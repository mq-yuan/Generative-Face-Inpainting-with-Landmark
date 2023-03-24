import torch
import os
from tqdm import tqdm
import json
import argparse
from models import Discriminator, Generator
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from utils import (
    MyDataset,
    Generator_Network_Loss,
    gen_input_MaskLayer,
    gen_Mask,
    poisson_blend,
    crop,
    sample_random_batch,
    Discriminator_Network_Loss,
)

app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(256, 256))

parser = argparse.ArgumentParser()
parser.add_argument('--model_G', default='./demo/model_cn')
parser.add_argument('--model_D', default=None)
parser.add_argument('--phase1', type=int, default=8)
parser.add_argument('--phase2', type=int, default=2)
parser.add_argument('--phase3', type=int, default=40)

def train(input_generator_size=160, batch_size=16, alpha=4e-4,
          num_epochs_pregen=10, num_epochs_predis=10, num_epochs_step3=20,
          mpv=None, init_model_G=None, init_model_D=None):
    """面部识别算法还没有先默认脸部中间下方， local_D的local大小还没有确定先默认为global的一半"""
    result_dir = './result/'
    data_dir = './datasets/'
    test_dir = './test_data/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for phase in ['phase_1', 'phase_2', 'phase_3']:
        if not os.path.exists(os.path.join(result_dir, phase)):
            os.makedirs(os.path.join(result_dir, phase))

    device = torch.device('cuda:0')
    transform = transforms.Compose([
        transforms.Resize(input_generator_size),
        transforms.RandomCrop((input_generator_size, input_generator_size)),
        transforms.ToTensor(),
    ])
    print('loading dataset')

    train_dataset = MyDataset(data_dir, transform)
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = MyDataset(test_dir, transform)

    if mpv is None:
        mpv = np.zeros(shape=(3,))
        pbar = tqdm(
            total=len(train_dataset.images),
            desc='computing mean pixel value of training dataset...')
        for imgpath in train_dataset.images:
            img = Image.open(imgpath)
            x = np.array(img) / 255.
            mpv += x.mean(axis=(0, 1))
            pbar.update()
        mpv /= len(train_dataset.images)
        pbar.close()
    else:
        mpv = np.array(mpv)

    # save training config
    mpv_json = []
    for i in range(3):
        mpv_json.append(float(mpv[i]))
    args_dict = {"data_dir": data_dir, "test_dir": test_dir, "result_dir": result_dir,
                 "init_model_G": init_model_G, "init_model_D": init_model_D,
                 "num_epochs_pregen": num_epochs_pregen, "num_epochs_predis": num_epochs_predis,
                 "num_epochs_step3": num_epochs_step3, "input_generator_size": input_generator_size,
                 "batch_size": batch_size, "alpha": alpha, 'mpv': mpv_json}
    with open(os.path.join(
            result_dir, 'config.json'),
            mode='w') as f:
        json.dump(args_dict, f)
    mpv = torch.tensor(
        mpv.reshape(1, 3, 1, 1),
        dtype=torch.float32).to(device)
    alpha = torch.tensor(alpha, dtype=torch.float32).to(device)

    # Training Phase 1
    # load generator network
    G = Generator()
    if init_model_G is not None:
        G.load_state_dict(torch.load(init_model_G, map_location='cpu'))
    G = G.to(device)
    optimizer_G = torch.optim.Adadelta(G.parameters())
    loss_G = Generator_Network_Loss

    epoch = tqdm(total=num_epochs_pregen)
    update_epoch = num_epochs_pregen // 10
    if update_epoch < 1:
        update_epoch = 1
    while epoch.n < num_epochs_pregen:
        for X in train_iter:
            X = X.to(device)
            mask_size, mask_location = gen_Mask(X)
            masklayer = gen_input_MaskLayer(
                MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                mask_size=mask_size,
                mask_area=mask_location)
            masklayer = masklayer.to(device)
            X_mask = X - X * masklayer + mpv * masklayer
            input_G = torch.cat((X_mask, masklayer), dim=1)
            output_G = G(input_G)
            loss = loss_G(X, output_G, masklayer)
    
            # backward
            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()
            epoch.set_description('phase 1 | train loss: %.5f' % loss.cpu())
        epoch.update()
    
        # test
        if epoch.n % update_epoch == 0:
            G.eval()
            with torch.no_grad():
                X = sample_random_batch(test_dataset, batch_size=8).to(device)
                mask_size, mask_location = gen_Mask(X)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location)
                masklayer = masklayer.to(device)
                X_mask = X - X * masklayer + mpv * masklayer
                input_G = torch.cat((X_mask, masklayer), dim=1)
                output_G = G(input_G)
                completed = poisson_blend(X_mask, output_G, masklayer)
                imgs = torch.cat((
                    X.cpu(),
                    X_mask.cpu(),
                    completed.cpu()), dim=0)
                imgpath = os.path.join(
                    result_dir,
                    'phase_1',
                    'phase_1_epoch%d.png' % epoch.n)
                model_generator_path = os.path.join(
                    result_dir,
                    'phase_1',
                    'phase_1_model_generator_epoch%d' % epoch.n)
                save_image(imgs, imgpath, nrow=8)
                torch.save(
                    G.state_dict(),
                    model_generator_path)
            G.train()
        if epoch.n >= num_epochs_pregen:
            break
    epoch.close()

    # Training Phase 2
    # load discriminator network
    D = Discriminator(
        local_input_shape=(3, input_generator_size // 2, input_generator_size // 2),
        global_input_shape=(3, input_generator_size, input_generator_size))
    if init_model_D is not None:
        D.load_state_dict(torch.load(init_model_D, map_location='cpu'))
    D = D.to(device)
    optimizer_D = torch.optim.Adadelta(D.parameters())
    loss_D = torch.nn.BCELoss()

    epoch = tqdm(total=num_epochs_predis)
    update_epoch = num_epochs_predis // 10
    if update_epoch < 1:
        update_epoch = 1
    while epoch.n < num_epochs_predis:
        for X in train_iter:
            fake = torch.zeros((X.shape[0], 1)).to(device)
            real = torch.ones((X.shape[0], 1)).to(device)
            # fake forward
            X = X.to(device)
            mask_size, mask_location = gen_Mask(X)
            masklayer = gen_input_MaskLayer(
                MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                mask_size=mask_size,
                mask_area=mask_location)
            masklayer = masklayer.to(device)
            X_mask = X - X * masklayer + mpv * masklayer
            input_G = torch.cat((X_mask, masklayer), dim=1)
            output_G = G(input_G)
            input_global_D_fake = output_G.detach()
            input_local_D_fake = crop(input_global_D_fake, (mask_size, mask_location))
            output_D_fake = D((input_local_D_fake, input_global_D_fake))
            loss_fake = loss_D(output_D_fake, fake)

            # real forward
            input_global_D_real = X
            input_local_D_real = crop(input_global_D_real, (mask_size, mask_location))
            output_D_real = D((input_local_D_real, input_global_D_real))
            loss_real = loss_D(output_D_real, real)

            # landmark forward
            real_img = X
            fake_img = output_G
            loss_landmark = Discriminator_Network_Loss(real_img, fake_img, input_generator_size)

            loss = loss_real + loss_fake + loss_landmark

            # backward
            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()
            epoch.set_description('phase 2 | train loss: %.5f' % loss.cpu())
        epoch.update()

        if epoch.n % update_epoch == 0:
            G.eval()
            with torch.no_grad():
                X = sample_random_batch(test_dataset, batch_size=8).to(device)
                mask_size, mask_location = gen_Mask(X)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location)
                masklayer = masklayer.to(device)
                X_mask = X - X * masklayer + mpv * masklayer
                input_G = torch.cat((X_mask, masklayer), dim=1)
                output_G = G(input_G)
                completed = poisson_blend(X_mask, output_G, masklayer)
                imgs = torch.cat((
                    X.cpu(),
                    X_mask.cpu(),
                    completed.cpu()), dim=0)
                imgpath = os.path.join(
                    result_dir,
                    'phase_2',
                    'phase_2_epoch%d.png' % epoch.n)
                model_discriminator_path = os.path.join(
                    result_dir,
                    'phase_2',
                    'phase_2_model_discriminator_epoch%d' % epoch.n)
                save_image(imgs, imgpath, nrow=8)
                torch.save(
                    D.state_dict(),
                    model_discriminator_path)
            G.train()
        if epoch.n >= num_epochs_predis:
            break
    epoch.close()

    # Training Phase 3
    epoch = tqdm(total=num_epochs_step3)
    update_epoch = num_epochs_step3 // 10
    if update_epoch < 1:
        update_epoch = 1
    while epoch.n < num_epochs_step3:
        for X in train_iter:
            # forward D
            X = X.to(device)
            mask_size, mask_location = gen_Mask(X)
            masklayer = gen_input_MaskLayer(
                MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                mask_size=mask_size,
                mask_area=mask_location)
            masklayer = masklayer.to(device)

            fake = torch.zeros((len(X), 1)).to(device)
            real = torch.ones((X.shape[0], 1)).to(device)

            X_mask = X - X * masklayer + mpv * masklayer
            input_G = torch.cat((X_mask, masklayer), dim=1)
            output_G = G(input_G)
            input_global_D_fake = output_G.detach()
            input_local_D_fake = crop(input_global_D_fake, (mask_size, mask_location))
            output_D_fake = D((input_local_D_fake, input_global_D_fake))
            loss_fake = loss_D(output_D_fake, fake)
            input_global_D_real = X
            input_local_D_real = crop(input_global_D_real, (mask_size, mask_location))
            output_D_real = D((input_local_D_real, input_global_D_real))
            loss_real = loss_D(output_D_real, real)
            real_img = X
            fake_img = output_G
            loss_landmark = Discriminator_Network_Loss(real_img, fake_img, input_generator_size)
            loss_d = (loss_fake + loss_real + loss_landmark) * alpha

            optimizer_D.zero_grad()
            loss_d.backward()
            optimizer_D.step()

            # forward G
            loss_G_1 = loss_G(X, output_G, masklayer)
            input_gD_fake = output_G
            input_lD_fake = crop(input_gD_fake, (mask_size, mask_location))
            output_fake = D((input_lD_fake, input_gD_fake))
            loss_G_2 = loss_D(output_fake, real)

            loss = alpha * loss_G_2 + loss_G_1

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            epoch.set_description(
                'phase 3 | train loss (D): %.5f (G): %.5f' % (
                    loss_d.cpu(),
                    loss.cpu()))
        epoch.update()

        if epoch.n % update_epoch == 0:
            G.eval()
            with torch.no_grad():
                X = sample_random_batch(test_dataset, batch_size=8).to(device)
                mask_size, mask_location = gen_Mask(X)
                masklayer = gen_input_MaskLayer(
                    MaskLayer_shape=(X.shape[0], 1, X.shape[2], X.shape[3]),
                    mask_size=mask_size,
                    mask_area=mask_location)
                masklayer = masklayer.to(device)
                X_mask = X - X * masklayer + mpv * masklayer
                input_G = torch.cat((X_mask, masklayer), dim=1)
                output_G = G(input_G)
                completed = poisson_blend(X_mask, output_G, masklayer)
                imgs = torch.cat((
                    X.cpu(),
                    X_mask.cpu(),
                    completed.cpu()), dim=0)
                imgpath = os.path.join(
                    result_dir,
                    'phase_3',
                    'phase_3_epoch%d.png' % epoch.n)
                model_discriminator_path = os.path.join(
                    result_dir,
                    'phase_3',
                    'phase_3_model_discriminator_epoch%d' % epoch.n)
                model_generator_path = os.path.join(
                    result_dir,
                    'phase_3',
                    'phase_3_model_generator_epoch%d' % epoch.n)
                save_image(imgs, imgpath, nrow=8)
                torch.save(
                    G.state_dict(),
                    model_generator_path)
                torch.save(
                    D.state_dict(),
                    model_discriminator_path)
            G.train()
        if epoch.n >= num_epochs_step3:
            break


if __name__ == '__main__':
    args = parser.parse_args()
    train(batch_size=32, num_epochs_pregen=args.phase1, num_epochs_predis=args.phase2, num_epochs_step3=args.phase3,
          init_model_G=args.model_G, init_model_D=args.model_D)
