import torch
import os
import json
import argparse
from models import Generator
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from torch.nn import functional as F
from utils import (
    MyDataset,
    gen_input_MaskLayer,
    gen_Mask,
    poisson_blend,
)



parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('config')
parser.add_argument('result_dir')
parser.add_argument('--mode')

###
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------测试--------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------------------- #
###


def main(input_generator_size=160, config='./demo/config.json', model='./demo/model_cn', data='./images/test/',
         result_dir='./results/No_landmark/', mode=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(config, 'r') as f:
        config = json.load(f)
    mpv = torch.tensor(config['mpv']).view(1, 3, 1, 1)
    mpv = mpv.to(device)
    G = Generator()
    G.load_state_dict(torch.load(model, map_location='cpu'))
    G = G.to(device)
    G.eval()

    transform = transforms.Compose([
        transforms.Resize(input_generator_size),
        transforms.CenterCrop((input_generator_size, input_generator_size)),
        transforms.ToTensor()
    ])
    test_dataset = MyDataset(data, transform)
    with torch.no_grad():
        if mode == 'cat':
            num_samples = len(test_dataset)
            batch = []
            for i in range(num_samples):
                x = torch.unsqueeze(test_dataset[i], dim=0)
                batch.append(x)
            x = torch.cat(batch, dim=0).to(device)
            mask_size, mask_location = gen_Mask(x)
            masklayer = gen_input_MaskLayer(MaskLayer_shape=(x.shape[0], 1, x.shape[2],
                                                                x.shape[3]),
                                            mask_size=mask_size,
                                            mask_area=mask_location)
            masklayer = masklayer.to(device)
            x_mask = x - x * masklayer + mpv * masklayer
            input_G = torch.cat((x_mask, masklayer), dim=1)
            output_G = G(input_G)
            completed = poisson_blend(x_mask, output_G, masklayer).to(device)
            imgs = torch.cat((
                x,
                x_mask,
                completed), dim=2)
            imgpath = os.path.join(
                result_dir,
                'predict.png' )
            save_image(imgs, imgpath, nrows=0)
            print(imgpath)
        else:
            number = 0
            for x in test_dataset:
                x = torch.unsqueeze(x, dim=0).to(device)
                mask_size, mask_location = gen_Mask(x)
                masklayer = gen_input_MaskLayer(MaskLayer_shape=(x.shape[0], 1, x.shape[2],
                                                                x.shape[3]),
                                                mask_size=mask_size,
                                                mask_area=mask_location)
                masklayer = masklayer.to(device)
                x_mask = x - x * masklayer + mpv * masklayer
                input_G = torch.cat((x_mask, masklayer), dim=1)
                output_G = G(input_G)
                completed = poisson_blend(x_mask, output_G, masklayer).to(device)
                imgs = torch.cat(( x, x_mask, completed ), dim=2)
                imgpath = os.path.join(
                    result_dir,
                    '%d.png' % number)
                save_image(imgs, imgpath, nrows=3)
                number += 1
            print('img saved in ', result_dir)

if __name__ == '__main__':
    args = parser.parse_args()
    main(config=args.config, model=args.model, result_dir=args.result_dir, mode=args.mode)
    # main(config='./config/config.json', model='./models/phase_3_model_generator_epoch64', result_dir='./results/')
    # main(mode='cat')