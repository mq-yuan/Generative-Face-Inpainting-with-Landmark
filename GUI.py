import torch
import os
import json
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
import tkinter as tk
from tkinter import filedialog



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
    device = torch.device('cuda:0')
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
    root = tk.Tk()

    config_path = tk.StringVar()
    config_path.set('./config/config.json')
    tk.Label(root, text="config路径: ").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=config_path).pack(side='top', anchor='n', fill='x')
    
    model_path = tk.StringVar()
    model_path.set('./models/phase_3_model_generator_epoch64')
    tk.Label(root, text="model路径: ").pack(side='top', anchor='nw')
    tk.Entry(root, textvariable=model_path).pack(side='top', anchor='n', fill='x')

    tk.Button(root, text="人脸补全", command=lambda: main(config=config_path.get(), model=model_path.get())).pack()
    root.mainloop()

    main(config='./config/config.json', model='./models/phase_3_model_generator_epoch64', result_dir='./results/')
    # main(mode='cat')