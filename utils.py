import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F
from PIL import Image
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch.utils import data
import numpy as np
import random
import torchvision
import cv2

device = torch.device('cuda:0')
def Generator_Network_Loss(input_G, output_G, mask):
    return F.mse_loss(output_G * mask, input_G * mask)

def get_landmark(img, app):
    img = img.clone().cpu()
    num_samples = img.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(img[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        faces = app.get(dstimg)
        # assert len(faces) == 1
        if len(faces) >= 1:
            face = faces[0]
            lmk = face.landmark_2d_106
            lmk = np.round(lmk).astype(int)
        else:
            lmk = np.zeros((106, 2)).astype(int)
        ret.append(lmk)
    return torch.Tensor(ret)


def draw_landmark(img, app):
    dstimg = transforms.functional.to_pil_image(img)
    dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
    faces = app.get(dstimg)
    # assert len(faces) == 1
    if len(faces) >= 1:
        face = faces[0]
        lmk = face.landmark_2d_106
        lmk = np.round(lmk).astype(int)
    else:
        lmk = np.zeros((106, 2)).astype(int)
    for coor in lmk:
        cv2.circle(dstimg, (int(coor[0]),int(coor[1])), 1, (0, 0, 255), 4)
    return dstimg


def Discriminator_Network_Loss(real_img, fake_img, det_size):
    real_landmark = get_landmark(real_img, det_size=det_size).to(device)
    fake_landmark = get_landmark(fake_img, det_size=det_size).to(device)
    return F.l1_loss(real_landmark, fake_landmark)


class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        images_path = Path(root)
        images_list = list(images_path.glob('*.jpg'))
        images_list_str = [str(x) for x in images_list]
        self.images = images_list_str

    def __getitem__(self, item):
        image_path = self.images[item]
        try:
            image = Image.open(image_path)
        except (OSError, NameError):
            image = cv2.imread(image_path)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.images)


def gen_input_MaskLayer(MaskLayer_shape, mask_size, mask_area):
    """生成遮挡口罩的图层

    :param MaskLayer_shape: 图层的大小(N, 1, X_shape, X_shape)
    :param mask_size: 遮挡口罩的大小(H, W)
    :param mask_area: 遮挡口罩区域的左上角坐标(x, y)
    :return 遮挡口罩图层， 需要遮挡部分值为1， 不需要遮挡部分值为0

    """
    MaskLayer = torch.zeros(MaskLayer_shape)
    batch_size, _, MaskLayer_h, MaskLayer_w = MaskLayer_shape
    mask_h, mask_w = mask_size
    upper_left_x, upper_left_y = mask_area
    for i in range(batch_size):
        MaskLayer[i, :, upper_left_y: upper_left_y + mask_h, upper_left_x: upper_left_x + mask_w] = 1.0
    return MaskLayer


def gen_Mask(face_img):
    """识别图片口罩区域， 或识别人脸下半部分用于戴口罩区域
    :param face_img: tensor类型的数组(N, C, H, W)
    :return (mask_size(h, w), mask_location(x, y))
    """
    H, W = face_img.shape[-2:]
    return (H // 2, W // 2), (W // 4, 2 * H // 4)


def poisson_blend(masked_img, output_G, masklayer):
    """通过cv2将output_G修补遮挡后的图片生成完成图片
    :param masked_img: Generator网络的输入，涂黑遮挡后的图片
    :type masked_img: torch.Tensor (N, 3, H, W)
    :param output_G: Generator网络的输出，利用它来修复涂黑遮挡的图片
    :type output_G: torch.Tensor (N, 3, H, W)
    :param masklayer: 遮挡层
    :type masklayer: torch.Tensor (N, 1, H, W)
    """
    masked_img = masked_img.clone().cpu()
    output_G = output_G.clone().cpu()
    mask = masklayer.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)
    num_samples = masked_img.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(masked_img[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output_G[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        xs, ys = [], []
        for j in range(msk.shape[0]):
            for k in range(msk.shape[1]):
                if msk[j, k, 0] == 255:
                    ys.append(j)
                    xs.append(k)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        dstimg = cv2.inpaint(dstimg, msk[:, :, 0], 1, cv2.INPAINT_TELEA)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret


def crop(img, crop_area):
    """裁剪生成local_D的输入
    :param img: global_D的输入(N, 3, H, W)
    :param crop_area: (area_size, area_location)
    """
    xmin, ymin = crop_area[1]
    h, w = crop_area[0]
    return img[:, :, ymin: ymin + h, xmin: xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)