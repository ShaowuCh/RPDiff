# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 17:02
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : utils.py
# @Software:  PyCharm python3.9
# @Description：
import os

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image

from utils.vis_utils import hist_line_stretch_pytorch


def log_img_local(save_dir, imgs, key, clamp=True):
    """
        imgs: tensor BCHW, 取值范围[-1 ,1]
    """
    b, c, h, w = imgs.shape
    if clamp:
        imgs = torch.clamp(imgs, 0, 1.)
    # Choose the rgb channel
    if c == 4:
        imgs = imgs[:, [2, 1, 0], :, :]
    elif c == 8:
        imgs = imgs[:, [4, 2, 1], :, :]
    elif c not in [1, 3, 4, 8]:
        raise ValueError(f"Not defined channel number:{c}!")
    # imgs = (imgs + 1.0) / 2.0  # 将取值范围变为 [0, 1]
    nbins = 255
    for i in range(b):
        imgs[i] = hist_line_stretch_pytorch(imgs[i], nbins=nbins)
    grid = torchvision.utils.make_grid(imgs, nrow=5)
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    filename = f"{key}.png"
    path = os.path.join(save_dir, filename)
    Image.fromarray(grid).save(path)


def save_gt(root, dataloaders):
    datas = {"LR": [],
             "GT": [],
             "REF": []}
    for batch in dataloaders:
        datas["LR"].append(batch["LR"])
        datas["GT"].append(batch["GT"])
        datas["REF"].append(batch["REF"])

    for k, imgs in datas.items():
        imgs = torch.cat(imgs, dim=0)
        log_img_local(root, imgs, k)


def save_record2csv(record_path, metrics, epoch):
    metrics = {k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
    data_frame = pd.DataFrame(data=metrics, index=[epoch])
    need_header = True if epoch == 0 else False
    data_frame.to_csv(record_path, mode='a', encoding='utf-8', header=need_header)
