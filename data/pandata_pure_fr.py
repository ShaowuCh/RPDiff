# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 20:07
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : pandata_pure.py
# @Software:  PyCharm python3.9
# @Description：
import os

import torch
from torch.utils.data import Dataset

from data import fileio


def get_middle_quarter(image_tensor):
    """
    Get center quarter sub image

    :param
    image_tensor (torch.Tensor):  (C, H, W)
    返回:
    torch.Tensor:
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise ValueError("The input must be PyTorch tensor")
    C, H, W = image_tensor.shape

    h_start = 3 * H // 8
    h_end = 5 * H // 8
    w_start = 3 * W // 8
    w_end = 5 * W // 8

    middle_quarter = image_tensor[:, h_start:h_end, w_start:w_end]
    return middle_quarter


class PanDataset(Dataset):
    def __init__(self, data_root_win,
                 data_root_linux,
                 operate_sys,
                 name,
                 scaledict,
                 img_range,
                 phase,  # train or val
                 lr_size,
                 wave_lengths=None,
                 clip=True,
                 **kwargs):
        super(PanDataset, self).__init__()
        self.phase = phase
        self.lr_size = lr_size
        self.name = name
        self.img_range = img_range
        self.clip = clip
        if operate_sys == "win":
            data_root = data_root_win
        elif operate_sys == "linux":
            data_root = data_root_linux
        else:
            raise NotImplementedError(f"{operate_sys} data root not defined!!!")
        self.img_paths_dict = {k: fileio.get_image_paths(os.path.join(data_root, k), ".npy") for k in scaledict}
        self.data_len = len(self.img_paths_dict['GT'])

        self.wave_length = [] if wave_lengths is None else wave_lengths

    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        pathdict = {t_key: t_value[i] for t_key, t_value in self.img_paths_dict.items()}
        imgbatch_dict = fileio.read_img(pathdict, ".npy")
        imgbatch_dict = fileio.np2tensor(imgbatch_dict, self.img_range)
        imgbatch_dict["sensor"] = self.name
        imgbatch_dict["img_range"] = self.img_range

        # # 裁剪
        if self.clip:
            imgbatch_dict['LR'] = get_middle_quarter(imgbatch_dict['GT'])
            imgbatch_dict['REF'] = get_middle_quarter(imgbatch_dict['REF_FR'])
        else:
            imgbatch_dict['LR'] = imgbatch_dict['GT']
            imgbatch_dict['REF'] = imgbatch_dict['REF_FR']

        imgbatch_dict["wave_length"] = torch.tensor(self.wave_length)
        return imgbatch_dict
