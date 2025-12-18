# -*- coding: utf-8 -*-
# @Time    : 2025/1/3 15:14
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : mixdata.py
# @Software:  PyCharm python3.9
# @Description：
import os

import torch
from torch.utils.data import Dataset

from data import fileio


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
                 **kwargs):
        super(PanDataset, self).__init__()
        self.phase = phase
        self.lr_size = lr_size
        self.name = name
        self.img_range = img_range
        if operate_sys == "win":
            data_root = data_root_win
        elif operate_sys == "linux":
            data_root = data_root_linux
        else:
            raise NotImplementedError(f"{operate_sys} data root not defined!!!")
        self.img_paths_dict = {k: fileio.get_image_paths(os.path.join(data_root, k), ".npy") for k in scaledict}
        self.data_len = len(self.img_paths_dict['REF'])

        self.wave_length = [] if wave_lengths is None else wave_lengths

    def __len__(self):
        return self.data_len

    def __getitem__(self, i):
        pathdict = {t_key: t_value[i] for t_key, t_value in self.img_paths_dict.items()}
        imgbatch_dict = fileio.read_img(pathdict, ".npy")
        if self.phase == "train":
            imgbatch_dict = fileio.augment(imgbatch_dict)
        imgbatch_dict = fileio.np2tensor(imgbatch_dict, self.img_range)

        imgbatch_dict["sensor"] = self.name
        imgbatch_dict["img_range"] = self.img_range

        imgbatch_dict["wave_length"] = torch.tensor(self.wave_length)
        return imgbatch_dict
