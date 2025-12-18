# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 17:08
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : vis_utils.py
# @Software:  PyCharm python3.9
# @Description：
import numpy as np
import torch
import os


def hist_line_stretch(img, nbins, bound=None):
    if bound is None:
        bound = [0.02, 0.98]

    def _line_strectch(img, nbins):
        nbins = int(nbins)
        # imgrange = [int(img.min()), np.ceil(img.max())]
        hist1, bins1 = np.histogram(img, bins=nbins, density=False)
        hist1 = hist1 / img.size
        cumhist = np.cumsum(hist1)
        lowThreshold = np.where(cumhist >= bound[0])[0][0]
        lowThreshold = bins1[lowThreshold]
        highThreshold = np.where(cumhist >= bound[1])[0][0]
        highThreshold = bins1[highThreshold]

        img[np.where(img < lowThreshold)] = lowThreshold
        img[np.where(img > highThreshold)] = highThreshold
        img = (img - lowThreshold) / (highThreshold - lowThreshold + np.finfo(float).eps)
        return img

    if img.ndim > 2:
        for i in range(img.shape[0]):
            img[i, :, :] = _line_strectch(img[i, :, :], nbins)
    else:
        img = _line_strectch(img, nbins)
    return img


def hist_line_stretch_pytorch(img, nbins, bound=None):  # CHW
    bound = bound if bound is not None else [0.02, 0.98]

    def _line_stretch(img, nbins):
        nbins = int(nbins)
        try:
            hist1, bins1 = torch.histogram(img, bins=nbins, density=False)
        except RuntimeError:
            torch.save(img, os.path.join("error_info", 'error_img.pt'))
        hist1 = hist1 / img.numel()
        cumhist = torch.cumsum(hist1, 0)
        lowThreshold = torch.where(cumhist >= bound[0])[0][0]
        lowThreshold = bins1[lowThreshold]
        highThreshold = torch.where(cumhist >= bound[1])[0][0]
        highThreshold = bins1[highThreshold]

        img[torch.where(img < lowThreshold)] = lowThreshold
        img[torch.where(img > highThreshold)] = highThreshold
        img = (img - lowThreshold) / (highThreshold - lowThreshold + torch.finfo(torch.float32).eps)
        return img

    if len(img.shape) == 3 and img.shape[0] >= 3:  # C H W 针对RGB图像
        for i in range(img.shape[0]):
            img[i] = _line_stretch(img[i], nbins)
    if len(img.shape) == 3 and img.shape[0] == 1:
        img[0] = _line_stretch(img[0], nbins)
    if len(img.shape) == 2:
        img = _line_stretch(img, nbins)
    return img


if __name__ == "__main__":
    img = torch.randn([3, 256, 256])
    a = hist_line_stretch_pytorch(img, nbins=255)
    print(a)
