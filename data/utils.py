# -*- coding: utf-8 -*-
# @Time    : 2024/10/29 20:18
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : utils.py
# @Software:  PyCharm python3.9
# @Description：

def make_patches(x, patch_size):
    """
    将图像转换为小点的patch
    :param x: tensor
    :param patch_size:
    :return:
    """
    if x.dim()>3:
        channel_dim = x.shape[1]
        patches = x.unfold(1, channel_dim, channel_dim).unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(-1, channel_dim, patch_size, patch_size)
    else:
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(-1, patch_size, patch_size)
    return patches, unfold_shape
