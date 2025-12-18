# -*- coding: utf-8 -*-
# @Time    : 2024/1/8 21:59
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : metrics.py
# @Software:  PyCharm python3.9
# @Description：
import math

import cv2
import numpy as np
from scipy.ndimage import sobel
from metrics.pan_metrics import onions_quality

EPS = 2.2204e-16  # matlab的eps值


def SAM(ms,ps,degs = True):
    result = np.double(ps)
    target = np.double(ms)
    if result.shape != target.shape:
        raise ValueError('Result and target arrays must have the same shape!')

    rnorm = (result ** 2).sum(axis=2)
    tnorm = (target ** 2).sum(axis=2)
    dotprod = (result * target).sum(axis=2)
    rnorm = np.sqrt(rnorm*tnorm)
    tnorm = rnorm
    rnorm = np.maximum(rnorm, EPS)
    cosines = (dotprod / rnorm)
    sam2d = np.arccos(cosines)
    if degs:
        sam2d = np.rad2deg(sam2d)

    rnorm = tnorm[tnorm>0]
    dotprod = dotprod[tnorm>0]
    sam =np.arccos(dotprod/rnorm).mean()

    if degs:
        sam = np.rad2deg(sam)
    # sam2d[np.invert(np.isfinite(sam2d))] = 0.  # arccos(1.) -> NaN
    return sam

def ERGAS(img_fake, img_real, scale=4):
    """ERGAS for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1].
    scale = spatial resolution of PAN / spatial resolution of MUL, default 4.
    """
    if not img_fake.shape == img_real.shape:
        raise ValueError('Input images must have the same dimensions.')
    img_fake_ = img_fake.astype(np.float64)
    img_real_ = img_real.astype(np.float64)
    if img_fake_.ndim == 2:
        mean_real = img_real_.mean()
        mean_real = mean_real ** 2
        mse = np.mean((img_fake_ - img_real_) ** 2)
        mean_real = np.maximum(mean_real, EPS)
        return 100.0 / scale * np.sqrt(mse / mean_real)
    elif img_fake_.ndim == 3:
        means_real = img_real_.reshape(-1, img_real_.shape[2]).mean(axis=0)
        means_real = means_real ** 2
        means_real = np.maximum(means_real, EPS)
        mses = ((img_fake_ - img_real_) ** 2).reshape(-1, img_fake_.shape[2]).mean(axis=0)
        return 100.0 / scale * np.sqrt((mses / means_real).mean())
    else:
        raise ValueError('Wrong input image dimensions.')


def q2n(I_GT, I_F, Q_blocks_size, Q_shift):  # HWC
    N1, N2, N3 = I_GT.shape

    size2 = Q_blocks_size

    stepx = math.ceil(N1 / Q_shift)  # 计算H，W分别能滑动多少步
    stepy = math.ceil(N2 / Q_shift)

    if (stepy <= 0):
        stepy = 1
        stepx = 1

    est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1  # 计算需要扩展的量，使得像素数刚好全部可以用上
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2
    # 对图片进行边界扩展， 这个代码有时间得重构一下
    if (est1 != 0) or (est2 != 0):
        refref = []
        fusfus = []

        for i in range(N3):
            a1 = np.squeeze(I_GT[:, :, 0])

            ia1 = np.zeros((N1 + est1, N2 + est2))
            ia1[0:N1, 0:N2] = a1
            ia1[:, N2:N2 + est2] = ia1[:, N2 - 1:N2 - est2 - 1:-1]
            ia1[N1:N1 + est1, :] = ia1[N1 - 1:N1 - est1 - 1:-1, :]

            if i == 0:
                refref = ia1
            elif i == 1:
                refref = np.concatenate((refref[:, :, None], ia1[:, :, None]), axis=2)
            else:
                refref = np.concatenate((refref, ia1[:, :, None]), axis=2)

            if (i < (N3 - 1)):
                I_GT = I_GT[:, :, 1:I_GT.shape[2]]

        I_GT = refref

        for i in range(N3):
            a2 = np.squeeze(I_F[:, :, 0])

            ia2 = np.zeros((N1 + est1, N2 + est2))
            ia2[0:N1, 0:N2] = a2
            ia2[:, N2:N2 + est2] = ia2[:, N2 - 1:N2 - est2 - 1:-1]
            ia2[N1:N1 + est1, :] = ia2[N1 - 1:N1 - est1 - 1:-1, :]

            if i == 0:
                fusfus = ia2
            elif i == 1:
                fusfus = np.concatenate((fusfus[:, :, None], ia2[:, :, None]), axis=2)
            else:
                fusfus = np.concatenate((fusfus, ia2[:, :, None]), axis=2)

            if (i < (N3 - 1)):
                I_F = I_F[:, :, 1:I_F.shape[2]]

        I_F = fusfus

    # I_F = np.uint16(I_F)
    # I_GT = np.uint16(I_GT)
    N1, N2, N3 = I_GT.shape
    # 对通道维度进行扩展使其通道数为2^n
    if (((math.ceil(math.log2(N3))) - math.log2(N3)) != 0):
        Ndif = (2 ** (math.ceil(math.log2(N3)))) - N3
        dif = np.zeros((N1, N2, Ndif))
        # dif = np.uint16(dif)
        I_GT = np.concatenate((I_GT, dif), axis=2)
        I_F = np.concatenate((I_F, dif), axis=2)

    N3 = I_GT.shape[2]

    valori = np.zeros((stepx, stepy, N3))

    for j in range(stepx):
        for i in range(stepy):
            o = onions_quality(
                I_GT[(j * Q_shift):(j * Q_shift) + Q_blocks_size, (i * Q_shift):(i * Q_shift) + size2, :],
                I_F[(j * Q_shift):(j * Q_shift) + Q_blocks_size, (i * Q_shift):(i * Q_shift) + size2, :], Q_blocks_size)
            valori[j, i, :] = o

    Q2n_index_map = np.sqrt(np.sum(valori ** 2, axis=2))

    Q2n_index = np.mean(Q2n_index_map)

    return Q2n_index, Q2n_index_map


def _qindex(img1, img2, block_size=8):
    """Q-index for 2D (one-band) image, shape (H, W); uint or float [0, 1]"""
    assert block_size > 1, 'block_size shold be greater than 1!'
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    window = np.ones((block_size, block_size)) / (block_size ** 2)
    # window_size = block_size**2
    # filter, valid
    pad_topleft = int(np.floor(block_size / 2))
    pad_bottomright = block_size - 1 - pad_topleft
    mu1 = cv2.filter2D(img1_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu2 = cv2.filter2D(img2_, -1, window)[pad_topleft:-pad_bottomright, pad_topleft:-pad_bottomright]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1_ ** 2, -1, window)[pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu1_sq
    sigma2_sq = cv2.filter2D(img2_ ** 2, -1, window)[pad_topleft:-pad_bottomright,
                pad_topleft:-pad_bottomright] - mu2_sq
    sigma12 = cv2.filter2D(img1_ * img2_, -1, window)[pad_topleft:-pad_bottomright,
              pad_topleft:-pad_bottomright] - mu1_mu2

    # all = 1, include the case of simga == mu == 0
    qindex_map = np.ones(sigma12.shape)
    # sigma == 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) == 0) * ((mu1_sq + mu2_sq) != 0)
    qindex_map[idx] = 2 * mu1_mu2[idx] / (mu1_sq + mu2_sq)[idx]
    # sigma !=0 and mu == 0
    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) == 0)
    qindex_map[idx] = 2 * sigma12[idx] / (sigma1_sq + sigma2_sq)[idx]
    # sigma != 0 and mu != 0
    idx = ((sigma1_sq + sigma2_sq) != 0) * ((mu1_sq + mu2_sq) != 0)
    qindex_map[idx] = ((2 * mu1_mu2[idx]) * (2 * sigma12[idx])) / (
            (mu1_sq + mu2_sq)[idx] * (sigma1_sq + sigma2_sq)[idx])
    return np.mean(qindex_map)


def Q_AVE(img1, img2, block_size=8): # matlab写死了是32
    """Q-index for 2D (H, W) or 3D (H, W, C) image; uint or float [0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return _qindex(img1, img2, block_size)
    elif img1.ndim == 3:
        qindexs = [_qindex(img1[..., i], img2[..., i], block_size) for i in range(img1.shape[2])]
        return np.array(qindexs).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def mPSNR(ref, tar):
    """
    The same implementation as Naoto's toolbox for HSMS image fusion.
    """
    if ref.ndim > 2:
        bands = ref.shape[2]
        ref = ref.reshape(-1, bands)
        tar = tar.reshape(-1, bands)
        msr = ((ref - tar) ** 2).mean(axis=0)
        # max2 = max(ref,[],1).^2;
        max2 = ref.max(axis=0) ** 2
        if (msr == 0).any():
            return float('inf')
        psnrall = 10 * np.log10(max2 / msr)
        # out.ave = mean(psnrall);
        return psnrall.mean()
    else:
        max2 = ref.max()
        msr = ((ref - tar) ** 2).mean()
        return 10 * np.log10(max2 / msr)


def sCC(ms, ps):
    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')
    return (np.sum(ps_sobel * ms_sobel) / np.sqrt(np.sum(ps_sobel * ps_sobel)) / np.sqrt(np.sum(ms_sobel * ms_sobel)))

def cal_all_rr_metrics(img_gt, img_pms, ratio=4, Q_block_size=32, dim_cut=None):
    """
    img_gt: numpy ndarray HWC
    img_pms: numpy ndarray HWC;数据的范围已经处理到正常的范围
    """

    if dim_cut is not None:
        img_gt = img_gt[dim_cut:-dim_cut, dim_cut:-dim_cut, :]
        img_pms = img_pms[dim_cut:-dim_cut, dim_cut:-dim_cut, :]

    q2n_index, _ = q2n(img_gt, img_pms, Q_block_size, Q_block_size)
    Q_index = Q_AVE(img_gt, img_pms, Q_block_size)
    sam_index = SAM(img_gt, img_pms)
    ergas_index = ERGAS(img_pms, img_gt, ratio)
    scc = sCC(img_gt, img_pms)
    return {"q2n":q2n_index,
            'qavg': Q_index,
            "sam":sam_index,
            "ergas":ergas_index,
            "scc":scc}






if __name__ == "__main__":
    pass
