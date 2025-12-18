# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""
from copy import deepcopy

from scipy.ndimage import sobel

from metrics.pan_metrics.metrics import _qindex

"""
 Description: 
           Reduced resolution quality indexes. 
 
 Interface:
           [Q2n_index, Q_index, ERGAS_index, SAM_index] = indexes_evaluation(I_F,I_GT,ratio,L,Q_blocks_size,flag_cut_bounds,dim_cut,th_values)

 Inputs:
           I_F:                Fused Image;
           I_GT:               Ground-Truth image;
           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
           L:                  Image radiometric resolution; 
           Q_blocks_size:      Block size of the Q-index locally applied;
           flag_cut_bounds:    Cut the boundaries of the viewed Panchromatic image;
           dim_cut:            Define the dimension of the boundary cut;
           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range.

 Outputs:
           Q2n_index:          Q2n index;
           Q_index:            Q index;
           ERGAS_index:        Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) index;
           SAM_index:          Spectral Angle Mapper (SAM) index.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""
#
# """
#
# 计算指标的新方法
# import torchmetrics.functional.image as MF
#     def save_full_ref(self, pred, gt, split="val"):
#         data_range = (0., 1.)
#         self.record_metrics('MAE', F.l1_loss(pred, gt), split)
#         self.record_metrics('SSIM', MF.structural_similarity_index_measure(pred, gt, data_range=data_range), split)
#         self.record_metrics('RMSE', MF.root_mean_squared_error_using_sliding_window(pred, gt), split)
#         self.record_metrics('ERGAS', MF.error_relative_global_dimensionless_synthesis(pred, gt), split)
#         self.record_metrics('SAM', MF.spectral_angle_mapper(pred, gt), split)
#         self.record_metrics('RASE', MF.relative_average_spectral_error(pred, gt), split)
#         self.record_metrics('PSNR', MF.peak_signal_noise_ratio(pred, gt, data_range=data_range), split)
#         self.record_metrics('UQI', MF.universal_image_quality_index(pred, gt), split)
#         self.record_metrics('CC', cross_correlation(pred, gt), split)
#
#     def save_no_ref(self, lrms, pan, pred, split="val"):
#         d_lambda = MF.spectral_distortion_index(lrms, pred)
#         d_s = MF.spatial_distortion_index(pred, lrms, pan.repeat(1, lrms.shape[1], 1, 1))
#         qnr = (1 - d_lambda) * (1 - d_s)
#         self.record_metrics('D_lambda', d_lambda, split)
#         self.record_metrics('D_s', d_s, split)
#         self.record_metrics('QNR', qnr, split)
# """

import numpy as np
from metrics.pan_metrics.ERGAS import ERGAS_np as ERGAS
from metrics.pan_metrics.SAM import SAM, sam
from metrics.pan_metrics.q2n import q2n


def Q_AVE(img1, img2, block_size=8):
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


def SCC(I_F, I_GT):
    # Define a function to apply the Sobel filter along both axes and compute the magnitude
    def sobel_magnitude(I):
        I_sobel_y = sobel(I, axis=0, mode='constant')
        I_sobel_x = sobel(I, axis=1, mode='constant')
        return np.sqrt(I_sobel_y ** 2 + I_sobel_x ** 2)

    # Apply the Sobel filter to each channel
    Im_Lap_F = np.zeros((I_F.shape[0] - 2, I_F.shape[1] - 2, I_F.shape[2]))
    for idim in range(I_F.shape[2]):
        Im_Lap_F[:, :, idim] = sobel_magnitude(I_F[1:-1, 1:-1, idim])

    Im_Lap_GT = np.zeros((I_GT.shape[0] - 2, I_GT.shape[1] - 2, I_GT.shape[2]))
    for idim in range(I_GT.shape[2]):
        Im_Lap_GT[:, :, idim] = sobel_magnitude(I_GT[1:-1, 1:-1, idim])

    # Compute sCC
    sCC = np.sum(Im_Lap_F * Im_Lap_GT)
    sCC /= np.sqrt(np.sum(Im_Lap_F ** 2))
    sCC /= np.sqrt(np.sum(Im_Lap_GT ** 2))

    # Compute SCCMap
    # SCCMap = np.sum(Im_Lap_F * Im_Lap_GT, axis=2)
    # SCCMap /= np.sqrt(np.sum(Im_Lap_GT**2))
    # SCCMap /= np.sqrt(np.sum(Im_Lap_GT**2))

    return sCC  # , SCCMap


def sCC(ms, ps):
    ps_sobel = sobel(ps, mode='constant')
    ms_sobel = sobel(ms, mode='constant')
    return (np.sum(ps_sobel * ms_sobel) / np.sqrt(np.sum(ps_sobel * ps_sobel)) / np.sqrt(np.sum(ms_sobel * ms_sobel)))


def indexes_evaluation(I_F, I_GT, ratio, Qblocks_size, flag_cut_bounds, dim_cut, th_values, ret_dict=True):
    """ cut bounds """
    if (flag_cut_bounds == 1):
        I_GT = I_GT[dim_cut - 1:I_GT.shape[0] - dim_cut, dim_cut - 1:I_GT.shape[1] - dim_cut, :]
        I_F = I_F[dim_cut - 1:I_F.shape[0] - dim_cut, dim_cut - 1:I_F.shape[1] - dim_cut, :]

    if th_values == 1:
        # I_F[I_F > 2**L] = 2**L
        # I_F[I_F < 0] = 0
        I_F[I_F > 1.0] = 1.0
        I_F[I_F < 0] = 0

    ERGAS_index = ERGAS(I_GT, I_F, ratio)
    SAM_index = SAM(I_GT, I_F)
    Qave = Q_AVE(I_GT, I_F, Qblocks_size)
    scc = SCC(I_F, I_GT)  # 上面的指标都不会修改I_F 和 I_GT的值，但是下面q2n指标会修改I_GT， I_F的值
    # if np.remainder(Qblocks_size,2) == 0:
    #     Q_index = Q(I_GT,I_F,Qblocks_size + 1)
    # else:
    #     Q_index = Q(I_GT,I_F,Qblocks_size)

    Q2n_index, Q2n_index_map = q2n(deepcopy(I_GT), I_F, Qblocks_size, Qblocks_size)
    if ret_dict:
        return {"q2n":Q2n_index, "qave":Qave, "ergas":ERGAS_index, "sam":SAM_index, "scc":scc}
    else:
        return Q2n_index, Qave, ERGAS_index, SAM_index, scc
