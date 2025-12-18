# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Spatial distortion index.
 
 Interface:
           D_s_index = D_s(I_F,I_MS,I_MS_LR,I_PAN,ratio,S,q)

 Inputs:
           I_F:                Pansharpened image;
           I_MS:               MS image resampled to panchromatic scale;
           I_MS_LR:            Original MS image;
           I_PAN:              Panchromatic image;
           ratio:              Scale ratio between MS and PAN. Pre-condition: Integer value;
           S:                  Block size;
           q:                  Exponent value (optional); Default value: q = 1.
 
 Outputs:
           D_s_index:          Spatial distortion index.
          
 Notes:
     Results very close to the MATLAB toolbox's ones. In particular, the results are more accurate than the MATLAB toolbox's ones
     because the Q-index is applied in a sliding window way. Instead, for computational reasons, the MATLAB toolbox uses a distinct block implementation
     of the Q-index.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from .interp23 import interp23
from .imresize import imresize
#
# def D_s(I_F,I_MS,I_MS_LR,I_PAN,ratio,S,q):
#
#     """ if 0, Toolbox 1.0, otherwise, original QNR paper """
#     flag_orig_paper = 0
#
#     if (I_F.shape != I_MS.shape):
#         print("The two images must have the same dimensions")
#         return -1
#
#     N = I_F.shape[0]
#     M = I_F.shape[1]
#     Nb = I_F.shape[2]
#
#     if (np.remainder(N,S-1) != 0):
#         print("Number of rows must be multiple of the block size")
#         return -1
#
#     if (np.remainder(M,S-1) != 0):
#         print("Number of columns must be multiple of the block size")
#         return -1
#
#     if (flag_orig_paper == 0):
#         """Opt. 1 (as toolbox 1.0)"""
#         if I_PAN.ndim==3 and I_PAN.shape[-1]==1: I_PAN = I_PAN.squeeze(axis=-1)
#         pan_filt = interp23(imresize(I_PAN,1/ratio), ratio)
#     else:
#         """ Opt. 2 (as paper QNR) """
#         pan_filt = imresize(I_PAN,1/ratio)
#
#     D_s_index = 0
#     for ii in range(Nb):
#         Q_high = ssim(I_F[:,:,ii],I_PAN, win_size=S, data_range=1.)
#
#         if (flag_orig_paper == 0):
#             """ Opt. 1 (as toolbox 1.0) """
#             Q_low = ssim(I_MS[:,:,ii],pan_filt, win_size=S, data_range=1.)
#         else:
#             """ Opt. 2 (as paper QNR) """
#             Q_low = ssim(I_MS_LR[:,:,ii],pan_filt, win_size=S, data_range=1.)
#
#         D_s_index = D_s_index + np.abs(Q_high-Q_low)**q
#
#     D_s_index = (D_s_index/Nb)**(1/q)
#
#     return D_s_index


from scipy.ndimage import zoom
from skimage.transform import resize

def uqi(x, y):
    x = x.flatten()
    y = y.flatten()
    mx = np.mean(x)
    my = np.mean(y)
    c = np.cov(x, y)

    Q = 4 * c[0, 1] * mx * my / (c[0, 0] + c[1, 1]) / (mx**2 + my**2)
    return Q

def D_s(I_F, I_MS, I_MS_LR, I_PAN, ratio, S, q):
    flag_orig_paper = 0  # if 0, Toolbox 1.0, otherwise, original QNR paper

    if I_F.shape != I_MS.shape:
        raise ValueError('The two images must have the same dimensions')

    N, M, Nb = I_F.shape

    if N % S != 0:
        raise ValueError('Number of rows must be multiple of the block size')

    if M % S != 0:
        raise ValueError('Number of columns must be multiple of the block size')

    if flag_orig_paper == 0:
        # Opt. 1 (as toolbox 1.0)
        if I_PAN.ndim == 3 and I_PAN.shape[-1] == 1: I_PAN = I_PAN.squeeze(axis=-1)
        pan_filt = interp23(imresize(I_PAN,1/ratio), ratio)
    else:
        # Opt. 2 (as paper QNR)
        pan_filt = imresize(I_PAN,1/ratio)

    D_s_index = 0
    for i in range(Nb):
        band1 = I_F[:, :, i]
        band2 = I_PAN
        Qmap_high = np.array([[uqi(band1[j:j+S, k:k+S], band2[j:j+S, k:k+S])
                               for k in range(0, band1.shape[1], S)]
                              for j in range(0, band1.shape[0], S)])
        Q_high = np.mean(Qmap_high)

        if flag_orig_paper == 0:
            band1 = I_MS[:, :, i]
            band2 = pan_filt
        else:
            band1 = I_MS_LR[:, :, i]
            band2 = pan_filt

        Qmap_low = np.array([[uqi(band1[j:j+S, k:k+S], band2[j:j+S, k:k+S])
                              for k in range(0, band1.shape[1], S)]
                             for j in range(0, band1.shape[0], S)])
        Q_low = np.mean(Qmap_low)

        D_s_index += abs(Q_high - Q_low) ** q

    D_s_index = (D_s_index / Nb) ** (1 / q)

    return D_s_index
