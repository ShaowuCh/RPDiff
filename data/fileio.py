import glob
import os
import random

import numpy as np
import torch
from utils.vis_utils import hist_line_stretch_pytorch


def multidata(func):
    def inner(input, *args, **kvargs):
        if isinstance(input, dict):
            return {t_key: func(tensor, *args, **kvargs) for t_key, tensor in input.items()}
        elif isinstance(input, (list, tuple)):
            return [func(tmp, *args, **kvargs) for tmp in input]
        else:
            return func(input, *args, **kvargs)

    return inner


def get_image_paths(path, ext):
    assert os.path.isdir(path), '[Error] [%s] is not a valid directory' % path
    images = glob.glob(os.path.join(path, '*' + ext))
    images.sort()
    assert images, '[%s] has no valid file' % path
    return images


@multidata
def read_img(path, ext):
    # read image by misc or from .npy
    # return: Numpy float32, HWC, RGB, [0,255]
    if ext == '.npy':
        img = np.load(path)
    # elif ext in IMG_EXT:
    #     import imageio
    #     img = imageio.imread(path, pilmode='RGB')
    # elif ext == '.mat':
    #     from scipy import io as sciio
    #     img = sciio.loadmat(file_name=path)
    # elif ext in ('.tif', '.TIF', '.tiff', '.TIFF'):
    #     from skimage import io as skimgio
    #     img = skimgio.imread(path)
    else:
        raise NotImplementedError(
            'Cannot read this type (%s) of data' % ext)
    if isinstance(img, np.ndarray) and img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


@multidata
def np2tensor(img, img_range, run_range=1.0):
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1))) / 1.0
    tensor = torch.from_numpy(np_transpose).float()
    # tensor = (tensor - tensor.min())/ (tensor.max() - tensor.min())
    tensor = tensor.mul_(run_range / img_range)  # [0, 1]
    # tensor = hist_line_stretch_pytorch(tensor, 255) # 这个直接放到forward中处理
    # tensor = tensor * 2 - 1.0  # [-1, 1]
    return tensor


def dict_random_crop(imgdict: dict, patch_size: int, op_type='np'):
    '''
    imgdict: a list of images whose resolution increase with index of the list
    scale_dict: list of scales for the corresponding images in imglist
    patch_size: the patch size for the fisrt image to be cropped.
    '''
    if op_type == 'np':
        sz_dict = {key: value.shape[0] for key, value in imgdict.items()}
    else:
        sz_dict = {key: value.shape[-1] for key, value in imgdict.items()}
    minkey = min(sz_dict.keys(), key=lambda x: sz_dict.get(x))
    scale_dict = {key: value // sz_dict[minkey] for key, value in sz_dict.items()}
    ih, iw = imgdict[minkey].shape[:2] if op_type == 'np' else imgdict[minkey].shape[-2:]
    ix = np.random.randint(0, ih - patch_size + 1)
    iy = np.random.randint(0, iw - patch_size + 1)
    pos_size_dict = {t_key: (ix * t_scale, iy * t_scale, patch_size * t_scale)
                     for t_key, t_scale in scale_dict.items()}
    if op_type == 'np':
        out_patch = {t_key: imgdict[t_key][ix:ix + t_psize, iy:iy + t_psize, :]
                     for t_key, (ix, iy, t_psize) in pos_size_dict.items()}
    elif op_type == 'torch':
        out_patch = {t_key: imgdict[t_key][:, :, ix:ix + t_psize, iy:iy + t_psize]
                     for t_key, (ix, iy, t_psize) in pos_size_dict.items()}
    return out_patch


def augment(input, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    @multidata
    def _augment(img, hflip, vflip, rot90):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return _augment(input, hflip, vflip, rot90)
