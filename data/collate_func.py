# -*- coding: utf-8 -*-
# @Time    : 2025/1/3 15:25
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : collate_func.py
# @Software:  PyCharm python3.9
# @Description：
import numpy as np
import torch


def get_sampled_wave_length(wave_length):
    """
    :param wave_length: (2, N) tensor
    :return:
    """
    mid = (wave_length[0] + wave_length[1]) / 2
    length = wave_length[1] - wave_length[0]
    start_tensor = mid - length / 4
    end_tensor = mid + length / 4
    random_samples = torch.stack([
        torch.randint(int(start.item()), int(end.item()), (1,)) for start, end in zip(start_tensor, end_tensor)
    ])
    return random_samples.flatten()


def mix_collect_func(batch):
    '''

    :param batch: 一个列表长度为batch size，每个列表元素对应dataset中的一个数据，我们这里对应的就是一个字典包含：LR,GT,REF,sensor,image_range,wave_length
    :return:
    '''
    res_batch_data = {}
    chans = [d["GT"].shape[0] for d in batch]
    res_batch_data['chans'] = chans
    padchans = max(chans) - np.array(chans)
    for key in ['GT', 'LR']:
        res_batch_data[key] = torch.stack(
            [torch.cat([d[key], torch.zeros((padchans[i], d[key].shape[1], d[key].shape[2]))], dim=0) for i, d in
             enumerate(batch)])

    res_batch_data["REF"] = torch.stack([d["REF"] for d in batch])
    res_batch_data["sensor"] = [d["sensor"] for d in batch]
    res_batch_data["img_range"] = torch.tensor([d["img_range"] for d in batch])

    # 计算选择的波长
    # 现在随机采样中间区域的50%区间的波作为采样区间
    wave_lengths = [get_sampled_wave_length(d["wave_length"]) for d in batch]
    res_batch_data["wave_length"] = torch.stack(
        [torch.cat([wl, torch.zeros((padchans[i],))], dim=0) for i, wl in enumerate(wave_lengths)])

    return res_batch_data


if __name__ == "__main__":
    # 在这个路径下跑不起来，在根目录新建一个py文件跑
    from utils.utils import instantiate_from_config
    from omegaconf import OmegaConf
    # 构建dataset
    from data.mixdata import PanDataset

    data_opt = OmegaConf.load("configs/data_opts/data_mix.yaml").data
    data = instantiate_from_config(data_opt)
    data.setup()
    train_loader = data.train_dataloader()

    # 迭代训练数据
    for batch in train_loader:
        # 在这里处理你的批次数据，例如训练模型
        print(batch)
