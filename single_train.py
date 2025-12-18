# -*- coding: utf-8 -*-
# @Time    : 2024/11/21 10:09
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : single_train.py
# @Software:  PyCharm python3.9
# @Description：
import os

from omegaconf import OmegaConf


def singledata_train4params(gpu_ids: str, configs: str, ckp_path: str, data_name: str):
    """
    :param gpu_ids:  "1,3"
    :param configs:  "method_name or config files name"
    :return:
    """
    gpu_ids = gpu_ids
    configs_prefix = "./configs/"  # "./configs/compared_methods/"
    configs_postfix = ".yaml"
    data_prefix = "./configs/data_opts/"

    configs = configs
    ckpoint_path = ckp_path
    data_opt = f"data_{data_name}.yaml"
    fr_data_opt = f"fr_data_{data_name}.yaml"

    for dim in [16, 32, 64, 128, 256]:
        for num_tokens in [64, 128, 256, 512, 1024]:
            config = OmegaConf.load(configs_prefix + configs + configs_postfix)
            config.model.params.dim = dim
            config.model.params.num_tokens = num_tokens
            config.lightning.callbacks.test_recoder.params.method_name = f"{configs.upper()}_dim{dim}_tk{num_tokens}"
            config.lightning.name = f"{configs.upper()}_dim{dim}_tk{num_tokens}"
            OmegaConf.save(config, configs_prefix + configs + configs_postfix)
            cmd = (f"CUDA_VISIBLE_DEVICES={gpu_ids} python main.py --base {configs_prefix + configs + configs_postfix} "
                   f"--phase train "
                   f"--data_opt {data_prefix + data_opt} "
                   f"--fr_data_opt {data_prefix + fr_data_opt}")
            retcode = os.system(cmd)
            print(retcode)


def singledata_train(gpu_ids: str, configs: str, ckp_path: str, data_name: str, iscompored=False):
    """
    :param gpu_ids:  "1,3"
    :param configs:  "method_name or config files name"
    :return:
    """
    gpu_ids = gpu_ids
    configs_prefix = "./configs/compared_methods/" if iscompored else "./configs/"
    # configs_prefix = "./configs/ablation/"
    configs_postfix = ".yaml"
    data_prefix = "./configs/data_opts/"

    configs = configs
    ckpoint_path = ckp_path
    data_opt = f"data_{data_name}.yaml"
    fr_data_opt = f"fr_data_{data_name}.yaml"

    # config = OmegaConf.load(configs_prefix + configs + configs_postfix)
    # config.model.params.dim = dim
    # config.model.params.num_tokens = num_tokens
    # config.lightning.callbacks.test_recoder.params.method_name = f"{configs.upper()}_dim{dim}_tk{num_tokens}"
    # config.lightning.name = f"{configs.upper()}_dim{dim}_tk{num_tokens}"
    # OmegaConf.save(config, configs_prefix + configs + configs_postfix)
    cmd = (f"CUDA_VISIBLE_DEVICES={gpu_ids} python main.py --base {configs_prefix + configs + configs_postfix} "
           f"--phase train "
           f"--data_opt {data_prefix + data_opt} "
           f"--fr_data_opt {data_prefix + fr_data_opt}")
    retcode = os.system(cmd)


if __name__ == '__main__':
    gpu_ids = "7"
    configs = "inr_diff"  # "vanilla" #
    ckpoint_path = "./logs/2024-11-18T00-57-31_arbrpn_GF1/checkpoints/epoch=383.ckpt"
    singledata_train(gpu_ids, configs, ckpoint_path, "wv2", iscompored=False)
