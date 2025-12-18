# -*- coding: utf-8 -*-
# @Time    : 2024/11/8 10:42
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : single_test.py
# @Software:  PyCharm python3.9
# @Description：

import os


def singledata_test(gpu_ids: str, configs: str, ckp_path: str, data_name: str, is_compare: bool=False):
    """
    :param gpu_ids:  "1,3"
    :param configs:  "method_name or config files name"
    :return:
    """
    gpu_ids = gpu_ids

    configs_prefix = "./configs/" if not is_compare else "./configs/compared_methods/" # "./configs/compared_methods/"
    # configs_prefix = "./configs/ablation/"
    configs_postfix = ".yaml"
    data_prefix = "./configs/data_opts/"

    configs = configs
    ckpoint_path = ckp_path
    data_opt = f"data_{data_name}.yaml"
    fr_data_opt = f"fr_data_{data_name}.yaml"

    cmd = (f"CUDA_VISIBLE_DEVICES={gpu_ids} python main.py --base {configs_prefix + configs + configs_postfix} "
           f"--phase test "
           f"--resume {ckpoint_path} "
           f"--data_opt {data_prefix + data_opt} "
           f"--fr_data_opt {data_prefix + fr_data_opt}")
    retcode = os.system(cmd)
    print(retcode)


if __name__ == '__main__':
    gpu_ids = "0"
    configs ="mslpt" #"vanilla"
    ckpoint_path = "./model_hub/MSLPT_dim32_tk256_stopqv_10epoch_QB/best_checkpoint.ckpt"
    singledata_test(gpu_ids, configs, ckpoint_path, "qb", is_compare=False)
