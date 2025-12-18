# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 17:32
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : test_callbacks.py
# @Software:  PyCharm python3.9
# @Description：
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from lightning_utilities.core.rank_zero import rank_zero_only
from scipy.io import savemat

from custom_callbacks.utils import save_record2csv


class RecordCallback(pl.Callback):
    def __init__(self, log_dir, method_name, dataset_name, epoch_freq=1):
        self.method_name = method_name
        self.dataset_name = dataset_name
        self.epoch_freq = epoch_freq
        self.local_dir = os.path.join(log_dir, "images", "test")
        os.makedirs(self.local_dir, exist_ok=True)

        record_dir = os.path.join(log_dir, "records", )
        os.makedirs(record_dir, exist_ok=True)
        self.record_path = os.path.join(record_dir, "test.csv")
    @rank_zero_only
    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # breakpoint()
        test_res = {k: np.concatenate(v, axis=0) for k, v in pl_module.test_res.items()}
        isFr = pl_module.isFR
        root = os.path.join("./results", self.method_name + '_' + self.dataset_name + f'{"_fr" if isFr else ""}')
        metrics = trainer._results.metrics(False)["callback"]
        metrics = {k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}
        data_frame = pd.DataFrame(data=metrics, index=[1])
        filename = f"test.csv"
        path = os.path.join(root, filename)
        os.makedirs(root, exist_ok=True)
        need_header = True
        data_frame.to_csv(path, mode='a', encoding='utf-8', header=need_header, index_label=1)

        mat_path = os.path.join(root, self.method_name + '_' + self.dataset_name + ".mat")
        savemat(mat_path, test_res)
        pl_module.test_res = {'gt': [], 'lms': [], 'pan': [], 'sr': []}

    @rank_zero_only
    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # TODO: save each test result as .bmp file
        metrics = trainer._results.metrics(False)["callback"]
        save_record2csv(self.record_path,metrics, trainer.current_epoch)
