# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 16:27
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : train_callbacks.py
# @Software:  PyCharm python3.9
# @Description：
import os
import shutil

import pytorch_lightning as pl
from lightning_utilities.core.rank_zero import rank_zero_only

from custom_callbacks.utils import save_record2csv


def get_yaml_files(directory):
    yaml_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                yaml_files.append(os.path.join(root, file))
    return yaml_files


class RecordCallback(pl.Callback):
    def __init__(self, log_dir, method_name, dataset_name, epoch_freq=1):
        super().__init__()
        root = os.path.join(log_dir, "records", )
        os.makedirs(root, exist_ok=True)

        filename = f"train.csv"
        self.path = os.path.join(root, filename)
        self.log_dir = log_dir
        self.epoch_freq = epoch_freq
        self.method_name = method_name
        self.dataset_name = dataset_name

    @rank_zero_only
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking: return
        # metrics = trainer.callback_metrics 这个会保存所有结果，包括 val中的
        epoch = trainer.current_epoch
        if epoch % self.epoch_freq == 0:
            metrics = trainer._results.metrics(False)["callback"]
            save_record2csv(self.path, metrics, epoch)

    @rank_zero_only
    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking: return
        # get the checkpoint path
        best_checkpoint_path = trainer.checkpoint_callback.best_model_path
        last_checkpoint_path = trainer.checkpoint_callback.last_model_path

        model_hub_path = f"./model_hub/{self.method_name}_{self.dataset_name}"
        os.makedirs(model_hub_path, exist_ok=True)

        dst_file_path = os.path.join(f"./model_hub/{self.method_name}_{self.dataset_name}", "best_checkpoint.ckpt")
        shutil.copyfile(best_checkpoint_path, dst_file_path)
        shutil.copyfile(last_checkpoint_path, os.path.join(f"./model_hub/{self.method_name}_{self.dataset_name}",
                                                           "last_checkpoint.ckpt"))

        config_path = os.path.join(self.log_dir, "configs")
        for yaml_file in get_yaml_files(config_path):
            shutil.copyfile(yaml_file, os.path.join(f"./model_hub/{self.method_name}_{self.dataset_name}",
                                                    os.path.basename(yaml_file)))
