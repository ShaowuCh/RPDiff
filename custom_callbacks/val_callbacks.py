# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 16:46
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : val_callbacks.py
# @Software:  PyCharm python3.9
# @Description：
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from lightning_utilities.core.rank_zero import rank_zero_only

from custom_callbacks.utils import save_gt, log_img_local, save_record2csv


class RecordCallback(pl.Callback):
    def __init__(self, log_dir, epoch_freq=1):
        super().__init__()
        self.epoch_freq = epoch_freq

        self.local_dir = os.path.join(log_dir, "images", "val")
        os.makedirs(self.local_dir, exist_ok=True)

        record_dir = os.path.join(log_dir, "records", )
        os.makedirs(record_dir, exist_ok=True)
        self.record_path = os.path.join(record_dir, "val.csv")

    @rank_zero_only
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if trainer.sanity_checking: return
        epoch = trainer.current_epoch
        if epoch == 0:
            save_gt(self.local_dir, dataloaders=trainer.val_dataloaders)

        if epoch % self.epoch_freq == 0:
            if pl_module.val_rec:
                with torch.no_grad():
                    imgs = torch.cat(pl_module.val_rec, dim=0).detach().cpu()
                    log_img_local(self.local_dir, imgs, f"rec_epoch_{trainer.current_epoch}")

        pl_module.val_rec = []
        metrics = trainer._results.metrics(False)["callback"]
        save_record2csv(self.record_path, metrics, epoch)
