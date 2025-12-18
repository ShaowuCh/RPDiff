# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 15:28
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : setup.py
# @Software:  PyCharm python3.9
# @Description：
import os

from omegaconf import OmegaConf
from pytorch_lightning import Callback


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config)
            if not self.resume:
                OmegaConf.save(self.config,
                               os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass
