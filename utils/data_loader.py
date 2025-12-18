# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 15:40
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : data_loader.py
# @Software:  PyCharm python3.9
# @Description：
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, default_collate, ConcatDataset
from utils.utils import instantiate_from_config
from data.collate_func import mix_collect_func

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, collect_type=1):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.datasets = dict()
        if collect_type == 0:
            self.collect_func = default_collate
        else:
            self.collect_func = mix_collect_func

    def setup(self, stage=None):
        for phase, v_config in self.dataset_configs.items():
            print(phase)
            print(v_config)
            self.datasets[phase] = [instantiate_from_config(v_config[k]) for k in v_config]
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(ConcatDataset(self.datasets["train"]), batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=self.collect_func,
                          persistent_workers=True)

    def _val_dataloader(self):
        return DataLoader(ConcatDataset(self.datasets["validation"]),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, collate_fn=self.collect_func,
                          persistent_workers=True)

    def _test_dataloader(self):
        return DataLoader(ConcatDataset(self.datasets["test"]), batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, collate_fn=self.collect_func,
                          persistent_workers=True)
