# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 21:17
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : base_plmodel.py
# @Software:  PyCharm python3.9
# @Description：
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import nn

from metrics.indexes_evaluation import indexes_evaluation
from metrics.indexes_evaluation_FS import indexes_evaluation_FS


def calculate_metrics_fr(sr, lr, pan, sensor, ms):
    metrics = {'D_lambda': 0.0, 'D_S': 0.0, 'HQNR_index': 0.0}
    with torch.no_grad():
        xrec = sr
        b = lr.shape[0]
        for i in range(b):
            img_lr = rearrange(lr[i], "c h w -> h w c")
            img_xrec = rearrange(xrec[i], 'c h w -> h w c')
            img_ms = rearrange(ms[i], 'c h w -> h w c')
            img_pan = rearrange(pan[i], 'c h w -> h w c')
            img_xrec = torch.clamp(img_xrec, 0.0, 1)
            img_lr = img_lr.detach().cpu().numpy()
            img_ms = img_ms.detach().cpu().numpy()
            img_pan = img_pan.detach().cpu().numpy()
            img_xrec = img_xrec.detach().cpu().numpy()
            # I_F, I_MS_LR, I_PAN, th_values, I_MS, sensor, ratio, Qblocks_size
            metric = indexes_evaluation_FS(img_xrec, img_lr, img_pan, 0, img_ms, sensor[i], ratio=4, Qblocks_size=32)
            metrics = {k: v + metric[k] for k, v in metrics.items()}
        metrics = {k: v / b for k, v in metrics.items()}
    return metrics


def calculate_metrics(sr, gt):
    metrics = {"q2n": 0., "sam": 0., "ergas": 0., "scc": 0.}
    with torch.no_grad():
        xrec = sr
        b = gt.shape[0]
        for i in range(b):
            img_gt = rearrange(gt[i], "c h w -> h w c")
            img_xrec = rearrange(xrec[i], 'c h w -> h w c')
            img_gt = torch.clamp(img_gt, 0.0, 1)
            img_xrec = torch.clamp(img_xrec, 0.0, 1)
            img_gt = img_gt.detach().cpu().numpy()
            img_xrec = img_xrec.detach().cpu().numpy()
            metric = indexes_evaluation(img_xrec, img_gt, ratio=4, Qblocks_size=32, flag_cut_bounds=1, dim_cut=21,
                                        th_values=0)
            metrics = {k: v + metric[k] for k, v in metrics.items()}
        metrics = {k: v / b for k, v in metrics.items()}
    return metrics


class BaseModel(pl.LightningModule):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.criterion = None
        self.isFR = False
        self.current_phase = None  # train or val or test

        self.val_rec = []
        self.test_res = {'gt': [], 'lms': [], 'pan': [], 'sr': []}

    def forward(self, batch):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        self.current_phase = 'train'
        # Note：调用Forward函数尽量在前，因为在Forward里可能修改batch中的内容
        out = self(batch)
        B = batch['GT'].shape[0]
        gt = batch['GT']
        loss = self.criterion(out['sr'], gt) if out['loss'] is None else out['loss']
        self.log('train/loss', loss, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        self.current_phase = 'val'
        B = batch['GT'].shape[0]
        out = self(batch)
        gt = batch['GT']
        loss = self.criterion(out['sr'], gt) if out['loss'] is None else out['loss']
        self.log('val/loss', loss, batch_size=B)
        metrics = calculate_metrics(out['sr'], gt)
        self.log_dict(metrics, prog_bar=True, batch_size=B)
        self.val_rec.append(out['sr'])
        return metrics

    def test_step(self, batch, batch_idx):
        self.current_phase = 'test'
        out = self(batch)
        if self.isFR:
            # LR have changed to gt in dataset_fr
            ms = nn.functional.interpolate(batch['LR'], scale_factor=4, mode='bilinear')
            batch['GT'] = ms
            # batch['sensor'] = batch['sensor'][0] if isinstance(batch['sensor'], list) else batch['sensor']
            metrics = calculate_metrics_fr(out['sr'], batch['LR'], batch['REF'], batch['sensor'], ms)

        else:
            gt = batch['GT']
            loss = self.criterion(out['sr'], gt) if out['loss'] is None else out['loss']
            self.log('test/loss', loss)
            metrics = calculate_metrics(out['sr'], gt)
        self.log_dict(metrics, prog_bar=True)
        img_range = batch["img_range"][:, None, None, None]
        self.save_testdata(out['sr'], batch, img_range)

        return metrics

    def configure_optimizers(self):
        raise NotImplementedError

    def save_testdata(self, sr, batch, img_range):
        if self.isFR:
            pan, lms, ms = batch['REF'], batch['LR'], batch['GT']
            pan, ms, lms, sr = map(lambda x: (x * img_range).permute(0, 2, 3, 1).cpu().numpy(), [pan, ms, lms, sr])
            self.test_res['pan'].append(pan)
            self.test_res['gt'].append(ms)  # Up sampled lms using bi-linear, gt is ms
            self.test_res['lms'].append(lms)
            self.test_res['sr'].append(sr)
        else:
            pan, lms, gt = batch['REF'], batch['LR'], batch['GT']
            pan, gt, lms, sr = map(lambda x: (x * img_range).permute(0, 2, 3, 1).cpu().numpy(), [pan, gt, lms, sr])
            self.test_res['pan'].append(pan)
            self.test_res['lms'].append(lms)
            self.test_res['sr'].append(sr)
            self.test_res['gt'].append(gt)

        # if self.isFR:
        #     pan, ms, lms, sr = map(lambda x: (x * img_range).permute(0, 2, 3, 1).cpu().numpy(), [pan, ms, lms, sr])
        #     self.test_res['pan'].append(pan)
        #     self.test_res['ms'].append(ms)
        #     self.test_res['lms'].append(lms)
        #     self.test_res['sr'].append(sr)
        # else:
        #     pan, gt, lms, sr = map(lambda x: (x * img_range).permute(0, 2, 3, 1).cpu().numpy(), [pan, gt, lms, sr])
        #     self.test_res['pan'].append(pan)
        #     self.test_res['lms'].append(lms)
        #     self.test_res['sr'].append(sr)
        #     self.test_res['gt'].append(gt)
