# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 14:58
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : main.py.py
# @Software:  PyCharm python3.9
# @Description：

import datetime
import os
import sys
from pathlib import Path

import torch
from pytorch_lightning import seed_everything, Trainer
from omegaconf import OmegaConf
from pytorch_lightning.loggers import CSVLogger

from utils.utils import get_parser, get_logidr, instantiate_from_config


def getstrtime():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    return now


def get_opts(opt, i, j):
    cfg_yaml = OmegaConf.load(opt.base[i])
    model_opt = cfg_yaml.model
    data_opt = OmegaConf.load(opt.data_opt[j]).data
    fr_data_opt = OmegaConf.load(opt.fr_data_opt[j]).data
    cfg_yaml.data = data_opt
    cfg_yaml.fr_data = fr_data_opt
    lightning_opt = cfg_yaml.lightning

    lightning_opt.callbacks.test_recoder.params.dataset_name = data_opt.name
    lightning_opt.callbacks.train_recoder.params.dataset_name = data_opt.name
    lightning_opt.callbacks.test_recoder.params.method_name = lightning_opt.name
    lightning_opt.callbacks.train_recoder.params.method_name = lightning_opt.name

    lightning_opt.postfix = f"_{data_opt.name}"
    lightning_opt.phase = opt.phase if opt.phase == "test" else lightning_opt.phase
    lightning_opt.name = False if opt.phase == "test" else lightning_opt.name
    lightning_opt.resume = opt.resume if opt.phase == "test" else False

    if "selfdefined_load_state_dict" not in lightning_opt:
        lightning_opt.selfdefined_load_state_dict = False
    return opt, model_opt, data_opt, fr_data_opt, lightning_opt, cfg_yaml


def set_callbacks_params(lightning_opt, now, logdir, ckptdir, cfgdir, cfg_yaml):
    lightning_opt.callbacks.setup_callback.params.resume = lightning_opt.resume
    lightning_opt.callbacks.setup_callback.params.now = now
    lightning_opt.callbacks.setup_callback.params.logdir = logdir
    lightning_opt.callbacks.setup_callback.params.ckptdir = ckptdir
    lightning_opt.callbacks.setup_callback.params.cfgdir = cfgdir
    lightning_opt.callbacks.setup_callback.params.config = cfg_yaml

    lightning_opt.callbacks.modelckpt.params.dirpath = ckptdir
    lightning_opt.callbacks.modelckpt.params.filename = '{epoch:03d}'

    local_dir = os.path.join(logdir, 'local')
    lightning_opt.callbacks.train_recoder.params.log_dir = local_dir
    lightning_opt.callbacks.val_recoder.params.log_dir = local_dir
    lightning_opt.callbacks.test_recoder.params.log_dir = local_dir
    return lightning_opt


def get_ckpt_path(ckptdir: str, type: str):
    # type = "best" or "last"
    directory = Path(ckptdir)
    files = [file.name for file in directory.iterdir() if file.is_file() and "last" not in file.name]
    if type == "best":
        return os.path.join(ckptdir, files[0])
    elif type == "last":
        return os.path.join(ckptdir, "last.ckpt")


def main(opt, i, j):
    now = getstrtime()
    opt, model_opt, data_opt, fr_data_opt, lightning_opt, cfg_yaml = get_opts(opt, i, j)
    logdir, nowname, ckptdir, cfgdir = get_logidr(lightning_opt, now)
    lightning_opt = set_callbacks_params(lightning_opt, now, logdir, ckptdir, cfgdir, cfg_yaml)

    seed_everything(opt.seed, workers=True)

    model = instantiate_from_config(model_opt)
    callbacks = [instantiate_from_config(lightning_opt.callbacks[k]) for k in lightning_opt.callbacks]
    csv_logger = CSVLogger(save_dir=os.path.join(logdir, 'csv_log'))
    trainer = Trainer(callbacks=callbacks, logger=csv_logger,
                      **lightning_opt.trainer)  # logger=wandb_logger, csv_logger

    data = instantiate_from_config(data_opt)
    data.setup()

    fr_data = instantiate_from_config(fr_data_opt)
    fr_data.setup()

    if lightning_opt.phase == "train":
        bs, base_lr = data_opt.params.batch_size, model_opt.base_learning_rate
        ngps = len(lightning_opt.trainer.devices)
        model.learning_rate = lightning_opt.trainer.accumulate_grad_batches * base_lr * ngps  # 这个计算学习率的方案保存一下

        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()}")

        if lightning_opt.resume:
            trainer.fit(model, datamodule=data, ckpt_path=lightning_opt.resume)
        else:
            trainer.fit(model, datamodule=data)
        # TODO: 测试
        if lightning_opt.selfdefined_load_state_dict:
            model_path = get_ckpt_path(ckptdir, "best")
            state_dict = torch.load(model_path)["state_dict"]  # 使用pytorch lightning自动保存时，保存的参数名是state_dict
            model.load_state_dict(state_dict)
            model.isFR = False
            trainer.test(model, data)
            model.isFR = True
            trainer.test(model, fr_data)

        else:
            model.isFR = False
            trainer.test(model, data, ckpt_path='best')
            model.isFR = True
            trainer.test(model, fr_data, ckpt_path='best')
    elif lightning_opt.phase == "test":
        # test时需要设置resume参数
        if lightning_opt.selfdefined_load_state_dict:
            model_path = lightning_opt.resume
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
            model.isFR = False
            trainer.test(model, data)
            model.isFR = True
            trainer.test(model, fr_data)

        else:
            model.isFR = False
            trainer.test(model, data, ckpt_path=lightning_opt.resume)
            # model.isFR = True
            # trainer.test(model, fr_data, ckpt_path=lightning_opt.resume)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    for i in range(len(opt.base)):  # methods loop
        for j in range(len(opt.data_opt)):  # datasets loop
            main(opt, i, j)
