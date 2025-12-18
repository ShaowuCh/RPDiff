# -*- coding: utf-8 -*-
# @Time    : 2024/10/22 15:07
# @Author  : Wu Shaowu
# @Email   : wshaow@whu.edu.cn
# @File    : utils.py
# @Software:  PyCharm python3.9
# @Description：


import argparse, os, importlib
import glob


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        const=True,
        default="train",
        nargs="?",
        help="flag of train or test, value train or test, default train",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-d",
        "--data_opt",
        nargs="*",
        metavar="data_config.yaml",
        help="paths to dataset configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=["./configs/data_opts/data_gf1.yaml"],
    )

    parser.add_argument(
        "-fd",
        "--fr_data_opt",
        nargs="*",
        metavar="data_config.yaml",
        help="paths to dataset configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=["./configs/data_opts/fr_data_gf1.yaml"],
    )

    return parser


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_logidr(opt, now):
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    if opt.resume: # for test and resume
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            if 'logs' in opt.resume:
                paths = opt.resume.split("/")
                idx = len(paths) - paths[::-1].index("logs") + 1
                logdir = "/".join(paths[:idx])
                ckpt = opt.resume
            else:
                ckpt = opt.resume
                return "./test_del","./test_del/checkpoints",ckpt,"./test_del/configs"

        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs") + 1]
    else:
        if opt.name:
            name = "_" + opt.name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    return logdir, nowname, ckptdir, cfgdir
