import os
import random
import shutil
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
import networks

import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumpertry
    from yaml import CLoader as Loader, CDumper as Dumper
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


yml_Loader, yml_Dumper = OrderedYaml()

def hist_line_stretch(img, nbins, bound=[0.02, 0.98]):
    def _line_strectch(img, nbins):
        nbins = int(nbins)
        # imgrange = [int(img.min()), np.ceil(img.max())]
        hist1, bins1 = np.histogram(img, bins=nbins, density=False)
        hist1 = hist1 / img.size
        cumhist = np.cumsum(hist1)
        lowThreshold = np.where(cumhist >= bound[0])[0][0]
        lowThreshold = bins1[lowThreshold]
        highThreshold = np.where(cumhist >= bound[1])[0][0]
        highThreshold = bins1[highThreshold]

        img[np.where(img < lowThreshold)] = lowThreshold
        img[np.where(img > highThreshold)] = highThreshold
        img = (img - lowThreshold) / (highThreshold - lowThreshold + np.finfo(np.float).eps)
        return img

    if img.ndim > 2:
        for i in range(img.shape[2]):
            img[:, :, i] = _line_strectch(img[:, :, i].squeeze(), nbins)
    else:
        img = _line_strectch(img, nbins)
    return img



def count_width(s, align_zh):
    s = str(s)

    count = 0
    for ch in s:
        if align_zh and u'\u4e00' <= ch <= u'\u9fff':  # 中文占两格
            count += 2
        else:
            count += 1

    return count


def print_dict_to_md_table(dict):
    columns, rows = [], []
    for key, value in dict.times():
        columns += [key]
        rows += [value]
    print_to_markdwon_table(columns, [rows])


def print_to_markdwon_table(column, rows, align_zh=False):
    widths = []
    column_str = ""
    separate = "----"
    separate_str = ""
    for ci, cname in enumerate(column):
        cw = count_width(cname, align_zh)
        for row in rows:
            item = row[ci]

            if count_width(item, align_zh) > cw:
                cw = count_width(item, align_zh)

        widths.append(cw)

        delete_count = count_width(cname, align_zh) - count_width(cname, False)

        column_str += f'|{cname:^{cw - delete_count + 2}}'
        separate_str += f'|{separate:^{cw + 2}}'

    column_str += "|"
    separate_str += "|"

    print(column_str)
    print(separate_str)

    for ri, row in enumerate(rows):
        row_str = ""
        for ci, item in enumerate(row):
            cw = widths[ci]

            delete_count = count_width(item, align_zh) - count_width(item, False)
            row_str += f'|{item:^{cw - delete_count + 2}}'

        row_str += "|"
        print(row_str)


def print_metrics(opt, testMetrics, idx_loader, loader_dataname, time):
    columns = ['Networks']
    rows = ['%8s' % opt['networks']['net_arch']]
    for key, value in testMetrics.items():
        columns += [key]
        rows += ['%.5f' % value]
    columns += ['Time']
    rows += ['%.5f' % (time)]
    print('[ valid ] The %d-th val set: [%s]' % (idx_loader + 1, loader_dataname))
    print_to_markdwon_table(columns, [rows])

def save_current_records(log, epoch, opt, data_name=None):
    data_name = data_name if data_name else "data"
    path = os.path.join(opt["logger"]["checkpoint_dir"], opt["networks"]["net_arch"] + "_".join(opt["logger"]["tags"]),
                        data_name.split("_")[0], "records")
    if not os.path.isdir(path):
        os.makedirs(path)
    data_frame = pd.DataFrame(
        data={key: value for key, value in log.items()},
        index=[epoch]
    )
    need_header = True if epoch == 1 else False
    data_name = data_name if data_name else ''
    data_frame.to_csv(os.path.join(path, f'train_records_{data_name}.csv'),
                      mode='a',encoding='utf-8',header=need_header, index=epoch)

def save_checkpoint(opt, model, optimizer, schedulre, log, is_best, data_name=None):
    """
    save checkpoint to experimental dir
    """
    data_name = data_name if data_name else "data"
    path = os.path.join(opt["logger"]["checkpoint_dir"], opt["networks"]["net_arch"] + "_".join(opt["logger"]["tags"]),
                        data_name, "check_point")
    if not os.path.isdir(path):
        os.makedirs(path)
    filename = os.path.join(path, 'last_ckp.pth')  #
    ckp = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_schedulre': schedulre.state_dict(),
        'log': log
    }
    torch.save(ckp, filename)
    if is_best:
        print('===> Saving best checkpoint to [%s] ...]' % filename.replace(
            'last_ckp', 'best_ckp'))
        torch.save(ckp, filename.replace('last_ckp', 'best_ckp'))

# def str2txt(opt, net_arch_str, n, exp_root):
#     net_lines = []
#     line = net_arch_str + '\n'
#     net_lines.append(line)
#     line = '====>Network structure: [{}], with parameters: [{:,d}]<===='.format(name, n)
#     net_lines.append(line)
#     net_lines = ''.join(net_lines)
#     if exp_root is not None and os.path.isdir(exp_root):
#         with open(os.path.join(exp_root, 'network_summary.txt'), 'w') as f:
#             f.writelines(net_lines)
#     return net_lines

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

################  { dir operation }  ################
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)

class NoneDict(dict):
    def __missing__(self, key):
        return None

def nonedict_to_dict(opt):
    if isinstance(opt, NoneDict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = nonedict_to_dict(sub_opt)
        return new_opt
    elif isinstance(opt, list):
        return [nonedict_to_dict(sub_opt) for sub_opt in opt]
    else:
        return opt

def save_setting(opt, Dumper=yml_Dumper):
    path = os.path.join(opt["logger"]["checkpoint_dir"], opt["networks"]["net_arch"] + "_".join(opt["logger"]["tags"]))
    dump_path = os.path.join(path, 'options.yml')
    network_file = opt["networks"]['net_arch'] + '.py'
    shutil.copy('./networks/' + network_file, os.path.join(path, network_file))
    with open(dump_path, 'w') as dump_file:
        yaml.dump(nonedict_to_dict(opt), dump_file, Dumper=Dumper)

def save_options_model_description(opt, model):
    path = os.path.join(opt["logger"]["checkpoint_dir"], opt["networks"]["net_arch"] + "_".join(opt["logger"]["tags"]))
    mkdir_and_rename(path)
    save_setting(opt)
    model_graph, params = networks.get_network_description(model)
    net_lines = []
    line = model_graph + '\n'
    net_lines.append(line)
    line = '====>Network structure: [{}], with parameters: [{:,d}]<===='.format(opt['networks']['net_arch'], params)
    net_lines.append(line)
    net_lines = ''.join(net_lines)

    if path is not None and os.path.isdir(path):
        with open(os.path.join(path, 'network_summary.txt'), 'w') as f:
            f.writelines(net_lines)

    return net_lines

select_band = lambda x: (2, 1, 0) if x.shape[2] > 2 else 0

def save_img(imgs, names, opt, img_id, epoch, data_name=None):
    data_name = data_name if data_name else "data"
    path = os.path.join(opt["logger"]["checkpoint_dir"], opt["networks"]["net_arch"] + "_".join(opt["logger"]["tags"]),
                        data_name.split("_")[0], "imgs", str(img_id))
    if not os.path.isdir(path):
        os.makedirs(path)
    for name in names:
        img = imgs[name][:, :, select_band(imgs[name])]
        img = hist_line_stretch(img.astype(float), nbins=255) * 255
        img = Image.fromarray(img.astype(np.uint8))

        img.save(os.path.join(path,f"{name}_{epoch}.jpg"))



if __name__ == "__main__":
    metrics = {"scc":None,
               "psnr":None,
               "sam":None,
               "egrs":None}
    for epoch in range(1,100):
        metrics["scc"] = random.random()
        metrics["psnr"] = random.random()
        metrics["sam"] = random.random()
        metrics["egrs"] = random.random()
        need_header = False
        if epoch == 1:
            need_header = True

        data_frame = pd.DataFrame(
            data={key: value for key, value in metrics.items()},
            index=[epoch]
        )
        data_frame.to_csv('test.csv',mode='a', encoding='utf-8', header=need_header, index=epoch)
