import copy
from collections import Sequence

import mmcv
from mmcv.runner import obj_from_dict
import torch

import matplotlib.pyplot as plt
import numpy as np
from ..concat_dataset import ConcatDataset
from ... import datasets


def get_dataset(data_cfg):
    if data_cfg['type'] == 'RepeatDataset':
        return RepeatDataset(
            get_dataset(data_cfg['dataset']), data_cfg['times'])

    if isinstance(data_cfg['ann_file'], (list, tuple)):
        ann_files = data_cfg['ann_file']
        num_dset = len(ann_files)
    else:
        ann_files = [data_cfg['ann_file']]
        num_dset = 1

    if isinstance(data_cfg['img_prefix'], (list, tuple)):
        img_prefixes = data_cfg['img_prefix']
    else:
        img_prefixes = [data_cfg['img_prefix']] * num_dset
    assert len(img_prefixes) == num_dset

    dsets = []
    for i in range(num_dset):
        data_info = copy.deepcopy(data_cfg)
        data_info['ann_file'] = ann_files[i]
        data_info['img_prefix'] = img_prefixes[i]
        dset = obj_from_dict(data_info, datasets)
        dsets.append(dset)
    if len(dsets) > 1:
        dset = ConcatDataset(dsets)
    else:
        dset = dsets[0]
    return dset
