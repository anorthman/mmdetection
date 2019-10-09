import os
import cv2
from PIL import Image
import torch

import mmcv
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


class CustomClassifyDataset(Dataset):
    
    def __init__(self, 
                 classes,
                 ann_file,
                 img_prefix,
                 transforms,
                 img_norm_cfg,
                 test_mode=False,
                 ):
        self.classes = classes
        self.img_prefix = img_prefix
        self.transforms = transforms
        self.img_norm_cfg = img_norm_cfg
        self.test_mode = test_mode
        self.img_infos = self.load_labels(ann_file)
        if not self.test_mode:
            self._set_group_flag()

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def load_labels(self, ann_file):
        img_infos = []
        with open(ann_file, 'r') as f:
            for line in f.readlines():
                path, labels = line.strip().split(' ')
                img = cv2.imread(os.path.join(self.img_prefix, path))
                h, w, _ = img.shape
                label = np.array([float(i) for i in labels.strip().split(',')])
                img_infos.append(dict(path=path, label=label, height=h, width=w))
        return img_infos

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        return self.prepare_train_img(idx)


    def prepare_test_img(self, idx):
        img_path = self.img_infos[idx]['path']
        label = torch.Tensor(self.img_infos[idx]['label'])
        img = cv2.imread(os.path.join(self.img_prefix, img_path))
        if self.img_norm_cfg.to_rgb:
            img = img[..., ::-1]
        img = self.transforms(Image.fromarray(img))
        return dict(img=img, label=label)

    def prepare_train_img(self, idx):
        img_path = self.img_infos[idx]['path']
        label = torch.Tensor(self.img_infos[idx]['label'])
        img = cv2.imread(os.path.join(self.img_prefix, img_path))
        if self.img_norm_cfg.to_rgb:
            img = img[...,::-1]
        img = self.transforms(Image.fromarray(img))
        return dict(img=img, label=label)

    def __len__(self):
        return len(self.img_infos)
        
        
        
                 
