import os
import cv2
from PIL import Image
import torch

import mmcv
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class ImageNetDataset(Dataset):

    def __init__(self,
                 data_root,
                 test_mode=False,**kwargs):
        self.classes = list(range(1000))
        normalize = T.Normalize(mean=[0.456], std=[1.0])
        #normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if not test_mode:
            traindir = os.path.join(data_root, 'train')
            self.dataset = ImageFolder(traindir, T.Compose([
                               T.Grayscale(num_output_channels=1),
                               T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),
                               normalize,
                               ]))
        else:
            valdir = os.path.join(data_root, 'val')
            self.dataset = ImageFolder(valdir, T.Compose([
                               T.Resize(256),
                               T.CenterCrop(224),
                               T.ToTensor(),
                               normalize,
                               ]))
        if not test_mode:
            self._set_group_flag()
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def __getitem__(self, idx):
        d = dict(img=self.dataset[idx][0], label=torch.tensor([self.dataset[idx][1]], dtype=torch.long))
        return d

    def __len__(self):
        return len(self.dataset)
        
        
        
                 
