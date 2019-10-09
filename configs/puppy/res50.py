import torch.nn as nn
import numpy as np
from torchvision import transforms as T
model = dict(
    type='Classifier',
    #pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        style='pytorch'),
    loss_head=dict(
        type='MultilabelsHead',
        num_classes=23,
        in_channels=2048
    ))

test_cfg = dict(score_thr=0.8)

dataset_type = 'CustomClassifyDataset'
data_root = 'food_data/'
classes = ('potato',
           'cherry tomato',
           'color pepper',
           'orange',
           'onion',
           'kiwifruit',
           'maize',
           'gumbo',
           'carrot',
           'mango',
           'apple',
           'pea pods',
           'cauliflower',
           'broccoli',
           'tomato',
           'needle mushroom',
           'green pepper',
           'mushroom',
           'banana',
           'egg',
           'hyacinth bean',
           'other')
img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train_labels.txt',
        img_prefix=data_root,
        transforms = T.Compose([
            T.RandomRotation(20),
            T.RandomHorizontalFlip(),
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(np.array(img_norm_cfg['mean']),
                        np.array(img_norm_cfg['std']))]), 
        img_norm_cfg=img_norm_cfg),
     test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val_labels.txt',
        img_prefix=data_root,
        transforms = T.Compose([
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(np.array(img_norm_cfg['mean']),
                        np.array(img_norm_cfg['std']))]),
        img_norm_cfg=img_norm_cfg))

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 3)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')])
total_epochs = 200
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/food_cls/res50'
load_from = None
resume_from = None
workflow = [('train', 1)]



