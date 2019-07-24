from torch import nn 
# model settings
model = dict(
    type='RetinaNet',
    pretrained=None,#'modelzoo://resnet50',
    backbone=dict(
        type='Base',
        base=[
        nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(8),
        nn.PReLU(8),
        nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(16),
        nn.PReLU(16),
        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(32),
        nn.PReLU(32)
            ],
        inchn=32,
        depth=(7,7,7),
        strides=(2,2,2),
        activate=nn.PReLU,
        dilations=[1,1,1],
	group=[1,4,1,4,1,4,1,1,8,1,8,1,8,1,1,16,1,16,1,16,1],
        kernel_size=[3,3,1,3,1,3,1]*3,
        num_stages=3,
        out_indices=(0, 1, 2),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256],
        out_channels=128,
        start_level=0,
        activation='relu',
        normalize=dict(type="BN"),
        add_extra_convs=False,
        num_outs=3),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=2,
        in_channels=128,
        stacked_convs=1,
        feat_channels=128,
        group=[8],
        octave_base_scale=1,
        scales_per_octave=3,
        anchor_ratios=[1.0,1.5],
        anchor_strides=[16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    smoothl1_beta=0.11,
    gamma=2.0,
    alpha=0.25,
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=40,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
dataset_type = 'CustomDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[256, 256, 256], to_rgb=True)
    #mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file="data/newlibraf_info/train_normal.pkl",
    	img_prefix='./data/',
        img_scale=[(1300, 576),(1300,512),(1300,640)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=64,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        extra_aug=dict(
                photo_metric_distortion=dict(
                    brightness_delta=32,
                    contrast_range=(0.5, 1.5),
                    saturation_range=(0.5, 1.5),
                    hue_delta=18),
                random_crop=dict(
                    min_ious=(0.3, 0.5, 0.7, 0.9), min_crop_size=0.6,keep_ratio=True),
        resize_keep_ratio=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        #ann_file="./data/qa/test.pkl",#data/newlibraf_info/test_normal.pkl",
        ann_file="data/newlibraf_info/test_normal.pkl",
    	img_prefix='./data/',
        img_scale=(512, 512),
        resize_keep_ratio=False,
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cosine',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 3)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 250
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/retinanet_r50_fpn_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
