_base_ = [
    "/home/ipad_ocr/ZLH/MMLab/CV_Task/configs/retinanet_r50_fpn_2x_truck.py"
]

# data num is too small, need strong augmentation to increase the training data
backend_args = None
color_space = [
    [dict(type='ColorTransform', prob=0.5)],
    [dict(type='AutoContrast', prob=0.5)],
    [dict(type='Equalize', prob=0.5)],
    [dict(type='Sharpness', prob=0.5)],
    [dict(type='Posterize', prob=0.5)],
    [dict(type='Solarize', prob=0.5)],
    [dict(type='Color', prob=0.5)],
    [dict(type='Contrast', prob=0.5)],
    [dict(type='Brightness', prob=0.5)],
]
geometric = [
    # [dict(type='Rotate', prob=0.2)],
    # [dict(type='ShearX')],
    # [dict(type='ShearY')],
    [dict(type='TranslateX', prob=0.2)],
    [dict(type='TranslateY', prob=0.2)],
]
scale = [(1920, 1080), (1333, 800)]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='RandomFlip', direction=['horizontal', 'vertical'], prob=[0.5, 0.5]),
    # dict(type='Rotate', max_mag=15.0, prob=0.2),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_batch_size = 2  # num of data per GPU
train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline)
)
test_dataloader = val_dataloader
train_cfg = dict(max_epochs=48, type='EpochBasedTrainLoop', val_interval=2)
# learning rate policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=125),
    dict(
        type='MultiStepLR',
        begin=2,
        end=48,
        by_epoch=True,
        milestones=[16, 22, 34],
        gamma=0.1)
]
# adjust the hooks
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=8))
