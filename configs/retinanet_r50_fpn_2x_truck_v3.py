_base_ = [
    "/home/ipad_ocr/ZLH/MMLab/CV_Task/configs/retinanet_r50_fpn_2x_truck.py"
]

# data num is too small, need strong augmentation to increase the training data
backend_args = None
color_space = [
    [dict(type='ColorTransform')],
    [dict(type='AutoContrast')],
    [dict(type='Equalize')],
    [dict(type='Sharpness')],
    [dict(type='Posterize')],
    [dict(type='Solarize')],
    [dict(type='Color')],
    [dict(type='Contrast')],
    [dict(type='Brightness')],
]
geometric = [
    [dict(type='Rotate')],
    [dict(type='ShearX')],
    [dict(type='ShearY')],
    [dict(type='TranslateX')],
    [dict(type='TranslateY')],
]
scale = [(1920, 1080), (1333, 800)]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=scale, keep_ratio=True),
    dict(type='RandAugment', aug_space=color_space, aug_num=1),
    dict(type='RandomFlip', prob=0.5),
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

train_batch_size = 4
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
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=48,
        by_epoch=True,
        milestones=[16, 32, 40],
        gamma=0.1)
]
# adjust the hooks
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=8))
