_base_ = [
    "/home/ipad_ocr/ZLH/MMLab/CV_Task/configs/retinanet_r50_fpn_2x_truck.py"
]

work_dir = "/home/ipad_ocr/ZLH/MMLab/CV_Task/retinanet_r50_fpn_2x_truck_v2/"

backend_args = None
# scale = [(1920, 1080), (1333, 800)]
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1920, 1080), keep_ratio=True),
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

train_batch_size = 2  # num of data per GPU
train_dataloader = dict(
    batch_size=train_batch_size,
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline)
)
test_dataloader = val_dataloader

# increase training epochs to 32
train_cfg = dict(max_epochs=24, type='EpochBasedTrainLoop', val_interval=2)
# adjust the hooks
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=4))
