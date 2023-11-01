_base_ = [
    "/home/ipad_ocr/ZLH/MMLab/CV_Task/configs/retinanet_r50_fpn_2x_truck.py"
]

train_batch_size = 6
train_dataloader = dict(batch_size=train_batch_size)

# increase training epochs to 32
train_cfg = dict(max_epochs=32, type='EpochBasedTrainLoop', val_interval=2)
