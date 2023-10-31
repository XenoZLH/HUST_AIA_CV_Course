# using retinanet as detector
_base_ = [
    "/home/ipad_ocr/ZLH/MMLab/mmdetection/configs/retinanet/retinanet_r50_fpn_2x_coco.py"
]

# change the num_class to 4
model = dict(
    bbox_head=dict(
        num_classes=4
    )
)

work_dir = "/home/ipad_ocr/ZLH/MMLab/CV_Task/"

data_root = "dataset/"
train_file = "annotations/train_det_data.json"
val_file = "annotations/val_det_data.json"
img_prefix = "images/"
classes = ("slagcar", "excavator", "bulldozer", "soilcompactor")
# change the dataset setting
train_dataloader = dict(
    dataset=dict(
        data_root=work_dir + data_root,
        ann_file=train_file,
        data_prefix=dict(img=img_prefix),
        metainfo=dict(classes=classes)
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=work_dir + data_root,
        ann_file=val_file,
        data_prefix=dict(img=img_prefix),
        metainfo=dict(classes=classes)
    )
)
test_dataloader = val_dataloader

# change the val evaluator
val_evaluator = dict(
    ann_file=work_dir + data_root + val_file
)
test_evaluator = val_evaluator

# validate every 2 epoch
train_cfg = dict(val_interval=2)

# load pre-trained model weights
load_from = work_dir + "retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
