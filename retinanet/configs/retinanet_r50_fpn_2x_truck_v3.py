_base_ = [
    "/home/ipad_ocr/ZLH/MMLab/CV_Task/configs/retinanet_r50_fpn_2x_truck_v2.py"
]

work_dir = "/home/ipad_ocr/ZLH/MMLab/CV_Task/retinanet_r50_fpn_2x_truck_v3/"
base_prefix = "/home/ipad_ocr/ZLH/MMLab/CV_Task/"

data_root = "dataset/"
train_file = "annotations/aug_train_det_data.json"
val_file = "annotations/val_det_data.json"
img_prefix = "aug_images/"
classes = ("slagcar", "excavator", "bulldozer", "soilcompactor")

# change the dataset setting
train_dataloader = dict(
    dataset=dict(
        data_root=base_prefix + data_root,
        ann_file=train_file,
        data_prefix=dict(img=img_prefix),
        metainfo=dict(classes=classes)
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=base_prefix + data_root,
        ann_file=val_file,
        data_prefix=dict(img=img_prefix),
        metainfo=dict(classes=classes)
    )
)
test_dataloader = val_dataloader

# change the val evaluator
val_evaluator = dict(
    ann_file=base_prefix + data_root + val_file
)
test_evaluator = val_evaluator
