# YOLOv5 Model
## 介绍
本仓库基于YOLOv8模型，用于保存HUST_AIA计算机视觉课设的代码。YOLOv8模型是[Ultralytics](https://www.ultralytics.com/)公司提出的一个高效目标分类模型，原代码链接位于[github仓库](https://github.com/ultralytics/ultralytics),关于它的介绍就不再赘述
## 使用方法
### 准备环境
使用[miniconda(推荐)或者anaconda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/download.html#anaconda-or-miniconda)进行环境配置。

首先安装[conda](https://www.anaconda.com/download/)或者[miniconda](https://docs.conda.io/en/latest/miniconda.html),然后依次运行如下代码
```
conda create -n yolov8 python=3.8
conda activate yolov8
pip install -r requirements.txt
```
### 准备数据集
首先需要将老师提供的数据集放在[yolov8](./)根文件夹下,并按照如下方式进行组织：
```
dataset
|- images
|- labels
|- classes.txt
|- train_det_data.json
|- val_det_data.json
```
其中，[train_det_data.json](../retinanet/Dataset/annotations/train_det_data.json)和[val_det_data.json](../retinanet/Dataset/annotations/val_det_data.json)为队友分割的测试和训练标签的coco格式json文件，需要自行加入。

根据train_det_data.json和val_det_data.json，我们可以运行代码`python split_dataset_fromjson.py`生成yolov5需要的文件格式，即在images和labels里面各自拆分训练和测试数据，并得到如下数据集组织结构:
```
dataset
|- images
|   |- train
|   |- test
|- labels
|   |- train
|   |- test
|- classes.txt
|- train_det_data.json
|- val_det_data.json
```
### 训练模型
运行如下代码
```
python train.py
```
其中，可以更改train.py中的config, cfg, data, hyp, batch-size, imgsz等参数，其中，当前的参数对应我本地环境：
```
OS: Ubuntu 22.04.3LST
NVIDIA driver: 535.113.01
CUDA version:12.2
NVIDA GeForce GTX TITAN X
```
### 验证模型
运行代码
```
python val.py
```
### 验证效果
当采用默认配置文件进行网络参数的配置，载入预训练模型[yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)得到的训练效果
| Class | Images | Instances | P | R | mAP<sub>50</sub> | mAP<sub>50-95</sub> 
| -- | -- | -- | -- | -- | -- | -- |
| all | 41 | 197 | 0.893 | 0.862 | 0.928 | 0.681 |
| slagcar | 41 | 167 | 0.95 | 0.801 | 0.926 | 0.622 |
| excavator | 41 | 26 | 1 | 0.648 | 0.794 | 0.454 |
| bulldozer | 41 | 3 | 0.862 | 1 | 0.995 | 0.753 |
| soilcompactor | 41 | 1 | 0.76 | 1 | 0.995 | 0.895 |

训练的的PR-curve如下图所示：
![pr-curve](runs/detect/val/PR_curve.png)