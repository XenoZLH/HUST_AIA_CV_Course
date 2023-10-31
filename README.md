# HUST_AIA_CVcourse
华中科技大学人工智能与自动化学院“计算机视觉课程设计”<br>
选题：非法挖山作业车识别方法

## 问题简介
### 任务描述
设计一个基于深度学习的目标识别模型，对无人机获取的图像进行检测，识别出作业车，并给出其类型和检测框。<br>
1. 识别目标：渣土车、挖掘机、推土机、压土机
2. 难点：无人机飞行高度、镜头焦距等参数变化，使得获取的图像目标尺度变化大；作业车目标数量不均匀，渣土车目标多，其他目标少；作业车类型多样，颜色、规格等因素使得同一类型的目标存在多种变式。

### 数据集说明
作业车训练数据集包括291张图像、标签和类别信息，图像均为无人机拍摄获取。其中，渣土车目标数量较多，压土机目标数量少。由于原始数据并非coco格式，自行根据原始数据生成了coco格式的json标注文件。数据集结构如下：<br>
```
Dataset
|- images
|- labels
|- annotations
|  |- train_det_data.json
|  |- val_det_data.json
|- classes.txt
|- det_data.json
```

### 评价指标
#### Precision
$Precision=\frac {TP} {TP + FP}$
#### Recall
$Recall=\frac {TP} {TP + FN}$

## 实验记录
update2023.10.31 <br>
使用mmdet进行基准性能实验，选用retinanet模型作为检测器，随机选取41个样本作为validation set，余下样本作为train set。使用coco评测器进行性能测试，结果如下：
| model | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| -- | -- | -- | -- | -- | -- | -- |
| Retinanet | 0.6380 | 0.8720 | 0.7340 | 0.0 | 0.4280 | 0.7450 |

对于precision和recall的评测需要自行实现新的评测器 <br>