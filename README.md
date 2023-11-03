# HUST_AIA_CVcourse
华中科技大学人工智能与自动化学院“计算机视觉课程设计”<br>
选题：非法挖山作业车识别方法

## 问题简介
### 任务描述
设计一个基于深度学习的目标识别模型，对无人机获取的图像进行检测，识别出作业车，并给出其类型和检测框。<br>
1. 识别目标：渣土车、挖掘机、推土机、压土机
2. 难点：无人机飞行高度、镜头焦距等参数变化，使得获取的图像目标尺度变化大；作业车目标数量不均匀，渣土车目标多，其他目标少；作业车类型多样，颜色、规格等因素使得同一类型的目标存在多种变式。

### 数据集说明
作业车训练数据集包括291张图像、标签和类别信息，图像均为无人机拍摄获取。其中，渣土车目标数量较多，压土机目标数量少。由于原始数据并非coco格式，自行根据原始数据生成了coco格式的json标注文件。<br>
由于并未提供测试集，故从训练集中随机抽取了41张图片作为验证集，同时为了实现数据增强，分别将所有图像进行了水平和垂直翻转，扩增后的全部图片保存至`Dataset/aug_images`下，扩增后的标注文件为`aug_train_det_data.json`，`aug_val_det_data.json`。<br>
数据集结构如下：<br>
```
Dataset
|- images
|- aug_images
|- labels
|- annotations
|  |- train_det_data.json
|  |- val_det_data.json
|  |- aug_train_det_data.json
|  |- aug_val_det_data.json
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
使用mmdet进行基准性能实验，选用retinanet模型作为检测器，使用mmdet默认retinanet的配置文件设置超参数。使用coco评测器进行性能测试，结果如下：
| model | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| -- | -- | -- | -- | -- | -- | -- |
| Retinanet | 0.638 | 0.872 | 0.734 | 0.0 | 0.428 | 0.745 |

update2023.11.3<br>
基于retinanet进行了新的实验：<br>
（1）考虑到数据集中存在一定的小尺寸样本，默认配置的1333×800分辨率可能会丢失信息，因此将分辨率设置为1920×1080进行测试
| model | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| -- | -- | -- | -- | -- | -- | -- |
| Retinanet | 0.628 | 0.886 | 0.724 | 0.0 | 0.494 | 0.727 |

（2）对数据集进行上下翻转和水平翻转，使用扩增后的数据集进行训练，分辨率设置为1920×1080
| model | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| -- | -- | -- | -- | -- | -- | -- |
| Retinanet | 0.583 | 0.812 | 0.677 | 0.0 | 0.428 | 0.731 |

（3）根据之前实验，1333×800分辨率的效果好于1920×1080，因此改用1333×800分辨率，在扩增数据集上进行实验
| model | AP | AP<sub>50</sub> | AP<sub>75</sub> | AP<sub>S</sub> | AP<sub>M</sub> | AP<sub>L</sub> |
| -- | -- | -- | -- | -- | -- | -- |
| Retinanet | 0.598 | 0.899 | 0.738 | 0.0 | 0.415 | 0.710 |

对于precision和recall的评测，计划使用`score_thr=0.5`时的输出作为基准 <br>