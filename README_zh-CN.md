# 基于MindSpore框架的FCENet网络实现
<div align="center">

[English](README.md) | 简体中文

</div>

# 目录
- [FCENet 简介](#FCENet-简介)
- [性能](#性能)
- [数据集](#数据集)
- [预训练模型](#预训练模型)
- [配置环境](#配置环境)
- [快速入门](#快速入门)

# [FCENet 简介](#目录)

傅里叶轮廓嵌入网络（FCENet）是一种文本检测器，能够很好地检测自然场景中的任意形状的文本。任意形状文本检测的主要挑战之一是设计一个好的文本实例表示，使网络能够学习不同的文本几何差异。大多数现有的方法在图像空间域中通过掩码或直角坐标系中的轮廓点序列对文本实例进行建模。然而，遮罩表示可能会导致昂贵的后处理，而点序列表示可能对具有高度弯曲形状的文本建模能力有限。为了解决这些问题，在傅里叶域对文本实例进行建模，并提出一种新颖的傅里叶轮廓嵌入（FCE）方法，将任意形状的文本轮廓表示为紧凑的签名。进一步用主干、特征金字塔网络（FPN）和反傅里叶变换（IFT）和非最大抑制（NMS）的简单后处理来构建FCENet。

<div align=center>
<img src="https://user-images.githubusercontent.com/49955700/202216513-8a47a3d9-c23a-403a-84b9-e589bb519563.png"/>
</div>

[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Fourier_Contour_Embedding_for_Arbitrary-Shaped_Text_Detection_CVPR_2021_paper.pdf):  Yiqin Zhu, Jianyong Chen, Lingyu Liang, Zhanghui Kuang, Lianwen Jin, Wayne Zhang; Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.

## 例子

<div align=center>
<img src="https://user-images.githubusercontent.com/49955700/202217983-81eddaa6-a37f-479e-b52b-e9ef2fb42ee6.jpg"/>
</div>

# [效果](#目录)

## FCENet 验证性能(ICDAR2105)
|  | Recall | Precision  | Hmean-iou 
|:-|:-:|:-:|:-:|
| Paper  | 84.2% | 85.1% | 84.6% |
| Torch  | 81.2% | 88.7% | 84.7% |
| MindSpore  | 80.7% | 88.4% | 84.4% |

## FCENet 验证性能(CTW1500)
|  | Recall | Precision  | Hmean-iou 
|:-|:-:|:-:|:-:|
| Paper  | 80.7% | 85.7% | 83.1% |
| Torch  | 79.1% | 83.0% | 81.0% |
| MindSpore  | 82.3% | 83.5% | 82.8% |

# [数据集](#目录)

## 概述


|      数据集      |                    链接                     |                                         
| :---------------: | :-------------------------------------------: |         
|      CTW1500      | [homepage](https://github.com/Yuliang-Liu/Curve-Text-Detector) |                     
|     ICDAR2015     | [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads) | 




## CTW1500

- 步骤1：从 [github](https://github.com/Yuliang-Liu/Curve-Text-Detector)下载 `train_images.zip`, `test_images.zip`, `train_labels.zip`, `test_labels.zip` 

  ```bash
  mkdir CTW1500 && cd CTW1500
  mkdir imgs && mkdir annotations

  # For annotations
  cd annotations
  wget -O train_labels.zip https://universityofadelaide.box.com/shared/static/jikuazluzyj4lq6umzei7m2ppmt3afyw.zip
  wget -O test_labels.zip https://cloudstor.aarnet.edu.au/plus/s/uoeFl0pCN9BOCN5/download
  unzip train_labels.zip && mv ctw1500_train_labels training
  unzip test_labels.zip -d test
  cd ..
  # For images
  cd imgs
  wget -O train_images.zip https://universityofadelaide.box.com/shared/static/py5uwlfyyytbb2pxzq9czvu6fuqbjdh8.zip
  wget -O test_images.zip https://universityofadelaide.box.com/shared/static/t4w48ofnqkdw7jyc4t11nsukoeqk9c3d.zip
  unzip train_images.zip && mv train_images training
  unzip test_images.zip && mv test_images test
  ```

- 步骤2：通过以下命令生成 `instances_training.txt` 和 `instances_test.txt`：

  ```bash
  python tools/ctw1500_converter.py /path/to/ctw1500 -o /path/to/ctw1500 --split-list training test
  ```

- 由此产生的目录结构看起来像下面这样：

  ```text
  ├── CTW1500
  │   ├── imgs
  │   ├── annotations
  │   ├── instances_training.txt
  │   └── instances_val.txt
  ```


## ICDAR2015

- 步骤1：从[homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载 `ch4_training_images.zip`, `ch4_test_images.zip`, `ch4_training_localization_transcription_gt.zip`, `Challenge4_Test_Task1_GT.zip`

  ```bash
  mkdir ICDAR2015 && cd ICDAR2015
  mkdir imgs && mkdir annotations

  unzip ch4_training_images.zip -d ch4_training_images
  unzip ./ch4_test_images.zip -d ch4_test_images
  unzip ch4_training_localization_transcription_gt.zip -d ch4_training_localization_transcription_gt
  unzip Challenge4_Test_Task1_GT.zip -d Challenge4_Test_Task1_GT
  # For images,
  mv ch4_training_images imgs/training
  mv ch4_test_images imgs/test
  # For annotations,
  mv ch4_training_localization_transcription_gt annotations/training
  mv Challenge4_Test_Task1_GT annotations/test
  ```

- 步骤2：通过以下命令生成 `instances_training.txt` 和 `instances_test.txt`：

  ```bash
  python tools/icdar2015_converter.py ./ICDAR2015 -o ./ICDAR2015 -d icdar2015 --split-list training test
  ```

- 由此产生的目录结构看起来像下面这样：

  ```text
  ├── ICDAR2015
  │   ├── imgs
  │   ├── annotations
  │   ├── instances_training.txt
  │   └── instances_val.txt
  ```

# [预训练模型](#目录)


下载pytorch预训练模型：[resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)，
将pytorch模型转换为mindpore模型

```shell
python tools/resnet_model_torch2mindspore.py --torch_file=/path_to_model/resnet50-19c8e357.pth --output_path=../
```

# [配置环境](#目录)

- 硬件（Ascend或GPU）。
    - 准备好带有Ascend处理器或GPU的硬件环境。
- 框架
    - [MindSpore](http://www.mindspore.cn/install/en)
- 更多信息，请查看以下资源： 
    - [MindSpore教程](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- 安装 [Mindspore1.8.1](https://www.mindspore.cn/install)
- 安装 [Opencv4.6.0](https://docs.opencv.org/4.6.0/)


# [快速入门](#目录)

通过官方网站安装MindSpore后，你可以开始训练和验证，具体步骤如下：

- 在GPU上运行

```shell
# 运行训练
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config_path='./configs/CTW1500_config.yaml' > CTW1500_out.log &

nohup python train.py --config_path='./configs/ICDAR2015_config.yaml' > ICDAR2015_out.log &

# 运行分布式训练
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/run_distribute_train_gpu.sh

# 运行测试
python test.py

python test.py --config_path='./configs/CTW1500_config.yaml'

python test.py --config_path='./configs/ICDAR2015_config.yaml'

# 推理展示检测结果
python infer_det.py

python infer_det.py --config_path='./configs/ICDAR2015_config.yaml'
```

- 在Ascend上运行

```shell
# 运行训练
nohup python train.py --config_path='./configs/CTW1500_config.yaml' --device_target='Ascend' > CTW1500_out.log &

nohup python train.py --config_path='./configs/ICDAR2015_config.yaml' --device_target='Ascend' > ICDAR2015_out.log &

# 运行分布式训练
sh scripts/run_distribute_train_ascend.sh

# 运行测试
python test.py

python test.py --config_path='./configs/CTW1500_config.yaml' --device_target='Ascend'

python test.py --config_path='./configs/ICDAR2015_config.yaml' --device_target='Ascend'

# 推理展示检测结果
python infer_det.py

python infer_det.py --config_path='./configs/ICDAR2015_config.yaml'
```

- 在ModelArts上运行
- 如果你想在modelarts上训练模型，你可以参考modelarts的【官方指导文件】（https://support.huaweicloud.com/modelarts/）。

