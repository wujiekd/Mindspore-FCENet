# FCENet in MindSpore
<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

# Contents
- [FCENet Description](#fCENet-description)
- [Dataset](#dataset)
- [Pretrained Model](#pretrained-model)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)

# [FCENet Description](#contents)

Fourier Contour Embedding Network (FCENet) is a text detector which is able to well detect the arbitrary-shape text in natural scene. One of the main challenges for arbitrary-shaped text detection is to design a good text instance representation that allows networks to learn diverse text geometry variances. Most of existing methods model text instances in image spatial domain via masks or contour point sequences in the Cartesian or the polar coordinate system. However, the mask representation might lead to expensive post-processing, while the point sequence one may have limited capability to model texts with highly-curved shapes. To tackle these problems, we model text instances in the Fourier domain and propose one novel Fourier Contour Embedding (FCE) method to represent arbitrary shaped text contours as compact signatures. We further construct FCENet with a backbone, feature pyramid networks (FPN) and a simple post-processing with the Inverse Fourier Transformation (IFT) and Non-Maximum Suppression (NMS). 

<div align=center>
<img src="https://user-images.githubusercontent.com/49955700/202216513-8a47a3d9-c23a-403a-84b9-e589bb519563.png"/>
</div>

[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhu_Fourier_Contour_Embedding_for_Arbitrary-Shaped_Text_Detection_CVPR_2021_paper.pdf):  Yiqin Zhu, Jianyong Chen, Lingyu Liang, Zhanghui Kuang, Lianwen Jin, Wayne Zhang; Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.


## Example
<div align=center>
<img src="https://user-images.githubusercontent.com/49955700/202217983-81eddaa6-a37f-479e-b52b-e9ef2fb42ee6.jpg"/>
</div>



# [Dataset](#contents)

## Overview


|      Dataset      |                    Link                     |                                         
| :---------------: | :-------------------------------------------: |         
|      CTW1500      | [homepage](https://github.com/Yuliang-Liu/Curve-Text-Detector) |                     
|     ICDAR2015     | [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads) | 




## CTW1500

- Step1: Download `train_images.zip`, `test_images.zip`, `train_labels.zip`, `test_labels.zip` from [github](https://github.com/Yuliang-Liu/Curve-Text-Detector)

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

- Step2: Generate `instances_training.txt` and `instances_test.txt` with following command:

  ```bash
  python tools/ctw1500_converter.py /path/to/ctw1500 -o /path/to/ctw1500 --split-list training test
  ```

- The resulting directory structure looks like the following:

  ```text
  ├── CTW1500
  │   ├── imgs
  │   ├── annotations
  │   ├── instances_training.txt
  │   └── instances_val.txt
  ```


## ICDAR2015

- Step1: Download `ch4_training_images.zip`, `ch4_test_images.zip`, `ch4_training_localization_transcription_gt.zip`, `Challenge4_Test_Task1_GT.zip` from [homepage](https://rrc.cvc.uab.es/?ch=4&com=downloads)

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

- Step2: Generate `instances_training.txt` and `instances_test.txt` with following command:

  ```bash
  python tools/icdar2015_converter.py ./ICDAR2015 -o ./ICDAR2015 -d icdar2015 --split-list training test
  ```

- The resulting directory structure looks like the following:

  ```text
  ├── ICDAR2015
  │   ├── imgs
  │   ├── annotations
  │   ├── instances_training.txt
  │   └── instances_val.txt
  ```

# [Pretrained Model](#contents)

download pytorch pretrained model: [resnet50-19c8e357.pth](https://download.pytorch.org/models/resnet50-19c8e357.pth)
transform pytorch model to mindspore model

```shell
python tools/resnet_model_torch2mindspore.py --torch_file=/path_to_model/resnet50-19c8e357.pth --output_path=../
```

# [Environment Requirements](#contents)

- Hardware（Ascend or GPU）
    - Prepare hardware environment with Ascend processor or GPU.
- Framework
    - [MindSpore](http://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)
- install [Mindspore1.8.1](https://www.mindspore.cn/install)
- install [Opencv4.6.0](https://docs.opencv.org/4.6.0/)


# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

- running on GPU

```shell
# run training example
CUDA_VISIBLE_DEVICES=1 nohup python train.py --config_path='./configs/CTW1500_config.yaml' > CTW1500_out.log &

nohup python train.py --config_path='./configs/ICDAR2015_config.yaml' > ICDAR2015_out.log &

# run distributed training example
CUDA_VISIBLE_DEVICES=0,1,2,3 sh scripts/run_distribute_train_gpu.sh

# run test.py
python test.py

python test.py --config_path='./configs/CTW1500_config.yaml'

python test.py --config_path='./configs/ICDAR2015_config.yaml'

# inference display detection results
python infer_det.py

python infer_det.py --config_path='./configs/ICDAR2015_config.yaml'
```

- running on Ascend

```shell
# run training example
nohup python train.py --config_path='./configs/CTW1500_config.yaml' --device_target='Ascend' > CTW1500_out.log &

nohup python train.py --config_path='./configs/ICDAR2015_config.yaml' --device_target='Ascend' > ICDAR2015_out.log &

# run distributed training example
sh scripts/run_distribute_train_ascend.sh

# run test.py
python test.py

python test.py --config_path='./configs/CTW1500_config.yaml' --device_target='Ascend'

python test.py --config_path='./configs/ICDAR2015_config.yaml' --device_target='Ascend'

# inference display detection results
python infer_det.py

python infer_det.py --config_path='./configs/ICDAR2015_config.yaml'
```


- running on ModelArts
- If you want to train the model on modelarts, you can refer to the [official guidance document] of modelarts (https://support.huaweicloud.com/modelarts/)

