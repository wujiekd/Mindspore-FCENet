# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import os
import cv2
from .transforms import *
from .fcenet_gentarget import *
from .loading import * 
import mindspore as ms
from mindspore import dataset as ds

__all__ = ['train_dataset_creator', 'test_dataset_creator']



class CTW_TrainDataset():
    def __init__(self,config):
        
        self.config = config
        self.root_dir = self.config.TRAIN_ROOT_DIR
        label_file_list = os.path.join(self.root_dir, "instances_training.txt")
        self.data_lines = self.get_image_info_list(label_file_list)
        
        self.opss = [
            #DecodeImage(img_mode='RGB',channel_first=False,ignore_orientation=True),
            DetLabelEncode(),
            ColorJitter(brightness=32.0 / 255, saturation=0.5, contrast=0.5),
            NormalizeImage(scale=1./255,mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],order='hwc'), 
            RandomScaling(size=800, scale=(3. / 4, 5. / 2)),
            RandomCropFlip(crop_ratio=0.5),
            RandomCropPolyInstances(crop_ratio=0.8,min_side_ratio=0.3),
            RandomRotatePolyInstances(rotate_ratio=0.5,max_angle=30,pad_with_fixed_color=False),
            SquareResizePad(target_size=800,pad_ratio=0.6),
            IaaAugment(augmenter_args=[{
                'type': 'Fliplr',
                'args': {
                    'p': 0.5
                }
            }]),
            Pad(),
            FCENetTargets(fourier_degree=5,level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0))),
            ToCHWImage(),
        ]
  
            
    def get_image_info_list(self, file_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                data_lines.extend(lines)
            
        data_lines = [line.decode('utf-8').strip("\n").split("\t") for line in data_lines]
        return data_lines

    def __getitem__(self, index):
        img_path = self.data_lines[index][0]
        label = self.data_lines[index][1]

        data = {'img_path': os.path.join(self.root_dir+'/imgs', img_path), 'label': label}
        data['image'] = cv2.imread(data['img_path'], cv2.IMREAD_COLOR)[:, :, ::-1]

        for ops in self.opss:  # augmentation and get label
            data = ops(data)

        p3_maps = data['p3_maps']
        p4_maps = data['p4_maps']
        p5_maps = data['p5_maps']

        return data['image'],p3_maps,p4_maps,p5_maps

    def __len__(self):
        return len(self.data_lines)
        

class ICDAR_TrainDataset():
    def __init__(self,config):
        self.config = config
        self.root_dir = self.config.TRAIN_ROOT_DIR
    
        label_file_list = os.path.join(self.root_dir, "instances_training.txt")
        self.data_lines = self.get_image_info_list(label_file_list)

        
        self.opss = [
            #DecodeImage(img_mode='RGB',channel_first=False,ignore_orientation=True),
            DetLabelEncode(),
            ColorJitter(brightness=32.0 / 255, saturation=0.5, contrast=0.5),
            NormalizeImage(scale=1./255,mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],order='hwc'), 
            RandomScaling(size=800, scale=(3. / 4, 5. / 2)),
            RandomCropFlip(crop_ratio=0.5),
            RandomCropPolyInstances(crop_ratio=0.8,min_side_ratio=0.3),
            RandomRotatePolyInstances(rotate_ratio=0.5,max_angle=30,pad_with_fixed_color=False),
            SquareResizePad(target_size=800,pad_ratio=0.6),
            IaaAugment(augmenter_args=[{
                'type': 'Fliplr',
                'args': {
                    'p': 0.5
                }
            }]),
            Pad(),
            FCENetTargets(fourier_degree=5,level_proportion_range=((0, 0.4), (0.3, 0.7), (0.6, 1.0))),
            ToCHWImage(),
        ]
            
    def get_image_info_list(self, file_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                data_lines.extend(lines)
            
        data_lines = [line.decode('utf-8').strip("\n").split("\t") for line in data_lines]
        return data_lines

    def __getitem__(self, index):
        img_path = self.data_lines[index][0]
        label = self.data_lines[index][1]

        data = {'img_path': os.path.join(self.root_dir+'/imgs', img_path), 'label': label}
        data['image'] = cv2.imread(data['img_path'], cv2.IMREAD_COLOR)[:, :, ::-1]

        for ops in self.opss:  # augmentation and get label
            data = ops(data)

        p3_maps = data['p3_maps']
        p4_maps = data['p4_maps']
        p5_maps = data['p5_maps']

        return data['image'],p3_maps,p4_maps,p5_maps

    def __len__(self):
        return len(self.data_lines)
        

def CTW_Test_Generator(root_dir):
    label_file_list = os.path.join(root_dir, "instances_training.txt")
    if isinstance(label_file_list, str):
            label_file_list = [label_file_list]
    data_lines = []
    for idx, file in enumerate(label_file_list):
        with open(file, "rb") as f:
            lines = f.readlines()
            data_lines.extend(lines)
        
    data_lines = [line.decode('utf-8').strip("\n").split("\t") for line in data_lines]

    opss = [
            #DecodeImage(img_mode='RGB',channel_first=False,ignore_orientation=True),
            DetLabelEncode(),
            DetResizeForTest(rescale_img=[1080, 736]),
            NormalizeImage(scale=1./255,mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],order='hwc'), 
            Pad(),
            ToCHWImage(),
        ]

        
    for index in range(len(data_lines)):
        img_path = data_lines[index][0]
        label = data_lines[index][1]

        data = {'img_path': os.path.join(root_dir+'/imgs', img_path), 'label': label}
        data['image'] = cv2.imread(data['img_path'], cv2.IMREAD_COLOR)[:, :, ::-1]
        for ops in opss:
            data = ops(data)

        yield data['image'],data['shape'], data['polys'], data['ignore_tags'],data['img_path']
        
        
def ICDAR_Test_Generator(root_dir):
    label_file_list = os.path.join(root_dir, "instances_test.txt")
    if isinstance(label_file_list, str):
            label_file_list = [label_file_list]
    data_lines = []
    for idx, file in enumerate(label_file_list):
        with open(file, "rb") as f:
            lines = f.readlines()
            data_lines.extend(lines)
        
    data_lines = [line.decode('utf-8').strip("\n").split("\t") for line in data_lines]

    opss = [
            #DecodeImage(img_mode='RGB',channel_first=False,ignore_orientation=True),
            DetLabelEncode(),
            DetResizeForTest(rescale_img=[2260, 2260]),
            NormalizeImage(scale=1./255,mean = [0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],order='hwc'), 
            Pad(),
            ToCHWImage(),
        ]
        
    for index in range(len(data_lines)):
        img_path = data_lines[index][0]
        label = data_lines[index][1]

        data = {'img_path': os.path.join(root_dir+'/imgs', img_path), 'label': label}
        data['image'] = cv2.imread(data['img_path'], cv2.IMREAD_COLOR)[:, :, ::-1]
        for ops in opss:
            data = ops(data)

        yield data['image'],data['shape'], data['polys'], data['ignore_tags'],data['img_path']




def train_dataset_creator(config,shuffle=True):
    ds.config.set_prefetch_size(42)
    if config.Data_NAME == "CTW1500":
        dataset = CTW_TrainDataset(config)
    else:
        dataset = ICDAR_TrainDataset(config)
    data_set = ds.GeneratorDataset(dataset, ['img', 'p3_maps', 'p4_maps', 'p5_maps'], num_parallel_workers=1,
                                   num_shards=config.device_num, shard_id=config.rank_id,
                                      shuffle=shuffle, max_rowsize=16)
    data_set = data_set.batch(config.TRAIN_BATCH_SIZE, drop_remainder=config.TRAIN_DROP_REMAINDER)
    return data_set


def test_dataset_creator(config):
    if config.Data_NAME == "CTW1500":
        data_set = ds.GeneratorDataset(CTW_Test_Generator(config.TEST_ROOT_DIR), ['image','shape', 'polys', 'ignore_tags','img_path'], num_parallel_workers=4)
    else:
        data_set = ds.GeneratorDataset(ICDAR_Test_Generator(config.TEST_ROOT_DIR), ['image','shape', 'polys', 'ignore_tags','img_path'], num_parallel_workers=4)
        
    data_set = data_set.batch(1, drop_remainder=config.TEST_DROP_REMAINDER)
    return data_set
