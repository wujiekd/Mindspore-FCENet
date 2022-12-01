# Copyright 2020 Huawei Technologies Co., Ltd
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
import math
import operator
from functools import reduce
import time
import numpy as np
import cv2
from tqdm import tqdm
from mindspore import Tensor, context
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.datasets.dataset import test_dataset_creator
from src.fcenet import FCENet
from src.postprocess.fce_process import FCEPostProcess
from src.metric.det_metric import DetFCEMetric
from src.config import config
import mindspore as ms

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                    save_graphs_path=".")

    
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test():
    
    ds = test_dataset_creator(config)
    config.INFERENCE = True
    net = FCENet(config)
    print(config.ckpt)
    param_dict = load_checkpoint(config.ckpt)
    load_param_into_net(net, param_dict)
    print('parameters loaded!')

    if config.Data_NAME ==  "CTW1500":
        PostProcess = FCEPostProcess(config.scales)
    elif config.Data_NAME ==  "ICDAR2015":
        PostProcess = FCEPostProcess(config.scales,text_repr_type='quad')
    eval_class = DetFCEMetric()
    get_data_time = AverageMeter()
    model_run_time = AverageMeter()
    post_process_time = AverageMeter()

    end_pts= start_pts = time.time()
    iters = ds.create_tuple_iterator(output_numpy=True)
    count = 0
    for batch in tqdm(iters):
        count += 1
        # get data
        image,img_shape, polys, ignore_tags,img_path = batch
        image = Tensor(image, ms.float32)
        
        get_data_pts = time.time()
        get_data_time.update(get_data_pts - end_pts)
        
        preds = net(image)
        model_run_pts = time.time()
        model_run_time.update(model_run_pts - get_data_pts)
        
        #batch = [item.numpy() for item in batch]
        
        
        post_result = PostProcess(preds, batch[1])
        eval_class(post_result, batch)
        
        post_process_pts = time.time()
        post_process_time.update(post_process_pts - model_run_pts)
        
        end_pts = time.time()
        
        
    metric = eval_class.get_metric()
    total_time = end_pts - start_pts
    metric['fps'] = 500 / (total_time)
    
    for k, v in metric.items():
        print('{}:{}'.format(k, v))
        


if __name__ == "__main__":
    test()
