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
from mindspore.communication.management import init, get_rank, get_group_size


def init_env(cfg):
    """初始化运行时环境."""
    ms.set_seed(cfg.seed)
    if cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg.device_target)


    if cfg.context_mode not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg.context_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg.device_target = ms.get_context("device_target")

    if cfg.device_target == "CPU":
        cfg.device_id = 0
        cfg.device_num = 1
        cfg.rank_id = 0

    if hasattr(cfg, "device_id") and isinstance(cfg.device_id, int):
        ms.set_context(device_id=cfg.device_id)
    if cfg.device_num > 1:
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        if cfg.device_num != group_size:
            raise ValueError(f"the setting device_num: {cfg.device_num} not equal to the real group_size: {group_size}")
        cfg.rank_id = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if hasattr(cfg, "all_reduce_fusion_config"):
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg.all_reduce_fusion_config)
    else:
        cfg.device_num = 1
        cfg.rank_id = 0
        print("run standalone!", flush=True)
        
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
    init_env(config)
    config.mode = False
    ds = test_dataset_creator(config)
    config.INFERENCE = True
    net = FCENet(config)
    print(config.ckpt)
    param_dict = load_checkpoint(config.ckpt)
    load_param_into_net(net, param_dict)
    print('parameters loaded!')
    
    
    if config.Data_NAME ==  "CTW1500":
        PostProcess = FCEPostProcess(config.scales,alpha=config.alpha,beta=config.beta,)
    elif config.Data_NAME ==  "ICDAR2015":
        PostProcess = FCEPostProcess(config.scales,text_repr_type='quad',alpha=config.alpha,beta=config.beta,)
    eval_class = DetFCEMetric()
    get_data_time = AverageMeter()
    model_run_time = AverageMeter()
    post_process_time = AverageMeter()

    end_pts= start_pts = time.time()
    iters = ds.create_tuple_iterator(output_numpy=True)
    for batch in tqdm(iters): 
        # get data
        image,img_shape, polys, ignore_tags,img_path,texts = batch
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
