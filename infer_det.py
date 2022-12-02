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
import json
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
        
def draw_det_res(dt_boxes, config, img, img_name, save_path):
    if len(dt_boxes) > 0:
        import cv2
        src_im = img
        for box in dt_boxes:
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(img_name))
        cv2.imwrite(save_path, src_im)
        print("The detected Image saved in {}".format(save_path))


def infer_det():
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

    iters = ds.create_tuple_iterator(output_numpy=True)
    
    save_res_path = config.save_res_path
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))
        
    with open(save_res_path, "wb") as fout:
        for batch in tqdm(iters):
            # get data
            image,img_shape, polys, ignore_tags,img_path = batch
            image = Tensor(image, ms.float32)
            preds = net(image)


            post_result = PostProcess(preds, batch[1])
            
            img_path = str(img_path[0]).split("'")[1]
            src_img = cv2.imread(img_path)

            dt_boxes_json = []
            # parser boxes if post_result is dict
            if isinstance(post_result, dict):
                det_box_json = {}
                for k in post_result.keys():
                    boxes = post_result[k][0]['points']
                    dt_boxes_list = []
                    for box in boxes:
                        tmp_json = {"transcription": ""}
                        tmp_json['points'] = box.tolist()
                        dt_boxes_list.append(tmp_json)
                    det_box_json[k] = dt_boxes_list
                    save_det_path = os.path.dirname(config.save_res_path) + "/det_results_{}/".format(k)
                    draw_det_res(boxes, config, src_img, img_path, save_det_path)
            else:
                boxes = post_result[0]['points']
                dt_boxes_json = []
                # write result
                for box in boxes:
                    tmp_json = {"transcription": ""}
                    tmp_json['points'] = box.tolist()
                    dt_boxes_json.append(tmp_json)
                save_det_path = os.path.dirname(config.save_res_path) + "/det_results/"
                draw_det_res(boxes, config, src_img, img_path, save_det_path)
            otstr = img_path + "\t" + json.dumps(dt_boxes_json) + "\n"
            fout.write(otstr.encode())
            
    fout.close()  

if __name__ == "__main__":
    infer_det()
