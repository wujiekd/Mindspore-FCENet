# Copyright 2020-2021 Huawei Technologies Co., Ltd
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


import ast
import operator
import mindspore.nn as nn
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from src.datasets.dataset import train_dataset_creator
from src.fcenet import FCENet
from src.loss.fce_loss import FCELoss 
from src.network_define import WithLossCell, LossCallBack
from src.schedule.lr_schedule import dynamic_lr
from src.config import config
import mindspore as ms
from mindspore.communication.management import init, get_rank, get_group_size
import os

binOps = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod
}

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def arithmeticeval(s):
    node = ast.parse(s, mode='eval')

    def _eval(node):
        if isinstance(node, ast.BinOp):
            return binOps[type(node.op)](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.Num):
            return node.n

        if isinstance(node, ast.Expression):
            return _eval(node.body)

        raise Exception('unsupported type{}'.format(node))
    return _eval(node.body)


def modelarts_pre_process():
    pass


def init_env(cfg):
    """初始化运行时环境."""
    ms.set_seed(cfg.seed)
    # 如果device_target设置是None，利用框架自动获取device_target，否则使用设置的。
    if cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg.device_target)

    # 配置运行模式，支持图模式和PYNATIVE模式
    if cfg.context_mode not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg.context_mode == "graph" else ms.PYNATIVE_MODE
    ms.set_context(mode=context_mode)

    cfg.device_target = ms.get_context("device_target")
    # 如果是CPU上运行的话，不配置多卡环境
    if cfg.device_target == "CPU":
        cfg.device_id = 0
        cfg.device_num = 1
        cfg.rank_id = 0

    # 设置运行时使用的卡
    if hasattr(cfg, "device_id") and isinstance(cfg.device_id, int):
        ms.set_context(device_id=cfg.device_id)
    if cfg.device_num > 1:
        # init方法用于多卡的初始化，不区分Ascend和GPU，get_group_size和get_rank方法只能在init后使用
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
        
def train():
    init_env(config)
    
    config.BASE_LR = arithmeticeval(config.BASE_LR)
    config.END_LR = arithmeticeval(config.END_LR)
    config.mode = True
    
    dataset = train_dataset_creator(config)
    step_size = dataset.get_dataset_size()
    print('Create dataset done!')

    config.INFERENCE = False
    net = FCENet(config)
    net = net.set_train()
    #print(net)
    if config.pre_trained:
        param_dict = load_checkpoint(config.pre_trained)
        load_param_into_net(net, param_dict, strict_load=True)
        print('Load Pretrained parameters done!')

    criterion = FCELoss(fourier_degree=config.fourier_degree,num_sample=config.num_sample)

    lrs = dynamic_lr(config.BASE_LR, config.END_LR, config.TRAIN_EPOCH , step_size)
    print('Load learning rate schedule done!')
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=lrs,
                 momentum=0.90, weight_decay=5e-4)
    net = WithLossCell(net, criterion)
    
    
    scale_sense = nn.FixedLossScaleUpdateCell(1)#(config.loss_scale) # 静态loss scale
    net = nn.TrainOneStepWithLossScaleCell(net, optimizer=opt, scale_sense=scale_sense)
    print('Load Network done!')
    
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossCallBack(per_print_times=step_size)
    ckpoint_cf = CheckpointConfig(save_checkpoint_steps=10*step_size, keep_checkpoint_max=50)
    ckpoint_cb = ModelCheckpoint(prefix="FCENET",
                                 config=ckpoint_cf,
                                 directory="{}/ckpt_{}".format(config.TRAIN_MODEL_SAVE_PATH,
                                                               config.rank_id))
    model = Model(net)
    print('Start training!')
    model.train(config.TRAIN_EPOCH,
                dataset,
                dataset_sink_mode=False,
                callbacks=[time_cb, loss_cb, ckpoint_cb])


if __name__ == '__main__':
    train()
