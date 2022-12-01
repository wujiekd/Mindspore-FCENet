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
"""lr generator for psenet"""
import math
from mindspore import nn,Tensor


def dynamic_lr(base_lr, end_lr , epochs, step_per_epoch, by_epoch=True):
    if by_epoch:
        
        lrs =  nn.polynomial_decay_lr(base_lr, end_lr, total_step = step_per_epoch*epochs, step_per_epoch = step_per_epoch, decay_epoch=epochs, power=0.9)
        return lrs



