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


from distutils.command.config import config
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor
import mindspore.common.dtype as mstype

from .model.base import _conv, _bn
from .model.resnet50 import ResNet, ResidualBlock
from .model.fpn import FCEFPN
from .model.fce_head import FCEHead

class FCENet(nn.Cell):
    def __init__(self, config):
        super(FCENet, self).__init__()
        self.inference = config.INFERENCE

        # backbone
        self.feature_extractor = ResNet(ResidualBlock,
                                        config.BACKBONE_LAYER_NUMS,
                                        config.BACKBONE_IN_CHANNELS,
                                        config.BACKBONE_OUT_CHANNELS)

        # neck
        self.feature_fusion = FCEFPN(config.BACKBONE_OUT_CHANNELS[1:],
                                  config.NECK_OUT_CHANNEL)


        # head
        self.head = FCEHead(in_channels = config.in_channels,
            scales = config.scales,
            fourier_degree= config.fourier_degree,
            num_sample=config.num_sample,
            alpha=config.alpha,
            beta=config.beta,
            mode = config.mode)
        

        print('FCENet initialized!')

    def construct(self, x):
        # backbone
        body_feats = self.feature_extractor(x)

        feature = self.feature_fusion(body_feats) 

        preds_p3,preds_p4,preds_p5 = self.head(feature)
        
        return preds_p3,preds_p4,preds_p5
