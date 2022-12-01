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


import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.ops as ops
from ..model.base import _conv, _bn,Xavier_conv
from collections import OrderedDict

# class FCEFPN(nn.Cell):
#     """
#     Feature Pyramid Network, see https://arxiv.org/abs/1612.03144
#     Args:
#         in_channels (list[int]): input channels of each level which can be 
#             derived from the output shape of backbone by from_config
#         out_channel (list[int]): output channel of each level
#         spatial_scales (list[float]): the spatial scales between input feature
#             maps and original input image which can be derived from the output 
#             shape of backbone by from_config
#         has_extra_convs (bool): whether to add extra conv to the last level.
#             default False
#         extra_stage (int): the number of extra stages added to the last level.
#             default 1
#         use_c5 (bool): Whether to use c5 as the input of extra stage, 
#             otherwise p5 is used. default True
#         norm_type (string|None): The normalization type in FPN module. If 
#             norm_type is None, norm will not be used after conv and if 
#             norm_type is string, bn, gn, sync_bn are available. default None
#         norm_decay (float): weight decay for normalization layer weights.
#             default 0.
#         freeze_norm (bool): whether to freeze normalization layer.  
#             default False
#         relu_before_extra_convs (bool): whether to add relu before extra convs.
#             default False
        
#     """

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  has_extra_convs=False,
#                  extra_stage=1,
#                  use_c5=True,
#                  norm_type=None,
#                  norm_decay=0.,
#                  freeze_norm=False,
#                  relu_before_extra_convs=True):
#         super(FCEFPN, self).__init__()
#         self.out_channels = out_channels
#         self.has_extra_convs = has_extra_convs
#         self.extra_stage = extra_stage
#         self.use_c5 = use_c5
#         self.relu_before_extra_convs = relu_before_extra_convs
#         self.norm_type = norm_type
#         self.norm_decay = norm_decay
#         self.freeze_norm = freeze_norm

#         self.lateral_convs = []
#         self.fpn_convs = []
#         fan = out_channels * 3 * 3

#         # stage index 0,1,2,3 stands for res2,res3,res4,res5 on ResNet Backbone
#         # 0 <= st_stage < ed_stage <= 3
       
#         self.convs_b3= Xavier_conv(in_channels[0], out_channels, kernel_size=1, has_bias=True)
#         self.convs_b4= Xavier_conv(in_channels[1], out_channels, kernel_size=1, has_bias=True)
#         self.convs_b5= Xavier_conv(in_channels[2], out_channels, kernel_size=1, has_bias=True)
            
        
#         self.fpn_convs_b3 = Xavier_conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True)
#         self.fpn_convs_b4 = Xavier_conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True)
#         self.fpn_convs_b5 = Xavier_conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True)
        
    
#     def construct(self, body_feats):
#         laterals = []
#         num_levels = len(body_feats)

#         laterals.append(self.convs_b3(body_feats[0]))
#         laterals.append(self.convs_b4(body_feats[1]))
#         laterals.append(self.convs_b5(body_feats[2]))

#         for i in range(1, num_levels):
#             lvl = num_levels - i
#             upsample = ops.interpolate(laterals[lvl],scales=(1.0,1.0,2.0,2.0) , mode="bilinear")
#             laterals[lvl - 1] += upsample

#         fpn_output = []

#         fpn_output.append(self.fpn_convs_b3(laterals[0]))
#         fpn_output.append(self.fpn_convs_b4(laterals[1]))
#         fpn_output.append(self.fpn_convs_b5(laterals[2]))
        
#         return fpn_output



class FCEFPN(nn.Cell):
    def __init__(self, in_channels, out_channel):
        super(FCEFPN, self).__init__()

        # reduce layers
        # self.reduce_conv_c2 = Xavier_conv(in_channels[0], out_channel, kernel_size=1, has_bias=True)
        # self.reduce_bn_c2 = _bn(out_channel)
        # self.reduce_relu_c2 = nn.ReLU()

        self.reduce_conv_c3 = Xavier_conv(in_channels[0], out_channel, kernel_size=1, has_bias=True)
        # self.reduce_bn_c3 = _bn(out_channel)
        # self.reduce_relu_c3 = nn.ReLU()

        self.reduce_conv_c4 = Xavier_conv(in_channels[1], out_channel, kernel_size=1, has_bias=True)
        # self.reduce_bn_c4 = _bn(out_channel)
        # self.reduce_relu_c4 = nn.ReLU()

        self.reduce_conv_c5 = Xavier_conv(in_channels[2], out_channel, kernel_size=1, has_bias=True)
        # self.reduce_bn_c5 = _bn(out_channel)
        # self.reduce_relu_c5 = nn.ReLU()

        # smooth layers
        self.smooth_conv_p5 = Xavier_conv(out_channel, out_channel, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        # self.smooth_bn_p5 = _bn(out_channel)
        # self.smooth_relu_p5 = nn.ReLU()
        
        self.smooth_conv_p4 = Xavier_conv(out_channel, out_channel, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        # self.smooth_bn_p4 = _bn(out_channel)
        # self.smooth_relu_p4 = nn.ReLU()

        self.smooth_conv_p3 = Xavier_conv(out_channel, out_channel, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
        # self.smooth_bn_p3 = _bn(out_channel)
        # self.smooth_relu_p3 = nn.ReLU()

        # self.smooth_conv_p2 = Xavier_conv(out_channel, out_channel, kernel_size=3, padding=1, has_bias=True)
        # self.smooth_bn_p2 = _bn(out_channel)
        # self.smooth_relu_p2 = nn.ReLU()

        # self._upsample_p2 = P.ResizeBilinear((800 // 4, 800 // 4), align_corners=True)

    def construct(self, body_feats):
          
        c3 = body_feats[0]
        c4 = body_feats[1]
        c5 = body_feats[2] 
        
        # _upsample_p4 = P.ResizeBilinear((c4.shape[-1], c4.shape[-1]), align_corners=True)
        # _upsample_p3 = P.ResizeBilinear((c3.shape[-1], c3.shape[-1]), align_corners=True)
        
        p5 = self.reduce_conv_c5(c5)
        # p5 = self.reduce_relu_c5(self.reduce_bn_c5(p5))

        c4 = self.reduce_conv_c4(c4)
        # c4 = self.reduce_relu_c4(self.reduce_bn_c4(c4))
        p4 = ops.interpolate(p5,scales=(1.0,1.0,2.0,2.0) , mode="bilinear")+ c4
        #p4 = _upsample_p4(p5) + c4
        # p4 = self.smooth_conv_p4(p4)
        # p4 = self.smooth_relu_p4(self.smooth_bn_p4(p4))

        c3 = self.reduce_conv_c3(c3)
        # c3 = self.reduce_relu_c3(self.reduce_bn_c3(c3))
        p3 = ops.interpolate(p4,scales=(1.0,1.0,2.0,2.0) , mode="bilinear")+ c3
        #p3 = _upsample_p3(p4) + c3
        
        p5 = self.smooth_conv_p5(p5)
        p4 = self.smooth_conv_p4(p4)
        p3 = self.smooth_conv_p3(p3)
        

        # c2 = self.reduce_conv_c2(c2)
        # c2 = self.reduce_relu_c2(self.reduce_bn_c2(c2))
        # p2 = self._upsample_p2(p3) + c2
        # p2 = self.smooth_conv_p2(p2)
        # p2 = self.smooth_relu_p2(self.smooth_bn_p2(p2))

        # p3 = self._upsample_p2(p3)
        # p4 = self._upsample_p3(p4)
        # p5 = self._upsample_p3(p5)

        # print(p3.shape)
        # print(p4.shape)
        # print(p5.shape)
        out = [p3, p4, p5] #self.concat((p3, p4, p5))

        return out
