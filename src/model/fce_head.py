
import Polygon as plg
from functools import partial
from abc import ABCMeta
import copy
from collections import defaultdict
from ..postprocess.fce_process import *
from .base import _conv, _bn


from mindspore import nn
from mindspore.common.initializer import Normal
import mindspore as ms
import mindspore.ops as ops
from mindspore.nn import Cell

class FCEHead(Cell):
    """The class for implementing FCENet head.
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped Text
    Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        in_channels (int): The number of input channels.
        scales (list[int]) : The scale of each layer.
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, FCEnet tends to be overfitting.
        score_thr (float) : The threshold to filter out the final
            candidates.
        nms_thr (float) : The threshold of nms.
        alpha (float) : The parameter to calculate final scores. Score_{final}
            = (Score_{text region} ^ alpha)
            * (Score{text center region} ^ beta)
        beta (float) :The parameter to calculate final scores.
    """

    def __init__(
        self,
        in_channels,
        scales,
        fourier_degree=5,
        num_sample=50,
        num_reconstr_points=50,
        decoding_type='fcenet',
        score_thr=0.3,
        nms_thr=0.1,
        alpha=1.0,
        beta=1.0,
        text_repr_type='poly',
        train_cfg=None,
        test_cfg=None,
        mode = False):
        super(FCEHead, self).__init__()
        assert isinstance(in_channels, int)

        self.downsample_ratio = 1.0
        self.in_channels = in_channels
        self.scales = scales
        self.fourier_degree = fourier_degree
        self.sample_num = num_sample
        self.num_reconstr_points = num_reconstr_points
        self.decoding_type = decoding_type
        self.score_thr = score_thr
        self.nms_thr = nms_thr
        self.alpha = alpha
        self.beta = beta
        self.text_repr_type = text_repr_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.out_channels_cls = 4
        self.out_channels_reg = (2 * self.fourier_degree + 1) * 2
        self.softmax = nn.Softmax(axis=1)
        self.training = mode
    
        self.out_conv_cls = _conv(self.in_channels, self.out_channels_cls, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)
   
        self.out_conv_reg = _conv(self.in_channels, self.out_channels_reg, kernel_size=3, padding=1, pad_mode='pad', has_bias=True)

    def construct(self, feats):

        cls_res = [self.out_conv_cls(feats[0]),self.out_conv_cls(feats[1]),self.out_conv_cls(feats[2])]
        reg_res = [self.out_conv_reg(feats[0]),self.out_conv_reg(feats[1]),self.out_conv_reg(feats[2])]
        level_num = len(cls_res)
        
        if not self.training:
            preds = []
            for i in range(level_num):
                tr_pred = self.softmax(cls_res[i][:, 0:2, :, :])
                tcl_pred = self.softmax(cls_res[i][:, 2:, :, :])
                preds.append(ops.concat([tr_pred, tcl_pred, reg_res[i]], axis=1))
            return preds[0],preds[1],preds[2]
        else:
            preds = [[cls_res[0], reg_res[0]],[cls_res[1], reg_res[1]],[cls_res[2], reg_res[2]]]
            return preds[0],preds[1],preds[2]

