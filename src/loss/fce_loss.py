import mindspore.numpy as np
from mindspore.common.tensor import Tensor
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer, Constant
import mindspore.numpy as mnp

class FCELoss(nn.Cell):
    """The class for implementing FCENet loss
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped
        Text Detection
    [https://arxiv.org/abs/2104.10442]
    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        ohem_ratio (float): the negative/positive ratio in OHEM.
    """

    def __init__(self, fourier_degree, num_sample, ohem_ratio=3.):
        super().__init__()
        self.fourier_degree = fourier_degree
        self.num_sample = num_sample
        self.ohem_ratio = ohem_ratio
        self.sort_descending = ms.ops.Sort(descending=True)
        equation = "ak, kn-> an"
        self.einsum = ops.Einsum(equation)
        self.threshold0 = Tensor(0.5, ms.float32)
        self.greater = ops.GreaterEqual()
        self.eps = ms.Tensor(1e-8, ms.float32)
        
    def construct(self, preds_p3,preds_p4,preds_p5, p3_maps, p4_maps, p5_maps):
        # assert isinstance(preds, list)
        # assert p3_maps[0].shape[0] == 4 * self.fourier_degree + 5,\
        #     'fourier degree not equal in FCEhead and FCEtarget'

        # device = preds[0][0].device
        # # to tensor
        # gts = [p3_maps, p4_maps, p5_maps]
        # for idx, maps in enumerate(gts):
        #     gts[idx] = torch.from_numpy(np.stack(maps)).float().to(device)

        # losses = multi_apply(self.forward_single, preds, gts)
        if p5_maps is None:
            losses = [self.forward_single(preds_p3, p3_maps),self.forward_single(preds_p4, p4_maps)]
        else:
            losses = [self.forward_single(preds_p3, p3_maps),self.forward_single(preds_p4, p4_maps),self.forward_single(preds_p5, p5_maps)]
        
        
        loss_tr = ms.Tensor(0., ms.float32) #torch.tensor(0., device=device).float()
        loss_tcl = ms.Tensor(0., ms.float32) #torch.tensor(0., device=device).float()
        loss_reg_x = ms.Tensor(0., ms.float32)#torch.tensor(0., device=device).float()
        loss_reg_y = ms.Tensor(0., ms.float32) #torch.tensor(0., device=device).float()
        all_loss = ms.Tensor(0., ms.float32)
        
           
        for i in range(len(losses)):
            loss_tr += losses[i][0]
            loss_tcl += losses[i][1]
            loss_reg_x += losses[i][2]
            loss_reg_y += losses[i][3]

                
            
        all_loss = loss_tr + loss_tcl + loss_reg_x + loss_reg_y

        # results = dict(
        #     all_loss = all_loss,
        #     loss_text=loss_tr,
        #     loss_center=loss_tcl,
        #     loss_reg_x=loss_reg_x,
        #     loss_reg_y=loss_reg_y,
        # )
        print(f"all_loss = {all_loss},loss_text={loss_tr},loss_center={loss_tcl},loss_reg_x={loss_reg_x},loss_reg_y={loss_reg_y}")
        return all_loss

    def mask_fun(self,pred, mask):
        mask = mask.astype('float32')>0.5
        return pred * mask
    
    def forward_single(self, pred, gt):
        cls_pred = ops.Transpose()(pred[0], (0, 2, 3, 1))
        reg_pred = ops.Transpose()(pred[1], (0, 2, 3, 1))
        gt = ops.Transpose()(gt, (0, 2, 3, 1))
        
        
        k = 2 * self.fourier_degree + 1
        tr_pred = cls_pred[:, :, :, :2].view((-1, 2))
        tcl_pred = cls_pred[:, :, :, 2:].view((-1, 2))
        x_pred = reg_pred[:, :, :, 0:k].view((-1, k))
        y_pred = reg_pred[:, :, :, k:2 * k].view((-1, k))

        tr_mask = gt[:, :, :, :1].view((-1))
        tcl_mask = gt[:, :, :, 1:2].view((-1))
        
        train_mask = gt[:, :, :, 2:3].view((-1))
        x_map = gt[:, :, :, 3:3 + k].view((-1, k))
        y_map = gt[:, :, :, 3 + k:].view((-1, k))

        tr_train_mask = train_mask * tr_mask
        # tr loss
        loss_tr = self.ohem(tr_pred, tr_mask.astype(ms.int32), train_mask.astype(ms.int32))

        # tcl loss
        loss_tcl = ms.Tensor(0., ms.float32)#torch.tensor(0.).float().to(device)
        tr_neg_mask = 1 - tr_train_mask

        #print(int(tr_train_mask.sum().item(0)))
        if int(tr_train_mask.sum().item(0)) > 0:
            loss_tcl_none = ops.cross_entropy(tcl_pred,tcl_mask.astype(ms.int32), reduction='none')
            loss_tcl_pos = (loss_tcl_none * self.greater(tr_train_mask, self.threshold0)).sum() /self.greater(tr_train_mask, self.threshold0).sum()
            
            loss_tcl_neg = (loss_tcl_none * self.greater(tr_neg_mask, self.threshold0)).sum() /self.greater(tr_neg_mask, self.threshold0).sum()
            
            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg

        
        # regression loss
        loss_reg_x = ms.Tensor(0., ms.float32)#torch.tensor(0.).float().to(device)
        loss_reg_y = ms.Tensor(0., ms.float32)#torch.tensor(0.).float().to(device)

        if int(tr_train_mask.sum().item(0)) > 0:
            weight = (tr_mask.astype('float32') + tcl_mask.astype('float32')) / 2
            weight = weight.view((-1, 1)) 
            ft_x, ft_y = self.fourier2poly(x_map, y_map)
            ft_x_pre, ft_y_pre = self.fourier2poly(x_pred, y_pred)
            dim = ft_x.shape[1]
            
            loss_x = ops.smooth_l1_loss(ft_x_pre,ft_x,reduction='none')
            loss_reg_x = weight * loss_x
            loss_reg_x = (loss_reg_x * self.greater(tr_train_mask.view((-1, 1)) , self.threshold0)).sum() /self.greater(tr_train_mask.view((-1, 1)), self.threshold0).sum()/dim
            
            loss_y = ops.smooth_l1_loss(ft_y_pre,ft_y,reduction='none')
            loss_reg_y = weight * loss_y
            loss_reg_y = (loss_reg_y * self.greater(tr_train_mask.view((-1, 1)) , self.threshold0)).sum() /self.greater(tr_train_mask.view((-1, 1)), self.threshold0).sum()/dim
        
        
        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y

    def ohem(self, predict, target, train_mask):
        # pos = ops.stop_gradient(((target * train_mask) > self.threshold0).astype(ms.float32))
        # neg = ops.stop_gradient((((1 - target) * train_mask) > self.threshold0).astype(ms.float32))

        # n_pos = ops.stop_gradient(pos.sum())
        # n_neg = ops.stop_gradient(min(neg.sum(), self.ohem_ratio * n_pos).astype(ms.int32))
        # if n_pos < 1:
        #     n_neg = ms.Tensor(100, ms.int32)
        # loss = ops.cross_entropy(predict, target, reduction='none')
        # loss_pos = (loss * pos).sum()
        # loss_neg = loss * neg
        # if neg.sum() > n_neg:
        #     negative_value, _ = self.sort_descending(loss_neg)  # Top K
        #     con_k = negative_value[n_neg - 1]
        #     con_mask = ops.stop_gradient((negative_value >= con_k).astype(negative_value.dtype))
        #     loss_neg = negative_value * con_mask
        # return (loss_pos + loss_neg.sum()) / (n_pos + n_neg.astype(ms.float32) + 1e-7)
        
        pos = self.greater(target * train_mask, self.threshold0)
        neg = self.greater((1 - target) * train_mask, self.threshold0) 

        n_pos = int(pos.sum().astype('float32').item(0))

        if n_pos > 0:
            loss_pos = ops.cross_entropy(predict, target, reduction='none')
            loss_pos =(loss_pos * pos).sum()
            
            loss_neg = ops.cross_entropy(predict, target, reduction='none')
            
            n_neg = min(
                int(neg.sum().astype('float32').item(0)),
                int(self.ohem_ratio * n_pos))
        else:
            loss_pos = ms.Tensor(0., ms.float32) 
            loss_neg = ops.cross_entropy(predict, target, reduction='none')
            n_neg = 100
        if len(loss_neg) > n_neg:
            negative_value, _ = self.sort_descending(loss_neg)  # Top K
            con_k = negative_value[n_neg - 1]
            con_mask = (negative_value >= con_k).astype(negative_value.dtype)
            loss_neg = negative_value * con_mask
            

        return (loss_pos + loss_neg.sum()) / (ms.Tensor(n_pos, ms.float32) + ms.Tensor(n_neg, ms.float32))

            

    def fourier2poly(self, real_maps, imag_maps):
        """Transform Fourier coefficient maps to polygon maps.
        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
        Returns
            x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """

        #device = real_maps.device

        # k_vect = torch.arange(
        #     -self.fourier_degree,
        #     self.fourier_degree + 1,
        #     dtype=torch.float,
        #     device=device).view(-1, 1)
        # i_vect = torch.arange(
        #     0, self.num_sample, dtype=torch.float, device=device).view(1, -1)
        k_vect = ms.Tensor(np.arange(
            -self.fourier_degree,
            self.fourier_degree + 1,
            dtype='float32')).view((-1, 1))
        i_vect = ms.Tensor(np.arange(
            0, self.num_sample, dtype='float32')).view((1, -1))
        

        transform_matrix = 2 * np.pi / self.num_sample * ops.matmul(
            k_vect, i_vect)

        x1 = self.einsum((real_maps,ops.cos(transform_matrix)))
        x2 = self.einsum((imag_maps,ops.sin(transform_matrix)))
        y1 = self.einsum((real_maps,ops.sin(transform_matrix)))
        y2 = self.einsum((imag_maps,ops.cos(transform_matrix)))

        
        x_maps = x1 - x2
        y_maps = y1 + y2

        return x_maps, y_maps