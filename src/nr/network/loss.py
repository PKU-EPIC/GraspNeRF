import torch
import torch.nn as nn
import numpy as np
import pyquaternion as pyq
import math
from network.ops import interpolate_feats
import torch.nn.functional as F
import torchmetrics
from utils.base_utils import calc_rot_error_from_qxyzw

class Loss:
    def __init__(self, keys):
        """
        keys are used in multi-gpu model, DummyLoss in train_tools.py
        :param keys: the output keys of the dict
        """
        self.keys=keys

    def __call__(self, data_pr, data_gt, step, **kwargs):
        pass

class ConsistencyLoss(Loss):
    default_cfg={
        'use_ray_mask': False,
        'use_dr_loss': False,
        'use_dr_fine_loss': False,
        'use_nr_fine_loss': False,
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_prob','loss_prob_fine'])

    def __call__(self, data_pr, data_gt, step, **kwargs):
        if 'hit_prob_self' not in data_pr: return {}
        prob0 = data_pr['hit_prob_nr'].detach()     # qn,rn,dn
        prob1 = data_pr['hit_prob_self']            # qn,rn,dn
        if self.cfg['use_ray_mask']:
            ray_mask = data_pr['ray_mask'].float()  # 1,rn
        else:
            ray_mask = 1
        ce = - prob0 * torch.log(prob1 + 1e-5) - (1 - prob0) * torch.log(1 - prob1 + 1e-5)
        outputs={'loss_prob': torch.mean(torch.mean(ce,-1),1)}
        if 'hit_prob_nr_fine' in data_pr:
            prob0 = data_pr['hit_prob_nr_fine'].detach()     # qn,rn,dn
            prob1 = data_pr['hit_prob_self_fine']            # qn,rn,dn
            ce = - prob0 * torch.log(prob1 + 1e-5) - (1 - prob0) * torch.log(1 - prob1 + 1e-5)
            outputs['loss_prob_fine']=torch.mean(torch.mean(ce,-1),1)
        return outputs

class RenderLoss(Loss):
    default_cfg={
        'use_ray_mask': True,
        'use_dr_loss': False,
        'use_dr_fine_loss': False,
        'use_nr_fine_loss': False,
        'disable_at_eval': True,
        'render_loss_weight': 0.01
    }
    def __init__(self, cfg):
        self.cfg={**self.default_cfg,**cfg}
        super().__init__([f'loss_rgb'])

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        if not is_train and self.cfg['disable_at_eval']:
            return {}
        rgb_gt = data_pr['pixel_colors_gt'] # 1,rn,3
        rgb_nr = data_pr['pixel_colors_nr'] # 1,rn,3
        def compute_loss(rgb_pr,rgb_gt):
            loss=torch.sum((rgb_pr-rgb_gt)**2,-1)        # b,n
            if self.cfg['use_ray_mask']:
                ray_mask = data_pr['ray_mask'].float() # 1,rn
                loss = torch.sum(loss*ray_mask,1)/(torch.sum(ray_mask,1)+1e-3)
            else:
                loss = torch.mean(loss, 1)
            return loss * self.cfg['render_loss_weight']

        results = {'loss_rgb_nr': compute_loss(rgb_nr, rgb_gt)}
        if self.cfg['use_dr_loss']:
            rgb_dr = data_pr['pixel_colors_dr']  # 1,rn,3
            results['loss_rgb_dr'] = compute_loss(rgb_dr, rgb_gt)
        if self.cfg['use_dr_fine_loss']:
            results['loss_rgb_dr_fine'] = compute_loss(data_pr['pixel_colors_dr_fine'], rgb_gt)
        if self.cfg['use_nr_fine_loss']:
            results['loss_rgb_nr_fine'] = compute_loss(data_pr['pixel_colors_nr_fine'], rgb_gt)
        return results

class DepthLoss(Loss):
    default_cfg={
        'depth_correct_thresh': 0.02,
        'depth_loss_type': 'l2',
        'depth_loss_l1_beta': 0.05,
        'depth_loss_weight': 1,
        'disable_at_eval': True,
    }
    def __init__(self, cfg):
        super().__init__(['loss_depth'])
        self.cfg={**self.default_cfg,**cfg}
        if self.cfg['depth_loss_type']=='smooth_l1':
            self.loss_op=nn.SmoothL1Loss(reduction='none',beta=self.cfg['depth_loss_l1_beta'])

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        if not is_train and self.cfg['disable_at_eval']:
            return {}
        if 'true_depth' not in data_gt['ref_imgs_info']:
            print('no')
            return {'loss_depth': torch.zeros([1], dtype=torch.float32, device=data_pr['pixel_colors_nr'].device)}
        coords = data_pr['depth_coords'] # rfn,pn,2
        depth_pr = data_pr['depth_mean'] # rfn,pn
        depth_maps = data_gt['ref_imgs_info']['true_depth'] # rfn,1,h,w
        rfn, _, h, w = depth_maps.shape
        depth_gt = interpolate_feats(
            depth_maps,coords,h,w,padding_mode='border',align_corners=True)[...,0]   # rfn,pn

        # transform to inverse depth coordinate
        depth_range = data_gt['ref_imgs_info']['depth_range'] # rfn,2
        near, far = -1/depth_range[:,0:1], -1/depth_range[:,1:2] # rfn,1
        def process(depth):
            depth = torch.clamp(depth, min=1e-5)
            depth = -1 / depth
            depth = (depth - near) / (far - near)
            depth = torch.clamp(depth, min=0, max=1.0)
            return depth
        depth_gt = process(depth_gt)

        # compute loss
        def compute_loss(depth_pr):
            if self.cfg['depth_loss_type']=='l2':
                loss = (depth_gt - depth_pr)**2
            elif self.cfg['depth_loss_type']=='smooth_l1':
                loss = self.loss_op(depth_gt, depth_pr)

            if data_gt['scene_name'].startswith('gso'):
                depth_maps_noise = data_gt['ref_imgs_info']['depth']  # rfn,1,h,w
                depth_aug = interpolate_feats(depth_maps_noise, coords, h, w, padding_mode='border', align_corners=True)[..., 0]  # rfn,pn
                depth_aug = process(depth_aug)
                mask = (torch.abs(depth_aug-depth_gt)<self.cfg['depth_correct_thresh']).float()
                loss = torch.sum(loss * mask, 1) / (torch.sum(mask, 1) + 1e-4)

            return loss.mean()

        outputs = {'loss_depth': compute_loss(depth_pr) * self.cfg['depth_loss_weight']}
        if 'depth_mean_fine' in data_pr:
            outputs['loss_depth_fine'] = compute_loss(data_pr['depth_mean_fine']) * self.cfg['depth_loss_weight']
        return outputs

def compute_mae(pr, gt, mask):
    return np.sum(np.abs(pr * mask - gt * mask)) / np.count_nonzero(mask)

class SDFLoss(Loss):
    default_cfg={
        'loss_sdf_weight': 1.0,
        'loss_eikonal_weight': 0.1,
        'show_sdf_mae': True,
        'record_s': True,
        'loss_s_weight': 0
    }
    def __init__(self, cfg):
        super().__init__(['loss_sdf'])
        self.cfg={**self.default_cfg,**cfg}
        self.loss_fn = nn.SmoothL1Loss()

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        outputs = {}
        if self.cfg['show_sdf_mae']:
            sdf_pr = data_pr['volume'][0,0].detach().cpu().numpy()
            sdf_gt = data_gt['ref_imgs_info']['sdf_gt'].detach().cpu().numpy()
            valid_mask = sdf_gt != -1.0
            outputs['sdf_mae'] = torch.tensor([compute_mae(sdf_pr, sdf_gt, valid_mask)],dtype=torch.float32)
        if self.cfg['loss_sdf_weight'] > 0:
            valid_mask = data_gt['ref_imgs_info']['sdf_gt'] != -1.0
            outputs['loss_sdf'] = self.loss_fn(data_gt['ref_imgs_info']['sdf_gt'] * valid_mask, data_pr['volume'][0,0] * valid_mask)[None] * self.cfg['loss_sdf_weight']
        if self.cfg['loss_eikonal_weight'] > 0:
            outputs['loss_eikonal'] = (data_pr['sdf_gradient_error']).mean()[None] * self.cfg['loss_eikonal_weight'] 
        if self.cfg['record_s']:
            outputs['variance'] = data_pr['s'][None]
        if self.cfg['loss_s_weight'] > 0:
            outputs['loss_s'] = torch.norm(data_pr['s']).mean()[None] * self.cfg['loss_s_weight']
        return outputs

class VGNLoss(Loss):
    default_cfg={
        'loss_vgn_weight': 1e-2,
    }
    def __init__(self, cfg):
        super().__init__(['loss_vgn'])
        self.cfg={**self.default_cfg,**cfg}

    def _loss_fn(self, y_pred, y, is_train):
        label_pred, rotation_pred, width_pred = y_pred
        _, label, rotations, width = y
        loss_qual = self._qual_loss_fn(label_pred, label)
        acc = self._acc_fn(label_pred, label)
        loss_rot_raw = self._rot_loss_fn(rotation_pred, rotations)
        loss_rot = label * loss_rot_raw
        loss_width_raw = 0.01 * self._width_loss_fn(width_pred, width)
        loss_width = label * loss_width_raw
        loss = loss_qual + loss_rot + loss_width
        loss_item =  {'loss_vgn': loss.mean()[None] * self.cfg['loss_vgn_weight'], 
                     'vgn_total_loss':loss.mean()[None],'vgn_qual_loss': loss_qual.mean()[None], 
                    'vgn_rot_loss':  loss_rot.mean()[None], 'vgn_width_loss':loss_width.mean()[None],
                    'vgn_qual_acc': acc[None]}

        num = torch.count_nonzero(label)
        angle_torch = label * self._angle_error_fn(rotation_pred, rotations, 'torch')
        loss_item['vgn_rot_err'] = (angle_torch.sum() / num)[None] if num else torch.zeros((1,),device=label.device)
        return loss_item

    def _qual_loss_fn(self, pred, target):
        return F.binary_cross_entropy(pred, target, reduction="none")

    def _acc_fn(self, pred, target):
        return 100 * (torch.round(pred) == target).float().sum() / target.shape[0]

    def _pr_fn(self, pred, target):
        p, r = torchmetrics.functional.precision_recall(torch.round(pred).to(torch.int), target.to(torch.int), 'macro',num_classes=2)
        return p[None] * 100, r[None] * 100

    def _rot_loss_fn(self, pred, target):
        loss0 = self._quat_loss_fn(pred, target[:, 0])
        loss1 = self._quat_loss_fn(pred, target[:, 1])
        return torch.min(loss0, loss1)

    def _angle_error_fn(self, pred, target, method='torch'):
        if method == 'np':
            def _angle_error(q1, q2, ):  
                q1 = pyq.Quaternion(q1[[3,0,1,2]])
                q1 /= q1.norm
                q2 = pyq.Quaternion(q2[[3,0,1,2]])
                q2 /= q2.norm
                qd = q1.conjugate * q2
                qdv = pyq.Quaternion(0, qd.x, qd.y, qd.z)
                err = 2 * math.atan2(qdv.norm, qd.w) / math.pi * 180
                return min(err, 360 - err)
            q1s = pred.detach().cpu().numpy()
            q2s = target.detach().cpu().numpy()
            err = []
            for q1,q2 in zip(q1s, q2s):
                err.append(min(_angle_error(q1, q2[0]), _angle_error(q1, q2[1])))
            return torch.tensor(err, device = pred.device)
        elif method == 'torch':
            return calc_rot_error_from_qxyzw(pred, target)
        else:
            raise NotImplementedError

    def _quat_loss_fn(self, pred, target):
        return 1.0 - torch.abs(torch.sum(pred * target, dim=1))

    def _width_loss_fn(self, pred, target):
        return F.mse_loss(pred, target, reduction="none")

    def __call__(self, data_pr, data_gt, step, is_train=True, **kwargs):
        return self._loss_fn(data_pr['vgn_pred'], data_gt['grasp_info'], is_train)

name2loss={
    'render': RenderLoss,
    'depth': DepthLoss,
    'consist': ConsistencyLoss,
    'vgn': VGNLoss,
    'sdf': SDFLoss
}