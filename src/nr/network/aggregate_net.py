import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
import numpy as np

from network.ibrnet import IBRNetWithNeuRay, IBRNetWithNeuRayNeus
from network.neus import SingleVarianceNetwork


def get_dir_diff(prj_dir,que_dir):
    rfn, qn, rn, dn, _ = prj_dir.shape
    dir_diff = prj_dir - que_dir.unsqueeze(0)  # rfn,qn,rn,dn,3
    dir_dot = torch.sum(prj_dir * que_dir.unsqueeze(0), -1, keepdim=True)
    dir_diff = torch.cat([dir_diff, dir_dot], -1)  # rfn,qn,rn,dn,4
    dir_diff = dir_diff.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)
    return dir_diff

class BaseAggregationNet(nn.Module):
    default_cfg={
        'sample_num': 64,
        'neuray_dim': 32,
        'use_img_feats': False,
    }
    def __init__(self, cfg):
        super().__init__()
        self.cfg={**self.default_cfg, **cfg}
        dim = self.cfg['neuray_dim']
        self.prob_embed = nn.Sequential(
            nn.Linear(2+32, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def _get_embedding(self, prj_dict, que_dir):
        """
        :param prj_dict
             prj_ray_feats: rfn,qn,rn,dn,f
             prj_hit_prob:  rfn,qn,rn,dn,1
             prj_vis:       rfn,qn,rn,dn,1
             prj_alpha:     rfn,qn,rn,dn,1
             prj_rgb:       rfn,qn,rn,dn,3
             prj_dir:       rfn,qn,rn,dn,3
        :param que_dir:       qn,rn,dn,3
        :return: qn,rn,dn
        """
        hit_prob_val = (prj_dict['hit_prob']-0.5)*2
        vis_val = (prj_dict['vis']-0.5)*2

        prj_hit_prob, prj_vis, prj_rgb, prj_dir, prj_ray_feats = \
            hit_prob_val, vis_val, prj_dict['rgb'], prj_dict['dir'], prj_dict['ray_feats']
        rfn,qn,rn,dn,_ = hit_prob_val.shape

        prob_embedding = self.prob_embed(torch.cat([prj_ray_feats, prj_hit_prob, prj_vis],-1))

        if que_dir is not None:
            dir_diff = get_dir_diff(prj_dir, que_dir)
        else:
            _,qn,rn,dn,_ = prj_hit_prob.shape
            dir_diff = torch.zeros((rfn, qn * rn, dn, 4)).permute(1, 2, 0, 3).to(prj_hit_prob.device)

        valid_mask = prj_dict['mask']
        valid_mask = valid_mask.float() # rfn,qn,rn,dn
        valid_mask = valid_mask.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)

        prj_img_feats = prj_dict['img_feats']
        prj_img_feats = torch.cat([prj_rgb, prj_img_feats], -1)
        prj_img_feats = prj_img_feats.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)
        prob_embedding = prob_embedding.reshape(rfn, qn * rn, dn, -1).permute(1, 2, 0, 3)
        return prj_img_feats, prob_embedding, dir_diff, valid_mask

class DefaultAggregationNet(BaseAggregationNet):
    def __init__(self,cfg):
        super().__init__(cfg)
        dim = self.cfg['neuray_dim']
        self.agg_impl = IBRNetWithNeuRay(dim,n_samples=self.cfg['sample_num'])

    def forward(self, prj_dict, que_dir, que_pts=None, que_dists=None):
        qn,rn,dn,_ = que_dir.shape
        prj_img_feats, prob_embedding, dir_diff, valid_mask = self._get_embedding(prj_dict, que_dir)
        outs = self.agg_impl(prj_img_feats, prob_embedding, dir_diff, valid_mask)
        colors = outs[...,:3] # qn*rn,dn,3
        density = outs[...,3] # qn*rn,dn,0
        return density.reshape(qn,rn,dn), colors.reshape(qn,rn,dn,3)


class NeusAggregationNet(BaseAggregationNet):
    neus_default_cfg = {
        'cos_anneal_end_iter': 0,
        'init_s': 0.3,
        'fix_s': False
    }
    def __init__(self,cfg):
        cfg = {**self.neus_default_cfg, **cfg}
        super().__init__(cfg)
        dim = self.cfg['neuray_dim']
        self.agg_impl = IBRNetWithNeuRayNeus(dim,n_samples=self.cfg['sample_num'])
        self.deviation_network = SingleVarianceNetwork(self.cfg['init_s'], self.cfg['fix_s'])
        self.step = 0
        self.cos_anneal_ratio = 1.0

    def _update_cos_anneal_ratio(self):
        self.cos_anneal_ratio = np.min([1.0, self.step / self.cfg['cos_anneal_end_iter']])

    def _get_alpha_from_sdf(self, sdf, grad, que_dir, que_dists):
        qn,rn,dn,_ = que_dir.shape
        inv_s = self.deviation_network(torch.zeros([1, 3], device=sdf.device))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(qn*rn, dn)
        true_cos = (-que_dir * grad).sum(-1, keepdim=True)
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     F.relu(-true_cos) * self.cos_anneal_ratio)[0].squeeze(-1)  # always non-positive
        # Estimate signed distances at section points
        estimated_next_sdf = sdf + iter_cos * que_dists[0] * 0.5
        estimated_prev_sdf = sdf - iter_cos * que_dists[0] * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(qn,rn,dn).clip(0.0, 1.0)

        return alpha

    def forward(self, prj_dict, que_dir, que_pts, que_dists, is_train):
        if self.cfg['cos_anneal_end_iter'] and is_train:
            self._update_cos_anneal_ratio()
        qn,rn,dn,_ = que_dir.shape
        prj_img_feats, prob_embedding, dir_diff, valid_mask = self._get_embedding(prj_dict, que_dir)
        outs, grad = self.agg_impl(prj_img_feats, prob_embedding, dir_diff, valid_mask, que_pts)
        colors = outs[...,:3] # qn*rn,dn,3
        sdf = outs[...,3] # qn*rn,dn,0
        if que_dists is None:
            return None, sdf.reshape(qn,rn,dn), colors.reshape(qn,rn,dn,3), None, None
        if is_train:
            self.step += 1
            self.deviation_network.set_step(self.step)
        alpha = self._get_alpha_from_sdf(sdf, grad, que_dir, que_dists)
        grad_error = torch.mean((torch.linalg.norm(grad.reshape(qn,rn,dn,3), ord=2, dim=-1) - 1.0) ** 2).reshape(1,1)
        return alpha.reshape(qn,rn,dn), sdf.reshape(qn,rn,dn), colors.reshape(qn,rn,dn,3), grad_error, self.deviation_network.variance.reshape(1,1)


name2agg_net={
    'default': DefaultAggregationNet,
    'neus': NeusAggregationNet
}