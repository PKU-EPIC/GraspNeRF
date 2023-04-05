import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.ops import interpolate_feats, masked_mean_var, ResEncoder, ResUNetLight, conv3x3, ResidualBlock, conv1x1

class CostVolumeInitNet(nn.Module):
    default_cfg={
        'cost_volume_sn': 64,
    }
    def __init__(self,cfg):
        super().__init__()
        self.cfg={**self.default_cfg,**cfg}

        imagenet_mean = torch.from_numpy(np.asarray([0.485, 0.456, 0.406], np.float32)).cuda()[None, :, None, None]
        imagenet_std = torch.from_numpy(np.asarray([0.229, 0.224, 0.225], np.float32)).cuda()[None, :, None, None]
        self.register_buffer('imagenet_mean', imagenet_mean)
        self.register_buffer('imagenet_std', imagenet_std)

        self.res_net = ResUNetLight(out_dim=32)
        norm_layer = lambda dim: nn.InstanceNorm2d(dim, track_running_stats=False, affine=True)


        in_dim = 32

        self.out_conv = nn.Sequential(
            conv3x3(in_dim, 32),
            ResidualBlock(32, 32, norm_layer=norm_layer),
            conv1x1(32, 32),
        )

    def forward(self, ref_imgs_info, src_imgs_info, is_train):
        ref_feats = self.res_net(ref_imgs_info['imgs'])
        return self.out_conv(torch.cat([ref_feats], 1))

name2init_net={
    'cost_volume': CostVolumeInitNet,
}