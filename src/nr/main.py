import sys, os
import time

sys.path.append("./src/nr")
from pathlib import Path
import numpy as np

import torch
from skimage.io import imsave, imread
from network.renderer import name2network
from utils.base_utils import load_cfg, to_cuda
from utils.imgs_info import build_render_imgs_info, imgs_info_to_torch, grasp_info_to_torch
from network.renderer import name2network
from utils.base_utils import color_map_forward
from network.loss import VGNLoss
from tqdm import tqdm
from scipy import ndimage
import cv2
from gd.utils.transform import Transform, Rotation
from gd.grasp import *


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=1.33,
    max_width=9.33,
    tsdf_thres_high = 0.5,
    tsdf_thres_low = 1e-3,
    n_grasp=0
):
    tsdf_vol = tsdf_vol.squeeze()  
    qual_vol = qual_vol.squeeze()  
    rot_vol = rot_vol.squeeze()  
    width_vol = width_vol.squeeze()
    # smooth quality volume with a Gaussian
    qual_vol = ndimage.gaussian_filter(
        qual_vol, sigma=gaussian_filter_sigma, mode="nearest"
    )

    # mask out voxels too far away from the surface
    outside_voxels = tsdf_vol > tsdf_thres_high
    inside_voxels = np.logical_and(tsdf_thres_low < tsdf_vol, tsdf_vol < tsdf_thres_high)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0
    
    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return qual_vol, rot_vol, width_vol


def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
    qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores, indexs = [], [], []
    for index in np.argwhere(mask):
        indexs.append(index)
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)
    return grasps, scores, indexs


def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    rot = rot_vol[:, i, j, k]
    ori = Rotation.from_quat(rot)
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score


class GraspNeRFPlanner(object):
    def set_params(self, args):
        self.args = args
        self.voxel_size = 0.3 / 40
        self.bbox3d =  [[-0.15, -0.15, -0.0503],[0.15, 0.15, 0.2497]]
        self.tsdf_thres_high = 0 
        self.tsdf_thres_low = -0.85

        self.renderer_root_dir = self.args.renderer_root_dir
        tp, split, scene_type, scene_split, scene_id, background_size = args.database_name.split('/')
        background, size = background_size.split('_')
        self.split = split
        self.tp = tp
        self.downSample = float(size) 
        tp2wh = {
            'vgn_syn': (640, 360)
        }
        src_wh = tp2wh[tp]
        self.img_wh = (np.array(src_wh) * self.downSample).astype(int)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.K = np.array([[892.62, 0.0, 639.5],
                           [0.0, 892.62, 359.5],
                           [0.0, 0.0, 1.0]]) 
        self.K[:2] = self.K[:2] * self.downSample
        if self.tp == 'vgn_syn':
            self.K[:2] /= 2
        self.depth_thres = {
            'vgn_syn': 0.8,
        }
        
        if args.object_set == "graspnet":
            dir_name = "pile_graspnet_test"
        else:
            if self.args.scene == "pile":
                dir_name = "pile_pile_test_200"
            elif self.args.scene == "packed":
                dir_name = "packed_packed_test_200"
            elif self.args.scene == "single":
                dir_name = "single_single_test_200"

        scene_root_dir = os.path.join(self.renderer_root_dir, "data/mesh_pose_list", dir_name)
        self.mesh_pose_list = [i for i in sorted(os.listdir(scene_root_dir))]
        self.depth_root_dir = ""
        self.depth_list = []

    def __init__(self, args=None, cfg_fn=None, debug_dir=None) -> None:
        default_render_cfg = {
        'min_wn': 3, # working view number
        'ref_pad_interval': 16, # input image size should be multiple of 16
        'use_src_imgs': False, # use source images to construct cost volume or not
        'cost_volume_nn_num': 3, # number of source views used in cost volume
        'use_depth': True, # use colmap depth in rendering or not,
        }
        # load render cfg
        if cfg_fn is None:
            self.set_params(args)
            cfg = load_cfg(args.cfg_fn)
        else:
            cfg = load_cfg(cfg_fn)

        print(f"[I] GraspNeRFPlanner: using ckpt: {cfg['name']}")
        render_cfg = cfg['train_dataset_cfg'] if 'train_dataset_cfg' in cfg else {}
        render_cfg = {**default_render_cfg, **render_cfg}
        cfg['render_rgb'] = False # only for training. Disable in grasping.
        # load model
        self.net = name2network[cfg['network']](cfg)
        ckpt_filename = 'model_best'
        ckpt = torch.load(Path('src/nr/ckpt') / cfg["group_name"] / cfg["name"] / f'{ckpt_filename}.pth')
        self.net.load_state_dict(ckpt['network_state_dict'])
        self.net.cuda()
        self.net.eval()
        self.step = ckpt["step"]
        self.output_dir = debug_dir
        if debug_dir is not None:
            if not Path(debug_dir).exists():
                Path(debug_dir).mkdir(parents=True)
        self.loss = VGNLoss({})
        self.num_input_views = render_cfg['num_input_views']
        print(f"[I] GraspNeRFPlanner: load model at step {self.step} of best metric {ckpt['best_para']}")

    def get_image(self, img_id, round_idx):
        img_filename = os.path.join(self.args.log_root_dir, "rendered_results/" + str(self.args.logdir).split("/")[-1], "rgb/%04d.png"%img_id)
        img = imread(img_filename)[:,:,:3]
        img = cv2.resize(img, self.img_wh)
        return np.asarray(img, dtype=np.float32)
    
    def get_pose(self, img_id):
        poses_ori = np.load(Path(self.renderer_root_dir) / 'camera_pose.npy')
        poses = [np.linalg.inv(p @ self.blender2opencv)[:3,:] for p in poses_ori]
        return poses[img_id].astype(np.float32).copy()
    
    def get_K(self, img_id):
        return self.K.astype(np.float32).copy()

    def get_depth_range(self,img_id, round_idx, fixed=False):
        if fixed:
            return np.array([0.2,0.8])
        depth = self.get_depth(img_id, round_idx)
        nf = [max(0, np.min(depth)), min(self.depth_thres[self.tp], np.max(depth))]
        return np.array(nf)
    
    def __call__(self, test_view_id, round_idx, n_grasp, gt_tsdf):
        # load data for test
        images = [self.get_image(i, round_idx) for i in test_view_id]
        images = color_map_forward(np.stack(images, 0)).transpose([0, 3, 1, 2])
        extrinsics = np.stack([self.get_pose(i) for i in test_view_id], 0)
        intrinsics = np.stack([self.get_K(i) for i in test_view_id], 0)
        depth_range = np.asarray([self.get_depth_range(i, round_idx, fixed = True) for i in test_view_id], dtype=np.float32)
        
        tsdf_vol, qual_vol_ori, rot_vol_ori, width_vol_ori, toc = self.core(images, extrinsics, intrinsics, depth_range, self.bbox3d)

        qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol_ori, rot_vol_ori, width_vol_ori, tsdf_thres_high=self.tsdf_thres_high, tsdf_thres_low=self.tsdf_thres_low, n_grasp=n_grasp)
        grasps, scores, indexs = select(qual_vol.copy(), rot_vol, width_vol)
        grasps, scores, indexs = np.asarray(grasps), np.asarray(scores), np.asarray(indexs)

        if len(grasps) > 0:
            np.random.seed(self.args.seed + round_idx + n_grasp)
            p = np.random.permutation(len(grasps))  
            grasps = [from_voxel_coordinates(g, self.voxel_size) for g in grasps[p]]
            scores = scores[p]
            indexs = indexs[p]

        return grasps, scores, toc
    
    def core(self, 
                images: np.ndarray, 
                extrinsics: np.ndarray, 
                intrinsics: np.ndarray, 
                depth_range=[0.2, 0.8], 
                bbox3d=[[-0.15, -0.15, -0.05],[0.15, 0.15, 0.25]], gt_info=None, que_id=0):
        """
        @args
            images: np array of shape (3, 3, h, w), image in RGB format
            extrinsics: np array of shape (3, 4, 4), the transformation matrix from world to camera
            intrinsics: np array of shape (3, 3, 3)
        @rets
            volume, label, rot, width: np array of shape (1, 1, res, res, res)
        """
        _, _, h, w = images.shape
        assert h % 32 == 0 and w % 32 == 0
        extrinsics = extrinsics[:, :3, :]
        que_imgs_info = build_render_imgs_info(extrinsics[que_id], intrinsics[que_id], (h, w), depth_range[que_id])
        src_imgs_info = {'imgs': images, 'poses': extrinsics.astype(np.float32), 'Ks': intrinsics.astype(np.float32), 'depth_range': depth_range.astype(np.float32), 
                                'bbox3d': np.array(bbox3d)}

        ref_imgs_info = src_imgs_info.copy()
        num_views = images.shape[0]
        ref_imgs_info['nn_ids'] = np.arange(num_views).repeat(num_views, 0)
        data = {'step': self.step , 'eval': True}
        if not gt_info:
            data['full_vol'] = True
        else:
            data['grasp_info'] = to_cuda(grasp_info_to_torch(gt_info))
        data['que_imgs_info'] = to_cuda(imgs_info_to_torch(que_imgs_info))
        data['src_imgs_info'] = to_cuda(imgs_info_to_torch(src_imgs_info))
        data['ref_imgs_info'] = to_cuda(imgs_info_to_torch(ref_imgs_info))

        with torch.no_grad():
            t0 = time.time()
            render_info = self.net(data)
            t = time.time() - t0
        
        if gt_info:
            return self.loss(render_info, data, self.step, False)

        label, rot, width = render_info['vgn_pred']
        
        return render_info['volume'].cpu().numpy(), label.cpu().numpy(), rot.cpu().numpy(), width.cpu().numpy(), t