import abc
import glob
import json
import os
import re
from pathlib import Path
import sys
import open3d as o3d
from utils.draw_utils import draw_gripper_o3d
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from skimage.io import imread, imsave

from asset import VGN_TRAIN_ROOT, VGN_TEST_ROOT, VGN_PILE_TRAIN_CSV,  VGN_PACK_TRAIN_CSV, VGN_PILE_TEST_CSV,VGN_PACK_TEST_CSV, VGN_SDF_DIR

from utils.draw_utils import draw_cube, draw_axis, draw_points, draw_gripper, draw_world_points
sys.path.append("../")
from gd.utils.transform import Rotation, Transform

class BaseDatabase(abc.ABC):
    def __init__(self, database_name):
        self.database_name = database_name

    @abc.abstractmethod
    def get_image(self, img_id):
        pass

    @abc.abstractmethod
    def get_K(self, img_id):
        pass

    @abc.abstractmethod
    def get_pose(self, img_id):
        pass

    @abc.abstractmethod
    def get_img_ids(self,check_depth_exist=False):
        pass

    @abc.abstractmethod
    def get_bbox(self, img_id):
        pass

    @abc.abstractmethod
    def get_depth(self,img_id):
        pass

    @abc.abstractmethod
    def get_mask(self,img_id):
        pass

    @abc.abstractmethod
    def get_depth_range(self,img_id):
        pass

class GraspSynDatabase(BaseDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        self.debug_save_dir = Path(f'output/nrvgn/{database_name}')
        tp, split, scene_type, scene_split, scene_id, background_size = database_name.split('/')
        background, size = background_size.split('_')
        self.split = split
        self.scene_id = scene_id
        self.scene_type = scene_type
        self.tp = tp
        self.downSample = float(size) 
        tp2wh = {
            'vgn_syn': (640, 360)
        }
        src_wh = tp2wh[tp]
        self.img_wh = (np.array(src_wh) * self.downSample).astype(int)

        root_dir = {'test':     {
                                 'vgn_syn':  VGN_TEST_ROOT,
                                },
                    'train':    {
                                 'vgn_syn': VGN_TRAIN_ROOT,
                                },
                    }

        if tp == 'vgn_syn':
            self.root_dir = Path(root_dir[split][tp]) / (scene_type + "_full") / scene_split / scene_id 
        else:
            raise NotImplementedError

        tp2len = {'grasp_syn': 256,
                 'vgn_syn':24}
        self.depth_img_ids = self.img_ids = list(range(tp2len[tp]))
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        self.K = np.array([[
            892.62,
            0.0,
            639.5
            ],
            [
                0.0,
                892.62,
                359.5
            ],
            [
                0.0,
                0.0,
                1.0
            ]]) 
        self.K[:2] = self.K[:2] * self.downSample
        if self.tp == 'vgn_syn':
            self.K[:2] /= 2
        self.poses_ori = np.load(self.root_dir / 'camera_pose.npy')
        self.poses = [np.linalg.inv(p @ self.blender2opencv)[:3,:] for p in self.poses_ori]
        
        
        self.depth_thres = {
            'grasp_syn': 1.5,
            'vgn_syn': 0.8,
        }
        self.fixed_depth_range = [0.2, 0.8]

        tp2bbox3d = {'grasp_syn': [[-0.35, -0.45, 0],
                                    [0.15, 0.05, 0.5]],
                    'vgn_syn': [[-0.15, -0.15, -0.05],
                                [0.15, 0.15, 0.25]]}
        self.bbox3d = tp2bbox3d[tp]

    def get_split(self):
        return self.split

    def get_image(self, img_id):
        img_filename = os.path.join(self.root_dir,
                            f'rgb/{img_id:04d}.png')
        img = imread(img_filename)[:,:,:3]
        img = cv2.resize(img, self.img_wh)
        #img[self.get_mask(img_id)] = 255
        return np.asarray(img, dtype=np.float32)

    def get_K(self, img_id):
        return self.K.astype(np.float32).copy()

    def get_pose(self, img_id):
        return self.poses[img_id].astype(np.float32).copy()

    def get_img_ids(self,check_depth_exist=False):
        if check_depth_exist: return self.depth_img_ids
        return self.img_ids

    def get_bbox3d(self, vis=False):
        if vis:
            img_id = 0
            img = self.get_image(img_id)
            cRb = self.poses[img_id][:3,:3]
            ctb = self.poses[img_id][:3,3]
            l = self.bbox3d[1][0] - self.bbox3d[0][0]
            img = draw_cube(img, cRb, ctb, self.K, length=l, bias=self.bbox3d[0])
            if not self.debug_save_dir.exists():
                self.debug_save_dir.mkdir(parents=True)
            imsave(str(self.debug_save_dir / 'bbox3d.jpg'), img)
        return self.bbox3d

    def get_bbox(self, img_id, vis=False):
        mask = self.get_mask(img_id,'obj')
        xs,ys=np.nonzero(mask)
        x_min,x_max=np.min(xs,0),np.max(xs,0)
        y_min,y_max=np.min(ys,0),np.max(ys,0)

        if vis:
            img = self.get_image(img_id)
            img = cv2.rectangle(img, (y_min, x_min), (y_max, x_max), (255,0,0), 2)

            imsave(str(self.debug_save_dir / 'box.jpg'), img)

        return [x_min,x_max,y_min,y_max]
        
    def _depth_existence(self,img_id):
        return True

    def get_depth(self, img_id):
        depth_filename = os.path.join(self.root_dir,
                    f'depth/{img_id:04d}.exr')
        depth_h = cv2.imread(depth_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]
        depth_h = cv2.resize(depth_h, self.img_wh, interpolation=cv2.INTER_NEAREST)

        return depth_h

    def get_mask(self, img_id, tp='desk'):
        if tp == 'desk':
            mask = self.get_depth(img_id) < self.depth_thres[self.tp]
            return (mask.astype(np.bool))
        elif tp == 'obj':
            mask_filename = os.path.join(self.root_dir,
                        f'mask/{img_id:04d}.exr')
            mask = cv2.imread(mask_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,0]
            mask = cv2.resize(mask, self.img_wh, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite('mask.jpg', mask * 256)

            return ~(mask.astype(np.bool))
        else:
            return np.ones((self.img_wh[1], self.img_wh[0]))

    def get_depth_range(self,img_id, fixed=True):
        if fixed:
            return np.array(self.fixed_depth_range)
        depth = self.get_depth(img_id)
        nf = [max(0, np.min(depth)), min(self.depth_thres[self.tp], np.max(depth))]
        return np.array(nf)

    def get_sdf(self):
        sdf_volume = np.load( Path(VGN_SDF_DIR) / f'{self.scene_id}.npz')['grid'][0]
        return sdf_volume * 2 - 1

class VGNSynDatabase(GraspSynDatabase):
    def __init__(self, database_name):
        super().__init__(database_name)
        split = self.get_split()

        if self.scene_type == 'packed':
            csv = VGN_PACK_TEST_CSV if split == 'test' else VGN_PACK_TRAIN_CSV
        elif self.scene_type == 'pile':
            csv = VGN_PILE_TEST_CSV if split == 'test' else VGN_PILE_TRAIN_CSV
        else:
            return

        self.df = csv
        self.df = self.df[self.df["scene_id"] == self.scene_id]
        assert len(self.df) > 0, f"empty grasping info {database_name}"

    def visualize_grasping(self, pos, rot, w, label=None, img_id=3,save_img=False):
        voxel_size = 0.3 / 40
        pts_w = pos * voxel_size  
        width = w * voxel_size
        
        img = self.get_image(img_id)
        
        t = np.array([[-0.15, -0.15, -0.05]]).repeat(pts_w.shape[0], axis=0)
        pts_b = pts_w + t
        
        cRb = self.poses[img_id][:3,:3]
        ctb = self.poses[img_id][:3,3] # + np.array([-0.15, -0.15, -0.05])

        for gid in range(pts_w.shape[0]):
            if label is not None and label[gid] == 0:
                continue
            btg = pts_b[gid]
            wRg = rot[gid]
            bRg = wRg 
            bTg = np.eye(4)
            bTg[:3,:3] = bRg
            bTg[:3,3] = btg
            cTb = self.poses[img_id]
            cTg = cTb @ bTg
            img = draw_gripper(img, cTg[:3,:3], cTg[:3,3], self.K, width[gid], 2)
            img = draw_world_points(img, pts_b[gid], cRb, ctb, self.K)

        if save_img:
            save_dir = str(self.debug_save_dir / f'gripper_test-{img_id}.jpg')
            print("save to", save_dir)
            imsave(save_dir, img)
        return img 

    def visualize_grasping_3d(self, pos, rot, w, label=None, voxel_size = 0.3 / 40):
        pts_w = pos * voxel_size
        width = w * voxel_size

        geometry = o3d.geometry.TriangleMesh()
        for gid in range(pts_w.shape[0]):
            if label is not None and label[gid] == 0:
                continue
            wRg = rot[gid]
            y_ccw_90 = np.array([[0, 0, -1], [0, 1,0], [1, 0, 0]])
            _R = wRg @ y_ccw_90
            _t = pts_w[gid]

            geometry_gripper = draw_gripper_o3d(_R, _t, width[gid])
            geometry += geometry_gripper

        o3d.io.write_triangle_mesh(str(self.debug_save_dir / f'gripper.ply'), geometry)

    def get_grasp_info(self):
        pos = self.df[["i","j","k"]].to_numpy(np.single)
        index = np.round(pos).astype(np.long)
        l = pos.shape[0]
        width = self.df[["width"]].to_numpy(np.single).reshape(l)
        label = self.df[["label"]].to_numpy(np.float32).reshape(l)
        rotations = np.empty((l, 2, 4), dtype=np.single)
        q = self.df[["qx","qy","qz","qw"]].to_numpy(np.single)
        ori = Rotation.from_quat(q)
        R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
        rotations[:,0] = ori.as_quat()
        rotations[:,1] = (ori * R).as_quat()

        # for i in range(4):
        #     self.visualize_grasping(pos, ori.as_matrix(), width, label, i)
        # exit()
        return (index, label, rotations, width)


def parse_database_name(database_name:str)->BaseDatabase:
    name2database={
        'vgn_syn': VGNSynDatabase,
    }
    database_type = database_name.split('/')[0]
    if database_type in name2database:
        return name2database[database_type](database_name)
    else:
        raise NotImplementedError

def get_database_split(database: BaseDatabase, split_type='val'):
    database_name = database.database_name
    if split_type.startswith('val'):
        splits = split_type.split('_')
        depth_valid = not(len(splits)>1 and splits[1]=='all')
        if database_name.startswith('vgn'):
            val_ids = database.get_img_ids()[2:24:8]# TODO
            train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
        else:
            raise NotImplementedError
    elif split_type.startswith('test'):
        splits = split_type.split('_')
        depth_valid = not(len(splits)>1 and splits[1]=='all')
        if database_name.startswith('vgn'):
            val_ids = database.get_img_ids()[2:24:8] + [0]# TODO
            train_ids = [img_id for img_id in database.get_img_ids(check_depth_exist=depth_valid) if img_id not in val_ids]
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return train_ids, val_ids