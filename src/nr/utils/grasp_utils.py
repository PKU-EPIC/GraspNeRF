import datetime
from pathlib import Path
import numpy as np
from scipy import ndimage
import sys
sys.path.append("./src")
import time

from nr.utils.base_utils import color_map_forward
from nr.utils.draw_utils import draw_cube, extract_surface_points_from_volume
from gd.utils.transform import Transform, Rotation
from skimage.io import imsave
import cv2

class Grasp(object):
    """Grasp parameterized as pose of a 2-finger robot hand.
    
    TODO(mbreyer): clarify definition of grasp frame
    """

    def __init__(self, pose, width):
        self.pose = pose
        self.width = width


def to_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation /= voxel_size
    width = grasp.width / voxel_size
    return Grasp(pose, width)


def from_voxel_coordinates(grasp, voxel_size):
    pose = grasp.pose
    pose.translation *= voxel_size
    width = grasp.width * voxel_size
    return Grasp(pose, width)


def process(
    tsdf_vol,
    qual_vol,
    rot_vol,
    width_vol,
    gaussian_filter_sigma=1.0,
    min_width=0,
    max_width=12,
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
    outside_voxels = tsdf_vol > 0.1
    inside_voxels = np.logical_and(-1 < tsdf_vol, tsdf_vol < -0.1)
    valid_voxels = ndimage.morphology.binary_dilation(
        outside_voxels, iterations=2, mask=np.logical_not(inside_voxels)
    )
    qual_vol[valid_voxels == False] = 0.0
    # reject voxels with predicted widths that are too small or too large
    qual_vol[np.logical_or(width_vol < min_width, width_vol > max_width)] = 0.0

    return tsdf_vol, qual_vol, rot_vol, width_vol

def select_index(qual_vol, rot_vol, width_vol, index):
    i, j, k = index
    score = qual_vol[i, j, k]
    ori = Rotation.from_quat(rot_vol[:, i, j, k])
    pos = np.array([i, j, k], dtype=np.float64)
    width = width_vol[i, j, k]
    return Grasp(Transform(ori, pos), width), score

def select(qual_vol, rot_vol, width_vol, threshold=0.90, max_filter_size=4):
    # threshold on grasp quality
    qual_vol[qual_vol < threshold] = 0.0

    # non maximum suppression
    max_vol = ndimage.maximum_filter(qual_vol, size=max_filter_size)
    qual_vol = np.where(qual_vol == max_vol, qual_vol, 0.0)
    mask = np.where(qual_vol, 1.0, 0.0)

    # construct grasps
    grasps, scores = [], []
    for index in np.argwhere(mask):
        grasp, score = select_index(qual_vol, rot_vol, width_vol, index)
        grasps.append(grasp)
        scores.append(score)

    return grasps, scores


def sim_grasp(database,  alp_vol, qual_vol, rot_vol, width_vol, top_k=10):
    from utils.grasp_utils import select, process
    qual_vol, rot_vol, width_vol = process(alp_vol, qual_vol, rot_vol, width_vol)
    grasps, scores = select(qual_vol.copy(), rot_vol, width_vol)
    grasps, scores = np.asarray(grasps), np.asarray(scores)
    
    img = None
    if len(grasps) > 0:
        p = np.argsort(scores)[::-1][:top_k]
        grasps = [g for g in grasps[p]]
        scores = scores[p]
        pos = np.array([ g.pose.translation for g in grasps ])
        rot = np.array([ g.pose.rotation.as_matrix() for g in grasps ])
        width =  np.array([ g.width for g in grasps ])

        img = database.visualize_grasping(pos, rot, width)
        database.visualize_grasping_3d(pos, rot, width, scores)
            

    return grasps, scores, img 


def run_real(run_id, model, images: list, extrinsics: list, intrinsic, save_img=True):
    extrinsics = np.stack(extrinsics, 0)
    intrinsics = np.repeat(np.expand_dims(intrinsic, 0), extrinsics.shape[0], axis=0)
    depth_range = np.repeat(np.expand_dims(np.r_[0.2, 0.8], 0), extrinsics.shape[0], axis=0).astype(np.float32)
    bbox3d = [[-0.15, -0.15, 0.00], [0.15, 0.15, 0.3]]

    if save_img:
        save_path = f'data/grasp_capture/{run_id}'
        if not Path(save_path).exists():
            Path(save_path).mkdir(parents=True)
        for i, img in enumerate(images):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img = draw_cube(img, extrinsics[i][:3,:3], extrinsics[i][:3,3], intrinsics[i], length=0.3, bias=bbox3d[0])
            cv2.imwrite(f"{save_path}/{i}.png", img)

    images = color_map_forward(np.stack(images, 0)).transpose([0, 3, 1, 2])
    
    t0 = time.time()
    tsdf_vol, qual_vol, rot_vol, width_vol = model(images, extrinsics, intrinsics, depth_range=depth_range, bbox3d=bbox3d, que_id=3)
    t = time.time() - t0

    tsdf_vol, qual_vol, rot_vol, width_vol = process(tsdf_vol, qual_vol, rot_vol, width_vol)
    grasps, scores = select(qual_vol.copy(), rot_vol, width_vol)
    grasps, scores = np.asarray(grasps), np.asarray(scores)

    if len(grasps) > 0:
        p = np.random.permutation(len(grasps))
        grasps = [from_voxel_coordinates(g, 0.3 / 40) for g in grasps[p]]
        scores = scores[p]

    pc = extract_surface_points_from_volume(tsdf_vol, (-0.2, 0.2))

    return grasps, scores, tsdf_vol, pc, t