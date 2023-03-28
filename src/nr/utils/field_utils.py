import torch
import numpy as np

def generate_grid_points_old(bound_min, bound_max, resolution):
    X = torch.linspace(bound_min[0], bound_max[0], resolution)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution)
    Z = torch.linspace(bound_max[2], bound_min[2], resolution) # from top to down to be like with training rays
    XYZ = torch.stack(torch.meshgrid(X, Y, Z), dim=-1)

    return XYZ

RESOLUTION = 40
VOLUME_SIZE = 0.3
VOXEL_SIZE = VOLUME_SIZE / RESOLUTION
HALF_VOXEL_SIZE = VOXEL_SIZE / 2

def generate_grid_points():
    points = []
    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            for z in range(RESOLUTION):
                points.append([x * VOXEL_SIZE + HALF_VOXEL_SIZE,
                                y * VOXEL_SIZE + HALF_VOXEL_SIZE,
                               z * VOXEL_SIZE + HALF_VOXEL_SIZE])
    return np.array(points).astype(np.float32)

TSDF_SAMPLE_POINTS = generate_grid_points()

if __name__ == "__main__":
    GT_POINTS = np.load('points.npy')
    TSDF_VOLUME_MASK = np.zeros((1, 40, 40, 40), dtype=np.bool8)
    idxs = []
    for point in GT_POINTS:
        i, j, k = np.floor(point / VOXEL_SIZE).astype(int)
        TSDF_VOLUME_MASK[0, i, j, k] = True
        idxs.append(i * (RESOLUTION * RESOLUTION) + j * RESOLUTION + k)
    print(TSDF_SAMPLE_POINTS[idxs], GT_POINTS)
    assert np.allclose(TSDF_SAMPLE_POINTS[idxs], GT_POINTS)

