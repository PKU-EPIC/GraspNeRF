import matplotlib
matplotlib.use('Agg')
import sys
sys.path.append("./src/nr")

from utils.base_utils import compute_relative_transformation, compute_F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import open3d as o3d

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def draw_correspondence(img0, img1, kps0, kps1, matches=None, colors=None, max_draw_line_num=None, kps_color=(0,0,255),vert=False):
    if len(img0.shape)==2:
        img0=np.repeat(img0[:,:,None],3,2)
    if len(img1.shape)==2:
        img1=np.repeat(img1[:,:,None],3,2)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    if matches is None:
        assert(kps0.shape[0]==kps1.shape[0])
        matches=np.repeat(np.arange(kps0.shape[0])[:,None],2,1)

    if vert:
        w = max(w0, w1)
        h = h0 + h1
        out_img = np.zeros([h, w, 3], np.uint8)
        out_img[:h0, :w0] = img0
        out_img[h0:, :w1] = img1
    else:
        h = max(h0, h1)
        w = w0 + w1
        out_img = np.zeros([h, w, 3], np.uint8)
        out_img[:h0, :w0] = img0
        out_img[:h1, w0:] = img1

    for pt in kps0:
        pt = np.round(pt).astype(np.int32)
        cv2.circle(out_img, tuple(pt), 1, kps_color, -1)

    for pt in kps1:
        pt = np.round(pt).astype(np.int32)
        pt = pt.copy()
        if vert:
            pt[1] += h0
        else:
            pt[0] += w0
        cv2.circle(out_img, tuple(pt), 1, kps_color, -1)

    if max_draw_line_num is not None and matches.shape[0]>max_draw_line_num:
        np.random.seed(6033)
        idxs=np.arange(matches.shape[0])
        np.random.shuffle(idxs)
        idxs=idxs[:max_draw_line_num]
        matches= matches[idxs]

        if colors is not None and (type(colors)==list or type(colors)==np.ndarray):
            colors=np.asarray(colors)
            colors= colors[idxs]

    for mi,m in enumerate(matches):
        pt = np.round(kps0[m[0]]).astype(np.int32)
        pr_pt = np.round(kps1[m[1]]).astype(np.int32)
        if vert:
            pr_pt[1] += h0
        else:
            pr_pt[0] += w0
        if colors is None:
            cv2.line(out_img, tuple(pt), tuple(pr_pt), (0, 255, 0), 1)
        elif type(colors)==list or type(colors)==np.ndarray:
            color=(int(c) for c in colors[mi])
            cv2.line(out_img, tuple(pt), tuple(pr_pt), tuple(color), 1)
        else:
            color=(int(c) for c in colors)
            cv2.line(out_img, tuple(pt), tuple(pr_pt), tuple(color), 1)

    return out_img

def draw_keypoints(img,kps,colors=None,radius=2):
    out_img=img.copy()
    for pi, pt in enumerate(kps):
        pt = np.round(pt).astype(np.int32)
        if colors is not None:
            color=[int(c) for c in colors[pi]]
            cv2.circle(out_img, tuple(pt), radius, color, -1, cv2.FILLED)
        else:
            cv2.circle(out_img, tuple(pt), radius, (0,255,0), -1)
    return out_img

def draw_epipolar_line(F, img0, img1, pt0, color):
    h1,w1=img1.shape[:2]
    hpt = np.asarray([pt0[0], pt0[1], 1], dtype=np.float32)[:, None]
    l = F @ hpt
    l = l[:, 0]
    a, b, c = l[0], l[1], l[2]
    pt1 = np.asarray([0, -c / b]).astype(np.int32)
    pt2 = np.asarray([w1, (-a * w1 - c) / b]).astype(np.int32)

    img0 = cv2.circle(img0, tuple(pt0.astype(np.int32)), 5, color, 2)
    img1 = cv2.line(img1, tuple(pt1), tuple(pt2), color, 2)
    return img0, img1

def draw_epipolar_lines(F, img0, img1,num=20):
    img0,img1=img0.copy(),img1.copy()
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape

    for k in range(num):
        color = np.random.randint(0, 255, [3], dtype=np.int32)
        color = [int(c) for c in color]
        pt = np.random.uniform(0, 1, 2)
        pt[0] *= w0
        pt[1] *= h0
        pt = pt.astype(np.int32)
        img0, img1 = draw_epipolar_line(F, img0, img1, pt, color)

    return img0, img1

def gen_color_map(error, clip_max=12.0, clip_min=2.0, cmap_name='viridis'):
    rectified_error=(error-clip_min)/(clip_max-clip_min)
    rectified_error[rectified_error<0]=0
    rectified_error[rectified_error>=1.0]=1.0
    viridis=cm.get_cmap(cmap_name,256)
    colors=[viridis(e) for e in rectified_error]
    return np.asarray(np.asarray(colors)[:,:3]*255,np.uint8)

def scale_float_image(image):
    max_val, min_val = np.max(image), np.min(image)
    image = (image - min_val) / (max_val - min_val) * 255
    return image.astype(np.uint8)

def concat_images(img0,img1,vert=False):
    if not vert:
        h0,h1=img0.shape[0],img1.shape[0],
        if h0<h1: img0=cv2.copyMakeBorder(img0,0,h1-h0,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        if h1<h0: img1=cv2.copyMakeBorder(img1,0,h0-h1,0,0,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=1)
    else:
        w0,w1=img0.shape[1],img1.shape[1]
        if w0<w1: img0=cv2.copyMakeBorder(img0,0,0,0,w1-w0,borderType=cv2.BORDER_CONSTANT,value=0)
        if w1<w0: img1=cv2.copyMakeBorder(img1,0,0,0,w0-w1,borderType=cv2.BORDER_CONSTANT,value=0)
        img = np.concatenate([img0, img1], axis=0)

    return img


def concat_images_list(*args,vert=False):
    if len(args)==1: return args[0]
    img_out=args[0]
    for img in args[1:]:
        img_out=concat_images(img_out,img,vert)
    return img_out


def get_colors_gt_pr(gt,pr=None):
    if pr is None:
        pr=np.ones_like(gt)
    colors=np.zeros([gt.shape[0],3],np.uint8)
    colors[gt & pr]=np.asarray([0,255,0])[None,:]     # tp
    colors[ (~gt) & pr]=np.asarray([255,0,0])[None,:] # fp
    colors[ gt & (~pr)]=np.asarray([0,0,255])[None,:] # fn
    return colors


def draw_hist(fn,vals,bins=100,hist_range=None,names=None):
    if type(vals)==list:
        val_num=len(vals)
        if hist_range is None:
            hist_range = (np.min(vals),np.max(vals))
        if names is None:
            names=[str(k) for k in range(val_num)]
        for k in range(val_num):
            plt.hist(vals[k], bins=bins, range=hist_range, alpha=0.5, label=names[k])
        plt.legend()
    else:
        if hist_range is None:
            hist_range = (np.min(vals),np.max(vals))
        plt.hist(vals,bins=bins,range=hist_range)

    plt.savefig(fn)
    plt.close()

def draw_pr_curve(fn,gt_sort):
    pos_num_all=np.sum(gt_sort)
    pos_nums=np.cumsum(gt_sort)
    sample_nums=np.arange(gt_sort.shape[0])+1
    precisions=pos_nums.astype(np.float64)/sample_nums
    recalls=pos_nums/pos_num_all

    precisions=precisions[np.arange(0,gt_sort.shape[0],gt_sort.shape[0]//40)]
    recalls=recalls[np.arange(0,gt_sort.shape[0],gt_sort.shape[0]//40)]
    plt.plot(recalls,precisions,'r-')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(fn)
    plt.close()

def draw_features_distribution(fn,feats,colors,ds_type='pca'):
    n,d=feats.shape
    if d>2:
        if ds_type=='pca':
            pca=PCA(2)
            feats=pca.fit_transform(feats)
        elif ds_type=='tsne':
            tsne=TSNE(2)
            feats=tsne.fit_transform(feats)
        elif ds_type=='pca-tsne':
            if d>50:
                tsne=PCA(50)
                feats=tsne.fit_transform(feats)
            tsne=TSNE(2,100.0)
            feats=tsne.fit_transform(feats)
        else:
            raise NotImplementedError

    colors=[np.array([c[0],c[1],c[2]],np.float64)/255.0 for c in colors]
    feats_min=np.min(feats,0,keepdims=True)
    feats_max=np.max(feats,0,keepdims=True)
    feats=(feats-(feats_min+feats_max)/2)*10/(feats_max-feats_min)
    plt.scatter(feats[:,0],feats[:,1],s=0.5,c=colors)
    plt.savefig(fn)
    plt.close()
    return feats

def draw_points(img,points):
    pts=np.round(points).astype(np.int32)
    h,w,_=img.shape
    pts[:,0]=np.clip(pts[:,0],a_min=0,a_max=w-1)
    pts[:,1]=np.clip(pts[:,1],a_min=0,a_max=h-1)
    img=img.copy()
    img[pts[:,1],pts[:,0]]=255
    # img[pts[:,1],pts[:,0]]+=np.asarray([127,0,0],np.uint8)[None,:]
    return img

def draw_bbox(img,bbox,color=None):
    if color is not None:
        color=[int(c) for c in color]
    else:
        color=(0,255,0)
    img=cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color)
    return img

def output_points(fn,pts,colors=None):
    with open(fn, 'w') as f:
        for pi, pt in enumerate(pts):
            f.write(f'{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} ')
            if colors is not None:
                f.write(f'{int(colors[pi,0])} {int(colors[pi,1])} {int(colors[pi,2])}')
            f.write('\n')

def compute_axis_points(pose):
    R=pose[:,:3] # 3,3
    t=pose[:,3:] # 3,1
    pts = np.concatenate([np.identity(3),np.zeros([3,1])],1) # 3,4
    pts = R.T @ (pts - t)
    colors = np.asarray([[255,0,0],[0,255,0,],[0,0,255],[0,0,0]],np.uint8)
    return pts.T, colors

def draw_epipolar_lines_func(img0,img1,Rt0,Rt1,K0,K1):
    Rt=compute_relative_transformation(Rt0,Rt1)
    F=compute_F(K0,K1,Rt[:,:3],Rt[:,3:])
    return concat_images_list(*draw_epipolar_lines(F,img0,img1))

def draw_axis(img, R, t, K, length=0.1, width=3, with_text=False, dist=None):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array (RGB format, not opencv BGR)
    :rotation_vec - euler rotations, numpy array of length 3,
                    use cv2.Rodrigues(R)[0] to convert from rotation matrix
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :length - factor to control the axis lengths
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    rotation_vec = cv2.Rodrigues(R)[0]
    img = img.astype(np.float32)
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = length * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)
    axis_points = axis_points.astype(np.int)

    if with_text:
        for pt,txt in zip(axis_points, "XYZO"):
            cv2.putText(img, txt, tuple(pt.ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[0].ravel()), (0, 0, 255), width)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), width)
    img = cv2.line(img, tuple(axis_points[3].ravel()), tuple(axis_points[2].ravel()), (255, 0, 0), width)
    return img

def draw_gripper(img, R, t, K, width, thickness=2, dist=None):
    rotation_vec = cv2.Rodrigues(R)[0]
    img = img.astype(np.float32)
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = np.float32([[0, -width/2, 0], [0, -width/2, 0.05], [0, width/2, 0], [0, width/2, 0.05], [0, 0, -0.05], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)
    axis_points = axis_points.astype(np.int)

    img = cv2.line(img, tuple(axis_points[0].ravel()), tuple(axis_points[2].ravel()), (0, 255, 0), thickness)
    img = cv2.line(img, tuple(axis_points[0].ravel()), tuple(axis_points[1].ravel()), (0, 255, 0), thickness)
    img = cv2.line(img, tuple(axis_points[2].ravel()), tuple(axis_points[3].ravel()), (0, 255, 0), thickness)

    img = cv2.line(img, tuple(axis_points[-2].ravel()), tuple(axis_points[-1].ravel()), (255, 0, 0), thickness)
    return img   


def draw_cube(img,  R, t, K, length=0.3, bias=[-0.15,-0.15,-0.], dist=None):
    rotation_vec = cv2.Rodrigues(R)[0]
    img = img.astype(np.float32)
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = length * np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                                    [0,0,1], [0,1,1], [1,1,1], [1,0,1]])
    points +=  np.float32(bias)
    imgpts, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(0,0,255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(255,0,0),3)
    return img

def draw_world_points(img, points, Rwc, twc, K, dist=None):
    rotation_vec = cv2.Rodrigues(Rwc)[0]
    img = img.astype(np.float32)
    dist = np.zeros(4, dtype=float) if dist is None else dist
    axis_points, _ = cv2.projectPoints(points, rotation_vec, twc, K, dist)
    img = draw_keypoints(img, axis_points.squeeze(1))

    return img

def extract_surface_points_from_volume(vol, rg, bound=(-1,1), color=(0,0,1), scale=0.3 / 40):
    assert len(vol.shape) == 3
    ind = np.transpose( ( (vol > rg[0]).astype(np.bool) & (vol < rg[1]).astype(np.bool) ).nonzero())
    print(ind.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ind)
    colors = np.array([color]).repeat(ind.shape[0],axis=0)
    
    if color == None:
            values = vol[ind[:,0], ind[:,1], ind[:,2]]
            a, b = bound
            m = (a + b) / 2
            r = np.where(values <= m, values - a, - values + b) 
            g = np.where(values <= m, 0, 1 - r)
            b = np.where(values <= m, 1 - r, 0) 
            colors = np.stack([r, g, b], axis=-1)
    else:
            colors = np.array([color]).repeat(ind.shape[0],axis=0)

    pcd.scale(scale, center=(0, 0, 0))
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def draw_volume_surface(logdir, vol, prefix='surface', rg=(-0.2,0.2)):    
    pcd = extract_surface_points_from_volume(vol, rg)

    o3d.io.write_point_cloud(f"{logdir}/{prefix}.ply", pcd)


def create_mesh_box(width, height, depth, dx=0, dy=0, dz=0):
    ''' Author: chenxi-wang
    Create box instance with mesh representation.
    '''
    box = o3d.geometry.TriangleMesh()
    vertices = np.array([[0,0,0],
                        [width,0,0],
                        [0,0,depth],
                        [width,0,depth],
                        [0,height,0],
                        [width,height,0],
                        [0,height,depth],
                        [width,height,depth]])
    vertices[:,0] += dx
    vertices[:,1] += dy
    vertices[:,2] += dz
    triangles = np.array([[4,7,5],[4,6,7],[0,2,4],[2,6,4],
                        [0,1,2],[1,3,2],[1,5,7],[1,7,3],
                        [2,3,7],[2,7,6],[0,4,1],[1,4,5]])
    box.vertices = o3d.utility.Vector3dVector(vertices)
    box.triangles = o3d.utility.Vector3iVector(triangles)
    return box

def draw_gripper_o3d(R, t, width, score=1, color=None):
    '''
    Author: chenxi-wang

    **Input:**

    - center: numpy array of (3,), target point as gripper center

    - R: numpy array of (3,3), rotation matrix of gripper

    - width: float, gripper width

    - score: float, grasp quality score

    **Output:**

    - open3d.geometry.TriangleMesh
    '''
    #x, y, z = center
    center = t
    #print(center)
    height = 0.004
    finger_width = 0.004
    tail_length = 0.04
    depth = 0.05
    depth_base = 0 # 0.02

    if color is not None:
        color_r, color_g, color_b = color
    else:
        color_r = score  # red for high score
        color_g = 0
        color_b = 1 - score  # blue for low score

    left = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    right = create_mesh_box(depth + depth_base + finger_width, finger_width, height)
    bottom = create_mesh_box(finger_width, width, height)
    tail = create_mesh_box(tail_length, finger_width, height)

    left_points = np.array(left.vertices)
    left_triangles = np.array(left.triangles)
    left_points[:, 0] -= depth_base + finger_width
    left_points[:, 1] -= width / 2 + finger_width
    left_points[:, 2] -= height / 2

    right_points = np.array(right.vertices)
    right_triangles = np.array(right.triangles) + 8
    right_points[:, 0] -= depth_base + finger_width
    right_points[:, 1] += width / 2
    right_points[:, 2] -= height / 2

    bottom_points = np.array(bottom.vertices)
    bottom_triangles = np.array(bottom.triangles) + 16
    bottom_points[:, 0] -= finger_width + depth_base
    bottom_points[:, 1] -= width / 2
    bottom_points[:, 2] -= height / 2

    tail_points = np.array(tail.vertices)
    tail_triangles = np.array(tail.triangles) + 24
    tail_points[:, 0] -= tail_length + finger_width + depth_base
    tail_points[:, 1] -= finger_width / 2
    tail_points[:, 2] -= height / 2

    vertices = np.concatenate([left_points, right_points, bottom_points, tail_points], axis=0)
    vertices = np.dot(R, vertices.T).T + center
    triangles = np.concatenate([left_triangles, right_triangles, bottom_triangles, tail_triangles], axis=0)
    colors = np.array([[color_r, color_g, color_b] for _ in range(len(vertices))])

    gripper = o3d.geometry.TriangleMesh()
    gripper.vertices = o3d.utility.Vector3dVector(vertices)
    gripper.triangles = o3d.utility.Vector3iVector(triangles)
    gripper.vertex_colors = o3d.utility.Vector3dVector(colors)
    return gripper

def transform_points(points, trans):
        '''
        Author: chenxi-wang

        **Input:**

        - points: numpy array of (N,3), point cloud

        - trans: numpy array of (4,4), transformation matrix

        **Output:**

        - numpy array of (N,3), transformed points.
        '''
        ones = np.ones([points.shape[0], 1], dtype=points.dtype)
        points_ = np.concatenate([points, ones], axis=-1)
        points_ = np.matmul(trans, points_.T).T
        return points_[:, :3]
