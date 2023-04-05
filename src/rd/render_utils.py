import os
import random
import bpy
import math
import numpy as np
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view
from rd.modify_material import set_modify_material, set_modify_raw_material, set_modify_table_material, set_modify_floor_material

# render parameter 
RENDERING_PATH = os.getcwd()
LIGHT_EMITTER_ENERGY = 5
LIGHT_ENV_MAP_ENERGY_IR = 0.035 
LIGHT_ENV_MAP_ENERGY_RGB = 1.0 
CYCLES_SAMPLE = 32
CAMERA_TYPE = "realsense"
NUM_FRAME_PER_SCENE = 24

# view point parameter
look_at_shift = np.array([0,0,0])
num_point_ver = 6
num_point_hor = 4
beta_range = (15*math.pi/180, 45*math.pi/180)
r = 0.5
tsdf2blender_coord_T_shift = np.array([-0.15, -0.15, -0.0503])
TABLE_CAD_MODEL_HEIGHT = 0.75

# material randomization mode (transparent, specular, mixed, raw)
my_material_randomize_mode = 'mixed'

# set depth sensor parameter
camera_width = 640
camera_height = 360
baseline_distance = 0.055

# set background parameter
background_size = 3.
background_position = (0., 0., 0.)
background_scale = (1., 1., 1.)


# set camera randomize paramater
start_point_range = ((0.5, 0.95), (-0.6, 0.6, -0.6, 0.6))
up_range = (-0.18, -0.18, -0.18, 0.18)
look_at_range = (background_position[0] - 0.05, background_position[0] + 0.05, 
                 background_position[1] - 0.05, background_position[1] + 0.05,
                 background_position[2] - 0.05, background_position[2] + 0.05)


g_syn_light_num_lowbound = 4
g_syn_light_num_highbound = 6
g_syn_light_dist_lowbound = 8
g_syn_light_dist_highbound = 12
g_syn_light_azimuth_degree_lowbound = 0
g_syn_light_azimuth_degree_highbound = 360
g_syn_light_elevation_degree_lowbound = 0
g_syn_light_elevation_degree_highbound = 90
g_syn_light_energy_mean = 3
g_syn_light_energy_std = 0.5
g_syn_light_environment_energy_lowbound = 0
g_syn_light_environment_energy_highbound = 1


g_shape_synset_name_pairs_all = {'02691156': 'aeroplane',
                                '02747177': 'ashtray',
                                '02773838': 'backpack',
                                '02801938': 'basket',
                                '02808440': 'tub',  # bathtub
                                '02818832': 'bed',
                                '02828884': 'bench',
                                '02834778': 'bicycle',
                                '02843684': 'mailbox', # missing in objectnet3d, birdhouse, use view distribution of mailbox
                                '02858304': 'boat',
                                '02871439': 'bookshelf',
                                '02876657': 'bottle',
                                '02880940': 'bowl', # missing in objectnet3d, bowl, use view distribution of plate
                                '02924116': 'bus',
                                '02933112': 'cabinet',
                                '02942699': 'camera',
                                '02946921': 'can',
                                '02954340': 'cap',
                                '02958343': 'car',
                                '02992529': 'cellphone',
                                '03001627': 'chair',
                                '03046257': 'clock',
                                '03085013': 'keyboard',
                                '03207941': 'dishwasher',
                                '03211117': 'tvmonitor',
                                '03261776': 'headphone',
                                '03325088': 'faucet',
                                '03337140': 'filing_cabinet',
                                '03467517': 'guitar',
                                '03513137': 'helmet',
                                '03593526': 'jar',
                                '03624134': 'knife',
                                '03636649': 'lamp',
                                '03642806': 'laptop',
                                '03691459': 'speaker',
                                '03710193': 'mailbox',
                                '03759954': 'microphone',
                                '03761084': 'microwave',
                                '03790512': 'motorbike',
                                '03797390': 'mug',  # missing in objectnet3d, mug, use view distribution of cup
                                '03928116': 'piano',
                                '03938244': 'pillow',
                                '03948459': 'rifle',  # missing in objectnet3d, pistol, use view distribution of rifle
                                '03991062': 'pot',
                                '04004475': 'printer',
                                '04074963': 'remote_control',
                                '04090263': 'rifle',
                                '04099429': 'road_pole',  # missing in objectnet3d, rocket, use view distribution of road_pole
                                '04225987': 'skateboard',
                                '04256520': 'sofa',
                                '04330267': 'stove',
                                '04379243': 'diningtable',  # use view distribution of dining_table
                                '04401088': 'telephone',
                                '04460130': 'road_pole',  # missing in objectnet3d, tower, use view distribution of road_pole
                                '04468005': 'train',
                                '04530566': 'washing_machine',
                                '04554684': 'dishwasher'}  # washer, use view distribution of dishwasher

g_synset_name_label_pairs = {#'aeroplane': 7,
                                #'bottle': 1,
                                #'bowl': 2,   
                                #'camera': 3,
                                #'can': 4,
                                #'car': 5,
                                #'mug': 6,    
                                'other': 0}   

material_class_instance_pairs = {'specular': ['metal', 'paintsp'],  # 'porcelain','plasticsp',
                                    'transparent': ['glass'],
                                    'diffuse': ['plastic','rubber','paper','leather','wood','clay','fabric'],
                                    'background': ['background']}


# material list
class_material_pairs = {'specular': ['other'],
                        'transparent': ['other'],
                        'diffuse': ['other']}

instance_material_except_pairs = {'metal': [],
                                    'porcelain': [],
                                    'plasticsp': [],
                                    'paintsp':[],

                                    'glass': [],#[8,9,18,19,20,24,25,26,27,28,29,30,31,32,34,43,59,72],
                                    
                                    'plastic': [],
                                    'rubber': [],     
                                    'leather': [],
                                    'wood':[],
                                    'paper':[],
                                    'fabric':[],
                                    'clay':[],   
                                    }
instance_material_include_pairs = {
                                    }

material_class_id_dict = {'raw': 0,
                        'diffuse': 1,
                        'transparent': 2,
                        'specular': 3}

material_type_id_dict = {'raw': 0,
                        'metal': 1,
                        'porcelain': 2,
                        'plasticsp': 3,
                        'paintsp':4,
                        'glass': 5, 
                        'plastic': 6,
                        'rubber': 7,     
                        'leather': 8,
                        'wood':9,
                        'paper':10,
                        'fabric':11,
                        'clay':12,               
                        }


def obj_centered_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180 * math.pi
    theta = float(azimuth_deg) / 180 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)

def quaternionFromYawPitchRoll(yaw, pitch, roll):
    c1 = math.cos(yaw / 2.0)
    c2 = math.cos(pitch / 2.0)
    c3 = math.cos(roll / 2.0)    
    s1 = math.sin(yaw / 2.0)
    s2 = math.sin(pitch / 2.0)
    s3 = math.sin(roll / 2.0)    
    q1 = c1 * c2 * c3 + s1 * s2 * s3
    q2 = c1 * c2 * s3 - s1 * s2 * c3
    q3 = c1 * s2 * c3 + s1 * c2 * s3
    q4 = s1 * c2 * c3 - c1 * s2 * s3
    return (q1, q2, q3, q4)

def camPosToQuaternion(cx, cy, cz):
    q1a = 0
    q1b = 0
    q1c = math.sqrt(2) / 2
    q1d = math.sqrt(2) / 2
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = cx / camDist
    cy = cy / camDist
    cz = cz / camDist    
    t = math.sqrt(cx * cx + cy * cy) 
    tx = cx / t
    ty = cy / t
    yaw = math.acos(ty)
    if tx > 0:
        yaw = 2 * math.pi - yaw
    pitch = 0
    tmp = min(max(tx*cx + ty*cy, -1),1)
    #roll = math.acos(tx * cx + ty * cy)
    roll = math.acos(tmp)
    if cz < 0:
        roll = -roll    
    print("%f %f %f" % (yaw, pitch, roll))
    q2a, q2b, q2c, q2d = quaternionFromYawPitchRoll(yaw, pitch, roll)    
    q1 = q1a * q2a - q1b * q2b - q1c * q2c - q1d * q2d
    q2 = q1b * q2a + q1a * q2b + q1d * q2c - q1c * q2d
    q3 = q1c * q2a - q1d * q2b + q1a * q2c + q1b * q2d
    q4 = q1d * q2a + q1c * q2b - q1b * q2c + q1a * q2d
    return (q1, q2, q3, q4)

def camRotQuaternion(cx, cy, cz, theta): 
    theta = theta / 180.0 * math.pi
    camDist = math.sqrt(cx * cx + cy * cy + cz * cz)
    cx = -cx / camDist
    cy = -cy / camDist
    cz = -cz / camDist
    q1 = math.cos(theta * 0.5)
    q2 = -cx * math.sin(theta * 0.5)
    q3 = -cy * math.sin(theta * 0.5)
    q4 = -cz * math.sin(theta * 0.5)
    return (q1, q2, q3, q4)

def quaternionProduct(qx, qy): 
    a = qx[0]
    b = qx[1]
    c = qx[2]
    d = qx[3]
    e = qy[0]
    f = qy[1]
    g = qy[2]
    h = qy[3]
    q1 = a * e - b * f - c * g - d * h
    q2 = a * f + b * e + c * h - d * g
    q3 = a * g - b * h + c * e + d * f
    q4 = a * h + b * g - c * f + d * e    
    return (q1, q2, q3, q4)

def quaternionToRotation(q):
    w, x, y, z = q
    r00 = 1 - 2 * y ** 2 - 2 * z ** 2
    r01 = 2 * x * y + 2 * w * z
    r02 = 2 * x * z - 2 * w * y

    r10 = 2 * x * y - 2 * w * z
    r11 = 1 - 2 * x ** 2 - 2 * z ** 2
    r12 = 2 * y * z + 2 * w * x

    r20 = 2 * x * z + 2 * w * y
    r21 = 2 * y * z - 2 * w * x
    r22 = 1 - 2 * x ** 2 - 2 * y ** 2
    r = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    return r

def quaternionToRotation_xyzw(q):
    x, y, z, w = q
    r00 = 1 - 2 * y ** 2 - 2 * z ** 2
    r01 = 2 * x * y + 2 * w * z
    r02 = 2 * x * z - 2 * w * y

    r10 = 2 * x * y - 2 * w * z
    r11 = 1 - 2 * x ** 2 - 2 * z ** 2
    r12 = 2 * y * z + 2 * w * x

    r20 = 2 * x * z + 2 * w * y
    r21 = 2 * y * z - 2 * w * x
    r22 = 1 - 2 * x ** 2 - 2 * y ** 2
    r = [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]
    return r

def quaternionFromRotMat(rotation_matrix):
    rotation_matrix = np.reshape(rotation_matrix, (1, 9))[0]
    w = math.sqrt(rotation_matrix[0]+rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    x = math.sqrt(rotation_matrix[0]-rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    y = math.sqrt(-rotation_matrix[0]+rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    z = math.sqrt(-rotation_matrix[0]-rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    a = [w,x,y,z]
    m = a.index(max(a))
    if m == 0:
        x = (rotation_matrix[7]-rotation_matrix[5])/(4*w)
        y = (rotation_matrix[2]-rotation_matrix[6])/(4*w)
        z = (rotation_matrix[3]-rotation_matrix[1])/(4*w)
    if m == 1:
        w = (rotation_matrix[7]-rotation_matrix[5])/(4*x)
        y = (rotation_matrix[1]+rotation_matrix[3])/(4*x)
        z = (rotation_matrix[6]+rotation_matrix[2])/(4*x)
    if m == 2:
        w = (rotation_matrix[2]-rotation_matrix[6])/(4*y)
        x = (rotation_matrix[1]+rotation_matrix[3])/(4*y)
        z = (rotation_matrix[5]+rotation_matrix[7])/(4*y)
    if m == 3:
        w = (rotation_matrix[3]-rotation_matrix[1])/(4*z)
        x = (rotation_matrix[6]+rotation_matrix[2])/(4*z)
        y = (rotation_matrix[5]+rotation_matrix[7])/(4*z)
    quaternion = (w,x,y,z)
    return quaternion

def quaternionFromRotMat_xyzw(rotation_matrix):
    rotation_matrix = np.reshape(rotation_matrix, (1, 9))[0]
    w = math.sqrt(rotation_matrix[0]+rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    x = math.sqrt(rotation_matrix[0]-rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    y = math.sqrt(-rotation_matrix[0]+rotation_matrix[4]-rotation_matrix[8]+1 + 1e-6)/2
    z = math.sqrt(-rotation_matrix[0]-rotation_matrix[4]+rotation_matrix[8]+1 + 1e-6)/2
    a = [x,y,z,w]
    m = a.index(max(a))
    if m == 0:
        x = (rotation_matrix[7]-rotation_matrix[5])/(4*w)
        y = (rotation_matrix[2]-rotation_matrix[6])/(4*w)
        z = (rotation_matrix[3]-rotation_matrix[1])/(4*w)
    if m == 1:
        w = (rotation_matrix[7]-rotation_matrix[5])/(4*x)
        y = (rotation_matrix[1]+rotation_matrix[3])/(4*x)
        z = (rotation_matrix[6]+rotation_matrix[2])/(4*x)
    if m == 2:
        w = (rotation_matrix[2]-rotation_matrix[6])/(4*y)
        x = (rotation_matrix[1]+rotation_matrix[3])/(4*y)
        z = (rotation_matrix[5]+rotation_matrix[7])/(4*y)
    if m == 3:
        w = (rotation_matrix[3]-rotation_matrix[1])/(4*z)
        x = (rotation_matrix[6]+rotation_matrix[2])/(4*z)
        y = (rotation_matrix[5]+rotation_matrix[7])/(4*z)
    quaternion = (x,y,z,w)
    return quaternion

def rotVector(q, vector_ori):
    r = quaternionToRotation(q)
    x_ori = vector_ori[0]
    y_ori = vector_ori[1]
    z_ori = vector_ori[2]
    x_rot = r[0][0] * x_ori + r[1][0] * y_ori + r[2][0] * z_ori
    y_rot = r[0][1] * x_ori + r[1][1] * y_ori + r[2][1] * z_ori
    z_rot = r[0][2] * x_ori + r[1][2] * y_ori + r[2][2] * z_ori
    return (x_rot, y_rot, z_rot)

def cameraLPosToCameraRPos(q_l, pos_l, baseline_dis):
    vector_camera_l_y = (1, 0, 0)
    vector_rot = rotVector(q_l, vector_camera_l_y)
    pos_r = (pos_l[0] + vector_rot[0] * baseline_dis,
             pos_l[1] + vector_rot[1] * baseline_dis,
             pos_l[2] + vector_rot[2] * baseline_dis)
    return pos_r

def getRTFromAToB(pointCloudA, pointCloudB):
    muA = np.mean(pointCloudA, axis=0)
    muB = np.mean(pointCloudB, axis=0)

    zeroMeanA = pointCloudA - muA
    zeroMeanB = pointCloudB - muB

    covMat = np.matmul(np.transpose(zeroMeanA), zeroMeanB)
    U, S, Vt = np.linalg.svd(covMat)
    R = np.matmul(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        print("[V]getRTFromAToB: Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T * U.T
    T = (-np.matmul(R, muA.T) + muB.T).reshape(3, 1)
    return R, T

def cameraPositionRandomize(start_point_range, look_at_range, up_range):
    r_range, vector_range = start_point_range
    r_min, r_max = r_range
    x_min, x_max, y_min, y_max = vector_range
    r = random.uniform(r_min, r_max)
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)
    z = math.sqrt(1 - x**2 - y**2)
    vector_camera_axis = np.array([x, y, z])

    x_min, x_max, y_min, y_max = up_range
    x = random.uniform(x_min, x_max)
    y = random.uniform(y_min, y_max)    
    z = math.sqrt(1 - x**2 - y**2)
    up = np.array([x, y, z])

    x_min, x_max, y_min, y_max, z_min, z_max = look_at_range
    look_at = np.array([random.uniform(x_min, x_max),
                        random.uniform(y_min, y_max),
                        random.uniform(z_min, z_max)])
    position = look_at + r * vector_camera_axis

    vectorZ = - (look_at - position)/np.linalg.norm(look_at - position)
    vectorX = np.cross(up, vectorZ)/np.linalg.norm(np.cross(up, vectorZ))
    vectorY = np.cross(vectorZ, vectorX)/np.linalg.norm(np.cross(vectorX, vectorZ))

    # points in camera coordinates
    pointSensor= np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])

    # points in world coordinates 
    pointWorld = np.array([position,
                            position + vectorX,
                            position + vectorY * 2,
                            position + vectorZ * 3])

    resR, resT = getRTFromAToB(pointSensor, pointWorld)
    resQ = quaternionFromRotMat(resR)
    return resQ, resT    


def genCameraPosition(look_at):
    quat_list = []
    rot_list = []
    trans_list = []
    position_list = []
    
    # alpha: 
    alpha = 0
    alpha_delta = (2 * math.pi) / num_point_ver
    for i in range(num_point_ver):
        alpha = alpha + alpha_delta
        flag_x = 1
        flag_y = 1
        alpha1 = alpha
        if alpha > math.pi/2 and alpha <= math.pi: 
            alpha1 = math.pi - alpha
            flag_x = -1
            flag_y = 1
        elif alpha > math.pi and alpha <= math.pi*(3/2):
            alpha1 = alpha - math.pi
            flag_x = -1
            flag_y = -1
        elif alpha > math.pi*(3/2):
            alpha1 = math.pi*2 - alpha
            flag_x = 1
            flag_y = -1
    
        beta = beta_range[0]
        beta_delta = (beta_range[1]-beta_range[0])/(num_point_hor-1)
        for j in range(num_point_hor):
            if j != 0:
                beta = beta + beta_delta

            x = flag_x * (r * math.sin(beta)) * math.cos(alpha1)
            y = flag_y * (r * math.sin(beta)) * math.sin(alpha1)
            z = r * math.cos(beta)
            position = np.array([x, y, z]) + look_at
            look_at = look_at
            up = np.array([0, 0, 1])

            vectorZ = - (look_at - position)/np.linalg.norm(look_at - position)
            vectorX = np.cross(up, vectorZ)/np.linalg.norm(np.cross(up, vectorZ))
            vectorY = np.cross(vectorZ, vectorX)/np.linalg.norm(np.cross(vectorX, vectorZ))

            # points in camera coordinates
            pointSensor= np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])

            # points in world coordinates 
            pointWorld = np.array([position,
                                   position + vectorX,
                                   position + vectorY * 2,
                                   position + vectorZ * 3])

            resR, resT = getRTFromAToB(pointSensor, pointWorld)
            resQ = quaternionFromRotMat(resR)

            quat_list.append(resQ)
            rot_list.append(resR)
            trans_list.append(resT)
            position_list.append(position)
    return quat_list, trans_list, rot_list 


def quanternion_mul(q1, q2):
    s1 = q1[0]
    v1 = np.array(q1[1:])
    s2 = q2[0]
    v2 = np.array(q2[1:])
    s = s1 * s2 - np.dot(v1, v2)
    v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
    return (s, v[0], v[1], v[2])

class BlenderRenderer(object):
    def __init__(self, viewport_size_x=640, viewport_size_y=360, DEVICE_LIST=None):
        '''
        viewport_size_x, viewport_size_y: rendering viewport resolution
        '''

        self.DEVICE_LIST = DEVICE_LIST

        # remove all objects, cameras and lights
        for obj in bpy.data.meshes:
            bpy.data.meshes.remove(obj)

        for cam in bpy.data.cameras:
            bpy.data.cameras.remove(cam)

        for light in bpy.data.lights:
            bpy.data.lights.remove(light)

        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        render_context = bpy.context.scene.render

        # add left camera
        camera_l_data = bpy.data.cameras.new(name="camera_l")
        camera_l_object = bpy.data.objects.new(name="camera_l", object_data=camera_l_data)
        bpy.context.collection.objects.link(camera_l_object)

        # add right camera
        camera_r_data = bpy.data.cameras.new(name="camera_r")
        camera_r_object = bpy.data.objects.new(name="camera_r", object_data=camera_r_data)
        bpy.context.collection.objects.link(camera_r_object)

        camera_l = bpy.data.objects["camera_l"]
        camera_r = bpy.data.objects["camera_r"]

        # set the camera postion and orientation so that it is in
        # the front of the object
        camera_l.location = (1, 0, 0)
        camera_r.location = (1, 0, 0)

        # add emitter light
        light_emitter_data = bpy.data.lights.new(name="light_emitter", type='SPOT')
        light_emitter_object = bpy.data.objects.new(name="light_emitter", object_data=light_emitter_data)
        bpy.context.collection.objects.link(light_emitter_object)

        light_emitter = bpy.data.objects["light_emitter"]
        light_emitter.location = (1, 0, 0)
        light_emitter.data.energy = LIGHT_EMITTER_ENERGY

        # render setting
        render_context.resolution_percentage = 100
        self.render_context = render_context

        self.camera_l = camera_l
        self.camera_r = camera_r

        self.light_emitter = light_emitter

        self.model_loaded = False
        self.background_added = None

        self.render_context.resolution_x = viewport_size_x
        self.render_context.resolution_y = viewport_size_y

        self.my_material = {}
        self.render_mode = 'IR'

        # output setting 
        self.render_context.image_settings.file_format = 'PNG'
        self.render_context.image_settings.compression = 0
        self.render_context.image_settings.color_mode = 'BW'
        self.render_context.image_settings.color_depth = '8'

        # cycles setting
        self.render_context.engine = 'CYCLES'
        bpy.context.scene.cycles.progressive = 'BRANCHED_PATH'
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'NLM'
        bpy.context.scene.cycles.film_exposure = 0.5

        # self.render_context.use_antialiasing = False
        ##########
        bpy.context.scene.view_layers["View Layer"].use_sky = True
        ##########

        # switch on nodes
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links
  
        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)
  
        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')

        # create output node
        self.fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        self.fileOutput.base_path = "./new_data/0000"
        self.fileOutput.format.file_format = 'OPEN_EXR'
        self.fileOutput.format.color_depth= '32'
        self.fileOutput.file_slots[0].path = 'depth#'
        links.new(rl.outputs[2], self.fileOutput.inputs[0])

        # depth sensor pattern
        self.pattern = []
        # environment map
        self.env_map = []
        self.realtable_img_list = []
        self.realfloor_img_list = []
        self.obj_texture_img_list = []

        self.src_energy_for_rgb_render = 0

    def loadImages(self, pattern_path, env_map_path, real_table_image_root_path, real_floor_image_root_path, obj_texture_image_root_path, obj_texture_image_idxfile, check_seen_scene):
        # load pattern image
        self.pattern = bpy.data.images.load(filepath=pattern_path)
        if check_seen_scene:
            env_map_path_list = os.listdir(env_map_path)
            real_table_image_root_path_list = os.listdir(real_table_image_root_path)
            real_floor_image_root_path_list = os.listdir(real_floor_image_root_path)
        else:
            env_map_path_list = sorted(os.listdir(env_map_path))
            real_table_image_root_path_list = sorted(os.listdir(real_table_image_root_path))
            real_floor_image_root_path_list = sorted(os.listdir(real_floor_image_root_path))
        # load env map
        for item in env_map_path_list:
            if item.split('.')[-1] == 'hdr':
                self.env_map.append(bpy.data.images.load(filepath=os.path.join(env_map_path, item)))
        # load real table images
        for item in real_table_image_root_path_list:
            if item.split('.')[-1] == 'jpg':
                self.realtable_img_list.append(bpy.data.images.load(filepath=os.path.join(real_table_image_root_path, item)))
        # load real floor images
        for item in real_floor_image_root_path_list:
            if item.split('.')[-1] == 'jpg':
                self.realfloor_img_list.append(bpy.data.images.load(filepath=os.path.join(real_floor_image_root_path, item)))
        # load obj texture images
        f_teximg_idx = open(os.path.join(obj_texture_image_root_path, obj_texture_image_idxfile),"r")
        lines = f_teximg_idx.readlines() 
        for item in lines:
            item = item[:-1]   
            self.obj_texture_img_list.append(bpy.data.images.load(filepath=os.path.join(obj_texture_image_root_path, "images", item)))


    def addEnvMap(self):
        # Get the environment node tree of the current scene
        node_tree = bpy.context.scene.world.node_tree
        tree_nodes = node_tree.nodes

        # Clear all nodes
        tree_nodes.clear()

        # Add Background node
        node_background = tree_nodes.new(type='ShaderNodeBackground')

        # Add Environment Texture node
        node_environment = tree_nodes.new('ShaderNodeTexEnvironment')
        # Load and assign the image to the node property
        # node_environment.image = bpy.data.images.load("/Users/zhangjiyao/Desktop/test_addon/envmap_lib/autoshop_01_1k.hdr") # Relative path
        node_environment.location = -300,0

        node_tex_coord = tree_nodes.new(type='ShaderNodeTexCoord')
        node_tex_coord.location = -700,0

        node_mapping = tree_nodes.new(type='ShaderNodeMapping')
        node_mapping.location = -500,0

        # Add Output node
        node_output = tree_nodes.new(type='ShaderNodeOutputWorld')   
        node_output.location = 200,0

        # Link all nodes
        links = node_tree.links
        links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
        links.new(node_background.outputs["Background"], node_output.inputs["Surface"])
        links.new(node_tex_coord.outputs["Generated"], node_mapping.inputs["Vector"])
        links.new(node_mapping.outputs["Vector"], node_environment.inputs["Vector"])

        #### bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 1.0
        random_energy = random.uniform(LIGHT_ENV_MAP_ENERGY_RGB * 0.8, LIGHT_ENV_MAP_ENERGY_RGB * 1.2)
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = random_energy
        ####


    def setEnvMap(self, env_map_id, rotation_elur_z):
        # Get the environment node tree of the current scene
        node_tree = bpy.context.scene.world.node_tree

        # Get Environment Texture node
        node_environment = node_tree.nodes['Environment Texture']
        # Load and assign the image to the node property
        node_environment.image = self.env_map[env_map_id]

        node_mapping = node_tree.nodes['Mapping']
        node_mapping.inputs[2].default_value[2] = rotation_elur_z


    def addMaskMaterial(self, num=20):
        background_material_name_list = ["mask_background", "mask_table", "mask_tableplane"]
        for material_name in background_material_name_list:
            material_class = (bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))         # test if material exists, if it does not exist, create it:

            # enable 'Use nodes'
            material_class.use_nodes = True
            node_tree = material_class.node_tree

            # remove default nodes
            material_class.node_tree.nodes.clear()

            # add new nodes  
            node_1 = node_tree.nodes.new('ShaderNodeOutputMaterial')
            node_2= node_tree.nodes.new('ShaderNodeBrightContrast')

            # link nodes
            node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
            node_2.inputs[0].default_value = (1, 1, 1, 1)
            self.my_material[material_name] =  material_class


        for i in range(num):
            class_name = str(i + 1)
            # set the material of background    
            material_name = "mask_" + class_name

            # test if material exists
            # if it does not exist, create it:
            material_class = (bpy.data.materials.get(material_name) or 
                bpy.data.materials.new(material_name))

            # enable 'Use nodes'
            material_class.use_nodes = True
            node_tree = material_class.node_tree

            # remove default nodes
            material_class.node_tree.nodes.clear()

            # add new nodes  
            node_1 = node_tree.nodes.new('ShaderNodeOutputMaterial')
            node_2= node_tree.nodes.new('ShaderNodeBrightContrast')

            # link nodes
            node_tree.links.new(node_1.inputs[0], node_2.outputs[0])

            if class_name.split('_')[0] == 'background' or class_name.split('_')[0] == 'table' or class_name.split('_')[0] == 'tableplane':
                node_2.inputs[0].default_value = (1, 1, 1, 1)
            else:
                node_2.inputs[0].default_value = ((i + 1)/255., 0., 0., 1)

            self.my_material[material_name] =  material_class


    def addNOCSMaterial(self):
        material_name = 'coord_color'
        mat = (bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))

        mat.use_nodes = True
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        nodes.clear()        

        links = node_tree.links
        links.clear()

        vcol_R = nodes.new(type="ShaderNodeVertexColor")
        vcol_R.layer_name = "Col_R" # the vertex color layer name
        vcol_G = nodes.new(type="ShaderNodeVertexColor")
        vcol_G.layer_name = "Col_G" # the vertex color layer name
        vcol_B = nodes.new(type="ShaderNodeVertexColor")
        vcol_B.layer_name = "Col_B" # the vertex color layer name

        node_Output = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_Emission = node_tree.nodes.new('ShaderNodeEmission')
        node_LightPath = node_tree.nodes.new('ShaderNodeLightPath')
        node_Mix = node_tree.nodes.new('ShaderNodeMixShader')
        node_Combine = node_tree.nodes.new(type="ShaderNodeCombineRGB")


        # make links
        node_tree.links.new(vcol_R.outputs[1], node_Combine.inputs[0])
        node_tree.links.new(vcol_G.outputs[1], node_Combine.inputs[1])
        node_tree.links.new(vcol_B.outputs[1], node_Combine.inputs[2])
        node_tree.links.new(node_Combine.outputs[0], node_Emission.inputs[0])

        node_tree.links.new(node_LightPath.outputs[0], node_Mix.inputs[0])
        node_tree.links.new(node_Emission.outputs[0], node_Mix.inputs[2])
        node_tree.links.new(node_Mix.outputs[0], node_Output.inputs[0])

        self.my_material[material_name] = mat


    def addNormalMaterial(self):
        material_name = 'normal'
        mat = (bpy.data.materials.get(material_name) or bpy.data.materials.new(material_name))
        mat.use_nodes = True
        node_tree = mat.node_tree
        nodes = node_tree.nodes
        nodes.clear()
            
        links = node_tree.links
        links.clear()
            
        # Nodes :
        new_node = nodes.new(type='ShaderNodeMath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (151.59744262695312, 854.5482177734375)
        new_node.name = 'Math'
        new_node.operation = 'MULTIPLY'
        new_node.select = False
        new_node.use_clamp = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5
        new_node.inputs[1].default_value = 1.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeLightPath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (602.9912719726562, 1046.660888671875)
        new_node.name = 'Light Path'
        new_node.select = False
        new_node.width = 140.0
        new_node.outputs[0].default_value = 0.0
        new_node.outputs[1].default_value = 0.0
        new_node.outputs[2].default_value = 0.0
        new_node.outputs[3].default_value = 0.0
        new_node.outputs[4].default_value = 0.0
        new_node.outputs[5].default_value = 0.0
        new_node.outputs[6].default_value = 0.0
        new_node.outputs[7].default_value = 0.0
        new_node.outputs[8].default_value = 0.0
        new_node.outputs[9].default_value = 0.0
        new_node.outputs[10].default_value = 0.0
        new_node.outputs[11].default_value = 0.0
        new_node.outputs[12].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeOutputMaterial')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.is_active_output = True
        new_node.location = (1168.93017578125, 701.84033203125)
        new_node.name = 'Material Output'
        new_node.select = False
        new_node.target = 'ALL'
        new_node.width = 140.0
        new_node.inputs[2].default_value = [0.0, 0.0, 0.0]

        new_node = nodes.new(type='ShaderNodeBsdfTransparent')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (731.72900390625, 721.4832763671875)
        new_node.name = 'Transparent BSDF'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = [1.0, 1.0, 1.0, 1.0]

        new_node = nodes.new(type='ShaderNodeCombineXYZ')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (594.4229736328125, 602.9271240234375)
        new_node.name = 'Combine XYZ'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.0
        new_node.inputs[1].default_value = 0.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = [0.0, 0.0, 0.0]

        new_node = nodes.new(type='ShaderNodeMixShader')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (992.7239990234375, 707.2142333984375)
        new_node.name = 'Mix Shader'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5

        new_node = nodes.new(type='ShaderNodeEmission')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (774.0802612304688, 608.2547607421875)
        new_node.name = 'Emission'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = [1.0, 1.0, 1.0, 1.0]
        new_node.inputs[1].default_value = 1.0

        new_node = nodes.new(type='ShaderNodeSeparateXYZ')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (-130.12167358398438, 558.1497802734375)
        new_node.name = 'Separate XYZ'
        new_node.select = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[0].default_value = 0.0
        new_node.outputs[1].default_value = 0.0
        new_node.outputs[2].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeMath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (162.43240356445312, 618.8094482421875)
        new_node.name = 'Math.002'
        new_node.operation = 'MULTIPLY'
        new_node.select = False
        new_node.use_clamp = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5
        new_node.inputs[1].default_value = 1.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeMath')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (126.8158187866211, 364.5539855957031)
        new_node.name = 'Math.001'
        new_node.operation = 'MULTIPLY'
        new_node.select = False
        new_node.use_clamp = False
        new_node.width = 140.0
        new_node.inputs[0].default_value = 0.5
        new_node.inputs[1].default_value = -1.0
        new_node.inputs[2].default_value = 0.0
        new_node.outputs[0].default_value = 0.0

        new_node = nodes.new(type='ShaderNodeVectorTransform')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.convert_from = 'WORLD'
        new_node.convert_to = 'CAMERA'
        new_node.location = (-397.0209045410156, 594.7037353515625)
        new_node.name = 'Vector Transform'
        new_node.select = False
        new_node.vector_type = 'VECTOR'
        new_node.width = 140.0
        new_node.inputs[0].default_value = [0.5, 0.5, 0.5]
        new_node.outputs[0].default_value = [0.0, 0.0, 0.0]

        new_node = nodes.new(type='ShaderNodeNewGeometry')
        new_node.active_preview = False
        new_node.color = (0.6079999804496765, 0.6079999804496765, 0.6079999804496765)
        new_node.location = (-651.8067016601562, 593.0455932617188)
        new_node.name = 'Geometry'
        new_node.width = 140.0
        new_node.outputs[0].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[1].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[2].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[3].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[4].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[5].default_value = [0.0, 0.0, 0.0]
        new_node.outputs[6].default_value = 0.0
        new_node.outputs[7].default_value = 0.0
        new_node.outputs[8].default_value = 0.0

        # Links :

        links.new(nodes["Light Path"].outputs[0], nodes["Mix Shader"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[0], nodes["Math"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[1], nodes["Math.002"].inputs[0])    
        links.new(nodes["Separate XYZ"].outputs[2], nodes["Math.001"].inputs[0])    
        links.new(nodes["Vector Transform"].outputs[0], nodes["Separate XYZ"].inputs[0])    
        links.new(nodes["Combine XYZ"].outputs[0], nodes["Emission"].inputs[0])    
        links.new(nodes["Math"].outputs[0], nodes["Combine XYZ"].inputs[0])    
        links.new(nodes["Math.002"].outputs[0], nodes["Combine XYZ"].inputs[1])    
        links.new(nodes["Math.001"].outputs[0], nodes["Combine XYZ"].inputs[2])    
        links.new(nodes["Transparent BSDF"].outputs[0], nodes["Mix Shader"].inputs[1])    
        links.new(nodes["Emission"].outputs[0], nodes["Mix Shader"].inputs[2])    
        links.new(nodes["Mix Shader"].outputs[0], nodes["Material Output"].inputs[0])    
        links.new(nodes["Geometry"].outputs[1], nodes["Vector Transform"].inputs[0])    

        self.my_material[material_name] = mat

    def addMaterialLib(self, material_class_instance_pairs):
        for mat in bpy.data.materials:
            name = mat.name
            name_class = str(name.split('_')[0])
            if name_class != 'Dots Stroke' and name_class != 'default':   
                if name_class not in self.my_material:
                    self.my_material[name_class] = [mat]
                else:
                    self.my_material[name_class].append(mat)    # e.g. self.my_material['metal'] = [.....]
    ###

    def setCamera(self, quaternion, translation, fov, baseline_distance):
        self.camera_l.data.angle = fov
        self.camera_r.data.angle = self.camera_l.data.angle
        cx = translation[0]
        cy = translation[1]
        cz = translation[2]

        self.camera_l.location[0] = cx
        self.camera_l.location[1] = cy 
        self.camera_l.location[2] = cz

        self.camera_l.rotation_mode = 'QUATERNION'
        self.camera_l.rotation_quaternion[0] = quaternion[0]
        self.camera_l.rotation_quaternion[1] = quaternion[1]
        self.camera_l.rotation_quaternion[2] = quaternion[2]
        self.camera_l.rotation_quaternion[3] = quaternion[3]

        self.camera_r.rotation_mode = 'QUATERNION'
        self.camera_r.rotation_quaternion[0] = quaternion[0]
        self.camera_r.rotation_quaternion[1] = quaternion[1]
        self.camera_r.rotation_quaternion[2] = quaternion[2]
        self.camera_r.rotation_quaternion[3] = quaternion[3]
        cx, cy, cz = cameraLPosToCameraRPos(quaternion, (cx, cy, cz), baseline_distance)
        self.camera_r.location[0] = cx
        self.camera_r.location[1] = cy 
        self.camera_r.location[2] = cz


    def setLighting(self):
        # emitter        
        #self.light_emitter.location = self.camera_r.location
        self.light_emitter.location = self.camera_l.location + 0.51 * (self.camera_r.location - self.camera_l.location)
        self.light_emitter.rotation_mode = 'QUATERNION'
        self.light_emitter.rotation_quaternion = self.camera_r.rotation_quaternion

        # emitter setting
        bpy.context.view_layer.objects.active = None
        # bpy.ops.object.select_all(action="DESELECT")
        self.render_context.engine = 'CYCLES'
        self.light_emitter.select_set(True)
        self.light_emitter.data.use_nodes = True
        self.light_emitter.data.type = "POINT"
        self.light_emitter.data.shadow_soft_size = 0.001
        random_energy = random.uniform(LIGHT_EMITTER_ENERGY * 0.9, LIGHT_EMITTER_ENERGY * 1.1)
        self.light_emitter.data.energy = random_energy

        # remove default node
        light_emitter = bpy.data.objects["light_emitter"].data
        light_emitter.node_tree.nodes.clear()

        # add new nodes
        light_output = light_emitter.node_tree.nodes.new("ShaderNodeOutputLight")
        node_1 = light_emitter.node_tree.nodes.new("ShaderNodeEmission")
        node_2 = light_emitter.node_tree.nodes.new("ShaderNodeTexImage")
        node_3 = light_emitter.node_tree.nodes.new("ShaderNodeMapping")
        node_4 = light_emitter.node_tree.nodes.new("ShaderNodeVectorMath")
        node_5 = light_emitter.node_tree.nodes.new("ShaderNodeSeparateXYZ")
        node_6 = light_emitter.node_tree.nodes.new("ShaderNodeTexCoord")

        # link nodes
        light_emitter.node_tree.links.new(light_output.inputs[0], node_1.outputs[0])
        light_emitter.node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
        light_emitter.node_tree.links.new(node_2.inputs[0], node_3.outputs[0])
        light_emitter.node_tree.links.new(node_3.inputs[0], node_4.outputs[0])
        light_emitter.node_tree.links.new(node_4.inputs[0], node_6.outputs[1])
        light_emitter.node_tree.links.new(node_4.inputs[1], node_5.outputs[2])
        light_emitter.node_tree.links.new(node_5.inputs[0], node_6.outputs[1])

        # set parameter of nodes
        node_1.inputs[1].default_value = 1.0        # scale
        node_2.extension = 'CLIP'
        # node_2.interpolation = 'Cubic'

        node_3.inputs[1].default_value[0] = 0.5
        node_3.inputs[1].default_value[1] = 0.5
        node_3.inputs[1].default_value[2] = 0
        node_3.inputs[2].default_value[0] = 0
        node_3.inputs[2].default_value[1] = 0
        node_3.inputs[2].default_value[2] = 0.05

        # scale of pattern
        node_3.inputs[3].default_value[0] = 0.6
        node_3.inputs[3].default_value[1] = 0.85
        node_3.inputs[3].default_value[2] = 0
        node_4.operation = 'DIVIDE'

        # pattern path
        node_2.image = self.pattern


    def lightModeSelect(self, light_mode):
        if light_mode == "RGB":
            self.light_emitter.hide_render = True
            ###
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = self.src_energy_for_rgb_render

        elif light_mode == "IR":
            self.light_emitter.hide_render = False
            # set the environment map energy
            random_energy = random.uniform(LIGHT_ENV_MAP_ENERGY_IR * 0.8, LIGHT_ENV_MAP_ENERGY_IR * 1.2)
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = random_energy
        
        elif light_mode == "Mask" or light_mode == "NOCS" or light_mode == "Normal":
            self.light_emitter.hide_render = True
            bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 0
        else:
            raise NotImplementedError   


    def outputModeSelect(self, output_mode):
        if output_mode == "RGB":
            self.render_context.image_settings.file_format = 'PNG'
            self.render_context.image_settings.compression = 0
            self.render_context.image_settings.color_mode = 'RGB'
            self.render_context.image_settings.color_depth = '8'
            bpy.context.scene.view_settings.view_transform = 'Filmic'
            bpy.context.scene.render.filter_size = 1.5
            self.render_context.resolution_x = 640 ### 1280
            self.render_context.resolution_y = 360 ### 720
        elif output_mode == "IR":
            self.render_context.image_settings.file_format = 'PNG'
            self.render_context.image_settings.compression = 0
            self.render_context.image_settings.color_mode = 'BW'
            self.render_context.image_settings.color_depth = '8'
            bpy.context.scene.view_settings.view_transform = 'Filmic'
            bpy.context.scene.render.filter_size = 1.5
            self.render_context.resolution_x = 640 ### 1280
            self.render_context.resolution_y = 360 ### 720
        elif output_mode == "Mask":
            self.render_context.image_settings.file_format = 'OPEN_EXR'
            self.render_context.image_settings.color_mode = 'RGB'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.filter_size = 0
            self.render_context.resolution_x = 640
            self.render_context.resolution_y = 360
        elif output_mode == "NOCS":
            # self.render_context.image_settings.file_format = 'OPEN_EXR'
            self.render_context.image_settings.file_format = 'PNG'            
            self.render_context.image_settings.color_mode = 'RGB'
            self.render_context.image_settings.color_depth = '8'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.filter_size = 0
            self.render_context.resolution_x = 640
            self.render_context.resolution_y = 360
        elif output_mode == "Normal":
            self.render_context.image_settings.file_format = 'OPEN_EXR'
            self.render_context.image_settings.color_mode = 'RGB'
            bpy.context.scene.view_settings.view_transform = 'Raw'
            bpy.context.scene.render.filter_size = 1.5
            self.render_context.resolution_x = 640
            self.render_context.resolution_y = 360
        else:
            raise NotImplementedError

    def renderEngineSelect(self, engine_mode):

        if engine_mode == "CYCLES":
            self.render_context.engine = 'CYCLES'
            bpy.context.scene.cycles.progressive = 'BRANCHED_PATH'
            bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.denoiser = 'NLM'
            bpy.context.scene.cycles.film_exposure = 1.0
            bpy.context.scene.cycles.aa_samples = CYCLES_SAMPLE

            ## Set the device_type
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA" # or "OPENCL"

            ## get_devices() to let Blender detects GPU device
            cuda_devices, _ = bpy.context.preferences.addons["cycles"].preferences.get_devices()
            #print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
            for d in bpy.context.preferences.addons["cycles"].preferences.devices:
                d["use"] = 1 # Using all devices, include GPU and CPU
                #print(d["name"], d["use"])
            device_list = self.DEVICE_LIST
            activated_gpus = []
            for i, device in enumerate(cuda_devices):
                if (i in device_list):
                    device.use = True
                    activated_gpus.append(device.name)
                else:
                    device.use = False


        elif engine_mode == "EEVEE":
            bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        else:
            print("Not support the mode!")    


    def addBackground(self, size, position, scale, default_background_texture_path):
        # set the material of background    
        material_name = "default_background"

        # test if material exists
        # if it does not exist, create it:
        material_background = (bpy.data.materials.get(material_name) or 
            bpy.data.materials.new(material_name))

        # enable 'Use nodes'
        material_background.use_nodes = True
        node_tree = material_background.node_tree

        # remove default nodes
        material_background.node_tree.nodes.clear()

        # add new nodes  
        node_1 = node_tree.nodes.new('ShaderNodeOutputMaterial')
        node_2 = node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        node_3 = node_tree.nodes.new('ShaderNodeTexImage')

        # link nodes
        node_tree.links.new(node_1.inputs[0], node_2.outputs[0])
        node_tree.links.new(node_2.inputs[0], node_3.outputs[0])

        # add texture image
        node_3.image = bpy.data.images.load(filepath=default_background_texture_path)
        self.my_material['default_background'] = material_background

        # add background plane
        for i in range(-2, 3, 1):
            for j in range(-2, 3, 1):
                position_i_j = (i * size + position[0], j * size + position[1], position[2] - TABLE_CAD_MODEL_HEIGHT)
                bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=position_i_j, scale=scale)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type = 'PASSIVE'
                bpy.context.object.rigid_body.collision_shape = 'BOX'
        for i in range(-2, 3, 1):
            for j in [-2, 2]:
                position_i_j = (i * size + position[0], j * size + position[1], position[2] - 0.25)# - TABLE_CAD_MODEL_HEIGHT)
                rotation_elur = (math.pi / 2., 0., 0.)
                bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=position_i_j, rotation = rotation_elur)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type = 'PASSIVE'
                bpy.context.object.rigid_body.collision_shape = 'BOX'    
        for j in range(-2, 3, 1):
            for i in [-2, 2]:
                position_i_j = (i * size + position[0], j * size + position[1], position[2] - 0.25)# - TABLE_CAD_MODEL_HEIGHT)
                rotation_elur = (0, math.pi / 2, 0)
                bpy.ops.mesh.primitive_plane_add(size=size, enter_editmode=False, align='WORLD', location=position_i_j, rotation = rotation_elur)
                bpy.ops.rigidbody.object_add()
                bpy.context.object.rigid_body.type = 'PASSIVE'
                bpy.context.object.rigid_body.collision_shape = 'BOX'        
        count = 0
        for obj in bpy.data.objects:
            if obj.type == "MESH":
                obj.name = "background_" + str(count)
                obj.data.name = "background_" + str(count)
                obj.active_material = material_background
                count += 1

        self.background_added = True


    def clearModel(self):
        '''
        # delete all meshes
        for item in bpy.data.meshes:
            bpy.data.meshes.remove(item)
        for item in bpy.data.materials:
            bpy.data.materials.remove(item)
        '''

        # remove all objects except background
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                bpy.data.meshes.remove(obj.data)
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                bpy.data.objects.remove(obj, do_unlink=True)

        # remove all default material
        for mat in bpy.data.materials:
            name = mat.name.split('.')
            if name[0] == 'Material':
                bpy.data.materials.remove(mat)


    def loadModel(self, file_path):
        self.model_loaded = True
        try:
            if file_path.endswith('obj'):
                bpy.ops.import_scene.obj(filepath=file_path)
            elif file_path.endswith('3ds'):
                bpy.ops.import_scene.autodesk_3ds(filepath=file_path)
            elif file_path.endswith('dae'):
                # Must install OpenCollada. Please read README.md
                bpy.ops.wm.collada_import(filepath=file_path)
            else:
                self.model_loaded = False
                raise Exception("Loading failed: %s" % (file_path))
        except Exception:
            self.model_loaded = False


    def render(self, image_name="tmp", image_path=RENDERING_PATH):
        # Render the object
        if not self.model_loaded:
            print("[W]render: Model not loaded.")
            return      

        if self.render_mode == "IR":
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("IR")
            self.outputModeSelect("IR")
            self.renderEngineSelect("CYCLES")

        elif self.render_mode == 'RGB':
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("RGB")
            self.outputModeSelect("RGB")
            self.renderEngineSelect("CYCLES")

        elif self.render_mode == "Mask":
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("Mask")
            self.outputModeSelect("Mask")
            # self.renderEngineSelect("EEVEE")
            self.renderEngineSelect("CYCLES")
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.aa_samples = 1

        elif self.render_mode == "NOCS":
            bpy.context.scene.use_nodes = False
            # set light and render mode
            self.lightModeSelect("NOCS")
            self.outputModeSelect("NOCS")
            # self.renderEngineSelect("EEVEE")
            self.renderEngineSelect("CYCLES")
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.aa_samples = 1

        elif self.render_mode == "Normal":
            bpy.context.scene.use_nodes = True
            self.fileOutput.base_path = image_path.replace("normal","depth")
            self.fileOutput.file_slots[0].path = image_name[:4]+"_#"# + 'depth_#'

            # set light and render mode
            self.lightModeSelect("Normal")
            self.outputModeSelect("Normal")
            # self.renderEngineSelect("EEVEE")
            self.renderEngineSelect("CYCLES")
            bpy.context.scene.cycles.use_denoising = False
            bpy.context.scene.cycles.aa_samples = 32

        else:
            print("[W]render: The render mode is not supported")
            return 

        bpy.context.scene.render.filepath = os.path.join(image_path, image_name)
        bpy.ops.render.render(write_still=True)  # save straight to file


    def set_material_randomize_mode(self, class_material_pairs, mat_randomize_mode, instance, material_type_in_mixed_mode):
        if mat_randomize_mode in ['mixed','diffuse','transparent','specular_tex','specular_texmix','specular_and_transparent']:
            if material_type_in_mixed_mode == 'raw':
                print("[V]set_material_randomize_mode", instance.name, 'material type: raw')
                set_modify_raw_material(instance)
            else:
                material = random.sample(self.my_material[material_type_in_mixed_mode], 1)[0]
                print("[V]set_material_randomize_mode", instance.name, 'material type: ', material_type_in_mixed_mode)
                ## graspnet
                set_modify_material(instance, material, self.obj_texture_img_list, mat_randomize_mode=mat_randomize_mode)
        elif mat_randomize_mode == 'specular':
            material = random.sample(self.my_material[material_type_in_mixed_mode], 1)[0]
            print("[V]set_material_randomize_mode", instance.name, 'material type: ', material_type_in_mixed_mode)
            set_modify_material(instance, material, self.obj_texture_img_list, mat_randomize_mode=mat_randomize_mode,
                                is_transfer=False)
        else:
            raise NotImplementedError("No such mat_randomize_mode!")


    def get_instance_pose(self):
        instance_pose = {}
        bpy.context.view_layer.update()
        cam = self.camera_l
        mat_rot_x = Matrix.Rotation(math.radians(180.0), 4, 'X')
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                instance_id = obj.name.split('_')[0]
                mat_rel = cam.matrix_world.inverted() @ obj.matrix_world
                # location
                relative_location = [mat_rel.translation[0],
                                     - mat_rel.translation[1],
                                     - mat_rel.translation[2]]
                # rotation
                # relative_rotation_euler = mat_rel.to_euler() # must be converted from radians to degrees
                relative_rotation_quat = [mat_rel.to_quaternion()[0],
                                          mat_rel.to_quaternion()[1],
                                          mat_rel.to_quaternion()[2],
                                          mat_rel.to_quaternion()[3]]
                quat_x = [0, 1, 0, 0]
                quat = quanternion_mul(quat_x, relative_rotation_quat)
                quat = [quat[0], - quat[1], - quat[2], - quat[3]]
                instance_pose[str(instance_id)] = [quat, relative_location]

        return instance_pose


    def check_visible(self, threshold=(0.1, 0.9, 0.1, 0.9)):
        w_min, x_max, h_min, h_max = threshold
        visible_objects_list = []
        bpy.context.view_layer.update()
        cs, ce = self.camera_l.data.clip_start, self.camera_l.data.clip_end
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
                obj_center = obj.matrix_world.translation
                co_ndc = world_to_camera_view(scene, self.camera_l, obj_center)
                if (w_min < co_ndc.x < x_max and
                    h_min < co_ndc.y < h_max and
                    cs < co_ndc.z <  ce):
                    obj.select_set(True)
                    visible_objects_list.append(obj)
                else:
                    obj.select_set(False)
        return visible_objects_list


def setModelPosition(instance, location, quaternion):
    instance.rotation_mode = 'QUATERNION'
    instance.rotation_quaternion[0] = quaternion[0]
    instance.rotation_quaternion[1] = quaternion[1]
    instance.rotation_quaternion[2] = quaternion[2]
    instance.rotation_quaternion[3] = quaternion[3]
    instance.location = location
###

def setRigidBody(instance):
    bpy.context.view_layer.objects.active = instance 
    object_single = bpy.context.active_object

    # add rigid body constraints to cube
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.mass = 1
    bpy.context.object.rigid_body.kinematic = True
    bpy.context.object.rigid_body.collision_shape = 'CONVEX_HULL'
    bpy.context.object.rigid_body.restitution = 0.01
    bpy.context.object.rigid_body.angular_damping = 0.8
    bpy.context.object.rigid_body.linear_damping = 0.99

    bpy.context.object.rigid_body.kinematic = False
    object_single.keyframe_insert(data_path='rigid_body.kinematic', frame=0)


def set_visiable_objects(visible_objects_list):
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and not obj.name.split('_')[0] == 'background':
            if obj in visible_objects_list:
                obj.hide_render = False
            else:
                obj.hide_render = True


def generate_CAD_model_list(scene_type, urdf_path_list, obj_uid_list):
    CAD_model_list = {}
    ###
    for idx in range(len(urdf_path_list)):
        urdf_path = urdf_path_list[idx]
        obj_uid = obj_uid_list[idx]
        class_name = 'other'
        urdf_path = str(urdf_path).replace("\\","/").split("/")
        if scene_type == "blocks":
            instance_path = "/".join(urdf_path[:-1]) + "/" + urdf_path[-1][:-5]+".obj"
        else:
            instance_path = "/".join(urdf_path[:-1])+"/"+urdf_path[-1][:-5]+"_visual.obj"
        class_list = []
        class_list.append([instance_path, class_name, obj_uid])
        if class_name == 'other' and 'other' in CAD_model_list:
            CAD_model_list[class_name] = CAD_model_list[class_name] + class_list
        else:
            CAD_model_list[class_name] = class_list
  
    return CAD_model_list


def generate_material_type(obj_name, class_material_pairs, instance_material_except_pairs, instance_material_include_pairs, material_class_instance_pairs, material_type):
    ###
    specular_type_for_ins_list = []
    transparent_type_for_ins_list = []
    diffuse_type_for_ins_list = []
    for key in instance_material_except_pairs:
            if key in material_class_instance_pairs['specular']:
                specular_type_for_ins_list.append(key)
            elif key in material_class_instance_pairs['transparent']:
                transparent_type_for_ins_list.append(key)
            elif key in material_class_instance_pairs['diffuse']:
                diffuse_type_for_ins_list.append(key)
    for key in instance_material_include_pairs:
            if key in material_class_instance_pairs['specular']:
                specular_type_for_ins_list.append(key)
            elif key in material_class_instance_pairs['transparent']:
                transparent_type_for_ins_list.append(key)
            elif key in material_class_instance_pairs['diffuse']:
                diffuse_type_for_ins_list.append(key)

    if material_type == "transparent":
        return random.sample(transparent_type_for_ins_list, 1)[0]
    elif material_type == "diffuse":
        return random.sample(diffuse_type_for_ins_list, 1)[0]
    elif material_type == "specular" or material_type == "specular_tex" or material_type == "specular_texmix":
        return random.sample(specular_type_for_ins_list, 1)[0]
    elif material_type == "specular_and_transparent":
        flag = random.randint(0, 2)
        if flag == 0:
            return random.sample(specular_type_for_ins_list, 1)[0]  ### 'specular'
        else:
            return random.sample(transparent_type_for_ins_list, 1)[0]  ### 'transparent'
    elif material_type == "mixed":
        # randomly pick material class
        flag = random.randint(0, 2) # D:S:T=1:2:2
        # select the raw material
        if flag == 0:
            return random.sample(diffuse_type_for_ins_list, 1)[0] ### 'diffuse'
        # select one from specular and transparent
        elif flag == 1:
            return random.sample(specular_type_for_ins_list, 1)[0] ### 'specular'
        else:
            return random.sample(transparent_type_for_ins_list, 1)[0]  ### 'transparent'
    else:
        raise ValueError(f"Material type error: {material_type}")