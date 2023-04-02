import os
import random
import bpy
import math
import numpy as np
from rd.modify_material import  set_modify_table_material, set_modify_floor_material
from rd.render_utils import *

def blender_init_scene(code_root, log_root_dir, obj_texture_image_root_path, scene_type, urdfs_and_poses_dict, round_idx, logdir, check_seen_scene, material_type, gpuid, output_modality_dict):
    if scene_type == "pile":
        seed = 1143+830+round_idx
    elif scene_type == "packed":
        seed = 111143+170+round_idx
    else:
        seed = 43+round_idx
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    DEVICE_LIST = [int(gpuid)]

    obj_texture_image_idxfile = "test_paths.txt"
    asset_root = code_root
    env_map_path = os.path.join(asset_root, "envmap_lib_test")
    real_table_image_root_path = os.path.join(asset_root, "realtable_test")
    real_floor_image_root_path = os.path.join(asset_root, "realfloor_test")

    emitter_pattern_path = os.path.join(asset_root, "pattern", "test_pattern.png")
    default_background_texture_path = os.path.join(asset_root, "texture", "texture_0.jpg")
    table_CAD_model_path = os.path.join(asset_root, "table_obj", "table.obj")

    output_root_path = os.path.join(log_root_dir, "rendered_results/" + str(logdir).split("/")[-1])
    
    obj_uid_list = [str(uid) for uid in urdfs_and_poses_dict]
    obj_scale_list = [value[0] for value in urdfs_and_poses_dict.values()]
    obj_quat_list = [value[1][[3, 0, 1, 2]] for value in urdfs_and_poses_dict.values()]
  
    obj_trans_list = []
    for value in urdfs_and_poses_dict.values():
        T = value[2]
        T = T + tsdf2blender_coord_T_shift
        obj_trans_list.append(T)
    
    urdf_path_list = [os.path.join(value[3]) for value in urdfs_and_poses_dict.values()] #"/".join(code_root.split("/")[:-2]), 

    max_instance_num = 20

    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    # generate CAD model list
    CAD_model_list = generate_CAD_model_list(scene_type, urdf_path_list, obj_uid_list)

    renderer = BlenderRenderer(viewport_size_x=camera_width, viewport_size_y=camera_height, DEVICE_LIST=DEVICE_LIST)
    renderer.loadImages(emitter_pattern_path, env_map_path, real_table_image_root_path, real_floor_image_root_path, obj_texture_image_root_path, obj_texture_image_idxfile, check_seen_scene)
    renderer.addEnvMap()
    renderer.addBackground(background_size, background_position, background_scale, default_background_texture_path)
    renderer.addMaterialLib(material_class_instance_pairs)  ###
    renderer.addMaskMaterial(max_instance_num)
    renderer.addNOCSMaterial()
    renderer.addNormalMaterial()

    renderer.clearModel()
    # set scene output path
    path_scene = output_root_path #os.path.join(output_root_path, uuid.uuid4().hex, "init") ### "scene_"+str(SCENE_NUM).zfill(4))
    if os.path.exists(path_scene)==False:
        os.makedirs(path_scene)

    # camera pose list, environment light list and background material_listz
    quaternion_list = []
    translation_list = []

    # environment map list
    env_map_id_list = []
    rotation_elur_z_list = []

    # background material list
    background_material_list = []

    # table material list
    table_material_list = []

    look_at = look_at_shift
    quat_list, trans_list, rot_list = genCameraPosition(look_at)

    rot_array = np.array(rot_list)  # (256, 3, 3)
    trans_array = np.array(trans_list)  #  (256, 3, 1)
    cam_RT = np.concatenate([rot_array, trans_array], 2)
    zero_one = np.expand_dims([[0, 0, 0, 1]],0).repeat(rot_array.shape[0],axis=0)
    cam_RT = np.concatenate([cam_RT, zero_one], 1)  # (256, 4, 4)

    # generate camara pose list
    for i in range(NUM_FRAME_PER_SCENE):
        quaternion = quat_list[i] #### cam_pose_list[i][0]
        translation = trans_list[i] #### cam_pose_list[i][1]
        quaternion_list.append(quaternion)
        translation_list.append(translation)

    flag_env_map = random.randint(0, len(renderer.env_map) - 1)
    flag_env_map_rot = random.uniform(-math.pi, math.pi)
    flag_realfloor = random.randint(0, len(renderer.realfloor_img_list) - 1)
    flag_realtable = random.randint(0, len(renderer.realtable_img_list) - 1)

    # generate environment map list
    env_map_id_list.append(flag_env_map)
    rotation_elur_z_list.append(flag_env_map_rot)

    
    # generate background material list 
    if my_material_randomize_mode == 'raw':
        background_material_list.append(renderer.my_material['default_background'])
    else:
        material_selected = random.sample(renderer.my_material['background'], 1)[0] ### renderer.my_material['background'][1] 
        background_material_list.append(material_selected)
        material_selected = random.sample(renderer.my_material['table'], 1)[0] ### renderer.my_material['table'][0]  
        table_material_list.append(material_selected)

    # read objects from floder
    meta_output = {}
    select_model_list = []
    select_model_list_other = []
    select_model_list_transparent = []
    select_model_list_dis = []
    select_number = 1

    for item in CAD_model_list:
        if item in ['other']:
            test = CAD_model_list[item]
            for model in test:
                select_model_list.append(model)
        else:
            raise ValueError("No such category!")
    
    # table model
    renderer.loadModel(table_CAD_model_path)
    obj = bpy.data.objects['table']
    # resize table, unit: m
    class_scale = 0.001
    obj.scale = (class_scale, class_scale, class_scale)
    y_transform = np.array([[0,0,-1],[0,1,0],[1,0,0]])
    transform = y_transform # z_transform @ y_transform
    obj_world_pose_quat = quaternionFromRotMat(transform)
    obj_world_pose_T_shift = np.array([0,0,-0.0751])
    obj_world_pose_T = obj_world_pose_T_shift ### np.array([cam_pose_T[0],cam_pose_T[1],0]) + obj_world_pose_T_shift
    setModelPosition(obj, obj_world_pose_T, obj_world_pose_quat)
    obj_world_pose_T_shift = np.array([0,0,0])
    obj_world_pose_T = obj_world_pose_T_shift ### np.array([cam_pose_T[0],cam_pose_T[1],0]) + obj_world_pose_T_shift
    # setModelPosition(obj, obj_world_pose_T, obj_world_pose_quat)
    bpy.ops.mesh.primitive_plane_add(size=1., enter_editmode=False, align='WORLD', location=obj_world_pose_T)
    bpy.ops.rigidbody.object_add()
    bpy.context.object.rigid_body.type = 'PASSIVE'
    bpy.context.object.rigid_body.collision_shape = 'BOX'
    obj = bpy.data.objects['Plane']
    obj.name = 'tableplane'
    obj.data.name = 'tableplane'
    obj.scale = (0.898, 1.3, 1.)
    ###

    instance_id = 1
    # set object parameters
    imported_obj_name_list = []
    for model in select_model_list:
        instance_path = model[0]
        class_name = model[1]
        instance_uid = model[2]
        instance_folder = model[0].split('/')[-1][:-4] 
        instance_name = str(instance_id) + "_" + class_name + "_" + instance_folder + "_" + instance_uid ### class_folder + "_" + instance_folder

        material_type_in_mixed_mode = generate_material_type(instance_name, class_material_pairs, instance_material_except_pairs, instance_material_include_pairs, material_class_instance_pairs, material_type)

        # download CAD model and rename
        renderer.loadModel(instance_path)
        import_obj_name = instance_folder #bpy.data.objects.keys()["instance_folder"] 

        obj = bpy.data.objects[import_obj_name]
        obj.name = instance_name
        obj.data.name = instance_name

        obj_world_pose_T = obj_trans_list[instance_id-1] ### obj_pose_list[instance_id-1][:3,3]
        obj_world_pose_quat = obj_quat_list[instance_id-1] ### quaternionFromRotMat(obj_world_pose_R)
        setModelPosition(obj, obj_world_pose_T, obj_world_pose_quat)

        # set object as rigid body
        setRigidBody(obj)

        # set material
        renderer.set_material_randomize_mode(class_material_pairs, my_material_randomize_mode, obj, material_type_in_mixed_mode)
        
        # generate meta file
        class_scale = obj_scale_list[instance_id-1] ### random.uniform(g_synset_name_scale_pairs[class_name][0], g_synset_name_scale_pairs[class_name][1])
        obj.scale = (class_scale, class_scale, class_scale)

        # query material type
        material_class_id = None
        for key in material_class_instance_pairs:
            if material_type_in_mixed_mode == 'raw':
                material_class_id = material_class_id_dict[material_type_in_mixed_mode]
                break
            elif material_type_in_mixed_mode in material_class_instance_pairs[key]:
                material_class_id = material_class_id_dict[key]
                break
        if material_class_id == None:
            raise ValueError("material_class_id error!")

        meta_output[str(instance_id)] = [#str(g_synset_name_label_pairs[class_name]),
                                         ### class_folder, 
                                         str(instance_folder), 
                                         ### str(class_scale),
                                         str(material_class_id), ###str(material_name_label_pairs[material_type_in_mixed_mode])]
                                         str(material_type_id_dict[material_type_in_mixed_mode])
                                         ]

        instance_id += 1

    if output_modality_dict['IR'] or output_modality_dict['RGB']:
        renderer.setEnvMap(env_map_id_list[0], rotation_elur_z_list[0])
        # pick real floor image
        selected_realfloor_img = renderer.realfloor_img_list[flag_realfloor]

        # pcik real table image
        selected_realtable_img = renderer.realtable_img_list[flag_realtable]
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name.split('_')[0] == 'background':
                if obj.name == 'background_0':
                    set_modify_floor_material(obj, background_material_list[0], selected_realfloor_img) ### renderer.realfloor_img_list)
                else:
                    background_0_obj = bpy.data.objects['background_0']
                    obj.active_material = background_0_obj.material_slots[0].material
            elif obj.type == "MESH" and obj.name == 'table':
                set_modify_table_material(obj, table_material_list[0], selected_realtable_img)### renderer.realtable_img_list)
            elif obj.type == "MESH" and obj.name == 'tableplane':
                    table_obj = bpy.data.objects['table']
                    obj.active_material = table_obj.material_slots[0].material

    return renderer, quaternion_list, translation_list, path_scene#, obj_trans_list, obj_quat_list


def blender_update_sceneobj(obj_name_list, obj_trans_list, obj_quat_list, obj_uid_list):
    for obj_name in bpy.data.objects.keys():
        obj = bpy.data.objects[obj_name]
        if obj.type == 'MESH' and obj_name[0:10] != "background" and obj_name not in ['camera_l', 'camera_r', 'light_emitter', 'table', 'tableplane']:
            obj_uid = obj_name.split("_")[-1]
            if obj_uid not in obj_uid_list:
                print("[V]blender_update_sceneobj: obj_uid [not in] obj_uid_list: ", obj_name, obj_name_list, obj_uid, obj_uid_list)
                obj.hide_render = True
            else:
                obj.hide_render = False
                obj_world_pose_T = obj_trans_list[obj_uid_list.index(obj_uid)] + tsdf2blender_coord_T_shift
                obj_world_pose_quat = obj_quat_list[obj_uid_list.index(obj_uid)]
                print("[V]blender_update_sceneobj: obj_uid [in] obj_uid_list: ", obj_name, obj_name_list, obj_uid, obj_uid_list, obj_world_pose_T, obj_world_pose_quat)
                setModelPosition(obj, obj_world_pose_T, obj_world_pose_quat)


def blender_render(renderer, quaternion_list, translation_list, path_scene, render_frame_list, output_modality_dict, camera_focal, is_init=False):
    # set the key frame
    scene = bpy.data.scenes['Scene']

    camera_fov = 2 * math.atan(camera_width / (2 * camera_focal))

    # render IR image and RGB image
    if output_modality_dict['IR'] or output_modality_dict['RGB']:
        if is_init:
            renderer.src_energy_for_rgb_render = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value

        for i in render_frame_list:  
            renderer.setCamera(quaternion_list[i], translation_list[i], camera_fov, baseline_distance)
            renderer.setLighting()

            # render RGB image
            if output_modality_dict['RGB']:
                rgb_dir_path = os.path.join(path_scene, 'rgb')
                if os.path.exists(rgb_dir_path) == False:
                    os.makedirs(rgb_dir_path)

                renderer.render_mode = "RGB"
                camera = bpy.data.objects['camera_l']
                scene.camera = camera
                save_path = rgb_dir_path
                save_name = str(i).zfill(4)
                renderer.render(save_name, save_path)

            # render IR image
            if output_modality_dict['IR']:
                ir_l_dir_path = os.path.join(path_scene, 'ir_l')
                if os.path.exists(ir_l_dir_path)==False:
                    os.makedirs(ir_l_dir_path)
                ir_r_dir_path = os.path.join(path_scene, 'ir_r')
                if os.path.exists(ir_r_dir_path)==False:
                    os.makedirs(ir_r_dir_path)

                renderer.render_mode = "IR"
                camera = bpy.data.objects['camera_l']
                scene.camera = camera
                save_path = ir_l_dir_path
                save_name = str(i).zfill(4)
                renderer.render(save_name, save_path)

                camera = bpy.data.objects['camera_r']
                scene.camera = camera
                save_path = ir_r_dir_path
                save_name = str(i).zfill(4)
                renderer.render(save_name, save_path)
        
    # render normal map and depth map
    if output_modality_dict['Normal']:
        # set normal as material
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.data.materials.clear()
                obj.active_material = renderer.my_material["normal"]

        # render normal map
        for i in render_frame_list:
            renderer.setCamera(quaternion_list[i], translation_list[i], camera_fov, baseline_distance)

            normal_dir_path = os.path.join(path_scene, 'normal')
            if os.path.exists(normal_dir_path)==False:
                os.makedirs(normal_dir_path)
            depth_dir_path = os.path.join(path_scene, 'depth')
            if os.path.exists(depth_dir_path)==False:
                os.makedirs(depth_dir_path)

            renderer.render_mode = "Normal"
            camera = bpy.data.objects['camera_l']
            scene.camera = camera
            save_path = normal_dir_path
            save_name = str(i).zfill(4)
            renderer.render(save_name, save_path)

    context = bpy.context
    for ob in context.selected_objects:
        ob.animation_data_clear()
