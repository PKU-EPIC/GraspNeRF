from pathlib import Path
import time
import os
import numpy as np
import pybullet

from gd.grasp import Label
from gd.perception import *
from gd.utils import btsim, workspace_lines
from gd.utils.transform import Rotation, Transform


class ClutterRemovalSim(object):
    def __init__(self, scene, object_set, gui=True, seed=None, renderer_root_dir="", args=None):
        assert scene in ["pile", "packed", "single"]

        self.urdf_root = Path(renderer_root_dir + "/data/urdfs")
        self.scene = scene
        self.object_set = object_set
        self.discover_objects()

        self.global_scaling = {"blocks": 1.67}.get(object_set, 1.0)
        self.gui = gui

        self.rng = np.random.RandomState(seed) if seed else np.random
        self.world = btsim.BtWorld(self.gui)
        self.gripper = Gripper(self.world)
        self.size = 6 * self.gripper.finger_depth
        intrinsic = CameraIntrinsic(640, 480, 540.0, 540.0, 320.0, 240.0) # TODO: cfg
        self.camera = self.world.add_camera(intrinsic, 0.1, 2.0)

        ##
        self.args = args
        self.renderer_root_dir = renderer_root_dir
        if self.args.load_scene_descriptor:
            if self.scene == "pile":
                dir_name = "pile_pile_test_200"
            elif self.scene == "packed":
                dir_name = "packed_packed_test_200"
            elif self.scene == "single":
                dir_name = "single_single_test_200"
            scene_root_dir = os.path.join(renderer_root_dir, "data/mesh_pose_list", dir_name)
            self.scene_descriptor_list = [os.path.join(scene_root_dir, i) for i in sorted(os.listdir(scene_root_dir))]

    @property
    def num_objects(self):
        return max(0, self.world.p.getNumBodies() - 1)  # remove table from body count

    def discover_objects(self):
        root = self.urdf_root / self.object_set
        self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]

    def save_state(self):
        self._snapshot_id = self.world.save_state()

    def restore_state(self):
        self.world.restore_state(self._snapshot_id)

    def reset(self, object_count, n_round):
        self.world.reset()
        self.world.set_gravity([0.0, 0.0, -9.81])
        self.draw_workspace()

        if self.gui:
            self.world.p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=0.0,
                cameraPitch=-45,
                cameraTargetPosition=[0.15, 0.50, -0.3],
            )

        table_height = self.gripper.finger_depth
        self.place_table(table_height)

        ##
        if self.args.gen_scene_descriptor:
            if self.scene == "pile":
                urdfs_and_poses_dict = self.generate_pile_scene(object_count, table_height)
                return urdfs_and_poses_dict
            elif self.scene == "packed":
                urdfs_and_poses_dict = self.generate_packed_scene(object_count, table_height)
                return urdfs_and_poses_dict
            else:
                raise ValueError("Invalid scene argument")
        elif self.args.load_scene_descriptor:
            scene_descriptor_npz = self.scene_descriptor_list[n_round]

            if self.scene == "pile":
                urdfs_and_poses_dict = self.generate_pile_scene(object_count, table_height, scene_descriptor_npz)
            elif self.scene == "packed":
                urdfs_and_poses_dict = self.generate_packed_scene(object_count, table_height, scene_descriptor_npz)
            elif self.scene == "single":
                urdfs_and_poses_dict = self.generate_packedsingle_scene(object_count, table_height, scene_descriptor_npz)
            else:
                raise ValueError("Invalid scene argument")
            return urdfs_and_poses_dict
        else:
            raise NotImplementedError

    def draw_workspace(self):
        points = workspace_lines(self.size)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.world.p.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )

    def place_table(self, height):
        urdf = self.urdf_root / "setup" / "plane.urdf"
        pose = Transform(Rotation.identity(), [0.15, 0.15, height])
        self.world.load_urdf(urdf, pose, scale=0.6)

        # define valid volume for sampling grasps
        lx, ux = 0.02, self.size - 0.02
        ly, uy = 0.02, self.size - 0.02
        lz, uz = height + 0.005, self.size
        self.lower = np.r_[lx, ly, lz]
        self.upper = np.r_[ux, uy, uz]

    def generate_seen_scene(self, table_height, mesh_pose_npz):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # read mesh_pose_npz
        print("########## scene name: ", mesh_pose_npz)
        if self.args.check_seen_scene:
            urdfs_and_poses_dict = np.load(mesh_pose_npz, allow_pickle=True)['pc']
            urdf_path_list = list(urdfs_and_poses_dict[:,0])
            obj_scale_list = list(urdfs_and_poses_dict[:,1])
            obj_RT_list = list(urdfs_and_poses_dict[:,2])

        urdfs_and_poses_dict = {}     ##
        for i in range(len(urdf_path_list)):
            urdf = os.path.join(self.renderer_root_dir, urdf_path_list[i].replace("_visual.obj",".urdf"))
            RT = obj_RT_list[i]
            R = RT[:3,:3]
            T = RT[:3,3]
            rotation = Rotation.from_matrix(R)
            pose = Transform(rotation ,T)
            scale = obj_scale_list[i]
            body = self.world.load_urdf(urdf, pose, scale=scale)
            body.set_pose(pose=Transform(rotation, T))

        # remove box
        self.world.remove_body(box)

        removed_object = True
        while removed_object:
            removed_object, obj_body_list = self.remove_objects_outside_workspace()

        for urdf, scale, rest_pose_quat, rest_pose_trans, body_uid in obj_body_list:
            urdfs_and_poses_dict[body_uid] = [scale, rest_pose_quat, rest_pose_trans, str(urdf)]

        return urdfs_and_poses_dict

    def generate_pile_scene(self, object_count, table_height, scene_descriptor_npz=None):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        urdfs_and_poses_dict = {}
        if self.args.gen_scene_descriptor:
            urdf_path_list = self.rng.choice(self.object_urdfs, size=object_count)
        elif self.args.load_scene_descriptor:
            dict = np.load(scene_descriptor_npz, allow_pickle=True).item()
            obj_scale_list = [value[0] for value in dict.values()]
            obj_quat_list = [value[1] for value in dict.values()]
            obj_xy_list = [value[2] for value in dict.values()]
            if self.scene != self.object_set:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[3].replace(self.scene, self.object_set)) for value in dict.values()]
            else:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[3]) for value in dict.values()]

        # drop objects
        for i in range(len(urdf_path_list)):
            if self.args.gen_scene_descriptor:
                rotation = Rotation.random(random_state=self.rng)
                xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
                pose = Transform(rotation, np.r_[xy, table_height + 0.2])
                scale = self.rng.uniform(0.8, 1.0)
                # save info
                urdfs_and_poses_dict[i] = [scale, pose.rotation.as_quat(), xy, str(urdf_path_list[i])]     # (x, y, z, w)
            elif self.args.load_scene_descriptor:
                rotation = Rotation.from_quat(obj_quat_list[i])
                xy = obj_xy_list[i]
                pose = Transform(rotation, np.r_[xy, table_height + 0.2])
                scale = obj_scale_list[i]
            self.world.load_urdf(urdf_path_list[i], pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)

        # remove box
        self.world.remove_body(box)
        obj_body_list = self.remove_and_wait()

        if self.args.gen_scene_descriptor:
            return urdfs_and_poses_dict
        else:
            for urdf, scale, rest_pose_quat, rest_pose_trans, body_uid in obj_body_list:
                urdfs_and_poses_dict[body_uid] = [scale, rest_pose_quat, rest_pose_trans, str(urdf)]
            return urdfs_and_poses_dict

    def generate_packed_scene(self, object_count, table_height, scene_descriptor_npz=None):
        attempts = 0
        max_attempts = 12

        if self.args.gen_scene_descriptor:
            urdfs_and_poses_dict = {}
        elif self.args.load_scene_descriptor:
            dict = np.load(scene_descriptor_npz, allow_pickle=True).item()
            obj_scale_list = [value[0] for value in dict.values()]
            obj_angle_list = [value[1] for value in dict.values()]
            obj_x_list = [value[2] for value in dict.values()]
            obj_y_list = [value[3] for value in dict.values()]
            if self.scene != self.object_set:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[4].replace(self.scene, self.object_set)) for value in dict.values()]
            else:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[4]) for value in dict.values()]

        while self.num_objects < object_count and attempts < max_attempts:
            self.save_state()
            if self.args.gen_scene_descriptor:
                urdf = self.rng.choice(self.object_urdfs)
                x = self.rng.uniform(0.08, 0.22)
                y = self.rng.uniform(0.08, 0.22)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                scale = self.rng.uniform(0.7, 0.9)
                # save info
                urdfs_and_poses_dict[attempts] = [scale, angle, x, y, str(urdf)]     # (x, y, z, w)
            elif self.args.load_scene_descriptor:
                urdf = urdf_path_list[attempts]
                angle = obj_angle_list[attempts]
                x = obj_x_list[attempts]
                y = obj_y_list[attempts]
                scale = obj_scale_list[attempts]

            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            z = 1.0
            pose = Transform(rotation, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))

            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

        if self.args.gen_scene_descriptor:
            return urdfs_and_poses_dict
        else:
            remain_obj_inws_infos = []
            for body in list(self.world.bodies.values()):
                urdf = self.world.bodies_urdfs[body.uid][0]
                scale = self.world.bodies_urdfs[body.uid][1]
                if str(urdf).split("/")[-1] != "box.urdf" and str(urdf).split("/")[-1] != "plane.urdf":
                    rest_pose = body.get_pose()
                    rest_pose_quat = rest_pose.rotation.as_quat()  # (x, y, z, w)
                    rest_pose_trans = rest_pose.translation
                    remain_obj_inws_infos.append([urdf, scale, rest_pose_quat, rest_pose_trans, str(body.uid)])
            urdfs_and_poses_dict = {}
            for urdf, scale, rest_pose_quat, rest_pose_trans, body_uid in remain_obj_inws_infos:
                urdfs_and_poses_dict[body_uid] = [scale, rest_pose_quat, rest_pose_trans, str(urdf)]
            return urdfs_and_poses_dict

    def generate_packedsingle_scene(self, object_count, table_height, scene_descriptor_npz=None):
        attempts = 0

        if self.args.gen_scene_descriptor:
            urdfs_and_poses_dict = {}
        elif self.args.load_scene_descriptor:
            dict = np.load(scene_descriptor_npz, allow_pickle=True).item()
            obj_scale_list = [value[0] for value in dict.values()]
            obj_angle_list = [value[1] for value in dict.values()]
            obj_x_list = [value[2] for value in dict.values()]
            obj_y_list = [value[3] for value in dict.values()]
            if self.scene != self.object_set:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[4].replace(self.scene, self.object_set)) for value in dict.values()]
            else:
                urdf_path_list = [os.path.join(self.renderer_root_dir, value[4]) for value in dict.values()]

        for _ in range(1):
            self.save_state()
            if self.args.gen_scene_descriptor:
                urdf = self.rng.choice(self.object_urdfs)
                x = self.rng.uniform(0.08, 0.22)
                y = self.rng.uniform(0.08, 0.22)
                angle = self.rng.uniform(0.0, 2.0 * np.pi)
                scale = self.rng.uniform(0.7, 0.9)
                # save info
                urdfs_and_poses_dict[attempts] = [scale, angle, x, y, str(urdf)]     # (x, y, z, w)
            elif self.args.load_scene_descriptor:
                urdf = urdf_path_list[attempts]
                angle = obj_angle_list[attempts]
                x = obj_x_list[attempts]
                y = obj_y_list[attempts]
                scale = obj_scale_list[attempts]

            rotation = Rotation.from_rotvec(angle * np.r_[0.0, 0.0, 1.0])
            z = 1.0
            pose = Transform(rotation, np.r_[x, y, z])
            body = self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            lower, upper = self.world.p.getAABB(body.uid)
            z = table_height + 0.5 * (upper[2] - lower[2]) + 0.002
            body.set_pose(pose=Transform(rotation, np.r_[x, y, z]))

            self.world.step()

            if self.world.get_contacts(body):
                self.world.remove_body(body)
                self.restore_state()
            else:
                self.remove_and_wait()
            attempts += 1

        if self.args.gen_scene_descriptor:
            return urdfs_and_poses_dict
        else:
            remain_obj_inws_infos = []
            for body in list(self.world.bodies.values()):
                urdf = self.world.bodies_urdfs[body.uid][0]
                scale = self.world.bodies_urdfs[body.uid][1]
                if str(urdf).split("/")[-1] != "box.urdf" and str(urdf).split("/")[-1] != "plane.urdf":
                    rest_pose = body.get_pose()
                    rest_pose_quat = rest_pose.rotation.as_quat()  # (x, y, z, w)
                    rest_pose_trans = rest_pose.translation
                    remain_obj_inws_infos.append([urdf, scale, rest_pose_quat, rest_pose_trans, str(body.uid)])
            urdfs_and_poses_dict = {}
            for urdf, scale, rest_pose_quat, rest_pose_trans, body_uid in remain_obj_inws_infos:
                urdfs_and_poses_dict[body_uid] = [scale, rest_pose_quat, rest_pose_trans, str(urdf)]
            return urdfs_and_poses_dict


    def acquire_tsdf(self, n, N=None):
        """Render synthetic depth images from n viewpoints and integrate into a TSDF.

        If N is None, the n viewpoints are equally distributed on circular trajectory.

        If N is given, the first n viewpoints on a circular trajectory consisting of N points are rendered.
        """
        tsdf = TSDFVolume(self.size, 40)
        high_res_tsdf = TSDFVolume(self.size, 120)

        origin = Transform(Rotation.identity(), np.r_[self.size / 2, self.size / 2, 0])
        r = 2.0 * self.size
        theta = np.pi / 6.0

        N = N if N else n
        phi_list = 2.0 * np.pi * np.arange(n) / N
        extrinsics = [camera_on_sphere(origin, r, theta, phi).as_matrix() for phi in phi_list]

        timing = 0.0
        for extrinsic in extrinsics:
            depth_img = self.camera.render(extrinsic)[1]
            tic = time.time()
            tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)
            timing += time.time() - tic
            high_res_tsdf.integrate(depth_img, self.camera.intrinsic, extrinsic)

        return tsdf, high_res_tsdf.get_cloud(), timing

    def execute_grasp(self, grasp, remove=True, allow_contact=False):
        # --grasp is the target containing pose and width
        # -- flag to control whether allow collision between pre-target and target
        # -- remove whether remove the objec from the scene after succesful grasp
        T_world_grasp = grasp.pose
        T_grasp_pregrasp = Transform(Rotation.identity(), [0.0, 0.0, -0.05])
        T_world_pregrasp = T_world_grasp * T_grasp_pregrasp

        # approach along z-axis of the gripper
        approach = T_world_grasp.rotation.as_matrix()[:, 2]
        angle = np.arccos(np.dot(approach, np.r_[0.0, 0.0, -1.0]))
        if angle > np.pi / 3.0:
            # side grasp, lift the object after establishing a grasp
            T_grasp_pregrasp_world = Transform(Rotation.identity(), [0.0, 0.0, 0.1])
            T_world_retreat = T_grasp_pregrasp_world * T_world_grasp
        else:
            T_grasp_retreat = Transform(Rotation.identity(), [0.0, 0.0, -0.1])
            T_world_retreat = T_world_grasp * T_grasp_retreat

        # move the gripper to pregrasp pose and detect the collision
        self.gripper.reset(T_world_pregrasp)

        if self.gripper.detect_contact():
            result = Label.FAILURE, self.gripper.max_opening_width
            print("0")
        else:
            #move the gripper to the target pose and detect collision
            self.gripper.move_tcp_xyz(T_world_grasp, abort_on_contact=True)
            """
            self.set_obj_pose_again(self.mesh_pose_npz)
            """
            if self.gripper.detect_contact() and not allow_contact:
                result = Label.FAILURE, self.gripper.max_opening_width
                print("1")
            else:
                self.gripper.move(0.0)      # shrink the gripper
                # lift the gripper up along z-axis of the world frame or z-axis of the gripper frame
                self.gripper.move_tcp_xyz(T_world_retreat, abort_on_contact=False)
                if self.check_success(self.gripper):
                    result = Label.SUCCESS, self.gripper.read()
                    print("2")
                    if remove:
                        contacts = self.world.get_contacts(self.gripper.body)
                        self.world.remove_body(contacts[0].bodyB, isRemoveObjPerGrasp=True)
                else:
                    result = Label.FAILURE, self.gripper.max_opening_width
                    print("3")
        self.world.remove_body(self.gripper.body)

        remain_obj_inws_infos = []
        if remove:
            remain_obj_inws_infos = self.remove_and_wait()  ### wait for blender to render updated scene

        return result, remain_obj_inws_infos

    def remove_and_wait(self):
        # wait for objects to rest while removing bodies that fell outside the workspace
        removed_object = True
        while removed_object:
            self.wait_for_objects_to_rest()
            removed_object, remain_obj_inws_infos = self.remove_objects_outside_workspace()
        return remain_obj_inws_infos

    def wait_for_objects_to_rest(self, timeout=2.0, tol=0.01):
        timeout = self.world.sim_time + timeout
        objects_resting = False
        while not objects_resting and self.world.sim_time < timeout:
            # simulate a quarter of a second
            for _ in range(60):
                self.world.step()
            # check whether all objects are resting
            objects_resting = True
            for _, body in self.world.bodies.items():
                if np.linalg.norm(body.get_velocity()) > tol:
                    objects_resting = False
                    break

    def remove_objects_outside_workspace(self):
        removed_object = False
        remain_obj_inws_infos = []   ##

        for body in list(self.world.bodies.values()):
            xyz = body.get_pose().translation
            if np.any(xyz < 0.0) or np.any(xyz > self.size):
                self.world.remove_body(body)
                removed_object = True
            else:
                urdf = self.world.bodies_urdfs[body.uid][0]
                scale = self.world.bodies_urdfs[body.uid][1]
                if str(urdf).split("/")[-1] != "box.urdf" and str(urdf).split("/")[-1] != "plane.urdf":
                    rest_pose = body.get_pose()
                    rest_pose_quat = rest_pose.rotation.as_quat()  # (x, y, z, w)
                    rest_pose_trans = rest_pose.translation
                    remain_obj_inws_infos.append([urdf, scale, rest_pose_quat, rest_pose_trans, str(body.uid)])
        return removed_object, remain_obj_inws_infos     ##

    def check_success(self, gripper):
        # check that the fingers are in contact with some object and not fully closed
        contacts = self.world.get_contacts(gripper.body)
        res = len(contacts) > 0 and gripper.read() > 0.1 * gripper.max_opening_width
        return res


class Gripper(object):
    """Simulated Panda hand."""

    def __init__(self, world):
        self.world = world
        self.urdf_path = Path("data/assets/data/urdfs/panda/hand.urdf") #TODO put in cfg
        self.max_opening_width = 0.08
        self.finger_depth = 0.05
        self.T_body_tcp = Transform(Rotation.identity(), [0.0, 0.0, 0.022])
        self.T_tcp_body = self.T_body_tcp.inverse()

    def reset(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.body = self.world.load_urdf(self.urdf_path, T_world_body)
        self.body.set_pose(T_world_body)  # sets the position of the COM, not URDF link
        self.constraint = self.world.add_constraint(
            self.body,
            None,
            None,
            None,
            pybullet.JOINT_FIXED,
            [0.0, 0.0, 0.0],
            Transform.identity(),
            T_world_body,
        )
        self.update_tcp_constraint(T_world_tcp)
        # constraint to keep fingers centered
        self.world.add_constraint(
            self.body,
            self.body.links["panda_leftfinger"],
            self.body,
            self.body.links["panda_rightfinger"],
            pybullet.JOINT_GEAR,
            [1.0, 0.0, 0.0],
            Transform.identity(),
            Transform.identity(),
        ).change(gearRatio=-1, erp=0.1, maxForce=50)
        self.joint1 = self.body.joints["panda_finger_joint1"]
        self.joint1.set_position(0.5 * self.max_opening_width, kinematics=True)
        self.joint2 = self.body.joints["panda_finger_joint2"]
        self.joint2.set_position(0.5 * self.max_opening_width, kinematics=True)

    def update_tcp_constraint(self, T_world_tcp):
        T_world_body = T_world_tcp * self.T_tcp_body
        self.constraint.change(
            jointChildPivot=T_world_body.translation,
            jointChildFrameOrientation=T_world_body.rotation.as_quat(),
            maxForce=300,
        )

    def set_tcp(self, T_world_tcp):
        T_word_body = T_world_tcp * self.T_tcp_body
        self.body.set_pose(T_word_body)
        self.update_tcp_constraint(T_world_tcp)

    def move_tcp_xyz(self, target, eef_step=0.002, vel=0.10, abort_on_contact=True):
        T_world_body = self.body.get_pose()
        T_world_tcp = T_world_body * self.T_body_tcp

        diff = target.translation - T_world_tcp.translation
        n_steps = int(np.linalg.norm(diff) / eef_step)
        dist_step = diff / n_steps
        dur_step = np.linalg.norm(dist_step) / vel

        for _ in range(n_steps):
            T_world_tcp.translation += dist_step
            self.update_tcp_constraint(T_world_tcp)
            for _ in range(int(dur_step / self.world.dt)):
                self.world.step()
            if abort_on_contact and self.detect_contact():
                return

    def detect_contact(self, threshold=5):
        if self.world.get_contacts(self.body):
            return True
        else:
            return False

    def move(self, width):
        self.joint1.set_position(0.5 * width)
        self.joint2.set_position(0.5 * width)
        for _ in range(int(0.5 / self.world.dt)):
            self.world.step()

    def read(self):
        width = self.joint1.get_position() + self.joint2.get_position()
        return width
