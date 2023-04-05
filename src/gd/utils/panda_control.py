from panda_robot import PandaArm
import rospy
import numpy as np
import quaternion
from gd.utils.transform import Transform
from scipy.spatial.transform import Rotation

class PandaCommander(object):
    def __init__(self):
        self.name = "panda_arm"
        self.r = PandaArm()
        self.r.enable_robot()
        rospy.loginfo("PandaCommander ready")
        self.moving = False
    def reset(self):
        rospy.logwarn("reset and go home!")
        self.r.enable_robot()
        self.home()

    def clear(self):
        self.r.exit_control_mode()

    def home(self):
        self.moving = True
        self.r.move_to_neutral()
        self.moving = False
        rospy.loginfo("PandaCommander: Arrived home!")

    def goto_joints(self, joints):
        self.moving = True
        self.r.move_to_joint_position(joints)
        self.moving = False

    def get_joints(self):
        return self.r.angles()
    def goto_pose(self, pose):
        rospy.loginfo("PandaCommander: goto pose " + str(pose.to_list()[-3:]))
        self.moving = True
        x, y, z, w = pose.rotation.as_quat()
        self.r.move_to_cartesian_pose(pose.translation.astype(np.float32), np.quaternion(w, x, y, z))
        safe = self.r.in_safe_state()
        if self.r.has_collided():
            rospy.logwarn("collided!")
            self.r.enable_robot()
            return
        error = self.r.error_in_current_state()
        if error or not safe:
            rospy.logwarn("error or not safe! reset and run again.")
            self.reset()
            self.goto_pose(pose)
            return 
        self.moving = False

    def get_pose(self):
            pos, rot = self.r.ee_pose()
            return Transform(Rotation.from_quat([rot.x, rot.y, rot.z, rot.w]), pos)
    def grasp(self, width=0.0, force=10.0):
        return self.r.exec_gripper_cmd(width, force=force)

    def move_gripper(self, width):
        return self.r.exec_gripper_cmd(width)

    def get_gripper_width(self):
        return np.sum(self.r.gripper_state()['position'])
