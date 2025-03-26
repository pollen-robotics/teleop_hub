import numpy as np
from reachy2_sdk import ReachySDK  # type: ignore
from scipy.spatial.transform import Rotation as R  # type: ignore
from utils import make_homogenous_matrix_from_rotation_matrix

from google.protobuf.wrappers_pb2 import FloatValue, Int32Value

from reachy2_sdk_api.arm_pb2 import (  # type: ignore
    ArmCartesianGoal,
    IKConstrainedMode,
    IKContinuousMode,
)
from reachy2_sdk_api.kinematics_pb2 import Matrix4x4  # type: ignore

IP = "localhost"

class Reachy2:
    def __init__(self):
        self.reachy = ReachySDK(IP)


    def init_robot(self):
        self.reachy.turn_on()
        if self.reachy.mobile_base is not None:
            self.reachy.mobile_base.reset_odometry()
        # self.reachy.head.l_antenna.turn_on()
        # self.reachy.head.r_antenna.turn_on()
        self.reachy.r_arm.gripper.open()
        self.reachy.l_arm.gripper.open()
        self.reachy.reset_default_limits()

        position = [0.36, -0.2, -0.28]
        orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
        r_pose = make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
        position = [0.36, 0.2, -0.28]
        orientation = R.from_euler('xyz', [0, -np.pi/2, 0], degrees=False)
        l_pose = make_homogenous_matrix_from_rotation_matrix(orientation.as_matrix(), position)
        head_joints = [0, 0, 0]

        joints = self.reachy.r_arm.inverse_kinematics(r_pose)
        self.reachy.r_arm.goto(joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False)
        joints = self.reachy.l_arm.inverse_kinematics(l_pose)
        self.reachy.l_arm.goto(joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=False)
        self.reachy.head.goto(head_joints, 3.0, degrees=True, interpolation_mode="minimum_jerk", wait=True)


    def fk(self, arm):
        if arm == "r_arm":
            return self.reachy.r_arm.forward_kinematics()
        else:
            return self.reachy.l_arm.forward_kinematics()


    def go_to_pose(self, pose, arm):
        if arm == "r_arm":
            request = ArmCartesianGoal(
                id=self.reachy.r_arm._part_id,
                goal_pose=Matrix4x4(data=pose.flatten().tolist()),
                continuous_mode=IKContinuousMode.UNFREEZE,
                constrained_mode=IKConstrainedMode.UNCONSTRAINED,
                preferred_theta=FloatValue(
                    value=-4 * np.pi / 6,
                ),
                d_theta_max=FloatValue(value=0.05),
                order_id=Int32Value(value=5),
            )
            self.reachy.r_arm._stub.SendArmCartesianGoal(request)
        elif arm == "l_arm":
            request = ArmCartesianGoal(
                id=self.reachy.l_arm._part_id,
                goal_pose=Matrix4x4(data=pose.flatten().tolist()),
                continuous_mode=IKContinuousMode.UNFREEZE,
                constrained_mode=IKConstrainedMode.UNCONSTRAINED,
                preferred_theta=FloatValue(
                    value=-4 * np.pi / 6,
                ),
                d_theta_max=FloatValue(value=0.05),
                order_id=Int32Value(value=5),
            )
            self.reachy.l_arm._stub.SendArmCartesianGoal(request)

    def move_gripper(self, joint, arm):
        if arm == "r_arm":
            self.reachy.r_arm.gripper.goal_position = joint
            self.reachy.r_arm.gripper.send_goal_positions()
        elif arm == "l_arm":
            self.reachy.l_arm.gripper.goal_position = joint
            self.reachy.l_arm.gripper.send_goal_positions()

    def move_mobile_base(self, x, y, theta):
        self.reachy.mobile_base.set_goal_speed(vx=x, vy=y, vtheta=theta)
        self.reachy.mobile_base.send_speed_command()
    
    def move_antenna(self, position, arm):
        if arm == "r_arm":
            self.reachy.head.r_antenna.goal_position = position
        else:
            self.reachy.head.l_antenna.goal_position = position
        self.reachy.send_goal_positions()

    def stop(self):
        self.reachy.turn_off_smoothly()


