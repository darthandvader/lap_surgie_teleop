from turtle import left
import numpy as np
import threading
import time
from enum import IntEnum

import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
import time
from pinocchio import casadi as cpin
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import os
import sys
import math
from math import sin, cos, radians, degrees, atan2, asin

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy  # Change this import based on your actual message type
from rclpy.executors import MultiThreadedExecutor


from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)  # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

kTopicLowCommand = "rt/lowcmd"
kTopicLowState = "rt/lowstate"
G1_29_Num_Motors = 35


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy  # Change this import based on your actual message type

from geometry_msgs.msg import PoseStamped

from scipy.spatial.transform import Rotation

parent2_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent2_dir)
from utils.weighted_moving_filter import WeightedMovingFilter
from scipy.spatial.transform import Rotation as R

from rcm_control import *





class RCMsuscriber(Node):
    def __init__(self):
        super().__init__("rcm_subscriber")
        self.subscription = self.create_subscription(
            PoseStamped,  # Replace with the actual message type
            "/RCM_pose_right",
            self.listener_callback,
            10,
        )
        
        self.subscription  # prevent unused variable warning
        self.x = 0.6
        self.y = -0.01
        self.z = -0.035

        self.subscription_ = self.create_subscription(
            PoseStamped,  # Replace with the actual message type
            "/RCM_pose_left",
            self.listener_callback_,
            10,
        )

        self.subscription  # prevent unused variable warning
        self.x_ = 0.6
        self.y_ = 0.19
        self.z_ = -0.035

    def listener_callback(self, msg):
        # Check the state of the RCM in the 'buttons' array
        # if len(msg.buttons) > 0:
        #     self.rcm_state = msg.buttons[0]  # Assuming the RCM is the first button
        #     if self.rcm_state == 1:
        #         self.get_logger().info("RCM is engaged.")
        #     else:
        #         self.get_logger().info("RCM is disengaged.")
        # else:
        #     self.get_logger().warn("No buttons data available.")    

         # subcribe RCM pose
        self.x = msg.pose.position.x  
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z

    def listener_callback_(self, msg):
        # # Check the state of the RCM in the 'buttons' array
        # if len(msg.buttons) > 0:
        #     self.rcm_state = msg.buttons[0]  # Assuming the RCM is the first button
        #     if self.rcm_state == 1:
        #         self.get_logger().info("RCM is engaged.")
        #     else:
        #         self.get_logger().info("RCM is disengaged.")
        # else:
        #     self.get_logger().warn("No buttons data available.")    

         # subcribe RCM pose
        self.x_ = msg.pose.position.x  
        self.y_ = msg.pose.position.y
        self.z_ = msg.pose.position.z





class FootpedalSubscriber(Node):
    def __init__(self):
        super().__init__("footpedal_subscriber")
        # Replace '/footpedals/clutch' with the actual topic name
        self.subscription_footpedal = self.create_subscription(
            Joy,  # Replace with the actual message type
            "/footpedals/clutch",
            self.listener_callback_footpedal,
            10,
        )
        self.subscription_controller = self.create_subscription(
            PoseStamped,  # Correct message type
            "/MTMR/measured_cp",
            self.listener_callback,
            10,
        )
        self.subscription_footpedal  # prevent unused variable warning
        self.subscription_controller
        self.clutch_state = 0
        self.x = 0
        self.y = 0
        self.z = 0
        self.plus = 0
        self.minus = 0
        self.quat = np.array([1, 0, 0, 0])

    def listener_callback_footpedal(self, msg):
        # Check the state of the clutch in the 'buttons' array
        if len(msg.buttons) > 0:
            self.clutch_state = msg.buttons[
                0
            ]
        else:
            self.get_logger().warn("No buttons data available.")

    def listener_callback(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z
        self.quat = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ]
        )
        
class FootpedalSubscriber_(Node):
    def __init__(self):
        super().__init__("footpedal_subscriber_")
        self.subscription_controller_ = self.create_subscription(
            PoseStamped,  # Correct message type
            "/MTML/measured_cp",
            self.listener_callback_,
            10,
        )
        self.subscription_controller_
        self.x_ = 0
        self.y_ = 0
        self.z_ = 0
        self.quat_ = np.array([1, 0, 0, 0])

    def listener_callback_(self, msg):
        self.x_ = msg.pose.position.x
        self.y_ = msg.pose.position.y
        self.z_ = msg.pose.position.z
        self.quat_ = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ]
        )
        # print(self.x_, self.y_, self.z_, self.quat)


class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None


class G1_29_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(G1_29_Num_Motors)]


class DataBuffer:
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()

    def GetData(self):
        with self.lock:
            return self.data

    def SetData(self, data):
        with self.lock:
            self.data = data


class G1_29_ArmController:
    def __init__(self):
        print("Initialize G1_29_ArmController...")
        self.q_target = np.zeros(14)
        self.tauff_target = np.zeros(14)

        self.kp_high = 300.0
        self.kd_high = 5.0
        self.kp_low = 60.0
        self.kd_low = 3.0
        self.kp_wrist = 55.0
        self.kd_wrist = 3.0

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        # initialize lowcmd publisher and lowstate subscriber
        ChannelFactoryInitialize(0)
        self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand, LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, LowState_)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        # initialize subscribe thread
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.01)
            print("[G1_29_ArmController] Waiting to subscribe dds...")

        # initialize hg's lowcmd msg
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        print(f"Current all body motor state q:\n{self.all_motor_q} \n")
        print(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        print("Lock all joints except two arms...\n")

        arm_indices = set(member.value for member in G1_29_JointArmIndex)
        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            if id.value in arm_indices:
                if self._Is_wrist_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_wrist
                    self.msg.motor_cmd[id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
            else:
                if self._Is_weak_motor(id):
                    self.msg.motor_cmd[id].kp = self.kp_low
                    self.msg.motor_cmd[id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[id].kp = self.kp_high
                    self.msg.motor_cmd[id].kd = self.kd_high
            self.msg.motor_cmd[id].q = self.all_motor_q[id]
        print("Lock OK!\n")

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        print("Initialize G1_29_ArmController OK!\n")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()
                for id in range(G1_29_Num_Motors):
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target

    def _ctrl_motor_state(self):
        while True:
            start_time = time.time()

            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target

            cliped_arm_q_target = self.clip_arm_q_target(
                arm_q_target, velocity_limit=self.arm_velocity_limit
            )

            for idx, id in enumerate(G1_29_JointArmIndex):
                self.msg.motor_cmd[id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[id].dq = 0
                self.msg.motor_cmd[id].tau = arm_tauff_target[idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)
            # print(f"arm_velocity_limit:{self.arm_velocity_limit}")
            # print(f"sleep_time:{sleep_time}")

    def ctrl_dual_arm(self, q_target, tauff_target):
        """Set control target values q & tau of the left and right arm motors."""
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        """Return current dds mode machine."""
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        """Return current state q of all body motors."""
        return np.array(
            [
                self.lowstate_buffer.GetData().motor_state[id].q
                for id in G1_29_JointIndex
            ]
        )

    def get_current_dual_arm_q(self):
        """Return current state q of the left and right arm motors."""
        return np.array(
            [
                self.lowstate_buffer.GetData().motor_state[id].q
                for id in G1_29_JointArmIndex
            ]
        )

    def get_current_dual_arm_dq(self):
        """Return current state dq of the left and right arm motors."""
        return np.array(
            [
                self.lowstate_buffer.GetData().motor_state[id].dq
                for id in G1_29_JointArmIndex
            ]
        )

    def ctrl_dual_arm_go_home(self):
        """Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero."""
        print("[G1_29_ArmController] ctrl_dual_arm_go_home start...")
        with self.ctrl_lock:
            self.q_target = np.zeros(14)
            # self.tauff_target = np.zeros(14)
        tolerance = 0.05  # Tolerance threshold for joint angles to determine "close to zero", can be adjusted based on your motor's precision requirements
        while True:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                print("[G1_29_ArmController] both arms have reached the home position.")
                break
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        """Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0."""
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        """set arms velocity to the maximum value immediately, instead of gradually increasing."""
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            G1_29_JointIndex.kLeftAnklePitch.value,
            G1_29_JointIndex.kRightAnklePitch.value,
            # Left arm
            G1_29_JointIndex.kLeftShoulderPitch.value,
            G1_29_JointIndex.kLeftShoulderRoll.value,
            G1_29_JointIndex.kLeftShoulderYaw.value,
            G1_29_JointIndex.kLeftElbow.value,
            # Right arm
            G1_29_JointIndex.kRightShoulderPitch.value,
            G1_29_JointIndex.kRightShoulderRoll.value,
            G1_29_JointIndex.kRightShoulderYaw.value,
            G1_29_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            G1_29_JointIndex.kLeftWristRoll.value,
            G1_29_JointIndex.kLeftWristPitch.value,
            G1_29_JointIndex.kLeftWristyaw.value,
            G1_29_JointIndex.kRightWristRoll.value,
            G1_29_JointIndex.kRightWristPitch.value,
            G1_29_JointIndex.kRightWristYaw.value,
        ]
        return motor_index.value in wrist_motors


class G1_29_JointArmIndex(IntEnum):
    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28


class G1_29_JointIndex(IntEnum):
    # Left leg
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    # Right leg
    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kWaistYaw = 12
    kWaistRoll = 13
    kWaistPitch = 14

    # Left arm
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitch = 20
    kLeftWristyaw = 21

    # Right arm
    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitch = 27
    kRightWristYaw = 28

    # not used
    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34


def main(args=None):
    rclpy.init(args=args)


# ---------- quaternion helpers ---------------------------------------------
def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_conjugate(q):
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quat_multiply(q2, q1):
    """
    Hamilton product q = q2 * q1  (active, right-multiplication)
    Both q1, q2 = [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
        ]
    )


def create_transformation(pose):
    """
    Create a 4x4 homogeneous transformation matrix from a numpy array
    pose = [x, y, z, roll, pitch, yaw].
    """
    T = np.eye(4)
    # T[0:3, 0:3] = euler_to_matrix(*pose[3:])
    T[0:3, 0:3] = quat_to_matrix(*pose[3:])  # Convert quat to matrix
    T[0:3, 3] = pose[:3]
    return T


def matrix_to_euler(R):
    """
    Convert a 3x3 rotation matrix into roll, pitch, yaw (in degrees).
    This assumes the same convention (Rz * Ry * Rx).
    """
    # pitch = asin(-R[2,0])
    # yaw   = atan2(R[1,0], R[0,0])
    # roll  = atan2(R[2,1], R[2,2])

    # Be mindful of potential numeric edge cases if R[2,0] is out of [-1,1].
    sy = -R[2, 0]
    sy = np.clip(sy, -1.0, 1.0)  # to avoid numeric issues if outside [-1,1]
    pitch_r = np.arcsin(sy)
    yaw_r = np.arctan2(R[1, 0], R[0, 0])
    roll_r = np.arctan2(R[2, 1], R[2, 2])

    roll_deg = degrees(roll_r)
    pitch_deg = degrees(pitch_r)
    yaw_deg = degrees(yaw_r)
    return roll_deg, pitch_deg, yaw_deg


def quat_to_matrix(x, y, z, w):
    """
    Convert quaternion → 3x3 rotation matrix.

    Parameters
    ----------
    q : array-like of length 4
        Quaternion in the form [x, y, z, w].

    Returns
    -------
    R : (3, 3) ndarray
        Rotation matrix.
    """
    # --- normalize (important if the quat is from sensor math) -------------
    n = x * x + y * y + z * z + w * w
    if n == 0.0:
        raise ValueError("Zero-norm quaternion is invalid")
    x, y, z, w = x / np.sqrt(n), y / np.sqrt(n), z / np.sqrt(n), w / np.sqrt(n)

    # --- compute matrix entries -------------------------------------------
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )
    return R


def euler_to_matrix(roll, pitch, yaw):
    """
    Convert roll, pitch, yaw (in degrees) into a 3x3 rotation matrix.
    Assumes roll -> pitch -> yaw about x, y, z respectively in that order.
    """
    # Convert degrees to radians
    roll_r = radians(roll)
    pitch_r = radians(pitch)
    yaw_r = radians(yaw)

    # Rotation about X-axis (roll)
    Rx = np.array(
        [[1, 0, 0], [0, cos(roll_r), -sin(roll_r)], [0, sin(roll_r), cos(roll_r)]]
    )
    # Rotation about Y-axis (pitch)
    Ry = np.array(
        [[cos(pitch_r), 0, sin(pitch_r)], [0, 1, 0], [-sin(pitch_r), 0, cos(pitch_r)]]
    )
    # Rotation about Z-axis (yaw)
    Rz = np.array(
        [[cos(yaw_r), -sin(yaw_r), 0], [sin(yaw_r), cos(yaw_r), 0], [0, 0, 1]]
    )

    # Typically, the total rotation for (roll, pitch, yaw) = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def scale_quat(q, s):
    """Return q with its rotation angle scaled by s (s<1 dampens, s>1 amplifies)."""
    q = q / np.linalg.norm(q)  # just to be safe
    v = q[:3]
    w = q[3]
    angle = 2 * np.arctan2(np.linalg.norm(v), w)

    if np.isclose(angle, 0.0):
        return np.array([0, 0, 0, 1])  # nothing to scale

    axis = v / np.linalg.norm(v)
    a2 = 0.5 * s * angle  # half‑angle after scaling
    return np.hstack((axis * np.sin(a2), np.cos(a2)))

def get_reprojection_err(T_curr, T_target):
    """
    Compute reprojection error between current and target poses.
    T_*: 4x4 homogeneous (world frame).
    Returns (pos_err_m, rot_err_deg).
    """
    # translation error (in world)
    pos_err = np.linalg.norm(T_curr[:3, 3] - T_target[:3, 3])

    # rotation error via relative rotation
    R_err = T_target[:3, :3].T @ T_curr[:3, :3]
    cos_theta = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    rot_err_deg = np.degrees(np.arccos(cos_theta))
    return pos_err, rot_err_deg


def point_to_rcm(v, up_hint=np.array([0,1,0])):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("align_x_to: direction vector is zero-length")
    x = v / n

    u = up_hint / np.linalg.norm(up_hint)
    # Avoid near-parallel up vectors
    if abs(np.dot(x, u)) > 0.99:
        u = np.array([0, 1, 0.0])

    # Build a right-handed ONB with x fixed
    z = np.cross(x, u); z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))   # columns are body axes in WORLD

def _min_jerk_s_and_ds(tau):
    # tau in [0,1]
    s  = 10*tau**3 - 15*tau**4 + 6*tau**5
    ds = 30*tau**2 - 60*tau**3 + 30*tau**4
    return s, ds


if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK
    import pinocchio as pin
    from rclpy.executors import SingleThreadedExecutor

    main()

    footpedal_subscriber = FootpedalSubscriber()
    footpedal_subscriber_ = FootpedalSubscriber_()
    rcm_subscriber = RCMsuscriber()

    arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=False)
    g1arm = G1_29_ArmController()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.2, 0.1]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.2, 0.1]),
    )


    last_cmd_T_r = np.eye(4)  # last feasible RIGHT SE3 we commanded
    last_cmd_T_r[:3,:3] = R_tf_target.rotation
    last_cmd_T_r[:3, 3] = R_tf_target.translation

    last_cmd_T_l = np.eye(4)  # last feasible LEFT SE3 we commanded
    last_cmd_T_l[:3,:3] = L_tf_target.rotation
    last_cmd_T_l[:3, 3] = L_tf_target.translation


    rotation_speed = 0.005
    noise_amplitude_translation = 0.001
    noise_amplitude_rotation = 0.1

    init_pose_tracker_1 = None
    init_pose_tracker_2 = None
    _delta_pos_l = np.zeros(3)
    _delta_rot_l = np.array([0, 0, 0, 1], dtype=np.float64)
    _delta_pos_r = np.zeros(3)
    _delta_rot_r = np.array([0, 0, 0, 1], dtype=np.float64)
    curr_delta_pos_l = np.zeros(3)
    curr_delta_rot_l = np.array([0, 0, 0, 1], dtype=np.float64)
    curr_delta_pos_r = np.zeros(3)
    curr_delta_rot_r = np.array([0, 0, 0, 1], dtype=np.float64)
    prev_delta_pos_l = None
    prev_delta_rot_l = None
    prev_delta_pos_r = None
    prev_delta_rot_r = None
    init_delta_flag = True
    track_clutch_button = True
    track_camera_button = False

    clip_r = False # for clipping poses outside of motion range.
    clip_l = False
    val_delta_pos_r = None
    val_delta_pos_l = None

    zero_pose_l = np.array([0.35, 0.2, 0.1])
    zero_pose_r = np.array([0.35, -0.2, 0.1])

    time.sleep(2)

    clutch_pressed = False

    scaling_factor = 0.5
    threshold = 0.15

    q_target = np.zeros(35)
    tauff_target = np.zeros(35)

    H1_prev_r= None
    H1_prev_l = None

    H1_init_r = None
    H1_init_l = None

    pos_err_r = None
    rot_err_r = None

    pos_err_l = None
    rot_err_l = None

    executor = SingleThreadedExecutor()
    executor.add_node(footpedal_subscriber)
    executor.add_node(footpedal_subscriber_)
    executor.add_node(rcm_subscriber)
    # executor = MultiThreadedExecutor(num_threads=2)
    # executor.add_node(footpedal_subscriber)
    # executor.add_node(footpedal_subscriber_)
    # executor.add_node(rcm_subscriber)
    # TODO: if rcm is none, use the last valid rcm

    # Spin ROS in a background thread (no blocking in your control loop)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # for i in range(10):
    #     rclpy.spin_once(footpedal_subscriber, timeout_sec=0.05)
    #     rclpy.spin_once(footpedal_subscriber_, timeout_sec=0.05)
    #     rclpy.spin_once(rcm_subscriber, timeout_sec=0.05)
    #     # executor.spin_once(timeout_sec=0.01)
    #     i +=1

    deadline = time.time() + 2.0
    while time.time() < deadline:
        time.sleep(0.01)

    rcm_l = None
    rcm_r = None

    smoother_R = SE3Smoother(tau_pos=0.02, tau_rot=0.02)   # tune as needed
    smoother_L = SE3Smoother(tau_pos=0.02, tau_rot=0.02)
    prev_time = time.perf_counter()

    while True:

        start = time.time()
        clutch_pressed = False
        # for i in range(10):
        # rclpy.spin_once(footpedal_subscriber, timeout_sec=0.02)
        # rclpy.spin_once(footpedal_subscriber_, timeout_sec=0.02)
        # rclpy.spin_once(rcm_subscriber, timeout_sec=0.05)
        # executor.spin_once(timeout_sec=0.0)
        clutch_pressed = footpedal_subscriber.clutch_state


        # print("Clutch state:", clutch_pressed)

        tracker_1_pose = np.array(
            [
                footpedal_subscriber_.x_,
                footpedal_subscriber_.y_,
                footpedal_subscriber_.z_,
                footpedal_subscriber_.quat_[0],
                -footpedal_subscriber_.quat_[1],
                -footpedal_subscriber_.quat_[2],
                footpedal_subscriber_.quat_[3],
            ]
        )
        tracker_2_pose = np.array(
            [
                footpedal_subscriber.x,
                footpedal_subscriber.y,
                footpedal_subscriber.z,
                footpedal_subscriber.quat[0],
                -footpedal_subscriber.quat[1],
                -footpedal_subscriber.quat[2],
                footpedal_subscriber.quat[3],
            ]
        )

        if init_pose_tracker_1 is None:
            init_pose_tracker_1 = tracker_1_pose
        if init_pose_tracker_2 is None:
            init_pose_tracker_2 = tracker_2_pose
            # init_pose_tracker_2[3:] = np.array([0, 0, 90])

        T_ref_inv_1 = np.linalg.inv(create_transformation(init_pose_tracker_1))
        T_ref_inv_2 = np.linalg.inv(create_transformation(init_pose_tracker_2))

        tracker_1_pose[:3] = [
            tracker_1_pose[1],
            -tracker_1_pose[0],
            tracker_1_pose[2],
        ]

        tracker_2_pose[:3] = [
            tracker_2_pose[1],
            -tracker_2_pose[0],
            tracker_2_pose[2],
        ]

        if np.allclose(tracker_1_pose[:3], 0.0, atol=1e-8) and np.allclose(tracker_1_pose[3:], [1,0,0,0], atol=1e-8):
                # skip this loop; wait for a real sample
                continue
        
        if np.allclose(tracker_2_pose[:3], 0.0, atol=1e-8) and np.allclose(tracker_2_pose[3:], [1,0,0,0], atol=1e-8):
            continue
        
        # print("Tracker 1 pose:", tracker_1_pose)
        # print("Tracker 2 pose:", tracker_2_pose)

        if not clutch_pressed:
        
            curr_delta_pos_r, curr_delta_rot_r = (
                tracker_2_pose[:3],
                tracker_2_pose[3:],
            )
            curr_delta_pos_l, curr_delta_rot_l = (
                tracker_1_pose[:3],
                tracker_1_pose[3:],
            )

        # for clutch RELEASE
        if track_clutch_button and not clutch_pressed:
            prev_delta_pos_r, prev_delta_rot_r = (
                curr_delta_pos_r.copy(),
                curr_delta_rot_r.copy(),
            )
            prev_delta_pos_l, prev_delta_rot_l = (
                curr_delta_pos_l.copy(),
                curr_delta_rot_l.copy(),
            )
            print("clutch released")
            track_clutch_button = False

        # if init_delta_flag:
        #     prev_delta_pos_l, prev_delta_rot_l = (
        #         curr_delta_pos_l.copy(),
        #         curr_delta_rot_l.copy(),
        #     )
        #     prev_delta_pos_r, prev_delta_rot_r = (
        #         curr_delta_pos_r.copy(),
        #         curr_delta_rot_r.copy(),
        #     )
        #     init_delta_flag = False

        # for clutch pressed

        if clutch_pressed:
            curr_delta_pos_r, curr_delta_rot_r = (
                prev_delta_pos_r.copy(),
                prev_delta_rot_r.copy(),
            )
            curr_delta_pos_l, curr_delta_rot_l = (
                prev_delta_pos_l.copy(),
                prev_delta_rot_l.copy(),
            )
            track_clutch_button = True

        # print("Current delta pos r:", curr_delta_pos_r, "Current delta rot r:", curr_delta_rot_r)
        # print("Previous delta pos r:", prev_delta_pos_r, "Previous delta rot r:", prev_delta_rot_r)

        # print("Current delta pos l:", curr_delta_pos_l, "Current delta rot l:", curr_delta_rot_l)
        # print("Previous delta pos l:", prev_delta_pos_l, "Previous delta rot l:", prev_delta_rot_l)

        ##########################Right arm############################################
        if clip_r and val_delta_pos_r is not None:
            _delta_pos_r = val_delta_pos_r
            clip_r = False
        else:
            _delta_pos_r += (curr_delta_pos_r - prev_delta_pos_r)*0.4


        # _delta_rot_r += curr_delta_rot_r - prev_delta_rot_r
        _delta_rot_r = quat_multiply(
            _delta_rot_r,
            quat_conjugate(
                quat_multiply(curr_delta_rot_r, quat_conjugate(prev_delta_rot_r))
            ),
        )

      
        curr_tracker_pose_1 = tracker_1_pose[3:]
        delta_tracker_pose_1 = quat_multiply(curr_tracker_pose_1, quat_conjugate(init_pose_tracker_1[3:]))



        curr_tracker_pose_2 = tracker_2_pose[3:]
        delta_tracker_pose_2 = quat_multiply(curr_tracker_pose_2, quat_conjugate(init_pose_tracker_2[3:]))


        ##########################Left arm############################################
        if clip_l and val_delta_pos_l is not None:
            _delta_pos_l = val_delta_pos_l
            clip_l = False
        else:
            _delta_pos_l += (curr_delta_pos_l - prev_delta_pos_l)*0.4

        _delta_rot_l = quat_multiply(
            _delta_rot_l,
            quat_conjugate(
                quat_multiply(curr_delta_rot_l, quat_conjugate(prev_delta_rot_l))
            ),
        )

        prev_delta_pos_l, prev_delta_rot_l = (
            curr_delta_pos_l.copy(),
            curr_delta_rot_l.copy(),
        )
        prev_delta_pos_r, prev_delta_rot_r = (
            curr_delta_pos_r.copy(),
            curr_delta_rot_r.copy(),
        )

        if rcm_l is None and rcm_r is None:
            # rcm_r = np.array([rcm_subscriber.x, rcm_subscriber.y, rcm_subscriber.z]) + np.array([-0.1, 0.0, 0.0])
            # rcm_l = np.array([rcm_subscriber.x_, rcm_subscriber.y_, rcm_subscriber.z_]) + np.array([-0.1, 0.0, 0.0])
            rcm_r = np.array([rcm_subscriber.x, rcm_subscriber.y, rcm_subscriber.z]) 
            rcm_l = np.array([rcm_subscriber.x_, rcm_subscriber.y_, rcm_subscriber.z_]) 
        
        
        # rcm_r = np.array([0.71, -0.2, 0.13])
        # rcm_l = np.array([0.71, 0.1, 0.13])
        t0 = time.time()
        print(f"before IK timer: {t0-start:.4f}s")


        ##########################Right arm############################################
        if H1_init_r is None:
            H1_init_r = np.eye(4)
            H1_init_r[:3, 3] = zero_pose_r
            rot_rcm_r = point_to_rcm(rcm_r - zero_pose_r)
            # H1_init[:3, :3] = Rotation.from_quat( rot_rcm ).as_matrix()  # initially it's identity (mtm reset)
            H1_init_r[:3, :3] = rot_rcm_r # point to rcm
            Ps2_init_r, _, _, _ = compute_shaft2_pose(H1_init_r, rcm_r, return_intermediate=False)
            zero_pos_ps2_r = Ps2_init_r[:3, 3]

        Ps2_gt_r = np.eye(4)
        zero_rot_ps2_r = Ps2_init_r[:3, :3]
        # Ps2_gt_r[:3, :3] =  np.linalg.inv(Rotation.from_quat( delta_tracker_pose_2).as_matrix()) @ zero_rot_ps2_r  # use global rotation
        Ps2_gt_r[:3, :3] =   zero_rot_ps2_r @ np.linalg.inv(Rotation.from_quat( delta_tracker_pose_2).as_matrix()) # use global rotation

        Ps2_gt_r[:3, 3] = zero_pos_ps2_r + _delta_pos_r

        H1_est_r = invert_shaft2_pose_angle_limits(Ps2_gt_r, rcm_r, H1_prev=H1_prev_r)

        if H1_est_r[:3, 3][0] >= rcm_r[0]:
            print("Estimated H1 pose is beyond RCM.", H1_est_r[:3, 3][0], rcm_r[0])
            continue

        H1_prev_r = H1_est_r.copy()
        H1_est_r[:3, 3] += np.array([-0.1, 0.0, 0.0])  

        R_tf_target.translation = H1_est_r[
            :3, 3
        ] 

        R_tf_target.rotation = H1_est_r[:3, :3]
        # R_tf_target.translation = zero_pose_r + _delta_pos_r
        # R_tf_target.rotation = Rotation.from_quat(_delta_rot_r).as_matrix()


        ##########################Left arm############################################

        if H1_init_l is None:
            H1_init_l = np.eye(4)
            H1_init_l[:3, 3] = zero_pose_l
            rot_rcm_l = point_to_rcm(rcm_l - zero_pose_l)
            H1_init_l[:3, :3] = rot_rcm_l # point to rcm
            Ps2_init_l, _, _, _ = compute_shaft2_pose(H1_init_l, rcm_l, return_intermediate=False)
            zero_pos_ps2_l = Ps2_init_l[:3, 3]

        Ps2_gt_l = np.eye(4)
        zero_rot_ps2_l = Ps2_init_l[:3, :3]
        # Ps2_gt_l[:3, :3] =  np.linalg.inv(Rotation.from_quat( delta_tracker_pose_1).as_matrix()) @ zero_rot_ps2_l  # use global rotation
        Ps2_gt_l[:3, :3] =   zero_rot_ps2_l @ np.linalg.inv(Rotation.from_quat( delta_tracker_pose_1).as_matrix()) # use global rotation

        Ps2_gt_l[:3, 3] = zero_pos_ps2_l + _delta_pos_l

        H1_est_l = invert_shaft2_pose_angle_limits(Ps2_gt_l, rcm_l, H1_prev=H1_prev_l)

        if H1_est_l[:3, 3][0] >= rcm_l[0]:
            continue

        H1_prev_l = H1_est_l.copy()
        H1_est_l[:3, 3] += np.array([-0.1, 0.0, 0.0])

        L_tf_target.translation = H1_est_l[:3, 3]

        L_tf_target.rotation = H1_est_l[:3, :3]

        print(f" IK process time: {time.time()-t0:.4f}s")
        # L_tf_target.rotation = Rotation.from_quat(_delta_rot_l).as_matrix()

        if arm_ik.Visualization and arm_ik.vis is not None:
        
            arm_ik.vis.viewer["Ps2_gt"].set_transform(Ps2_gt_r)
            T_rcm_r = np.eye(4)
            T_rcm_r[:3, 3] = rcm_r
            arm_ik.vis.viewer["RCM_point"].set_transform(T_rcm_r)

            arm_ik.vis.viewer["Ps2_gt_l"].set_transform(Ps2_gt_l)
            T_rcm_l = np.eye(4)
            T_rcm_l[:3, 3] = rcm_l
            arm_ik.vis.viewer["RCM_point_l"].set_transform(T_rcm_l)

        current_lr_arm_q = g1arm.get_current_dual_arm_q()
        current_lr_arm_dq = g1arm.get_current_dual_arm_dq()

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now


        # IK on smoothed targets
        sol_q, sol_tauff = arm_ik.solve_ik(
            L_tf_target.homogeneous,
            R_tf_target.homogeneous,
            current_lr_arm_q,
            current_lr_arm_dq,
        )

        
        T_L_curr, T_R_curr = arm_ik.get_current_ee_poses()   # get current actual end-effector poses

        # reprojection error check:
        # pos_err_l, rot_err_l = arm_ik.get_reprojection_err(T_L_curr, L_tf_target.homogeneous)
        # TODO: when reprojection error is too high, clip the delta pose
        # TODO: figure out a way then the solved poses are beyond the rcm

        if pos_err_r is None or rot_err_r is None:
            pos_err_r, rot_err_r = 0.0, 0.0
            t1p, t2p = 0.0, 0.0
        else:
            pos_err_r, rot_err_r = get_reprojection_err(T_R_curr, R_tf_target.homogeneous)
            # add a check of the angles
            reproj_Ps2, _, t1p, t2p = compute_shaft2_pose(R_tf_target.homogeneous, rcm_r, return_intermediate = True)
        # print(f"Left reprojection error: {pos_err_l:.4f} m, {rot_err_l:.4f} deg")
        # print(f"Right reprojection error: {pos_err_r:.4f} m, {rot_err_r:.4f} deg")
        if pos_err_r >  0.1 or t1p > 0.90 * np.pi /2 or t2p > 0.90 * np.pi /2:
            print("Reprojection error too high, clipping delta.")
            clip_r = True
            continue
        
        val_delta_pos_r = _delta_pos_r.copy()

        if pos_err_l is None or rot_err_l is None:
            pos_err_l, rot_err_l = 0.0, 0.0
            t1p, t2p = 0.0, 0.0
        else:
            pos_err_l, rot_err_l = get_reprojection_err(T_L_curr, L_tf_target.homogeneous)
            reproj_Ps2, _, t1p, t2p = compute_shaft2_pose(L_tf_target.homogeneous, rcm_l, return_intermediate = True)
        # print(f"Left reprojection error: {pos_err_l:.4f} m, {rot_err_l:.4f} deg")
        if pos_err_l >  0.1 or t1p > 0.90 * np.pi /2 or t2p > 0.90 * np.pi /2:
            print("Reprojection error too high, clipping delta.")
            clip_l = True
            continue

        val_delta_pos_l = _delta_pos_l.copy()

        g1arm.ctrl_dual_arm(sol_q, sol_tauff)

        print("total time:", time.time() - start)
        time.sleep(0.005)

    footpedal_subscriber.destroy_node()
    rclpy.shutdown()