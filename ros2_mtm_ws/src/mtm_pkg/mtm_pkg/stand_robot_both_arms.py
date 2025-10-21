import time
import sys
import threading

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
)

from robot_arm_ik import G1_29_ArmIK
import pinocchio as pin
from teleop.vive_tracker.origin_init import (
    create_transformation,
    transform_pose,
)
import traceback

from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
import numpy as np
from sensor_msgs.msg import Joy  # Change this import based on your actual message type

from geometry_msgs.msg import PoseStamped
from math import sin, cos, radians, degrees, atan2, asin
from rcm_control import *
from scipy.spatial.transform import Rotation as R

kPi = 3.141592654
kPi_2 = 1.57079632


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
        # subcribe RCM pose
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z

    def listener_callback_(self, msg):
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
            self.clutch_state = msg.buttons[0]
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


class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14  # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20  # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21  # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28  # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29  # NOTE: Weight


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


def point_to_rcm(v, up_hint=np.array([0, 1, 0])):
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
    z = np.cross(x, u)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.column_stack((x, y, z))  # columns are body axes in WORLD


def node_launch(args=None):
    rclpy.init(args=args)


class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02
        self.duration_ = 3.0
        self.counter_ = 0
        self.weight = 0.0
        self.weight_rate = 0.2
        # self.kp = 150.0
        # self.kd = 3.0
        # self.dq = 0.0

        self.kp_high = 350.0
        self.kd_high = 5.0
        self.kp_low  = 50.0
        self.kd_low  = 3.0
        self.kp_wrist = 50.0
        self.kd_wrist = 3.0

        self.tau_ff = 0.0
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = None
        self.first_update_low_state = False
        self.crc = CRC()
        self.done = False

        self.target_pos = [
            0.0,
            kPi_2,
            0.0,
            kPi_2,
            0.0,
            0.0,
            0.0,
            0.0,
            -kPi_2,
            0.0,
            kPi_2,
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
        ]

        self.arm_joints = [
            G1JointIndex.LeftShoulderPitch,
            G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftShoulderYaw,
            G1JointIndex.LeftElbow,
            G1JointIndex.LeftWristRoll,
            G1JointIndex.LeftWristPitch,
            G1JointIndex.LeftWristYaw,
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,
            G1JointIndex.RightElbow,
            G1JointIndex.RightWristRoll,
            G1JointIndex.RightWristPitch,
            G1JointIndex.RightWristYaw,
            G1JointIndex.WaistYaw,
            G1JointIndex.WaistRoll,
            G1JointIndex.WaistPitch,
        ]

        time.sleep(2)
        self.init_exception_flag = False

    def _is_wrist_joint(self, j):
        return j in [
            G1JointIndex.LeftWristRoll,
            G1JointIndex.LeftWristPitch,
            G1JointIndex.LeftWristYaw,
            G1JointIndex.RightWristRoll,
            G1JointIndex.RightWristPitch,
            G1JointIndex.RightWristYaw,
        ]

    def _is_weak_joint(self, j):
        # Same idea as robot_arm_both_arms.py: shoulders & elbows run "low"
        return j in [
            G1JointIndex.LeftShoulderPitch,
            G1JointIndex.LeftShoulderRoll,
            G1JointIndex.LeftShoulderYaw,
            G1JointIndex.LeftElbow,
            G1JointIndex.RightShoulderPitch,
            G1JointIndex.RightShoulderRoll,
            G1JointIndex.RightShoulderYaw,
            G1JointIndex.RightElbow,
        ]

    def _kp_kd_for(self, j):
        if self._is_wrist_joint(j):
            return self.kp_wrist, self.kd_wrist
        elif self._is_weak_joint(j):
            return self.kp_low, self.kd_low
        else:
            return self.kp_high, self.kd_high


    def Init(self):
        # create publisher #
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.first_update_low_state == False:
            time.sleep(1)

        if self.first_update_low_state == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.first_update_low_state == False:
            self.first_update_low_state = True

    def LowCmdWrite(self):
        self.time_ += self.control_dt_

        if self.time_ < self.duration_:
            # [Stage 1]: set robot to zero posture
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = (
                1  # 1:Enable arm_sdk, 0:Disable arm_sdk
            )
            for i, joint in enumerate(self.arm_joints):
                ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
                self.low_cmd.motor_cmd[joint].tau = 0.0
                self.low_cmd.motor_cmd[joint].q = (
                    1.0 - ratio
                ) * self.low_state.motor_state[joint].q
                self.low_cmd.motor_cmd[joint].dq = 0.0
                kp, kd = self._kp_kd_for(joint)
                self.low_cmd.motor_cmd[joint].kp = kp
                self.low_cmd.motor_cmd[joint].kd = kd

        elif self.time_ < self.duration_ * 3:
            try:
                if self.init_exception_flag == False:
                    footpedal_subscriber = FootpedalSubscriber()
                    footpedal_subscriber_ = FootpedalSubscriber_()
                    rcm_subscriber = RCMsuscriber()

                    executor = SingleThreadedExecutor()
                    executor.add_node(footpedal_subscriber)
                    executor.add_node(footpedal_subscriber_)
                    executor.add_node(rcm_subscriber)

                    # Spin ROS in a background thread (no blocking in your control loop)
                    spin_thread = threading.Thread(target=executor.spin, daemon=True)
                    spin_thread.start()

                    deadline = time.time() + 2.0
                    while time.time() < deadline:
                        time.sleep(0.01)


                    arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=False)
                    # initial positon
                    L_tf_target = pin.SE3(
                        pin.Quaternion(1, 0, 0, 0),
                        np.array([0.25, +0.2, 0.1]),
                    )

                    R_tf_target = pin.SE3(
                        pin.Quaternion(1, 0, 0, 0),
                        np.array([0.25, -0.2, 0.1]),
                    )

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
                    track_clutch_button = True

                    zero_pose_l = np.array([0.35, 0.2, 0.1])
                    zero_pose_r = np.array([0.35, -0.2, 0.1])

                    clutch_pressed = False

                    H1_prev_r= None
                    H1_prev_l = None

                    H1_init_r = None
                    H1_init_l = None

                    pos_err_r = None
                    rot_err_r = None

                    pos_err_l = None
                    rot_err_l = None

                    rcm_l = None
                    rcm_r = None

                    # smoother_R = SE3Smoother(tau_pos=0.02, tau_rot=0.02)   # tune as needed
                    # smoother_L = SE3Smoother(tau_pos=0.02, tau_rot=0.02)
                    prev_time = time.perf_counter()
                    i = 0
                    while True:
                        start_time = time.time()
                        clutch_pressed = False
                        clutch_pressed = footpedal_subscriber.clutch_state
                        try:
                            # Get pose data for the tracker device and format as a string
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

                            if np.allclose(
                                tracker_1_pose[:3], 0.0, atol=1e-8
                            ) and np.allclose(tracker_1_pose[3:], [1, 0, 0, 0], atol=1e-8):
                                # skip this loop; wait for a real sample
                                continue

                            if np.allclose(
                                tracker_2_pose[:3], 0.0, atol=1e-8
                            ) and np.allclose(tracker_2_pose[3:], [1, 0, 0, 0], atol=1e-8):
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

                            _delta_pos_r += (
                                curr_delta_pos_r - prev_delta_pos_r
                            ) * 0.4
                            # _delta_rot_r += curr_delta_rot_r - prev_delta_rot_r
                            _delta_rot_r = quat_multiply(
                                _delta_rot_r,
                                quat_conjugate(
                                    quat_multiply(
                                        curr_delta_rot_r,
                                        quat_conjugate(prev_delta_rot_r),
                                    )
                                ),
                            )
                        
                
                            # left arm
                            curr_tracker_pose_1 = tracker_1_pose[3:]
                            delta_tracker_pose_1 = quat_multiply(
                                curr_tracker_pose_1,
                                quat_conjugate(init_pose_tracker_1[3:]),
                            )

                            # right arm
                            curr_tracker_pose_2 = tracker_2_pose[3:]
                            delta_tracker_pose_2 = quat_multiply(
                                curr_tracker_pose_2,
                                quat_conjugate(init_pose_tracker_2[3:]),
                            )

                            _delta_pos_l += (
                                curr_delta_pos_l - prev_delta_pos_l
                            ) * 0.4
                            # _delta_rot_l += curr_delta_rot_l - prev_delta_rot_l
                            _delta_rot_l = quat_multiply(
                                _delta_rot_l,
                                quat_conjugate(
                                    quat_multiply(
                                        curr_delta_rot_l,
                                        quat_conjugate(prev_delta_rot_l),
                                    )
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
                            # print("Delta pos right arm:", _delta_pos_r)
                            # print("Delta pos left arm:", _delta_pos_l)
                            # rcm_l = np.array([0.60, -0.15, 0.0])
                            # rcm_r = np.array([0.60, 0.15, 0.0])
                            if rcm_l is None and rcm_r is None:
                                rcm_r = np.array([rcm_subscriber.x, rcm_subscriber.y, rcm_subscriber.z]) + np.array([0.05, 0.0, 0.0])
                                rcm_l = np.array([rcm_subscriber.x_, rcm_subscriber.y_, rcm_subscriber.z_]) + np.array([0.05, 0.0, 0.0])

                            # print("RCM right arm:", rcm_r)
                            # print("RCM left arm:", rcm_l)
                            t0 = time.time()
                            # print(f"before IK timer: {t0-start_time:.4f}s")
                            ##########################Right arm############################################
                            if H1_init_r is None:
                                H1_init_r = np.eye(4)
                                H1_init_r[:3, 3] = zero_pose_r
                                rot_rcm_r = point_to_rcm(rcm_r - zero_pose_r)
                                # H1_init[:3, :3] = Rotation.from_quat( rot_rcm ).as_matrix()  # initially it's identity (mtm reset)
                                H1_init_r[:3, :3] = rot_rcm_r  # point to rcm
                                Ps2_init_r, _, _, _ = compute_shaft2_pose(
                                    H1_init_r, rcm_r, return_intermediate=False
                                )
                                zero_pos_ps2_r = Ps2_init_r[:3, 3]

                            Ps2_gt_r = np.eye(4)
                            zero_rot_ps2_r = Ps2_init_r[:3, :3]
                            # Ps2_gt_r[:3, :3] =  np.linalg.inv(Rotation.from_quat( delta_tracker_pose_2).as_matrix()) @ zero_rot_ps2_r  # use global rotation
                            Ps2_gt_r[:3, :3] = zero_rot_ps2_r @ np.linalg.inv(
                                Rotation.from_quat(delta_tracker_pose_2).as_matrix()
                            )  # use global rotation

                            Ps2_gt_r[:3, 3] = zero_pos_ps2_r + _delta_pos_r

                            H1_est_r = invert_shaft2_pose_angle_limits(
                                Ps2_gt_r, rcm_r, H1_prev=H1_prev_r
                            )
                            # if i == 0:
                            #     print("H1_est_r", H1_est_r, "rcm_r", rcm_r, "Ps2_gt_r", Ps2_gt_r)

                            if H1_est_r[:3, 3][0] >= rcm_r[0]:
                                # print("H1_est_r", H1_est_r, "rcm_r", rcm_r)
                                raise ValueError(
                                    "target pose is beyond RCM, which is not valid."
                                )

                            H1_prev_r = H1_est_r.copy()
                            H1_est_r[:3, 3] += np.array(
                                [-0.1, 0.0, 0.0]
                            )  # TODO: use the new offset

                            R_tf_target.translation = H1_est_r[:3, 3]

                            R_tf_target.rotation = H1_est_r[:3, :3]

                            ##########################Left arm############################################

                            if H1_init_l is None:
                                H1_init_l = np.eye(4)
                                H1_init_l[:3, 3] = zero_pose_l
                                rot_rcm_l = point_to_rcm(rcm_l - zero_pose_l)
                                H1_init_l[:3, :3] = rot_rcm_l  # point to rcm
                                Ps2_init_l, _, _, _ = compute_shaft2_pose(
                                    H1_init_l, rcm_l, return_intermediate=False
                                )
                                zero_pos_ps2_l = Ps2_init_l[:3, 3]

                            Ps2_gt_l = np.eye(4)
                            zero_rot_ps2_l = Ps2_init_l[:3, :3]
                            # Ps2_gt_l[:3, :3] =  np.linalg.inv(Rotation.from_quat( delta_tracker_pose_1).as_matrix()) @ zero_rot_ps2_l  # use global rotation
                            Ps2_gt_l[:3, :3] = zero_rot_ps2_l @ np.linalg.inv(
                                Rotation.from_quat(delta_tracker_pose_1).as_matrix()
                            )  # use global rotation

                            Ps2_gt_l[:3, 3] = zero_pos_ps2_l + _delta_pos_l

                            H1_est_l = invert_shaft2_pose_angle_limits(
                                Ps2_gt_l, rcm_l, H1_prev=H1_prev_l
                            )
                            # if i == 0:
                            #     print("H1_est_l", H1_est_l, "rcm_l", rcm_l, "Ps2_gt_l", Ps2_gt_l)
    
                            if H1_est_l[:3, 3][0] >= rcm_l[0]:
                                raise ValueError(
                                    "target pose is beyond RCM, which is not valid."
                                )

                            H1_prev_l = H1_est_l.copy()
                            H1_est_l[:3, 3] += np.array([-0.1, 0.0, 0.0])

                            L_tf_target.translation = H1_est_l[:3, 3]

                            L_tf_target.rotation = H1_est_l[:3, :3]

                            print(f" IK process time: {time.time()-t0:.4f}s")

                            # except Exception as e:
                            #     print("error in controller", e)

                            rotation_l = R.from_quat(_delta_rot_l)
                            _rot_l = rotation_l.as_euler("xyz", degrees=True)
                            rotation_r = R.from_quat(_delta_rot_r)
                            _rot_r = rotation_r.as_euler("xyz", degrees=True)

                            current_lr_arm_q = self.get_current_dual_arm_q()[:-3]
                            current_lr_arm_dq = self.get_current_dual_arm_dq()[:-3]
                            current_lr_arm_tau_est = self.get_current_dual_arm_tau_est()[
                                :-3
                            ]

                            now = time.perf_counter()
                            dt = now - prev_time
                            prev_time = now

                            # L_tf_target = L_tf_target
                            # R_tf_target = smoother_R.step(R_tf_target, dt)


                            sol_q, sol_tauff = arm_ik.solve_ik(
                                L_tf_target.homogeneous,
                                R_tf_target.homogeneous,
                                current_lr_arm_q,
                                current_lr_arm_dq,
                                # current_lr_arm_tau_est,
                            )

                            T_L_curr, T_R_curr = (
                                arm_ik.get_current_ee_poses()
                            )  # get current actual end-effector poses

                            # reprojection error check:
                            # pos_err_l, rot_err_l = arm_ik.get_reprojection_err(T_L_curr, L_tf_target.homogeneous)
                            if pos_err_r is None or rot_err_r is None:
                                pos_err_r, rot_err_r = 0.0, 0.0
                                t1p, t2p = 0.0, 0.0
                            else:
                                pos_err_r, rot_err_r = get_reprojection_err(
                                    T_R_curr, R_tf_target.homogeneous
                                )
                                # add a check of the angles
                                reproj_Ps2, _, t1p, t2p = compute_shaft2_pose(
                                    R_tf_target.homogeneous,
                                    rcm_r,
                                    return_intermediate=True,
                                )
                            # print(f"Left reprojection error: {pos_err_l:.4f} m, {rot_err_l:.4f} deg")
                            # print(f"Right reprojection error: {pos_err_r:.4f} m, {rot_err_r:.4f} deg")
                            if (
                                pos_err_r > 0.15
                                or t1p > 0.90 * np.pi / 2
                                or t2p > 0.90 * np.pi / 2
                            ):
                                print("Reprojection error too high, skipping IK solve.")
                                continue

                            if pos_err_l is None or rot_err_l is None:
                                pos_err_l, rot_err_l = 0.0, 0.0
                                t1p, t2p = 0.0, 0.0
                            else:
                                pos_err_l, rot_err_l = get_reprojection_err(
                                    T_L_curr, L_tf_target.homogeneous
                                )
                                reproj_Ps2, _, t1p, t2p = compute_shaft2_pose(
                                    L_tf_target.homogeneous,
                                    rcm_l,
                                    return_intermediate=True,
                                )
                            # print(f"Left reprojection error: {pos_err_l:.4f} m, {rot_err_l:.4f} deg")
                            if (
                                pos_err_l > 0.15
                                or t1p > 0.90 * np.pi / 2
                                or t2p > 0.90 * np.pi / 2
                            ):
                                print("Reprojection error too high, skipping IK solve.")
                                continue

                            # print(sol_q)
                            self.target_pos = sol_q
                            self.target_pos = np.append(self.target_pos, [0, 0, 0])
                            sol_tauff = np.append(sol_tauff, [0, 0, 0])

                            i += 1
                            if i > 100:
                                # [Stage 3]: set robot back to zero posture
                                cliped_arm_q_target = self.clip_arm_q_target(
                                    self.target_pos, velocity_limit=30.0
                                )
                            else:
                                cliped_arm_q_target = self.clip_arm_q_target(
                                    self.target_pos, velocity_limit=2.0
                                )
                            # print(cliped_arm_q_target)
                            # print("Right arm pos:", H1_est_r[:3,3], "rot (rpy):", _rot_r)
                            for j, joint in enumerate(self.arm_joints):
                                ratio = np.clip(
                                    (self.time_ - self.duration_) / (self.duration_ * 2),
                                    0.0,
                                    1.0,
                                )

                                self.low_cmd.motor_cmd[joint].tau = sol_tauff[j]
                                self.low_cmd.motor_cmd[joint].q = cliped_arm_q_target[j]
                                self.low_cmd.motor_cmd[joint].dq = 0.0
                                kp, kd = self._kp_kd_for(joint)
                                self.low_cmd.motor_cmd[joint].kp = kp
                                self.low_cmd.motor_cmd[joint].kd = kd

                            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                            self.arm_sdk_publisher.Write(self.low_cmd)

                            current_time = time.time()
                            all_t_elapsed = current_time - start_time
                            sleep_time = max(0, (self.control_dt_ - all_t_elapsed))
                            # time.sleep(sleep_time)
                        except Exception as e:
                            # Print just the error line and type
                            exc_type, exc_obj, exc_tb = sys.exc_info()
                            fname = exc_tb.tb_frame.f_code.co_filename
                            line_no = exc_tb.tb_lineno
                            print(f"⚠️ Error: {e}")
                            print(f"Occurred in file: {fname}, line: {line_no}")

                            # Optionally, print full traceback for debugging
                            print("Full Traceback:")
                            traceback.print_exc()
            except KeyboardInterrupt:
                if self.init_exception_flag == False:
                    self.time_ = 9.0
                    self.init_exception_flag = True

        elif self.time_ >= self.duration_ * 3:
            # [Stage 3]: set robot back to zero posture
            for i, joint in enumerate(self.arm_joints):
                ratio = np.clip(
                    (self.time_ - self.duration_ * 3) / (self.duration_ * 3), 0.0, 1.0
                )
                self.low_cmd.motor_cmd[joint].tau = 0.0
                self.low_cmd.motor_cmd[joint].q = (
                    1.0 - ratio
                ) * self.low_state.motor_state[joint].q
                self.low_cmd.motor_cmd[joint].dq = 0.0
                kp, kd = self._kp_kd_for(joint)
                self.low_cmd.motor_cmd[joint].kp = kp
                self.low_cmd.motor_cmd[joint].kd = kd

        elif self.time_ > self.duration_ * 7:
            # [Stage 4]: release arm_sdk
            for i, joint in enumerate(self.arm_joints):
                ratio = np.clip(
                    (self.time_ - self.duration_ * 6) / (self.duration_), 0.0, 1.0
                )
                self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = (
                    1 - ratio
                )  # 1:Enable arm_sdk, 0:Disable arm_sdk

        else:
            self.done = True

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

    def get_current_dual_arm_q(self):
        return np.array([self.low_state.motor_state[id].q for id in self.arm_joints])

    def get_current_dual_arm_dq(self):
        return np.array([self.low_state.motor_state[id].dq for id in self.arm_joints])

    def get_current_dual_arm_tau_est(self):
        return np.array(
            [self.low_state.motor_state[id].tau_est for id in self.arm_joints]
        )

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt_)
        cliped_arm_q_target = current_q + delta / max(motion_scale, 1.0)
        return cliped_arm_q_target


if __name__ == "__main__":

    print(
        "WARNING: Please ensure there are no obstacles around the robot while running this example."
    )
    input("Press Enter to continue...")

    node_launch()

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    while True:
        time.sleep(1)
        if custom.done:
            print("Done!")
            sys.exit(-1)
