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


from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_                                 # idl
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
        self.subscription_footpedal_plus = self.create_subscription(
            Joy,  # Replace with the actual message type
            "/footpedals/cam_plus",
            self.listener_callback_footpedal_plus,
            10,
        )
        self.subscription_footpedal_minus = self.create_subscription(
            Joy,  # Replace with the actual message type
            "/footpedals/cam_minus",
            self.listener_callback_footpedal_minus,
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
        self.quat = np.array([0, 0, 0, 0])

    def listener_callback_footpedal(self, msg):
        # Check the state of the clutch in the 'buttons' array
        if len(msg.buttons) > 0:
            self.clutch_state = msg.buttons[
                0
            ]  # Assuming the clutch is the first button
            if self.clutch_state == 1:
                self.get_logger().info("Clutch is engaged.")
            else:
                self.get_logger().info("Clutch is disengaged.")
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
        print(self.x, self.y, self.z, self.quat)

    def listener_callback_footpedal_plus(self, msg):
        self.plus = msg.buttons[0]

    def listener_callback_footpedal_minus(self, msg):
        self.minus = msg.buttons[0]

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
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

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
            self.msg.motor_cmd[id].q  = self.all_motor_q[id]
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
                    lowstate.motor_state[id].q  = msg.motor_state[id].q
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
                arm_q_target     = self.q_target
                arm_tauff_target = self.tauff_target

            cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit = self.arm_velocity_limit)

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
        '''Set control target values q & tau of the left and right arm motors.'''
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        '''Return current dds mode machine.'''
        return self.lowstate_subscriber.Read().mode_machine
    
    def get_current_motor_q(self):
        '''Return current state q of all body motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointIndex])
    
    def get_current_dual_arm_q(self):
        '''Return current state q of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].q for id in G1_29_JointArmIndex])
    
    def get_current_dual_arm_dq(self):
        '''Return current state dq of the left and right arm motors.'''
        return np.array([self.lowstate_buffer.GetData().motor_state[id].dq for id in G1_29_JointArmIndex])
    
    def ctrl_dual_arm_go_home(self):
        '''Move both the left and right arms of the robot to their home position by setting the target joint angles (q) and torques (tau) to zero.'''
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

    def speed_gradual_max(self, t = 5.0):
        '''Parameter t is the total time required for arms velocity to gradually increase to its maximum value, in seconds. The default is 5.0.'''
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        '''set arms velocity to the maximum value immediately, instead of gradually increasing.'''
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
    return np.array([-x, -y, -z,  w])

def quat_multiply(q2, q1):
    """
    Hamilton product q = q2 * q1  (active, right-multiplication)
    Both q1, q2 = [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w2*x1 + x2*w1 + y2*z1 - z2*y1,
        w2*y1 - x2*z1 + y2*w1 + z2*x1,
        w2*z1 + x2*y1 - y2*x1 + z2*w1,
        w2*w1 - x2*x1 - y2*y1 - z2*z1
    ])

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
    n = x*x + y*y + z*z + w*w
    if n == 0.0:
        raise ValueError("Zero-norm quaternion is invalid")
    x, y, z, w = x/np.sqrt(n), y/np.sqrt(n), z/np.sqrt(n), w/np.sqrt(n)

    # --- compute matrix entries -------------------------------------------
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z

    R = np.array([[1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
                  [    2*(xy + wz), 1 - 2*(xx + zz),         2*(yz - wx)],
                  [    2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)]])
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


if __name__ == "__main__":
    from robot_arm_ik import G1_29_ArmIK
    import pinocchio as pin 

    main()

    footpedal_subscriber = FootpedalSubscriber()

    arm_ik = G1_29_ArmIK(Unit_Test = True, Visualization = False)
    g1arm = G1_29_ArmController()

    # initial positon
    L_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.2, 0.2]),
    )

    R_tf_target = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.2, 0.2]),
    )


    rotation_speed = 0.005
    noise_amplitude_translation = 0.001
    noise_amplitude_rotation = 0.1

    init_pose_tracker_1 = None
    init_pose_tracker_2 = None
    _delta_pos_l = np.zeros(3)
    _delta_rot_l = np.zeros(3)
    _delta_pos_r = np.zeros(3)
    _delta_rot_r = np.array([0, 0, 0, 1])
    curr_delta_pos_l = np.zeros(3)
    curr_delta_rot_l = np.zeros(3)
    curr_delta_pos_r = np.zeros(3)
    curr_delta_rot_r = np.array([0, 0, 0, 1])
    prev_delta_pos_l = None
    prev_delta_rot_l = None
    prev_delta_pos_r = None
    prev_delta_rot_r = None
    init_delta_flag = True
    track_clutch_button = False
    track_camera_button = False

    zero_pose_l = np.array([0.25, 0.2, 0.2])
    zero_pose_r = np.array([0.25, -0.2, 0.2])

    time.sleep(2)

    clutch_pressed = False
    camera_pressed = False

    scaling_factor = 0.5
    threshold = 0.15

    q_target = np.zeros(35)
    tauff_target = np.zeros(35)


    while True:
        try:
            clutch_pressed, camera_pressed = False, False
            rclpy.spin_once(footpedal_subscriber, timeout_sec=0.05)
            clutch_pressed = footpedal_subscriber.clutch_state

            tracker_1_pose = np.array([0, 0, 0, 0, 0, 0])
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

            print(tracker_2_pose)

            if init_pose_tracker_1 is None:
                init_pose_tracker_1 = tracker_1_pose
                init_pose_tracker_1[3:] = np.array([0, 0, 90])
            if init_pose_tracker_2 is None:
                init_pose_tracker_2 = tracker_2_pose
                # init_pose_tracker_2[3:] = np.array([0, 0, 90])

            # T_ref_inv_1 = np.linalg.inv(create_transformation(init_pose_tracker_1))
            T_ref_inv_2 = np.linalg.inv(create_transformation(init_pose_tracker_2))

            # tracker_2_pose[3:] = [
            #     tracker_2_pose[4],
            #     tracker_2_pose[3],
            #     tracker_2_pose[5],
            # ]
            tracker_2_pose[:3] = [
                tracker_2_pose[1],
                -tracker_2_pose[0],
                tracker_2_pose[2],
            ]

            if not clutch_pressed:
                curr_delta_pos_r, curr_delta_rot_r = (
                    tracker_2_pose[:3],
                    tracker_2_pose[3:],
                )
            if not camera_pressed:
                curr_delta_pos_l, curr_delta_rot_l = (
                    tracker_1_pose[:3],
                    tracker_1_pose[3:],
                )

            if track_clutch_button and not clutch_pressed:
                prev_delta_pos_r, prev_delta_rot_r = (
                    curr_delta_pos_r.copy(),
                    curr_delta_rot_r.copy(),
                )
                print("clutch accessed")
                track_clutch_button = False
            if track_camera_button and not camera_pressed:
                prev_delta_pos_l, prev_delta_rot_l = (
                    curr_delta_pos_l.copy(),
                    curr_delta_rot_l.copy(),
                )
                print("camera accessed")
                track_camera_button = False

            if init_delta_flag:
                prev_delta_pos_l, prev_delta_rot_l = (
                    curr_delta_pos_l.copy(),
                    curr_delta_rot_l.copy(),
                )
                prev_delta_pos_r, prev_delta_rot_r = (
                    curr_delta_pos_r.copy(),
                    curr_delta_rot_r.copy(),
                )
                init_delta_flag = False

            if clutch_pressed:
                curr_delta_pos_r, curr_delta_rot_r = (
                    prev_delta_pos_r.copy(),
                    prev_delta_rot_r.copy(),
                )
                track_clutch_button = True
            if camera_pressed:
                curr_delta_pos_l, curr_delta_rot_l = (
                    prev_delta_pos_l.copy(),
                    prev_delta_rot_l.copy(),
                )
                track_camera_button = True

            _delta_pos_l += curr_delta_pos_l - prev_delta_pos_l
            _delta_rot_l += curr_delta_rot_l - prev_delta_rot_l
            _delta_pos_r += curr_delta_pos_r - prev_delta_pos_r
            _delta_rot_r = quat_multiply(_delta_rot_r, quat_conjugate(quat_multiply(curr_delta_rot_r, quat_conjugate(prev_delta_rot_r))))

            prev_delta_pos_l, prev_delta_rot_l = (
                curr_delta_pos_l.copy(),
                curr_delta_rot_l.copy(),
            )
            prev_delta_pos_r, prev_delta_rot_r = (
                curr_delta_pos_r.copy(),
                curr_delta_rot_r.copy(),
            )

            # print(_delta_pos_l, _delta_rot_l, _delta_pos_r, _delta_rot_r)
            print()

            L_tf_target.translation = zero_pose_l + _delta_pos_l
            L_tf_target.rotation = Rotation.from_euler(
                "xyz", _delta_rot_l, degrees=True
            ).as_matrix()

            R_tf_target.translation = zero_pose_r + _delta_pos_r
            # R_tf_target.rotation = Rotation.from_euler(
            #     "xyz", _delta_rot_r, degrees=True
            # ).as_matrix()
            R_tf_target.rotation = Rotation.from_quat(_delta_rot_r).as_matrix()

        except Exception as e:
            print(f"Error in reading: {e}")

        current_lr_arm_q  = g1arm.get_current_dual_arm_q()
        current_lr_arm_dq = g1arm.get_current_dual_arm_dq()

        sol_q, sol_tauff = arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous, current_lr_arm_q, current_lr_arm_dq)

        g1arm.ctrl_dual_arm(sol_q, sol_tauff)

        time.sleep(0.01)
    

    footpedal_subscriber.destroy_node()
    rclpy.shutdown()
