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





class G1_29_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        if not self.Unit_Test:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                "../../../../assets/g1/g1_body29_hand14.urdf", "../../../../assets/g1/"
            )
        else:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                "../../../../assets/g1/g1_body29_hand14.urdf", "../../../../assets/g1/"
            )

        self.mixed_jointsToLockIDs = [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_hand_thumb_0_joint",
            "left_hand_thumb_1_joint",
            "left_hand_thumb_2_joint",
            "left_hand_middle_0_joint",
            "left_hand_middle_1_joint",
            "left_hand_index_0_joint",
            "left_hand_index_1_joint",
            "right_hand_thumb_0_joint",
            "right_hand_thumb_1_joint",
            "right_hand_thumb_2_joint",
            "right_hand_index_0_joint",
            "right_hand_index_1_joint",
            "right_hand_middle_0_joint",
            "right_hand_middle_1_joint",
        ]

        self.reduced_robot = self.robot.buildReducedRobot(
            list_of_joints_to_lock=self.mixed_jointsToLockIDs,
            reference_configuration=np.array([0.0] * self.robot.model.nq),
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "L_ee",
                self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        self.reduced_robot.model.addFrame(
            pin.Frame(
                "R_ee",
                self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                pin.FrameType.OP_FRAME,
            )
        )

        # for i in range(self.reduced_robot.model.nframes):
        #     frame = self.reduced_robot.model.frames[i]
        #     frame_id = self.reduced_robot.model.getFrameId(frame.name)
        #     print(f"Frame ID: {frame_id}, Name: {frame.name}")

        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(
                        self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T
                    ),
                    cpin.log3(
                        self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T
                    ),
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)  # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )

        opts = {
            "ipopt": {"print_level": 0, "max_iter": 50, "tol": 1e-6},
            "print_time": False,  # print or not
            "calc_lam_p": False,  # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None

        if self.Visualization:
            # Initialize the Meshcat visualizer for visualization
            self.vis = MeshcatVisualizer(
                self.reduced_robot.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model,
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(
                True, frame_ids=[101, 102], axis_length=0.15, axis_width=5
            )
            self.vis.display(pin.neutral(self.reduced_robot.model))

            # Enable the display of end effector target frames with short axis lengths and greater width.
            frame_viz_names = ["L_ee_target", "R_ee_target"]
            FRAME_AXIS_POSITIONS = (
                np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
                )
                .astype(np.float32)
                .T
            )
            FRAME_AXIS_COLORS = (
                np.array(
                    [
                        [1, 0, 0],
                        [1, 0.6, 0],
                        [0, 1, 0],
                        [0.6, 1, 0],
                        [0, 0, 1],
                        [0, 0.6, 1],
                    ]
                )
                .astype(np.float32)
                .T
            )
            axis_length = 0.1
            axis_width = 10
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    # If the robot arm is not the same size as your arm :)
    def scale_arms(
        self,
        human_left_pose,
        human_right_pose,
        human_arm_length=0.60,
        robot_arm_length=0.75,
    ):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
    ):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        # left_wrist, right_wrist = self.scale_arms(left_wrist, right_wrist)
        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(
                left_wrist
            )  # for visualization
            self.vis.viewer["R_ee_target"].set_transform(
                right_wrist
            )  # for visualization

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)  # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            return sol_q, sol_tauff

        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            print(
                f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}"
            )
            if self.Visualization:
                self.vis.display(sol_q)  # for visualization

            # return sol_q, sol_tauff
            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)


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
    main()
    footpedal_subscriber = FootpedalSubscriber()

    arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=True)

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

    # rclpy.spin_once(footpedal_subscriber, timeout_sec=0.05)
    # initial_teleop_pose = np.array([footracker_2_pose[3:] = [
            #     tracker_2_pose[4],
            #     tracker_2_pose[3],
            #     tracker_2_pose[5],
            # ]tpedal_subscriber.x, footpedal_subscriber.y, footpedal_subscriber.z])
    # clutch_pressed = False
    # teleop_pose = np.array([footpedal_subscriber.x, footpedal_subscriber.y, footpedal_subscriber.z])
    # scaling_factor = 1

    # original_quat = pin.Quaternion(footpedal_subscriber.quat[0], footpedal_subscriber.quat[1], footpedal_subscriber.quat[2], footpedal_subscriber.quat[3])
    # original_rot = original_quat.toRotationMatrix()
    # # last_valid_pose = initial_teleop_pose  # Initialize with the initial pose

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

            print(tracker_2_pose[3:])

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

        arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)

        time.sleep(0.1)

    footpedal_subscriber.destroy_node()
    rclpy.shutdown()
