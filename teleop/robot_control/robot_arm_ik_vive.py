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

parent2_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent2_dir)

from teleop.utils.weighted_moving_filter import WeightedMovingFilter
from scipy.spatial.transform import Rotation

from teleop.vive_tracker.track import ViveTrackerModule
from IPython import embed
from teleop.vive_tracker.fairmotion_vis import camera
from teleop.vive_tracker.fairmotion_ops import conversions, math as fairmotion_math
from teleop.vive_tracker.origin_init import (
    euler_to_matrix,
    matrix_to_euler,
    create_transformation,
    decompose_transformation,
    transform_pose,
)

import hid

VID = 0x06C2  # Replace with your VID
PID = 0x0036  # Replace with your PID


class G1_29_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        if not self.Unit_Test:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                "../assets/g1/g1_inspire_gen4.urdf", "../assets/g1/"
            )
        else:
            self.robot = pin.RobotWrapper.BuildFromURDF(
                "../../assets/g1/g1_inspire_gen4.urdf", "../../assets/g1/"
            )  # for test

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
            "left_wrist_pitch_joint",
            "left_thumb_1_joint",
            "left_thumb_2_joint",
            "left_thumb_3_joint",
            "left_thumb_4_joint",
            "left_middle_1_joint",
            "left_index_1_joint",
            "left_ring_1_joint",
            "left_little_1_joint",
            "left_middle_2_joint",
            "left_index_2_joint",
            "left_ring_2_joint",
            "left_little_2_joint",
            "right_thumb_1_joint",
            "right_thumb_2_joint",
            "right_thumb_3_joint",
            "right_thumb_4_joint",
            "right_middle_1_joint",
            "right_index_1_joint",
            "right_ring_1_joint",
            "right_little_1_joint",
            "right_middle_2_joint",
            "right_index_2_joint",
            "right_ring_2_joint",
            "right_little_2_joint",
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
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 13)
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
                    ),
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
        current_lr_arm_motor_tau_est=None,
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

            # # right arm
            # J = pin.computeFrameJacobian(
            #     self.reduced_robot.model,
            #     self.reduced_robot.data,
            #     current_lr_arm_motor_q,
            #     self.R_hand_id,
            # )
            # J_r = J[:, 6:]
            # F_desired = np.array([150, 150, 150, 0, 0, 0])
            # F_ee = np.linalg.pinv(J_r.T) @ current_lr_arm_motor_tau_est[6:]
            # K_p = np.diag([0, 0, 0.0001, 0, 0, 0])
            # F_control = K_p @ (F_desired - F_ee)
            # # Compute the torque using the Jacobian transpose
            # tau_impedance = J_r.T @ F_control
            # sol_tauff[6:] += tau_impedance

            # # left arm
            # J = pin.computeFrameJacobian(
            #     self.reduced_robot.model,
            #     self.reduced_robot.data,
            #     current_lr_arm_motor_q,
            #     self.L_hand_id,
            # )
            # J_l = J[:, :6]
            # F_desired = np.array([150, 150, 150, 0, 0, 0])
            # F_ee = np.linalg.pinv(J_l.T) @ current_lr_arm_motor_tau_est[:6]
            # K_p = np.diag([0, 0, 0.0001, 0, 0, 0])
            # # Compute impedance force
            # F_control = K_p @ (F_desired - F_ee)
            # # Compute the torque using the Jacobian transpose
            # tau_impedance = J_l.T @ F_control
            # sol_tauff[:6] += tau_impedance

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


if __name__ == "__main__":
    arm_ik = G1_29_ArmIK(Unit_Test=True, Visualization=True)

    # Open the foot pedal
    device = hid.device()
    device.open(VID, PID)

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
    _delta_rot_r = np.zeros(3)
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

    v_tracker = ViveTrackerModule()
    v_tracker.print_discovered_objects()
    tracker_1 = v_tracker.devices["tracker_1"]
    tracker_2 = v_tracker.devices["tracker_2"]

    clutch_pressed = False
    camera_pressed = False

    scaling_factor = 0.5
    threshold = 0.15

    while True:
        try:
            # Read device input
            clutch_pressed, camera_pressed = False, False
            input_data = 255 - device.read(64)[1]
            clutch_pressed = input_data in {1, 3}
            camera_pressed = input_data in {2, 3}

            # Get pose data for the tracker device and format as a string
            tracker_1_pose = np.array([val for val in tracker_1.get_pose_euler()])
            tracker_2_pose = np.array([val for val in tracker_2.get_pose_euler()])

            if init_pose_tracker_1 is None:
                init_pose_tracker_1 = tracker_1_pose
                init_pose_tracker_1[3:] = np.array([0, 0, 90])
            if init_pose_tracker_2 is None:
                init_pose_tracker_2 = tracker_2_pose
                init_pose_tracker_2[3:] = np.array([0, 0, 90])

            T_ref_inv_1 = np.linalg.inv(create_transformation(init_pose_tracker_1))
            T_ref_inv_2 = np.linalg.inv(create_transformation(init_pose_tracker_2))

            tracker_1_pose = np.array(transform_pose(tracker_1_pose, T_ref_inv_1))
            tracker_2_pose = np.array(transform_pose(tracker_2_pose, T_ref_inv_2))

            tracker_1_pose[3:] = [
                tracker_1_pose[5],
                -tracker_1_pose[3],
                -tracker_1_pose[4],
            ]
            tracker_1_pose[:3] = [
                -tracker_1_pose[1],
                -tracker_1_pose[2],
                tracker_1_pose[0],
            ]

            tracker_2_pose[3:] = [
                -tracker_2_pose[5],
                -tracker_2_pose[3],
                tracker_2_pose[4],
            ]
            tracker_2_pose[:3] = [
                -tracker_2_pose[1],
                -tracker_2_pose[2],
                tracker_2_pose[0],
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

            # if not init_delta_flag:
            #     if camera_pressed:
            #         threshold = 0.75 / scaling_factor
            #     else:
            #         threshold = 0.75
            #     if np.any(np.abs(curr_delta_pos_r) > threshold) or np.any(
            #         np.abs(curr_delta_pos_l) > threshold
            #     ):
            #         curr_delta_pos_l, curr_delta_rot_l = (
            #             prev_delta_pos_l.copy(),
            #             prev_delta_rot_l.copy(),
            #         )
            #         curr_delta_pos_r, curr_delta_rot_r = (
            #             prev_delta_pos_r.copy(),
            #             prev_delta_rot_r.copy(),
            #         )

            # if camera_pressed:
            #     curr_delta_pos_r[2] = max(curr_delta_pos_r[2], -0.75/scaling_factor)
            #     curr_delta_pos_l[2] = max(curr_delta_pos_l[2], -0.75/scaling_factor)
            # else:
            #     curr_delta_pos_r[2] = max(curr_delta_pos_r[2], -0.75)
            #     curr_delta_pos_l[2] = max(curr_delta_pos_l[2], -0.75)
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

            # if camera_pressed:
            #     print("Camera pressed")
            #     _delta_pos_l += (curr_delta_pos_l - prev_delta_pos_l) * scaling_factor
            #     _delta_rot_l += (curr_delta_rot_l - prev_delta_rot_l) * scaling_factor
            #     _delta_pos_r += (curr_delta_pos_r - prev_delta_pos_r) * scaling_factor
            #     _delta_rot_r += (curr_delta_rot_r - prev_delta_rot_r) * scaling_factor
            # else:
            #     _delta_pos_l += curr_delta_pos_l - prev_delta_pos_l
            #     _delta_rot_l += curr_delta_rot_l - prev_delta_rot_l
            #     _delta_pos_r += curr_delta_pos_r - prev_delta_pos_r
            #     _delta_rot_r += curr_delta_rot_r - prev_delta_rot_r

            _delta_pos_l += curr_delta_pos_l - prev_delta_pos_l
            _delta_rot_l += curr_delta_rot_l - prev_delta_rot_l
            _delta_pos_r += curr_delta_pos_r - prev_delta_pos_r
            _delta_rot_r += curr_delta_rot_r - prev_delta_rot_r

            _delta_rot_r = np.array([60.0, -15.0, 0.0])

            prev_delta_pos_l, prev_delta_rot_l = (
                curr_delta_pos_l.copy(),
                curr_delta_rot_l.copy(),
            )
            prev_delta_pos_r, prev_delta_rot_r = (
                curr_delta_pos_r.copy(),
                curr_delta_rot_r.copy(),
            )

            print(_delta_pos_l, _delta_rot_l, _delta_pos_r, _delta_rot_r)

            L_tf_target.translation = zero_pose_l + _delta_pos_l
            L_tf_target.rotation = Rotation.from_euler(
                "xyz", _delta_rot_l, degrees=True
            ).as_matrix()

            R_tf_target.translation = zero_pose_r + _delta_pos_r
            R_tf_target.rotation = Rotation.from_euler(
                "xyz", _delta_rot_r, degrees=True
            ).as_matrix()

        except Exception as e:
            print(f"Error in reading: {e}")

        arm_ik.solve_ik(L_tf_target.homogeneous, R_tf_target.homogeneous)

        time.sleep(0.1)
