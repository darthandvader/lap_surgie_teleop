import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco
import numpy as np
import transforms3d as tf3d
from tf2_ros import TransformBroadcaster
from concurrent.futures import Future
from scipy.spatial.transform import Rotation, Slerp


def wait_for_message(node: Node, topic, msg_type, timeout=None):
    future = Future()

    def callback(msg):
        if not future.done():
            future.set_result(msg)

    subscription = node.create_subscription(msg_type, topic, callback, 10)
    try:
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    finally:
        node.destroy_subscription(subscription)
    return future.result() if future.done() else None


class ArucoPosePublisher(Node):
    def __init__(self):
        super().__init__('aruco_pose_publisher')

        # Bridge and storage for incoming data
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.last_image_msg = None
        self.last_image = None
        self.time_stamp = None

        # Subscriptions just to store data
        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            1
        )
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Camera intrinsics: wait a bit longer and fall back if needed ---
        color_cam_info = wait_for_message(
            self, '/camera/camera/color/camera_info', CameraInfo, timeout=5
        )
        if color_cam_info is not None:
            self.camera_matrix = np.array(color_cam_info.k).reshape((3, 3))
            self.dist_coeffs = np.array(color_cam_info.d)
            self.get_logger().info("Got /camera/camera/color/camera_info, intrinsics set.")
        else:
            self.get_logger().error(
                "No /camera/camera/color/camera_info received within timeout. "
                "Using dummy intrinsics so overlay can still publish."
            )
            self.camera_matrix = np.eye(3)
            self.dist_coeffs = np.zeros(5)

        # debug overlay publisher
        self.show_overlay = True
        self.overlay_pub = self.create_publisher(Image, '/debug/rcm_overlay', 1)

        # Publishers for RCM poses with respect to camera frame
        self.rcm_left_pub = self.create_publisher(PoseStamped, '/RCM_pose_left', 1)
        self.rcm_right_pub = self.create_publisher(PoseStamped, '/RCM_pose_right', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/marker_pose', 1)

        # ------------------ RCM OFFSETS (your logic kept) ------------------ #
        self.marker_T_right_rcm = np.eye(4)
        self.marker_T_right_rcm[:3, :3] = Rotation.from_euler('x', -30, degrees=True).as_matrix()

        self.offset1 = np.eye(4)
        self.offset1[:3, 3] = np.array([0.0, -0.06, 0.0])
        self.offset2 = np.eye(4)
        self.offset2[:3, 3] = np.array([0.0, -0.05, 0.0])

        self.marker_T_left_rcm = np.eye(4)
        self.marker_T_left_rcm[:3, :3] = Rotation.from_euler('x', -30, degrees=True).as_matrix()
        # ------------------------------------------------------------------- #

        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.marker_length = 0.08  # meters

        # Use two different marker IDs: LEFT = 1, RIGHT = 0
        self.left_marker_id = 1
        self.right_marker_id = 0

        # caches for poses and transforms
        self.last_rcm_left = None
        self.last_rcm_right = None
        self.last_cam_T_rcm_left = None
        self.last_cam_T_rcm_right = None

        self.max_translation_jump = 0.06   # meters
        # Detection timer (e.g. 10 Hz)
        self.create_timer(1.0 / 10.0, self.detection_callback)

        self.get_logger().info(
            "ArucoPosePublisher started.\n"
            f"Using marker ID {self.left_marker_id} for LEFT RCM and "
            f"marker ID {self.right_marker_id} for RIGHT RCM.\n"
            "When a marker is not detected, last pose will be re-published."
        )
    
    def clamp_pose(self, prev_T, new_T):
        if prev_T is None:
            return True
        dt = np.linalg.norm(prev_T[:3, 3] - new_T[:3, 3])
        return dt < self.max_translation_jump

    def moving_avg_pose(self, prev_T, new_T, alpha=0.1):
        if prev_T is None:
            return new_T

        avg_T = np.eye(4)

        # translation EMA
        avg_T[:3, 3] = alpha * new_T[:3, 3] + (1.0 - alpha) * prev_T[:3, 3]

        # rotations
        q_prev_wxyz = tf3d.quaternions.mat2quat(prev_T[:3, :3])   # [w,x,y,z]
        q_new_wxyz  = tf3d.quaternions.mat2quat(new_T[:3, :3])    # [w,x,y,z]

        # convert to SciPy format [x,y,z,w]
        q_prev_xyzw = np.array([q_prev_wxyz[1], q_prev_wxyz[2], q_prev_wxyz[3], q_prev_wxyz[0]])
        q_new_xyzw  = np.array([q_new_wxyz[1],  q_new_wxyz[2],  q_new_wxyz[3],  q_new_wxyz[0]])

        key_times = [0, 1]
        key_rots = Rotation.from_quat([q_prev_xyzw, q_new_xyzw])

        slerp = Slerp(key_times, key_rots)
        q_avg_xyzw = slerp([alpha])[0].as_quat()

        # convert back to [w,x,y,z] for transforms3d
        q_avg_wxyz = np.array([q_avg_xyzw[3], q_avg_xyzw[0], q_avg_xyzw[1], q_avg_xyzw[2]])

        avg_T[:3, :3] = tf3d.quaternions.quat2mat(q_avg_wxyz)
        return avg_T

    def _draw_rcm_axes_on_image(self, img, rcm_R_cam, rcm_t_cam, axis_len=0.05, label='RCM'):
        """
        Draw a 3D axes triad for the RCM pose onto the image.
        Colors (BGR): X=red, Y=green, Z=blue
        """
        obj_pts = np.float32([
            [0, 0, 0],                  # origin
            [axis_len, 0, 0],           # X
            [0, axis_len, 0],           # Y
            [0, 0, axis_len],           # Z
        ]).reshape(-1, 3)

        rvec, _ = cv2.Rodrigues(rcm_R_cam)
        tvec = rcm_t_cam.reshape(3, 1)

        img_pts, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        img_pts = img_pts.reshape(-1, 2).astype(int)

        O, X, Y, Z = img_pts
        cv2.circle(img, O, 3, (255, 255, 255), -1)
        cv2.line(img, O, X, (0, 0, 255), 2)   # X red
        cv2.line(img, O, Y, (0, 255, 0), 2)   # Y green
        cv2.line(img, O, Z, (255, 0, 0), 2)   # Z blue
        cv2.putText(
            img, label, (O[0] + 5, O[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA
        )

    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV bridge error: {e}')
            return
        self.last_image_msg = msg
        self.last_image = cv_img
        self.time_stamp = msg.header.stamp

    def detection_callback(self):
        if (self.last_image is None or
            self.camera_matrix is None or
            self.dist_coeffs is None or
            self.last_image_msg is None):
            self.get_logger().warn(
                f"Early return: last_image={self.last_image is None}, "
                f"K={self.camera_matrix is None}, D={self.dist_coeffs is None}, "
                f"last_image_msg={self.last_image_msg is None}"
            )
            return

        cam_frame = getattr(self.last_image_msg.header, 'frame_id', 'camera_frame')
        stamp = self.time_stamp if self.time_stamp is not None else self.get_clock().now().to_msg()

        try:
            gray = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f'cv2.cvtColor failed: {e}')
            return
        overlay = self.last_image.copy()

        # --- detect markers ---
        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )

        detected_left = False
        detected_right = False

        if ids is not None and len(ids) > 0:
            self.get_logger().debug(f'Detected marker ids: {ids.flatten()}')

            try:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length, self.camera_matrix, self.dist_coeffs
                )
            except Exception as e:
                self.get_logger().error(f'estimatePoseSingleMarkers failed: {e}')
                return

            flat_ids = [int(i[0]) for i in ids]
            for idx, marker_id in enumerate(flat_ids):
                rvec = rvecs[idx][0]
                tvec = tvecs[idx][0]
                R_cam_marker, _ = cv2.Rodrigues(rvec)

                cam_T_marker = np.eye(4)
                cam_T_marker[:3, :3] = R_cam_marker
                cam_T_marker[:3, 3] = tvec

                # publish raw marker pose if you want
                q_marker = tf3d.quaternions.mat2quat(R_cam_marker)
                marker_pose_msg = PoseStamped()
                marker_pose_msg.header.stamp = stamp
                marker_pose_msg.header.frame_id = cam_frame
                marker_pose_msg.pose.position.x = float(tvec[0])
                marker_pose_msg.pose.position.y = float(tvec[1])
                marker_pose_msg.pose.position.z = float(tvec[2])
                marker_pose_msg.pose.orientation.x = float(q_marker[1])
                marker_pose_msg.pose.orientation.y = float(q_marker[2])
                marker_pose_msg.pose.orientation.z = float(q_marker[3])
                marker_pose_msg.pose.orientation.w = float(q_marker[0])
                # self.pose_pub.publish(marker_pose_msg)

                # broadcast TF for marker
                t_marker = TransformStamped()
                t_marker.header.stamp = stamp
                t_marker.header.frame_id = cam_frame
                t_marker.child_frame_id = f'marker_{marker_id}'
                t_marker.transform.translation.x = float(tvec[0])
                t_marker.transform.translation.y = float(tvec[1])
                t_marker.transform.translation.z = float(tvec[2])
                t_marker.transform.rotation.x = float(q_marker[1])
                t_marker.transform.rotation.y = float(q_marker[2])
                t_marker.transform.rotation.z = float(q_marker[3])
                t_marker.transform.rotation.w = float(q_marker[0])
                self.tf_broadcaster.sendTransform(t_marker)

                # ----- LEFT RCM (marker id = self.left_marker_id) -----
                if marker_id == self.left_marker_id:
                    # cam_T_rcm_left = cam_T_marker @ self.offset1 @ self.marker_T_left_rcm @ self.offset2
                    raw_left = cam_T_marker @ self.offset1 @ self.marker_T_left_rcm @ self.offset2

                    # --- safety clamp ---
                    # accepted = self.clamp_pose(self.last_cam_T_rcm_left, raw_left)
                    # if not accepted:
                    #     self.get_logger().warn("LEFT RCM jump detected → clamped")
                    #     raw_left = self.last_cam_T_rcm_left
                        
                    # --- moving average filter ---
                    cam_T_rcm_left = self.moving_avg_pose(self.last_cam_T_rcm_left, raw_left, alpha=0.1)

                    # --- publish ---
                    rcm_msg = PoseStamped()
                    rcm_msg.header.stamp = stamp
                    rcm_msg.header.frame_id = cam_frame
                    rcm_msg.pose.position.x = float(cam_T_rcm_left[0, 3])
                    rcm_msg.pose.position.y = float(cam_T_rcm_left[1, 3])
                    rcm_msg.pose.position.z = float(cam_T_rcm_left[2, 3])
                    q_rcm = tf3d.quaternions.mat2quat(cam_T_rcm_left[:3, :3])
                    rcm_msg.pose.orientation.x = float(q_rcm[1])
                    rcm_msg.pose.orientation.y = float(q_rcm[2])
                    rcm_msg.pose.orientation.z = float(q_rcm[3])
                    rcm_msg.pose.orientation.w = float(q_rcm[0])

                    self.rcm_left_pub.publish(rcm_msg)
                    self.last_rcm_left = rcm_msg
                    self.last_cam_T_rcm_left = cam_T_rcm_left.copy()
                    detected_left = True

                    if self.show_overlay:
                        try:
                            self._draw_rcm_axes_on_image(
                                overlay,
                                cam_T_rcm_left[:3, :3],
                                cam_T_rcm_left[:3, 3],
                                axis_len=0.05,
                                label='RCM_L'
                            )
                        except Exception as e:
                            self.get_logger().error(f'Projection error for RCM_L: {e}')

                # ----- RIGHT RCM (marker id = self.right_marker_id) -----
                if marker_id == self.right_marker_id:
                    # cam_T_rcm_right = cam_T_marker @ self.offset1 @ self.marker_T_right_rcm @ self.offset2
                    raw_right = cam_T_marker @ self.offset1 @ self.marker_T_right_rcm @ self.offset2
                    
                    # --- safety clamp ---
                    # accepted = self.clamp_pose(self.last_cam_T_rcm_right, raw_right)
                    # if not accepted:
                    #     self.get_logger().warn("RIGHT RCM jump detected → clamped")
                    #     raw_right = self.last_cam_T_rcm_right

                    # --- moving average filter ---
                    cam_T_rcm_right = self.moving_avg_pose(self.last_cam_T_rcm_right, raw_right, alpha=0.1)

                    # --- publish ---   
                    rcm_msg = PoseStamped()
                    rcm_msg.header.stamp = stamp
                    rcm_msg.header.frame_id = cam_frame
                    rcm_msg.pose.position.x = float(cam_T_rcm_right[0, 3])
                    rcm_msg.pose.position.y = float(cam_T_rcm_right[1, 3])
                    rcm_msg.pose.position.z = float(cam_T_rcm_right[2, 3])
                    q_rcm = tf3d.quaternions.mat2quat(cam_T_rcm_right[:3, :3])
                    rcm_msg.pose.orientation.x = float(q_rcm[1])
                    rcm_msg.pose.orientation.y = float(q_rcm[2])
                    rcm_msg.pose.orientation.z = float(q_rcm[3])
                    rcm_msg.pose.orientation.w = float(q_rcm[0])

                    self.rcm_right_pub.publish(rcm_msg)
                    self.last_rcm_right = rcm_msg
                    self.last_cam_T_rcm_right = cam_T_rcm_right.copy()
                    detected_right = True

                    if self.show_overlay:
                        try:
                            self._draw_rcm_axes_on_image(
                                overlay,
                                cam_T_rcm_right[:3, :3],
                                cam_T_rcm_right[:3, 3],
                                axis_len=0.05,
                                label='RCM_R'
                            )
                        except Exception as e:
                            self.get_logger().error(f'Projection error for RCM_R: {e}')

        # --- if not detected this frame, publish / draw last cached pose ---
        stamp_now = self.time_stamp if self.time_stamp is not None else self.get_clock().now().to_msg()

        if not detected_left and self.last_rcm_left is not None:
            self.last_rcm_left.header.stamp = stamp_now
            self.rcm_left_pub.publish(self.last_rcm_left)
            if self.show_overlay and self.last_cam_T_rcm_left is not None:
                try:
                    self._draw_rcm_axes_on_image(
                        overlay,
                        self.last_cam_T_rcm_left[:3, :3],
                        self.last_cam_T_rcm_left[:3, 3],
                        axis_len=0.05,
                        label='RCM_L'
                    )
                except Exception as e:
                    self.get_logger().error(f'Frozen RCM_L overlay error: {e}')

        if not detected_right and self.last_rcm_right is not None:
            self.last_rcm_right.header.stamp = stamp_now
            self.rcm_right_pub.publish(self.last_rcm_right)
            if self.show_overlay and self.last_cam_T_rcm_right is not None:
                try:
                    self._draw_rcm_axes_on_image(
                        overlay,
                        self.last_cam_T_rcm_right[:3, :3],
                        self.last_cam_T_rcm_right[:3, 3],
                        axis_len=0.05,
                        label='RCM_R'
                    )
                except Exception as e:
                    self.get_logger().error(f'Frozen RCM_R overlay error: {e}')

        # --- publish overlay image ---
        self._publish_overlay(overlay)

    def _publish_overlay(self, overlay):
        if not self.show_overlay:
            return
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = self.last_image_msg.header
            self.overlay_pub.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f'Overlay publish error: {e}')

    def destroy_node(self):
        super().destroy_node()


def main(argv=None):
        rclpy.init(args=argv)
        node = ArucoPosePublisher()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
