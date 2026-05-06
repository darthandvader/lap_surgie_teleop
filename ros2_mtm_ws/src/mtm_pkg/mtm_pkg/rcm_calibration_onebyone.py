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
from scipy.spatial.transform import Rotation


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

        # debug publisher
        self.show_overlay = True
        self.overlay_pub = self.create_publisher(Image, '/debug/rcm_overlay', 1)

        # Publishers for RCM poses with respect to camera frame
        self.rcm_left_pub = self.create_publisher(PoseStamped, '/RCM_pose_left', 1)
        self.rcm_right_pub = self.create_publisher(PoseStamped, '/RCM_pose_right', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/marker_pose', 1)

        # ------------------ RCM OFFSETS (your logic kept) ------------------ #
        self.marker_T_right_rcm = np.eye(4)
        # self.marker_T_right_rcm[:3, :3] = Rotation.from_euler('x', -30, degrees=True).as_matrix()

        self.offset1 = np.eye(4)
        # self.offset1[:3, 3] = np.array([0.0, -0.14, 0.0])
        # self.offset2 = np.eye(4)
        # self.offset2[:3, 3] = np.array([0.0, -0.05, 0.0])
        self.marker_T_right_rcm[:3, 3] = np.array([0.0, -0.155, 0.0])

        self.marker_T_left_rcm = np.eye(4)
        # self.marker_T_left_rcm[:3, :3] = Rotation.from_euler('x', -30, degrees=True).as_matrix()
        self.marker_T_left_rcm[:3, 3] = np.array([0.0, -0.155, 0.0])
        # ------------------------------------------------------------------- #

        # ArUco setup
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters()
        self.marker_length = 0.1  # meters

        # Single marker ID used for both left and right, sequentially
        self.marker_id = 0

        # Detection timer (e.g. 10 Hz)
        self.create_timer(1.0 / 10.0, self.detection_callback)

        # --- calibration state machine ---
        # 0: using marker as LEFT (live)
        # 1: LEFT frozen, using marker as RIGHT (live)
        # 2: both frozen
        self.calib_state = 0

        # caches for poses and transforms
        self.last_rcm_left = None
        self.last_rcm_right = None
        self.last_cam_T_rcm_left = None
        self.last_cam_T_rcm_right = None

        # tiny window to capture keys
        cv2.namedWindow("k", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("k", 1, 1)
        cv2.moveWindow("k", 0, 0)

        self.get_logger().info(
            "ArucoPosePublisher started with single marker ID 0.\n"
            "Stage 0: use marker as LEFT RCM. Press 'f' to freeze left and "
            "start RIGHT calibration (Stage 1). Press 'f' again to freeze right (Stage 2)."
        )

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
        # Debug: make sure this actually runs
        self.get_logger().debug("detection_callback tick")

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

        # --- keyboard-driven calibration stages ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            if self.calib_state == 0:
                self.calib_state = 1
                self.get_logger().info(
                    'Left RCM frozen (using last cached LEFT pose). '
                    'Now calibrating RIGHT (still using marker ID 0). '
                    'Press f again to freeze right.'
                )
            elif self.calib_state == 1:
                self.calib_state = 2
                self.get_logger().info(
                    'Right RCM frozen (using last cached RIGHT pose). '
                    'Both RCMs locked.'
                )
            else:
                self.get_logger().info(
                    'Both RCMs already frozen; f has no further effect.'
                )

        cam_frame = getattr(self.last_image_msg.header, 'frame_id', 'camera_frame')
        stamp = self.time_stamp if self.time_stamp is not None else self.get_clock().now().to_msg()

        try:
            gray = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f'cv2.cvtColor failed: {e}')
            return
        overlay = self.last_image.copy()

        corners, ids, _ = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.aruco_params
        )
        if ids is not None and len(ids) > 0:
            self.get_logger().debug(f'Detected marker ids: {ids.flatten()}')
        if ids is None or len(ids) == 0:
            # still draw any frozen RCMs and re-publish
            self.get_logger().debug("No markers detected, publishing overlay with any frozen RCMs.")
            self._draw_and_publish_frozen(overlay)
            self._publish_overlay(overlay)
            return

        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
        except Exception as e:
            self.get_logger().error(f'estimatePoseSingleMarkers failed: {e}')
            return

        flat_ids = [int(i[0]) for i in ids]
        for idx, marker_id in enumerate(flat_ids):
            # Only care about our single marker ID
            if marker_id != self.marker_id:
                self.get_logger().info(f'Detected marker id {marker_id}, ignoring (we only use {self.marker_id})')
                continue

            # Stage logic: how do we interpret the detection?
            # Stage 0: marker → LEFT (live)
            # Stage 1: marker → RIGHT (live)
            # Stage 2: both frozen → ignore updates
            if self.calib_state >= 2:
                # everything frozen, ignore new detections
                continue

            rvec = rvecs[idx][0]
            tvec = tvecs[idx][0]
            R, _ = cv2.Rodrigues(rvec)

            cam_T_marker = np.eye(4)
            cam_T_marker[:3, :3] = R
            cam_T_marker[:3, 3] = tvec

            q_marker = tf3d.quaternions.mat2quat(R)
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
            # Optional: publish raw marker pose if needed
            # self.pose_pub.publish(marker_pose_msg)

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

            # Interpret marker as LEFT or RIGHT depending on stage
            if self.calib_state == 0:
                cam_T_rcm = cam_T_marker @ self.marker_T_left_rcm 
                label = 'RCM_L'
            elif self.calib_state == 1:
                cam_T_rcm = cam_T_marker @ self.marker_T_right_rcm 
                label = 'RCM_R'
            else:
                # calib_state >= 2 already handled above; just be safe
                continue

            # Draw live pose
            if self.show_overlay:
                rcm_R_cam = cam_T_rcm[:3, :3]
                rcm_t_cam = cam_T_rcm[:3, 3]
                try:
                    self._draw_rcm_axes_on_image(
                        overlay, rcm_R_cam, rcm_t_cam, axis_len=0.05, label=label
                    )
                except Exception as e:
                    self.get_logger().error(f'Projection error for {label}: {e}')

            # Publish and cache as left or right depending on stage
            rcm_msg = PoseStamped()
            rcm_msg.header.stamp = stamp
            rcm_msg.header.frame_id = cam_frame
            rcm_msg.pose.position.x = float(cam_T_rcm[0, 3])
            rcm_msg.pose.position.y = float(cam_T_rcm[1, 3])
            rcm_msg.pose.position.z = float(cam_T_rcm[2, 3])
            q_rcm = tf3d.quaternions.mat2quat(cam_T_rcm[:3, :3])
            rcm_msg.pose.orientation.x = float(q_rcm[1])
            rcm_msg.pose.orientation.y = float(q_rcm[2])
            rcm_msg.pose.orientation.z = float(q_rcm[3])
            rcm_msg.pose.orientation.w = float(q_rcm[0])

            if self.calib_state == 0:
                self.rcm_left_pub.publish(rcm_msg)
                self.last_rcm_left = rcm_msg
                self.last_cam_T_rcm_left = cam_T_rcm.copy()
                self.get_logger().debug('Updated LEFT RCM cache (stage 0).')
            elif self.calib_state == 1:
                self.rcm_right_pub.publish(rcm_msg)
                self.last_rcm_right = rcm_msg
                self.last_cam_T_rcm_right = cam_T_rcm.copy()
                self.get_logger().debug('Updated RIGHT RCM cache (stage 1).')

        # Draw & publish frozen poses on top
        self._draw_and_publish_frozen(overlay)
        self._publish_overlay(overlay)

    def _draw_and_publish_frozen(self, overlay):
        """Draw frozen RCMs and re-publish their poses with updated timestamps."""
        try:
            if self.show_overlay:
                # Left is frozen in stage >= 1
                if self.calib_state >= 1 and self.last_cam_T_rcm_left is not None:
                    Rl = self.last_cam_T_rcm_left[:3, :3]
                    tl = self.last_cam_T_rcm_left[:3, 3]
                    self._draw_rcm_axes_on_image(
                        overlay, Rl, tl, axis_len=0.05, label='RCM_L'
                    )
                # Right is frozen only in stage == 2
                if self.calib_state == 2 and self.last_cam_T_rcm_right is not None:
                    Rr = self.last_cam_T_rcm_right[:3, :3]
                    tr = self.last_cam_T_rcm_right[:3, 3]
                    self._draw_rcm_axes_on_image(
                        overlay, Rr, tr, axis_len=0.05, label='RCM_R'
                    )
        except Exception as e:
            self.get_logger().error(f'Frozen RCM overlay error: {e}')

        # Re-publish frozen poses with fresh timestamps
        stamp_now = self.time_stamp if self.time_stamp is not None else self.get_clock().now().to_msg()
        if self.calib_state >= 1 and self.last_rcm_left is not None:
            self.last_rcm_left.header.stamp = stamp_now
            self.rcm_left_pub.publish(self.last_rcm_left)
        if self.calib_state == 2 and self.last_rcm_right is not None:
            self.last_rcm_right.header.stamp = stamp_now
            self.rcm_right_pub.publish(self.last_rcm_right)

    def _publish_overlay(self, overlay):
        if not self.show_overlay:
            return
        try:
            self.get_logger().debug("Publishing /debug/rcm_overlay frame")
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


# TODO: add real time rcm calibration + testing script