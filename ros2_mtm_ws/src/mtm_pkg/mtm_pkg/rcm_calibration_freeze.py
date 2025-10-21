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
import rclpy
from rclpy.node import Node
from concurrent.futures import Future

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

        # Subscriptions just to store data
        self.create_subscription(Image,
                                 '/camera/camera/color/image_rect_raw',
                                 self.image_callback, 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        color_cam_info = wait_for_message(self, '/camera/camera/color/camera_info', CameraInfo, timeout=1)
        if color_cam_info is not None:
            self.camera_matrix = np.array(color_cam_info.k).reshape((3, 3))
            self.dist_coeffs = np.array(color_cam_info.d)

        # debug publisher
        self.show_overlay = True
        self.overlay_pub = self.create_publisher(Image, '/debug/rcm_overlay', 1)

        self.rcm_left_pub  = self.create_publisher(PoseStamped, '/RCM_pose_left', 1)
        self.rcm_right_pub = self.create_publisher(PoseStamped, '/RCM_pose_right', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/marker_pose', 1)

        # RCM offsets (edit as needed)
        self.marker_T_right_rcm = np.eye(4)
        self.marker_T_right_rcm[:3, 3] = np.array([-0.04, 0.0, 0.0])
        self.marker_T_left_rcm = np.eye(4)
        self.marker_T_left_rcm[:3, 3] = np.array([0.04, 0.0, 0.0])

        self.base_T_cam = np.eye(4)
        R = tf3d.euler.euler2mat(-np.pi/2, 0, -np.pi/2, axes='sxyz')
        self.base_T_cam[:3, :3] = R
        self.base_T_cam[:3, 3] = np.array([0.079, 0.0065, -0.00])

        # ArUco setup
        self.aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()
        self.marker_length = 0.06  # meters
        self.marker_id_l = 0
        self.marker_id_r = 1

        # Detection timer (e.g. 10 Hz)
        self.create_timer(1.0/10.0, self.detection_callback)

        # --- NEW: freeze + caches ---
        self.freeze = False
        self.last_rcm_left = None
        self.last_rcm_right = None
        self.last_cam_T_rcm_left = None   # camera-frame transforms for overlay while frozen
        self.last_cam_T_rcm_right = None

        # tiny window to capture keys
        cv2.namedWindow("k", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("k", 1, 1)
        cv2.moveWindow("k", 0, 0)

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

        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        img_pts = img_pts.reshape(-1, 2).astype(int)

        O, X, Y, Z = img_pts
        cv2.circle(img, O, 3, (255, 255, 255), -1)
        cv2.line(img, O, X, (0, 0, 255), 2)   # X red
        cv2.line(img, O, Y, (0, 255, 0), 2)   # Y green
        cv2.line(img, O, Z, (255, 0, 0), 2)   # Z blue
        cv2.putText(img, label, (O[0] + 5, O[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    def image_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV bridge error: {e}')
            return
        self.last_image_msg = msg
        self.last_image     = cv_img
        self.time_stamp = msg.header.stamp

    def detection_callback(self):
        if (self.last_image is None or
            self.camera_matrix is None or
            self.dist_coeffs is None or
            self.last_image_msg is None):
            return

        # --- minimal keyboard toggle ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):  # press 'f' to toggle freeze
            self.freeze = not self.freeze
            self.get_logger().info(f'freeze -> {self.freeze}')

        # If frozen: re-publish last poses and overlay the *cached cam-frame* poses on the *current* image
        if self.freeze:
            stamp_now = getattr(self, 'time_stamp', self.get_clock().now().to_msg())
            if self.last_rcm_left is not None:
                self.last_rcm_left.header.stamp = stamp_now
                self.rcm_left_pub.publish(self.last_rcm_left)
            if self.last_rcm_right is not None:
                self.last_rcm_right.header.stamp = stamp_now
                self.rcm_right_pub.publish(self.last_rcm_right)

            if self.show_overlay and self.last_image is not None:
                overlay = self.last_image.copy()
                try:
                    if self.last_cam_T_rcm_left is not None:
                        Rl = self.last_cam_T_rcm_left[:3, :3]
                        tl = self.last_cam_T_rcm_left[:3, 3]
                        self._draw_rcm_axes_on_image(overlay, Rl, tl, axis_len=0.05, label='RCM_L')
                    if self.last_cam_T_rcm_right is not None:
                        Rr = self.last_cam_T_rcm_right[:3, :3]
                        tr = self.last_cam_T_rcm_right[:3, 3]
                        self._draw_rcm_axes_on_image(overlay, Rr, tr, axis_len=0.05, label='RCM_R')
                    overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                    overlay_msg.header = self.last_image_msg.header
                    self.overlay_pub.publish(overlay_msg)
                except Exception as e:
                    self.get_logger().error(f'Overlay re-draw error (freeze): {e}')
            return

        # ---- Normal (live) path ----
        cam_frame = getattr(self.last_image_msg.header, 'frame_id', 'camera_frame')
        stamp = getattr(self, 'time_stamp', self.get_clock().now().to_msg())

        try:
            gray = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            self.get_logger().error(f'cv2.cvtColor failed: {e}')
            return
        overlay = self.last_image.copy()

        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is not None and len(ids) > 0:
            self.get_logger().info(f'Detected marker ids: {ids.flatten()}')
        if ids is None or len(ids) == 0:
            if self.show_overlay:
                try:
                    overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                    overlay_msg.header = self.last_image_msg.header
                    self.overlay_pub.publish(overlay_msg)
                except Exception as e:
                    self.get_logger().error(f'Overlay publish error (no ids): {e}')
            return

        try:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
        except Exception as e:
            self.get_logger().error(f'estimatePoseSingleMarkers failed: {e}')
            return

        cam_T_base = np.linalg.inv(self.base_T_cam)
        t_base = TransformStamped()
        t_base.header.stamp = stamp
        t_base.header.frame_id = cam_frame
        t_base.child_frame_id = 'robot_base'
        t_base.transform.translation.x = float(cam_T_base[0, 3])
        t_base.transform.translation.y = float(cam_T_base[1, 3])
        t_base.transform.translation.z = float(cam_T_base[2, 3])
        q_base = tf3d.quaternions.mat2quat(cam_T_base[:3, :3])
        t_base.transform.rotation.x = float(q_base[1])
        t_base.transform.rotation.y = float(q_base[2])
        t_base.transform.rotation.z = float(q_base[3])
        t_base.transform.rotation.w = float(q_base[0])
        self.tf_broadcaster.sendTransform(t_base)

        flat_ids = [int(i[0]) for i in ids]
        for idx, marker_id in enumerate(flat_ids):
            if marker_id not in (self.marker_id_l, self.marker_id_r):
                self.get_logger().info(f'Detected unknown marker id {marker_id}, ignoring')
                continue

            rvec = rvecs[idx][0]
            tvec = tvecs[idx][0]
            R, _ = cv2.Rodrigues(rvec)

            cam_T_marker = np.eye(4)
            cam_T_marker[:3, :3] = R
            cam_T_marker[:3, 3] = tvec

            q_marker = tf3d.quaternions.mat2quat(R)
            pose_msg = PoseStamped()
            pose_msg.header.stamp = stamp
            pose_msg.header.frame_id = cam_frame
            pose_msg.pose.position.x = float(tvec[0])
            pose_msg.pose.position.y = float(tvec[1])
            pose_msg.pose.position.z = float(tvec[2])
            pose_msg.pose.orientation.x = float(q_marker[1])
            pose_msg.pose.orientation.y = float(q_marker[2])
            pose_msg.pose.orientation.z = float(q_marker[3])
            pose_msg.pose.orientation.w = float(q_marker[0])
            self.pose_pub.publish(pose_msg)

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

            # Compose RCM in camera frame
            if marker_id == self.marker_id_r:
                cam_T_rcm = cam_T_marker @ self.marker_T_right_rcm
                label = 'RCM_R'
            else:
                cam_T_rcm = cam_T_marker @ self.marker_T_left_rcm
                label = 'RCM_L'

            # Draw onto the live frame
            if self.show_overlay:
                rcm_R_cam = cam_T_rcm[:3, :3]
                rcm_t_cam = cam_T_rcm[:3, 3]
                try:
                    self._draw_rcm_axes_on_image(overlay, rcm_R_cam, rcm_t_cam, axis_len=0.05, label=label)
                except Exception as e:
                    self.get_logger().error(f'Projection error for {label}: {e}')

            # Publish RCM pose in robot base frame
            base_T_rcm = self.base_T_cam @ cam_T_rcm
            q_rcm = tf3d.quaternions.mat2quat(base_T_rcm[:3, :3])
            rcm_msg = PoseStamped()
            rcm_msg.header.stamp = stamp
            rcm_msg.header.frame_id = 'robot_base'
            rcm_msg.pose.position.x = float(base_T_rcm[0, 3])
            rcm_msg.pose.position.y = float(base_T_rcm[1, 3])
            rcm_msg.pose.position.z = float(base_T_rcm[2, 3])
            rcm_msg.pose.orientation.x = float(q_rcm[1])
            rcm_msg.pose.orientation.y = float(q_rcm[2])
            rcm_msg.pose.orientation.z = float(q_rcm[3])
            rcm_msg.pose.orientation.w = float(q_rcm[0])
            (self.rcm_right_pub if marker_id == self.marker_id_r else self.rcm_left_pub).publish(rcm_msg)

            # --- NEW: cache for freeze mode (both Pose and camera-frame transform) ---
            if marker_id == self.marker_id_r:
                self.last_rcm_right = rcm_msg
                self.last_cam_T_rcm_right = cam_T_rcm.copy()
            else:
                self.last_rcm_left = rcm_msg
                self.last_cam_T_rcm_left = cam_T_rcm.copy()

        # Publish the overlay
        if self.show_overlay:
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
