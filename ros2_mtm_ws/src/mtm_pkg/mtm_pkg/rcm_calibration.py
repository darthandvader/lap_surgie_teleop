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
    """
    Waits for a single message on the specified topic.    :param node: The rclpy node instance.
    :param topic: The topic name to subscribe to.
    :param msg_type: The message type (e.g., std_msgs.msg.String).
    :param timeout: Timeout in seconds, or None for no timeout.
    :return: The received message, or None if the timeout occurs.
    """
    future = Future()    
    def callback(msg):
        if not future.done():
            future.set_result(msg)    # Create a temporary subscription
    subscription = node.create_subscription(msg_type, topic, callback, 10)    # Spin until a message is received or timeout occurs
    try:
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    finally:
        # Clean up subscription
        node.destroy_subscription(subscription)
    return future.result() if future.done() else None


class ArucoPosePublisher(Node):
    def __init__(self):
        super().__init__('aruco_pose_publisher')

        # Bridge and storage for incoming data
        self.overlay_pub = self.create_publisher(Image, '/aruco/overlay', 1)
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

        self.rcm_left_pub  = self.create_publisher(PoseStamped, '/RCM_pose_left', 1)
        self.rcm_right_pub = self.create_publisher(PoseStamped, '/RCM_pose_right', 1)
        self.pose_pub = self.create_publisher(PoseStamped, '/marker_pose', 1)

        # TODO:
        self.marker_T_right_rcm = np.eye(4)
        self.marker_T_right_rcm[:3, 3] = np.array([0.085, 0.0, 0.0])
        self.marker_T_left_rcm = np.eye(4)
        self.marker_T_left_rcm[:3, 3] = np.array([-0.085, 0.0, 0.0])

        self.base_T_cam = np.eye(4)
        R = tf3d.euler.euler2mat(-np.pi/2, 0, -np.pi/2, axes='sxyz')
        self.base_T_cam[:3, :3] = R
        self.base_T_cam[:3, 3] = np.array([0.079, 0.0065, -0.00])

        # ArUco setup
        self.aruco_dict   = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_params = aruco.DetectorParameters_create()
        self.marker_length = 0.075  # meters
        self.marker_id = 0

        # Detection timer (e.g. 15 Hz)
        self.create_timer(1.0/10.0, self.detection_callback)

    def image_callback(self, msg: Image):
        # store the latest image and its header
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV bridge error: {e}')
            return

        self.last_image_msg = msg
        self.last_image     = cv_img
        self.time_stamp = msg.header.stamp  
        
    def _rt_from_T(self, T44: np.ndarray):
        """Return OpenCV rvec/tvec from a 4x4 transform (camera -> object)."""
        R = T44[:3, :3].astype(np.float64)
        t = T44[:3, 3].reshape(3, 1).astype(np.float64)
        rvec, _ = cv2.Rodrigues(R)
        return rvec, t
    
    def detection_callback(self):
        # only proceed if we have both camera info and an image
        if self.last_image is not None:
            gray = cv2.cvtColor(self.last_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params)

            # if exactly two, identify left/right by ID and publish
            if ids is not None:
                # flatten ids and find left/right
                flat_ids = [i[0] for i in ids]

                # estimate poses
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length,
                    self.camera_matrix, self.dist_coeffs)

                for idx, marker_id in enumerate(flat_ids):
                    if marker_id == self.marker_id:
                        self.get_logger().info(f'Detected target id {self.marker_id}')
                        rvec = rvecs[idx][0]
                        tvec = tvecs[idx][0]
                        R, _ = cv2.Rodrigues(rvec)
                        
                        # This gives you cam_T_marker
                        q = tf3d.quaternions.mat2quat(R)
                        p = tvec
                        self.cam_T_marker = np.eye(4)
                        self.cam_T_marker[:3, :3] = R
                        self.cam_T_marker[:3, 3] = p

                        # publish the pose in the camera frame
                        pose_msg = PoseStamped()
                        pose_msg.header.frame_id = 'camera_frame'
                        pose_msg.pose.position.x = float(p[0])
                        pose_msg.pose.position.y = float(p[1])
                        pose_msg.pose.position.z = float(p[2])
                        pose_msg.pose.orientation.x = float(q[1])
                        pose_msg.pose.orientation.y = float(q[2])
                        pose_msg.pose.orientation.z = float(q[3])
                        pose_msg.pose.orientation.w = float(q[0])
                        self.pose_pub.publish(pose_msg)

                        # broadcast the transform
                        t = TransformStamped()
                        t.header.stamp = self.time_stamp
                        t.header.frame_id = 'camera_frame'
                        t.child_frame_id = f'marker_{marker_id}'
                        t.transform.translation.x = float(p[0])
                        t.transform.translation.y = float(p[1])
                        t.transform.translation.z = float(p[2])
                        t.transform.rotation.x = float(q[1])
                        t.transform.rotation.y = float(q[2])
                        t.transform.rotation.z = float(q[3])
                        t.transform.rotation.w = float(q[0])
                        self.tf_broadcaster.sendTransform(t)
                        self.get_logger().info(f'Published pose for marker {marker_id}')

                        # publish cam_T_base
                        cam_T_base = np.linalg.inv(self.base_T_cam)
                        t = TransformStamped()
                        t.header.stamp = self.time_stamp
                        t.header.frame_id = 'camera_frame'
                        t.child_frame_id = 'robot_base'
                        t.transform.translation.x = float(cam_T_base[0, 3])
                        t.transform.translation.y = float(cam_T_base[1, 3])
                        t.transform.translation.z = float(cam_T_base[2, 3])
                        q = tf3d.quaternions.mat2quat(cam_T_base[:3, :3])
                        t.transform.rotation.x = float(q[1])
                        t.transform.rotation.y = float(q[2])
                        t.transform.rotation.z = float(q[3])
                        t.transform.rotation.w = float(q[0])
                        self.tf_broadcaster.sendTransform(t)

                        # publish RCM poses
                        rcm_pose = self.base_T_cam @ self.cam_T_marker @ self.marker_T_right_rcm
                        rcm_msg = PoseStamped()
                        rcm_msg.header.frame_id = 'robot_base'
                        rcm_msg.pose.position.x = float(rcm_pose[0, 3])
                        rcm_msg.pose.position.y = float(rcm_pose[1, 3])
                        rcm_msg.pose.position.z = float(rcm_pose[2, 3])
                        q = tf3d.quaternions.mat2quat(rcm_pose[:3, :3])
                        rcm_msg.pose.orientation.x = float(q[1])
                        rcm_msg.pose.orientation.y = float(q[2])
                        rcm_msg.pose.orientation.z = float(q[3])
                        rcm_msg.pose.orientation.w = float(q[0])
                        self.rcm_right_pub.publish(rcm_msg)

                        rcm_pose = self.base_T_cam @ self.cam_T_marker @ self.marker_T_left_rcm
                        rcm_msg = PoseStamped()
                        rcm_msg.header.frame_id = 'robot_base'
                        rcm_msg.pose.position.x = float(rcm_pose[0, 3])
                        rcm_msg.pose.position.y = float(rcm_pose[1, 3])
                        rcm_msg.pose.position.z = float(rcm_pose[2, 3])
                        q = tf3d.quaternions.mat2quat(rcm_pose[:3, :3])
                        rcm_msg.pose.orientation.x = float(q[1])
                        rcm_msg.pose.orientation.y = float(q[2])
                        rcm_msg.pose.orientation.z = float(q[3])
                        rcm_msg.pose.orientation.w = float(q[0])
                        self.rcm_left_pub.publish(rcm_msg)

                        # --- build 4x4 transforms in CAMERA frame ---
                        cam_T_marker = self.cam_T_marker.copy()
                        cam_T_rcm_right = cam_T_marker @ self.marker_T_right_rcm
                        cam_T_rcm_left  = cam_T_marker @ self.marker_T_left_rcm

                        # --- make a copy of the BGR image to draw on ---
                        overlay = self.last_image.copy()

                        # --- draw axes for each pose (axis length in meters) ---
                        axis_len = 0.05  # 5 cm axes

                        def draw_axes_on(overlay_img, T44, K, D, axis_length, label_text):
                            rvec, tvec = self._rt_from_T(T44)
                            # Draw axes
                            cv2.drawFrameAxes(
                                overlay_img,
                                K.astype(np.float64),
                                D.astype(np.float64),
                                rvec, tvec,
                                axis_length
                            )
                            # Project label point (origin) for annotation
                            pts2d, _ = cv2.projectPoints(
                                np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                                rvec, tvec, K.astype(np.float64), D.astype(np.float64)
                            )
                            u, v = int(pts2d[0,0,0]), int(pts2d[0,0,1])
                            cv2.putText(overlay_img, label_text, (u+5, v-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

                        # camera intrinsics/distortion
                        K = self.camera_matrix
                        D = self.dist_coeffs

                        draw_axes_on(overlay, cam_T_marker,    K, D, axis_len, 'marker')
                        draw_axes_on(overlay, cam_T_rcm_right, K, D, axis_len, 'RCM_R')
                        draw_axes_on(overlay, cam_T_rcm_left,  K, D, axis_len, 'RCM_L')

                        # publish overlay image
                        img_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
                        # preserve original header (timestamp/frame_id) so RViz sync works with TF
                        if self.last_image_msg is not None:
                            img_msg.header = self.last_image_msg.header
                        self.overlay_pub.publish(img_msg)

                # also broadcast TF
        # if pose8 is not None and pose_10 is not None:
        #     p = pose8[1]
        #     r = pose8[0]
        #     R, _ = cv2.Rodrigues(r)
        #     cam_T_8 = np.eye(4)
        #     cam_T_8[:3, :3] = R
        #     cam_T_8[:3, 3] = p

        #     p = pose_10[1]
        #     r = pose_10[0]
        #     R, _ = cv2.Rodrigues(r)
        #     q = tf3d.quaternions.mat2quat(R)
        #     cam_T_10 = np.eye(4)
        #     cam_T_10[:3, :3] = R
        #     cam_T_10[:3, 3] = p

        #     _8_T_10 = np.linalg.inv(cam_T_8) @ cam_T_10

        #     # print('8_T_10:', _8_T_10[:3, 3])
        #     tr = _8_T_10[:3, 3]
        #     tr = np.array([float(tr[2]+0.06), float(tr[0]), float(tr[1])-0.055])
        #     print('8_T_10:', tr)
        # print('broadcasting: ', self.get_clock().now().to_msg())



        # else: too few or too many, no pose publishes

    def destroy_node(self):
        # clean up if needed
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