import rclpy
from rclpy.node import Node
import h5py
import numpy as np
from datetime import datetime
from sensor_msgs.msg import Image, JointState, Joy
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from collections import deque
import os
import time


class DVRKDataCollector(Node):

    '''
    Ros2 node for collecting dVRK data.

    This node subscribes to various topics to collect images,
    joint states, and Cartesian poses from the dVRK system.

    control:
    - 3 fast psm1 jaw pinch: start recording
    - 3 fast psm2 jaw pinch: save recording
    - foot camera button: discard recording
    
    '''

    def __init__(self):
        super().__init__('dvrk_data_collector')
        self.bridge = CvBridge()

        # Subscribers
        self.create_subscription(Image, '/stereo/left/rectified_downscaled_image', self.left_image_callback, 10)
        self.create_subscription(Image, '/stereo/right/rectified_downscaled_image', self.right_image_callback, 10)
        self.create_subscription(JointState, '/PSM1/measured_js', self.psm1_js_callback, 10)
        self.create_subscription(JointState, '/PSM2/measured_js', self.psm2_js_callback, 10)
        self.create_subscription(PoseStamped, '/PSM1/measured_cp', self.psm1_cp_callback, 10)
        self.create_subscription(PoseStamped, '/PSM2/measured_cp', self.psm2_cp_callback, 10)

        # Timer: run at 30 Hz 
        self.frequency = 15.  # Hz
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)
        
        # Control with footpedal and gripper pattern
        self.create_subscription
        # self.create_subscription(Joy, '/footpedals/cam_plus', self.pedal_callback_plus, 10)
        # self.create_subscription(Joy, '/footpedals/cam_minus', self.pedal_callback_minus, 10)
        self.create_subscription(Joy, '/footpedals/camera', self.pedal_callback_camera, 10)
        # self.pedal_plus_prev = 0
        # self.pedal_minus_prev = 0
        self.pedal_camera_prev = 0

        self.create_subscription(JointState, '/PSM1/jaw/measured_js', self.jaw1_callback, 10)
        self.jaw1_transition_times = deque()
        self.jaw1_state = 'open'
        self.create_subscription(JointState, '/PSM2/jaw/measured_js', self.jaw2_callback, 10)
        self.jaw2_transition_times = deque()
        self.jaw2_state = 'open'

        self.n = 3
        self.time_window = 2.0

        # Data buffers
        self.reset_buffer()
        self.recording = False
        self.demo_count = 0

        # Output HDF5 file

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = f'/docker-rostask_done/ws/data/dvrk_demos_{timestamp}.hdf5'
        self.hdf = h5py.File(self.output_path, 'w')
        metadata = self.hdf.create_group('metadata')
        metadata.attrs['sampling_frequency_hz'] = self.frequency
        self.get_logger().info(f"Recording initialized. Output: {self.output_path}")

    # def pedal_callback_plus(self, msg: Joy):
    #     if len(msg.buttons) > 0 and msg.buttons[0] == 1 and self.pedal_plus_prev == 0:
    #         if not self.recording:
    #             self.start_recording()
    #     self.pedal_plus_prev = msg.buttons[0]

    # def pedal_callback_minus(self, msg: Joy):
    #     if len(msg.buttons) > 0 and msg.buttons[0] == 1 and self.pedal_minus_prev == 0:
    #         if self.recording:
    #             self.stop_and_save()
    #     self.pedal_minus_prev = msg.buttons[0]

    def pedal_callback_camera(self, msg: Joy):
        if len(msg.buttons) > 0 and msg.buttons[0] == 1 and self.pedal_camera_prev == 0:
            if self.recording:
                self.recording = False
                self.reset_buffer()
                self.get_logger().info("Recording discarded by foot pedal.")
        self.pedal_camera_prev = msg.buttons[0]

    def jaw1_callback(self, msg: JointState):
        position = msg.position[0]
        threshold = 0.05

        new_state = 'closed' if position > threshold else 'open'

        if new_state != self.jaw1_state:
            now = time.time()
            self.jaw1_transition_times.append(now)
            self.jaw1_state = new_state

            # Clean old timestamps
            while self.jaw1_transition_times and now - self.jaw1_transition_times[0] > self.time_window:
                self.jaw1_transition_times.popleft()

            # Check for n full open-close cycles (2 transitions = 1 cycle)
            if len(self.jaw1_transition_times) >= self.n * 2:
                self.jaw1_transition_times.clear()
                if not self.recording:
                    self.start_recording()

    def jaw2_callback(self, msg: JointState):
        position = msg.position[0]
        threshold = 0.05

        new_state = 'closed' if position > threshold else 'open'

        if new_state != self.jaw2_state:
            now = time.time()
            self.jaw2_transition_times.append(now)
            self.jaw2_state = new_state

            # Clean old timestamps
            while self.jaw2_transition_times and now - self.jaw2_transition_times[0] > self.time_window:
                self.jaw2_transition_times.popleft()

            # Check for n full open-close cycles (2 transitions = 1 cycle)
            if len(self.jaw2_transition_times) >= self.n * 2:
                self.jaw2_transition_times.clear()
                if self.recording:
                    self.stop_and_save()

    def left_image_callback(self, msg):
        self.left_image= self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def right_image_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def psm1_js_callback(self, msg: JointState):
        self.psm1_js = msg.position
        
    def psm2_js_callback(self, msg: JointState):
        self.psm2_js = msg.position

    def psm1_cp_callback(self, msg: PoseStamped):
        self.psm1_cp = msg.pose
    
    def psm2_cp_callback(self, msg: PoseStamped):
        self.psm2_cp = msg.pose

    def timer_callback(self):
        if not self.recording:
            return
 
        if hasattr(self, 'left_image') and \
           hasattr(self, 'right_image') and \
           hasattr(self, 'psm1_js') and \
           hasattr(self, 'psm2_js') and \
           hasattr(self, 'psm1_cp') and \
           hasattr(self, 'psm2_cp'):
            
            self.left_image_trajectory.append(self.left_image)
            self.right_image_trajectory.append(self.right_image)
            self.psm1_js_trajectory.append(self.psm1_js)
            self.psm2_js_trajectory.append(self.psm2_js)
            self.psm1_cp_trajectory.append(self.psm1_cp)
            self.psm2_cp_trajectory.append(self.psm2_cp)

    def start_recording(self):
        self.images = []
        self.joint_positions = []
        self.recording = True
        self.get_logger().info(f"Started recording demo_{self.demo_count:04d}")

    def stop_and_save(self):
        self.recording = False
        demo_id = f"demo_{self.demo_count:04d}"
        self.get_logger().info(f"Stopping recording. Saving {demo_id}...")

        demo = self.hdf.create_group(f'/data/{demo_id}')

        # Save joint states
        demo.create_dataset('psm1_joint_positions', data=np.array(self.psm1_js_trajectory))
        demo.create_dataset('psm2_joint_positions', data=np.array(self.psm2_js_trajectory))

        def pose_to_array(pose):
            return np.array([
                pose.position.x, pose.position.y, pose.position.z,
                 pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
            ])

        psm1_cp_array = np.array([pose_to_array(pose) for pose in self.psm1_cp_trajectory])
        psm2_cp_array = np.array([pose_to_array(pose) for pose in self.psm2_cp_trajectory])
        demo.create_dataset('psm1_cartesian_pose', data=psm1_cp_array)
        demo.create_dataset('psm2_cartesian_pose', data=psm2_cp_array)

        # Save images
        obs_group = demo.create_group('obs')
        obs_group.create_dataset('left_image', data=np.stack(self.left_image_trajectory))
        obs_group.create_dataset('right_image', data=np.stack(self.right_image_trajectory))

        self.get_logger().info(f"Saved {demo_id} with {len(self.left_image_trajectory)} frames.")

        # Reset for next demo
        self.reset_buffer()
        self.demo_count += 1

    def reset_buffer(self):
        self.left_image, self.right_image = None, None
        self.psm1_js, self.psm2_js = None, None
        self.psm1_cp, self.psm2_cp = None, None
        self.left_image_trajectory = []
        self.right_image_trajectory = []
        self.psm1_js_trajectory = []
        self.psm2_js_trajectory = []
        self.psm1_cp_trajectory = []
        self.psm2_cp_trajectory = []
    
    def destroy_node(self):
        super().destroy_node()
        if self.hdf:
            self.hdf.close()
            self.get_logger().info("HDF5 file closed.")
    

def main(args=None):
    rclpy.init(args=args)
    collector = DVRKDataCollector()
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        collector.get_logger().info("Interrupted. Exiting.")
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()