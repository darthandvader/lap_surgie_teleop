import rclpy
from rclpy.node import Node
import numpy as np
from datetime import datetime
from sensor_msgs.msg import Image, JointState, Joy
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from collections import deque
import os
import time

from writer import EpisodeWriter

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
        # Don't run Image topics in VScode 
        self.create_subscription(Image, '/stereo/left/rectified_downscaled_image', self.left_image_callback, 10)
        self.create_subscription(Image, '/stereo/right/rectified_downscaled_image', self.right_image_callback, 10)
        self.create_subscription(JointState, '/PSM1/measured_js', self.psm1_js_callback, 10)
        self.create_subscription(JointState, '/PSM2/measured_js', self.psm2_js_callback, 10)
        self.create_subscription(PoseStamped, '/PSM1/measured_cp', self.psm1_cp_callback, 10)
        self.create_subscription(PoseStamped, '/PSM2/measured_cp', self.psm2_cp_callback, 10)

        # Timer: run at 30 Hz 
        # self.frequency = 15.  # Hz
        self.frequency = 30.  # Hz
        self.timer = self.create_timer(1/self.frequency, self.timer_callback)

        
        # Control with footpedal and gripper pattern
        self.create_subscription(Joy, '/footpedals/camera', self.pedal_callback_camera, 10)
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
        self.recording = False

        # Output writer
        self.writer = EpisodeWriter(task_dir='/docker-ros/ws/data/lifting/',)


    def pedal_callback_camera(self, msg: Joy):
        if len(msg.buttons) > 0 and msg.buttons[0] == 1 and self.pedal_camera_prev == 0:
            if self.recording:
                self.recording = False
                self.reset_buffer()
                self.get_logger().info("Recording discarded by foot pedal.")
        self.pedal_camera_prev = msg.buttons[0]

    def jaw1_callback(self, msg: JointState):
        self.jaw1_position = msg.position[0]
        threshold = 0.05

        new_state = 'closed' if self.jaw1_position > threshold else 'open'

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
                self.get_logger().info('jaw1')
                if not self.recording:
                    self.start_recording()

    def jaw2_callback(self, msg: JointState):
        
        self.jaw2_position = msg.position[0]
        threshold = 0.05

        new_state = 'closed' if self.jaw2_position > threshold else 'open'

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
                self.get_logger().info('jaw2')
                if self.recording:
                    self.stop_and_save()

    def left_image_callback(self, msg):
        # Don't run Image topics in VScode
        self.left_image= self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def right_image_callback(self, msg):
        # Don't run Image topics in VScode
        self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def psm1_js_callback(self, msg: JointState):
        # print("PSM1 joint state callback triggered")
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
           hasattr(self, 'psm2_cp') and \
           hasattr(self, 'jaw2_position') and \
           hasattr(self, 'jaw1_position'):

           

            colors = {"left_image": self.left_image,
                      "right_image": self.right_image}
            

            states = {
                        "psm_cutter_js": {                                                                    
                            "qpos": self.psm2_js.tolist(), 
                            "gripper": self.jaw2_position,              
                        }, 
                        "psm_retraction_js": {                                                                    
                            "qpos": self.psm1_js.tolist(),       
                            "gripper": self.jaw1_position,
                        
                        },               
                        "psm_cutter_ee": {
                            "psm_cutter_pos": [self.psm2_cp.position.x,
                                                self.psm2_cp.position.y,
                                                self.psm2_cp.position.z],
                            "psm_cutter_quat": [self.psm2_cp.orientation.w,
                                                self.psm2_cp.orientation.x,
                                                self.psm2_cp.orientation.y,
                                                self.psm2_cp.orientation.z],
                        },
                        "psm_retraction_ee": {
                            "psm_retraction_pos": [self.psm1_cp.position.x,
                                                    self.psm1_cp.position.y,
                                                    self.psm1_cp.position.z],
                            "psm_retraction_quat": [self.psm1_cp.orientation.w,
                                                    self.psm1_cp.orientation.x,
                                                    self.psm1_cp.orientation.y,
                                                    self.psm1_cp.orientation.z],
                        },
                    }

            self.writer.add_item(colors=colors, states=states)
            
    def start_recording(self):
        self.writer.create_episode()
        self.recording = True
        self.get_logger().info(f"Started recording demo_{self.writer.episode_id:04d}")

    def stop_and_save(self):
        self.writer.save_episode()
        self.recording = False
        self.get_logger().info(f"End recording demo_{self.writer.episode_id:04d}")
        
    def destroy_node(self):
        self.writer.close()
        super().destroy_node()


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