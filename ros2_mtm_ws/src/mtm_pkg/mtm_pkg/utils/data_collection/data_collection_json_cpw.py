import rclpy
from rclpy.node import Node
import numpy as np
from datetime import datetime
from sensor_msgs.msg import Image, JointState, Joy
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import threading
from collections import deque
import os
import time
import cv2
import select

from writer import EpisodeWriter
from image_labeler import ImageLabeler

import torch
import sys
sys.path.insert(0 ,'src')

from lite_tracker.src.lite_tracker import LiteTracker

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

        # setup
        self.bridge = CvBridge()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize variables
        # self.initial_joint_state = np.load(os.path.join(os.path.dirname(__file__), 'initial_joint_state.npy'))
        # self.initial_jaw_state = np.load(os.path.join(os.path.dirname(__file__), 'initial_jaw_state.npy'))[0]
        # import pdb; pdb.set_trace()
        self.initial_retractor_joint_state = np.load(os.path.join(os.path.dirname(__file__), 'psm1_initial_joint_state.npy'))
        self.initial_retractor_jaw_state = np.load(os.path.join(os.path.dirname(__file__), 'psm1_initial_jaw_state.npy'))

        self.initial_cutter_joint_state = np.load(os.path.join(os.path.dirname(__file__), 'psm2_initial_joint_state.npy'))
        self.initial_cutter_jaw_state = np.load(os.path.join(os.path.dirname(__file__), 'psm2_initial_jaw_state.npy'))


        self.dissection_points = None

        # colors for visualize tracked points
        self.colors = [
                    (0, 0, 255),   # Red
                    (0, 255, 0),   # Green
                    (255, 0, 0),   # Blue
                    (0, 255, 255), # Yellow
                    # Add more if you have more points
                ]
        
        self.vis_left_image = None
        self.pred_coords = None

        # Subscribers
        # Don't run Image topics in VScode 
        self.create_subscription(Image, '/stereo/left/rectified_downscaled_image', self.left_image_callback, 10)
        self.create_subscription(Image, '/stereo/right/rectified_downscaled_image', self.right_image_callback, 10)
        self.create_subscription(JointState, '/PSM1/measured_js', self.psm1_js_callback, 10)
        self.create_subscription(JointState, '/PSM2/measured_js', self.psm2_js_callback, 10)
        self.create_subscription(PoseStamped, '/PSM1/measured_cp', self.psm1_cp_callback, 10)
        self.create_subscription(PoseStamped, '/PSM2/measured_cp', self.psm2_cp_callback, 10)

        # Publishers
        # this is for large movement, large movement will not work with servo_jp 
        self.retractorMove_pub = self.create_publisher(
            JointState,
            '/PSM1/move_jp',
            10,
        )

        self.cutterMove_pub = self.create_publisher(
            JointState,
            '/PSM2/move_jp',
            10,
        )

        self.set_retractor_gripperServo_pub = self.create_publisher(
            JointState,
            '/PSM1/jaw/servo_jp',
            10,
        )

        self.set_cutter_gripperServo_pub = self.create_publisher(
            JointState,
            '/PSM2/jaw/servo_jp',
            10,
        )

        

        # Timer: run at 30 Hz 
        self.frequency = 30  # Hz 
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

        # Arrange the model and queries
        self.tracker = LiteTracker()
        with open('weights/scaled_online.pth', "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        self.tracker.load_state_dict(state_dict)
        self.tracker = self.tracker.to(self.device)
        self.tracker.eval()

        # Data buffers
        self.recording = False

        # Output writer
        self.writer = EpisodeWriter(task_dir=f'/docker-ros/ws/data/1005_phantom_{self.frequency}hz/',)
        # self.writer = EpisodeWriter(task_dir=f'/docker-ros/ws/data/revealing_data/',)
        # self.writer = EpisodeWriter(task_dir=f'/docker-ros/ws/data/cholin_00{self.frequency}hz/',)

        # print("Moving to initial position...")
        self._PSMMove(
            self.retractorMove_pub,
            'retractor',
            self.initial_retractor_joint_state,
            sleep_time=3,
        )
        self._PSMMove(
            self.cutterMove_pub,
            'cutter',
            self.initial_cutter_joint_state,
            sleep_time=3,
        )

        self._jawServo(
            self.set_retractor_gripperServo_pub,
            self.initial_retractor_jaw_state,
            sleep_time=0.1,
        )

        self._jawServo(
            self.set_cutter_gripperServo_pub,
            self.initial_cutter_jaw_state,
            sleep_time=0.1,
        )
        # print("Moved to initial position...")

    def record_psm1_joint_positions(self):
        joint_states = self.psm1_js
        np.save(os.path.join(os.path.dirname(__file__), f'psm1_initial_joint_state.npy'), np.array(joint_states))

    def record_psm1_jaw_positions(self):
        jaw_states = self.jaw1_position
        np.save(os.path.join(os.path.dirname(__file__), f'psm1_initial_jaw_state.npy'), np.array(jaw_states))

    def record_psm2_joint_positions(self):
        joint_states = self.psm2_js
        np.save(os.path.join(os.path.dirname(__file__), f'psm2_initial_joint_state.npy'), np.array(joint_states))

    def record_psm2_jaw_positions(self):
        jaw_states = self.jaw2_position
        np.save(os.path.join(os.path.dirname(__file__), f'psm2_initial_jaw_state.npy'), np.array(jaw_states))



    def init_tracker(self):
        """
        Initialize the tracker with the model weights.
        """
        self.tracker = LiteTracker()
        with open('weights/scaled_online.pth', "rb") as f:
            state_dict = torch.load(f, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
        self.tracker.load_state_dict(state_dict)
        self.tracker = self.tracker.to(self.device)
        self.tracker.eval()

    def tracking(self, points, frame):
        def _process_step(frame):
            with torch.no_grad():
                frame = (
                    torch.tensor(frame, device=self.device)
                    .permute(2, 0, 1)[None]
                    .float()
                )  # shape is (B, C, H, W)

                return self.tracker(
                    frame,
                    queries=queries,
                )
        with torch.autocast(
            device_type=self.device,
            enabled=True,
            ):
            queries = torch.cat(
                    [
                        torch.ones_like(points[:, :, :1]) * 0,
                        points,
                    ],
                    dim=2,
                ).to(self.device)
            pred_coords, viss, confs = _process_step(
                    frame,
                )
        return pred_coords
    
    def _PSMMove( 
        self,
        pub,
        psm_name: str,
        goal_state: float,
        sleep_time: float = 0.1,
    ):
        """

        Set a retractor's joint angle
        args:
            pub: ROS publisher of the retractor's joint angle
            goal_state: target joint angle in radians
        """

        j_msg = JointState()
        j_msg.name = [f'{psm_name}']
        j_msg.position: list[float] = goal_state.astype(float).tolist()
        j_msg.velocity = [0.0]
        j_msg.effort   = [0.0]
        pub.publish(j_msg)
        time.sleep(sleep_time)  # Wait for the retractor to respond

    def _jawServo(
        self,
        pub,
        end_pos: float,
        sleep_time: float = 0.01,
    ):
        """
        Set a gripper's jaw angle
        args:
            pub: ROS publisher of the gripper's jaw angle
            end_pos: target jaw angle in radians
        """
        j_msg = JointState()
        j_msg.name = ['jaw']  # Assuming jaw state is the first element
        j_msg.position: list[float] = [float(end_pos)]  # end_pos is a scalar
        j_msg.velocity = [0.0]
        j_msg.effort   = [0.0]
        pub.publish(j_msg)
        time.sleep(sleep_time)  # Wait for the gripper to respond


    def pedal_callback_camera(self, msg: Joy):
        if len(msg.buttons) > 0 and msg.buttons[0] == 1 and self.pedal_camera_prev == 0:
            if self.recording:
                self.recording = False
                self.reset_buffer()
                self.get_logger().info("Recording discarded by foot pedal.")
        self.pedal_camera_prev = msg.buttons[0]

    def jaw1_callback(self, msg: JointState):
        self.jaw1_position = msg.position[0]
        self.jaw1_effort = msg.effort[0]
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
        self.jaw2_effort = msg.effort[0]
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
        self.psm1_js = msg.position
        self.psm1_js_vel = msg.velocity
        self.psm1_js_effort = msg.effort
        
    def psm2_js_callback(self, msg: JointState):
        self.psm2_js = msg.position
        self.psm2_js_vel = msg.velocity
        self.psm2_js_effort = msg.effort

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
            
            self.vis_left_image = self.left_image.copy()
            pred_coords = self.tracking(torch.tensor(self.dissection_points).unsqueeze(0), self.vis_left_image)
            pred_coords.squeeze().cpu().numpy().tolist()
                        
            # visualize tracked points on left image
            if pred_coords is not None:
                pred_coords = pred_coords.squeeze().cpu().numpy()
                for idx, coord in enumerate(pred_coords):
                    color = self.colors[idx % len(self.colors)]  # Repeat colors if more points than colors
                    cv2.circle(self.vis_left_image, (int(coord[0]), int(coord[1])), 5, color, -1)
            # cv2.imshow("Tracked Points", vis_left_image)
            # cv2.waitKey(1)

            states = {
                        "psm_cutter_js": {                                                                    
                            "qpos": self.psm2_js.tolist(), 
                            "qvel": self.psm2_js_vel.tolist(),
                            "qeffort": self.psm2_js_effort.tolist(),
                            "gripper": self.jaw2_position,   
                            "gripper_effort": self.jaw2_effort           
                        }, 
                        "psm_retraction_js": {                                                                    
                            "qpos": self.psm1_js.tolist(), 
                            "qvel": self.psm1_js_vel.tolist(),
                            "qeffort": self.psm1_js_effort.tolist(),      
                            "gripper": self.jaw1_position,
                            "gripper_effort": self.jaw1_effort
                        
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
                        "dissection_target": {                                                                    
                            "points": pred_coords.tolist(),       
                        }, 
                    }

            self.writer.add_item(colors=colors, states=states)

    def label_dissection_target(self):
        labeler = ImageLabeler("Image Labeler", self.left_image)
        labeler.show()
        task_dir = self.writer.task_dir
        labeler.save(os.path.join(task_dir, f"label_{self.writer.episode_id+1:04d}.png"))
        self.dissection_points = labeler.get_keypoints()

    def move_to_initial_position(self):
        print("Moving to initial position...")
        self._PSMMove(
            self.retractorMove_pub,
            'retractor',
            self.initial_retractor_joint_state,
            sleep_time=3,
        )

        self._PSMMove(
            self.cutterMove_pub,
            'cutter',
            self.initial_cutter_joint_state,
            sleep_time=3,
        )

        self._jawServo(
            self.set_retractor_gripperServo_pub,
            self.initial_retractor_jaw_state,
            sleep_time=0.1,
        )

        self._jawServo(
            self.set_cutter_gripperServo_pub,
            self.initial_cutter_jaw_state,
            sleep_time=0.1,
        )
        print("Moved to initial position...")

    def start_recording(self):
        self.get_logger().info(f"Started recording demo_{self.writer.episode_id:04d}")
        self.init_tracker()  # Initialize the tracker
        self.writer.create_episode()
        self.recording = True

        # if a new frame is ready, show it
        if self.vis_left_image is not None:
            cv2.imshow("Tracked Points", self.vis_left_image)
            cv2.imshow("Tracked Points", cv2.rotate(self.vis_left_image, cv2.ROTATE_180))
           
    def stop_and_save(self):
        self.writer.save_episode()
        self.recording = False
        self.vis_left_image = None
        self.dissection_points = None
        self.get_logger().info(f"End recording demo_{self.writer.episode_id:04d}")
        cv2.destroyWindow("Tracked Points")
        
    def destroy_node(self):
        self.writer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DVRKDataCollector()
    # MultiThreadedExecutor for handling callbacks
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    # Start the executor in a separate thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    frame_num = 0
    try:
        while rclpy.ok():
            # pump ROS once, but don’t block the thread
            rclpy.spin_once(node, timeout_sec=0.01)

            # (optional) process keyboard commands coming from stdin
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip()
                if cmd == "l":
                    node.label_dissection_target()
                    node.move_to_initial_position()
                elif cmd == 'r':
                    node.record_psm1_joint_positions()
                    node.get_logger().info(f"Record PSM1 joint positions")
                    node.record_psm2_joint_positions()
                    node.get_logger().info(f"Record PSM2 joint positions")
                    node.record_psm1_jaw_positions()
                    node.record_psm2_jaw_positions()
                    
                elif cmd == 'init':
                    node.move_to_initial_position()

                elif cmd == "q":
                    break

            # if a new frame is ready, show it
            if getattr(node, "vis_left_image", None) is not None and node.recording == True:
                cv2.namedWindow("Tracked Points", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Tracked Points", 1920, 1080)
                cv2.imshow("Tracked Points", cv2.rotate(node.vis_left_image, cv2.ROTATE_180))
                # cv2.imwrite(
                #     os.path.join(node.writer.task_dir, f"tracked_points_{frame_num:04d}.png"),
                #     node.vis_left_image,
                # )
                # frame_num += 1
                # cv2.imshow("Tracked Points", node.vis_left_image)
                # keep the window responsive
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    continue

        # # Main thread loop with user input
        # while rclpy.ok():
        #     user_input = input("Enter a command: ")
        #     match user_input:
        #         case "q":
        #             print("Stopping node...")
        #             break
        #         case "l":
        #             print("Starting labeling...")
        #             node.label_dissection_target()
        #             print("Labeling completed.")
        #             node.move_to_initial_position()
        #         case _:
        #             print('Invalid command')

            
                

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        executor.shutdown()
        executor_thread.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()