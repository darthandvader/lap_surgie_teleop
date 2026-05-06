from rclpy.node import Node
import rclpy
import time
import numpy as np
from sensor_msgs.msg import Joy, JointState
from geometry_msgs.msg import WrenchStamped, Vector3, Quaternion, PoseStamped
from std_msgs.msg import Bool, Empty
from pynput import keyboard
from robot_arm_both_arms_user_study import point_to_rcm


class SendDummyTipPose(Node):

    def __init__(self):
        super().__init__("mtm_setup")
        self.coag_pressed = None

        # Gripper and trigger flags
        self.gripper_closed_l = False
        self.gripper_triggered_l = False
        self.gripper_closed_r = False
        self.gripper_triggered_r = False

        # Clutch states
        self.clutch_pressed = False
        self.prev_clutch_pressed = False

        # coag states 
        self.coag_pressed = 1

        # Latest orientation caches
        self.latest_orientation_l = Quaternion()
        self.latest_orientation_r = Quaternion()

        # Subscriptions

        self.subscribe_gripper_closed_l = self.create_subscription(Bool, "/MTML/gripper/closed", self.gripper_callback_l, 10)
        self.subscribe_gripper_closed_r = self.create_subscription(Bool, "/MTMR/gripper/closed", self.gripper_callback_r, 10)
        self.subscribe_pose_l = self.create_subscription(PoseStamped, "/MTML/measured_cp", self.pose_callback_l, 10)
        self.subscribe_pose_r = self.create_subscription(PoseStamped, "/MTMR/measured_cp", self.pose_callback_r, 10)
        self.subscribe_joint_r = self.create_subscription(JointState, "/MTMR/measured_js", self.joint_callback_r, 10)
        self.subscribe_joint_l = self.create_subscription(JointState, "/MTML/measured_js", self.joint_callback_l, 10)

        # Publishers
        self.publish_servo_cf_r = self.create_publisher(WrenchStamped, '/MTMR/spatial/servo_cf', 10)
        self.publish_use_gravity_comp_r = self.create_publisher(Bool, 'MTMR/use_gravity_compensation', 10)
        self.publish_hold_arm_r = self.create_publisher(Empty, 'MTMR/hold', 10)
        self.publish_lock_orientation_r = self.create_publisher(Quaternion, '/MTMR/lock_orientation', 10)
        self.publish_unlock_orientation_r = self.create_publisher(Empty, '/MTMR/unlock_orientation', 10)
        self.publish_reset_joint_r = self.create_publisher(JointState, '/MTMR/servo_jp', 10)

        self.publish_servo_cf_l = self.create_publisher(WrenchStamped, '/MTML/spatial/servo_cf', 10)
        self.publish_use_gravity_comp_l = self.create_publisher(Bool, 'MTML/use_gravity_compensation', 10)
        self.publish_hold_arm_l = self.create_publisher(Empty, 'MTML/hold', 10)
        self.publish_lock_orientation_l = self.create_publisher(Quaternion, '/MTML/lock_orientation', 10)
        self.publish_unlock_orientation_l = self.create_publisher(Empty, '/MTML/unlock_orientation', 10)
        self.publish_reset_joint_l = self.create_publisher(JointState, '/MTML/servo_jp', 10)

        self.coag_pub = self.create_publisher(Joy, "/footpedals/coag", 10)
        self.clutch_pub = self.create_publisher(Joy, "/footpedals/clutch", 10)

        self.timer = self.create_timer(0.01, self.timer_function)

        # Track key states
        self.key_states = {
            'c': False,  # coag
            'a': False   # clutch
        }

        # Start keyboard listener
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

        self.get_logger().info("Tracking 'a' (coag) and 'c' (clutch) states...")

    def on_press(self, key):
        try:
            if key.char in self.key_states:
                self.key_states[key.char] = True
        except AttributeError:
            pass  # Ignore special keys

    def on_release(self, key):
        try:
            if key.char in self.key_states:
                self.key_states[key.char] = False
        except AttributeError:
            pass

    def publish_joy(self, publisher, value):
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = [0.0]  # example placeholder
        msg.buttons = [int(value)]
        publisher.publish(msg)

    def reset_mtm_joints(self):
        r_target = [0.0] * 7
        r_target[-2] = 0.4
        l_target = [0.0] * 7
        l_target[-2] = -0.4
        steps = 200
        sleep_time = 0.01

        js_r_traj = np.linspace(np.array(self.latest_joint_state_r.position), np.array(r_target), steps)
        js_l_traj = np.linspace(np.array(self.latest_joint_state_l.position), np.array(l_target), steps)
        # import pdb; pdb.set_trace()  # Debugging breakpoint
        for i in range(steps):
            js_r = JointState()
            js_r.position = js_r_traj[i].tolist()
            self.publish_reset_joint_r.publish(js_r)

            js_l = JointState()
            js_l.position = js_l_traj[i].tolist()
            self.publish_reset_joint_l.publish(js_l)

            time.sleep(sleep_time)

            
    def timer_function(self):
        
        self.publish_joy(self.coag_pub, self.key_states['c'])
        self.publish_joy(self.clutch_pub, self.key_states['a'])
    
        if self.key_states['c'] == False:
            self.gripper_triggered_l = False
            self.gripper_triggered_r = False

        # Always enable gravity compensation
        g = Bool()
        g.data = True
        self.publish_use_gravity_comp_l.publish(g)
        self.publish_use_gravity_comp_r.publish(g)

        self.clutch_pressed = self.key_states['a']
        # Detect clutch edge trigger (for lock/unlock orientation)
        if self.clutch_pressed and not self.prev_clutch_pressed:
            # LEFT arm
            if self.coag_pressed == 1 and self.gripper_triggered_l:
                self.publish_lock_orientation_l.publish(self.latest_orientation_l)

            # RIGHT arm
            if self.coag_pressed == 1 and self.gripper_triggered_r:
                self.publish_lock_orientation_r.publish(self.latest_orientation_r)

        elif not self.clutch_pressed and self.prev_clutch_pressed:
            # Clutch released, unlock orientation
            self.publish_unlock_orientation_l.publish(Empty())
            self.publish_unlock_orientation_r.publish(Empty())

        # Update previous clutch state
        self.prev_clutch_pressed = self.clutch_pressed

        # Regular servo/hold logic
        if self.coag_pressed == 1:
            # LEFT arm
            if self.gripper_triggered_l:
                servo_cf_l = WrenchStamped()
                servo_cf_l.wrench.force = Vector3(x=0.0, y=0.0, z=0.0)
                self.publish_servo_cf_l.publish(servo_cf_l)
            else:
                self.publish_hold_arm_l.publish(Empty())

            # RIGHT arm
            if self.gripper_triggered_r:
                servo_cf_r = WrenchStamped()
                servo_cf_r.wrench.force = Vector3(x=0.0, y=0.0, z=0.0)
                self.publish_servo_cf_r.publish(servo_cf_r)
            else:
                self.publish_hold_arm_r.publish(Empty())
        else:
            # Reset both arms to hold
            self.publish_hold_arm_l.publish(Empty())
            self.publish_hold_arm_r.publish(Empty())

    
    def joint_callback_r(self, msg: JointState):
        self.latest_joint_state_r = msg

    def joint_callback_l(self, msg: JointState):
        self.latest_joint_state_l = msg

    def coag_callback(self, coag_value: Joy):
        self.coag_pressed = coag_value.buttons[0]
        if self.coag_pressed == 0:
            self.gripper_triggered_l = False
            self.gripper_triggered_r = False

    def clutch_callback(self, msg: Joy):
        self.clutch_pressed = bool(msg.buttons[0])  # Assuming button[0] is clutch

    def gripper_callback_l(self, msg: Bool):
        self.gripper_closed_l = msg.data
        if self.coag_pressed == 1 and self.gripper_closed_l:
            self.gripper_triggered_l = True

    def gripper_callback_r(self, msg: Bool):
        self.gripper_closed_r = msg.data
        if self.coag_pressed == 1 and self.gripper_closed_r:
            self.gripper_triggered_r = True

    def pose_callback_l(self, msg: PoseStamped):
        self.latest_orientation_l = msg.pose.orientation

    def pose_callback_r(self, msg: PoseStamped):
        self.latest_orientation_r = msg.pose.orientation

def main(args=None):
    rclpy.init(args=args)
    node = SendDummyTipPose()

    # Wait until joint states are received
    while not hasattr(node, 'latest_joint_state_r') or not hasattr(node, 'latest_joint_state_l'):
        rclpy.spin_once(node, timeout_sec=0.1)

    # Smoothly reset MTMs to zero position
    node.reset_mtm_joints()

    # Enter normal spin loop
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
