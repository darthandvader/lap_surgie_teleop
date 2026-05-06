import roslibpy

# import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
# from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32
# from std_msgs import Bool
from sensor_msgs.msg import JointState


import base64
import rclpy
from rclpy.node import Node


class PoseRepublisher(Node):
    def __init__(self):
        super().__init__('pose_republisher')
        self.Lpose = self.create_publisher(PoseStamped, '/MTML/measured_cp_ros2', 10)
        self.Rpose = self.create_publisher(PoseStamped, '/MTMR/measured_cp_ros2', 10)
        self.gripperLjs = self.create_publisher(JointState, '/MTML/gripper/measured_js_ros2', 10)
        self.gripperRjs = self.create_publisher(JointState, '/MTMR/gripper/measured_js_ros2', 10)

        self.client = roslibpy.Ros(host='25.20.60.167', port=9090)
        self.client.run()

        self.Lpose_listener = roslibpy.Topic(self.client, '/MTML/measured_cp', 'geometry_msgs/PoseStamped')
        self.Rpose_listener = roslibpy.Topic(self.client, '/MTMR/measured_cp', 'geometry_msgs/PoseStamped')
        self.gripperLjs_listener = roslibpy.Topic(self.client, 'MTML/gripper/measured_js', 'sensor_msgs/JointState')
        self.gripperRjs_listener = roslibpy.Topic(self.client, 'MTMR/gripper/measured_js', 'sensor_msgs/JointState')

        self.Lpose_listener.subscribe(lambda msg, pub=self.Lpose: self.on_message_pose(msg, pub))
        self.Rpose_listener.subscribe(lambda msg, pub=self.Rpose: self.on_message_pose(msg, pub))
        self.gripperLjs_listener.subscribe(lambda msg, pub=self.gripperLjs: self.on_message_gripper(msg, pub))
        self.gripperRjs_listener.subscribe(lambda msg, pub=self.gripperRjs: self.on_message_gripper(msg, pub))

    def on_message_gripper(self, msg, pub):
        gripper_msg = JointState()

        gripper_msg.header.stamp = self.get_clock().now().to_msg()
        gripper_msg.name = msg.get('name', [])
        gripper_msg.position = [float(pos) for pos in msg.get('position', [])]
        gripper_msg.velocity = [float(vel) for vel in msg.get('velocity', [])]
        gripper_msg.effort = [float(eff) for eff in msg.get('effort', [])]

        print("Republishing gripper joint state to ROS2 topic: ", gripper_msg)
        pub.publish(gripper_msg)

    def on_message_pose(self, msg, pub):
        pose_msg = PoseStamped()

        header = msg.get('header', {})
        pose_msg.header.frame_id = header.get('frame_id', '')
        pose_msg.header.stamp = self.get_clock().now().to_msg()

        pose = msg.get('pose', {})
        position = pose.get('position', {})
        orientation = pose.get('orientation', {})

        pose_msg.pose.position.x = float(position.get('x', 0.0))
        pose_msg.pose.position.y = float(position.get('y', 0.0))
        pose_msg.pose.position.z = float(position.get('z', 0.0))

        pose_msg.pose.orientation.x = float(orientation.get('x', 0.0))
        pose_msg.pose.orientation.y = float(orientation.get('y', 0.0))
        pose_msg.pose.orientation.z = float(orientation.get('z', 0.0))
        pose_msg.pose.orientation.w = float(orientation.get('w', 1.0))

        print("Republishing pose to ROS2 topic: ", pose_msg)
        pub.publish(pose_msg)

def main():
    rclpy.init()
    node = PoseRepublisher()a


    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.Lpose_listener.unsubscribe()
        node.Rpose_listener.unsubscribe()
        node.gripperLjs_listener.unsubscribe()
        node.gripperRjs_listener.unsubscribe()
        node.client.terminate()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()