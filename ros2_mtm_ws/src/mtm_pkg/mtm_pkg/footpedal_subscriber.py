# my_footpedals_pkg/my_footpedals_pkg/footpedal_subscriber.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy  # Change this import based on your actual message type
from geometry_msgs.msg import PoseStamped

class FootpedalSubscriber(Node):
    def __init__(self):
        super().__init__('footpedal_subscriber')
        # Replace '/footpedals/clutch' with the actual topic name
        self.subscription_footpedal = self.create_subscription(
            Joy,  # Replace with the actual message type
            '/footpedals/clutch',
            self.listener_callback_footpedal,
            10
        )
        self.subscription_controller = self.create_subscription(
            PoseStamped,  # Correct message type
            '/MTML/measured_cp',
            self.listener_callback,
            10
        )
        self.subscription_footpedal  # prevent unused variable warning
        self.subscription_controller
        self.clutch_state = 0
        self.x = 0
        self.y = 0
        self.z = 0

    def listener_callback_footpedal(self, msg):
        # Check the state of the clutch in the 'buttons' array
        if len(msg.buttons) > 0:
            self.clutch_state = msg.buttons[0]  # Assuming the clutch is the first button
            if self.clutch_state == 1:
                self.get_logger().info("Clutch is engaged.")
            else:
                self.get_logger().info("Clutch is disengaged.")
        else:
            self.get_logger().warn("No buttons data available.")
    
    def listener_callback(self, msg):
        print(msg)
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.z = msg.pose.position.z

def main(args=None):
    rclpy.init(args=args)

    footpedal_subscriber = FootpedalSubscriber()

    rclpy.spin(footpedal_subscriber)

    footpedal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()