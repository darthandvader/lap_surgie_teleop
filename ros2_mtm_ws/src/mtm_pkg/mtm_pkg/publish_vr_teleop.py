# vr_to_dvrk_bridge.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Joy
import numpy as np
from typing import Dict, Tuple, Any
from oculus_reader.reader import OculusReader

def quat_from_rot(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Robust rotation-matrix → quaternion (w, x, y, z)."""
    # Ensure numeric stability
    R = R.astype(np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2,1] - R[1,2]) / s
        qy = (R[0,2] - R[2,0]) / s
        qz = (R[1,0] - R[0,1]) / s
    else:
        i = np.argmax([R[0,0], R[1,1], R[2,2]])
        if i == 0:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
    # Normalize to avoid drift
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q) + 1e-12
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])

def pose_from_T(T: np.ndarray, frame_id: str) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame_id
    ps.header.stamp.sec = 0
    ps.header.stamp.nanosec = 0
    
    # 1) your existing mapping (both R and t)
    T_rot_mtm = np.array([
        [-1., 0., 0.],  # new x = -old x
        [ 0., 0., 1.],  # new y =  old z
        [ 0., 1., 0.],  # new z =  old y
    ], dtype=float)

    R1 = T_rot_mtm @ T[:3, :3]
    t1 = T_rot_mtm @ T[:3, 3]   # keep translation from your previous mapping

    Sxy = np.array([
        [ 0.,  1., 0.],
        [-1.,  0., 0.],
        [ 0.,  0., 1.],
    ], dtype=float)

    # If signs look flipped, try the -90° version:
    # Sxy = np.array([[0., -1., 0.],
    #                 [1.,  0., 0.],
    #                 [0.,  0., 1.]], dtype=float)

    # IMPORTANT: post-multiply to change the CHILD frame (column permutation)
    R_final = R1 @ Sxy
    t_final = t1

    ps.pose.position.x = float(t_final[0])
    ps.pose.position.y = float(t_final[1])
    ps.pose.position.z = float(t_final[2])

    qw, qx, qy, qz = quat_from_rot(R_final)   # your helper (w, x, y, z)
    ps.pose.orientation.w = qw
    ps.pose.orientation.x = qx
    ps.pose.orientation.y = qy
    ps.pose.orientation.z = qz
    return ps

class VrToDvrkBridge(Node):
    """
    Bridges your VR packet:
      ({'l': 4x4, 'r': 4x4}, {'A': bool, ..., 'LG': bool/tuple, ...})
    to topics:
      - /MTMR/measured_cp : PoseStamped (right)
      - /MTML/measured_cp : PoseStamped (left)
      - /footpedals/clutch: Joy (buttons[0] is clutch state)
    """
    def __init__(self):
        super().__init__("vr_to_dvrk_bridge")

        # Publishers to match your subscribers
        self.pub_mtmr = self.create_publisher(PoseStamped, "/MTMR/measured_cp", 10)
        self.pub_mtml = self.create_publisher(PoseStamped, "/MTML/measured_cp", 10)
        self.pub_clutch = self.create_publisher(Joy, "/footpedals/clutch", 10)

        # Publish at 60 Hz (adjust to your VR stream rate)
        self.timer = self.create_timer(1.0 / 30.0, self.tick)

        self.oculus_reader = OculusReader()

    # If you have your own VR feed, call this to update inputs each tick.
    def get_vr_packet(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        # Replace with: l_T, r_T, buttons = your_live_feed()
        msg = self.oculus_reader.get_transformations_and_buttons()
        transform, buttons = msg
        l_T = transform['l']
        r_T = transform['r']
        print(l_T, r_T, buttons)
        return l_T, r_T, buttons

    def tick(self):
        try:
            l_T, r_T, buttons = self.get_vr_packet()

            # Map controllers → dVRK topics:
            #  - Right controller → /MTMR/measured_cp
            #  - Left  controller → /MTML/measured_cp
            msg_mtmr = pose_from_T(r_T, frame_id="vr_world")
            msg_mtml = pose_from_T(l_T, frame_id="vr_world")
            self.pub_mtmr.publish(msg_mtmr)
            self.pub_mtml.publish(msg_mtml)

            # Clutch mapping: use left grip ("LG") → buttons[0]
            # (Change "LG" → your preferred button, e.g., "RG" or "LTr")
            lg = buttons.get("LG", False)
            clutch_pressed = int(bool(lg))  # 1 if pressed, else 0

            joy = Joy()
            joy.axes = []        # your subscriber ignores axes
            joy.buttons = [clutch_pressed]  # subscriber reads buttons[0]
            self.pub_clutch.publish(joy)
        except Exception as e:
            self.get_logger().error(f"Error in tick: {e}")

def main():
    rclpy.init()
    node = VrToDvrkBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()