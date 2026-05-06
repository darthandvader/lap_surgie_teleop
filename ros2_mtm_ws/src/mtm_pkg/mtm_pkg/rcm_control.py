import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from numba import njit

def rot_x(theta: float) -> np.ndarray:
    """Rotation about X (right‑hand rule, radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta: float) -> np.ndarray:
    """Rotation about Y (right‑hand rule, radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta: float) -> np.ndarray:
    """Rotation about Z (right‑hand rule, radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# Core mapping


def set_axes_equal(ax):
    """Set equal aspect ratio for 3D axes."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max([x_range, y_range, z_range])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])



def _plot_pose(ax, T: np.ndarray, name: str, length: float = 0.04):
    colors = ["r", "g", "b"]
    origin = T[:3, 3]
    R = T[:3, :3]
    for i in range(3):
        ax.quiver(
            *origin, *(R[:, i] * length), color=colors[i], label=f'{name}-{"XYZ"[i]}'
        )

@njit(cache=True, fastmath=True)
def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,    c,   -s],
                     [0.0,    s,    c]], dtype=np.float64)

@njit(cache=True, fastmath=True)
def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[   c, 0.0,    s],
                     [ 0.0, 1.0,  0.0],
                     [  -s, 0.0,    c]], dtype=np.float64)

@njit(cache=True, fastmath=True)
def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[   c,   -s, 0.0],
                     [   s,    c, 0.0],
                     [ 0.0,  0.0, 1.0]], dtype=np.float64)


@njit(cache=True, fastmath=True)
def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x




def set_axes_equal(ax):
    xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    max_range = max(abs(xl[1] - xl[0]), abs(yl[1] - yl[0]), abs(zl[1] - zl[0]))
    xm, ym, zm = [np.mean(l) for l in [xl, yl, zl]]
    ax.set_xlim3d(xm - max_range / 2, xm + max_range / 2)
    ax.set_ylim3d(ym - max_range / 2, ym + max_range / 2)
    ax.set_zlim3d(zm - max_range / 2, zm + max_range / 2)


def plot_frame(ax, T, name="", length=0.05, linewidth=2):
    origin = T[:3, 3]
    R = T[:3, :3]
    colors = ["r", "g", "b"]
    labels = ["X", "Y", "Z"]
    for i in range(3):
        ax.quiver(*origin, *(R[:, i] * length), color=colors[i], linewidth=linewidth)
        ax.text(
            *(origin + R[:, i] * length * 1.15), f"{name}{labels[i]}", color=colors[i]
        )


# ---------- forward model (user routine) ----------
@njit(cache=True, fastmath=True)
def compute_shaft2_pose(
    handle1_pose, xcm, l12=0.1, l1=0.6, k=2, return_intermediate=False
):  # k = 2
    Rh1, xh1 = handle1_pose[:3, :3], handle1_pose[:3, 3]
    n1 = Rh1 @ np.array([0.0, 0.0, 1.0])
    n2 = xcm - xh1
    dir_h1h2 = np.cross(n1, np.cross(n1, n2))
    dir_h1h2 /= np.linalg.norm(dir_h1h2)
    xh2 = xh1 + l12 * dir_h1h2
    l0 = np.linalg.norm(xh2 - xcm)
    xs2 = xh2 + (l1 / l0) * (xcm - xh2)
    cos_t1 = clamp(
        (l12**2 + l0**2 - np.linalg.norm(xcm - xh1) ** 2) / (2 * l12 * l0), -1.0, 1.0
    )
    t1 = np.arccos(cos_t1)
    v1 = xh1 - xh2
    v2 = Rh1 @ np.array([1.0, 0.0, 0.0])
    cos_t2 = clamp(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    t2 = np.arccos(cos_t2)
    t1p = (
        np.sign(np.dot(np.cross(xcm - xh2, xh1 - xh2), Rh1 @ np.array([0.0, 0.1, 0.0])))
        * t1
    )
    t2p = (
        np.sign(
            np.dot(
                np.cross(xh1 - xh2, Rh1 @ np.array([0.1, 0.0, 0.0])),
                Rh1 @ np.array([0.0, 0.0, 0.1]),
            )
        )
        * t2
    )
    Rs2 = Rh1 @ rot_z(-t2p) @ rot_y(-t1p) @ rot_y(k * t1p) @ rot_z(k * t2p)
    Ps2 = np.eye(4)
    Ps2[:3, :3] = Rs2
    Ps2[:3, 3] = xs2
    # if return_intermediate:
    return Ps2, xh2, t1p, t2p
    # return Ps2


# ---------- Lie helpers ----------
def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_map(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def log_map(R):
    # th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    # if th < 1e-12:
    #     return np.zeros(3)
    # return (
    #     th
    #     / (2 * np.sin(th))
    #     * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    # )
    # Map SO(3) -> so(3) rotation vector
    tr = R[0,0] + R[1,1] + R[2,2]
    c = 0.5 * (tr - 1.0)
    c = np.clip(c, -1.0, 1.0)
    th = np.arccos(c)
    v = np.array([R[2,1] - R[1,2],
                  R[0,2] - R[2,0],
                  R[1,0] - R[0,1]], dtype=np.float64)
    if th < 1e-6:
        # w ≈ 1/2 vee(R - Rᵀ)
        return 0.5 * v
    s = np.sin(th)
    return (th / (2.0 * s)) * v


def vec2pose(q):
    T = np.eye(4)
    T[:3, 3] = q[:3]
    T[:3, :3] = exp_map(q[3:])
    return T


def pose2vec(T):
    return np.hstack([T[:3, 3], log_map(T[:3, :3])])


from scipy.optimize import least_squares


# def invert_shaft2_pose(Ps2_meas, xcm, l12=0.1, l1=0.6, k=2):
#     Rs2, xs2 = Ps2_meas[:3, :3], Ps2_meas[:3, 3]
#     axis = xcm - xs2
#     n_axis = axis / np.linalg.norm(axis)
#     xh2_seed = xs2 + l1 * n_axis
#     perp = np.cross([0, 1, 0], n_axis)
#     if np.linalg.norm(perp) < 1e-6:
#         perp = np.cross([1, 0, 0], n_axis)
#     perp /= np.linalg.norm(perp)
#     xh1_seed = xh2_seed + l12 * perp
#     q0 = pose2vec(np.block([[np.eye(3), xh1_seed[:, None]], [0, 0, 0, 1]]))

#     def resid(q):
#         H1 = vec2pose(q)
#         Ps2_hat = compute_shaft2_pose(H1, xcm, l12, l1, k)
#         return np.hstack([Ps2_hat[:3, 3] - xs2, log_map(Ps2_hat[:3, :3].T @ Rs2)])

#     sol = least_squares(resid, q0, max_nfev=100, xtol=1e-12, ftol=1e-12)
#     return vec2pose(sol.x)



# def invert_shaft2_pose_angle_limits(Ps2_meas, xcm, l12=0.1, l1=0.6, k=2, w_t=10.0, H1_prev=None):
#     Rs2, xs2 = Ps2_meas[:3, :3], Ps2_meas[:3, 3]
#     axis = xcm - xs2
#     n_axis = axis / np.linalg.norm(axis)
#     xh2_seed = xs2 + l1 * n_axis
#     perp = np.cross([0, 1, 0], n_axis)
#     if np.linalg.norm(perp) < 1e-6:
#         perp = np.cross([1, 0, 0], n_axis)
#     perp /= np.linalg.norm(perp)
#     xh1_seed = xh2_seed + l12 * perp
#     if H1_prev is None:
#         q0 = pose2vec(np.block([[np.eye(3), xh1_seed[:, None]],
#                                 [0, 0, 0, 1]]))
#     else:
#         q0 = pose2vec(H1_prev)

#     def resid(q):
#         H1  = vec2pose(q)
#         Ps2, _, t1p, t2p = compute_shaft2_pose(H1, xcm, l12, l1, k, return_intermediate=True)
       
#         # print(f"t1p: {np.angle(t1p)}, t2p: {np.angle(t2p)}")
#         r_pos = w_t * (Ps2[:3, 3] - xs2)          # weighted
#         r_rot = log_map(Ps2[:3, :3].T @ Rs2)
#         w_limt = 30.0
#         MAX_T = np.pi/4
#         penalty = w_limt * np.array([
#         max(0.0, abs(t1p) - MAX_T),
#         max(0.0, abs(t2p) - MAX_T),
#         ])
        
#         return np.hstack([r_pos, r_rot, penalty])

#     sol = least_squares(resid, q0, max_nfev=40,
#                         xtol=1e-10, ftol=1e-10)
#     # return vec2pose(sol.x), sol.cost
#     return vec2pose(sol.x)



def invert_shaft2_pose_angle_limits(Ps2_meas, xcm, l12=0.1, l1=0.6, k=2, w_t=10.0, H1_prev=None, w_smooth=1.0):

    Rs2, xs2 = Ps2_meas[:3, :3], Ps2_meas[:3, 3]
    axis = xcm - xs2
    n_axis = axis / np.linalg.norm(axis)
    xh2_seed = xs2 + l1 * n_axis

    # Get a perpendicular direction for initial seed
    perp = np.cross([0, 1, 0], n_axis)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross([1, 0, 0], n_axis)
    perp /= np.linalg.norm(perp)
    xh1_seed = xh2_seed + l12 * perp

    # Initial pose seed q0
    if H1_prev is None:
        q0 = pose2vec(np.block([[np.eye(3), xh1_seed[:, None]],
                                [0, 0, 0, 1]]))
    else:
        q0 = pose2vec(H1_prev)

    def resid(q):
        H1 = vec2pose(q)
        t0 = time.time()
        Ps2, _, t1p, t2p = compute_shaft2_pose(H1, xcm, l12, l1, k, return_intermediate=True)
        t1 = time.time()
        # print(f"compute_shaft2_pose time: {t1 - t0:.6f} seconds")
        # Primary residuals
        r_pos = w_t * (Ps2[:3, 3] - xs2)
        r_rot = log_map(Ps2[:3, :3].T @ Rs2)

        # Joint limit penalty
        w_limt = 30.0
        MAX_T = np.pi / 4
        penalty = w_limt * np.array([
            max(0.0, abs(t1p) - MAX_T),
            max(0.0, abs(t2p) - MAX_T)
        ])

        # Temporal smoothing regularizer
        # if H1_prev is not None:
        #     q_prev = pose2vec(H1_prev)
        #     r_smooth = w_smooth * (q - q_prev)
        #     return np.hstack([r_pos, r_rot, penalty, r_smooth])
        # else:
        return np.hstack([r_pos, r_rot, penalty])

    # sol = least_squares(resid, q0, max_nfev=10, xtol=1e-6, ftol=1e-6)
    sol = least_squares(resid, q0, method='trf', max_nfev=3, xtol=1e-3, ftol=1e-3)

    return vec2pose(sol.x)





def invert_shaft2_pose(Ps2_meas, xcm, l12=0.1, l1=0.6, k=2, w_t=10.0, H1_prev=None):


    Rs2, xs2 = Ps2_meas[:3, :3], Ps2_meas[:3, 3]
    axis = xcm - xs2
    n_axis = axis / np.linalg.norm(axis)
    xh2_seed = xs2 + l1 * n_axis
    perp = np.cross([0, 1, 0], n_axis)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross([1, 0, 0], n_axis)
    perp /= np.linalg.norm(perp)
    xh1_seed = xh2_seed + l12 * perp
    if H1_prev is None:
        q0 = pose2vec(np.block([[np.eye(3), xh1_seed[:, None]],
                                [0, 0, 0, 1]]))
    else:
        q0 = pose2vec(H1_prev)

    def resid(q):
        H1  = vec2pose(q)
        Ps2 = compute_shaft2_pose(H1, xcm, l12, l1, k)
        r_pos = w_t * (Ps2[:3, 3] - xs2)          # weighted
        r_rot = log_map(Ps2[:3, :3].T @ Rs2)
        return np.hstack([r_pos, r_rot])

    sol = least_squares(resid, q0, max_nfev=80,
                        xtol=1e-10, ftol=1e-10)
    # return vec2pose(sol.x), sol.cost
    return vec2pose(sol.x)


import pinocchio as pin
from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":

    H1 = np.eye(4)
    H1[:3, 3] = np.array([0, 0, 0])
    H1[:3, :3] = rot_y(np.pi / 8) @ rot_z(np.pi / 8)  # 45° rotation around Z
    xcm = np.array([0.3, 0, 0])

    Ps2, xh2, theta1, theta2 = compute_shaft2_pose(
        H1, xcm, visualize=True, return_intermediate=True
    )

    # print("\nComputed Shaft‑2 pose:\n", Ps2)

    z_vector = H1[:3, :3] @ np.array([0, 0, 1])
    h1_h2 = H1[:3, 3] - xh2

    # print(
    #     f"check angle:{np.degrees(np.arccos(np.clip(np.dot(z_vector, h1_h2) / (np.linalg.norm(z_vector) * np.linalg.norm(h1_h2)), -1.0, 1.0)))}"
    # )





