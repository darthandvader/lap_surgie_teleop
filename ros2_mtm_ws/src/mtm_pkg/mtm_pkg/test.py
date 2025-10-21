

# from rcm_control import *
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_map(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)



def log_map(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-12:
        return np.zeros(3)
    return (
        th
        / (2 * np.sin(th))
        * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    )


def vec2pose(q):
    T = np.eye(4)
    T[:3, 3] = q[:3]
    T[:3, :3] = exp_map(q[3:])
    return T


def pose2vec(T):
    return np.hstack([T[:3, 3], log_map(T[:3, :3])])


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])



# ---------- forward model (user routine) ----------
def compute_shaft2_pose(
    handle1_pose, xcm, l12=0.1, l1=0.6, k=2, return_intermediate=False
):
    Rh1, xh1 = handle1_pose[:3, :3], handle1_pose[:3, 3]
    n1 = Rh1 @ np.array([0, 0, 1])
    n2 = xcm - xh1
    dir_h1h2 = np.cross(n1, np.cross(n1, n2))
    dir_h1h2 /= np.linalg.norm(dir_h1h2)
    xh2 = xh1 + l12 * dir_h1h2
    l0 = np.linalg.norm(xh2 - xcm)
    xs2 = xh2 + (l1 / l0) * (xcm - xh2)
    cos_t1 = np.clip(
        (l12**2 + l0**2 - np.linalg.norm(xcm - xh1) ** 2) / (2 * l12 * l0), -1, 1
    )
    t1 = np.arccos(cos_t1)
    v1 = xh1 - xh2
    v2 = Rh1 @ np.array([1, 0, 0])
    cos_t2 = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
    t2 = np.arccos(cos_t2)
    t1p = (
        np.sign(np.dot(np.cross(xcm - xh2, xh1 - xh2), Rh1 @ np.array([0, 0.1, 0])))
        * t1
    )
    t2p = (
        np.sign(
            np.dot(
                np.cross(xh1 - xh2, Rh1 @ np.array([0.1, 0, 0])),
                Rh1 @ np.array([0, 0, 0.1]),
            )
        )
        * t2
    )
    Rs2 = Rh1 @ rot_z(-t2p) @ rot_y(-t1p) @ rot_y(k * t1p) @ rot_z(k * t2p)
    Ps2 = np.eye(4)
    Ps2[:3, :3] = Rs2
    Ps2[:3, 3] = xs2
    if return_intermediate:
        return Ps2, xh2
    return Ps2


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




Ps2_gt = np.eye(4)
Ps2_gt[:3, :3] = Rotation.from_euler('xyz', [-48.45016264938367, 26.840908061425587, -8.16938475209237], degrees=True).as_matrix()
Ps2_gt[:3, 3] = np.array([0.05, 0.00, -0.05]) + np.array([0.64, -0.05, 0.02]) + np.array([-0.03974, 0.09816, 0.00356])
rcm = np.array([0.515, -0.08, -0.01])

# Ps2_gt, xh2_gt = compute_shaft2_pose(H1_true, rcm, return_intermediate=True)
H1_est = invert_shaft2_pose(Ps2_gt, rcm)
# Ps2_est, xh2_est = compute_shaft2_pose(H1_est, rcm, return_intermediate=True)

# robot_pos = required_robot_pose[:3, 3]
# robot_to_pivot_vec = pivot - robot_pos
print(H1_est)


# before jerkiness

Ps2_gt = np.eye(4)
Ps2_gt[:3, :3] = Rotation.from_euler('xyz', [-48.47035894046265, 26.63143145100563, -8.20849740575957], degrees=True).as_matrix()
Ps2_gt[:3, 3] = np.array([0.05, 0.00, -0.05]) + np.array([0.64, -0.05, 0.02]) + np.array([-0.03998,  0.09857,  0.00335])
rcm = np.array([0.515, -0.08, -0.01])

# Ps2_gt, xh2_gt = compute_shaft2_pose(H1_true, rcm, return_intermediate=True)
H1_est= invert_shaft2_pose(Ps2_gt, rcm)
# Ps2_est, xh2_est = compute_shaft2_pose(H1_est, rcm, return_intermediate=True)

# robot_pos = required_robot_pose[:3, 3]
# robot_to_pivot_vec = pivot - robot_pos
print(H1_est)