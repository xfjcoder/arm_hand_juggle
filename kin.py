"""Kinematics of the palm: pose, Jacobian, damped least-squares IK over the six arm joints.

The palm frame is the hand's `grasp_site` (centre of the palm's inner face): its z axis points along the
fingers and its -y axis out of the palm. "Palm up, fingers forward" is PALM_UP below.
"""
import mujoco as mj
import numpy as np

from scene import ARM_JOINTS, Ids

PALM_SITE = "grasp_site"
# columns = palm x, y, z in world: fingers along +x, palm normal (-y) along +z
PALM_UP = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
HOME = np.array([0.0, -1.9, 1.9, -1.57, -1.57, 0.0])      # rough seed: elbow up, wrist bent down


def palm_pose(m, d):
    s = m.site(PALM_SITE).id
    return d.site_xpos[s].copy(), d.site_xmat[s].reshape(3, 3).copy()


def palm_jac(m, d, ids):
    """6 x 6: palm linear (rows 0-2) and angular (rows 3-5) velocity per arm joint velocity."""
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mj.mj_jacSite(m, d, jp, jr, m.site(PALM_SITE).id)
    return np.vstack([jp[:, ids.arm_v], jr[:, ids.arm_v]])


def rot_err(R, Rt):
    """Rotation vector taking R to Rt, in world coordinates."""
    E = Rt @ R.T
    q = np.zeros(4)
    mj.mju_mat2Quat(q, E.ravel())
    v = np.zeros(3)
    mj.mju_quat2Vel(v, q, 1.0)
    return v


def ik(m, d, ids, pos, R=PALM_UP, q0=None, iters=200, tol=1e-4, damp=1e-3):
    """Arm joint angles putting the palm at (pos, R). Leaves d at that configuration. Returns q, error."""
    q = HOME.copy() if q0 is None else np.array(q0, float)
    for _ in range(iters):
        d.qpos[ids.arm_q] = q
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        p, Rc = palm_pose(m, d)
        e = np.concatenate([pos - p, rot_err(Rc, R)])
        if np.linalg.norm(e) < tol:
            break
        J = palm_jac(m, d, ids)
        q = q + J.T @ np.linalg.solve(J @ J.T + damp * np.eye(6), e)
        q = (q + np.pi) % (2 * np.pi) - np.pi          # keep every joint in [-pi, pi) (elbow's real range)
    mj.mj_forward(m, d)
    return q, float(np.linalg.norm(e))
