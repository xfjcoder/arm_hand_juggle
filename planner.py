"""Arm kinematics for in-flight planning, on a scratch MjData so it never touches the live simulation."""
import mujoco as mj
import numpy as np

import kin


class Planner:
    """Kinematics on a scratch MjData, so in-flight planning never touches the live simulation."""

    def __init__(self, J):
        self.J = J
        self.d = mj.MjData(J.m)

    def fk(self, q):
        m, d, ids = self.J.m, self.d, self.J.ids
        d.qpos[ids.arm_q] = q
        mj.mj_kinematics(m, d)
        mj.mj_comPos(m, d)
        s = self.J.site
        return d.site_xpos[s].copy(), d.site_xmat[s].reshape(3, 3).copy()

    def ball_point(self, q):
        p, R = self.fk(q)
        return p + R @ self.J.ball_off

    def jac6(self, q):
        """6 x 6: ball-point linear velocity and palm angular velocity per arm joint rate."""
        m, d, ids = self.J.m, self.d, self.J.ids
        p, R = self.fk(q)
        pt = p + R @ self.J.ball_off
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mj.mj_jac(m, d, jp, jr, pt, m.site_bodyid[self.J.site])
        return np.vstack([jp[:, ids.arm_v], jr[:, ids.arm_v]])

    def ik_ball(self, target, R_t, q0, iters=60):
        """Arm joints putting the resting-ball point at `target` with palm orientation R_t."""
        q = q0.copy()
        for _ in range(iters):
            p, R = self.fk(q)
            e = np.r_[target - (p + R @ self.J.ball_off), kin.rot_err(R, R_t)]
            if np.linalg.norm(e) < 1e-5:
                break
            Jm = self.jac6(q)
            q = q + Jm.T @ np.linalg.solve(Jm @ Jm.T + 1e-4 * np.eye(6), e)
        return q


class BallFit:
    """Least-squares ballistic fit of the measured ball positions since release: the position and velocity
    used for planning come from the fit, not from one (noisy) sample or the simulator's true velocity."""

    def __init__(self, g=9.81, window=0.5):
        self.g, self.window = g, window
        self.t, self.p = [], []

    def ready(self, span=0.02):
        """Enough samples (at least `span` seconds of them) for a usable velocity."""
        return len(self.t) > 1 and self.t[-1] - self.t[0] >= span

    def add(self, t, p):
        self.t.append(t)
        self.p.append(np.array(p, float))

    def state(self, t):
        ts = np.array(self.t)
        keep = ts >= ts[-1] - self.window
        ts, ps = ts[keep] - t, np.array(self.p)[keep]
        ps = ps + np.outer(0.5 * self.g * ts ** 2, [0, 0, 1])     # remove gravity: p + g s^2/2 = p0 + v0 s
        A = np.c_[np.ones_like(ts), ts]
        if len(ts) < 2:
            return ps[-1], np.zeros(3)
        c, *_ = np.linalg.lstsq(A, ps, rcond=None)
        return c[0], c[1]
