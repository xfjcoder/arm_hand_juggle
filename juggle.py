"""Palm-space controller and scripted toss / catch for the UR10e + Shadow Hand.

Arm: operational-space computed torque. The palm (grasp_site) tracks a desired position / velocity /
acceleration with fixed palm-up orientation:
    qdd = J^-1 (a_d + Kp e + Kd e_dot - Jdot qdot),   tau = M qdd + bias - passive,  clipped to UR10e limits.
Hand: fingers held in a shallow cup (position servos), wrist joints at zero.
Trajectories: quintic segments in palm space, re-planned during flight from the ball's measured state.
"""
import mujoco as mj
import numpy as np

import kin
import scene

G = 9.81
# shallow cup: fingers slightly curled at the knuckles, little finger's palm joint curled, thumb across
CUP = {"rh_A_WRJ2": 0.0, "rh_A_WRJ1": 0.0,
       "rh_A_THJ5": 0.3, "rh_A_THJ4": 0.9, "rh_A_THJ3": 0.0, "rh_A_THJ2": 0.0, "rh_A_THJ1": 0.3,
       "rh_A_FFJ4": 0.0, "rh_A_FFJ3": 0.35, "rh_A_FFJ0": 0.5,
       "rh_A_MFJ4": 0.0, "rh_A_MFJ3": 0.35, "rh_A_MFJ0": 0.5,
       "rh_A_RFJ4": 0.0, "rh_A_RFJ3": 0.35, "rh_A_RFJ0": 0.5,
       "rh_A_LFJ5": 0.4, "rh_A_LFJ4": 0.0, "rh_A_LFJ3": 0.35, "rh_A_LFJ0": 0.5}
# open hand for the release: fingers straight, thumb out of the palm, so the ball leaves without dragging
FLAT = dict(CUP, **{"rh_A_THJ4": 0.2, "rh_A_THJ1": 0.0, "rh_A_THJ5": 0.0,
                    "rh_A_FFJ3": 0.0, "rh_A_FFJ0": 0.0, "rh_A_MFJ3": 0.0, "rh_A_MFJ0": 0.0,
                    "rh_A_RFJ3": 0.0, "rh_A_RFJ0": 0.0, "rh_A_LFJ5": 0.0, "rh_A_LFJ3": 0.0, "rh_A_LFJ0": 0.0})
# deeper cup for holding the ball while the palm is tilted fingers-down (human-style cocked pose); the thumb
# curls across the palm edge as a wall: with the thumb across the palm the ball could wedge between thumb
# and first finger, off the palm, and a throw from there flies 0.2-0.4 m/s sideways
HOLD = dict(CUP, **{"rh_A_THJ4": 1.2, "rh_A_THJ5": 0.0, "rh_A_THJ2": 0.0, "rh_A_THJ1": 0.0,
                    "rh_A_FFJ3": 0.7, "rh_A_FFJ0": 0.9, "rh_A_MFJ3": 0.7, "rh_A_MFJ0": 0.9,
                    "rh_A_RFJ3": 0.7, "rh_A_RFJ0": 0.9, "rh_A_LFJ3": 0.7, "rh_A_LFJ0": 0.9, "rh_A_LFJ5": 0.5})
# Stage 0 throwing pose: palm at (1.0, 0, 0.3), palm up, fingers along +y (logs/stage0_speed.json)
POSE_Q = np.array([-0.5721, -0.5549, 0.7827, -0.2278, -0.5721, 1.5708])
POSE_R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)   # palm frame columns in world at POSE_Q


def quintic(x0, v0, a0, x1, v1, a1, T):
    """Coefficients (6, dim) of x(s) = sum c_k s^k on s in [0, T] matching both end states."""
    x0, v0, a0, x1, v1, a1 = (np.atleast_1d(np.asarray(v, float)) for v in (x0, v0, a0, x1, v1, a1))
    A = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]])
    rhs = np.stack([x1 - (x0 + v0 * T + a0 * T ** 2 / 2), v1 - (v0 + a0 * T), a1 - a0])
    c345 = np.linalg.solve(A, rhs)
    return np.vstack([x0, v0, a0 / 2, c345])


class Seg:
    def __init__(self, t0, T, start, end):
        self.t0, self.T = t0, T
        self.c = quintic(*start, *end, T)

    def __call__(self, t):
        s = np.clip(t - self.t0, 0, self.T)
        k = np.arange(6)
        p = (self.c.T * s ** k).sum(1)
        v = (self.c[1:].T * (k[1:] * s ** (k[1:] - 1))).sum(1)
        a = (self.c[2:].T * (k[2:] * (k[2:] - 1) * s ** (k[2:] - 2))).sum(1)
        if t - self.t0 > self.T:                     # past the end: hold the end state's position
            v = v * 0
            a = a * 0
        return p, v, a


class Juggler:
    def __init__(self, n_balls=1, kp=600.0, kd=None, kr=600.0, human=False):
        self.m = scene.build(n_balls, human)
        self.d = mj.MjData(self.m)
        self.ids = scene.Ids(self.m)
        self.kp, self.kd = kp, (2 * np.sqrt(kp) if kd is None else kd)
        self.kr, self.kdr = kr, 2 * np.sqrt(kr)
        m = self.m
        self.site = m.site(kin.PALM_SITE).id
        hand = set()
        for b in range(m.nbody):
            x = b
            while x > 0:
                if x == self.ids.forearm:
                    hand.add(b)
                    break
                x = m.body_parentid[x]
        self.hand_bodies = hand
        self.cup = self.posture(CUP)
        self.flat = self.posture(FLAT)
        self.hold = self.posture(HOLD)
        self.hand_target = self.cup
        self.R_d = POSE_R
        self.stats = dict(qd_ratio=0.0, tau_sat=0, steps=0)

    def posture(self, named):
        u = np.zeros(self.m.nu)
        for k, v in named.items():
            u[self.m.actuator(k).id] = v
        return u

    # ---- state ----
    def palm(self):
        m, d = self.m, self.d
        p = d.site_xpos[self.site].copy()
        R = d.site_xmat[self.site].reshape(3, 3).copy()
        J = kin.palm_jac(m, d, self.ids)
        v = J @ d.qvel[self.ids.arm_v]
        return p, R, v, J

    def ball(self, i=0):
        q, v = self.ids.ball_q[i], self.ids.ball_v[i]
        return self.d.qpos[q:q + 3].copy(), self.d.qvel[v:v + 3].copy()

    def ball_touching_hand(self, i=0):
        bid = self.ids.balls[i]
        for c in self.d.contact[:self.d.ncon]:
            b1, b2 = self.m.geom_bodyid[c.geom1], self.m.geom_bodyid[c.geom2]
            if (b1 == bid and b2 in self.hand_bodies) or (b2 == bid and b1 in self.hand_bodies):
                return True
        return False

    # ---- control ----
    def control(self, p_d, v_d, a_d):
        m, d, ids = self.m, self.d, self.ids
        p, R, v6, J = self.palm()
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        mj.mj_jacDot(m, d, jp, jr, d.site_xpos[self.site], m.site_bodyid[self.site])
        Jd = np.vstack([jp[:, ids.arm_v], jr[:, ids.arm_v]])
        qd = d.qvel[ids.arm_v]
        acc = np.concatenate([a_d + self.kp * (p_d - p) + self.kd * (v_d - v6[:3]),
                              self.kr * kin.rot_err(R, self.R_d) - self.kdr * v6[3:]])
        qdd = np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), J.T @ (acc - Jd @ qd))
        M = np.zeros((m.nv, m.nv))
        mj.mj_fullM(m, d, M)
        tau = M[np.ix_(ids.arm_v, ids.arm_v)] @ qdd + d.qfrc_bias[ids.arm_v] - d.qfrc_passive[ids.arm_v]
        sat = np.abs(tau) > scene.TORQUE_MAX
        d.ctrl[ids.arm_u] = np.clip(tau, -scene.TORQUE_MAX, scene.TORQUE_MAX)
        d.ctrl[ids.hand_u] = self.hand_target[ids.hand_u]
        self.stats["qd_ratio"] = max(self.stats["qd_ratio"], float(np.max(np.abs(qd) / scene.JOINT_VMAX)))
        self.stats["tau_sat"] += int(sat.any())
        self.stats["steps"] += 1

    def control_joint(self, q_d, qd_d, qdd_d, kp=400.0, kd=40.0):
        """Joint-space computed torque for the arm; hand held at self.hand_target."""
        m, d, ids = self.m, self.d, self.ids
        q, qd = d.qpos[ids.arm_q], d.qvel[ids.arm_v]
        qdd = qdd_d + kp * (q_d - q) + kd * (qd_d - qd)
        M = np.zeros((m.nv, m.nv))
        mj.mj_fullM(m, d, M)
        tau = M[np.ix_(ids.arm_v, ids.arm_v)] @ qdd + d.qfrc_bias[ids.arm_v] - d.qfrc_passive[ids.arm_v]
        sat = np.abs(tau) > scene.TORQUE_MAX
        d.ctrl[ids.arm_u] = np.clip(tau, -scene.TORQUE_MAX, scene.TORQUE_MAX)
        d.ctrl[ids.hand_u] = self.hand_target[ids.hand_u]
        self.stats["qd_ratio"] = max(self.stats["qd_ratio"], float(np.max(np.abs(qd) / scene.JOINT_VMAX)))
        self.stats["tau_sat"] += int(sat.any())
        self.stats["steps"] += 1

    def reset(self, q=POSE_Q, settle=0.8, ball_offset=(0.0, 0.0, 0.06), grip=None, ball_spot=None):
        """Arm at q, hand cupped, ball dropped onto the palm and allowed to settle."""
        m, d, ids = self.m, self.d, self.ids
        mj.mj_resetData(m, d)
        self.hand_target = self.cup if grip is None else grip
        d.qpos[ids.arm_q] = q
        d.ctrl[ids.hand_u] = self.cup[ids.hand_u]
        for u in ids.hand_u:                          # fingers start at their cup targets
            j = m.actuator_trnid[u, 0]
            if m.actuator_trntype[u] == mj.mjtTrn.mjTRN_JOINT:
                d.qpos[m.jnt_qposadr[j]] = self.cup[u]
        mj.mj_forward(m, d)
        p0, R, _, _ = self.palm()
        self.p_home = p0
        self.R_d = R                                  # hold the pose's own palm orientation
        for i in range(len(ids.balls)):
            q0 = ids.ball_q[i]
            d.qpos[q0:q0 + 3] = (p0 + R @ np.array(ball_spot) if ball_spot is not None   # palm-frame spot
                                 else p0 + np.array(ball_offset) + [0, 0, 0.08 * i])
            d.qpos[q0 + 3:q0 + 7] = [1, 0, 0, 0]
        mj.mj_forward(m, d)
        for _ in range(int(settle / m.opt.timestep)):
            self.control(p0, np.zeros(3), np.zeros(3))
            mj.mj_step(m, d)
        b, _ = self.ball(0)
        p, R, _, _ = self.palm()
        self.ball_off = R.T @ (b - p)                 # resting ball centre in the palm frame
        self.stats = dict(qd_ratio=0.0, tau_sat=0, steps=0)
        return self.ball_off

    def run(self, plan_fn, T, record=None, every=1):
        """Step the sim for T seconds; plan_fn(t) -> (p, v, a)."""
        m, d = self.m, self.d
        n = int(T / m.opt.timestep)
        t0 = d.time
        for k in range(n):
            p, v, a = plan_fn(d.time - t0)
            self.control(p, v, a)
            mj.mj_step(m, d)
            if record is not None and k % every == 0:
                record(self)

    def render(self, distance=1.6, azimuth=160, elevation=-12, lookat_dz=0.35, lookat=None):
        """One RGB frame from a fixed camera aimed above the home palm position, or at `lookat` (needs self.renderer)."""
        cam = mj.MjvCamera()
        cam.lookat[:] = self.p_home + [0, 0, lookat_dz] if lookat is None else lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        self.renderer.update_scene(self.d, cam)
        return self.renderer.render()
