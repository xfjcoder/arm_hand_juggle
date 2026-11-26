"""Human-style throw: upper arm hanging, elbow bent, the throw made by shoulder + elbow + wrist flexion.

UR10e shoulder_lift, elbow and wrist_1 rotate about parallel axes, so together they form a planar
3-joint chain in the vertical x-z plane, like a human shoulder / elbow / wrist flexing. The throw is
planned in joint space on those three joints; the others are held:

  release state  at the release pose q_rel (palm level), joint rates that give the BALL CENTRE the
                 velocity (vx, 0, vz) with the smallest palm pitch rate whose joint use is <= u_max of
                 the real UR10e speed limits (an LP); joint accelerations that give the ball centre
                 (0, 0, -g a_rel) so it separates from the palm right at release
  throw          quintic in joint space from rest at q_cock = q_rel - qd_rel T / 2 to the release state
  brake          quintic from the release state to rest

Arm control is joint-space computed torque (Juggler.control_joint).
"""
import mujoco as mj
import numpy as np
from scipy.optimize import linprog

import juggle
import scene
from juggle import G

P = np.array([1, 2, 3])                      # shoulder_lift, elbow, wrist_1 (indices into the 6 arm joints)
# release pose: upper arm hanging 21 deg forward of vertical, elbow bent, forearm up 34 deg, palm level
Q_REL = np.array([0.0, 1.2, -1.8, 0.6, np.pi / 2, np.pi / 2])


def ball_jac(J, q, with_dot=False, qd=None):
    """3 x 6 Jacobian (and optionally Jdot qd) of the resting ball centre, at arm joints q."""
    m, d, ids = J.m, J.d, J.ids
    d.qpos[ids.arm_q] = q
    if qd is not None:
        d.qvel[ids.arm_v] = qd
    mj.mj_kinematics(m, d)
    mj.mj_comPos(m, d)
    s = J.site
    R = d.site_xmat[s].reshape(3, 3)
    pt = d.site_xpos[s] + R @ J.ball_off
    body = m.site_bodyid[s]
    jp = np.zeros((3, m.nv))
    mj.mj_jac(m, d, jp, None, pt, body)
    Jb = jp[:, ids.arm_v]
    if not with_dot:
        return Jb, pt
    mj.mj_comVel(m, d)
    jdp = np.zeros((3, m.nv))
    mj.mj_jacDot(m, d, jdp, None, pt, body)
    return Jb, pt, jdp[:, ids.arm_v] @ d.qvel[ids.arm_v]


def release_state(J, q_rel, v, vx=0.0, a_rel=1.0, u_max=0.85):
    """(qd_rel, qdd_rel, use) for a ball-centre release velocity (vx, 0, v) at q_rel (planar joints only)."""
    Jb, _ = ball_jac(J, q_rel)
    A = Jb[[0, 2]][:, P]
    vm = scene.JOINT_VMAX[P]
    # variables: qd (3), w = |pitch rate|; minimise w
    c = np.array([0, 0, 0, 1.0])
    A_eq = np.hstack([A, np.zeros((2, 1))])
    A_ub = np.array([[1, 1, 1, -1], [-1, -1, -1, -1]], float)
    for u in (u_max, 0.9, 0.95, 1.0):
        r = linprog(c, A_ub=A_ub, b_ub=[0, 0], A_eq=A_eq, b_eq=[vx, v],
                    bounds=[(-u * x, u * x) for x in vm] + [(0, None)], method="highs")
        if r.status == 0:
            break
    else:
        raise ValueError(f"release speed {v} m/s not reachable at this pose")
    qd = np.zeros(6)
    qd[P] = r.x[:3]
    Jb, _, jdqd = ball_jac(J, q_rel, with_dot=True, qd=qd)
    A = Jb[[0, 2]][:, P]
    qdd = np.zeros(6)
    qdd[P] = np.linalg.pinv(A) @ (np.array([0.0, -G * a_rel]) - jdqd[[0, 2]])
    return qd, qdd, float(np.max(np.abs(qd[P]) / vm))


class Throw:
    """Joint-space throw + brake. __call__(t) -> (q, qd, qdd); t = 0 is the start of the throw."""

    def __init__(self, J, v, q_rel=Q_REL, T=0.3, vx=0.0, a_rel=1.0, T_brake=None, u_max=0.85):
        self.q_rel = q_rel.copy()
        self.qd_rel, self.qdd_rel, self.use = release_state(J, q_rel, v, vx, a_rel, u_max)
        self.q_cock = q_rel - self.qd_rel * T / 2
        self.T = T
        z = np.zeros(6)
        self.throw = juggle.Seg(0.0, T, (self.q_cock, z, z), (q_rel, self.qd_rel, self.qdd_rel))
        # brake to rest: as fast as ~2x the throw's average deceleration allows, at least 0.12 s
        self.Tb = T_brake or max(0.12, T / 2)
        self.q_stop = q_rel + self.qd_rel * self.Tb / 2
        self.brake = juggle.Seg(T, self.Tb, (q_rel, self.qd_rel, self.qdd_rel), (self.q_stop, z, z))

    def __call__(self, t):
        return self.throw(t) if t < self.T else self.brake(t)


# ---------------------------------------------------------------------------------------------------------
# Whole-throw planning as a small constrained optimisation (SLSQP over 12 numbers, kinematics + M(q) only)
# ---------------------------------------------------------------------------------------------------------
def ball_terms(J, q, qd, qdd):
    """Ball-centre velocity, and its normal 'grip' N/m = a.n + g n_z (> 0: pressed into the palm)."""
    m, d, ids = J.m, J.d, J.ids
    d.qpos[ids.arm_q] = q
    d.qvel[:] = 0
    d.qvel[ids.arm_v] = qd
    mj.mj_kinematics(m, d)
    mj.mj_comPos(m, d)
    mj.mj_comVel(m, d)
    s = J.site
    R = d.site_xmat[s].reshape(3, 3)
    pt = d.site_xpos[s] + R @ J.ball_off
    body = m.site_bodyid[s]
    jp = np.zeros((3, m.nv))
    jdp = np.zeros((3, m.nv))
    mj.mj_jac(m, d, jp, None, pt, body)
    mj.mj_jacDot(m, d, jdp, None, pt, body)
    vel = jp[:, ids.arm_v] @ qd
    acc = jp[:, ids.arm_v] @ qdd + jdp[:, ids.arm_v] @ qd
    n = -R[:, 1]
    return vel, acc @ n + G * n[2], n


def torque(J, q, qd, qdd):
    m, d, ids = J.m, J.d, J.ids
    d.qpos[ids.arm_q] = q
    d.qvel[:] = 0
    d.qvel[ids.arm_v] = qd
    d.qacc[:] = 0
    d.qacc[ids.arm_v] = qdd
    mj.mj_inverse(m, d)
    return d.qfrc_inverse[ids.arm_v] - d.qfrc_passive[ids.arm_v]


# PlannedThrow solution for v = 4.0 m/s, vx = 0, T = 0.35 (qd_rel, qdd_rel / 10, cocking offset, stop offset)
X_NOMINAL = np.array([-1.2666, -2.1128, -1.345, 0.3014, -0.2201, 1.3231, -0.4069, -0.2161, 0.1229,
                      -0.0497, -0.0682, -0.0089])


class PlannedThrow:
    """Throw + brake from an optimisation. __call__(t) -> (q, qd, qdd); release at t = T."""

    def __init__(self, J, v, vx=0.0, q_rel=Q_REL, T=0.35, Tb=0.2, use=0.85, grip_margin=1.5, max_tilt=0.5,
                 n_samp=24, verbose=False, x0=None):
        from scipy.optimize import minimize
        self.q_rel, self.T, self.Tb = q_rel.copy(), T, Tb
        vm = scene.JOINT_VMAX[P]
        tm = scene.TORQUE_MAX
        z6 = np.zeros(6)
        ts = np.linspace(0, T, n_samp)[1:-1]
        tb = np.linspace(T, T + Tb, n_samp // 2)[1:]

        def unpack(x):
            qd, qdd, dq, dstop = (np.zeros(6) for _ in range(4))
            qd[P], qdd[P], dq[P], dstop[P] = x[0:3], x[3:6] * 10, x[6:9], x[9:12]
            return qd, qdd, dq, dstop

        def segs(x):
            qd, qdd, dq, dstop = unpack(x)
            th = juggle.Seg(0, T, (q_rel - dq, z6, z6), (q_rel, qd, qdd))
            br = juggle.Seg(T, Tb, (q_rel, qd, qdd), (q_rel + dstop, z6, z6))
            return th, br

        def eq(x):
            qd, qdd, _, _ = unpack(x)
            vel, grip, _ = ball_terms(J, q_rel, qd, qdd)
            return np.array([vel[0] - vx, vel[2] - v, grip])          # on target, grip exactly 0 at release

        def ineq(x, labels=None):
            th, br = segs(x)
            out = []

            def add(name, vals):
                vals = np.atleast_1d(vals)
                out.extend(vals)
                if labels is not None:
                    labels.extend([name] * len(vals))
            for t in ts:                                               # pressed in until release; speed limits
                q, qd, qdd = th(t)
                add("grip before release", ball_terms(J, q, qd, qdd)[1] - grip_margin)
                add("joint speed (throw)", use - qd[P] / vm)
                add("joint speed (throw)", use + qd[P] / vm)
                if t > 0.5 * T:
                    tq = torque(J, q, qd, qdd)[P] / tm[P]
                    add("torque (throw)", 0.9 - tq)
                    add("torque (throw)", 0.9 + tq)
            for t in tb:                                               # palm falls away from the ball; limits
                q, qd, qdd = br(t)
                if t < T + 0.06:
                    add("palm falls away", -ball_terms(J, q, qd, qdd)[1] - grip_margin)
                add("joint speed (brake)", use - qd[P] / vm)
                add("joint speed (brake)", use + qd[P] / vm)
                tq = torque(J, q, qd, qdd)[P] / tm[P]
                add("torque (brake)", 0.9 - tq)
                add("torque (brake)", 0.9 + tq)
            q0 = th(0)[0]
            tilt = q0[P].sum() - q_rel[P].sum()                        # palm tilt in the cocked pose
            add("cocked tilt", [max_tilt - tilt, max_tilt + tilt])
            return np.array(out)

        def cost(x):
            qd, qdd, dq, dstop = unpack(x)
            return np.sum((qd[P] / vm) ** 2) + 0.01 * np.sum(x[3:6] ** 2) + 0.1 * np.sum(dstop[P] ** 2)

        # start from the 4 m/s solution (T 0.35) scaled to v: SLSQP from the LP guess gets stuck at lower speeds
        s = v / 4.0
        base = X_NOMINAL * np.r_[[s] * 3, [s * s] * 3, [s] * 6]
        rng = np.random.default_rng(0)
        for attempt in range(1 if x0 is not None else 6):
            if x0 is None or attempt > 0:
                x0 = base if attempt == 0 else base * (1 + 0.15 * rng.standard_normal(12))
            r = minimize(cost, x0, method="SLSQP", constraints=[dict(type="eq", fun=eq), dict(type="ineq", fun=ineq)],
                         options=dict(maxiter=300, ftol=1e-6))
            self.ok = bool(r.success and np.all(ineq(r.x) > -1e-3) and np.all(np.abs(eq(r.x)) < 1e-3))
            if self.ok:
                break
        self.violation = float(max(0, -ineq(r.x).min(), np.abs(eq(r.x)).max()))
        labels = []
        vals = ineq(r.x, labels)
        self.worst = {}
        for name, val in zip(labels, vals):
            self.worst[name] = min(self.worst.get(name, np.inf), float(val))
        self.worst["equalities"] = -float(np.abs(eq(r.x)).max())
        self.result = r
        self.x = r.x
        self.throw, self.brake = segs(r.x)
        self.qd_rel, self.qdd_rel, dq, dstop = unpack(r.x)
        self.q_cock = q_rel - dq
        self.use = max(float(np.max(np.abs(self(t)[1][P]) / vm)) for t in np.linspace(0, T + Tb, 120))
        if verbose:
            print(f"  plan: ok {self.ok} ({r.message}) cost {r.fun:.3f} peak use {self.use:.2f}  qd_rel {self.qd_rel[P].round(2)}"
                  f" cock tilt {self.q_cock[P].sum() - q_rel[P].sum():+.2f} rad")

    def __call__(self, t):
        return self.throw(t) if t < self.T else self.brake(t)


def plan(J, v, vx=0.0, step=0.1, verbose=False, **kw):
    """PlannedThrow by continuation from the known 4 m/s solution: SLSQP stalls when started far away,
    so walk (v, vx) from (4.0, 0) to the target in small steps, each warm-started from the last."""
    n = int(np.ceil(max(abs(v - 4.0), abs(vx)) / step))
    th = PlannedThrow(J, 4.0, 0.0, **kw)
    for i in range(1, n + 1):
        f = i / n
        th = PlannedThrow(J, 4.0 + f * (v - 4.0), f * vx, x0=th.x, **kw)
        if verbose:
            print(f"   continuation {i}/{n}: v {4.0 + f * (v - 4.0):.2f} vx {f * vx:+.2f} ok {th.ok} viol {th.violation:.3f}")
        if not th.ok:
            break
    return th


def sim_throw(v_cmd, vx_cmd, T=0.35, Tb=0.2, open_at=0.85, plan=None):
    """Simulate one planned throw from the cocked pose; returns (plan, measured release velocity or None)."""
    J = juggle.Juggler(1, human=True)
    J.reset(q=Q_REL)
    off = J.ball_off.copy()
    th = plan or PlannedThrow(J, v_cmd, vx=vx_cmd, T=T, Tb=Tb)
    J.reset(q=th.q_cock, grip=J.hold)
    J.ball_off = off
    m, d = J.m, J.d
    t, gap = 0.0, 0
    while t < T + Tb + 0.1:
        q, qd, qdd = th(t)
        if t >= open_at * T:
            J.hand_target = J.flat
        J.control_joint(q, qd, qdd)
        mj.mj_step(m, d)
        t += m.opt.timestep
        gap = 0 if J.ball_touching_hand() else gap + 1
        if gap >= 3 and t > 0.5 * T:
            return th, J.ball()[1]
    return th, None


def calibrate(v, T=0.35, Tb=0.2, iters=6, tol=0.01, verbose=False):
    """Commanded (vx, vz) such that the MEASURED release is (0, v): the ball rolls forward on the turning palm,
    a systematic offset the planner's rigid-ball model does not see. Gain-1 fixed-point iteration."""
    cmd = np.array([0.0, v])
    for i in range(iters):
        th, bv = sim_throw(cmd[1], cmd[0], T, Tb)
        if bv is None:
            raise RuntimeError("ball not released")
        err = np.array([0.0, v]) - bv[[0, 2]]
        if verbose:
            print(f"  calib {i}: cmd ({cmd[0]:+.3f}, {cmd[1]:.3f}) -> release ({bv[0]:+.3f}, {bv[2]:.3f})  plan ok {th.ok}")
        if np.abs(err).max() < tol:
            break
        cmd = cmd + err
    return cmd, th
