"""Tune the catch with CEM, every candidate scored in the full simulation (a ball dropped from the apex).

Running the throw backwards does not work as a catch: the tuned throw moves the palm forward/backward at
release (to cancel the ball's roll-off), so its reverse slides a vertically falling ball off the palm.
The catch here is its own design:

  approach  hold the stop pose; Tb_c before the predicted arrival, quintic to the catch state:
            palm where the ball will be (IK, palm orientation of the release pose), BALL-POINT velocity
            (0, 0, -k v_ball) (moving straight down with the ball), palm pitch rate w_c, accel qdd_c
  dip       at the arrival time, quintic from the catch state to rest at the cocked pose over T_d;
            the fingers curl to the holding grip at the arrival time

Parameters (8): k, w_c, qdd_c (3 planar joints, / 10), T_d, Tb_c, grip delay (fingers curl this long after
the arrival; the palm alone carries the ball until then).
Cost per drop: 100 if the ball is not held at the end, + 40 * distance between where the ball settles at the
cocked pose and where the throw starts it (the palm centre: a throw from elsewhere goes astray),
+ 50 max(0, joint-speed ratio - 0.9) + 20 torque-saturated fraction; averaged over 3 drops (nominal and
+-2 cm / +-3 cm apex offsets).

    PYTHONPATH=. python3 cem_catch.py --throw runs/throw_v443.json --out runs/catch_v443.json
"""
import argparse
import json
import os

import mujoco as mj
import numpy as np

import cem_throw
import juggle
from juggle import G, Seg
from robolab import evalpool
from planner import BallFit, Planner

P = np.array([1, 2, 3])
DROPS = [(0.0, 0.0, 0.0), (0.02, -0.02, 0.03), (-0.02, 0.02, -0.03)]


def random_drops(n, seed=0, spin=15.0):
    """Drops like the juggling cycle sees them: apex offset +-3 cm across, +-5 cm high, sideways velocity
    N(0, 0.05 m/s) (lands up to ~5 cm off), 2 mm measurement noise (the noise seed per drop), and spin
    N(0, 15 rad/s) per axis: the thrown ball rolls off the turning palm and flies spinning at 20-30 rad/s,
    and friction turns that spin into sideways motion when it lands."""
    rng = np.random.default_rng(seed)
    return [tuple(np.r_[rng.uniform(-0.03, 0.03, 2), rng.uniform(-0.05, 0.05), rng.normal(0, 0.05, 2), 0.002, k,
                        rng.normal(0, spin, 3)])
            for k in range(n)]


class Catcher:
    """Plans the approach to the catch state and the dip, from the measured ball."""

    def __init__(self, J, pl, params, q_rel, q_cock, R_rel, ball_off):
        self.J, self.pl = J, pl
        k, w_c, a1, a2, a3, T_d, Tb_c, grip_delay = params
        self.k, self.w_c, self.T_d, self.Tb_c = k, w_c, max(0.15, T_d), max(0.1, Tb_c)
        self.grip_delay = max(0.0, grip_delay)
        self.qdd_c = np.zeros(6)
        self.qdd_c[P] = np.array([a1, a2, a3]) * 10
        self.q_rel, self.q_cock, self.R_rel, self.ball_off = q_rel, q_cock, R_rel, ball_off
        self.match_horizontal = True

    def catch_state(self, hit, v_hit):
        J, pl = self.J, self.pl
        J.ball_off = self.ball_off
        q_c = pl.ik_ball(hit, self.R_rel, self.q_rel)
        Jm = pl.jac6(q_c)
        qd_c = np.zeros(6)
        if self.match_horizontal:
            # all six joints: ball point moving with the ball sideways (x, y) and down at k v_z, palm pitching at
            # w_c about the planar chain's axis, no other palm rotation; a ball arriving with sideways speed
            # otherwise lands rolling across the open palm and rolls off before the fingers close
            axis = Jm[3:, 1] / np.linalg.norm(Jm[3:, 1])
            qd_c = np.linalg.solve(Jm, np.r_[v_hit[0], v_hit[1], self.k * v_hit[2], self.w_c * axis])
        else:
            A = np.vstack([Jm[0, P], Jm[2, P], np.ones(3)])       # ball-point vx, vz; palm pitch rate
            qd_c[P] = np.linalg.solve(A, [0.0, self.k * v_hit[2], self.w_c])
        return q_c, qd_c, self.qdd_c

    def dip(self, t, end):
        return Seg(t, self.T_d, end, (self.q_cock, np.zeros(6), np.zeros(6)))


def drop_test(params, x, spec_rel, drop=(0, 0, 0), T=0.35, Tb=0.2, frames=None, J=None, settle=0.3, ball_state=None):
    th, br, q_rel = cem_throw.segments(x, T, Tb)
    q_cock, q_stop = th(0)[0], br(T + Tb)[0]
    J = J or juggle.Juggler(1, human=True)
    J.reset(q=q_stop, grip=J.flat)
    pl = Planner(J)
    ball_off = np.array(spec_rel["off"])
    J.ball_off = ball_off
    _, R_rel = pl.fk(q_rel)
    c = Catcher(J, pl, params, q_rel, q_cock, R_rel, ball_off)
    v = spec_rel["v"][2]
    p_rel = np.array(spec_rel["p"])
    z_catch = p_rel[2]
    apex = p_rel + np.array(drop[:3]) + [0, 0, v * v / (2 * G)]
    m, d, ids = J.m, J.d, J.ids
    d.qpos[ids.ball_q[0]:ids.ball_q[0] + 3] = apex
    d.qvel[ids.ball_v[0]:ids.ball_v[0] + 6] = 0
    vx, vy, noise, nseed, wx, wy, wz = (tuple(drop[3:]) + (0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0)[len(drop) - 3:])
    d.qvel[ids.ball_v[0]:ids.ball_v[0] + 2] = vx, vy
    d.qvel[ids.ball_v[0] + 3:ids.ball_v[0] + 6] = wx, wy, wz
    if ball_state is not None:                      # replay: exact ball position + 6-D velocity
        d.qpos[ids.ball_q[0]:ids.ball_q[0] + 3] = ball_state[0]
        d.qvel[ids.ball_v[0]:ids.ball_v[0] + 6] = ball_state[1]
    rng = np.random.default_rng(int(nseed))
    fit = BallFit()
    mj.mj_forward(m, d)
    J.stats = dict(qd_ratio=0.0, tau_sat=0, steps=0)
    seg, t_hit, next_plan, phase, t = None, None, 0.0, "fall", 0.0
    hold = (q_stop, np.zeros(6), np.zeros(6))
    t_end = None
    while True:
        b, bv = J.ball()
        if phase == "fall":
            fit.add(t, b + (rng.normal(0, noise, 3) if noise else 0))
            b, bv = fit.state(t)
            q, qd, qdd = seg(t) if seg is not None else hold
            if seg is not None and t >= t_hit:
                end = seg(seg.t0 + seg.T - 1e-9)
                seg, phase = c.dip(t, end), "dip"
                q, qd, qdd = end
                t_grip = t + c.grip_delay
                t_end = t + c.T_d + settle
            elif t >= next_plan and fit.ready() and (t_hit is None or t_hit - t > 0.03):
                disc = bv[2] ** 2 + 2 * G * (b[2] - z_catch)
                if disc > 0:
                    tau = (bv[2] + np.sqrt(disc)) / G
                    if 0.03 < tau <= c.Tb_c + 0.005:
                        hit = b + bv * tau + [0, 0, -0.5 * G * tau ** 2]
                        seg = Seg(t, tau, (q, qd, qdd), c.catch_state(hit, bv + [0, 0, -G * tau]))
                    t_hit = t + tau
                next_plan = t + 0.02
        else:
            q, qd, qdd = seg(t)
            if t >= t_grip:
                J.hand_target = J.hold
        J.control_joint(q, qd, qdd)
        mj.mj_step(m, d)
        if frames is not None and int(round(t / m.opt.timestep)) % 16 == 0:
            frames.append(J.render())
        t += m.opt.timestep
        if (t_end is not None and t >= t_end) or b[2] < -1.4 or t > 3:
            break
    p, R, _, _ = J.palm()
    b, bv = J.ball()
    held = bool(J.ball_touching_hand() and np.linalg.norm(bv) < 0.5)
    spot = float(np.linalg.norm(R.T @ (b - p) - np.array(spec_rel["start"])))
    return dict(held=held, spot=spot, qd_ratio=J.stats["qd_ratio"], tau_sat=J.stats["tau_sat"] / max(1, J.stats["steps"]))


def cost_of(r):
    return (0 if r["held"] else 100) + 40 * min(r["spot"], 0.2) + 50 * max(0, r["qd_ratio"] - 0.9) + 20 * r["tau_sat"]


def evaluate_one(i, pop, x, rel, drops=DROPS):
    # a drop is an offset tuple, or ("state", ball_state, noise_seed): a replayed apex from collect_apex.py
    rs = [drop_test(pop[i], x, rel, (0, 0, 0, 0, 0, 0.002, dr[2]), ball_state=dr[1]) if dr[0] == "state"
          else drop_test(pop[i], x, rel, dr) for dr in drops]
    return dict(cost=float(np.mean([cost_of(r) for r in rs])), held=sum(r["held"] for r in rs),
                spot=round(float(np.mean([r["spot"] for r in rs])), 3), qd=round(max(r["qd_ratio"] for r in rs), 3),
                sat=round(max(r["tau_sat"] for r in rs), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--throw", default="runs/throw_v443.json")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pop", type=int, default=36)
    ap.add_argument("--out", default="runs/catch_v443.json")
    ap.add_argument("--init", help="start from this catch's parameters")
    ap.add_argument("--drops", type=int, default=0, help="score on this many random drops (0: the 3 fixed ones)")
    ap.add_argument("--bank", help="score on --drops apex states from this bank (collect_apex.py) instead")
    a = ap.parse_args()
    spec = json.load(open(a.throw))
    x = np.array(spec["x"])
    r = cem_throw.simulate(x, spec["v"])
    rel = dict(v=r["release"]["v"].tolist(), p=r["release"]["p"].tolist(), off=r["release"]["off"].tolist())
    th, _, _ = cem_throw.segments(x, 0.35, 0.2)
    J0 = juggle.Juggler(1, human=True)
    J0.reset(q=th(0)[0], grip=J0.hold)                 # where the ball rests at the start of the first throw
    p0, R0, _, _ = J0.palm()
    rel["start"] = (R0.T @ (J0.ball()[0] - p0)).tolist()
    print("throw release:", {k: np.round(v, 3).tolist() for k, v in rel.items()})
    # start: palm moving down at 0.9 of the ball's speed, the throw's pitch rate mirrored, gentle accel, 0.35 s dip, 0.2 s approach
    ap_init = json.load(open(a.init))["params"] if a.init else None
    mu = np.array(ap_init + [0.2] if ap_init is not None and len(ap_init) == 7 else
                  ap_init if ap_init is not None else [0.9, 5.3, 0.0, 0.0, 0.0, 0.35, 0.2, 0.2])
    sig = np.array([0.08, 1.0, 1.0, 1.0, 1.0, 0.08, 0.05, 0.12])
    rng = np.random.default_rng(0)
    if a.bank:
        states = json.load(open(a.bank))["states"]
        pick = np.random.default_rng(1).choice(len(states), a.drops or 16, replace=False)
        bank = [("state", states[j], k) for k, j in enumerate(pick)]
    best = (np.inf, None, None)
    procs = min(12, os.cpu_count())
    for it in range(a.iters):
        X = mu + sig * rng.standard_normal((a.pop, len(mu)))
        X[0] = mu
        drops = bank if a.bank else random_drops(a.drops) if a.drops else DROPS   # the same every generation
        R = evalpool.evaluate(evaluate_one, range(a.pop), procs=procs, args=(X, x, rel, drops))
        costs = np.array([q["cost"] for q in R])
        o = np.argsort(costs)
        if costs[o[0]] < best[0]:
            best = (costs[o[0]], X[o[0]].copy(), R[o[0]])
        E = X[o[:8]]
        mu = E.mean(0)
        sig = 0.7 * sig + 0.3 * E.std(0) + 1e-3
        print(f"iter {it:2d}: best {best[0]:.3f} {best[2]}  params {np.round(best[1], 3).tolist()}  "
              f"gen median {np.median(costs):.2f}", flush=True)
        if best[0] < 0.2 and not a.drops and not a.bank:
            break
    json.dump(dict(throw=a.throw, params=best[1].tolist(), cost=best[0], result=best[2], rel=rel),
              open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
