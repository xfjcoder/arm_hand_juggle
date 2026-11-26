"""Tune a human-style throw with CEM, every candidate scored in the full simulation.

Parameters (13): the PlannedThrow vector (release joint rates, release accelerations / 10, cocking offset,
stop offset) plus the palm pitch at release `phi` (added to wrist_1). The SLSQP planner's rigid-ball model
misses the ball rolling on the turning palm (a ~0.8 m/s forward fling), so here the score is what the
simulation measures:

    cost = 25 |release velocity - (0, 0, v)|^2 + 50 max(0, joint-speed ratio - 0.9) + 20 torque-saturated fraction
           + 100 if the ball is not released / falls out before the throw

    PYTHONPATH=. python3 cem_throw.py --v 4.43 --out runs/throw_v443.json
"""
import argparse
import json
import os
from pathlib import Path

import mujoco as mj
import numpy as np

import human
import juggle
from juggle import G, Seg
from robolab import evalpool

P = human.P


def segments(x, T, Tb):
    qd, qdd, dq, dstop = (np.zeros(6) for _ in range(4))
    qd[P], qdd[P], dq[P], dstop[P] = x[0:3], x[3:6] * 10, x[6:9], x[9:12]
    q_rel = human.Q_REL + np.array([0, 0, 0, x[12], 0, 0])
    z = np.zeros(6)
    th = Seg(0, T, (q_rel - dq, z, z), (q_rel, qd, qdd))
    br = Seg(T, Tb, (q_rel, qd, qdd), (q_rel + dstop, z, z))
    return th, br, q_rel


def simulate(x, v, T=0.35, Tb=0.2, open_at=0.85, ball_offset=(0.0, 0.0, 0.06), frames=None, J=None, ball_spot=None):
    """One throw from the cocked pose. Returns dict(release velocity, limits, ...)."""
    th, br, q_rel = segments(x, T, Tb)
    J = J or juggle.Juggler(1, human=True)
    J.reset(q=th(0)[0], grip=J.hold, ball_offset=ball_offset, ball_spot=ball_spot)
    m, d = J.m, J.d
    t, gap, rel, off = 0.0, 0, None, None
    p_start = J.ball()[0]
    while t < T + Tb + 0.05:
        q, qd, qdd = th(t) if t < T else br(t)
        if t >= open_at * T:
            J.hand_target = J.flat
        J.control_joint(q, qd, qdd)
        mj.mj_step(m, d)
        if frames is not None and int(round(t / m.opt.timestep)) % 16 == 0:
            frames.append(J.render())
        t += m.opt.timestep
        b, bv = J.ball()
        if rel is None:
            gap = 0 if J.ball_touching_hand() else gap + 1
            if J.ball_touching_hand():
                p_, R_, _, _ = J.palm()
                off = R_.T @ (b - p_)                      # where the ball sits in the palm while thrown
            if gap >= 3 and t > 0.5 * T:
                rel = dict(v=bv.copy(), p=b.copy(), t=t - T, off=off)
        if b[2] < p_start[2] - 0.3:               # fell out of the hand
            break
    return dict(release=rel, qd_ratio=J.stats["qd_ratio"], tau_sat=J.stats["tau_sat"] / max(1, J.stats["steps"]))


def cost_of(r, v):
    c = 50 * max(0.0, r["qd_ratio"] - 0.9) + 20 * r["tau_sat"]
    if r["release"] is None or r["release"]["t"] < -0.05:
        return c + 100
    e = r["release"]["v"] - np.array([0.0, 0.0, v])
    return c + 25 * float(e @ e)


def evaluate_one(i, pop, v, ball_spot=None):
    """Cost of one candidate; with a list of start spots, the mean over throws from each (worst release shown)."""
    spots = ball_spot if ball_spot is not None and np.ndim(ball_spot) == 2 else [ball_spot]
    rs = [simulate(pop[i], v, ball_spot=s) for s in spots]
    cs = [cost_of(r, v) for r in rs]
    w = rs[int(np.argmax(cs))]
    return dict(cost=float(np.mean(cs)), rel=None if w["release"] is None else w["release"]["v"].round(3).tolist(),
                qd=round(max(r["qd_ratio"] for r in rs), 3), sat=round(max(r["tau_sat"] for r in rs), 4))


def cem(v, x0, sigma0, iters=25, pop=48, elite=8, procs=12, seed=0, log=print, ball_spot=None):
    rng = np.random.default_rng(seed)
    mu, sig = x0.copy(), sigma0.copy()
    best = (np.inf, None, None)
    for it in range(iters):
        X = mu + sig * rng.standard_normal((pop, len(mu)))
        X[0] = mu
        R = evalpool.evaluate(evaluate_one, range(pop), procs=procs, args=(X, v, ball_spot))
        costs = np.array([r["cost"] for r in R])
        order = np.argsort(costs)
        if costs[order[0]] < best[0]:
            best = (costs[order[0]], X[order[0]].copy(), R[order[0]])
        E = X[order[:elite]]
        mu = E.mean(0)
        sig = 0.7 * sig + 0.3 * E.std(0) + 1e-4
        b = best[2]
        log(f"iter {it:2d}: best {best[0]:.4f} (release {b['rel']}, qd {b['qd']}, sat {b['sat']})  "
            f"gen median {np.median(costs):.3f}")
        if best[0] < 0.002:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=4.43)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--out", default="runs/throw.json")
    ap.add_argument("--init", help="start from this throw's parameters instead of the SLSQP plan")
    ap.add_argument("--ball-spot", type=float, nargs=3, help="ball start position in the palm frame (m)")
    ap.add_argument("--ball-spots", type=float, nargs="+",
                    help="several start positions (x y z ...), the cost averaged over throws from each")
    a = ap.parse_args()
    if a.ball_spots:
        a.ball_spot = np.reshape(a.ball_spots, (-1, 3)).tolist()
    J = juggle.Juggler(1, human=True)
    J.reset(q=human.Q_REL)
    if a.init:
        x0 = np.array(json.load(open(a.init))["x"])
        sigma0 = np.r_[np.abs(x0[:3]) * 0.05 + 0.03, np.abs(x0[3:6]) * 0.08 + 0.05, np.full(6, 0.015), 0.05]
    else:
        th = human.plan(J, a.v, 0.0)              # SLSQP start (rigid-ball model), phi = -0.15 as a nudge
        x0 = np.r_[th.x, -0.15]
        sigma0 = np.r_[np.abs(th.x[:3]) * 0.1 + 0.05, np.abs(th.x[3:6]) * 0.15 + 0.1, np.full(6, 0.03), 0.1]
    best = cem(a.v, x0, sigma0, iters=a.iters, pop=a.pop, procs=min(12, os.cpu_count()), ball_spot=a.ball_spot)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(v=a.v, x=best[1].tolist(), cost=best[0], result=best[2], ball_spot=a.ball_spot),
              open(a.out, "w"), indent=1)
    print("saved", a.out)


if __name__ == "__main__":
    main()
