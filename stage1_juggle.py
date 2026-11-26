"""Stage 1: continuous one-ball juggling with the human-style throw and catch.

One cycle:
  throw   cocked pose (rest) -> release -> brake to rest at the stop pose       (CEM-tuned, cem_throw.py)
  wait    while the ball flies
  catch   Tb_c before the predicted arrival, quintic to the catch state: palm where the ball will be (IK),
          moving straight down with it, re-planned every 20 ms from the measured ball   (CEM-tuned, cem_catch.py)
  dip     from the catch state to rest at the cocked pose over T_d, fingers curled to the holding grip
and the next throw starts. A catch counts when the ball is still in the hand at the end of the dip.

    PYTHONPATH=. python3 stage1_juggle.py --catch runs/catch_v443.json --cycles 30 --gif media_full/stage1.gif
"""
import argparse
import json

import mujoco as mj
import numpy as np

import cem_throw
import juggle
from cem_catch import Catcher
from juggle import G, Seg
from planner import Planner


def juggle_run(J, x, catch_params, ball_off0, cycles=30, T=0.35, Tb=0.2, open_at=0.85, seed=None, noise=0.0,
               ball_offset=(0.0, 0.0, 0.06), settle_max=0.6, frames=None, every=16, log=None):
    rng = np.random.default_rng(seed)
    th, br, q_rel = cem_throw.segments(x, T, Tb)
    q_cock = th(0)[0]
    J.reset(q=q_cock, grip=J.hold, ball_offset=ball_offset)
    p_, R_, _, _ = J.palm()
    start_spot = (R_.T @ (J.ball()[0] - p_)).round(3).tolist()
    pl = Planner(J)
    _, R_rel = pl.fk(q_rel)
    c = Catcher(J, pl, catch_params, q_rel, q_cock, R_rel, np.array(ball_off0))
    m, d = J.m, J.d
    dt = m.opt.timestep
    t, step = 0.0, 0
    phase, t_throw = "throw", 0.0
    seg, t_hit, next_plan, gap, z_catch, t_dip_end = None, None, 0.0, 0, None, None
    off_now = np.array(ball_off0)
    catches, releases, spots = 0, [], []
    while True:
        b, bv = J.ball()
        if phase == "throw":
            tl = t - t_throw
            q, qd, qdd = th(tl) if tl < T else br(tl)
            if tl >= open_at * T:
                J.hand_target = J.flat
            touching = J.ball_touching_hand()
            gap = 0 if touching else gap + 1
            if touching:                                              # where the ball sits in the palm
                p_, R_, _, _ = J.palm()
                off_now = R_.T @ (b - p_)
            if gap >= 3 and tl > 0.5 * T:
                releases.append(bv.round(3).tolist())
                c.ball_off = off_now                                  # catch it back onto that spot
                z_catch = b[2]
                phase, seg, t_hit, next_plan = "flight", None, None, t
        elif phase == "flight":
            q, qd, qdd = seg(t) if seg is not None else br(t - t_throw)
            if seg is not None and t >= t_hit:
                end = seg(seg.t0 + seg.T - 1e-9)                      # the catch state (a Seg holds zero
                seg, phase = c.dip(t, end), "dip"                     # velocity past its end)
                q, qd, qdd = end
                t_grip = t + c.grip_delay
                t_dip_end = t + c.T_d
            elif t >= next_plan and (t_hit is None or t_hit - t > 0.03):
                bm = b + (rng.normal(0, noise, 3) if noise else 0)
                disc = bv[2] ** 2 + 2 * G * (bm[2] - z_catch)
                if disc > 0:
                    tau = (bv[2] + np.sqrt(disc)) / G
                    if 0.03 < tau <= c.Tb_c + 0.005:
                        hit = bm + bv * tau + [0, 0, -0.5 * G * tau ** 2]
                        target = c.catch_state(hit, bv + [0, 0, -G * tau])
                        if np.abs(target[0] - q_rel).max() > 0.6:         # out of reach: a drop, not a flail
                            break
                        seg = Seg(t, tau, (q, qd, qdd), target)
                    t_hit = t + tau
                next_plan = t + 0.02
        else:                                                         # dip, then settle
            q, qd, qdd = seg(t)
            if t >= t_grip:
                J.hand_target = J.hold
            if t >= t_dip_end + 0.05:
                held = J.ball_touching_hand() and np.linalg.norm(bv) < 0.5
                if not held:
                    break
                # wait (up to settle_max) for the ball to come to rest against the curled fingers, where
                # it starts the first throw: the tuned throw's aim depends on where the ball sits
                if np.linalg.norm(bv) > 0.03 and t < t_dip_end + settle_max:
                    pass
                else:
                    catches += 1
                    p_, R_, _, _ = J.palm()
                    spots.append((R_.T @ (b - p_)).round(3).tolist())
                    if log:
                        log(f"  catch {catches}: release {releases[-1]}  ball in palm before next throw {spots[-1]}")
                    if catches >= cycles:
                        break
                    phase, t_throw, gap = "throw", t, 0
        J.control_joint(q, qd, qdd)
        mj.mj_step(m, d)
        if frames is not None and step % every == 0:
            frames.append(J.render())
        t += dt
        step += 1
        if b[2] < -1.4 or t > 120:
            break
    return dict(catches=catches, releases=releases, spots=spots, start_spot=start_spot, qd_ratio=round(J.stats["qd_ratio"], 3),
                tau_sat=round(J.stats["tau_sat"] / max(1, J.stats["steps"]), 4), t=round(t, 2))


def load(catch_path):
    cs = json.load(open(catch_path))
    spec = json.load(open(cs["throw"]))
    return np.array(spec["x"]), np.array(cs["params"]), np.array(cs["rel"]["off"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catch", default="runs/catch_v443.json")
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--gif")
    a = ap.parse_args()
    x, cp, off = load(a.catch)
    J = juggle.Juggler(1, human=True)
    frames = None
    if a.gif:
        J.renderer = mj.Renderer(J.m, 480, 640)
        frames = []
    out = juggle_run(J, x, cp, off, cycles=a.cycles, frames=frames, log=print)
    print(json.dumps({k: v for k, v in out.items() if k not in ("releases", "spots")}))
    if a.gif:
        J.renderer.close()
        from robolab.media import save_gif
        save_gif(frames, a.gif, fps=30)


if __name__ == "__main__":
    main()
