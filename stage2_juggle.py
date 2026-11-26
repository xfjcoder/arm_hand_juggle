"""Stage 2: two balls, one hand.

The hand runs the periodic cycle from the Stage 2 gate (stage2_timing_cem.py):
    dwell   catch state -> release state, D s: absorb the caught ball, swing in, throw (hand opens D_open
            before the release)
    vacant  release -> catch state for the other ball, re-planned every 20 ms from the camera's ballistic
            fit of that ball, so the palm meets it where and when it actually comes down
Each ball flies F = 2 V + D: thrown, the other ball is caught V later and thrown Tc = V + D later, and the
first comes back V after that. Caught 0.20 m to the side of the throw point (world y), so the rising ball
passes the falling one about 9 cm apart (a one-hand fountain; 0.10 m left 1.6 cm and the balls collided).

Start: the first ball is thrown from rest (a start-up stroke into the release state); at that release a
feeder puts the second ball on the matching path, as if it had been thrown one cycle earlier. A catch counts
when the caught ball is thrown again.

    PYTHONPATH=.:../robolab python3 stage2_juggle.py --cycles 10 --gif media_full/stage2.gif
"""
import argparse
import json

import mujoco as mj
import numpy as np

import cem_throw
import juggle
import scene
import stage1_juggle
import stage2_timing
from cem_catch import Catcher
from juggle import G, Seg
from planner import BallFit, Planner
from sensing import Camera, Tactile, PRESETS
from stage1_juggle import peak_ratio

CATCH = "runs/catch5_v443.json"


def load_cycle(J, pl, gate="logs/stage2_timing_cem_d[0.0,0.2].json", catch=CATCH, d_out=(0.0, 0.2)):
    """The nominal cycle: the best gate solution, plus the Catcher (for catch states) it was built with."""
    x, cp, off = stage1_juggle.load(catch)
    R = json.load(open(gate))
    b = min(R, key=lambda r: max(r["speed"], r["torque"]))
    th, br, q_rel = cem_throw.segments(x, 0.35, 0.2)
    _, R_rel = pl.fk(q_rel)
    c = Catcher(J, pl, cp, q_rel, th(0)[0], R_rel, np.array(off))
    J.ball_off = c.ball_off
    cy = stage2_timing.cycle_states(J, pl, c, q_rel, np.array(b["z"]), b["vz_set"], b["V"], d_out)
    c.k, c.qdd_c, c.R_rel = cy["k"], cy["qdd_c"], pl.fk(cy["catch"][0])[1]
    # start-up stroke: from rest at the Stage 1 cocking offset below the release pose (0.94 of the speed
    # limit, 0.78 of the torque limit over 0.4 s)
    cy["q_start"], cy["c"] = cy["rel"][0] - (q_rel - th(0)[0]), c
    return cy


def juggle2(J, cy, cycles=10, seed=None, kick=0.0, preset="real", noise=0.002, T_start=0.4, D_open=0.05,
            grip_delay=0.03, vmax_ratio=0.97, frames=None, every=16, log=None, trace=None):
    rng = np.random.default_rng(seed)
    pl = Planner(J)
    c = cy["c"]
    J.ball_off = c.ball_off
    V, D = cy["V"], cy["D"]
    z_catch = cy["hit"][2]
    cfg = PRESETS[preset]
    srng = np.random.default_rng(None if seed is None else seed + 104729)
    cams = [Camera(J, srng, noise=noise, ball=i, **cfg["camera"]) for i in (0, 1)]
    tacs = [Tactile(J, srng, ball=i, **cfg["tactile"]) for i in (0, 1)]
    J.reset(q=cy["q_start"], grip=J.hold)
    m, d, ids = J.m, J.d, J.ids
    dt = m.opt.timestep
    seg = Seg(0.0, T_start, (cy["q_start"], np.zeros(6), np.zeros(6)), cy["rel"])
    if log and peak_ratio(seg) > vmax_ratio:
        log(f"start-up stroke peak {peak_ratio(seg):.2f}")
    phase, held, t, step = "dwell", 0, 0.0, 0                    # held: the ball in (or leaving) the hand
    t_open, t_grip, t_seg_end = T_start - D_open, None, T_start
    fits, gap, gap_h, leaving = [None, None], 0, 0, None
    t_hit, next_plan, fed = None, 0.0, False
    catches, releases, why = 0, [], None
    Tc = V + D
    # the feeder's ball, one cycle behind the first: as if thrown Tc before the start-up release, so it is
    # tracked for most of its flight like every later ball (fed late, its short track missed by 3 cm)
    tau0 = Tc - T_start
    d.qpos[ids.ball_q[1]:ids.ball_q[1] + 3] = cy["p_rel"] + cy["v_rel"] * tau0 + np.r_[0, 0, -0.5 * G * tau0 ** 2]
    d.qvel[ids.ball_v[1]:ids.ball_v[1] + 6] = np.r_[cy["v_rel"] + np.r_[0, 0, -G * tau0], 0, 0, 0]
    mj.mj_forward(m, d)
    fits[1], fed = BallFit(), True
    fits[1].t0 = -1.0
    while True:
        readings = [tc.step(t) for tc in tacs]
        frames_in = [cam.step(t) for cam in cams]
        for i in (0, 1):
            if fits[i] is not None:
                for tcap, p in frames_in[i]:
                    if tcap > fits[i].t0:
                        fits[i].add(tcap, p)
        if leaving is not None:                                        # the thrown ball: watch for its release
            r = readings[leaving]
            gap = 0 if (r is not None and r[1]) else gap + 1
            if gap >= 3:
                if kick:
                    d.qvel[ids.ball_v[leaving]:ids.ball_v[leaving] + 3] += rng.normal(0, kick, 3)
                    mj.mj_forward(m, d)
                releases.append(J.ball(leaving)[1].round(3).tolist())
                if log:
                    log(f"  release ball {leaving} at {t:.3f}: v {releases[-1]}")
                fits[leaving] = BallFit()
                fits[leaving].t0 = t
                leaving = None
        if phase == "dwell":
            q, qd, qdd = seg(t)
            if t_grip is not None and t >= t_grip:
                J.hand_target = J.hold
            if t >= t_open:
                J.hand_target = J.flat
                if leaving is None and (not releases or t_grip is not None):
                    leaving, gap = held, 0
            elif t_grip is not None and t > t_grip + 0.1:              # the caught ball must stay in the hand
                r = readings[held]
                gap_h = gap_h + 1 if (r is not None and not r[1]) else 0
                if gap_h > 20:
                    why = "not held"
                    break
            if t >= t_seg_end:                                         # the release state: start the vacant
                if t_grip is not None:
                    catches += 1
                    if catches >= cycles:
                        break
                held = 1 - held
                # until the incoming ball's fit is ready: the nominal catch, V from now
                vac = Seg(t, V, cy["rel"], cy["catch"])
                phase, t_hit, next_plan = "vacant", t + V, t
                q, qd, qdd = vac(t)
        elif phase == "vacant":
            q, qd, qdd = vac(t)
            f = fits[held]
            if t >= next_plan and f is not None and f.ready() and t_hit - t > 0.03:
                b, bv = f.state(t)
                disc = bv[2] ** 2 + 2 * G * (b[2] - z_catch)
                if disc > 0:
                    tau = (bv[2] + np.sqrt(disc)) / G
                    if tau > 0.03:
                        hit = b + bv * tau + np.r_[0, 0, -0.5 * G * tau ** 2]
                        target = c.catch_state(hit, bv + np.r_[0, 0, -G * tau])
                        new = Seg(t, tau, (q, qd, qdd), target)
                        if peak_ratio(new) > vmax_ratio:
                            why = "approach too fast"
                            if log:
                                log(f"  approach: tau {tau:.3f} hit {hit.round(3)} peak {peak_ratio(new):.2f}")
                            break
                        vac, t_hit = new, t + tau
                        q, qd, qdd = vac(t)
                next_plan = t + 0.02
            if t >= t_hit:
                end = vac(vac.t0 + vac.T - 1e-9)                       # a Seg holds zero velocity past its end
                seg = Seg(t, D, end, cy["rel"])
                t_open, t_grip, t_seg_end = t + D - D_open, t + grip_delay, t + D
                phase, gap_h = "dwell", 0
                q, qd, qdd = end
        J.control_joint(q, qd, qdd)
        mj.mj_step(m, d)
        if trace is not None:
            trace.append(dict(t=t, phase=phase, held=held, b0=J.ball(0)[0].copy(), b1=J.ball(1)[0].copy(),
                              q=d.qpos[ids.arm_q].copy(), qd_d=np.asarray(qd).copy()))
        if frames is not None and step % every == 0:
            frames.append(J.render())
        t += dt
        step += 1
        zs = [J.ball(i)[0][2] for i in ((0, 1) if fed else (0,))]
        if min(zs) < -1.4 or t > 60:
            why = why or ("ball dropped" if min(zs) < -1.4 else "time")
            break
    return dict(catches=catches, why=why if catches < cycles else None, releases=releases,
                qd_ratio=round(J.stats["qd_ratio"], 3), tau_sat=round(J.stats["tau_sat"] / max(1, J.stats["steps"]), 4),
                t=round(t, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=10)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--kick", type=float, default=0.0)
    ap.add_argument("--sensors", default="real", choices=["ideal", "real"])
    ap.add_argument("--gif")
    a = ap.parse_args()
    J = juggle.Juggler(2, human=True)
    pl = Planner(J)
    cy = load_cycle(J, pl)
    frames = None
    if a.gif:
        J.renderer = mj.Renderer(J.m, 720, 540)
        render = J.render
        J.render = lambda: render(distance=3.1, azimuth=160, elevation=-6, lookat=[1.0, 0.2, 0.2])
        frames = []
    out = juggle2(J, cy, cycles=a.cycles, seed=a.seed, kick=a.kick, preset=a.sensors, frames=frames, log=print)
    print(json.dumps({k: v for k, v in out.items() if k != "releases"}))
    if a.gif:
        J.renderer.close()
        from robolab.media import save_gif
        save_gif(frames, a.gif, fps=30)


if __name__ == "__main__":
    main()
