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
from inhand import InHand
from planner import BallFit, Planner
from sensing import Sensors, render_views


def juggle_run(J, x, catch_params, ball_off0, cycles=30, T=0.35, Tb=0.2, open_at=0.85, seed=None, noise=0.0,
               ball_offset=(0.0, 0.0, 0.06), settle_max=0.6, frames=None, every=16, log=None, trace=None, preposition=True,
               vmax_ratio=0.97, kick=0.0, inhand=True, ball_spot=None, inhand_kw=None, sensors=None, z_release=None):
    # vmax_ratio: planned motions above this fraction of the joint speed limits end the run (a drop); below 1.0
    # because tracking adds a little: a plan at 0.995 of the limit measured 1.001
    # sensors: None reads the ball as Stage 1 was tuned (tracker every step with `noise`, ideal in-hand
    # sensor); "ideal"/"real" or a sensing.Sensors: a camera in flight and a tactile skin in the hand.
    # With sensors the catch aims for the release spot and height the catch was tuned with (ball_off0,
    # z_release): measured ones would come from a ball rolling fast across the palm, seen by a 100 Hz skin
    rng = np.random.default_rng(seed)
    if isinstance(sensors, str):
        sensors = Sensors(J, sensors, seed=seed, noise=noise)
    cam = sensors.camera if sensors else None
    tac = sensors.tactile if sensors else None
    th, br, q_rel = cem_throw.segments(x, T, Tb)
    q_cock = th(0)[0]
    J.reset(q=q_cock, grip=J.hold, ball_offset=ball_offset, ball_spot=ball_spot)
    p_, R_, _, _ = J.palm()
    start_spot = (R_.T @ (J.ball()[0] - p_)).round(3).tolist()
    pl = Planner(J)
    _, R_rel = pl.fk(q_rel)
    q_stop = br(T + Tb)[0]
    _, R_stop = pl.fk(q_stop)
    c = Catcher(J, pl, catch_params, q_rel, q_cock, R_rel, np.array(ball_off0))
    # the target is the pocket the throw was tuned from (x 0.015-0.018), not wherever this start settled
    ih = InHand(J, pl, 0.016, rng=np.random.default_rng(None if seed is None else seed + 7919),
                **(inhand_kw or {})) if inhand else None
    m, d = J.m, J.d
    dt = m.opt.timestep
    t, step = 0.0, 0
    phase, t_throw, t_rel = "throw", 0.0, 0.0
    for sn in (cam, tac):
        if sn:
            sn.reset()
    seg, t_hit, next_plan, gap, z_catch, t_dip_end = None, None, 0.0, 0, None, None
    off_now = np.array(ball_off0)
    catches, releases, spots = 0, [], []
    pre, why = None, None
    while True:
        b, bv = J.ball()
        reading = tac.step(t) if tac else None                       # newest tactile reading
        frames_in = cam.step(t) if cam else None                     # camera frames delivered now
        if phase == "throw":
            tl = t - t_throw
            q, qd, qdd = th(tl) if tl < T else br(tl)
            if tl >= open_at * T:
                J.hand_target = J.flat
            if tac:                                                   # release seen by the skin
                touching = reading is not None and reading[1]
            else:
                touching = J.ball_touching_hand()
                if touching:                                          # where the ball sits in the palm
                    p_, R_, _, _ = J.palm()
                    off_now = R_.T @ (b - p_)
            gap = 0 if touching else gap + 1
            if gap >= 3 and tl > 0.5 * T:
                if kick:                                              # throw-to-throw variation (robustness test)
                    J.d.qvel[J.ids.ball_v[0]:J.ids.ball_v[0] + 3] += rng.normal(0, kick, 3)
                    mj.mj_forward(m, J.d)
                    b, bv = J.ball()
                releases.append(bv.round(3).tolist())
                c.ball_off = off_now                                  # catch it back onto that spot
                if tac:
                    z_catch = z_release
                else:
                    z_catch = b[2]
                phase, seg, t_hit, next_plan, pre, t_rel = "flight", None, None, t, None, t
                fit = BallFit()
        elif phase == "flight":
            if cam:
                for tc, pc in frames_in:
                    if tc >= t_rel:                                # in free flight only
                        fit.add(tc, pc)
            else:
                fit.add(t, b + (rng.normal(0, noise, 3) if noise else 0))  # what a ball tracker would measure
            if fit.t:
                b, bv = fit.state(t)
            tl = t - t_throw                                          # the ball can leave early (a bad throw):
            hold = pre(t) if pre is not None else th(tl) if tl < T else br(tl)   # finish the throw, then brake
            q, qd, qdd = seg(t) if seg is not None else hold
            if preposition and pre is None and seg is None and t - t_throw >= T + Tb and fit.ready():
                # the brake is done and the ball's flight is known: move (slowly) to under where it will land,
                # so the tuned approach only has to correct the last few cm
                disc = bv[2] ** 2 + 2 * G * (b[2] - z_catch)
                tau = (bv[2] + np.sqrt(max(disc, 0.0))) / G
                hit = b + bv * tau + [0, 0, -0.5 * G * tau ** 2]
                J.ball_off = c.ball_off
                shift = np.r_[hit[:2] - pl.ball_point(q_rel)[:2], 0.0]
                q_pre = pl.ik_ball(pl.ball_point(q_stop) + shift, R_stop, q_stop)
                T_pre = max(0.15, tau - c.Tb_c - 0.04)
                pre = Seg(t, T_pre, (q, qd, qdd), (q_pre, np.zeros(6), np.zeros(6)))
                if log and peak_ratio(pre) > vmax_ratio:
                    log(f"  pre: shift {shift.round(3)} T {T_pre:.2f} peak {peak_ratio(pre):.2f} dq {(q_pre - q).round(2)}")
                if peak_ratio(pre) > vmax_ratio:
                    why = "pre-position too fast"
                    break
                q, qd, qdd = pre(t)
            if seg is not None and t >= t_hit:
                end = seg(seg.t0 + seg.T - 1e-9)                      # the catch state (a Seg holds zero
                seg, phase = c.dip(t, end), "dip"                     # velocity past its end)
                q, qd, qdd = end
                t_grip = t + c.grip_delay
                t_dip_end = t + c.T_d
                if ih:
                    ih.reset()
            elif t >= next_plan and fit.ready() and (t_hit is None or t_hit - t > 0.03):
                bm = b
                disc = bv[2] ** 2 + 2 * G * (bm[2] - z_catch)
                if disc > 0:
                    tau = (bv[2] + np.sqrt(disc)) / G
                    if 0.03 < tau <= c.Tb_c + 0.005:
                        hit = bm + bv * tau + [0, 0, -0.5 * G * tau ** 2]
                        target = c.catch_state(hit, bv + [0, 0, -G * tau])
                        if np.abs(target[0] - q_rel).max() > 0.9:         # out of reach: a drop, not a flail
                            why = "out of reach"
                            break
                        seg = Seg(t, tau, (q, qd, qdd), target)
                        if peak_ratio(seg) > vmax_ratio:                 # would break the joint-speed limits
                            if log:
                                log(f"  approach: tau {tau:.3f} hit {hit.round(3)} fit v {bv.round(2)} true v "
                                    f"{J.ball()[1].round(2)} peak {peak_ratio(seg):.2f} dq {(target[0] - q).round(2)}")
                            why = "approach too fast"
                            break
                    t_hit = t + tau
                next_plan = t + 0.02
        else:                                                         # dip, then settle
            q, qd, qdd = seg(t)
            if ih:                                                    # in-hand feedback: stop the roll
                ih.step(t, dt, q, qd, reading=reading)
                q, qd, qdd = ih.apply(q, qd, qdd)
            if t >= t_grip:
                J.hand_target = J.hold
            if t >= t_dip_end + 0.05:
                held = J.ball_touching_hand() and np.linalg.norm(bv) < 0.5
                if not held:
                    why = "not held"
                    break
                # wait (up to settle_max) for the ball to come to rest against the curled fingers, where
                # it starts the first throw: the tuned throw's aim depends on where the ball sits
                still = ih.settled() if ih else np.linalg.norm(bv) <= 0.03
                if not still and t < t_dip_end + settle_max:
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
        if trace is not None:
            p_, R_, _, _ = J.palm()
            trace.append(dict(t=t, phase=phase, cycle=catches, qd=(d.qvel[J.ids.arm_v] / juggle.scene.JOINT_VMAX).copy(),
                              qd_d=np.asarray(qd) / juggle.scene.JOINT_VMAX, off=R_.T @ (b - p_), b=b.copy(), bv=bv.copy()))
        if frames is not None and step % every == 0:
            frames.append(J.render())
        t += dt
        step += 1
        z_true = J.ball()[0][2]
        if z_true < -1.4 or t > 120:
            why = why or ("ball dropped" if z_true < -1.4 else "time")
            break
    return dict(catches=catches, why=why if catches < cycles else None, releases=releases, spots=spots, start_spot=start_spot, qd_ratio=round(J.stats["qd_ratio"], 3),
                tau_sat=round(J.stats["tau_sat"] / max(1, J.stats["steps"]), 4), t=round(t, 2),
                cam_lost=round((cam.hidden + cam.dropped) / max(1, cam.frames), 3) if cam else None)


def peak_ratio(seg, n=40):
    """Largest planned joint speed over a segment, as a fraction of the UR10e limits."""
    return max(float(np.max(np.abs(seg(seg.t0 + seg.T * i / n)[1]) / juggle.scene.JOINT_VMAX)) for i in range(n + 1))


def load(catch_path):
    cs = json.load(open(catch_path))
    spec = json.load(open(cs["throw"]))
    return np.array(spec["x"]), np.array(cs["params"]), np.array(cs["rel"]["off"])


def release_height(catch_path):
    """The release height the catch was tuned with (cem_catch.py)."""
    return json.load(open(catch_path))["rel"]["p"][2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catch", default="runs/catch_v443.json")
    ap.add_argument("--cycles", type=int, default=30)
    ap.add_argument("--gif")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--kick", type=float, default=0.0, help="random ball velocity kick per release (m/s, per axis)")
    ap.add_argument("--noise", type=float, default=0.0, help="ball tracker position noise (m)")
    ap.add_argument("--no-inhand", action="store_true")
    ap.add_argument("--sensors", choices=["ideal", "real"], help="sensor model (default: as Stage 1 was tuned)")
    a = ap.parse_args()
    x, cp, off = load(a.catch)
    J = juggle.Juggler(1, human=True)
    sensors = Sensors(J, a.sensors, seed=a.seed, noise=a.noise) if a.sensors else None
    frames = None
    if a.gif:
        J.renderer = mj.Renderer(J.m, 720, 540)              # portrait: the cocked palm (z -0.7) to the apex (1.1)
        render = J.render
        outside = dict(distance=3.1, azimuth=160, elevation=-6, lookat=[1.0, 0.2, 0.2])
        if sensors:                                          # plus what the tracking camera sees
            J.render = lambda: render_views(J, J.renderer, sensors, render(**outside))
        else:
            J.render = lambda: render(**outside)
        frames = []
    out = juggle_run(J, x, cp, off, cycles=a.cycles, frames=frames, log=print, seed=a.seed, kick=a.kick, noise=a.noise,
                     inhand=not a.no_inhand, settle_max=1.0, sensors=sensors,
                     z_release=release_height(a.catch))
    print(json.dumps({k: v for k, v in out.items() if k not in ("releases", "spots")}))
    if a.gif:
        J.renderer.close()
        from robolab.media import save_gif
        save_gif(frames, a.gif, fps=30)


if __name__ == "__main__":
    main()
