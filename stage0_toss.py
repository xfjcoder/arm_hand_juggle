"""Stage 0: one scripted toss and catch with privileged ball state.

Phases (palm space, vertical throw from the Stage 0 pose):
  lower   rest -> rest, palm down by `stroke`
  throw   quintic up to (release height, v_r, accel -g); the hand opens flat at 85% of the stroke so the
          ball does not drag on the thumb / curled fingers; then a brake (~1.6 g) to rest
  flight  every 20 ms, re-plan a quintic from the current command to the predicted catch point and time,
          arriving with velocity k * ball velocity (so the ball meets the palm at (1 - k) of its speed)
  catch   on first contact (or at the predicted time), decelerate to rest `give` lower, then hold.

    PYTHONPATH=. python3 stage0_toss.py --v 2.0 --gif media_full/stage0_toss.gif
"""
import argparse
import json

import mujoco as mj
import numpy as np

import juggle
from juggle import G, Seg


def toss(J, v_r=2.0, stroke=0.3, t_throw=0.36, k=0.5, give=0.12, t_give=0.25, hold=1.0, noise=0.0, a_rel=1.0, open_at=0.85, brake_g=1.6, brake_v=0.0,
         rng=None, frames=None, fps=30):
    m, d = J.m, J.d
    dt = m.opt.timestep
    p0 = J.p_home.copy()
    off_w = J.R_d @ J.ball_off                       # resting ball centre relative to the palm site
    z_r = p0[2]
    low = p0 - [0, 0, stroke]
    zero = np.zeros(3)
    log = dict(release=None, apex=-9.0, catch=None, phase=[])
    state = dict(seg=Seg(0, 0.4, (p0, zero, zero), (low, zero, zero)), phase="lower", t_phase=0.0,
                 next_plan=0.0, t_hit=None, no_touch=0)
    t = 0.0
    render_every = max(1, int(round(1 / (fps * dt))))
    step = 0
    T_total = 0.4 + 0.25 + t_throw + 2 * v_r / G + t_give + hold + 0.3
    while t < T_total:
        ph = state["phase"]
        b, bv = J.ball()
        log["apex"] = max(log["apex"], b[2])
        if ph == "lower" and t >= 0.65:
            rel = low.copy()
            rel[2] = z_r
            state.update(seg=Seg(t, t_throw, (low, zero, zero), (rel, [0, 0, v_r], [0, 0, -G * a_rel])),
                         phase="throw")
        elif ph == "throw":
            if open_at is not None and t >= state["seg"].t0 + open_at * t_throw and not state.get("braking"):
                J.hand_target = J.flat                # open the hand late in the stroke
            if t >= state["seg"].t0 + state["seg"].T and not state.get("braking"):
                # hard brake after the throw (about 3 g, the arm manages ~4 g downward): the palm ends moving
                # DOWN at 0.3 m/s, away from the ball, so it cannot chase the ball and bump it again
                p_c, v_c, a_c = state["seg"](state["seg"].t0 + state["seg"].T - 1e-9)
                T_b = (v_r + brake_v) / (brake_g * G)
                state.update(seg=Seg(t, T_b, (p_c, v_c, a_c), (p_c + [0, 0, (v_r - brake_v) * T_b / 2], [0, 0, -brake_v], zero)),
                             braking=True)
            state["no_touch"] = 0 if J.ball_touching_hand() else state["no_touch"] + 1
            # release only counts near the stroke's end: an off-centre ball can lose contact briefly mid-stroke
            if state["no_touch"] >= 3 and (state.get("braking") or t >= state["seg"].t0 + 0.8 * state["seg"].T):
                log["release"] = dict(t=t, ball_v=bv.round(3).tolist(), ball_z=round(b[2], 4))
                state.update(phase="flight", next_plan=t)
        elif ph == "flight":
            if J.hand_target is J.flat and (bv[2] < 0.5 or b[2] - J.palm()[0][2] > 0.08):
                J.hand_target = J.cup                 # cup again well before the catch
            # a touch only counts as the catch once the ball is falling (fingers can brush it just after release)
            touching = J.ball_touching_hand() and bv[2] < 0
            if touching or (state["t_hit"] is not None and t >= state["t_hit"] + 0.02):
                log["catch"] = dict(t=t, contact=bool(touching), ball_v=bv.round(3).tolist(),
                                    rel_v=(bv - J.palm()[2][:3]).round(3).tolist())
                p_c, v_c, a_c = state["seg"](t)
                state.update(seg=Seg(t, t_give, (p_c, v_c, a_c), (p_c - [0, 0, give], zero, zero)), phase="catch")
            elif t >= state["next_plan"] and (state["t_hit"] is None or state["t_hit"] - t > 0.04):
                bm = b + (rng.normal(0, noise, 3) if noise else 0)
                # ball centre reaches the catch height z_r + off_w[2] on the way down
                zc = z_r + off_w[2]
                disc = bv[2] ** 2 + 2 * G * (bm[2] - zc)
                if disc > 0:
                    tau = (bv[2] + np.sqrt(disc)) / G
                    hit = bm + bv * tau + [0, 0, -0.5 * G * tau ** 2]
                    v_hit = bv + [0, 0, -G * tau]
                    if tau > 0.03:
                        p_c, v_c, a_c = state["seg"](t)
                        state["seg"] = Seg(t, tau, (p_c, v_c, a_c), (hit - off_w, k * v_hit, zero))
                        state["t_hit"] = t + tau
                state["next_plan"] = t + 0.02
        log["phase"].append(ph)
        p, v, a = state["seg"](t)
        J.control(p, v, a)
        mj.mj_step(m, d)
        if frames is not None and step % render_every == 0:
            frames.append(J.render())
        t += dt
        step += 1
    b, bv = J.ball()
    p, R, _, _ = J.palm()
    rest_err = float(np.linalg.norm(R.T @ (b - p) - J.ball_off))
    held = bool(J.ball_touching_hand() and abs(b[2] - p[2]) < 0.04 and np.linalg.norm(bv) < 0.2)
    log.update(held=held, caught=bool(held and rest_err < 0.04),        # caught = held AND back in the palm centre
               rest_err=round(rest_err, 4), apex_above_release=round(log["apex"] - (z_r + off_w[2]), 3),
               qd_ratio=round(J.stats["qd_ratio"], 3), tau_sat_frac=round(J.stats["tau_sat"] / J.stats["steps"], 4))
    del log["phase"]
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v", type=float, default=2.0)
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--gif")
    a = ap.parse_args()
    J = juggle.Juggler(1)
    J.reset()
    frames = [] if a.gif else None
    if a.gif:
        J.renderer = mj.Renderer(J.m, 480, 640)
    log = toss(J, v_r=a.v, k=a.k, frames=frames)
    print(json.dumps(log, indent=1))
    if a.gif:
        J.renderer.close()
        from robolab.media import save_gif
        save_gif(frames, a.gif, fps=30)


if __name__ == "__main__":
    main()
