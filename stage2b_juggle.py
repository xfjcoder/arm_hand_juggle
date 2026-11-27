"""Stage 2b: two balls, two arms, on the three-ball cascade rhythm (every third throw empty).

Both hands run the periodic cycle from the two-hand gate (stage2b_timing.py), the left half a hand period
behind the right:
    dwell   catch state -> release state: absorb the caught ball, carry it 10 cm inward, throw it across
            (hand opens D_open before the release)
    vacant  release state -> catch state, re-planned every 20 ms from the camera's ballistic fit of the
            incoming ball, so the palm meets it where and when it actually comes down
Throws are clock-locked: throw k at T_start + k Tc / 2, right hand for even k, left for odd. The ball thrown
at k is caught by the other hand F later and thrown again at k + 3, so slot k carries ball k mod 3; with two
balls slot 2 is empty and its hand goes through the motion without one.

The left arm is driven as the mirror of a right arm (kin.mirror): its incoming ball is mirrored into the
right arm's half (y -> -y), planned with the right arm's catch model, and the joint plan mirrored back.

Start: each hand holds its ball at rest at the cocked pose and makes a start-up stroke into its first
release (right at T_start, left Tc / 2 later). A catch counts when the caught ball is thrown again.

    PYTHONPATH=.:../robolab python3 stage2b_juggle.py --throws 20 --gif media_full/stage2b.gif
"""
import argparse
import json

import mujoco as mj
import numpy as np

import juggle
import kin
import scene
import stage2b_cem
from juggle import G, Seg
from planner import BallFit, Planner
from sensing import Camera, Tactile, PRESETS, camera_pos
from stage1_juggle import peak_ratio

CATCH = "runs/catch5_v443.json"
CAMERA = camera_pos(lookat=(1.0, 0.0, 0.4))           # the tracking camera, in front of the two-arm robot
MIR = np.array([1.0, -1.0, 1.0])
MIRX = np.array([-1.0, 1.0, 1.0])              # palm frame: the left hand's x axis is flipped


def load_cycle(JR, run="runs/cycle2b_v5.json", start="runs/start2b_v5.json"):
    """The nominal right-hand cycle (stage2b_cem.py's tuned x, or the gate solution if run is None) in the
    two-arm robot's coordinates (mirror plane y = 0), the Catcher it was built with, and the start-up throw."""
    R = json.load(open(run)) if run else {}
    b = stage2b_cem.gate_row(R.get("pick", stage2b_cem.PICK))
    pl = Planner(JR)
    c, q_rel, q_cock = stage2b_cem.catcher(JR, pl)
    x = np.array(R["x"]) if run else np.r_[b["z"], 0.0, 0.0, 0.0, 0.05, 0.03, 0.0]
    cy = stage2b_cem.build(JR, pl, c, q_rel, b, x, centre=0.0)
    xs = np.array(json.load(open(start))["xs"]) if start else np.r_[0.4, 0.0, 0.0, 0.0, cy["D_open"]]
    q_r, qd_r, qdd_r = cy["rel"]
    cy.update(q_start=q_r - (q_rel - q_cock), T_start=float(np.clip(xs[0], 0.3, 0.8)),
              D_open_start=float(np.clip(xs[4], 0.0, 0.15)),
              rel_start=(q_r, qd_r + np.linalg.solve(pl.jac6(q_r), np.r_[xs[1:4], 0, 0, 0]), qdd_r))
    return cy


class Hand:
    """One hand's cycle. All planning happens in the right arm's half: for the left hand ball states come in
    mirrored and joint states go out mirrored."""

    def __init__(self, J, cy, mirrored, first_throw, T_start, D_open, grip_delay, grip, vmax_ratio, log):
        self.J, self.cy, self.mir = J, cy, mirrored
        self.c, self.pl = cy["c"], cy["pl"]
        self.V, self.D, self.Tc = cy["V"], cy["D"], cy["Tc"]
        self.k = first_throw                                   # index of this hand's next throw
        self.T_start, self.D_open, self.grip_delay, self.grip = T_start, D_open, grip_delay, grip
        self.vmax_ratio, self.log = vmax_ratio, log
        self.phase = "rest"
        self.seg = Seg(self.throw_time(self.k) - T_start, T_start, (cy["q_start"], np.zeros(6), np.zeros(6)),
                       cy["rel_start"])
        self.t_open, self.t_grip = self.throw_time(self.k) - cy["D_open_start"], None
        self.held = None                                       # ball in (or leaving) the hand
        self.incoming = None
        self.gap_h, self.leaving, self.gap, self.last_reading = 0, None, 0, None
        self.fb = self.fb_thrown = self.last_off_t = None
        self.offs, self.catch_dev = [], np.zeros(3)

    def throw_time(self, k):
        return self.T_start + k * self.Tc / 2

    def out(self, s):
        q, qd, qdd = s
        return (kin.mirror(q), -qd, -qdd) if self.mir else (q, qd, qdd)

    def ball_in(self, b, bv):
        return (b * MIR, bv * MIR) if self.mir else (b, bv)


def reset2(m, d, hands, balls, settle=0.8):
    """Both arms at their start poses, hand holding, ball b on hand balls[b]'s palm; settle."""
    mj.mj_resetData(m, d)
    for h in hands:
        J = h.J
        J.hand_target = J.hold
        d.qpos[J.ids.arm_q] = h.out((h.cy["q_start"], np.zeros(6), np.zeros(6)))[0]
        for u in J.ids.hand_u:
            if m.actuator_trntype[u] == mj.mjtTrn.mjTRN_JOINT:
                d.qpos[m.jnt_qposadr[m.actuator_trnid[u, 0]]] = J.cup[u]
    mj.mj_forward(m, d)
    ids = hands[0].J.ids
    for b in range(len(balls), len(ids.balls)):          # the rest out of the way
        d.qpos[ids.ball_q[b]:ids.ball_q[b] + 3] = [3.0, 3.0 + b, -1.4]
    for b, h in enumerate(balls):
        p, R, _, _ = h.J.palm()
        q0 = ids.ball_q[b]
        d.qpos[q0:q0 + 3] = p + np.r_[0, 0, 0.06]
        d.qpos[q0 + 3:q0 + 7] = [1, 0, 0, 0]
    mj.mj_forward(m, d)
    home = []
    for h in hands:                                  # palm-space hold, as Juggler.reset (the start throw was tuned so)
        p, R, _, _ = h.J.palm()
        h.J.R_d = R
        home.append(p)
    for _ in range(int(settle / m.opt.timestep)):
        for h, p in zip(hands, home):
            h.J.control(p, np.zeros(3), np.zeros(3))
        mj.mj_step(m, d)
    for h in hands:
        h.J.stats = dict(qd_ratio=0.0, tau_sat=0, steps=0)


def juggle2b(JR, JL, cy, n_balls=2, throws=20, seed=None, kick=0.0, preset="ideal", noise=0.002, vmax_ratio=0.97,
             frames=None, render=None, every=16, log=None, correct=None, lead=0.25, data=None,
             strict=False):
    m, d = JR.m, JR.d
    dt = m.opt.timestep
    rng = np.random.default_rng(seed)
    cfg = PRESETS[preset]
    srng = np.random.default_rng(None if seed is None else seed + 104729)
    cams = [Camera(JR, srng, noise=noise, ball=i, pos=CAMERA, **cfg["camera"]) for i in range(n_balls)]
    T_start, D_open, grip_delay = cy["T_start"], cy["D_open"], cy["grip_delay"]
    grip = [J.hold + cy["cage"] * (J.cage - J.hold) for J in (JR, JL)]
    hands = [Hand(J, cy, J is JL, k0, T_start, D_open, grip_delay, g, vmax_ratio, log)
             for J, k0, g in ((JR, 0, grip[0]), (JL, 1, grip[1]))]
    tacs = [[Tactile(h.J, srng, ball=i, **cfg["tactile"]) for i in range(n_balls)] for h in hands]
    reset2(m, d, hands, [hands[i % 2] for i in range(min(n_balls, 2))])
    for i, h in enumerate(hands):
        h.held = i if i < n_balls else None
    fits = [None] * n_balls
    if n_balls > 2:
        # the third ball: fed in flight, as if the left hand had thrown it Tc / 2 before the right hand's
        # first throw (a Shadow Hand cannot start with two balls); the right hand catches it first
        s0 = cy["Tc"] / 2 - T_start
        p0, v0 = cy["p_rel"] * MIR, np.r_[cy["v_in"], G * cy["F"] / 2]
        q2, v2 = JR.ids.ball_q[2], JR.ids.ball_v[2]
        d.qpos[q2:q2 + 3] = p0 + v0 * s0 + np.r_[0, 0, -0.5 * G * s0 ** 2]
        d.qpos[q2 + 3:q2 + 7] = [1, 0, 0, 0]
        d.qvel[v2:v2 + 6] = np.r_[v0 + np.r_[0, 0, -G * s0], 0, 0, 0]
        mj.mj_forward(m, d)
        fits[2] = BallFit()
        fits[2].t0 = -1.0
    slot = lambda k: (k % 3) if (k % 3) < n_balls else None
    z_catch = cy["hit"][2]
    t, step, why, catches, thrown = 0.0, 0, None, 0, 0
    releases = []
    while True:
        frames_in = [cam.step(t) for cam in cams]
        for i in range(n_balls):
            if fits[i] is not None:
                for tcap, p in frames_in[i]:
                    if tcap > fits[i].t0:
                        fits[i].add(tcap, p)
        for hi, h in enumerate(hands):
            J = h.J
            readings = [tc.step(t) for tc in tacs[hi]]
            if h.leaving is not None:                             # the thrown ball: watch for its release
                r = readings[h.leaving]                           # two new readings without contact
                if r is not None and r[0] != h.last_reading:
                    h.last_reading = r[0]
                    h.gap = 0 if r[1] else h.gap + 1
                if h.gap >= 2:
                    b = h.leaving
                    if kick:
                        d.qvel[J.ids.ball_v[b]:J.ids.ball_v[b] + 3] += rng.normal(0, kick, 3)
                        mj.mj_forward(m, d)
                    pb, vb = JR.ball(b)
                    disc = vb[2] ** 2 + 2 * G * (pb[2] - z_catch)
                    Fr = (vb[2] + np.sqrt(max(disc, 0.0))) / G
                    land = pb + vb * Fr + np.r_[0, 0, -0.5 * G * Fr ** 2]
                    target = cy["hit"] * (MIR if not h.mir else 1)          # the other hand's catch spot
                    releases.append(dict(t=round(t, 3), hand="LR"[not h.mir], v=vb.round(3).tolist(), ball=b,
                                         t_throw=round(h.throw_time(h.k - 2), 3),
                                         land_err=(land - target)[:2].round(3).tolist(),
                                         t_err=round(t + Fr - (h.throw_time(h.k - 2) + cy["F"]), 3)))
                    if data is not None and h.fb_thrown is not None:
                        e = releases[-1]
                        data.append(dict(f=h.fb_thrown.round(5).tolist(),
                                         y=[e["land_err"][0], e["land_err"][1] * (-1 if h.mir else 1), e["t_err"]]))
                    if log:
                        log(f"  {releases[-1]}")
                    fits[b] = BallFit()
                    fits[b].t0 = t
                    h.leaving = None
            if h.phase in ("rest", "dwell"):
                q, qd, qdd = h.seg(t) if t >= h.seg.t0 else h.seg(h.seg.t0)
                if h.t_grip is not None and t >= h.t_grip:
                    J.hand_target = h.grip
                if t >= h.t_open:
                    J.hand_target = J.flat
                    if h.held is not None and h.leaving is None:
                        h.leaving, h.gap = h.held, 0
                elif h.held is not None and h.t_grip is not None and t > h.t_grip + 0.1:
                    r = readings[h.held]                          # the caught ball must stay in the hand
                    h.gap_h = h.gap_h + 1 if (r is not None and not r[1]) else 0
                    if h.gap_h > 20:
                        why = f"not held ({'LR'[not h.mir]})"
                if h.held is not None and h.t_grip is not None and h.fb is None:
                    # in-hand feedback: the skin's ball position (palm frame, the left palm's x flipped into the
                    # right's) plus where and when the ball was caught predict the throw's landing error; at
                    # `lead` before the release the rest of the stroke is re-planned to cancel it
                    r = readings[h.held]
                    if r is not None and r[1] and r[2] is not None and r[0] != h.last_off_t:
                        h.last_off_t = r[0]
                        h.offs.append(r[2] * (MIRX if h.mir else 1))
                    T_rel = h.throw_time(h.k)
                    if t >= T_rel - lead and h.offs:
                        h.fb = np.r_[np.mean(h.offs[-3:], 0), h.catch_dev]
                        if correct is not None:
                            pred = np.asarray(correct["A"]) @ h.fb + np.asarray(correct["b"])
                            gain = correct.get("gain", 1.0)
                            dv = -gain * np.r_[pred[0] / cy["F"], pred[1] / cy["F"], 0.5 * G * pred[2]]
                            q_r, qd_r, qdd_r = cy["rel"]
                            qd_r = qd_r + np.linalg.solve(h.pl.jac6(q_r), np.r_[dv, 0, 0, 0])
                            h.seg = Seg(t, T_rel - t, (q, qd, qdd), (q_r, qd_r, qdd_r))
                if t >= h.throw_time(h.k):                        # the release state: start the vacant
                    if h.held is not None:
                        thrown += 1
                        if h.t_grip is not None:
                            catches += 1
                    h.fb_thrown, h.fb = h.fb, None
                    h.held = None
                    h.incoming = slot(h.k - 1)
                    h.vac = Seg(t, h.V, h.seg(h.seg.t0 + h.seg.T - 1e-9), cy["catch"])   # on from where the stroke ends
                    h.phase, h.t_hit, h.next_plan = "vacant", t + h.V, t
                    h.hit_plan = None                             # set by the first feasible plan
                    h.k += 2
                    q, qd, qdd = h.vac(t)
            else:
                q, qd, qdd = h.vac(t)
                f = fits[h.incoming] if h.incoming is not None else None
                if t >= h.next_plan and f is not None and f.ready() and h.t_hit - t > 0.03:
                    b, bv = h.ball_in(*f.state(t))
                    # catch at the nominal height; if the hand cannot get there in time within the speed limits,
                    # lower (the ball falls further, which buys up to ~0.06 s)
                    ok = pk = hit = None
                    for dz in (0.0, 0.1, 0.2, 0.3):
                        disc = bv[2] ** 2 + 2 * G * (b[2] - (z_catch - dz))
                        if disc <= 0:
                            continue
                        tau = (bv[2] + np.sqrt(disc)) / G
                        if tau <= 0.03:
                            continue
                        hit = b + bv * tau + np.r_[0, 0, -0.5 * G * tau ** 2]
                        new = Seg(t, tau, (q, qd, qdd), h.c.catch_state(hit, bv + np.r_[0, 0, -G * tau]))
                        pk = peak_ratio(new)
                        if pk <= vmax_ratio:
                            ok = (new, tau, hit)
                            break
                    if ok is None and (strict or h.hit_plan is None):
                        why = f"approach too fast ({'LR'[not h.mir]})"
                        if log and pk is not None:
                            tb, tv = h.ball_in(*JR.ball(h.incoming))
                            log(f"  approach: hit {hit.round(3)} peak {pk:.2f}; fit {b.round(3)} {bv.round(2)}, "
                                f"true {tb.round(3)} {tv.round(2)}, fit from {f.t0:.3f}, {len(f.t)} frames")
                    elif ok is not None:
                        h.vac, h.t_hit = ok[0], t + ok[1]
                        h.hit_plan = ok[2]
                        q, qd, qdd = h.vac(t)
                    h.next_plan = t + 0.02
                if t >= h.t_hit:
                    end = h.vac(h.vac.t0 + h.vac.T - 1e-9)                 # a Seg holds zero velocity past its end
                    T_rel = h.throw_time(h.k)
                    h.seg = Seg(t, T_rel - t, end, cy["rel"])
                    h.t_open = T_rel - h.D_open
                    h.held = h.incoming
                    h.t_grip = t + h.grip_delay if h.held is not None else None
                    t_nom = h.throw_time(h.k - 2) + h.V
                    hp = cy["hit"] if h.hit_plan is None else h.hit_plan
                    h.catch_dev = np.r_[(hp - cy["hit"])[:2], t - t_nom]
                    h.offs, h.fb = [], None
                    h.phase, h.gap_h = "dwell", 0
                    q, qd, qdd = end
            J.control_joint(*h.out((q, qd, qdd)))
        mj.mj_step(m, d)
        if frames is not None and step % every == 0:
            frames.append(render())
        t += dt
        step += 1
        if why or catches >= throws:
            break
        zs = [JR.ball(i)[0][2] for i in range(n_balls)]
        if min(zs) < -1.4 or t > 120:
            why = "ball dropped" if min(zs) < -1.4 else "time"
            break
    st = [h.J.stats for h in hands]
    return dict(catches=catches, why=why if catches < throws else None, t=round(t, 2),
                qd_ratio=round(max(s["qd_ratio"] for s in st), 3),
                tau_sat=round(max(s["tau_sat"] / max(1, s["steps"]) for s in st), 4), releases=releases)


def make(n_balls=2):
    m = scene.build(n_balls, human=True, arms=2)
    d = mj.MjData(m)
    JR = juggle.Juggler(side="R", m=m, d=d)
    JL = juggle.Juggler(side="L", m=m, d=d)
    return JR, JL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--balls", type=int, default=2)
    ap.add_argument("--throws", type=int, default=20)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--kick", type=float, default=0.0)
    ap.add_argument("--sensors", default="ideal", choices=["ideal", "real"])
    ap.add_argument("--run", default="runs/cycle2b_v5.json")
    ap.add_argument("--start", default="runs/start2b_v5.json")
    ap.add_argument("--gif")
    a = ap.parse_args()
    JR, JL = make(a.balls)
    cy = load_cycle(JR, a.run or None, a.start or None)
    frames = render = None
    if a.gif:
        # front and three-quarter views; frame from the hands' lowest pose (z -0.85) to the apex (0.85)
        r = mj.Renderer(JR.m, 720, 540)
        for i, rgba in enumerate(([1.0, 0.25, 0.2, 1], [0.35, 0.85, 1.0, 1], [1.0, 0.85, 0.2, 1])[:a.balls]):
            JR.m.geom_rgba[JR.m.body_geomadr[JR.ids.balls[i]]] = rgba        # bright on the dark background
        JR.m.vis.headlight.ambient[:], JR.m.vis.headlight.diffuse[:] = [0.35] * 3, [0.6] * 3   # balls lit at the apex
        cams = []
        for az in (180, 145):
            cam = mj.MjvCamera()
            cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = [0.9, 0.0, 0.0], 2.8, az, -6
            cams.append(cam)
        def render():
            out = []
            for cam in cams:
                r.update_scene(JR.d, cam)
                out.append(r.render())
            return np.concatenate(out, 1)
        frames = []
    out = juggle2b(JR, JL, cy, n_balls=a.balls, throws=a.throws, seed=a.seed, kick=a.kick, preset=a.sensors,
                   frames=frames, render=render, log=print)
    print(json.dumps({k: v for k, v in out.items() if k != "releases"}))
    if a.gif:
        r.close()
        from robolab.media import save_gif
        save_gif(frames, a.gif, fps=30)


if __name__ == "__main__":
    main()
