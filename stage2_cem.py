"""Stage 2: tune one catch-dwell-throw unit of the two-ball cycle in full simulation (CEM).

One evaluation: the arm starts at the release state (moving), runs the vacant segment to meet an incoming
ball, re-planned every 20 ms from its true state, then the dwell (absorb, carry 0.2 m sideways, throw).
Scored on where the thrown ball would land relative to the next catch point and when, whether the ball
stayed in the hand, and the joint limits.

Parameters: the gate cycle (stage2_timing.cem_search: catch and release accelerations, catch height and
speed factor, release pose and palm pitch rate), plus
    D_open      how long before the release the hand opens flat
    grip_delay  how long after the catch the fingers curl
    tilt        feedforward palm tilt during the dwell, as a fraction of the angle that points the
                gravity-plus-inertia force into the palm (tested: the wrist has no speed to spare for it at
                this tempo, 1.5x the limits at a third of the full tilt; kept at 0 by default)
    cage        the grip during the carry, from the Stage 1 hold (0) to the fingers closed over the ball
                (1, juggle.CAGE): the carry pushes the ball sideways at up to 9.5 m/s^2 against 10-15 m/s^2
                into the palm, so on an open palm it rolls off
Incoming balls (common random numbers, the same for every candidate in a generation): the nominal path with
a velocity kick, a position offset and a random spin (thrown balls spin 20-30 rad/s).

    PYTHONPATH=.:../robolab python3 stage2_cem.py --gens 40 --out runs/cycle2_v1.json
"""
import argparse
import json
import os

import mujoco as mj
import numpy as np

import juggle
import scene
import stage2_juggle
import stage2_timing
from juggle import G, Seg
from planner import Planner
from robolab import evalpool
from stage1_juggle import peak_ratio

NZ = 12                       # gate parameters
EXTRA = ["D_open", "grip_delay", "tilt", "cage"]


class Tilt:
    """Feedforward palm tilt about the palm's finger axis (z), through the ball point, along a planned
    segment: theta(t) = gain * atan2(a_x, g + a_n), with a_x and a_n the ball point's planned acceleration
    across the palm and along its normal (palm frame), so the net of gravity and inertia points into the palm."""

    def __init__(self, J, pl, seg, gain, t_end, ramp=0.08, h=2e-3):
        self.J, self.pl, self.seg, self.gain, self.h = J, pl, seg, gain, h
        self.t_end, self.ramp = t_end, ramp           # faded in after the catch, out before the hand opens

    def theta(self, t):
        seg, pl, h = self.seg, self.pl, self.h
        t = min(max(t, seg.t0 + h), seg.t0 + seg.T - h)
        p0, p1, p2 = (pl.ball_point(seg(t + k * h)[0]) for k in (-1, 0, 1))
        a = (p2 - 2 * p1 + p0) / h ** 2 + np.r_[0.0, 0.0, G]      # inertial + gravity, the ball must feel
        R = pl.fk(seg(t)[0])[1]
        ap = R.T @ a
        # the palm site's y axis points out of the back of the hand: -ap[1] is the push into the palm
        th = float(np.clip(self.gain * np.arctan2(ap[0], max(-ap[1], 0.3 * G)), -0.5, 0.5))
        s = min(1.0, max(0.0, min(t - seg.t0, self.t_end - t) / self.ramp))
        return th * s * s * (3 - 2 * s)

    def apply(self, t, q, qd, qdd, dt=2e-3):
        th0, th1, th2 = (self.theta(t + k * dt) for k in (-1, 0, 1))
        Jm = self.pl.jac6(q)
        axis = self.pl.fk(q)[1][:, 2]
        k = np.linalg.solve(Jm, np.r_[0.0, 0.0, 0.0, axis])
        return q + k * th1, qd + k * (th2 - th0) / (2 * dt), qdd + k * (th2 - 2 * th1 + th0) / dt ** 2


def build(J, pl, base, x):
    """The cycle for parameters x (gate z + extras)."""
    c = base["c"]
    cy = stage2_timing.cycle_states(J, pl, c, base["q_rel"], np.asarray(x[:NZ]), base["vz"], base["V"], base["d_out"])
    c.k, c.qdd_c, c.R_rel = cy["k"], cy["qdd_c"], pl.fk(cy["catch"][0])[1]
    cy["c"] = c
    cy.update(D_open=float(np.clip(x[NZ], 0.0, 0.15)), grip_delay=float(np.clip(x[NZ + 1], 0.0, 0.2)),
              tilt=float(np.clip(x[NZ + 2], -1.5, 1.5)), cage=float(np.clip(x[NZ + 3], 0.0, 1.0)))
    return cy


def incoming(n, seed):
    rng = np.random.default_rng(seed)
    return [dict(kick=rng.normal(0, 0.05, 3), off=np.r_[rng.uniform(-0.01, 0.01, 2), 0.0], spin=rng.normal(0, 20, 3))
            for _ in range(n)]


def unit(J, pl, cy, inc, frames=None, every=16):
    """One catch-dwell-throw. Returns the cost terms."""
    m, d, ids = J.m, J.d, J.ids
    c = cy["c"]
    J.ball_off = c.ball_off
    V, D, Tc = cy["V"], cy["D"], cy["V"] + cy["D"]
    mj.mj_resetData(m, d)
    d.qpos[ids.arm_q] = cy["rel"][0]
    d.qvel[ids.arm_v] = cy["rel"][1]
    for u in ids.hand_u:                                          # fingers at the flat (open) targets
        j = m.actuator_trnid[u, 0]
        if m.actuator_trntype[u] == mj.mjtTrn.mjTRN_JOINT:
            d.qpos[m.jnt_qposadr[j]] = J.flat[u]
    J.hand_target = J.flat
    # the incoming ball: thrown one cycle ago from the release point, now V from its catch, plus disturbances
    p0 = cy["p_rel"] + cy["v_rel"] * Tc + np.r_[0, 0, -0.5 * G * Tc ** 2] + inc["off"]
    v0 = cy["v_rel"] + np.r_[0, 0, -G * Tc] + inc["kick"]
    d.qpos[ids.ball_q[0]:ids.ball_q[0] + 3] = p0
    d.qvel[ids.ball_v[0]:ids.ball_v[0] + 6] = np.r_[v0, inc["spin"]]
    d.qpos[ids.ball_q[1]:ids.ball_q[1] + 3] = [3.0, 3.0, -1.4]   # the other ball, out of the way
    mj.mj_forward(m, d)
    J.stats = dict(qd_ratio=0.0, tau_sat=0, steps=0)
    dt = m.opt.timestep
    z_catch = cy["hit"][2]
    vac = Seg(0.0, V, cy["rel"], cy["catch"])
    t, step, t_hit, next_plan = 0.0, 0, V, 0.0
    phase, seg, tilt, gap, gap_r, rel, held_lost = "vacant", None, None, 0, 0, None, False
    peak = 0.0
    while t < V + D + 0.15:
        b, bv = J.ball(0)
        if phase == "vacant":
            q, qd, qdd = vac(t)
            if t >= next_plan and t_hit - t > 0.03:
                disc = bv[2] ** 2 + 2 * G * (b[2] - z_catch)
                if disc > 0:
                    tau = (bv[2] + np.sqrt(disc)) / G
                    if tau > 0.03:
                        hit = b + bv * tau + np.r_[0, 0, -0.5 * G * tau ** 2]
                        new = Seg(t, tau, (q, qd, qdd), c.catch_state(hit, bv + np.r_[0, 0, -G * tau]))
                        peak = max(peak, peak_ratio(new))
                        vac, t_hit = new, t + tau
                        q, qd, qdd = vac(t)
                next_plan = t + 0.02
            if t >= t_hit:
                end = vac(vac.t0 + vac.T - 1e-9)
                seg = Seg(t, D, end, cy["rel"])
                t_grip, t_open, t_end = t + cy["grip_delay"], t + D - cy["D_open"], t + D
                tilt = Tilt(J, pl, seg, cy["tilt"], t_open) if cy["tilt"] else None
                nxt = Seg(t_end, V, cy["rel"], cy["catch"])
                grip = J.hold + cy["cage"] * (J.cage - J.hold)
                phase, q, qd, qdd = "dwell", *end
        if phase == "dwell":
            # past the release state the hand goes on into the next vacant segment, as in the juggling loop
            q, qd, qdd = seg(t) if t < t_end else nxt(t)
            if tilt is not None and t < t_open:
                q, qd, qdd = tilt.apply(t, q, qd, qdd)
            J.hand_target = grip if t_grip <= t < t_open else J.flat
            touching = J.ball_touching_hand(0)
            if t_grip + 0.05 < t < t_open and not touching:
                gap += 1
                if gap > 15:
                    held_lost = True
                    break
            elif t < t_open:
                gap = 0
            if t >= t_open - 0.1 and rel is None:
                gap_r = 0 if touching else gap_r + 1
                if gap_r >= 3:
                    rel = (t, b.copy(), bv.copy(), J.d.qvel[ids.ball_v[0] + 3:ids.ball_v[0] + 6].copy())
                    break
        J.control_joint(q, qd, qdd)
        mj.mj_step(m, d)
        if frames is not None and step % every == 0:
            frames.append(J.render())
        t += dt
        step += 1
        if J.ball(0)[0][2] < z_catch - 0.6:
            held_lost = True
            break
    caught = phase == "dwell"
    out = dict(qd_ratio=J.stats["qd_ratio"], tau_sat=J.stats["tau_sat"] / max(1, J.stats["steps"]), peak=peak,
               held=not held_lost and rel is not None,
               held_frac=float(np.clip((t - t_hit) / D, 0.0, 1.0)) if caught else 0.0)
    if rel is not None:
        t_r, p, v, w = rel
        disc = v[2] ** 2 + 2 * G * (p[2] - z_catch)
        F = (v[2] + np.sqrt(max(disc, 0.0))) / G
        land = p + v * F + np.r_[0, 0, -0.5 * G * F ** 2]
        # the next catch of this ball: at the nominal catch point, Tc + V after this catch
        t_land = t_r + F
        out.update(land_err=(land - cy["hit"])[:2].tolist(), t_err=float(t_land - (t_hit + Tc + V)),
                   v=v.tolist(), spin=w.tolist(), t_rel=float(t_r - t_hit))
    return out


def cost(r):
    over = max(r["qd_ratio"], r["peak"]) - 0.97
    if over > 0 or r["tau_sat"] > 0.01:                          # outside the limits: as bad as a drop, or worse
        return 8.0 + 10 * max(over, 0.0) + 10 * r["tau_sat"]
    if not r["held"]:
        return 10.0 - 4.0 * r["held_frac"]                         # graded: how far through the dwell it stayed
    c = np.sum((np.array(r["land_err"]) / 0.03) ** 2) + (r["t_err"] / 0.03) ** 2
    c += 20 * max(0.0, r["qd_ratio"] - 0.93) + 20 * max(0.0, r["peak"] - 0.93)
    return float(6 * c / (c + 50))                                 # 0-6: below every dropped ball (6-10)


_BASE = {}


def _base():
    if not _BASE:
        J = juggle.Juggler(2, human=True)
        pl = Planner(J)
        cy0 = stage2_juggle.load_cycle(J, pl)
        R = json.load(open("logs/stage2_timing_cem_d[0.0,0.2].json"))
        b = min(R, key=lambda r: max(r["speed"], r["torque"]))
        import cem_throw, stage1_juggle
        x, _, _ = stage1_juggle.load(stage2_juggle.CATCH)
        _, _, q_rel = cem_throw.segments(x, 0.35, 0.2)
        _BASE.update(J=J, pl=pl, base=dict(c=cy0["c"], q_rel=q_rel, vz=b["vz_set"], V=b["V"], d_out=(0.0, 0.2)),
                     z0=np.array(b["z"]))
    return _BASE


def evaluate_one(i, pop, incs):
    B = _base()
    k, j = divmod(i, len(incs))
    cy = build(B["J"], B["pl"], B["base"], pop[k])
    r = unit(B["J"], B["pl"], cy, incs[j])
    return dict(k=k, cost=cost(r), held=int(r["held"]), qd=r["qd_ratio"], peak=r["peak"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=36)
    ap.add_argument("--incs", type=int, default=6)
    ap.add_argument("--init", help="start from a previous run's x")
    ap.add_argument("--out", default="runs/cycle2_v1.json")
    a = ap.parse_args()
    B = _base()
    x0 = np.r_[B["z0"], 0.05, 0.03, 0.0, 1.0] if not a.init else np.array(json.load(open(a.init))["x"])
    sig = np.r_[np.full(6, 0.5), 0.02, 0.05, 0.05, 0.05, 0.05, 0.4, 0.02, 0.02, 0.0, 0.2]
    rng = np.random.default_rng(0)
    mu, best = x0.copy(), (np.inf, x0.copy())
    for g in range(a.gens):
        incs = incoming(a.incs, seed=1000 + g)
        pop = np.vstack([mu, mu + sig * rng.standard_normal((a.pop - 1, len(mu)))])
        R = evalpool.evaluate(evaluate_one, range(len(pop) * len(incs)), procs=min(12, os.cpu_count()),
                              kwargs=dict(pop=pop, incs=incs))
        C = np.zeros(len(pop)); H = np.zeros(len(pop))
        for r in R:
            C[r["k"]] += r["cost"] / len(incs)
            H[r["k"]] += r["held"] / len(incs)
        idx = np.argsort(C)[:max(4, a.pop // 6)]
        if C[idx[0]] < best[0]:
            best = (C[idx[0]], pop[idx[0]].copy())
        mu, sig = pop[idx].mean(0), 0.7 * sig + 0.3 * pop[idx].std(0)
        print(f"gen {g:2d}  best {C[idx[0]]:.2f}  held {H[idx[0]]:.2f}  mean(elite) {C[idx].mean():.2f}  "
              f"mu-cost {C[0]:.2f} held {H[0]:.2f}  extras {np.round(pop[idx[0]][NZ:], 3).tolist()}", flush=True)
        json.dump(dict(x=best[1].tolist(), cost=best[0], gen=g), open(a.out, "w"))


if __name__ == "__main__":
    main()
