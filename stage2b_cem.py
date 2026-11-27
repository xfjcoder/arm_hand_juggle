"""Stage 2b: tune one catch-dwell-throw unit of the two-hand cascade cycle in full simulation (CEM).

The gate (stage2b_timing.py) treats the ball as fixed in the palm. In the simulation it rolls about 3 cm along
the fingers during the throw and leaves with ~1 m/s of unplanned forward speed, so the cycle is tuned here as
Stage 1's throw was: on the full simulation's outcome.

One evaluation, the right arm in the one-arm model (its dynamics are the two-arm robot's right arm; the left
is its exact mirror): the arm starts at the release state, moving, with the hand open; runs the vacant
segment to meet a ball thrown by the (mirrored) left hand, re-planned every 20 ms from the ball's true state;
catches it, carries it inward and throws it at the scheduled time (V + D after the vacant began). Scored on
where the thrown ball would land relative to the left hand's catch spot and when, whether the ball stayed in
the hand, and the joint limits.

Parameters: the gate's 11 (catch and release accelerations, catch speed factor, release pose, palm pitch
rate), a bias on the commanded release velocity (3), D_open (the hand opens flat that long before the
release), grip_delay (the fingers curl that long after the catch) and cage (the carry grip, from the Stage 1
hold, 0, to the fingers closed over the ball, 1).
Incoming balls (common random numbers per generation): the nominal path with a velocity kick, a position
offset and a random spin.

    PYTHONPATH=.:../robolab python3 stage2b_cem.py --gens 40 --out runs/cycle2b_v1.json
    PYTHONPATH=.:../robolab python3 stage2b_cem.py --start runs/cycle2b_v1.json --out runs/start2b_v1.json

The start-up throw (from rest, ball settled in the hand, into the release state) is tuned the same way with
--start: its duration, a bias on its release velocity and D_open.
"""
import argparse
import json
import os

import mujoco as mj
import numpy as np

import cem_throw
import juggle
import stage1_juggle
import stage2b_timing as gate
from cem_catch import Catcher
from juggle import G, Seg
from planner import Planner
from robolab import evalpool
from stage1_juggle import peak_ratio

NZ = 11                                 # gate parameters
REACH_MIN = None                        # --reach-min: minimum front-back catch reach (m), None: not scored
EXTRA = ["vb_x", "vb_y", "vb_z", "D_open", "grip_delay", "cage"]
PICK = dict(pattern="cascade", vz=4.43, d_y=0.10, V=0.35)


def gate_row(pick=PICK, log="logs/stage2b_timing.json"):
    R = json.load(open(log))
    return [r for r in R if all(abs(r[k] - v) < 0.005 if isinstance(v, float) else r[k] == v for k, v in pick.items())][0]


def catcher(J, pl):
    x, cp, off = stage1_juggle.load("runs/catch5_v443.json")
    th, br, q_rel = cem_throw.segments(x, 0.35, 0.2)
    _, R_rel = pl.fk(q_rel)
    c = Catcher(J, pl, cp, q_rel, th(0)[0], R_rel, np.array(off))
    J.ball_off = c.ball_off
    return c, q_rel, th(0)[0]


def build(J, pl, c, q_rel, b, x, centre=gate.CENTRE_Y):
    """The cycle for parameters x (gate z + extras), right arm, mirror plane y = centre."""
    x = np.asarray(x, float)
    cy = gate.states(J, pl, c, q_rel, x[:NZ], b["vz"], b["V"], b["D"], b["c_off"], centre=centre, vb=x[NZ:NZ + 3])
    c.k, c.qdd_c = cy["k"], cy["catch"][2]
    c.R_rel = pl.fk(cy["catch"][0])[1]
    cy.update(c=c, pl=pl, Tc=b["V"] + b["D"], centre=centre, D_open=float(np.clip(x[NZ + 3], 0.0, 0.3)),
              grip_delay=float(np.clip(x[NZ + 4], 0.0, 0.2)), cage=float(np.clip(x[NZ + 5], 0.0, 1.0)))
    return cy


def incoming(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return [dict(kick=scale * rng.normal(0, 0.05, 3), off=scale * np.r_[rng.uniform(-0.01, 0.01, 2), 0.0],
                 spin=rng.normal(0, 20, 3)) for _ in range(n)]


def unit(J, pl, cy, inc, frames=None, render=None, every=16):
    """One catch-dwell-throw. Returns the cost terms."""
    m, d, ids = J.m, J.d, J.ids
    c = cy["c"]
    J.ball_off = c.ball_off
    V, D, F = cy["V"], cy["D"], cy["F"]
    mj.mj_resetData(m, d)
    d.qpos[ids.arm_q] = cy["rel"][0]
    d.qvel[ids.arm_v] = cy["rel"][1]
    for u in ids.hand_u:                                          # fingers at the flat (open) targets
        if m.actuator_trntype[u] == mj.mjtTrn.mjTRN_JOINT:
            d.qpos[m.jnt_qposadr[m.actuator_trnid[u, 0]]] = J.flat[u]
    J.hand_target = J.flat
    # the incoming ball: thrown by the left hand from the mirror of our release spot, F - V ago
    s = F - V
    v_th = np.r_[cy["v_in"], G * F / 2]
    p0 = gate.mirror_p(cy["p_rel"], cy["centre"]) + v_th * s + np.r_[0, 0, -0.5 * G * s * s] + inc["off"]
    v0 = v_th + np.r_[0, 0, -G * s] + inc["kick"]
    d.qpos[ids.ball_q[0]:ids.ball_q[0] + 3] = p0
    d.qvel[ids.ball_v[0]:ids.ball_v[0] + 6] = np.r_[v0, inc["spin"]]
    for i in range(1, len(ids.balls)):                            # other balls out of the way
        d.qpos[ids.ball_q[i]:ids.ball_q[i] + 3] = [3.0, 3.0 + i, -1.4]
    mj.mj_forward(m, d)
    J.stats = dict(qd_ratio=0.0, tau_sat=0, steps=0)
    dt = m.opt.timestep
    z_catch = cy["hit"][2]
    vac = Seg(0.0, V, cy["rel"], cy["catch"])
    t_end = V + D                                                 # the scheduled release
    t, step, t_hit, next_plan = 0.0, 0, V, 0.0
    phase, gap, gap_r, rel, held_lost, peak, t_sep = "vacant", 0, 0, None, False, 0.0, None
    grip = J.hold + cy["cage"] * (J.cage - J.hold)
    nxt = Seg(t_end, V, cy["rel"], cy["catch"])
    while t < t_end + 0.25:
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
                seg = Seg(t, t_end - t, end, cy["rel"])
                t_grip, t_open = t + cy["grip_delay"], t_end - cy["D_open"]
                phase, q, qd, qdd = "dwell", *end
        if phase == "dwell":
            q, qd, qdd = seg(t) if t < t_end else nxt(t)          # past the release: on into the next vacant
            J.hand_target = grip if t_grip <= t < t_open else J.flat
            touching = J.ball_touching_hand(0)
            if t_grip + 0.05 < t < t_open and not touching:
                gap += 1
                if gap > 15:
                    held_lost = True
                    break
            elif t < t_open:
                gap = 0
            if t >= t_open - 0.1:
                gap_r = 0 if touching else gap_r + 1
                if gap_r >= 3 and t_sep is None:
                    t_sep = t
                if touching:
                    t_sep = None                                  # pushed again: not released yet
                if t_sep is not None and t >= t_sep + 0.06:       # scored on the free flight after leaving
                    rel = (t, b.copy(), bv.copy())
                    break
        J.control_joint(q, qd, qdd)
        mj.mj_step(m, d)
        if frames is not None and step % every == 0:
            frames.append(render())
        t += dt
        step += 1
        if J.ball(0)[0][2] < -1.3:                                  # on the floor (the dwell dips ~0.6 m)
            held_lost = True
            break
    caught = phase == "dwell"
    out = dict(qd_ratio=J.stats["qd_ratio"], tau_sat=J.stats["tau_sat"] / max(1, J.stats["steps"]), peak=peak,
               held=not held_lost and rel is not None,
               held_frac=float(np.clip((t - t_hit) / D, 0.0, 1.0)) if caught else 0.0)
    if rel is not None:
        t_r, p, v = rel
        disc = v[2] ** 2 + 2 * G * (p[2] - z_catch)
        Fr = (v[2] + np.sqrt(max(disc, 0.0))) / G
        land = p + v * Fr + np.r_[0, 0, -0.5 * G * Fr ** 2]
        target = gate.mirror_p(cy["hit"], cy["centre"])           # the left hand's catch spot
        out.update(land_err=(land - target)[:2].tolist(), t_err=float(t_r + Fr - (t_end + F)), v=v.tolist(),
                   t_rel=float(t_r - t_end))
    return out


def reach_x(cy, vmax_ratio=0.97, iters=8):
    """Smallest front-back catch reach (m) of the nominal vacant segment: how far the catch point can move
    toward or away from the body, arriving at the same time, within vmax_ratio of the speed limits."""
    c = cy["c"]
    out = []
    for sign in (1, -1):
        lo, hi = 0.0, 0.3
        for _ in range(iters):
            mid = (lo + hi) / 2
            hit = cy["hit"] + np.r_[sign * mid, 0.0, 0.0]
            ok = peak_ratio(Seg(0.0, cy["V"], cy["rel"], c.catch_state(hit, cy["v_hit"]))) <= vmax_ratio
            lo, hi = (mid, hi) if ok else (lo, mid)
        out.append(lo)
    return min(out)


def cost(r):
    over = max(r["qd_ratio"], r["peak"]) - 0.97
    if over > 0 or r["tau_sat"] > 0.01:                          # outside the limits: as bad as a drop, or worse
        return 8.0 + 10 * max(over, 0.0) + 10 * r["tau_sat"]
    if not r["held"]:
        return 10.0 - 4.0 * r["held_frac"]                         # graded: how far through the dwell it stayed
    c = np.sum((np.array(r["land_err"]) / 0.03) ** 2) + (r["t_err"] / 0.03) ** 2
    c += 20 * max(0.0, r["qd_ratio"] - 0.93) + 20 * max(0.0, r["peak"] - 0.93)
    if "reach" in r:                                               # keep room to chase imperfect throws
        c += (max(0.0, REACH_MIN - r["reach"]) / 0.01) ** 2
    return float(6 * c / (c + 50))                                 # 0-6: below every dropped ball (6-10)


_BASE = {}


def _base(pick=None):
    if not _BASE or (pick is not None and _BASE["pick"] != pick):
        J = juggle.Juggler(1, human=True)
        pl = Planner(J)
        c, q_rel, _ = catcher(J, pl)
        pick = pick or PICK
        _BASE.update(J=J, pl=pl, c=c, q_rel=q_rel, b=gate_row(pick), pick=pick)
    return _BASE


def evaluate_one(i, pop, incs, pick=None, reach_min=None):
    global REACH_MIN
    REACH_MIN = reach_min
    B = _base(pick)
    k, j = divmod(i, len(incs))
    cy = build(B["J"], B["pl"], B["c"], B["q_rel"], B["b"], pop[k])
    r = unit(B["J"], B["pl"], cy, incs[j])
    if reach_min is not None:
        r["reach"] = reach_x(build(B["J"], B["pl"], B["c"], B["q_rel"], B["b"], pop[k]))
    return dict(k=k, cost=cost(r), held=int(r["held"]), qd=r["qd_ratio"], peak=r["peak"],
                land=r.get("land_err"), t_err=r.get("t_err"), reach=r.get("reach"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=40)
    ap.add_argument("--pop", type=int, default=36)
    ap.add_argument("--incs", type=int, default=6)
    ap.add_argument("--init", help="start from a previous run's x")
    ap.add_argument("--out", default="runs/cycle2b_v1.json")
    ap.add_argument("--start", help="tune the start-up throw for this cycle run instead (writes --out)")
    ap.add_argument("--vz", type=float, default=PICK["vz"], help="gate cycle: toss speed")
    ap.add_argument("--V", type=float, default=PICK["V"], help="gate cycle: empty time")
    ap.add_argument("--reach-min", type=float, help="penalise front-back catch reach below this (m)")
    ap.add_argument("--inc-scale", type=float, default=1.0, help="incoming-ball disturbance scale")
    a = ap.parse_args()
    if a.start:
        return tune_start(a.start, a.out)
    pick = dict(PICK, vz=a.vz, V=a.V)
    B = _base(pick)
    x0 = np.r_[B["b"]["z"], 0.0, 0.0, 0.0, 0.07, 0.0, 0.86] if not a.init else np.array(json.load(open(a.init))["x"])
    sig = np.r_[np.full(6, 0.5), 0.05, 0.05, 0.05, 0.05, 0.4, 0.3, 0.3, 0.3, 0.02, 0.02, 0.2]
    rng = np.random.default_rng(0)
    mu, best = x0.copy(), (np.inf, x0.copy())
    for g in range(a.gens):
        incs = incoming(a.incs, seed=1000 + g, scale=a.inc_scale)
        pop = np.vstack([mu, mu + sig * rng.standard_normal((a.pop - 1, len(mu)))])
        R = evalpool.evaluate(evaluate_one, range(len(pop) * len(incs)), procs=min(12, os.cpu_count()),
                              kwargs=dict(pop=pop, incs=incs, pick=pick, reach_min=a.reach_min))
        C, H = np.zeros(len(pop)), np.zeros(len(pop))
        for r in R:
            C[r["k"]] += r["cost"] / len(incs)
            H[r["k"]] += r["held"] / len(incs)
        idx = np.argsort(C)[:max(4, a.pop // 6)]
        if C[idx[0]] < best[0]:
            best = (C[idx[0]], pop[idx[0]].copy())
        mu, sig = pop[idx].mean(0), 0.7 * sig + 0.3 * pop[idx].std(0)
        lands = [r["land"] for r in R if r["k"] == idx[0] and r["land"] is not None]
        reach = [r["reach"] for r in R if r["k"] == idx[0] and r["reach"] is not None]
        print(f"gen {g:2d}  best {C[idx[0]]:.2f}  held {H[idx[0]]:.2f}  mean(elite) {C[idx].mean():.2f}  "
              f"mu-cost {C[0]:.2f} held {H[0]:.2f}  land {np.round(np.mean(lands, 0), 3).tolist() if lands else None}  "
              f"reach {reach[0] if reach else None}  extras {np.round(pop[idx[0]][NZ:], 3).tolist()}", flush=True)
        json.dump(dict(x=best[1].tolist(), cost=best[0], gen=g, pick=pick, reach_min=a.reach_min), open(a.out, "w"))


def start_unit(J, pl, cy, xs, q_start, frames=None, render=None, every=16):
    """The start-up throw: from rest at q_start, ball settled in the holding hand, a stroke of T_s into the
    release state (its commanded ball velocity biased by vb_s), hand open D_open before the end.
    xs = [T_s, vb_s (3), D_open]. Returns the landing error at the left hand's catch spot and the time error."""
    T_s, vb, D_open = float(np.clip(xs[0], 0.3, 0.8)), np.asarray(xs[1:4]), float(np.clip(xs[4], 0.0, 0.15))
    m, d = J.m, J.d
    J.reset(q=q_start, grip=J.hold)
    q_r, qd_r, qdd_r = cy["rel"]
    qd_r = qd_r + np.linalg.solve(pl.jac6(q_r), np.r_[vb, 0, 0, 0])
    seg = Seg(0.0, T_s, (q_start, np.zeros(6), np.zeros(6)), (q_r, qd_r, qdd_r))
    nxt = Seg(T_s, cy["V"], (q_r, qd_r, qdd_r), cy["catch"])
    t, gap, rel, step, t_sep = 0.0, 0, None, 0, None
    z_catch = cy["hit"][2]
    while t < T_s + 0.25:
        q, qd, qdd = seg(t) if t < T_s else nxt(t)
        J.hand_target = J.hold if t < T_s - D_open else J.flat
        if t >= T_s - D_open - 0.05:
            touching = J.ball_touching_hand(0)
            gap = 0 if touching else gap + 1
            if gap >= 3 and t_sep is None:
                t_sep = t
            if touching:
                t_sep = None
            if t_sep is not None and t >= t_sep + 0.06:
                rel = (t, *J.ball(0))
                break
        J.control_joint(q, qd, qdd)
        mj.mj_step(m, d)
        if frames is not None and step % every == 0:
            frames.append(render())
        t += m.opt.timestep
        step += 1
    out = dict(qd_ratio=J.stats["qd_ratio"], tau_sat=J.stats["tau_sat"] / max(1, J.stats["steps"]),
               peak=peak_ratio(seg), held=rel is not None, held_frac=1.0 if rel is not None else 0.0)
    if rel is not None:
        t_r, p, v = rel
        disc = v[2] ** 2 + 2 * G * (p[2] - z_catch)
        Fr = (v[2] + np.sqrt(max(disc, 0.0))) / G
        land = p + v * Fr + np.r_[0, 0, -0.5 * G * Fr ** 2]
        out.update(land_err=(land - gate.mirror_p(cy["hit"], cy["centre"]))[:2].tolist(),
                   t_err=float(t_r + Fr - (T_s + cy["F"])), v=v.tolist())
    return out


def tune_start(cycle_run, out, gens=25, pop=24):
    """CEM on start_unit (deterministic: one evaluation per candidate)."""
    run = json.load(open(cycle_run))
    B = _base(run.get("pick", PICK))
    x = run["x"]
    cy = build(B["J"], B["pl"], B["c"], B["q_rel"], B["b"], x)
    _, q_rel, q_cock = catcher(B["J"], B["pl"])
    q_start = cy["rel"][0] - (q_rel - q_cock)
    rng = np.random.default_rng(0)
    mu, sig = np.r_[0.4, 0.0, 0.0, 0.0, 0.05], np.r_[0.08, 0.3, 0.3, 0.3, 0.02]
    best = (np.inf, mu)
    for g in range(gens):
        P = np.vstack([mu, mu + sig * rng.standard_normal((pop - 1, 5))])
        R = [start_unit(B["J"], B["pl"], cy, xs, q_start) for xs in P]
        C = np.array([cost(r) for r in R])
        idx = np.argsort(C)[:6]
        if C[idx[0]] < best[0]:
            best = (C[idx[0]], P[idx[0]].copy(), R[idx[0]])
        mu, sig = P[idx].mean(0), 0.7 * sig + 0.3 * P[idx].std(0)
        print(f"start gen {g:2d} best {C[idx[0]]:.3f} land {np.round(R[idx[0]].get('land_err', []), 3).tolist()} "
              f"t_err {R[idx[0]].get('t_err', 0):+.3f} peak {R[idx[0]]['peak']:.2f}  xs {np.round(P[idx[0]], 3).tolist()}", flush=True)
    json.dump(dict(xs=best[1].tolist(), cost=best[0], result={k: v for k, v in best[2].items()}), open(out, "w"))


if __name__ == "__main__":
    main()
