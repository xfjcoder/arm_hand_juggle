"""Stage 2 gate: can the arm keep a two-ball, one-hand rhythm within the UR10e limits?

Two balls in one hand: the hand throws every Tc, holding a ball for the dwell D and empty for V = Tc - D.
Each ball flies F = 2 Tc - D = 2 V + D (it is thrown, the other ball is caught V later, thrown Tc later, and the
first comes back V after that). So the toss height sets F, and F must cover two empty intervals and one dwell.

The hand motion is periodic, two quintic joint segments:
    vacant  release state -> catch state, V seconds (brake, move out, meet the falling ball)
    dwell   catch state -> release state, D seconds (absorb, swing in, throw)
Release: the tuned Stage 1 throw's release state (runs/throw2_v443.json) with its joint speeds scaled by s,
so the ball leaves at about s * 4.4 m/s, plus a sideways component so it lands `d` m outward (a one-hand
fountain: thrown from the inside, caught on the outside, so the rising ball misses the falling one).
Catch: the tuned catch state (runs/catch5_v443.json) at the landing point.

Checked along the cycle: joint speeds against the UR10e limits, joint torques (inverse dynamics with the
hand) against the UR10e limits, and during the dwell that the palm never pulls away from the ball (its
acceleration along the palm normal stays above -g). Prints, per toss height, the feasible (V, D) with the
most margin.

    PYTHONPATH=. python3 stage2_timing.py
"""
import argparse

import mujoco as mj
import numpy as np

import cem_throw
import juggle
import scene
import stage1_juggle
from cem_catch import Catcher
from juggle import G, Seg
from planner import Planner
from scipy.optimize import minimize

P = [1, 2, 3]          # the planar chain: shoulder_lift, elbow, wrist_1


def cycle_check(J, pl, c, rel, s, V, D, d_out, acc=None, n=60, dz=0.0, k=None):
    """Peak speed ratio, peak torque ratio, the lowest vertical acceleration of the ball point while the ball
    is held (dwell, before the last 10%) and the release acceleration (end of the dwell) for one cycle.
    acc: accelerations of the three planar joints at the catch and at the release (6 values, rad/s^2 / 10);
    None keeps the tuned ones. dz: catch height relative to the release (the ball falls further and flies
    longer: D is then recomputed as F - 2 V); k: palm's downward speed at the catch as a fraction of the ball's."""
    q_r, qd_r, qdd_r = rel
    qd_r, qdd_r = s * qd_r, s * s * qdd_r
    J.ball_off = c.ball_off
    Jm = pl.jac6(q_r)
    v_r = Jm[:3] @ qd_r                                  # ball-point velocity at release
    F = 2 * V + D
    if dz:
        F = (v_r[2] + np.sqrt(v_r[2] ** 2 - 2 * G * dz)) / G
        D = F - 2 * V
    side = np.array([1.0, 0.0, 0.0]) * d_out / F         # outward drift (world x: away from the column)
    qd_r = qd_r + np.linalg.solve(Jm, np.r_[side, 0, 0, 0])
    p_r = pl.ball_point(q_r)
    hit = p_r + np.r_[d_out, 0.0, dz]
    v_hit = np.r_[side[0], 0.0, v_r[2] - G * F]
    k0 = c.k
    if k is not None:
        c.k = k
    q_c, qd_c, qdd_c = c.catch_state(hit, v_hit)
    c.k = k0
    if acc is not None:
        qdd_c, qdd_r = qdd_c.copy(), qdd_r.copy()
        qdd_c[P], qdd_r[P] = np.asarray(acc[:3]) * 10, np.asarray(acc[3:]) * 10
    vac = Seg(0.0, V, (q_r, qd_r, qdd_r), (q_c, qd_c, qdd_c))
    dwl = Seg(0.0, D, (q_c, qd_c, qdd_c), (q_r, qd_r, qdd_r))
    d = pl.d
    ids = J.ids
    d.qpos[ids.ball_q[0]:ids.ball_q[0] + 3] = [10.0, 10.0, 10.0]
    spd = tau = 0.0
    jt = js = 0
    for seg, T in ((vac, V), (dwl, D)):
        for i in range(n + 1):
            q, qd, qdd = seg(T * i / n)
            rs = np.abs(qd) / scene.JOINT_VMAX
            if rs.max() > spd:
                spd, js = float(rs.max()), int(rs.argmax())
            d.qpos[ids.arm_q], d.qvel[:] = q, 0.0
            d.qvel[ids.arm_v] = qd
            d.qacc[:] = 0.0
            d.qacc[ids.arm_v] = qdd
            mj.mj_inverse(J.m, d)
            rt = np.abs(d.qfrc_inverse[ids.arm_v]) / scene.TORQUE_MAX
            if rt.max() > tau:
                tau, jt = float(rt.max()), int(rt.argmax())

    def az(t, h=2e-3):                                   # ball point's vertical acceleration (finite differences)
        t = min(max(t, h), D - h)
        p0, p1, p2 = (pl.ball_point(dwl(t + k * h)[0]) for k in (-1, 0, 1))
        return float((p2 - 2 * p1 + p0)[2] / h ** 2)

    acc_min = min(az(D * i / n) for i in range(int(0.9 * n)))
    return dict(speed=spd, torque=tau, acc_min=acc_min, acc_rel=az(D), F=F, D=D, vz=float(v_r[2]), js=js, jt=jt)


def best_cycle(J, pl, c, rel, s, V, D, d_out, iters=400):
    """Accelerations at the catch and release that keep the cycle furthest inside the limits (Nelder-Mead):
    cost = max(speed ratio, torque ratio), plus penalties if the palm pulls away from the held ball
    or does not pull away at the release."""
    def cost(a):
        r = cycle_check(J, pl, c, rel, s, V, D, d_out, acc=a, n=24)
        pen = max(0.0, -G - r["acc_min"]) / G + max(0.0, r["acc_rel"] + 1.2 * G) / G
        return max(r["speed"], r["torque"]) + 2 * pen
    a0 = np.r_[c.qdd_c[P], s * s * rel[2][P]] / 10
    res = minimize(cost, a0, method="Nelder-Mead", options=dict(maxiter=iters, xatol=1e-3, fatol=1e-4))
    return res.x, cycle_check(J, pl, c, rel, s, V, D, d_out, acc=res.x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catch", default="runs/catch5_v443.json")
    ap.add_argument("--d-out", type=float, default=0.12, help="catch point outward of the throw point (m)")
    a = ap.parse_args()
    x, cp, off = stage1_juggle.load(a.catch)
    J = juggle.Juggler(1, human=True)
    pl = Planner(J)
    th, br, q_rel = cem_throw.segments(x, 0.35, 0.2)
    q_cock = th(0)[0]
    _, R_rel = pl.fk(q_rel)
    c = Catcher(J, pl, cp, q_rel, q_cock, R_rel, np.array(off))
    rel = th(0.35 - 1e-9)
    J.ball_off = c.ball_off
    print(f"catch point {a.d_out} m outward; feasible: speed and torque ratio <= 1, ball held through the dwell,"
          " palm pulls away at release")
    print("  s  vz m/s apex m  F s |  V     D   | speed torque held-acc rel-acc | ok")
    for s in (1.0, 1.1, 1.2, 1.3):
        vz = float((pl.jac6(q_rel)[:3] @ (s * rel[1]))[2])
        F = 2 * vz / G
        for V in np.arange(0.16, 0.5 * F - 0.05, 0.04):
            D = F - 2 * V
            acc, r = best_cycle(J, pl, c, rel, s, V, D, a.d_out)
            ok = r["speed"] <= 1.0 and r["torque"] <= 1.0 and r["acc_min"] >= -G and r["acc_rel"] < -G
            print(f"{s:4.2f} {vz:5.2f} {vz * vz / (2 * G):5.2f} {F:5.2f} | {V:4.2f}  {D:4.2f} | {r['speed']:5.2f} "
                  f"{r['torque']:5.2f} {r['acc_min']:7.1f} {r['acc_rel']:7.1f} | {'yes' if ok else ''}", flush=True)


if __name__ == "__main__":
    main()


def free_search(J, pl, c, rel, s, V, d_out, iters=800):
    """As best_cycle, also choosing the catch height dz (m, at most 0 here: lower) and the catch speed factor k."""
    def unpack(z):
        return z[:6], -abs(z[6]), float(np.clip(z[7], 0.0, 1.0))
    def cost(z):
        a, dz, k = unpack(z)
        r = cycle_check(J, pl, c, rel, s, V, 0.0, d_out, acc=a, n=24, dz=dz, k=k)
        if r["D"] < 0.1:
            return 10.0
        pen = max(0.0, -G - r["acc_min"]) / G + max(0.0, r["acc_rel"] + 1.2 * G) / G
        return max(r["speed"], r["torque"]) + 2 * pen
    z0 = np.r_[c.qdd_c[P], s * s * rel[2][P]] / 10
    z0 = np.r_[z0, 0.05, c.k]
    res = minimize(cost, z0, method="Nelder-Mead", options=dict(maxiter=iters, xatol=1e-3, fatol=1e-4))
    a, dz, k = unpack(res.x)
    return dz, k, cycle_check(J, pl, c, rel, s, V, 0.0, d_out, acc=a, dz=dz, k=k)


def release_state(J, pl, q_rel, dq, vz, w_r, a_r):
    """A release from pose q_rel + dq (planar joints) with the ball point moving straight up at vz and the
    palm pitching at w_r, accelerations a_r (planar joints, rad/s^2 / 10)."""
    q = q_rel.copy()
    q[P] += dq
    Jm = pl.jac6(q)
    axis = Jm[3:, 1] / np.linalg.norm(Jm[3:, 1])
    A = np.vstack([Jm[0, P], Jm[2, P], axis @ Jm[3:, P]])
    qd = np.zeros(6)
    qd[P] = np.linalg.solve(A, [0.0, vz, w_r])
    qdd = np.zeros(6)
    qdd[P] = np.asarray(a_r) * 10
    return q, qd, qdd


def free_search2(J, pl, c, q_rel, vz, V, d_out, z0=None, iters=3000):
    """As free_search, also choosing the release: pose (3 planar joints) and palm pitch rate, for a ball
    leaving straight up at vz. z = [a_catch(3), a_rel(3), dz, k, dq(3), w_r]."""
    def unpack(z):
        return z[:3], z[3:6], -abs(z[6]), float(np.clip(z[7], 0.0, 1.0)), np.clip(z[8:11], -0.4, 0.4), z[11]
    def run(z, n=24):
        ac, ar, dz, k, dq, w = unpack(z)
        rel = release_state(J, pl, q_rel, dq, vz, w, ar)
        return cycle_check(J, pl, c, rel, 1.0, V, 0.0, d_out, acc=np.r_[ac, ar], n=n, dz=dz or -1e-6, k=k)
    def cost(z):
        r = run(z)
        if r["D"] < 0.1:
            return 10.0
        pen = max(0.0, -G - r["acc_min"]) / G + max(0.0, r["acc_rel"] + 1.2 * G) / G
        return max(r["speed"], r["torque"]) + 2 * pen
    if z0 is None:
        z0 = np.r_[c.qdd_c[P] / 10, 0.0, 3.3, 0.0, 0.05, c.k, 0.0, 0.0, 0.0, 0.0]
    res = minimize(cost, z0, method="Nelder-Mead", options=dict(maxiter=iters, maxfev=iters, xatol=1e-4, fatol=1e-5))
    return res.x, run(res.x, n=60)


def cem_search(J, pl, c, q_rel, vz, V, d_out, z0, sig0, pop=64, elite=8, gens=60, seed=0):
    """CEM over z (as free_search2): Nelder-Mead stalls in 12-D."""
    rng = np.random.default_rng(seed)

    def run(z, n=24):
        ac, ar, dz, k, dq, w = z[:3], z[3:6], -abs(z[6]), float(np.clip(z[7], 0.0, 1.0)), np.clip(z[8:11], -0.4, 0.4), z[11]
        rel = release_state(J, pl, q_rel, dq, vz, w, ar)
        return cycle_check(J, pl, c, rel, 1.0, V, 0.0, d_out, acc=np.r_[ac, ar], n=n, dz=dz or -1e-6, k=k)

    def cost(z):
        r = run(z)
        if r["D"] < 0.1:
            return 10.0
        pen = max(0.0, -G - r["acc_min"]) / G + max(0.0, r["acc_rel"] + 1.2 * G) / G
        return max(r["speed"], r["torque"]) + 2 * pen

    mu, sig = np.array(z0, float), np.array(sig0, float)
    best = (cost(mu), mu)
    for g in range(gens):
        Z = mu + sig * rng.standard_normal((pop, len(mu)))
        C = np.array([cost(z) for z in Z])
        idx = np.argsort(C)[:elite]
        if C[idx[0]] < best[0]:
            best = (C[idx[0]], Z[idx[0]])
        mu, sig = Z[idx].mean(0), 0.7 * sig + 0.3 * Z[idx].std(0)
    return best[1], run(best[1], n=60)
