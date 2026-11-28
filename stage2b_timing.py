"""Stage 2b gate: can two arms keep a two-ball / three-ball rhythm within the UR10e limits?

The left arm is the exact mirror of the right one (kin.mirror: same joint speeds, torques mirrored to 1e-15),
so only the right arm's cycle is checked, in the one-arm model (whose dynamics are the right arm's; its base
sits 0.35 m to the right of the two-arm robot's centre plane, which is y = +0.35 here).

The hand cycle is periodic, two quintic joint segments as in the one-hand gate (stage2_timing.py):
    vacant  release state -> catch state, V s
    dwell   catch state -> release state, D s
Catch at the release height (dz = 0, as Stage 1 with sensing), so every ball flies F = 2 vz / g.

Patterns (Shannon: (F + D) hands = (V + D) balls, hand period Tc = V + D):
    exchange   2 balls, both hands throw at the same time across (F = V), each catching where it throws.
               Mirror-symmetric throws meet at the centre, so the left hand throws `delta` s after the right:
               right hand V = F + delta, left V = F - delta (its mirror), same Tc. Result: the arm limits are
               fine, but the balls still collide (two mirror-image paths shifted in time meet at the point
               halfway between their apex times), so this pattern needs the hands offset front to back.
    cascade    3 balls, throws alternating every Tc / 2, thrown from the inside (the Stage 1 spot) and caught
               d_y further out: D = 2 F - 3 V. Two balls on the same cycle (one throw in three empty) is the
               standard stepping stone, so the same check covers both.
Each candidate is tuned with CEM over 11 numbers: catch and release accelerations of the planar joints (3 + 3),
the catch speed factor k, the release pose (3 planar joints) and the palm pitch rate at release. Feasible:
speed and torque ratio <= 1, the held ball never left behind (ball-point vertical acceleration >= -g in the
dwell), the palm pulling away at release (< -1.2 g). Also the closest approach between balls in flight.

    PYTHONPATH=.:../robolab python3 stage2b_timing.py      # ~15 min on 12 processes
    PYTHONPATH=.:../robolab python3 stage2b_timing.py --wide 0.15 0.20    # add wider cascade spacings
"""
import json

import numpy as np

import cem_throw
import juggle
import stage1_juggle
import stage2_timing as st
from cem_catch import Catcher
from juggle import G, Seg
from planner import Planner

CENTRE_Y = 0.35            # the two-arm robot's mirror plane, in the one-arm (right arm) frame
NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]


def mirror_p(p, centre=CENTRE_Y):
    return np.r_[p[0], 2 * centre - p[1], p[2]]


LIM = dict(v=1.0, t=1.0, clip=0.4)    # what-ifs (--four2): joint speed and torque limits x v, x t; release pose range


def unpack(z):
    return z[:3], z[3:6], float(np.clip(z[6], 0.0, 1.0)), np.clip(z[7:10], -LIM["clip"], LIM["clip"]), z[10]


def states(J, pl, c, q_rel, z, vz, V, D, c_off, centre=CENTRE_Y, vb=(0.0, 0.0, 0.0)):
    """The right hand's nominal cycle for CEM solution z: release and catch states plus the geometry. It
    throws from the release state (ball across to the left hand's catch spot, the mirror of ours, in the plane
    y = centre) and catches the left hand's ball (thrown from the mirror of our release spot) at our release
    spot + c_off. vb: a bias added to the commanded release velocity of the ball point (tuned in full
    simulation, stage2b_cem.py: the ball rolls in the palm during the throw)."""
    ac, ar, k, dq, w = unpack(z)
    q_r, qd_r, qdd_r = st.release_state(J, pl, q_rel, dq, vz, w, ar)
    J.ball_off = c.ball_off
    Jm = pl.jac6(q_r)
    F = 2 * vz / G
    p_r = pl.ball_point(q_r)
    hit = p_r + np.asarray(c_off, float)
    v_out = (mirror_p(hit, centre) - p_r)[:2] / F            # across to the left hand's catch spot
    v_in = (hit - mirror_p(p_r, centre))[:2] / F             # the left hand's throw, arriving here
    qd_r = qd_r + np.linalg.solve(Jm, np.r_[v_out + np.asarray(vb[:2]), vb[2], 0, 0, 0])
    v_hit = np.r_[v_in, vz - G * F]
    k0, c.k = c.k, k
    q_c, qd_c, _ = c.catch_state(hit, v_hit)
    c.k = k0
    qdd_c = np.zeros(6)
    qdd_c[st.P] = np.asarray(ac) * 10
    return dict(rel=(q_r, qd_r, qdd_r), catch=(q_c, qd_c, qdd_c), V=V, D=D, F=F, k=k, p_rel=p_r, hit=hit,
                v_rel=Jm[:3] @ qd_r, v_hit=v_hit, v_out=v_out, v_in=v_in)


def cycle(J, pl, c, q_rel, z, vz, V, D, c_off, n=24):
    """One right-hand cycle (states()): the limit ratios, accelerations and the geometry."""
    s = states(J, pl, c, q_rel, z, vz, V, D, c_off)
    (q_r, qd_r, qdd_r), (q_c, qd_c, qdd_c) = s["rel"], s["catch"]
    F, p_r, hit, v_out, v_in = s["F"], s["p_rel"], s["hit"], s["v_out"], s["v_in"]
    vac = Seg(0.0, V, (q_r, qd_r, qdd_r), (q_c, qd_c, qdd_c))
    dwl = Seg(0.0, D, (q_c, qd_c, qdd_c), (q_r, qd_r, qdd_r))
    spd, tau, js, jt = limits(J, pl, ((vac, V), (dwl, D)), n)

    def az(t, h=2e-3):
        t = min(max(t, h), D - h)
        p0, p1, p2 = (pl.ball_point(dwl(t + i * h)[0]) for i in (-1, 0, 1))
        return float((p2 - 2 * p1 + p0)[2] / h ** 2)

    acc_min = min(az(D * i / n) for i in range(int(0.9 * n)))
    return dict(speed=spd, torque=tau, js=js, jt=jt, acc_min=acc_min, acc_rel=az(D), F=F, V=V, D=D,
                p_rel=p_r.tolist(), hit=hit.tolist(), v_out=v_out.tolist(), v_in=v_in.tolist())


def limits(J, pl, segs, n):
    """Peak joint speed and torque ratios (inverse dynamics with the hand, no ball) along the segments."""
    import mujoco as mj
    import scene
    d, ids = pl.d, J.ids
    d.qpos[ids.ball_q[0]:ids.ball_q[0] + 3] = [10.0, 10.0, 10.0]
    spd = tau = 0.0
    js = jt = 0
    for seg, T in segs:
        for i in range(n + 1):
            q, qd, qdd = seg(T * i / n)
            rs = np.abs(qd) / (LIM["v"] * scene.JOINT_VMAX)
            if rs.max() > spd:
                spd, js = float(rs.max()), int(rs.argmax())
            d.qpos[ids.arm_q], d.qvel[:], d.qacc[:] = q, 0.0, 0.0
            d.qvel[ids.arm_v], d.qacc[ids.arm_v] = qd, qdd
            mj.mj_inverse(J.m, d)
            rt = np.abs(d.qfrc_inverse[ids.arm_v]) / (LIM["t"] * scene.TORQUE_MAX)
            if rt.max() > tau:
                tau, jt = float(rt.max()), int(rt.argmax())
    return spd, tau, js, jt


def cost(r):
    pen = max(0.0, -G - r["acc_min"]) / G + max(0.0, r["acc_rel"] + 1.2 * G) / G
    return max(r["speed"], r["torque"]) + 2 * pen


def feasible(r):
    return r["speed"] <= 1 and r["torque"] <= 1 and r["acc_min"] >= -G and r["acc_rel"] < -1.2 * G


def setup():
    x, cp, off = stage1_juggle.load("runs/catch5_v443.json")
    J = juggle.Juggler(1, human=True)
    pl = Planner(J)
    th, br, q_rel = cem_throw.segments(x, 0.35, 0.2)
    _, R_rel = pl.fk(q_rel)
    c = Catcher(J, pl, cp, q_rel, th(0)[0], R_rel, np.array(off))
    J.ball_off = c.ball_off
    q, qd, qdd = th(0.35 - 1e-9)
    Jm = pl.jac6(q)
    axis = Jm[3:, 1] / np.linalg.norm(Jm[3:, 1])
    z0 = np.r_[c.qdd_c[st.P] / 10, qdd[st.P] / 10, c.k, 0, 0, 0, float(axis @ (Jm[3:] @ qd))]
    return J, pl, c, q_rel, z0


def search(job, pop=64, elite=8, gens=60):
    LIM.update(v=job.get("vscale", 1.0), t=job.get("tscale", 1.0), clip=job.get("clip", 0.4))
    J, pl, c, q_rel, z0 = setup()
    rng = np.random.default_rng(job["seed"])
    run = lambda z, n=24: cycle(J, pl, c, q_rel, z, job["vz"], job["V"], job["D"], job["c_off"], n)
    mu, sig = z0.copy(), np.r_[np.full(6, 1.0), 0.1, 0.1, 0.1, 0.1, 1.0]
    best = (cost(run(mu)), mu)
    for _ in range(gens):
        Z = mu + sig * rng.standard_normal((pop, len(mu)))
        C = np.array([cost(run(z)) for z in Z])
        idx = np.argsort(C)[:elite]
        if C[idx[0]] < best[0]:
            best = (C[idx[0]], Z[idx[0]])
        mu, sig = Z[idx].mean(0), 0.7 * sig + 0.3 * Z[idx].std(0)
    r = run(best[1], n=60)
    return dict(job, z=best[1].tolist(), ok=bool(feasible(r)), **r)


def min_gap(throws, F, t_end):
    """Closest centre distance between balls in flight. throws: (t, p0, v0 horizontal) per throw, vz = g F / 2."""
    vz = G * F / 2
    best = np.inf
    for t in np.arange(0.0, t_end, 0.002):
        ps = []
        for t0, p0, vh in throws:
            s = t - t0
            if 0.05 < s < F - 0.05:                              # in the air, clear of the hands
                ps.append(np.r_[p0[:2] + vh * s, p0[2] + vz * s - 0.5 * G * s * s])
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                best = min(best, float(np.linalg.norm(ps[i] - ps[j])))
    return best


def jobs():
    out = []
    for vz in (4.43, 5.0):
        F = 2 * vz / G
        for dV in (-0.2, -0.1, 0.0, 0.1, 0.2):
            for D in (0.4, 0.6, 0.8, 1.0):
                out.append(dict(pattern="exchange", vz=vz, V=round(F + dV, 3), dV=dV, D=D, c_off=[0, 0, 0]))
    for vz in (4.43, 5.0, 5.5):
        F = 2 * vz / G
        for d_y in (0.05, 0.10):
            for i in range(5):
                V = round(F * (0.35 + 0.04 * i), 3)                # D = 2F - 3V from 0.95 F down to 0.47 F
                out.append(dict(pattern="cascade", vz=vz, V=V, D=round(2 * F - 3 * V, 3), d_y=d_y,
                                c_off=[0, -d_y, 0]))
    for i, j in enumerate(out):
        j["seed"] = i
    return out


def wide_jobs(d_ys, vz=4.43):
    """Cascade candidates with the catch spot further out (wider throw-catch spacing, more room between the
    rising and the falling ball where their paths cross)."""
    F = 2 * vz / G
    out = []
    for d_y in d_ys:
        for i in range(5):
            V = round(F * (0.35 + 0.04 * i), 3)
            out.append(dict(pattern="cascade", vz=vz, V=V, D=round(2 * F - 3 * V, 3), d_y=d_y, c_off=[0, -d_y, 0],
                            seed=1000 + int(round(100 * d_y)) * 10 + i))
    return out


def four_jobs():
    """Four balls on two hands: Shannon gives F = 2 V + D (any two-hand pattern, crossing or not). The arm limit
    check of one hand's cycle at the higher throw speeds; geometry as the 10 cm cascade (only the limits are
    looked at here, the pattern itself is a later choice)."""
    out = []
    for vz in (5.0, 5.5, 6.0, 6.5):
        F = 2 * vz / G
        for V in (0.30, 0.35, 0.40, 0.45):
            D = round(F - 2 * V, 3)
            if D >= 0.30:
                out.append(dict(pattern="four", vz=vz, V=V, D=D, d_y=0.10, c_off=[0, -0.10, 0],
                                seed=2000 + len(out)))
    return out


def four2_jobs():
    """Four-ball what-ifs: a faster arm (speed and torque limits x 1.25, x 1.5), a catch spot 0.20 m to the side
    (Stage 2a: brings in shoulder_pan), a wider release-pose range (+-0.8 rad)."""
    out = []
    for vs, dy, clip in ((1.0, 0.20, 0.4), (1.0, 0.10, 0.8), (1.0, 0.20, 0.8), (1.25, 0.10, 0.4), (1.5, 0.10, 0.4)):
        for vz in (4.5, 5.0, 5.5, 6.0, 6.5):
            F = 2 * vz / G
            for V in (0.30, 0.35, 0.40):
                D = round(F - 2 * V, 3)
                if D >= 0.25:
                    out.append(dict(pattern="four", vz=vz, V=V, D=D, d_y=dy, c_off=[0, -dy, 0], vscale=vs,
                                    tscale=vs, clip=clip, seed=3000 + len(out)))
    return out


def main():
    import argparse
    from robolab import evalpool
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", type=float, nargs="*", help="only these extra cascade spacings d_y (m), added to the log")
    ap.add_argument("--four", action="store_true", help="four-ball cycles (F = 2 V + D), added to the log")
    ap.add_argument("--four2", action="store_true", help="four-ball what-ifs (four2_jobs), logs/stage4_gate2.*")
    a = ap.parse_args()
    if a.four2:
        R = evalpool.evaluate(search, four2_jobs(), procs=14)
        json.dump(R, open("logs/stage4_gate2.json", "w"))
        lines = ["four balls, what-ifs: limits x scale, catch d_y, release pose range; ratios to the scaled limits",
                 " scale d_y  pose |  vz    F  |  V     D  | speed      torque     | ok"]
        for r in sorted(R, key=lambda r: (r["vscale"], r["d_y"], r["clip"], r["vz"], r["V"])):
            lines.append(f"  {r['vscale']:.2f} {r['d_y']:.2f} {r['clip']:.1f} | {r['vz']:.2f} {r['F']:.2f} | {r['V']:.2f} "
                         f"{r['D']:.2f} | {r['speed']:.2f} {NAMES[r['js']][:8]:8s} {r['torque']:.2f} {NAMES[r['jt']][:8]:8s} "
                         f"| {'yes' if r['ok'] else ''}")
        print("\n".join(lines))
        open("logs/stage4_gate2.txt", "w").write("\n".join(lines) + "\n")
        return
    if a.four:
        R = [r for r in json.load(open("logs/stage2b_timing.json")) if r["pattern"] != "four"]
        R += evalpool.evaluate(search, four_jobs(), procs=12)
        json.dump(R, open("logs/stage2b_timing.json", "w"))
        lines = ["four balls, two hands: F = 2 V + D",
                 "  vz    F    apex |  V     D    Tc  | speed      torque     held  rel  | ok"]
        for r in [r for r in R if r["pattern"] == "four"]:
            lines.append(f"  {r['vz']:.2f} {r['F']:.2f} {r['vz'] ** 2 / (2 * G):.2f} | {r['V']:.2f} {r['D']:.2f} "
                         f"{r['V'] + r['D']:.2f} | {r['speed']:.2f} {NAMES[r['js']][:8]:8s} {r['torque']:.2f} "
                         f"{NAMES[r['jt']][:8]:8s} {r['acc_min']:5.1f} {r['acc_rel']:5.1f} | {'yes' if r['ok'] else ''}")
        print("\n".join(lines))
        open("logs/stage4_timing.txt", "w").write("\n".join(lines) + "\n")
        return
    if a.wide:
        R = json.load(open("logs/stage2b_timing.json"))
        R = [r for r in R if not (r["pattern"] == "cascade" and any(abs(r["d_y"] - d) < 1e-6 for d in a.wide))]
        R += evalpool.evaluate(search, wide_jobs(a.wide), procs=12)
    else:
        R = evalpool.evaluate(search, jobs(), procs=12)
    json.dump(R, open("logs/stage2b_timing.json", "w"))
    report(R)


def report(R):
    lines = []
    p = lambda s: (print(s), lines.append(s))
    p("exchange, 2 balls (F = V +- delta; the right hand takes V = F + delta, the left V = F - delta):")
    p("  vz    F   |  V - F   D   | speed      torque     held  rel  | ok")
    for r in [r for r in R if r["pattern"] == "exchange"]:
        p(f"  {r['vz']:.2f} {r['F']:.2f} | {r['dV']:+.1f}  {r['D']:.1f} | {r['speed']:.2f} {NAMES[r['js']][:8]:8s} "
          f"{r['torque']:.2f} {NAMES[r['jt']][:8]:8s} {r['acc_min']:5.1f} {r['acc_rel']:5.1f} | {'yes' if r['ok'] else ''}")
    p("pairs (right V = F + delta, left V = F - delta, same hand period Tc), both feasible:")
    ex = {(r["vz"], r["dV"], r["D"]): r for r in R if r["pattern"] == "exchange"}
    for (vz, dV, D), r in sorted(ex.items()):
        if dV < 0:
            continue
        o = ex.get((vz, -dV, round(D + 2 * dV, 1)))
        if o is None or not (r["ok"] and o["ok"]):
            continue
        pr = np.array(r["p_rel"])
        gap = min_gap([(0.0, pr, np.array(r["v_out"])), (dV, mirror_p(pr), np.array(o["v_out"]) * [1, -1])],
                      r["F"], r["F"] + dV + 0.01)
        p(f"  vz {vz:.2f} delta {dV:.1f}: Tc {r['V'] + D:.2f}  right D {D:.1f} left D {o['D']:.1f}  "
          f"worst ratio {max(r['speed'], r['torque'], o['speed'], o['torque']):.2f}  balls pass {100 * gap:.0f} cm apart")
    p("cascade, 3 balls (and 2 balls with one throw in three empty): D = 2F - 3V")
    p("  vz    F   d_y |  V     D    Tc  | speed      torque     held  rel  | gap 3 balls | ok")
    for r in [r for r in R if r["pattern"] == "cascade"]:
        pr, Tc = np.array(r["p_rel"]), r["V"] + r["D"]
        vo = np.array(r["v_out"])
        throws = []
        for n in range(8):                                        # alternating hands every Tc / 2
            if n % 2 == 0:
                throws.append((n * Tc / 2, pr, vo))
            else:
                throws.append((n * Tc / 2, mirror_p(pr), vo * [1, -1]))
        gap = min_gap(throws, r["F"], 4 * Tc)
        p(f"  {r['vz']:.2f} {r['F']:.2f} {r['d_y']:.2f} | {r['V']:.2f} {r['D']:.2f} {Tc:.2f} | {r['speed']:.2f} "
          f"{NAMES[r['js']][:8]:8s} {r['torque']:.2f} {NAMES[r['jt']][:8]:8s} {r['acc_min']:5.1f} {r['acc_rel']:5.1f} | "
          f"{100 * gap:5.0f} cm    | {'yes' if r['ok'] else ''}")
    open("logs/stage2b_timing.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
