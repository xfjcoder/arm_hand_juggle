"""Stage 0 gate: how fast can a UR10e throw straight up with a Shadow Hand palm-up?

For many palm-up poses (IK from several seeds, self-collision free) this solves two small LPs:

  speed  max v_z  s.t. palm velocity = (0, 0, v_z) and |qdot_i| <= real UR10e joint speed limits
         "flat":  no palm rotation at all
         "flick": palm may pitch about its lateral axis (a wrist flick), every other rate zero
  accel  max a_z  from rest s.t. palm acceleration = (0, 0, a_z), no rotation, |tau_i| <= torque limits
         (gravity included), and the same straight downward for the catch / post-release brake.

A ball thrown at v_z rises v^2 / 2g and is in the air 2 v / g. Writes logs/stage0_speed.json.
"""
import json

import mujoco as mj
import numpy as np
from scipy.optimize import linprog

import kin
import scene

G = 9.81


def yaw(R, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ R


def lp_max(c, A_eq, b_eq, bounds):
    r = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return -r.fun if r.status == 0 else 0.0


def speeds(J):
    """Max vertical palm speed (flat, flick) under the joint speed limits."""
    bounds = list(zip(-scene.JOINT_VMAX, scene.JOINT_VMAX))
    flat = lp_max(J[2], J[[0, 1, 3, 4, 5]], np.zeros(5), bounds)
    return flat, J


def flick_speed(J, R):
    """As flat, but the palm may rotate about its own lateral (x) axis."""
    bounds = list(zip(-scene.JOINT_VMAX, scene.JOINT_VMAX))
    Jw = R.T @ J[3:]                                  # angular rates in the palm frame
    A = np.vstack([J[[0, 1]], Jw[[1, 2]]])            # no sideways motion, no roll / yaw of the palm
    return lp_max(J[2], A, np.zeros(4), bounds)


def accels(m, d, ids, J):
    """Max straight-up and straight-down palm acceleration from rest (m/s^2), torque-limited."""
    M = np.zeros((m.nv, m.nv))
    mj.mj_fullM(m, d, M)
    Ma = M[np.ix_(ids.arm_v, ids.arm_v)]
    g = d.qfrc_bias[ids.arm_v]                        # at rest: gravity only
    # variables qdd; tau = Ma qdd + g within +-TORQUE_MAX  ->  A_ub qdd <= b_ub
    A_ub = np.vstack([Ma, -Ma])
    b_ub = np.concatenate([scene.TORQUE_MAX - g, scene.TORQUE_MAX + g])
    out = []
    for sign in (1, -1):
        r = linprog(-sign * J[2], A_ub=A_ub, b_ub=b_ub, A_eq=J[[0, 1, 3, 4, 5]], b_eq=np.zeros(5),
                    bounds=[(None, None)] * 6, method="highs")
        out.append(-r.fun if r.status == 0 else 0.0)
    return out


def self_collision(m, d):
    for c in d.contact[:d.ncon]:
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        names = m.body(b1).name + m.body(b2).name
        if "ball" in names:
            continue
        return True
    return False


def main():
    m = scene.build(1)
    d = mj.MjData(m)
    ids = scene.Ids(m)
    d.qpos[ids.ball_q[0]:ids.ball_q[0] + 3] = [5, 5, 5]   # ball out of the way
    rng = np.random.default_rng(0)
    rows = []
    for x in np.arange(0.7, 1.41, 0.1):
        for y in np.arange(-0.5, 0.51, 0.25):
            for z in np.arange(-0.3, 0.51, 0.2):
                for a in (0.0, np.pi / 2, -np.pi / 2):
                    R = yaw(kin.PALM_UP, a)
                    for s in range(6):
                        q0 = rng.uniform(-np.pi, np.pi, 6)
                        q, e = kin.ik(m, d, ids, np.array([x, y, z]), R, q0=q0, iters=400)
                        if e > 1e-3:
                            continue
                        d.qpos[ids.ball_q[0]:ids.ball_q[0] + 3] = [5, 5, 5]
                        mj.mj_forward(m, d)
                        if self_collision(m, d) or d.site_xpos[m.site(kin.PALM_SITE).id][2] < -0.75:
                            continue
                        J = kin.palm_jac(m, d, ids)
                        flat, _ = speeds(J)
                        flick = flick_speed(J, R)
                        up, down = accels(m, d, ids, J)
                        rows.append(dict(pos=[round(x, 2), round(y, 2), round(z, 2)], yaw=round(a, 3),
                                         q=q.round(4).tolist(), flat=flat, flick=flick, a_up=up, a_down=down))
    rows.sort(key=lambda r: -r["flat"])
    json.dump(rows, open("logs/stage0_speed.json", "w"), indent=0)
    print(f"{len(rows)} palm-up poses (IK solved, no self-collision)")
    for k in ("flat", "flick", "a_up", "a_down"):
        v = np.array([r[k] for r in rows])
        print(f"  {k:6s} median {np.median(v):6.2f}  90th pct {np.percentile(v, 90):6.2f}  max {v.max():6.2f}")
    print("best poses by flat vertical speed:")
    for r in rows[:8]:
        h = r["flat"] ** 2 / (2 * G)
        print(f"  pos {r['pos']} yaw {r['yaw']:+.2f}  flat {r['flat']:.2f} m/s (apex {h:.2f} m, air "
              f"{2 * r['flat'] / G:.2f} s)  flick {r['flick']:.2f}  a_up {r['a_up']:.0f}  a_down {r['a_down']:.0f}")
    for need, label in ((2.0, "1 ball"), (2.5, "2 balls"), (3.4, "3 balls (low)"), (4.5, "3 balls (high)")):
        nf = sum(r["flat"] >= need for r in rows)
        nk = sum(r["flick"] >= need for r in rows)
        print(f"  poses reaching {need} m/s ({label}): flat {nf}, flick {nk}")


if __name__ == "__main__":
    main()


def stroke_rank(stroke=0.25, n=5, top=400):
    """Re-rank the best poses by the WORST flat speed / up-accel over a vertical stroke below them."""
    m = scene.build(1)
    d = mj.MjData(m)
    ids = scene.Ids(m)
    rows = json.load(open("logs/stage0_speed.json"))[:top]
    out = []
    for r in rows:
        R = yaw(kin.PALM_UP, r["yaw"])
        q = np.array(r["q"])
        fl, up = [], []
        for dz in np.linspace(0, stroke, n):
            q, e = kin.ik(m, d, ids, np.array(r["pos"]) - [0, 0, dz], R, q0=q)
            if e > 1e-3 or self_collision(m, d):
                fl = [0]
                break
            J = kin.palm_jac(m, d, ids)
            fl.append(speeds(J)[0])
            up.append(accels(m, d, ids, J)[0])
        if fl[0] > 0:
            out.append(dict(r, stroke_flat=min(fl), stroke_up=min(up)))
    out.sort(key=lambda r: -r["stroke_flat"])
    json.dump(out, open("logs/stage0_stroke.json", "w"), indent=0)
    for r in out[:10]:
        print(f"  pos {r['pos']} yaw {r['yaw']:+.2f}  worst flat over {stroke} m stroke {r['stroke_flat']:.2f} m/s, "
              f"worst a_up {r['stroke_up']:.1f}  q {np.round(r['q'], 2).tolist()}")
