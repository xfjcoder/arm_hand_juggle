"""Stage 2 gate, CEM version: the periodic two-ball one-hand cycle with catch and release both free
(accelerations, catch height and speed factor, release pose and palm pitch rate), per toss speed and empty
time V. Results in logs/stage2_timing_cem.json / .txt. ~3 min on 12 processes.

    PYTHONPATH=.:../robolab python3 stage2_timing_cem.py
"""
import numpy as np, juggle, stage1_juggle, cem_throw, stage2_timing as st, json
from cem_catch import Catcher
from planner import Planner
from robolab import evalpool
NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
import sys
D_OUT = [float(a) for a in sys.argv[1:3]] if len(sys.argv) > 2 else (float(sys.argv[1]) if len(sys.argv) > 1 else 0.10)
grid = [(vz, V) for vz in (4.5, 5.0) for V in (0.20, 0.22, 0.24, 0.26, 0.28, 0.30)]
def job2(i):
    vz, V = grid[i]
    x, cp, off = stage1_juggle.load("runs/catch5_v443.json")
    J = juggle.Juggler(1, human=True); pl = Planner(J)
    th, br, q_rel = cem_throw.segments(x, 0.35, 0.2)
    _, R_rel = pl.fk(q_rel)
    c = Catcher(J, pl, cp, q_rel, th(0)[0], R_rel, np.array(off)); J.ball_off = c.ball_off
    q, qd, qdd = th(0.35 - 1e-9)
    Jm = pl.jac6(q); axis = Jm[3:, 1] / np.linalg.norm(Jm[3:, 1]); w_r = float(axis @ (Jm[3:] @ qd))
    z0 = np.r_[c.qdd_c[st.P] / 10, qdd[st.P] / 10, 0.05, 0.75, 0, 0, 0, w_r]
    sig = np.r_[np.full(6, 1.0), 0.05, 0.1, 0.1, 0.1, 0.1, 1.0]
    z, r = st.cem_search(J, pl, c, q_rel, vz, V, D_OUT, z0, sig, seed=i)
    return dict(vz_set=vz, V=V, d_out=D_OUT, z=z.tolist(), w0=w_r, **{k: (float(v) if not isinstance(v, int) else v) for k, v in r.items()})
R = evalpool.evaluate(job2, range(len(grid)), procs=12)
json.dump(R, open(f"logs/stage2_timing_cem_d{str(D_OUT).replace(' ', '')}.json", "w"))
for r in sorted(R, key=lambda r: max(r["speed"], r["torque"])):
    z = r["z"]
    print(f"vz {r['vz']:.1f} V {r['V']:.2f} | F {r['F']:.2f} D {r['D']:.2f} | speed {r['speed']:.2f} ({NAMES[r['js']]}) "
          f"torque {r['torque']:.2f} ({NAMES[r['jt']]}) held {r['acc_min']:.1f} rel {r['acc_rel']:.1f} | dz {-abs(z[6]):+.3f} k {np.clip(z[7],0,1):.2f} dq {np.round(z[8:11],2)} w {z[11]:.1f} (tuned {r['w0']:.1f})")
