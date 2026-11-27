"""Stage 2b: catch reach of each cascade cycle in the gate. From the release state, the vacant segment is
re-planned to a catch point shifted by dx (front-back) or dy (sideways), arriving at the same time; reach is
the largest shift whose plan stays within `vmax_ratio` of the joint speed limits (the loop's rule).

    PYTHONPATH=.:../robolab python3 stage2b_reach.py
"""
import json

import numpy as np

import stage2b_cem
import stage2b_timing as gate
from juggle import Seg
from stage1_juggle import peak_ratio


def reach(J, pl, c, q_rel, r, vmax_ratio=0.97):
    cy = gate.states(J, pl, c, q_rel, np.array(r["z"]), r["vz"], r["V"], r["D"], r["c_off"])
    c.k = cy["k"]
    out = {}
    for name, axis, sign in (("+x", 0, 1), ("-x", 0, -1), ("+y", 1, 1), ("-y", 1, -1)):
        lo, hi = 0.0, 0.4
        for _ in range(12):
            mid = (lo + hi) / 2
            hit = cy["hit"].copy()
            hit[axis] += sign * mid
            ok = peak_ratio(Seg(0.0, r["V"], cy["rel"], c.catch_state(hit, cy["v_hit"]))) <= vmax_ratio
            lo, hi = (mid, hi) if ok else (lo, mid)
        out[name] = lo
    return out


def main():
    B = stage2b_cem._base()
    R = [r for r in json.load(open("logs/stage2b_timing.json")) if r["pattern"] == "cascade" and r["d_y"] == 0.10]
    print(" vz    F    V     D   ratio |  +x    -x    +y    -y  (m)")
    for r in R:
        e = reach(B["J"], B["pl"], B["c"], B["q_rel"], r)
        print(f"{r['vz']:.2f} {r['F']:.2f} {r['V']:.2f} {r['D']:.2f} {max(r['speed'], r['torque']):.2f} | "
              f"{e['+x']:.3f} {e['-x']:.3f} {e['+y']:.3f} {e['-y']:.3f}", flush=True)


if __name__ == "__main__":
    main()
