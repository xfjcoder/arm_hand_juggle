"""Collect the ball states the catch really faces: the exact ball state (position, velocity, spin) at the apex
of every cycle of kicked juggling runs. The catch is then tuned on drops replayed from this bank (cem_catch
--bank), so that its test cases come from what the cycle produces, spin included, not from a guessed
distribution.

    PYTHONPATH=.:../robolab python3 collect_apex.py --catch runs/catch4_v443.json --n 48 --out data/apex_bank.json
"""
import argparse
import json
import os

import numpy as np

import juggle
import stage1_juggle
from robolab import evalpool


def episode(seed, catch, kick, noise, cycles=8):
    x, cp, off = stage1_juggle.load(catch)
    J = juggle.Juggler(1, human=True)
    q0, v0 = J.ids.ball_q[0], J.ids.ball_v[0]
    rec, tr = [], []
    orig = J.control_joint

    def cj(*a, **k):
        rec.append((J.d.qpos[q0:q0 + 3].tolist(), J.d.qvel[v0:v0 + 6].tolist()))
        return orig(*a, **k)
    J.control_joint = cj
    r = stage1_juggle.juggle_run(J, x, cp, off, cycles=cycles, seed=seed, noise=noise, kick=kick, trace=tr)
    states = []
    for c in range(r["catches"] + 1):
        idx = [i for i, q in enumerate(tr) if q["cycle"] == c and q["phase"] == "flight"]
        if idx:
            states.append(rec[min(idx, key=lambda i: abs(rec[i][1][2]))])
    return dict(states=states, catches=r["catches"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--catch", default="runs/catch4_v443.json")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--seed0", type=int, default=20000)     # tuning seeds, apart from the 50000+ test seeds
    ap.add_argument("--kick", type=float, default=0.05)
    ap.add_argument("--noise", type=float, default=0.002)
    ap.add_argument("--out", default="data/apex_bank.json")
    a = ap.parse_args()
    R = evalpool.evaluate(episode, range(a.seed0, a.seed0 + a.n), procs=min(12, os.cpu_count()),
                          args=(a.catch, a.kick, a.noise))
    bank = [s for r in R for s in r["states"]]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(catch=a.catch, kick=a.kick, states=bank), open(a.out, "w"))
    P = np.array([s[0] for s in bank]); W = np.array([s[1][3:] for s in bank])
    print(f"{len(bank)} apex states; position mean {P.mean(0).round(3)} sd {P.std(0).round(3)}; "
          f"spin mean {W.mean(0).round(1)} sd {W.std(0).round(1)}")
