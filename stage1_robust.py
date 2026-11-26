"""Stage 1 check: continuous one-ball juggling (1 m tosses) over many randomised seeds.

Per seed: the ball dropped into the hand up to 1.5 cm off-centre, a random velocity kick N(0, 0.05 m/s) per
axis added to the ball at every release (throw-to-throw variation), 2 mm noise on the ball position the
catch planner sees. Up to 30 cycles; a cycle counts when the ball is caught and at rest in the hand before
the next throw. A run stops at the first drop, or as soon as a planned motion would exceed the UR10e joint
speed limits. Pass: median >= 20 catches in a row, all within the limits.

    PYTHONPATH=.:../robolab python3 stage1_robust.py --n 120
"""
import argparse
import os

import numpy as np

import juggle
import stage1_juggle
from robolab import evalpool


def episode(seed, catch="runs/catch3_v443.json", cycles=30, kick=0.05, noise=0.002):
    rng = np.random.default_rng(seed)
    x, cp, off = stage1_juggle.load(catch)
    J = juggle.Juggler(1, human=True)
    o = rng.uniform(-0.015, 0.015, 2)
    r = stage1_juggle.juggle_run(J, x, cp, off, cycles=cycles, seed=seed, noise=noise, kick=kick,
                                 ball_offset=(o[0], o[1], 0.06))
    return dict(catches=r["catches"], all30=int(r["catches"] >= cycles), why=r["why"], qd_ratio=r["qd_ratio"],
                within_limits=int(r["qd_ratio"] <= 1.0), tau_sat=r["tau_sat"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--catch", default="runs/catch3_v443.json")
    ap.add_argument("--kick", type=float, default=0.05)
    ap.add_argument("--out", default="logs/stage1_robust.jsonl")
    a = ap.parse_args()
    R = evalpool.evaluate(episode, range(a.seed0, a.seed0 + a.n), procs=min(12, os.cpu_count()),
                          kwargs=dict(catch=a.catch, kick=a.kick))
    evalpool.save_jsonl(R, a.out)
    c = np.array([r["catches"] for r in R])
    print(evalpool.summarise(R, rates=["all30", "within_limits"], means=["catches", "qd_ratio", "tau_sat"]))
    print(f"catches in a row: median {np.median(c):.0f}, min {c.min()}, quartiles {np.percentile(c, [25, 75]).tolist()}")
    whys = [r["why"] for r in R if r["why"]]
    print("why runs ended early:", {w: whys.count(w) for w in set(whys)})
