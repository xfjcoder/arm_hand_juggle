"""Stage 2b in-hand feedback: fit the throw-error predictor.

Runs the two-arm loop without correction on training seeds (70000+, separate from the 50000+ / 90000+ test
seeds), logging per throw the features available `lead` s before the release (the skin's ball position in the
palm frame, mean of the last 3 readings; where and when the ball was caught relative to the nominal catch)
and the landing error (front-back, sideways, arrival time; right arm's half). Fits a ridge linear model,
reports held-out R^2, writes runs/inhand2b_v1.json for stage2b_juggle.juggle2b(correct=...).

    PYTHONPATH=.:../robolab python3 stage2b_inhand_fit.py --n 120
"""
import argparse
import json
import os

import numpy as np

import stage2b_juggle
from robolab import evalpool

_W = {}
NAMES = ["off_x", "off_y", "off_z", "catch_dx", "catch_dy", "catch_dt"]


def collect(seed, kick, preset, lead, correct=None):
    if not _W:
        JR, JL = stage2b_juggle.make(2)
        _W.update(JR=JR, JL=JL, cy=stage2b_juggle.load_cycle(JR))
    D = []
    stage2b_juggle.juggle2b(_W["JR"], _W["JL"], _W["cy"], throws=30, seed=seed, kick=kick, preset=preset,
                            lead=lead, data=D, correct=correct)
    return dict(rows=D)


def fit(X, Y, lam=1e-6):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    A = np.linalg.solve(Z.T @ Z + lam * len(Z) * np.eye(Z.shape[1]), Z.T @ (Y - Y.mean(0)))
    return (A / sd[:, None]).T, Y.mean(0) - (A / sd[:, None]).T @ mu       # y = A_raw x + b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=70000)
    ap.add_argument("--preset", default="ideal")
    ap.add_argument("--lead", type=float, default=0.25)
    ap.add_argument("--out", default="runs/inhand2b_v1.json")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.n))
    R = evalpool.evaluate(collect, seeds, procs=min(12, os.cpu_count()),
                          kwargs=dict(kick=0.035, preset=a.preset, lead=a.lead))
    rows = [x for r in R for x in r["rows"]]
    X, Y = np.array([r["f"] for r in rows]), np.array([r["y"] for r in rows])
    n = len(X)
    k = int(0.8 * n)
    A, b = fit(X[:k], Y[:k])
    res = Y[k:] - (X[k:] @ A.T + b)
    print(f"{n} throws ({a.preset} sensing, lead {a.lead} s); held-out R^2 and residual std:")
    for i, nm in enumerate(["land x", "land y", "arrival t"]):
        r2 = 1 - res[:, i].var() / Y[k:, i].var()
        print(f"  {nm:9s}: std {Y[k:, i].std():.4f} -> {res[:, i].std():.4f}  R^2 {r2:.2f}  "
              f"coef {dict(zip(NAMES, np.round(A[i], 2)))}")
    A, b = fit(X, Y)
    json.dump(dict(A=A.tolist(), b=b.tolist(), lead=a.lead, preset=a.preset, features=NAMES, n=n), open(a.out, "w"))


if __name__ == "__main__":
    main()
