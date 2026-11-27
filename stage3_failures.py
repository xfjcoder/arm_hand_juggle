"""Failure cases of the two-arm cascade: classify every failed run, and render GIFs of chosen ones.

A ball-ball collision is linked to a failure if it happens within 1 s before the run ends (a lost ball is
detected ~0.3 s after its catch, the collisions come ~0.2 s before the catch). Categories:
    start collision     collision before the first catch-and-rethrow (the start-up throws)
    collision -> lost   collision, then the ball is not held ("not held")
    collision -> reach  collision, then the hand cannot get to the ball in time ("approach too fast")
    reach               no collision, the hand cannot get to the ball in time

    PYTHONPATH=.:../robolab python3 stage3_failures.py classify --balls 3 --kick 0.05 --out logs/stage3_failures_b3.json
    PYTHONPATH=.:../robolab python3 stage3_failures.py gif --balls 3 --kick 0.05 --seed 50000 --gif media_full/fail.gif
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stage2b_juggle as S

_W = {}
NAMES = ("red", "cyan", "yellow")


def world(balls):
    if _W.get("balls") != balls:
        JR, JL = S.make(balls)
        _W.update(balls=balls, JR=JR, JL=JL, cy=S.load_cycle(JR))
    return _W["JR"], _W["JL"], _W["cy"]


def collision_watch(JR, n_balls, events):
    """on_step hook: append one event per ball-ball contact episode (first step of contact)."""
    m, d = JR.m, JR.d
    bodies = {JR.ids.balls[i]: i for i in range(n_balls)}
    touching = set()

    def on_step(t):
        now = set()
        for c in d.contact[:d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 in bodies and b2 in bodies:
                now.add(tuple(sorted((bodies[b1], bodies[b2]))))
        for pr in now - touching:
            (p0, v0), (p1, v1) = JR.ball(pr[0]), JR.ball(pr[1])
            events.append(dict(t=round(t, 3), balls=list(pr), z=round(float((p0[2] + p1[2]) / 2), 3),
                               vz=[round(float(v0[2]), 2), round(float(v1[2]), 2)],
                               v_rel=round(float(np.linalg.norm(v0 - v1)), 2)))
        touching.clear()
        touching.update(now)
    return on_step


def run(seed, balls, kick, sensors, throws=30, aim=False):
    JR, JL, cy = world(balls)
    events = []
    r = S.juggle2b(JR, JL, cy, n_balls=balls, throws=throws, seed=seed, kick=kick, preset=sensors,
                   on_step=collision_watch(JR, balls, events), aim=S.AIM if aim else None)
    rel = r.pop("releases")
    near = [e for e in events if r["why"] and r["t"] - 1.0 <= e["t"] <= r["t"]]
    if not r["why"]:
        cat = None
    elif near and r["catches"] == 0:
        cat = "start collision"
    elif near:
        cat = "collision -> lost" if r["why"].startswith("not held") else (
            "collision -> reach" if r["why"].startswith("approach") else "collision -> other")
    else:
        cat = "reach" if r["why"].startswith("approach") else "other"
    # landing error (x, y, time) of the last throw towards the failing hand
    side = r["why"][-2] if r["why"] and r["why"].endswith(")") else None
    last = [e for e in rel if side and e["hand"] != side]
    land = last[-1]["land_err"] + [last[-1]["t_err"]] if last else None
    return dict(seed=seed, catches=r["catches"], why=r["why"], t=r["t"], category=cat, collision=near[-1] if near else None,
                n_collisions=len(events), land_err=land)


def classify(a):
    from robolab import evalpool
    R = evalpool.evaluate(run, range(a.seed0, a.seed0 + a.n), procs=min(12, os.cpu_count()),
                          kwargs=dict(balls=a.balls, kick=a.kick, sensors=a.sensors, aim=a.aim))
    cats = {}
    for r in R:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    C = [r["collision"] for r in R if r["collision"]]
    summary = dict(balls=a.balls, kick=a.kick, sensors=a.sensors, aim=a.aim, n=a.n, seed0=a.seed0, categories=cats,
                   collision_z=np.percentile([c["z"] for c in C], [10, 50, 90]).round(3).tolist() if C else None,
                   collision_v_rel=np.percentile([c["v_rel"] for c in C], [10, 50, 90]).round(2).tolist() if C else None,
                   one_rising_one_falling=sum((c["vz"][0] > 0) != (c["vz"][1] > 0) for c in C), n_linked=len(C))
    reach = [r["land_err"] for r in R if r["category"] == "reach" and r["land_err"]]
    if reach:
        e = np.array(reach)
        summary["reach_miss_xy_cm"] = np.percentile(np.linalg.norm(e[:, :2], axis=1) * 100, [10, 50, 90]).round(1).tolist()
        summary["reach_miss_t_ms"] = np.percentile(np.abs(e[:, 2]) * 1000, [10, 50, 90]).round(0).tolist()
    json.dump(dict(summary=summary, runs=R), open(a.out, "w"), indent=1)
    print(json.dumps(summary))


def caption(img, lines, colour=(255, 255, 255), band=96):
    """Text in a black band above the views, clear of the balls at the top of their flight."""
    im = Image.new("RGB", (img.shape[1], img.shape[0] + band))
    im.paste(Image.fromarray(img), (0, band))
    dr = ImageDraw.Draw(im)
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(path, 22) if os.path.exists(path) else ImageFont.load_default()
    y = 6
    for txt, col in lines:
        dr.text((12, y), txt, fill=col or colour, font=font)
        y += 29
    return np.asarray(im)


def gif(a):
    JR, JL, cy = world(a.balls)
    # pass 1: when it fails and when the balls collide (the run is deterministic for a seed)
    r = run(a.seed, a.balls, a.kick, a.sensors)
    t_end = r["t"]
    t_c = r["collision"]["t"] if r["collision"] else None
    t0 = max(0.0, (t_c if t_c is not None else t_end) - a.before)
    # x8 slow around the collision, or (out of reach, no collision) over the miss: the planner gives up
    # ~0.3 s before the ball comes down
    slow = (t_c - 0.12, t_c + 0.3) if t_c is not None else (t_end - 0.05, t_end + 0.45)
    print(r)
    hist = [[] for _ in range(a.balls)]

    def trails():                       # each ball's last 0.3 s, fading
        out = []
        for i, H in enumerate(hist):
            for j, p in enumerate(H[-150::5]):
                fade = (j + 1) / len(H[-150::5])
                out.append((p, 0.012, np.r_[S.BALL_RGBA[i][:3], 0.6 * fade]))
        return out
    render, close = S.renderer(JR, a.balls, extra=trails)
    frames, clock = [], {}

    def on_step(t):
        clock["t"] = t
        for i in range(a.balls):
            hist[i].append(JR.ball(i)[0])
            del hist[i][:-150]
    watch = collision_watch(JR, a.balls, [])

    def rend():
        # 4 ms per frame inside the slow window (x8 slower at 30 fps), 32 ms outside (about real time)
        t = clock.get("t", 0.0)
        k = round(t / 0.004)
        if t < t0 or (not (slow[0] <= t <= slow[1]) and k % 8):
            return None
        img = render()
        tag = "x8 slow" if slow[0] <= t <= slow[1] else "real time"
        lines = [(a.title + (f" ({NAMES[r['collision']['balls'][0]]} hits {NAMES[r['collision']['balls'][1]]})"
                            if t_c is not None else ""), None), (f"t = {t:5.2f} s   {tag}", (200, 200, 200))]
        if t_c is not None and t_c - 0.004 <= t <= t_c + 0.06:
            lines.append(("COLLISION", (255, 80, 80)))
        if t >= t_end:
            side = {"L": "left", "R": "right"}.get(r["why"][-2], "")
            msg = (f"failed: the {side} hand cannot reach the ball in time" if r["why"].startswith("approach")
                   else f"failed: ball not held by the {side} hand" if r["why"].startswith("not held") else r["why"])
            lines.append((msg, (255, 80, 80)))
        return caption(img, lines)

    def both(t):
        on_step(t)
        watch(t)
    S.juggle2b(JR, JL, cy, n_balls=a.balls, throws=30, seed=a.seed, kick=a.kick, preset=a.sensors,
               frames=frames, render=rend, every=2, on_step=both, tail=a.tail)
    close()
    frames = [f for f in frames if f is not None]
    from robolab.media import save_gif
    save_gif(frames, a.gif, fps=30)
    print(a.gif, len(frames), "frames")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["classify", "gif"])
    ap.add_argument("--balls", type=int, default=3)
    ap.add_argument("--kick", type=float, default=0.05)
    ap.add_argument("--sensors", default="real", choices=["ideal", "real"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--out", default="logs/stage3_failures.json")
    ap.add_argument("--seed", type=int, default=50000)
    ap.add_argument("--gif")
    ap.add_argument("--aim", action="store_true", help="with the release-point offset (stage2b_juggle.AIM)")
    ap.add_argument("--title", default="")
    ap.add_argument("--before", type=float, default=1.6, help="seconds shown before the collision / failure")
    ap.add_argument("--tail", type=float, default=0.8, help="seconds shown after the failure")
    a = ap.parse_args()
    classify(a) if a.mode == "classify" else gif(a)


if __name__ == "__main__":
    main()
