"""Sensor models: what the controller is allowed to know about the ball.

Two sensors, each with an "ideal" and a "real" preset (`PRESETS`):

Camera (ball tracker, in flight)
    3-D ball position at `rate` Hz, delivered `latency` s after capture, with `noise` (m, per axis). A frame is
    lost when the line of sight from the camera to the ball centre hits the arm or hand first (MuJoCo ray
    cast), or at random with probability `drop`. Frames carry their capture time, so the ballistic fit
    (planner.BallFit) extrapolates over the latency itself. The camera stands in front of the robot, about
    2.5 m away and 1.4 m up (`pos`), roughly where a tracking camera would be mounted.

Tactile (palm and finger skin)
    Whether the ball touches the hand (total normal force above `force`), and where its centre is in the
    palm frame, estimated from the contacts: each contact point plus one ball radius along its normal, force
    weighted, as a skin reporting contact location and surface normal would give. `rate` Hz, `latency` s,
    `noise` (m, per axis). The real Shadow Hand has tactile sensing only in the fingertips, so the "real"
    preset assumes an add-on palm skin (taxel arrays give a contact centroid to about 2 mm).

The "ideal" presets reproduce what Stage 1 was tuned and tested with: the tracker every simulation step with
no delay or occlusion, and the in-hand position at 100 Hz with 1 mm noise and no delay.
"""
from collections import deque

import mujoco as mj
import numpy as np

import scene

PRESETS = {
    "ideal": dict(camera=dict(rate=None, latency=0.0, occlusion=False, drop=0.0),
                  tactile=dict(rate=100.0, latency=0.0, noise=0.001, force=0.0)),
    "real": dict(camera=dict(rate=120.0, latency=0.020, occlusion=True, drop=0.02),
                 tactile=dict(rate=100.0, latency=0.010, noise=0.002, force=0.05)),
}


CAMERA_VIEW = dict(lookat=(1.0, 0.2, 0.4), distance=2.5, azimuth=160.0, elevation=-20.0)


def camera_pos(lookat=CAMERA_VIEW["lookat"], distance=CAMERA_VIEW["distance"], azimuth=CAMERA_VIEW["azimuth"],
               elevation=CAMERA_VIEW["elevation"]):
    """Position of a camera aimed at `lookat` (MuJoCo free-camera convention)."""
    a, e = np.radians(azimuth), np.radians(elevation)
    fwd = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return np.array(lookat) - distance * fwd


class Camera:
    def __init__(self, J, rng, noise=0.0, rate=None, latency=0.0, occlusion=False, drop=0.0, pos=None, ball=0):
        self.J, self.rng, self.noise = J, rng, noise
        self.period = None if rate is None else 1.0 / rate
        self.latency, self.occlusion, self.drop = latency, occlusion, drop
        self.pos = camera_pos() if pos is None else np.array(pos, float)
        self.ball = ball
        self.ball_geom = J.m.body_geomadr[J.ids.balls[ball]]
        self.reset()

    def reset(self):
        self.next_t, self.queue = -1.0, deque()
        self.frames = self.hidden = self.dropped = 0
        self.last, self.status = None, None          # newest delivered frame, fate of the newest capture

    def visible(self):
        p = self.J.ball(self.ball)[0]
        gid = np.zeros(1, np.int32)
        mj.mj_ray(self.J.m, self.J.d, self.pos, p - self.pos, None, 1, -1, gid)
        return gid[0] == self.ball_geom

    def step(self, t):
        """Call every simulation step; returns the (capture time, position) frames delivered by time t."""
        if self.period is None or t >= self.next_t:
            if self.period is not None:
                self.next_t = t + self.period if self.next_t < 0 else self.next_t + self.period
            self.frames += 1
            p = self.J.ball(self.ball)[0] + (self.rng.normal(0, self.noise, 3) if self.noise else 0)
            if self.occlusion and not self.visible():
                self.hidden += 1
                self.status = "hidden"
            elif self.drop and self.rng.random() < self.drop:
                self.dropped += 1
                self.status = "dropped"
            else:
                self.queue.append((t, p))
                self.status = "ok"
        out = []
        while self.queue and self.queue[0][0] + self.latency <= t + 1e-9:
            out.append(self.queue.popleft())
        if out:
            self.last = out[-1]
        return out


class Tactile:
    def __init__(self, J, rng, rate=100.0, latency=0.0, noise=0.001, force=0.0, ball=0):
        self.J, self.rng = J, rng
        self.period, self.latency, self.noise, self.force = 1.0 / rate, latency, noise, force
        self.ball = ball
        self.reset()

    def reset(self):
        self.next_t, self.queue, self.last = -1.0, deque(), None

    def measure(self):
        """(touching, ball centre in the palm frame or None) from the ball-hand contacts right now."""
        J, m, d = self.J, self.J.m, self.J.d
        bid = J.ids.balls[self.ball]
        f6 = np.zeros(6)
        pts, ws = [], []
        for i, c in enumerate(d.contact[:d.ncon]):
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == bid and b2 in J.hand_bodies:
                sign = -1.0                                   # the normal points from geom1 (ball) to the hand
            elif b2 == bid and b1 in J.hand_bodies:
                sign = 1.0
            else:
                continue
            mj.mj_contactForce(m, d, i, f6)
            n = c.frame[:3]
            pts.append(c.pos + sign * scene.BALL_RADIUS * n)
            ws.append(max(f6[0], 0.0))
        w = float(np.sum(ws)) if ws else 0.0
        if not pts or w <= self.force:
            return False, None
        centre = np.average(pts, axis=0, weights=ws) if w > 0 else np.mean(pts, axis=0)
        p, R, _, _ = J.palm()
        return True, R.T @ (centre - p)

    def step(self, t):
        """Call every simulation step; returns the newest reading (capture time, touching, offset) delivered
        by time t, or None if there is none yet."""
        if t >= self.next_t:
            self.next_t = t + self.period if self.next_t < 0 else self.next_t + self.period
            touching, off = self.measure()
            if off is not None and self.noise:
                off = off + self.rng.normal(0, self.noise, 3)
            self.queue.append((t, touching, off))
        while self.queue and self.queue[0][0] + self.latency <= t + 1e-9:
            self.last = self.queue.popleft()
        return self.last


class Sensors:
    """The camera and the tactile skin, from a preset name ("ideal" / "real") plus overrides."""

    def __init__(self, J, preset="ideal", seed=None, noise=0.0, camera=None, tactile=None):
        cfg = PRESETS[preset]
        rng = np.random.default_rng(None if seed is None else seed + 104729)
        self.camera = Camera(J, rng, noise=noise, **dict(cfg["camera"], **(camera or {})))
        self.tactile = Tactile(J, rng, **dict(cfg["tactile"], **(tactile or {})))
        self.preset = preset


def render_views(J, renderer, sensors, left):
    """Side by side: the outside view (image `left`) and the tracking camera's view, with what the
    controller gets drawn in: the newest delivered camera frame (green; it trails the ball by the latency) and
    a status line."""
    from PIL import Image, ImageDraw, ImageFont
    cam_, tac = sensors.camera, sensors.tactile
    v = mj.MjvCamera()
    v.lookat[:] = CAMERA_VIEW["lookat"]
    v.distance, v.azimuth, v.elevation = CAMERA_VIEW["distance"], CAMERA_VIEW["azimuth"], CAMERA_VIEW["elevation"]
    renderer.update_scene(J.d, v)
    scn = renderer.scene
    touching = tac.last is not None and tac.last[1]
    if cam_.last is not None and not touching and scn.ngeom < scn.maxgeom:
        mj.mjv_initGeom(scn.geoms[scn.ngeom], mj.mjtGeom.mjGEOM_SPHERE, np.r_[scene.BALL_RADIUS * 0.8, 0, 0],
                        cam_.last[1], np.eye(3).ravel(), np.array([0.1, 1.0, 0.2, 0.8], np.float32))
        scn.geoms[scn.ngeom].emission = 0.8                    # bright regardless of the lighting
        scn.ngeom += 1
    right = Image.fromarray(renderer.render())
    dr = ImageDraw.Draw(right)
    font = ImageFont.load_default(size=20)
    status = {"ok": "tracked", "hidden": "HIDDEN by the hand", "dropped": "frame dropped"}.get(cam_.status, "")
    dr.text((12, 10), "tracking camera  120 Hz, 20 ms delay", fill=(255, 255, 255), font=font)
    dr.text((12, 36), "green: latest fix the controller has", fill=(120, 255, 140), font=font)
    dr.text((12, 62), ("in hand: tactile skin" if touching else "in flight: ") + ("" if touching else status),
            fill=(255, 220, 120), font=font)
    return np.concatenate([left, np.asarray(right)], axis=1)
