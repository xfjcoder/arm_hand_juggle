"""In-hand feedback: keep a caught ball from rolling off the palm, and bring it to the throw's start spot.

After a catch the ball still carries some sideways speed and spin, and on a nearly flat palm nothing stops it:
most Stage 1 drops were balls rolling across the palm and over the thumb-side edge during the dip, 0.5 s
after a clean catch. Here the palm is tilted about its finger axis (palm z) against the roll, like balancing a
ball on a plate, from the catch until the next throw.

Sensing is modelled on what a real hand could have (palm tactile sensors or a wrist camera): the ball's
position in the palm frame at 100 Hz with 1 mm noise, and only while the ball touches the hand. The ball's
speed across the palm comes from a straight-line fit over the last 8 samples.

Control, across the palm only (palm x; along the fingers the curled fingers stop the ball):
    wanted ball acceleration  a = -(kp e + kd v),   e = x - x_spot
    a rolling ball accelerates at 5/7 of gravity's component along the palm, and a small palm rotation dth
    about palm z changes that component by g_y dth (g_y: gravity along the palm normal, in the palm frame), so
    dth = (7/5) a / g_y, clipped to +-max_tilt and smoothed (time constant tau)
applied as a rotation of the palm about palm z through the ball point, on top of the planned joint motion. The
tilt rate is capped at max_rate and further so that planned motion + correction stays within `vmax_frac` of
every UR10e joint speed limit; its change is capped at max_acc (rad/s^2) so the wrist, carrying the hand, can
track it without overshooting.
"""
import numpy as np

import scene

from juggle import G


class InHand:
    # defaults from a 24-seed comparison: the unfiltered version (tau 0.03, 5-sample fit, no acceleration cap)
    # caught as well but chased sensor noise at the rate limit and overshot wrist_3's speed limit (1.16x)
    def __init__(self, J, pl, x_spot, kp=40.0, kd=10.0, max_tilt=0.3, tau=0.08, rate=100.0, noise=0.001,
                 rng=None, max_rate=2.0, max_acc=60.0, vmax_frac=0.80, nfit=8):
        self.J, self.pl, self.x_spot = J, pl, x_spot
        self.kp, self.kd, self.max_tilt, self.tau = kp, kd, max_tilt, tau
        self.period, self.noise = 1.0 / rate, noise
        self.max_rate, self.max_acc, self.vmax_frac, self.nfit = max_rate, max_acc, vmax_frac, nfit
        self.rng = rng or np.random.default_rng()
        self.reset()

    def reset(self):
        self.th_acc = 0.0
        self.th, self.th_rate, self.ts, self.xs, self.zs, self.next_t = 0.0, 0.0, [], [], [], -1.0
        self.e, self.v, self.speed = None, None, None

    def sense(self, t, reading=None):
        """Sample the in-hand sensor (if due); returns True when a new estimate is available.

        reading: (capture time, touching, ball offset in the palm frame) from a sensing.Tactile skin; without
        it, the built-in ideal sensor (true position, 1 mm noise, no delay)."""
        if reading is not None:
            tc, touching, off = reading
            if tc <= self.next_t:                           # nothing new delivered
                return False
            self.next_t = tc
        else:
            if t < self.next_t:
                return False
            self.next_t, tc = t + self.period, t
            touching, off = self.J.ball_touching_hand(), None
        if not touching or (reading is not None and off is None):   # no contact, no in-hand reading
            self.ts, self.xs, self.zs = [], [], []
            self.e = self.v = self.speed = None
            return False
        if off is None:
            p, R, _, _ = self.J.palm()
            off = R.T @ (self.J.ball()[0] - p) + self.rng.normal(0, self.noise, 3)
        self.ts.append(tc); self.xs.append(off[0]); self.zs.append(off[2])
        n = self.nfit
        self.ts, self.xs, self.zs = self.ts[-n:], self.xs[-n:], self.zs[-n:]
        self.e = self.xs[-1] - self.x_spot
        if len(self.ts) >= 3:
            A = np.c_[np.ones(len(self.ts)), np.array(self.ts) - t]
            self.v = float(np.linalg.lstsq(A, np.array(self.xs), rcond=None)[0][1])
            vz = float(np.linalg.lstsq(A, np.array(self.zs), rcond=None)[0][1])
            self.speed = float(np.hypot(self.v, vz))
            self.e += self.v * (t - tc)                     # carry the reading over the sensor's delay
        return True

    def step(self, t, dt, q=None, qd=None, reading=None):
        """Update the tilt; with the planned (q, qd), the tilt rate is limited to keep joint speeds legal."""
        self.sense(t, reading)
        want = 0.0
        if self.e is not None and self.v is not None:
            R = self.J.palm()[1]
            g_y = (R.T @ np.array([0.0, 0.0, -G]))[1]
            if abs(g_y) > 1.0:
                a = -(self.kp * self.e + self.kd * self.v)
                want = float(np.clip(1.4 * a / g_y, -self.max_tilt, self.max_tilt))
        rate = (want - self.th) / max(self.tau, dt)
        lim = self.max_rate
        if q is not None:                                   # joint speed budget left by the planned motion
            k = self.gain(q)
            vmax = self.vmax_frac * scene.JOINT_VMAX
            for kk, qq, vm in zip(k, qd, vmax):
                if abs(kk) > 1e-9:
                    room = (vm - qq * np.sign(rate * kk)) / abs(kk)
                    lim = min(lim, max(room, 0.0))
        rate = float(np.clip(rate, -lim, lim))
        # and the rate changes smoothly: a jump in commanded rate makes the wrist (carrying the hand) overshoot
        rate = float(np.clip(rate, self.th_rate - self.max_acc * dt, self.th_rate + self.max_acc * dt))
        self.th_acc = (rate - self.th_rate) / dt
        self.th_rate, self.th = rate, self.th + rate * dt

    def settled(self, e_tol=0.004, v_tol=0.03, th_tol=0.02):
        return (self.e is not None and self.speed is not None and abs(self.e) < e_tol and self.speed < v_tol
                and abs(self.th) < th_tol)

    def gain(self, q):
        """Joint motion per radian of palm rotation about palm z, through the ball point."""
        Jm = self.pl.jac6(q)
        axis = self.pl.fk(q)[1][:, 2]
        return np.linalg.solve(Jm, np.r_[0.0, 0.0, 0.0, axis])

    def apply(self, q, qd, qdd):
        """The planned joint state plus the palm rotation th about palm z, through the ball point."""
        if self.th == 0.0 and self.th_rate == 0.0 and self.th_acc == 0.0:
            return q, qd, qdd
        k = self.gain(q)
        return q + k * self.th, qd + k * self.th_rate, qdd + k * self.th_acc      # feedforward: no lag, no overshoot
