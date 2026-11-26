# arm_hand_juggle

A UR10e arm carrying a Shadow Hand (E3M5) in MuJoCo. The goal is continuous toss-and-catch, first with one ball,
then two, then three. Everything runs on a CPU, so the method is model-based: ballistic prediction, scripted palm
trajectories and computed-torque control, with tuning and learning added only where they are needed.

Models come from `mujoco_menagerie` (`universal_robots_ur10e`, `shadow_hand`) and are combined with MjSpec in
`scene.py`. The hand weighs 3.79 kg, which is more than a Panda's 3 kg payload, so the arm is a UR10e
(12.5 kg payload). Joint torques are capped at the real UR10e values (330/330/150/56/56/56 N m). The real joint
speed limits (120/120/180/180/180/180 deg/s) are not in the model, so they are checked on every run.

## Stage 0: feasibility gate (done)

![stage 0 toss](media/stage0_toss.gif)

**How fast can it throw?** `stage0_speed.py` solves small linear programs over 1977 palm-up poses that are
reachable without self-collision (`logs/stage0_speed.txt`, `logs/stage0_stroke.txt`). It reports the fastest
straight-up palm speed allowed by the joint speed limits, and the vertical acceleration allowed by the torque
limits.

| | median | best |
|---|---|---|
| straight-up palm speed (no palm rotation) | 1.60 m/s | 2.95 m/s |
| same, worst point over a 0.25 m throwing stroke | | 2.18 m/s |
| with a wrist flick allowed (palm may pitch) | 2.64 m/s | 7.48 m/s |
| upward acceleration from rest | 18 m/s^2 | 39 m/s^2 |

What this means for the plan, in one hand:
- **1 ball** (about 2 m/s, 0.2 m high) fits within the real limits.
- **2 balls** (about 2.5 m/s) is borderline with straight throws.
- **3 balls** (3.4 m/s or more) cannot be done with straight throws on this arm. It would need wrist flicks,
  a second arm and hand, or a faster-than-real arm.

**One toss and catch** (`juggle.py`, `stage0_toss.py`). The palm is driven by operational-space computed
torque. Each toss runs in phases:
1. Lower the palm 0.3 m.
2. Throw upward with a quintic trajectory to 2 m/s. The hand opens flat at 85% of the stroke.
3. Brake to a stop.
4. While the ball is in flight, re-plan the catch every 20 ms from the ball's measured position. The palm meets
   the ball moving at half the ball's speed.
5. Absorb the catch over 0.12 m, then hold.

**Robustness** (`stage0_robust.py`, 120 fresh seeds from 50000). Each seed starts the ball up to 1.5 cm
off-centre, scales the throw speed by a random factor between 0.9 and 1.1, and adds 2 mm noise to the ball
position measurement.

| metric | result |
|---|---|
| ball held after the catch | **120/120** (97-100%) |
| ...and resting back in the palm centre | 116/120 (92-99%); the other 4 rest 4-10 cm off-centre |
| within the real UR10e joint speed limits | 119/120 (peak 0.88 of the limit on average) |
| release speed / apex | 2.14 m/s / 0.24 m |

Bugs found and fixed on the way to these numbers:
- The throw segment ended with a velocity jump. The palm overshot, touched the ball just after release, and this
  was taken for the catch.
- A gentle brake let the palm chase the ball and bump it again.
- The ball dragged on the thumb and curled fingers at release, losing up to half its speed. Opening the hand
  late in the stroke fixed it.
- A ball rocking in the cup mid-stroke was taken for the release.

Comparisons are in `logs/stage0_release_grid.txt`. The 8 variants there land between 109 and 117 of 120, mostly
within noise of each other. The clear gain is that release speed matches the command once the hand opens.

## Next

- **Stage 1:** juggle one ball continuously. Pass means a median of 20 or more catches in a row over 120 seeds,
  within the joint limits. This needs re-centring after off-centre catches and a joined catch-throw cycle.
- **Stage 2:** two balls in one hand.
- **Stage 3:** three balls, decided using the speed numbers above.

```
PYTHONPATH=. MUJOCO_GL=egl python3 stage0_speed.py              # throw-speed feasibility (~35 s)
PYTHONPATH=. MUJOCO_GL=egl python3 stage0_toss.py --gif media_full/stage0_toss.gif
PYTHONPATH=. MUJOCO_GL=egl python3 stage0_robust.py --n 120     # needs robolab (pip install -e ~/robo/robolab)
```
