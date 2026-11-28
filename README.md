# arm_hand_juggle

A UR10e arm carrying a Shadow Hand (E3M5) in MuJoCo. The goal is continuous toss-and-catch, first with one ball,
then two, then three. Everything runs on a CPU, so the method is model-based: ballistic prediction, scripted palm
trajectories and computed-torque control, with tuning and learning added only where they are needed.

Models come from `mujoco_menagerie` (`universal_robots_ur10e`, `shadow_hand`) and are combined with MjSpec in
`scene.py`. The hand weighs 3.79 kg, which is more than a Panda's 3 kg payload, so the arm is a UR10e
(12.5 kg payload). Joint torques are capped at the real UR10e values (330/330/150/56/56/56 N m). The real joint
speed limits (120/120/180/180/180/180 deg/s) are not in the model, so they are checked on every run.

## Status: closed

| stage | result |
|---|---|
| 0: feasibility | one hand: 1 ball fits the limits, 3 balls need a second arm (done) |
| 1: one ball, one hand | continuous juggling with realistic sensing (done) |
| 2a: two balls, one hand | stopped at the arm's limits |
| 2b: two balls, two arms | cascade cycle, realistic sensing (done) |
| 3: three balls, two arms | 30 catches in 120/120 runs with no kick or 0.02 m/s kicks; at 0.05 m/s 94/120 (held-out 92), 2 balls 98/120 (held-out 99) |
| 4: four balls | not built: within the UR10e limits only at 0.97-1.00 of them; an arm ~1.25x faster at the elbow and wrist would leave the 3-ball cycle's headroom |

Left open: the remaining 3-ball losses at 0.05 m/s (19 balls lost from the hand, 7 out of reach), four balls on
a faster arm, and the path to hardware (UR arms take position/velocity streaming, not torque).

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

## Stage 1: continuous one-ball juggling, human-style arm, 1 m tosses (passed)

![stage 1 juggling](media/stage1_juggle.gif)

Eight catches in a row with 1 m-high tosses, under the test disturbances: a random velocity kick on every throw
and 2 mm noise on the ball tracker. Joint speed peaks at 0.86 of the UR10e limits, with no torque saturation.

**Arm pose.** The upper arm hangs and the elbow is bent. The base sits on a 1.45 m column. The shoulder_lift,
elbow and wrist_1 joints have parallel axes, so they form a planar 3-joint chain whose speeds add at the palm,
like a human elbow and wrist flick. The same LP as in Stage 0 gives 6.1-6.4 m/s of palm speed in this pose
family, against 2.2-2.95 m/s for the level-palm layout (`logs/stage1_planar_speed.txt`). A 1 m apex needs
4.4 m/s.

**Throw** (`human.py`, `cem_throw.py`). Quintic joint segments: cock, release, brake.
- SLSQP on a rigid-ball model gives the starting plan. That model misses the ball rolling on the turning palm
  (a ~0.8 m/s forward fling).
- CEM then tunes 13 parameters, each candidate scored in the full simulation.
- The throw is scored from all three spots where the ball can rest in the hand
  (`runs/throw2_v443.json`). Worst release: (0.01, 0.04, 4.51) m/s.

**Catch** (`cem_catch.py`).
- Running the throw in reverse does not work as a catch: the palm's sideways motion slides the falling ball
  off. The catch is its own design:
  - Just before arrival, the palm goes to where the ball will be, moving down with it and sideways at the
    ball's horizontal speed.
  - A dip then takes it back to the cocked pose while the fingers curl.
- CEM tunes 8 parameters. The cost includes how far the ball comes to rest from the spot the throw starts
  from, so the cycle closes on itself.
- The ball is tracked from noisy positions with a least-squares parabola fit, never from the simulator's true
  velocity.

**What went wrong on the way:**
- The ball sometimes came to rest wedged between the thumb and first finger, and the throw from there flew
  0.2-0.4 m/s sideways. With the thumb curled along the palm edge as a wall, that resting spot is gone.
- The thrown ball spins at 20-30 rad/s. Catch tests with dropped, non-spinning balls passed while real cycles
  failed. The catch is now tuned on 178 real apex ball states recorded from juggling runs
  (`collect_apex.py`, `data/apex_bank.json`).
- With fewer than 20 ms of samples, the parabola fit reported zero velocity and planned an impossible motion.
  The planner now waits until the fit has enough samples.
- Even after a clean catch, three quarters of the remaining drops happened later: the ball kept its leftover
  sideways speed and spin, rolled across the nearly flat palm and went over the thumb-side edge about 0.5 s
  after the catch. The fix is in-hand feedback (below).
- The first version of that feedback chased sensor noise at its rate limit. It caught as well, but it
  overshot wrist_3's speed limit (1.16x). Filtering, a cap on tilt acceleration with feedforward, and a
  joint-speed budget fixed it.
- When the ball slipped off early in a throw, the controller switched to flight mode and asked the brake
  segment for a time before it had started. The arm's target jumped and the shoulder overshot to 1.48x its
  limit. The arm now finishes the throw, then brakes.

**In-hand feedback** (`inhand.py`). From the catch until the next throw, the palm tilts about its finger axis
against the ball's roll, like balancing a ball on a plate. The sensor is modelled on what a real hand could
have (palm tactile sensors or a wrist camera): the ball's position in the palm at 100 Hz with 1 mm noise, and
only while the ball touches the hand.
- Wanted ball acceleration: a = -(kp e + kd v), where e is the offset from the throw's start spot.
- A rolling ball accelerates at 5/7 of gravity's component along the palm, so the tilt is
  dth = (7/5) a / g_normal.
- The tilt is filtered and rate-limited to a joint-speed budget of 0.8 of the limits.
- The next throw starts only when the sensor reads the ball at rest in the pocket.

**Robustness** (`stage1_robust.py`). Runs are capped at 30 cycles.
- Disturbances: each throw gets a random velocity kick of N(0, 0.05 m/s) per axis, and the ball tracker has
  2 mm of position noise.
- Start: the ball is placed in the hand up to 1 cm from its resting spot.
- Joint limits: a planned motion above 0.97 of the joint speed limits counts as a drop, leaving room for
  tracking error. The limits are also checked on the measured motion.

| controller | seeds | median catches in a row | mean (95% CI) | reach 30/30 | within joint limits |
|---|---|---|---|---|---|
| catch4 (tuned on random spin-free drops) | 50000+ | 1 | 4.1 | 3/120 | 118/120 |
| catch5 (tuned on real apex states) | 50000+ | 4 | 7.4 | 6/120 | 120/120 |
| catch5 + in-hand feedback | 50000+ | **30** | 22.1 (20.2-24.0) | 68/120 | **120/120** |
| catch5 + in-hand feedback, fresh seeds | 90000+ | **30** | 21.4 (19.4-23.4) | 62/120 | **120/120** |

**Stage 1 passes** (median of 20 or more catches in a row, within the joint limits) on both the test seeds and
seeds never used before. The early-start drops in the catch5 row were mostly a test artifact: the ball was
dropped onto the hand up to 1.5 cm off-centre and sometimes not held.

Remaining drops on the fresh seeds:
- 32 runs: the ball fell after the catch.
- 14 runs: the ball was not held.
- 12 runs: the landing point was too far to reach within the speed limits.

## Stage 1 with realistic sensing (passed, narrowly)

![stage 1 with realistic sensors](media/stage1_sensors_real.gif)

Left: outside view. Right: the tracking camera's own view, from where the occlusion ray casts start. The green
marker is the newest camera fix the controller has; it trails the ball by the 20 ms delay. While the ball is in
the hand, the tactile skin takes over. Eight catches in a row, same disturbances as above.

`sensing.py` replaces what the controller knows about the ball with sensor models. `--sensors ideal|real` turns
it on; without the flag, the Stage 1 code path above runs unchanged.
- **Camera** (in flight): the ball's 3-D position at a frame rate, delivered after a delay. A frame is lost when
  the arm or hand blocks the line of sight (MuJoCo ray cast from the camera) or at random. Frames carry their
  capture time, so the parabola fit predicts over the delay.
- **Tactile skin** (palm and fingers): contact (force above a threshold) and the ball's position, estimated from
  the contact points and normals (0.2 mm error on average). The real Shadow Hand has tactile sensing only in the
  fingertips, so this assumes an add-on palm skin.

| preset | camera | tactile |
|---|---|---|
| ideal | every step, no delay, 2 mm noise | 100 Hz, no delay, 1 mm noise |
| real | 120 Hz, 20 ms delay, 2 mm noise, occlusion, 2% dropped frames | 100 Hz, 10 ms delay, 2 mm noise, 0.05 N threshold |

| sensing | seeds | median catches in a row | mean (95% CI) | reach 30/30 | within joint limits |
|---|---|---|---|---|---|
| Stage 1 path | 50000+ | 30 | 22.1 (20.2-24.0) | 68/120 | 120/120 |
| ideal | 50000+ | 30 | 25.0 (23.4-26.6) | 85/120 | 120/120 |
| real | 50000+ | 30 | 21.8 (19.9-23.7) | 64/120 | 120/120 |
| real, fresh seeds | 90000+ | **22** | 19.4 (17.5-21.3) | 47/120 | **120/120** |

- **A bug found on the way.** The first version took the ball's spot at release from the skin. At release the
  ball rolls across the palm at about 0.8 m/s, and a reading up to 10 ms old misplaced it by several mm. The
  catch aims for that spot, and runs lost about 7 catches on average (24-seed isolation runs). The catch now aims
  for the spot and height it was tuned with, as a real robot would; the skin only detects the release. That is
  also why "ideal" beats the Stage 1 path.
- **Occlusion.** With the camera in front of and above the robot (2.5 m away, looking down 20 degrees), the hand
  never hid the ball. A level side-on camera lost 7% of frames and one below the palm 56%, mostly while the ball
  sat in the hand, where the skin takes over.
- **Main cost of realistic sensing:** catch motions that would break the speed limits ("approach too fast",
  23 of the 120 fresh-seed runs, against 5 with ideal sensing). Delayed, sparser camera data means a later,
  faster correction. Stage 2 will need earlier catch planning or more margin.

## Stage 2a: two balls in one hand (stopped: at the arm's limits)

![two balls, one hand: the three cycles tried](media/stage2_onehand.gif)

The two-ball loop with each cycle tried, about 1/8 speed (red: ball 1, blue: ball 2, fed one cycle behind).
1: the open palm loses the caught ball in the carry. 2: the cage grip, tuned without hard limits, moves the
arm at 1.58x its speed limits and still cannot reach the next catch. 3: with the limits enforced, ball 2 is
caught and held, but re-thrown at 1.4 m/s instead of 4.5.

![close-up: catch, carry, throw](media/stage2_closeup.gif)

One catch -> carry -> throw per cycle (an undisturbed incoming ball): the whole scene on top, following the
thrown ball until it comes back down, and a close-up of the hand below. 1: the ball rolls across the
open palm and drops out during the carry. 2: the cage holds it, but the arm runs at 1.70x its speed limits
and the throw is 3.0 m/s. 3: within the limits, the cage holds it through the carry and it slips out early in
the throw, at 0.9 m/s.

**The timing.** With two balls in one hand, each ball flies F = 2V + D: the hand holds a ball for the dwell D
(catch, carry, throw) and is empty for V (from a throw to the next catch). A 1 m toss flies 0.92 s. Stage 1's
catch-to-throw takes 1.5-2.4 s (a long dip, settling in the hand, a 0.35 s stroke from rest), so the cycle
has to be redesigned, about four times faster.

**Gate** (`stage2_timing.py`, `stage2_timing_cem.py`). A periodic hand motion of two quintic joint segments,
release -> catch -> release, checked for joint speeds, joint torques (inverse dynamics with the hand) and the
palm not pulling away from the held ball.

| cycle | peak joint speed / limit | peak torque / limit |
|---|---|---|
| Stage 1 throw and catch states, squeezed in time | 1.09 or more | 1.09 or more |
| throw and catch co-designed (CEM), catch 0.10 m outward | 0.95 | 0.95 |
| same, catch 0.20 m to the side (world y) | **0.92** | **0.91** |

The best: 1 m tosses, V = 0.22 s, D = 0.48 s, about 1.4 throws per second. Feasible on paper, with
almost no margin for correcting imperfect throws (Stage 1's catch corrections already reach 0.97).

**Full simulation** (`stage2_juggle.py`, `stage2_cem.py`). What went wrong, in order:
- With the catch 0.10 m outward, the rising and falling balls passed 1.6 cm apart and collided. 0.20 m to
  the side leaves about 9 cm and costs the arm less (it uses the otherwise idle shoulder_pan).
- The fed second ball, put in flight only 0.22 s before its catch, had too short a camera track and was
  predicted 3 cm off. It is now fed one cycle earlier.
- The carry pushes the ball sideways at up to 9.5 m/s^2 against 10-15 m/s^2 into the palm, so it rolls
  across the open palm and off, while the palm turns about 40 degrees for the release.
- Tilting the palm against the roll needs about 3 rad/s of extra wrist speed. The wrist has none to spare.
- A cage grip (fingers closed over the ball, thumb opposed; `juggle.CAGE`) holds the ball. A CEM over one
  catch-carry-throw unit found it held 6 of 6, but only at 1.70x the joint speed limits, throwing at 3 m/s.
- With the limits as a hard failure, the ball stays in the cage through the carry but slips out early in
  the throw stroke, at 1-2.4 m/s instead of 4.5.

**Conclusion.** Two balls in one hand is at the edge of what this arm and hand can do. The arm (120/180 deg/s
joints carrying a 3.8 kg hand) leaves about 8% margin for a perfect cycle, and none of the simple ways to keep
the ball seated (palm tilt, finger cage, CEM-tuned timing) fit inside it. Not proven impossible: a hand
motion optimised together with the ball in the hand, a throw designed from scratch for the short carry, or a
funnel-shaped hand (as in Ploeger et al. 2020) might. Three balls in one hand would need ~7 m/s throws and is
out of reach, so the project moves to two arms, where each hand gets far more time per ball.

## Stage 2b: two balls, two arms (passed, ideal and realistic sensing, up to moderate throw errors)

![two arms, two balls, cascade rhythm](media/stage2b_ideal.gif)

Two balls on the three-ball cascade rhythm, real time, front and three-quarter views, ideal sensing, a random
0.02 m/s kick at every release (12 catches shown).

**The robot** (`scene.build(arms=2)`, `stage2b_scene.py`). Two UR10e arms 0.7 m apart, a right and a left
Shadow Hand; in the Stage 1 poses the palms are 0.35 m apart.
![two-arm scene](media/two_arm_scene.png)
The left arm in `kin.MIRROR_OFF - q` is the exact mirror image of the right arm in q (positions, palm frames,
and inverse-dynamics torques all match to 1e-15), so only the right arm's motion is designed and checked, and
the left arm runs it mirrored, half a hand period later.

**Gate** (`stage2b_timing.py`). Two patterns, each hand cycle tuned by CEM for joint speed, torque and keeping
the ball seated:

| pattern | arm limits | balls in the air |
|---|---|---|
| exchange: both hands throw across at once (F = V) | fine (0.66-1.00) | **collide**: mirror-image throws meet at the centre, even with one hand delayed |
| cascade: alternate throws every Tc / 2, throw inside, catch 10 cm outside (D = 2F - 3V) | **0.72-0.99** for every candidate | 10 cm apart |

So two balls run on the cascade cycle with every third throw empty, which is also exactly the Stage 3 cycle.
Picked: 1 m tosses (F 0.90 s), hand empty V = 0.35 s, holding D = 0.75 s, a throw every 0.55 s.

**Full simulation** (`stage2b_juggle.py`, `stage2b_cem.py`). What it took:
- The gate treats the ball as fixed in the palm. In the simulation it rolls a few mm along the fingers during
  the throw, and 3 mm there is 6 cm at the catch. The catch-carry-throw unit is tuned on the full simulation
  (CEM, 17 numbers: the gate's plus a release-velocity bias, when the hand opens, the finger grip), with 12
  disturbed incoming balls per candidate: front-back landing spread 7.2 cm -> 2.7 cm, worst 50 cm -> 14 cm.
  Opening the hand early is far worse (the ball rolls forward off the fingers); the grip stays closed until
  0.07 s before the release.
- The start-up throw (from rest) is tuned the same way. Its first tuning gamed the score (ball velocity at the
  first instant of separation, before being pushed again); both units now score the free flight 0.06 s later.
- The loop must continue from where the previous segment ends (a jump in commanded velocity while the ball
  is still on the palm cost 0.4 m/s of throw).
- A constant bias in the loop (2.5 cm forward, 11 ms early) is taken out of the release velocity.

**Catch reach** (`stage2b_reach.py`): how far the catch point can move front-back and still be reached in the
0.35 s the hand is empty, within 0.97 of the joint speed limits. The gate's cycle reaches 13-17 cm; the first
full-simulation tunings traded that down to 8 cm toward the body, which is exactly where the loop failed. The
final tuning (`--reach-min 0.12`) keeps 12 cm, and trains on incoming balls twice as disturbed (`--inc-scale 2`).

Tried and dropped: a 1.3 m toss with the hand empty 0.53 s (reach 18-21 cm, but throws scatter twice as much
and errors compound from catch to throw: 15/120 at 0.05 m/s); a linear in-hand predictor of the throw error
from the skin's ball position and the catch error (`stage2b_inhand_fit.py`, R^2 below 0.25: the errors left
are the random kicks themselves, or rare bad throws a linear model cannot see).

**Robustness** (`stage2b_robust.py`, 120 seeds, ideal sensing, a random velocity kick at every release):

| kick per axis | first cycle (v2b) | final cycle (v5), seeds 50000+ | v5, fresh seeds 90000+ |
|---|---|---|---|
| none | 106/120 reach 30 | | **120/120** |
| 0.02 m/s | 113/120 | **119/120** | **119/120** |
| 0.05 m/s (Stage 1's test) | 12/120, median 4 | 35/120, median 16 | 32/120, median 14 |

Throw spread in the tuning unit: 0.9-1.3 cm front-back. At 0.05 m/s the kick alone puts balls ~4.5 cm off per
axis (~6 cm radial). One ball, same 0.05 test: 85/120 (its hand is empty ~0.9 s). The "not held" failures
were first read as off-centre catches; they are ball-ball collisions (see Stage 3, *Where the losses come from*).

**Realistic sensing** (`--sensors real`: camera 120 Hz, 20 ms delay, occlusion by the arms, 2% dropped
frames; skin 100 Hz, 10 ms delay, 2 mm noise, 0.05 N threshold), same 120-seed test:

| kick per axis | seeds 50000+ | fresh seeds 90000+ |
|---|---|---|
| none | **120/120** | **120/120** |
| 0.02 m/s | **119/120** | **119/120** |
| 0.05 m/s | 30/120, median 14 | |

No measurable cost: each ball is tracked for ~0.5 s of its 0.9 s flight before the catch is committed, so 20 ms
of delay, a few hidden or dropped frames and 2 mm noise barely move the predicted catch point (one-ball Stage 1
was hurt more because its catch corrections were already near the speed limit).

![two arms, two balls, realistic sensing](media/stage2b_real.gif)

## Stage 3: three balls, two arms, realistic sensing (passed, up to moderate throw errors)

![three-ball cascade, realistic sensing](media/stage3_real.gif)

The three-ball cascade, real time, realistic sensing, a random 0.02 m/s kick at every release (15 catches shown).

Same cycle as Stage 2b with every throw filled (`stage2b_juggle.py --balls 3`): each ball is thrown every 1.65 s
and spends 0.90 s in the air, 0.75 s in a hand; a throw every 0.55 s. A Shadow Hand cannot start with two
balls, so the third is fed in flight at the start, as if the left hand had thrown it half a hand period
before the first throw. Balls pass about 10 cm apart where rising and falling paths cross (worst 6 cm in one
100-throw run with kicks; over many seeds they do collide, see below).

**Robustness** (`stage2b_robust.py --balls 3`, 120 seeds, 30 catches):

| kick per axis | ideal, 50000+ | realistic, 50000+ | realistic, fresh 90000+ |
|---|---|---|---|
| none | **120/120** | **120/120** | **120/120** |
| 0.02 m/s | **114/120** | **114/120** | **114/120** |
| 0.05 m/s | | 16/120, median 8 | |

A 100-throw run (realistic, 0.02 m/s) completes. As with two balls, realistic sensing costs nothing measurable.

**Where the losses come from.** With every catch logged on 60 seeds at 0.05 m/s, the arm reaches its planned
catch point to ~1 cm, but in the 41 lost catches the ball is 3-25 cm *above* it at the catch time, its velocity
0.2-1.8 m/s off the camera fit. All 41 had hit another ball ~0.18 s before. Rerun with ball-ball collisions
switched off (`stage2b_robust.py --ghost`, diagnostic only), realistic sensing, 120 seeds:

| | collisions on | collisions off |
|---|---|---|
| 3 balls, 0.02 m/s | 114/120 | **120/120** |
| 3 balls, 0.05 m/s | 16/120 | **99/120** (21 out of reach) |
| 2 balls, 0.05 m/s | 30/120 | **92/120** (28 out of reach) |

With collisions off no ball is lost from a hand; what is left is catches out of reach.

### Failure cases

`stage3_failures.py classify` sorts every failed run (120 seeds, 0.05 m/s kicks, realistic sensing; a
collision counts if it comes within 1 s before the failure):

| | 3 balls | 2 balls |
|---|---|---|
| reached 30 catches | 16 | 30 |
| collision, then the ball is not held | 68 | 43 |
| collision at the start (before the first re-throw) | 18 | 21 |
| collision, then out of reach | 8 | 6 |
| out of reach, no collision | 10 | 20 |

- **Where the balls collide:** 0.55 m above the hands (z 0.36-0.43 against catch height -0.17), one ball
  rising at ~2.4 m/s and one falling at ~2.2 m/s, 5 m/s apart, in every case. It is the cascade's crossing: a
  ball thrown up out of a hand ~0.18 s earlier meets the ball that hand will catch ~0.18 s later. Nominally
  they pass ~10 cm apart there; ~6 cm throw errors close the gap.
- **What happens next:** the falling ball is knocked 0.2-1.8 m/s off course. The camera fit catches up too
  late to re-plan, so the hand arrives where the ball should have been; the ball meets the hand late and off
  the palm centre and is not held, or its new path is out of reach.
- **At the start:** a fifth (3 balls) to a third (2 balls) of the collisions happen at the very first crossing
  (t = 1.20-1.23 s), between the two start-up throws. With no kicks the start is clean (120/120).
- **Out of reach, no collision:** the ball lands a median 12 cm off target (90th percentile 14 cm), just past
  the 12 cm catch reach the cycle was tuned to keep. The planner sees this ~0.3 s before the ball comes
  down and stops the run.

Each GIF below shows about 1.6 s in real time, then x8 slow motion around the collision (or the miss), and
continues for 0.8 s after the failure. Balls are red, cyan and yellow, with trails of their last 0.3 s.

**Collision, ball lost** (3 balls, seed 50000, after 12 catches). The rising cyan ball and the falling red
ball bend at the contact; the red ball comes down off target and the left hand does not hold it:

![collision, ball lost](media/fail_b3_collision_lost.gif)

**Collision at the start** (3 balls, seed 50005). The first two throws meet at the first crossing:

![collision at the start](media/fail_b3_start_collision.gif)

**Collision, then out of reach** (3 balls, seed 50024, after 14 catches):

![collision, out of reach](media/fail_b3_collision_reach.gif)

**Out of reach, no collision** (3 balls, seed 50002, after 4 catches). The cyan ball comes down ~14 cm
from the catch spot and falls past the left hand:

![out of reach](media/fail_b3_reach.gif)

**Two balls, collision, ball lost** (seed 50014, after 10 catches). The same crossing, with one slot empty:

![two balls, collision](media/fail_b2_collision_lost.gif)

### Avoiding the collisions

![three balls with the release-point offset, 0.05 m/s kicks](media/stage3_aim.gif)

Three balls, realistic sensing, a random 0.05 m/s kick at every release (seed 50003, which loses a ball after
6 catches without the offset; 15 catches shown).

Two fixes, tried one at a time and together (120 seeds, realistic sensing, 30 catches):

| | 10 cm spacing (Stage 3) | 15 cm | 20 cm | 10 cm + offset | 15 cm + offset | 20 cm + offset |
|---|---|---|---|---|---|---|
| 3 balls, 0.05 m/s | 16 | 28 | 20 | **69** | 34 | 20 |
| 2 balls, 0.05 m/s | 30 | 32 | 19 | **76** | 36 | 19 |
| 3 balls, 0.02 m/s | 114 | 108 | 118 | **119** | 109 | 118 |
| 3 balls, no kick | 120 | | 120 | **120** | | |

Held-out seeds 90000+, Stage 3 cycle without / with the offset: 3 balls at 0.05 m/s 14 -> **71**, 2 balls
35 -> **81**, 3 balls at 0.02 m/s 114 -> **120/120**.

- **Release-point offset** (`stage2b_juggle.aim_offset`, `--aim`): 0.2 s before each release the hand takes the
  camera's fit of the ball it will catch next, which is the ball its throw will cross, and predicts their
  closest approach. If it is under 10 cm, the release point moves up to 4 cm away from that ball and the throw
  velocity changes by -offset / F, so the ball still lands on the same spot at the same time; at the
  crossing the paths move ~0.8 x the offset apart. The stroke is re-planned from the current state within the
  speed limit (else half the offset). Two things that do not work: shifting the release time (gravity cancels
  between two balls, so relative to each other they move in a straight line, ~5 m/s and mostly vertical where
  they cross; a time shift only slides along it), and a sideways velocity change alone (~5 cm of landing
  error per cm gained). The offset is off for the two start-up throws: their separately tuned stroke does not
  take it (with up to 4 cm: 49/120; 2 cm: 54/120 and 11 runs over the torque limit; 1 cm: 73/120, within
  noise of none).
- **Wider spacing** (catch spot 15 or 20 cm outside the throw spot; `stage2b_timing.py --wide`,
  `stage2b_cem.py --d-y`): the gate cycles stay within the limits and the balls pass 15 / 19 cm apart, so
  balls lost from the hand nearly vanish (20 cm, 0.05 m/s: no "not held" in 120 runs; 15 cm: 16, against 86
  at 10 cm), and three balls at 0.02 m/s improve to 118/120. But
  at 0.05 m/s nearly every run then ends out of reach: the catch spot further out leaves 7 cm of outward
  reach instead of 12, and the tuned throws are less precise front to back (2.0 cm -> 4.5-5.4 cm spread at 0.02
  m/s; unit cost 0.12-0.21 against 0.02). A re-tune with the reach floor applied sideways too (`--reach-y`)
  could only reach 9.6 cm and threw worse (5/120).
- With the offset, what is left (`stage3_failures.py classify --aim`, 0.05 m/s): 3 balls 18 start-up
  collisions, 23 collisions during juggling (was 68), 10 out of reach; 2 balls 21 start-up collisions, 5
  collisions (was 43), 18 out of reach.

### Start-up throws and a reach-aware offset

![three balls, re-tuned start-up throw and reach-aware offset, 0.05 m/s kicks](media/stage3_start.gif)

Seed 50005, which used to lose a ball to a start-up collision before its first catch (the
`fail_b3_start_collision` case above), now with both changes (15 catches shown; it reaches 30 in the
evaluation).

120 seeds each, realistic sensing, 0.05 m/s kicks, runs reaching 30 catches (all within the limits):

| | 3 balls | 3 balls, held-out | 2 balls | 2 balls, held-out |
|---|---|---|---|---|
| offset (above) | 69 | 71 | 76 | 81 |
| + start-up stroke with separation | 78 | 82 | 95 | 93 |
| + reach-aware offset | **94** | **92** | **98** | **99** |

At 0.02 m/s and with no kick, the final version stays at 120/120.

- **Start-up stroke** (`stage2b_cem.py --start ... --gap-min 0.13`, `runs/start2b_v5g13.json`): both start
  crossings (the right hand's first throw against the fed third ball, the left hand's first throw against
  ball 0) passed only 9.1-9.7 cm apart, centre to centre. The start-up stroke is re-tuned with its own
  release-point offset (2 more parameters, the same landing-preserving velocity change) and a penalty below a
  gap target. The gap is computed from the released ball against its own mirror image released Tc / 2
  earlier, which is exact, because the right hand's start-up throw is the left's mirror image. The third
  ball is now fed along that mirror path. Targets 12 / 13 / 14 cm all tune to < 1 mm landing error; in the
  two-arm simulation the crossings are then 15 cm apart. Over 120 seeds the three are 81 / 78 / 81 (3 balls)
  and 82 / 95 / 86 (2 balls); the 13 cm one is kept. Losses before the first catch from a dropped ball go
  from 18-21 to 0-1.
- **Reach-aware offset** (`stage2b_juggle.aim_reach`, now the default `AIM`): instead of one direction (away
  from the incoming ball) and a growing size, 17 candidate offsets (none, and 8 directions at 2 and 4 cm).
  Candidates whose re-planned stroke exceeds the speed limit are dropped. Of those that clear the 10 cm gap
  (or, if none does, the ones with the largest gap), the choice is the lowest peak speed ratio of the empty
  move from that release to the predicted catch of the incoming ball, plus 0.05 per 4 cm of offset. The
  peak ratios differ little between candidates (~0.005), so how much each part contributes is not clear: with
  the offset size weighted 10 instead of 0.05 (in effect: the smallest offset that clears the gap, any
  direction) it is 87 (3 balls) and 94 (2 balls), between the old offset and the reach-aware one.
- A harness bug found on the way: with the wider cycles the ball lifts off the opening hand before the
  release, the release detector fired there and again at the real release, and each such throw got two kicks.
  Releases now count only once the stroke has reached the release point (the Stage 3 cycle never triggered it,
  so its earlier numbers stand).

### Four balls: feasibility gate (not started)

Four balls on two hands need F = 2 V + D (Shannon), whatever the pattern. The gate (`stage2b_timing.py
--four`, `logs/stage4_timing.txt`) over throw speeds 5-6.5 m/s (apex 1.3-2.2 m): only cycles with the shortest
empty time, V = 0.30 s, stay within the arm limits, and only at 0.97-0.99 of them (the loop plans at <= 0.97,
the tuned cycles run at 0.93). V = 0.35 s already leaves only ~12 cm of catch reach; at 0.30 s and the speed
limit there is essentially none to chase an imperfect throw. Also, 4 balls on two hands are either two
one-hand fountains (Stage 2a: stopped at the arm's limits) or a synchronous crossing whose mirror-image throws
meet unless the hands throw one in front of the other (the left arm would no longer be the right arm's
mirror). Not started.

**What-ifs** (`stage2b_timing.py --four2`, `logs/stage4_gate2.txt`), each cycle re-optimised:

| change | best V = 0.30 s cycles | best V = 0.35 s cycles |
|---|---|---|
| none (UR10e) | 0.97-0.99 | 1.04-1.07 |
| catch spot 20 cm to the side (Stage 2a: brings in shoulder_pan) | 0.98-1.00 | 1.05-1.09 |
| release-pose range +-0.8 rad instead of +-0.4 | 0.97-0.99 | 1.04-1.06 |
| both | 0.99-1.00 | 1.04-1.07 |
| speed and torque limits x 1.25 | **0.78-0.80** | **0.82-0.85** |
| speed and torque limits x 1.5 | 0.65-0.66 | 0.70-0.71 |

(peak ratio to the limits in force, speed and torque, throw speeds 4.5-6.5 m/s.) A different pose does not
help. The ratios fall as 1 / scale: the optimiser keeps the same motion, limited by wrist_1 speed (180 deg/s)
and elbow torque (150 N m) together. The 3-ball cycle in use sits at 0.77 in this gate, so an arm about 1.25x
faster and stronger at the elbow and wrist would give 4 balls the same headroom at V = 0.30, and ~1.4x at V
= 0.35 (the 3-ball cycle's empty time). Within the UR10e's limits, 4 balls stay out.

## Next

- What is left at 0.05 m/s (final version, seeds 50000+): 3 balls 19 lost from the hand and 7 out of reach;
  2 balls 8 lost and 14 out of reach. Not yet re-classified; before, nearly all losses from the hand followed a
  collision.
- Four balls: no pose change helps; an arm ~1.25-1.4x faster and stronger at the elbow and wrist would (gate
  above). Building it in simulation means a hypothetical arm (scaled limits) and a new pattern (with an even
  number of balls each hand catches its own throws: two one-hand fountains, or a synchronous crossing).
- Path to hardware: UR arms take position/velocity streaming, not torque; finger speeds; friction and spin
  randomisation; safety.

```
PYTHONPATH=. MUJOCO_GL=egl python3 stage0_speed.py              # throw-speed feasibility (~35 s)
PYTHONPATH=. MUJOCO_GL=egl python3 stage0_toss.py --gif media_full/stage0_toss.gif
PYTHONPATH=. MUJOCO_GL=egl python3 stage0_robust.py --n 120     # needs robolab (pip install -e ~/robo/robolab)
PYTHONPATH=. MUJOCO_GL=egl python3 stage1_juggle.py --catch runs/catch5_v443.json --cycles 8 --seed 50001 --kick 0.05 --noise 0.002 --gif media_full/stage1_juggle.gif
PYTHONPATH=. MUJOCO_GL=egl python3 stage1_robust.py --n 120 --catch runs/catch5_v443.json             # seeds 50000+
PYTHONPATH=. MUJOCO_GL=egl python3 stage1_robust.py --n 120 --seed0 90000 --catch runs/catch5_v443.json --out logs/stage1_verify_90000.jsonl
PYTHONPATH=. MUJOCO_GL=egl python3 stage1_robust.py --n 120 --catch runs/catch5_v443.json --sensors real --out logs/stage1_sensors_real.jsonl
PYTHONPATH=. MUJOCO_GL=egl python3 stage1_juggle.py --catch runs/catch5_v443.json --cycles 8 --seed 50001 --kick 0.05 --noise 0.002 --sensors real --gif media_full/stage1_sensors_real.gif
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2_timing_cem.py 0 0.2      # two balls, one hand: periodic-cycle gate (~3 min)
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2_cem.py --gens 50 --out runs/cycle2_v2.json   # catch-carry-throw unit CEM (~25 min)
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2_onehand_gif.py media_full/stage2_onehand.gif
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2_closeup_gif.py media_full/stage2_closeup.gif
PYTHONPATH=. MUJOCO_GL=egl python3 stage2b_scene.py                              # two-arm scene images
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_timing.py                   # two-hand gate (~5 min)
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_cem.py --gens 40 --incs 12 --inc-scale 2 --reach-min 0.12 --init runs/cycle2b_v4.json --out runs/cycle2b_v5.json   # (~25 min)
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_cem.py --start runs/cycle2b_v5.json --out runs/start2b_v5.json
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_reach.py
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_robust.py --n 120 --kick 0.02 --sensors ideal --out logs/stage2b_v5_ideal_kick0.02.jsonl
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_juggle.py --throws 12 --seed 50000 --kick 0.02 --gif media_full/stage2b_ideal.gif
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_robust.py --balls 3 --n 120 --seed0 90000 --kick 0.02 --sensors real --out logs/stage3_real_kick0.02_90000.jsonl
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_juggle.py --balls 3 --throws 15 --seed 7 --kick 0.02 --sensors real --gif media_full/stage3_real.gif
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_robust.py --balls 3 --kick 0.05 --sensors real --ghost --out logs/stage3_ghost_b3_kick0.05.jsonl
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage3_failures.py classify --balls 3 --out logs/stage3_failures_b3.json
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_robust.py --balls 3 --kick 0.05 --sensors real --run runs/cycle2b_v5.json --start runs/start2b_v5.json --aim 0.10 0.04 0.2 --aim-start 0 --out logs/fix_aimns_b3_kick0.05.jsonl
# (stage3_aim.gif and the failure GIFs were made with the old start-up stroke, --start runs/start2b_v5.json, and the old offset; the defaults are now the new ones)
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_timing.py --wide 0.15 0.20
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_cem.py --gens 40 --incs 12 --inc-scale 2 --reach-min 0.12 --d-y 0.20 --init runs/cycle2b_v5.json --out runs/cycle2b_w20.json
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_timing.py --four
PYTHONPATH=.:../robolab python3 stage2b_timing.py --four2
PYTHONPATH=.:../robolab python3 stage2b_cem.py --start runs/cycle2b_v5.json --init runs/start2b_v5.json --gap-min 0.13 --gens-start 40 --out runs/start2b_v5g13.json
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_robust.py --balls 3 --kick 0.05 --sensors real --run runs/cycle2b_v5.json --start runs/start2b_v5g13.json --aim 0.10 0.04 0.2 --aim-start 0 --aim-reach 0.05 --out logs/fix_sg13reach_b3_kick0.05_50000.jsonl
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage2b_juggle.py --balls 3 --throws 15 --seed 50005 --kick 0.05 --sensors real --aim --gif media_full/stage3_start.gif
PYTHONPATH=.:../robolab MUJOCO_GL=egl python3 stage3_failures.py gif --balls 3 --seed 50000 --title "3 balls: collision, ball lost" --gif media_full/fail_b3_collision_lost.gif
```
