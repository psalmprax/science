# PREREG-PHYSICS-J1-2026-11: the physics judge's blind test on clips rendered here (gate J, part 1)

**Status:** draft, not frozen. Written 27 November 2026, on the owner's word "go ahead to test all" (02:32 Berlin
time). That word took up two proposals of `docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md`:

- test the judge mainly on clips simulated and rendered here;
- keep PhysInOne as a second check beside gate J.

Freezing has two stages:

1. **The judge.** Its files are frozen by `preregistration/DIGEST-PHYSICS-J1-JUDGE.json` after an independent
   review of this plan.
2. **The breakages it was not developed against.** An independent reviewer who took no part in building the
   judge then writes them, aimed at the frozen judge. They are frozen with the plan by
   `preregistration/DIGEST-PHYSICS-J1.json`.

After that the test runs once, on a branch `physics-j1-run-1`.

## What gate J asks, and what this part reads

Gate J (the plan's proposed numbers, which the owner kept on 27 November at 00:07) covers falls, throws, rolls,
bounces and pendulums:

- at least 80 % of true-physics clips pass;
- at most 10 % of broken clips pass, counted separately for every kind of breakage, the reviewer's unseen kinds
  included;
- agreement with the human majority on human-labelled videos, Cohen's kappa at least 0.3.

This test reads the first two. The third needs an object tracker on generated video (the judge reads tracks,
and real or generated footage has no masks). It is **part 2**, a later preregistration. Gate J holds only if
both parts hold.

## The judge

`apedp_dyn/epistemic/physics_judge.py` with its development settings, as at freezing. Its development history
is in `CHANGELOG.md` and the plan:

- PhysInOne development runs 1, 3 and 4;
- 20 to 30 development clips per kind rendered here (`--mode dev`: seeds 0 to 29);
- 27 synthetic tests.

## Clips

- **Rendered here**, `apedp_dyn/video/physics_clips_pybullet.py` (PyBullet, 240 Hz physics, 320 x 240 frames
  at 30 per second, a nearly level side camera, exact segmentation masks): **60 clips of each of the five
  gate kinds**, seeds 1,000,000 to 1,000,059. No development run used them; the module refuses them unless a
  frozen run asks. Collisions get 60 clips too, judged and reported **beside** the gate, as the owner chose.
- **The development breakages** (`apedp_dyn/video/physics_perturbations.py`, as frozen):
  - a jump of three object sizes;
  - a hold that then carries on;
  - the vertical motion mirrored for a stretch;
  - time reversal of clips that lose energy.
- **The reviewer's breakages** (`apedp_dyn/video/physics_perturbations_unseen.py`). The reviewer writes them
  after stage 1 and may read the frozen judge to aim at what it might miss. Each kind must:
  - break the law the clip shows (the reviewer writes down which law and how);
  - change the track by more than its noise;
  - be a function `perturb_unseen(kind, tracks, how, rng)` that returns `None` where it does not apply.

  The builder does not see them before stage 1.
- **PhysInOne beside**: 30 scenes of each kind from its blind pool (backgrounds that are multiples of five; 60
  scenes for collisions), as in development. The same true and broken variants, reported and not read. Its
  cameras all look 30 to 33 degrees down, its rigid scenes step irregularly, and its pendulums are pushed. No
  PhysInOne track is written to the record.

## Expectations, fixed now (gate J, part 1)

Over the 300 rendered true clips of the five gate kinds and their breakages:

1. **True clips:** at least 80 % pass. An untestable verdict counts as not passing.
2. **Every kind of breakage**, each development kind and each of the reviewer's, pooled over the five kinds: at
   most 10 % of the clips it applied to pass.
   - A kind of breakage that applies to fewer than 20 clips is reported, not read: there, two clips would
     already be 10 %.
   - At least one of the reviewer's kinds must be read, or part 1 is not read.

Untestable broken clips do not pass; their counts are reported beside each kind.

## What each outcome proposes (the owner decides; AGENTS.md 1.5)

- **Both hold:** part 1 of gate J holds. Propose part 2 (an object tracker on generated video, then agreement
  with PhyGround's or VideoPhy-2's human labels) as the next preregistration. Nothing goes in force: the judge
  stays experimental and no route reads it.
- **Either fails:** the plan's stop rule applies, and step 1 stops here with the result recorded. The judge is
  listed as retired (deprecated, kept for replication), under the owner's standing word of 27 November 00:41.

## Limits

- **In-distribution true clips.** The judge was developed on this generator from 27 November, so the blind
  rendered clips measure performance on the distribution it was tuned on, though on seeds and scenes it never
  saw. The arms it was not developed against are:
  - the reviewer's breakages, aimed at the frozen judge after freezing (AGENTS.md 1.6);
  - PhysInOne, another generator, reported beside.
- **Exact masks.** Tracks come from the renderer's segmentation buffer. A generated video would need a tracker,
  whose errors this test does not see; part 2 does.
- **Narrow scenes.** One object (two for collisions), a plain floor, a side view, three seconds.

## Disclosed before freezing

- The development runs and every development change are in `CHANGELOG.md` (27 November). The runner's
  `--mode dev` reproduces the development tally on seeds 0 to 29.
- The judge's settings are its development numbers, none pinned. Gate J's two numbers are the plan's, kept by
  the owner. The 20-clip floor for reading a kind of breakage is this test's own number.
