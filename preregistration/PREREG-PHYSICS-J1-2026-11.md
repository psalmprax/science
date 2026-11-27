# PREREG-PHYSICS-J1-2026-11: the physics judge's blind test on clips rendered here (gate J, part 1)

**Status:** draft, not frozen; revised after the independent review of 27 November (below). Written on 27
November 2026, on the owner's word "go ahead to test all" (02:32 Berlin time). That word took up two proposals of
`docs/PLAN_PHYSICS_VIDEO_STEP1_2026-11-26.md`:
- test the judge mainly on clips simulated and rendered here;
- keep PhysInOne as a second check beside gate J.

Freezing has two stages:

1. **The judge.** After a check of this revision, `preregistration/DIGEST-PHYSICS-J1-JUDGE.json` freezes the
   judge, its breakages, the generator, this plan, the runner, the workflow and the tests. It is the pin of
   gate J's numbers for this run.
2. **The breakages it was not developed against.** The independent reviewer writes them after the stage-1
   commit, aimed at the frozen judge. `preregistration/DIGEST-PHYSICS-J1.json` then freezes them with everything
   in stage 1, the stage-1 digest included.

The test then runs once on a branch `physics-j1-run-1`. Its blind seeds are used up afterwards.

## What gate J asks, and what this part reads

Gate J (the plan's proposed numbers, kept by the owner on 27 November at 00:07) asks three things:
- at least 80 % of true-physics clips pass;
- at most 10 % of broken clips pass, counted separately for every kind of breakage, the reviewer's unseen kinds
  included;
- agreement with the human majority on human-labelled videos, Cohen's kappa at least 0.3.

This test reads the first two. The third needs an object tracker on generated video: the judge reads tracks, and
real or generated footage has no masks. It is **part 2**, a later preregistration. Gate J holds only if both parts
hold.

**Which scene kinds.** All six: falls, throws, rolls, bounces, pendulums and collisions. The owner's choice of
00:07 kept collisions out "until they pass in development". On the development clips rendered here they now do: 28
of 30 true clips pass, and no broken one passes above the limit. So they join, and the owner can take them out.

**How the counts are read.** Gate J's text does not say whether a kind of breakage is counted over all scene kinds
together or for each one. This test reads it **for each scene kind**: every scene kind's true clips, and every
scene kind against every kind of breakage. The pooled reading is reported beside. The outcome follows from both
readings:

| Per scene kind | Pooled | State |
| :--- | :--- | :--- |
| holds | holds | holds |
| fails | holds | the owner decides |
| fails | fails | fails |

## The judge, its breakages and the clips

**The judge** is `apedp_dyn/epistemic/physics_judge.py` with its development settings as frozen in stage 1. It
was developed on:
- PhysInOne development runs 1, 3 and 4;
- the 30 development clips per kind rendered here (seeds 0 to 29);
- 27 synthetic tests.

Every development change is in `CHANGELOG.md`.

**The clips.** `apedp_dyn/video/physics_clips_pybullet.py` renders them: our own generator, PyBullet rigid bodies
at 240 Hz, 320 x 240 frames at 30 per second, a nearly level side camera, exact segmentation masks.
- **60 blind clips of each scene kind.** Their seeds start at a base derived from the stage-2 digest
  (`seeds("final")`), so no blind clip existed before stage 2. The module refuses seeds from 1,000,000 up
  unless a frozen run asks.

**The development breakages** (`apedp_dyn/video/physics_perturbations.py`):
- a jump of three object sizes;
- a hold that then carries on;
- the vertical motion mirrored for a stretch, wherever that moves the object at least 5 px, contacts included;
- time reversal of clips that lose energy: those that end at rest, or whose last swing is at least a tenth below
  the first on the same side and by four noise widths.

Where each applies is set by the physics and the track's noise, not by the judge's tolerances. Each breakage
draws from its own stream, named by a hash of the breakage, and gets its own copy of the tracks.

**The reviewer's breakages** (`apedp_dyn/video/physics_perturbations_unseen.py`) follow these rules, written now:

1. **Who and when.** The reviewer alone writes them, after the stage-1 commit. The builder does not see them
   before stage 2. The module names the stage-1 digest's SHA-256 (`STAGE1_SHA256`), and the collection checks in
   git history that the file was first committed after stage 1.
2. **Interface.**
   - `UNSEEN_KINDS` gives at least two kinds, with names other than the development kinds and "true".
   - `LAWS` maps each kind to the law it breaks.
   - `perturb_unseen(kind, tracks, how, rng)` returns broken tracks, or `None` where the kind does not apply. It
     is deterministic, changes nothing it is given, and raises nothing on the 180 development clips.
3. **Applicability.** It is computed from the true track only, without calling the judge. The reviewer records
   it per scene kind on the development clips before stage 2.
4. **Physics.** Each kind breaks a named law. The broken track must lie outside the lawful family under the
   judge's declared invariances: a time scaling, a changed g or a left-right mirror are lawful and not allowed.
5. **Size.** The breakage moves the track by at least 5 px and 10 noise widths, a floor independent of the judge.
6. **Testability.** A broken clip the judge calls untestable counts as not caught, so "untestable" is no escape.
7. **Coverage.** At least two kinds, covering different laws. Every one of them must apply to some clip in the
   blind run, or part 1 is not read.

The reviewer may look beyond the blind spots the review of this plan named. The builder fixed some of those before
stage 1, so the traps must reach further.

**PhysInOne beside.** 30 scenes of each kind from its blind pool (backgrounds that are multiples of five; 60
scenes for collisions), with the same true and broken variants. These are reported, not read. Breakages are also
reported among the clips whose true version passed, since about half of PhysInOne's true clips fail. No
PhysInOne track is written to a record or a log.

## Expectations, fixed now (gate J, part 1)

For each of the six scene kinds:

1. **True clips:** at least 80 % of the 60 pass. An untestable verdict or an error counts as not passing.
2. **Every kind of breakage**, development and reviewer's alike: at most 10 % of the clips it applied to are not
   caught. Only a fail is caught; pass, untestable and error are not.
   - A cell applied to fewer than 30 clips is read by count: it fails if more than 2 of its clips are not caught.
     At 30 clips, 0 not caught bounds the share below 10 % at 95 %.

The pooled reading takes the same two expectations over all six scene kinds together. One-sided 95 %
Clopper-Pearson bounds are reported beside every share, and untestable counts beside every kind of breakage. Those
bounds are not read: reading the limits on the bounds would change gate J's meaning, and that is the owner's to
decide.

**Not read.** The run is not read in four cases:
- a clip or a breakage row is missing or duplicated;
- a shard did not finish;
- a loaded file differs from the checkout;
- fewer than two reviewer kinds were read, or git history does not show the reviewer's file after stage 1.

A run that is not read is void. A new run would draw fresh seeds from a new stage-2 digest.

## What each outcome proposes (the owner decides; AGENTS.md 1.5)

- **Holds:** part 1 of gate J holds. Part 2 is proposed as the next preregistration: an object tracker on
  generated video, then agreement with PhyGround's or VideoPhy-2's human labels. Nothing goes in force; the judge
  stays experimental and no route reads it.
- **Fails:** the plan's stop rule applies, step 1 stops here and the result is recorded. The judge is listed as
  retired (deprecated, kept for replication), under the owner's standing word of 27 November 00:41.
- **The owner decides:** the pooled reading holds and a scene kind fails. Whether gate J is pooled is the owner's
  to say. Nothing is retired and nothing proceeds until they do.
- **Not read:** void, as above.

## Limits

- **The true clips are in-distribution.** The judge was developed on this generator from 27 November, so the blind
  rendered clips measure it on the distribution it was tuned on, though on seeds it never saw. Two arms were not
  part of its development: the reviewer's breakages, aimed at the frozen judge after freezing (AGENTS.md 1.6), and
  PhysInOne, reported beside.
- **On PhysInOne the judge is far from gate J.** In development run 4, 70 of 150 true clips passed, and of the time
  reversals 23 of 68 were not caught. The move to clips rendered here followed those failures. PhysInOne's own data
  issues explain part of them: downward cameras, irregular steps, pushed pendulums.
- **Exact masks.** Tracks come from the renderer's segmentation buffer. A generated video would need a tracker,
  whose errors this test does not see; part 2 does.
- **Narrow scenes.** One object (two for collisions), a plain floor, a side view, three seconds.
- **Some breakages look lawful.** A stretch mirrored at the moment of landing mostly moves where the object comes
  to rest, or makes a small bounce. In development, 9 of 138 mirrored falls, throws and bounces were of this
  kind and were not caught. The rule that applies a mirror wherever it moves the object 5 px counts them anyway,
  as the review asked.
- **The development arms' outcome is largely foreseeable.** In development, true clips pass 177 of 180, and the
  development breakages are not caught in 0 to 14 % of a cell. The reviewer's kinds are what this test really
  tests.

## Disclosed before freezing

- **Development tally, 30 clips per kind, seeds 0 to 29, with the judge and breakages as now:**

  | Scene kind | True passing | Jumps | Holds | Mirrored stretches | Time reversals |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | falls | 29 of 30 | 0 of 30 | 2 of 30 | 4 of 28 | 0 of 30 |
  | throws | 30 | 0 of 30 | 0 of 30 | 3 of 29 | 0 of 30 |
  | rolls | 30 | 0 of 30 | 0 of 30 | 0 of 26 | none apply |
  | bounces | 30 | 0 of 30 | 0 of 30 | 2 of 29 | 0 of 29 |
  | pendulums | 30 | 0 of 30 | 0 of 30 | 0 of 26 | 0 of 20 |
  | collisions | 28 of 30 | 0 of 30 | 1 of 30 | none apply | 0 of 29 |

  The broken columns count clips not caught. Read per scene kind, mirrored falls (4 of 28) would already miss
  the 10 % limit. Pooled, mirrored stretches are 9 of 138 not caught.
- **The independent review of 27 November** found twelve problems. Each was addressed:
  - **Rules that excused the judge.** Two changes to the breakages that day had set their applicability at the
    judge's own tolerances. A swing then lost energy only below three quarters of its first. A mirrored stretch
    applied only where the object kept moving vertically for half of it, or moved an object size inside it.
    Both rules are replaced by physics-and-noise rules: a tenth and four noise widths; 5 px wherever, contacts
    included.
  - **What the new rules exposed, fixed in the judge as development:**
    - pendulums mirrored off their arc (10 of 26 passed): the judge now also reads each window's distance from the
      arc, not only the whole clip's robust spread;
    - reversed swings growing 13 % a swing (2 of 20 passed): growth is now allowed by noise, four widths, or 3 %
      of the swing, against 15 % before (true swings here never grew: at most 0.994 of the one before);
    - mirrored rolls rising while speeding up (3 of 26 passed): a new rule says a passive object cannot gain speed
      while it rises.
  - **Pooling hid a failing scene kind.** The counts are now read per scene kind, with the pooled reading beside.
  - **A reviewer kind could be set aside, and one kind sufficed.** Every reviewer kind must now be read, and
    there must be at least two.
  - **Collect did not check completeness, and a shard lost everything on one error.** It now checks every clip
    and breakage exactly once. Shards keep rows as they go, and errors are rows.
  - **"Not read" came out as "fails".** It is now its own state, with its own consequence.
  - **The freeze could be rewritten.** Stage 2 now includes the stage-1 digest. The reviewer's module names it,
    git history is checked, and loaded files and versions are recorded.
  - **The reviewer's rules were too thin.** They are written above.
  - **Too few clips to read a small cell.** The floor is raised from 20 to 30 clips, with bounds beside.
  - **Blind seeds were public.** They are now derived from the stage-2 digest, and streams are named by hash.
  - **Collisions and PhysInOne's numbers.** Collisions join the gate under the owner's own rule, and PhysInOne's
    numbers are in Limits.
  - **The workflow.** It now catches `BadZipFile`, uploads the record, pins NumPy and PyBullet and fetches git
    history. Separately, PhysInOne tracks had been printed in the development workflow's public logs; they no
    longer are.
- **Pendulum scene and loss rule changed earlier on 27 November, before the review.** After untestable clips, the
  pendulum scene's length (0.35 to 0.8 m) and damping (0 to 0.4) were changed so that clips turn at least three
  times and some lose energy visibly.
- **Thresholds.** The judge's settings are its development numbers, and none is pinned. Gate J's two numbers are
  the plan's, kept by the owner. The 30-clip floor, the 2-clip count for small cells, the two reviewer kinds and
  the 5 px and noise floors are this test's own. The stage-1 digest pins them all.
