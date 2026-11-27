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
2. **The breakages it was not developed against, and the reading.** The independent reviewer writes the
   breakages after the stage-1 commit, aimed at the frozen judge. The owner's choice of reading is written to
   `preregistration/PHYSICS-J1-READING.json`. The reviewer then writes and commits
   `preregistration/DIGEST-PHYSICS-J1.json` in its canonical form. It freezes the breakages and the reading with
   everything in stage 1, the stage-1 digest included.

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

**How the counts are read: the owner's choice, made before stage 2.** Gate J's text does not say whether a kind of
breakage is counted over all scene kinds together or for each scene kind. Both readings are computed and reported
in every case; the owner chooses which one decides, and the choice is written to
`preregistration/PHYSICS-J1-READING.json`. That file is frozen in stage 2, and the run refuses to start without it.

| Reading | Holds when |
| :--- | :--- |
| `per_kind` | every scene kind holds on its own (its true clips, and every kind of breakage), **and** the pooled reading holds (small cells can each hold while their sum fails) |
| `pooled` | the pooled reading holds: all six scene kinds together, for true clips and for every kind of breakage |

On the development numbers the two readings already disagree: `per_kind` would fail on mirrored falls and throws,
and `pooled` would hold (Disclosed, below). The choice is asked of the owner now, before any blind clip exists, not
after the run. The builder recommended `per_kind`, the stricter one, because pooling can hide a scene kind the judge
fails.

## The judge, its breakages and the clips

**The judge** is `apedp_dyn/epistemic/physics_judge.py` with its development settings as frozen in stage 1. It
was developed on:
- PhysInOne development runs 1, 3 and 4;
- the 30 development clips per kind rendered here (seeds 0 to 29);
- 27 synthetic tests.

Every development change is in `CHANGELOG.md`.

**The clips.** `apedp_dyn/video/physics_clips_pybullet.py` renders them: our own generator, PyBullet rigid bodies
at 240 Hz, 320 x 240 frames at 30 per second, a nearly level side camera, exact segmentation masks.
- **60 blind clips of each scene kind.** Their seeds start at a base derived from the stage-2 digest's bytes
  (`seeds("final")`), so no blind clip existed before stage 2. The digest must be in its one canonical byte form
  (indent 1, sorted keys, a final newline, no other keys), and the reviewer writes and commits it, so its writer
  cannot choose the seeds by its formatting. The generator refuses seeds from 1,000,000 up unless the caller
  passes `allow_blind=True`. Only the frozen runner does, in final mode, after `check_frozen`.

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
7. **Coverage.** At least two kinds, covering different laws. Every one of them must apply to at least 30 clips
   over all scene kinds in the blind run, or part 1 is not read. The reviewer checks this on the development
   seeds before stage 2.

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

**Not read.** The run is not read in five cases:
- a clip or a breakage row is missing or duplicated;
- a shard did not finish;
- a loaded file differs from the checkout;
- fewer than two reviewer kinds applied to 30 clips each;
- git history does not show the freeze as written. Each digest must come from exactly one commit, the reviewer's
  file must be first added after stage 1, stage 2 must come after the reviewer's last change, and between stage 1
  and the run no frozen or loaded file may change except the reviewer's file and the stage-2 digest. The run
  branch is cut from the stage-2 commit.

A clip that fails to render gets an error row for its true version and every breakage. Those count against the
judge (not passing, not caught), so no row is left out.

A run that is not read is void. A new run would draw fresh seeds from a new stage-2 digest.

## What each outcome proposes (the owner decides; AGENTS.md 1.5)

- **Holds:** part 1 of gate J holds. Part 2 is proposed as the next preregistration: an object tracker on
  generated video, then agreement with PhyGround's or VideoPhy-2's human labels. Nothing goes in force; the judge
  stays experimental and no route reads it.
- **Fails:** the plan's stop rule applies, step 1 stops here and the result is recorded. The judge is listed as
  retired (deprecated, kept for replication), under the owner's standing word of 27 November 00:41.
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
- **Mirrors at landing.** A stretch mirrored at the moment of landing mostly moves where the object comes to rest,
  or makes a small bounce. In development, 9 of 138 mirrored falls, throws and bounces were of this kind and not
  caught. Seven of them leave the object within about one size of where it really rests. Two leave it floating
  about two sizes above: free_fall seed 7 (34 px) and throw seed 8 (23 px). Those are clearly broken, and the
  judge misses them. A rule "comes to rest above its lowest point" was tried in development and rejected, because
  true clips rest up to 0.57 sizes above their lowest point. The rule that applies a mirror wherever it moves the
  object 5 px counts all nine, as the review asked.
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

  The broken columns count clips not caught. Read per scene kind, development would already miss: mirrored falls,
  4 of 28, are over 10 % of a cell of 30 clips or so, and mirrored throws, 3 of 29, are over the count of 2 for a
  small cell. Pooled, mirrored stretches are 9 of 138 not caught, which holds. So on development numbers the state
  would differ by reading. That is why the reading is put to the owner before stage 2, not after the run.
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
- **The same reviewer checked the revision** and found three blocking points and five more:
  - the gate said "holds" when the pooled reading failed (now it fails);
  - git history did not catch a rewrite after stage 2 (now the checks above);
  - "the owner decides" was already the development outcome, so the reading is now the owner's choice, made before
    stage 2 and frozen with it, and that state is gone;
  - a rendering error voided the run (now error rows count against the judge);
  - the stage-2 digest's formatting could choose the seeds (now one canonical form, written by the reviewer);
  - a reviewer kind applied to two clips could be read (now 30 pooled);
  - the two floating mirrored misses were not named (now in Limits);
  - the seed refusal was overstated (now worded as the code has it).
- **Thresholds.** The judge's settings are its development numbers, and none is pinned. Gate J's two numbers are
  the plan's, kept by the owner. The 30-clip floor, the 2-clip count for small cells, the two reviewer kinds and
  the 5 px and noise floors are this test's own. The stage-1 digest pins them all.
