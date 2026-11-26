# Plan: physics-first video, step 1 and its first gates

**Status:** plan, 26 November 2026; part 1a in development since the owner's word "so, build" (22:45 Berlin
time); see "Development of part 1a" at the end. Nothing in it is frozen or in force. The owner asked for it
("yes let us try in that direction", 22:15 Berlin time) after the answer to "if you are to make it top notch,
what will you do with it differently". It becomes a preregistration once its code exists and an independent reviewer has
checked it (AGENTS.md 1.6). Every number marked *proposed* is a new named threshold: it needs the owner's
word, a pin and a `CHANGELOG.md` entry before it is used (AGENTS.md 1.3), and none is borrowed from another
test.

## The aim of the whole line

An open video model that obeys simple physics, with a referee that can show it does. Not a race with the
large labs on speed or picture quality, where a small model trained from scratch cannot win (the JetFlow
rehearsals showed that much on toy clips).

## Step 1 in one sentence

Put a small physics adapter on an open pretrained model (Wan2.1-T2V-1.3B, Apache 2.0, about 8 GB of GPU
memory to run), train it on physics footage, and test whether it raises a physics pass rate measured by a
judge that has itself been shown to work, without making the pictures worse.

Step 1 has three parts, run in order. Each has its own stop rule.

## 1a. The judge, and gate J (CPU, before any GPU work)

A gate measured by an unchecked judge proves nothing, and a reward taken from it later would be gamed.
So the judge is built and tested first.

**Scenes.** Six kinds, each with a law that can be read from one object's track: free fall, a throw
(projectile), rolling down a slope, a bounce, a pendulum, and a head-on collision of two balls.

**What it does with a clip.**

1. *Track.* A text-prompted segmenter (for example Grounding DINO plus SAM2) finds the named object in each
   frame and records its centroid and size. Tracking runs where a GPU is (Colab); only the tracks are kept.
2. *Check the camera.* The background's motion is tracked too. A clip whose camera moves more than a
   *proposed* limit, or whose object moves less than a *proposed* minimum, is marked untestable.
3. *Test the law.* On GitHub's CPU runners, the APEDP-X referee fits the track (time against position) with
   the scene's law as the hypothesis and named competitors as rivals: constant acceleration in the vertical
   and constant velocity in the horizontal for a throw; a speed reversal with lower speed after each bounce;
   a constant period for a pendulum; equal momentum before and after a collision, up to an unknown mass
   ratio that must stay the same across the clip. It tests the law's form, not its constants: pixel scale
   and frame rate are unknown, so a fall must show constant acceleration, not 9.81 m/s².
4. *Verdict.* Pass, fail or untestable, with the reason. For every gate, untestable counts as not passing,
   so a generator cannot gain by making clips the judge cannot read; the untestable rate is reported
   beside every pass rate.

**Validation data.**

- True physics: PhysInOne clips of the six phenomena (a published synthetic set of 2 million physics videos;
  its licence is checked before anything is downloaded). A development set and a separate blind set, split
  by scene.
- Broken physics, made from the same clips: time reversal of clips that lose energy, a jump of the object
  between two frames, frames held still, the vertical motion reversed for part of a clip. For the blind set,
  the independent reviewer also writes perturbations the judge was not developed against, and keeps them
  from the builder until the judge is frozen (AGENTS.md 1.6).
- Human labels: the released human judgements of PhyGround (13 laws, 459 annotators) for the solid-body
  laws the judge covers; if those videos are not released with the labels, VideoPhy-2's released
  human-labelled videos instead.

**Gate J (proposed numbers).** On the blind set: at least 80% of true-physics clips pass; at most 10% of
broken clips pass, counted separately for every kind of breakage (the reviewer's unseen kinds included);
and on the human-labelled videos, agreement with the human majority of at least Cohen's kappa 0.3.

**Stop rule.** If the judge fails gate J, step 1 stops there and the result is recorded; 1b and 1c do not
run until a revised judge passes a fresh blind set.

**Exploratory, not a gate: a tensor-train fingerprint for fluids.** Real fluid velocity fields compress well
into tensor trains (matrix product states), and the compression keeps turbulence's signatures. In
development only, the judge computes the optical flow of fluid clips, compresses it as a tensor train, and
records the rank needed for a fixed error, for real clips and for generated ones. If real and generated
fluids separate cleanly, that becomes a candidate fluid check for a later step, with its own blind test.

## 1b. The base model's baseline (GPU)

**Prompts.** 120 prompts, 20 for each scene kind, written before anything is generated: 60 for development
and 60 held out. The held-out prompts' hash goes in the preregistration; their text is read only by the
final run.

**Generation.** Wan2.1-T2V-1.3B as published, with its own sampler, at one fixed size and length chosen in
the rehearsal (480p and about two seconds as the starting point), two seeds per prompt.

**Scores.** The judge's pass rate; picture quality (VBench's imaging-quality and subject-consistency
scores); and prompt adherence (a text-video similarity score). The same three are read for every arm in 1c.

## 1c. The adapters, and gate 1 (GPU)

**Training data.** About 2,000 PhysInOne clips of the six phenomena with their captions, from scenes that
appear in neither the judge's blind set nor the prompts.

**Arms.** All trained for the same number of steps on the same clips in the same order; each arm's learning
rate is chosen from three values by the flow-matching loss on held-back training clips (no generation, no
judge), so the judge never shapes training.

| Arm | What changes | Why it is there |
| :--- | :--- | :--- |
| A0 | nothing (the base model) | the baseline |
| A1 | LoRA, rank 16, on the self-attention projections of every block | the standard adapter |
| A2 | tensor-train adapter (TT-LoRA) with A1's parameter count | does a tensor train do better at equal size? |
| A3 | TT-LoRA with about a tenth of A1's parameters | does a tensor train keep the gain while much smaller? |
| A4 | hybrid: LoRA on attention, TT-LoRA on the feed-forward layers, A1's total parameter count | the hybrid |
| A5 | A1's LoRA trained on ordinary footage (the UCF101 subset's train split) instead of physics | is the gain physics, or just any fine-tuning? |

The tensor-train adapters reuse `TTLinear` from `packages/jetflow-video/jetflow_video/tensor_train.py` where
it fits. Exact parameter counts are fixed in the build and matched within 1%.

**Gate 1 (proposed numbers).** On the 60 held-out prompts, two seeds each, read pair by pair against A0:

- the best adapter among A1 to A4 raises the physics pass rate by at least 10 percentage points, with an
  exact one-sided McNemar test at p < 0.01 after Holm's correction over the four arms;
- its imaging-quality score falls by no more than 0.02 (on VBench's 0-to-1 scale) and its prompt adherence
  by no more than 0.01;
- its gain beats A5's by at least 5 points. If A5 gains as much, the judge is rewarding something other
  than physics, and that is recorded as a finding about the judge.

**What the arms say about tensor trains.** A tensor train is worth keeping for adapters if A2 or A4 gains at
least as much as A1 at the same size, or A3 keeps at least 80% of A1's gain at a tenth of its size; if
neither, LoRA stays and tensor trains stay out of this line except as an export format.

**What each outcome proposes** (to the owner; nothing changes what is in force without their word): gate 1
met → step 2 (the judge as a reward signal for fine-tuning) is planned; gate 1 missed by every arm → the
result is recorded, and whether to try step 2 anyway is the owner's call.

## What step 1 does not do

- No fine-tuning with the judge as a reward (step 2, only after gates J and 1).
- No one- or two-call sampling with MeanFlow (step 3). JetFlow's corrected Jet2 step joins step 3 as a
  sampler arm only if tonight's head-to-head shows it beats Euler at equal calls.
- No claim beyond the six scene kinds.

## Where it runs and how results come back

- The judge's law test runs on GitHub's CPU runners, like every other referee run, so the verdict is not
  produced inside the GPU notebook.
- Tracking, generation and training run in Colab notebooks kept in the repository (the owner has said
  GPU training is acceptable there). The owner starts them; this workspace cannot
  reach Colab.
- Each notebook writes its record (tracks, scores, the commit, hashes of weights and prompts) to a private
  Hugging Face dataset repository under the owner's account, which is read back through the Hugging Face
  connector and committed.
- Compute, to be measured in a small rehearsal before the run: roughly 25 to 35 hours of an A100-class GPU
  for 1b and 1c together (five adapters with three learning rates each, and about 720 generated clips).
  On a smaller GPU it takes two to three times longer. If it does not fit, seeds are cut before prompts.

## Order

1. Build the judge (1a) and its tests; rehearse on the development set.
2. Independent review; freeze gate J; the reviewer's blind perturbations; one blind run; record.
3. If gate J holds: write 1b and 1c as a preregistration, review, rehearse on five prompts, freeze, run once,
   record with an independent check of every number.

## Development of part 1a (the judge), 26 November

Built: `apedp_dyn/epistemic/physics_judge.py` (the judge), `apedp_dyn/video/physics_perturbations.py` (the four
development perturbations), `apedp_dyn/video/physics_tracks_synthetic.py` (worlds for its unit tests),
`tests/test_physics_judge.py`, and `scripts/run_physics_judge_dev.py` with `.github/workflows/physics-judge-dev.yml`
(development runs on PhysInOne). Every setting is a development number; none is pinned.

On the synthetic worlds (in distribution: the judge was developed on them) 35 of 36 true clips passed and
every applicable perturbation failed. On real PhysInOne tracks it is far from gate J (which asks for 80 % of
true clips passing and at most 10 % of each kind of broken clip): on the first run's 150 development clips,
re-judged after the changes below, 76 true clips pass (falls 10 of 30, throws 22, rolls 23, soft bounces 20,
collisions 1), and of the broken ones 1 of 141 jumps, 7 of 139 holds, 15 of 101 mirrored stretches and 26
of 71 time reversals pass. These numbers are from the development pool and from a judge tuned on it. What the first
development run (`benchmarks/physics_judge/dev_run_1.json`) showed, and what was done:

- **PhysInOne's rigid scenes move on a three-frame cadence**: an object's position changes in one large step
  out of every three frames (a fixed simulation step against the render's 30 frames per second). The judge
  read the small steps as stops and starts. The adapter now takes every third frame, at the phase after the
  large step; the clips are 90 to 210 frames long, so 30 to 70 samples.
- **The first listed static camera often looks steeply down**, so a fall hardly accelerates in the image
  ("no segment shows a downward acceleration" on 10 of 30 falls). The second development run takes the most
  nearly level static camera, as generated clips are usually framed.
- **The phenomenon chosen for bounces, ElasticFall, is a soft body**: its contacts last several frames and
  push upward, where the judge assumes rigid contacts. The owner may prefer rigid bounces instead (from
  VerticalFall or SpringboardRebound scenes).
- **Collisions are seen in three dimensions**: MovingHitsMoving objects move in depth and hide each other,
  and judging their momentum from the image alone failed on nearly every true clip. A judge that lifts the
  tracks to three dimensions (PhysInOne's depth maps for validation, a monocular depth model for generated
  clips) is the likely fix; it is a change to the plan and waits for the owner.
- **Time reversal of a gradual slowdown is not caught**: contact rules see only abrupt changes. A check that
  speed is gained only by descending (depth taken out with the object's size) was tried and dropped: it failed
  most true rolls.
- Made more robust on what the data showed: noise and jump limits relative to the object's size, launches
  and speed gains counted only when substantial, durations in seconds, no "must speed up" rule for rolls
  (a roll receding from the camera slows in the image), collisions paired only when the objects touch.
