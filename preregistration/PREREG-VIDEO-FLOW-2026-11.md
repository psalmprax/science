# PREREG-VIDEO-FLOW-2026-11: JetFlow, plain flow matching and MeanFlow at equal training time

**Status:** frozen on 26 November 2026 by `preregistration/DIGEST-VIDEO-FLOW.json`, which pins this file,
`scripts/run_prereg_video_flow.py`, `packages/jetflow-video/jetflow_video/head_to_head.py`,
`.github/workflows/video-flow.yml`, `tests/test_video_flow_head_to_head.py`, the JetFlow package files it uses
(`__init__.py`, `model.py`, `stepper.py`, `noether.py`, `physics.py`, `diffusion_baseline.py`, `bridge.py`,
`tensor_train.py`, `text_encoder.py`, `depth_se3.py`, `trainer.py`) and the toy worlds and split rule
(`apedp_dyn/video/fluid_domain.py`, `collision_domain.py`, `eval_gate.py`), after the independent review below
and its second look; every job also records the commit it ran. Runs once, on GitHub, by pushing the frozen
commit to a branch `video-flow-run-1`; every job checks the digest before it reads anything. Owner's words:
"yes build it" (26 November, 19:07) and "fix the bug in jetflow" (26 November, 21:56).

## Why

On 26 November the owner asked whether JetFlow (`packages/jetflow-video`) is a better approach to
training a video model, and, if not, whether something else is. The answer then, from the repository's
own records: not shown. JetFlow had been measured only on ten 16 x 16 toy clips; on the collision world a
diffusion model of the same size won the preregistered confirmation; on the fluid world it was a tie; its
training-time claim compared baselines whose times differed tenfold. The alternative named was MeanFlow
(Geng et al., 2025), a flow-matching objective that learns the average velocity over an interval, so that
one network call goes from noise to data. The owner then said "yes build it". This is that test, on more
clips, three seeds and real footage, with the comparison fixed before the run.

## What is compared

`packages/jetflow-video/jetflow_video/head_to_head.py`. All arms share one latent space (the fixed wavelet
autoencoder, no learned parameters, which round-trips a clip exactly), one clip shape (six frames,
16 x 16, RGB), one clip per training step (JetFlow's backbone is only correct at batch one), AdamW at
1e-3, and the same training time. Every model sees the same constant zero prompt in training and in
sampling (an unconditional model, guidance off).

| Arm | Network | Objective | Samplers scored (network calls per clip) |
| :--- | :--- | :--- | :--- |
| `fm-jetflow` | `JetFlowVideoModel`, hidden 48, two layers (210,637 parameters, of which its forward uses the backbone's 135,580) | flow matching, t uniform in (0, 1) | Euler 1, 2, 4, 8, 32; JetFlow's sampler (its Jet2 step and its latent velocity projection; its steering is the identity at default settings), 1, 2, 4 steps = 2, 4, 8 calls |
| `fm-unet` | the compact video UNet, width 40, matched to JetFlow's backbone (152,492) | flow matching, as above | Euler 1, 2, 4, 8, 32; the Jet2 step alone, 1, 2, 4 steps; **hybrid:** JetFlow's sampler on the UNet, 2 steps = 4 calls |
| `meanflow-unet` | the same UNet with the gap t - r embedded beside t (154,812) | MeanFlow: u(z_t, r, t) against sg(v - (t - r) du/dt), du/dt by one forward-mode product along (v, 0, 1); (t, r) logit-normal(-0.4, 1), r = t on 75 % of steps; weight sg(1 / (squared norm of the error + 0.001)), the paper's adaptive weight with power 1 | MeanFlow 1, 2, 4 steps = 1, 2, 4 calls; Euler on u(z, t, t) 8, 32; **hybrid:** the Jet2 step on u(z, t, t), 2 steps = 4 calls |
| `meanflow-jetflow` (**hybrid**) | JetFlow's network; the gap t - r reaches it the way it reaches the UNet, added to the time embedding (a forward hook on the backbone's time embedding adds a Fourier embedding of the gap through a layer that starts at zero), so JetFlow's code is unchanged | MeanFlow as above, du/dt by a finite difference along (v, 0, 1) with step 0.001 (the target is not differentiated, so no forward-mode product is needed) | MeanFlow 1, 2, 4 steps; Euler 8, 32 |

**JetFlow's step had its curvature term reversed, and is fixed.** From 6 November (commit 99f15f2) to 26
November `JetFlowStepper` used -dv/dt in its second-order step (found by this plan's independent review:
on dz/dt = z one step from t = 1 landed at -0.5 z, where the exact answer is z / e and Euler gives 0).
On the owner's word of 26 November (21:56) it was fixed in the package before this plan was frozen
(`docs/ERRATUM_JETFLOW_JET2_SIGN_2026-11-26.md`; pinned by `tests/test_jetflow_stepper_curvature.py`). This
run scores the fixed step; no recorded JetFlow result was measured with the reversed one. The three
rehearsals below were, and say so.

## Data

- **fluid** and **collision**: the repository's toy worlds (`apedp_dyn/video/fluid_domain.py`,
  `collision_domain.py`), clips 0 to 499 split by the manifests' rule (seed 13): 400 to train on, 100
  held out. Development used clips 3000 to 3499 (split with seed 14), never these.
- **ucf101**: real footage, the UCF101 subset on the Hugging Face Hub (`sayakpaul/ucf101-subset`, ten
  action classes), up to four windows of six frames from each video, every third frame, resized to
  16 x 16 by area. The run trains on the subset's train split and scores on its test split. Development
  used the train split alone, holding out its groups 8 to 10; nothing has read the test split. The
  archive's SHA-256 is pinned in the runner from the rehearsal.

## Protocol

Each of the 36 jobs (four arms, three data sets, seeds 0, 1 and 2) runs alone on one GitHub runner:
train for 1,200 seconds, then generate 256 clips with every sampler of the arm's grid, from the same
noise for every arm and sampler, and decode them. (JetFlow's sampler is run from that shared noise; a test
pins that it gives exactly what `generate_video` gives from the same noise.)

**Quality** is the Frechet distance (FD) between the generated clips and the held-out real clips, in the
first 16 principal components of the training clips. Its **floor** is the FD of 256 training clips (the
same 256 for every arm and seed) to the held-out ones (what a model that reproduces its training distribution exactly would score). Reported
beside it: a **copying ratio** (mean distance of a generated clip to its nearest training clip, over the
same for held-out clips; well below 1 means copying), **temporal roughness** (mean second difference in
time) against the held-out clips' own, network calls and seconds per clip, training steps.

## Expectations, fixed now

"A beats B" means: over the three seeds, A's mean FD is at most 0.9 times B's, **and** A's FD is lower
than B's on at least two of the three seeds (paired by seed). Each expectation is read on each data set.

1. **JetFlow beats plain flow matching at four calls:** `fm-jetflow` with its own sampler at two steps
   (four calls) beats `fm-unet` with Euler at four steps.
2. **MeanFlow beats plain flow matching at one call:** `meanflow-unet` at one step beats `fm-unet` with
   Euler at one step.
3. **MeanFlow at one call is no worse than JetFlow at four:** `meanflow-unet`'s mean FD at one step is at
   most 1/0.9 times `fm-jetflow`'s at two steps of its own sampler.
4. **The Jet2 step beats Euler at four calls on the same network:** on `fm-unet` (Jet2, two steps, against
   Euler four) and on `fm-jetflow` (its own sampler, two steps, against Euler four), read separately.
5. **JetFlow's network beats the UNet with the same sampler:** `fm-jetflow` with Euler at 32 steps beats
   `fm-unet` with Euler at 32.
6. **Hybrids** (on the owner's word of 26 November, 20:32: "test hybrid approach in all the suggested
   solutions"), each read on its own:
   a. MeanFlow helps JetFlow's network: `meanflow-jetflow` at one step beats `fm-jetflow` with Euler at
      one step;
   b. MeanFlow on JetFlow's network beats MeanFlow on the UNet at one step;
   c. the Jet2 step on MeanFlow's velocity, two steps (four calls), beats MeanFlow's own four steps;
   d. JetFlow's sampler on the UNet, two steps (four calls), beats Euler at four steps on the UNet.

A miss is reported as a miss. A job that fails is reported as missing; each expectation that needs it is
then marked not read, and the others are read.

## What each outcome proposes (the owner decides; nothing goes in force, AGENTS.md 1.5)

"Holds where it counts" means on ucf101 and on at least one toy world. Something that holds on ucf101
alone is reported and proposes a larger test of it, not its adoption; something that holds on toy worlds
only is reported, not proposed.

- **1 holds where it counts:** propose continuing to build the video work on JetFlow. **1 fails on
  ucf101:** propose not building the video work on JetFlow's approach; if 4 or 5 holds where it counts,
  propose keeping only that part (the Jet2 step, or the network).
- **2 and 3 hold where they count:** propose MeanFlow as the few-step training objective for the video
  package, under its own preregistration at a larger size.
- **6a and 6b both hold where they count:** propose `meanflow-jetflow` (JetFlow's network trained by
  MeanFlow) in place of both `fm-jetflow` and `meanflow-unet`. **6c:** propose the Jet2 step in
  place of MeanFlow's own multi-step sampler at four calls. **6d:** propose JetFlow's sampler in place of
  Euler for the UNet at four calls. Each under its own preregistration at a larger size.
- **Nothing above holds:** plain flow matching with a many-step sampler stays the reference; nothing new
  is proposed.

## Disclosed before freezing

- **Development numbers (local, two cores, 15-second and 8-second runs, 16 to 24 clips per sampler).**
  Fluid, 80 training clips, 15 s: FD with JetFlow's network 116.5 (Euler 2) and 133.5 (its own sampler, 2
  steps); the UNet 92.3 (Euler 2) and 148.3 (Jet2, 1 step); MeanFlow 107.0 (1 step) and 70.9 (Euler 4).
  Collision, 400 clips, 8 s: the floor 1.20; JetFlow's network 1.58 to 1.75; the UNet's Jet2 4 steps 2.26.
  These are too short to say anything; they show the pipeline runs.
- **Choices made:** the clip shape and JetFlow's size are those of the earlier matched benchmark; the
  UNet's width is now matched to JetFlow's backbone (the earlier match counted a text encoder and a depth
  engine JetFlow's forward does not use, which made the UNet 1.6 times larger); the learning rate is the
  repository's (1e-3 for every arm, not tuned); the MeanFlow settings are the paper's (its adaptive weight
  on the squared norm of the error, power 1); 16 principal components; 256 clips per sampler.
- **Rehearsal** on the development splits, the 27 jobs then planned, 120 seconds of training and 64 clips
  per sampler (run 45875359514; its collect job failed for a missing package, `tqdm`, which the jobs had
  and the collect job did not; the 27 records were read from the jobs' logs and collected here,
  `benchmarks/video/video_flow_rehearsal.json`). UCF101 subset: 300 videos in its train split, 36 of them
  in groups 8 to 10; 1,056 training and 144 held-out clips; archive SHA-256 e9fcc76a...71eb, now pinned in
  the runner. Mean FD over three seeds (floor): fluid (19.0) JetFlow's sampler 2 steps 114.9, the UNet's
  Euler 4 37.4, MeanFlow 1 step 69.1 against Euler 1 75.2; collision (0.69) 2.67, 1.96, 2.23 against 2.23;
  ucf101 (34.0) 166.9, 89.0, 180.9 against 140.6. Read as the plan reads: 1 missed on all three, 2 missed
  on all three (on fluid lower on 2 of 3 seeds but not by 10 %), 3 held on all three (because JetFlow's
  sampler did badly), 4 missed for both networks on all three (the Jet2 step was worse than Euler at
  equal calls), 5 missed on fluid and ucf101 (on collision lower on 2 of 3 seeds, not by 10 %); on
  collision 2 was also lower on 2 of 3 seeds. Over three seeds the MeanFlow-trained UNet with Euler at 8
  calls did not beat the flow-matching UNet at 32 (fluid 59.0 against 33.2, ucf101 91.6 against 76.0,
  collision 1.53 against 1.59). These rehearsal numbers come from the code before the review's changes
  (the packaged step only, the larger UNet, t in [0.05, 0.95]). Jobs took 3.5 to 4.5 minutes; the final run's 20
  minutes of training and four times the clips come to about 30, inside the 70-minute step limit.
- **Changes after the rehearsal, before freezing.** The hybrid arm `meanflow-jetflow`, the hybrid
  samplers and expectation 6, on the owner's word of 26 November (20:32). Then, on the independent review
  (below): the corrected Jet2 step beside the packaged one; the hybrid's gap through a channel shift (the
  token route got no gradient); JetFlow's sampler from the shared noise; the UNet matched to JetFlow's
  backbone; flow matching on t in (0, 1); MeanFlow's weight on the squared norm; the floor on the same
  clips for every seed; each expectation read on its own; the consequences tightened; the frozen list
  widened and each job's commit recorded. The collect job now installs the jobs' packages; the archive's
  SHA-256 is pinned. After the review's second look: the gap through the time embedding, seeded starting
  weights, the test's float tolerance. A rehearsal of all 36 jobs runs this code before freezing
  (`video-flow-smoke-4`; the second stopped on the test's tolerance, the third was cancelled for the
  second look's changes). Then the owner's word to fix JetFlow's step (21:56): the step is fixed in the
  package; the comparison's own corrected copy of it and the duplicate samplers it fed are removed; every
  expectation reads the fixed step. A last rehearsal of this code runs before freezing
  (`video-flow-smoke-5`). All rehearsals before it used the reversed step.
- **Last rehearsal**, of the frozen code with the fixed step (run 45895464640, commit c9f53c6, development
  splits, 120 seconds of training, 64 clips per sampler; `benchmarks/video/video_flow_rehearsal5.json`). All
  36 jobs and the collect job ran; no FD is NaN; at most 77 seconds of generation per job for 256 clips. The
  hybrid trained: its MeanFlow loss sits at 1.0, as the adaptive weight makes it, and its one-call FD differs
  from its Euler rows on every job (collision seed 0: 0.85 against 1.90 at 8 calls). Mean FD over three seeds
  (floor): fluid (15.2) JetFlow's sampler 2 steps 64.5, the UNet's Euler 4 37.5, MeanFlow 1 step 44.4 against
  Euler 1 82.3; collision (0.61) 2.24, 1.93, 2.10 against 2.13; ucf101 (35.1) 96.8, 127.4, 129.9 against
  163.1. The fixed Jet2 step against Euler at four calls: on JetFlow's network lower in the mean on all three
  data sets (fluid 64.5 against 67.1, lower on 2 seeds; collision 2.24 against 2.33, 3 seeds; ucf101 96.8
  against 109.3, 3 seeds; held by the plan's 10 % margin on ucf101 only); on the UNet lower on ucf101 only
  (119.5 against 127.4, 3 seeds) and higher on fluid (38.5 against 37.5) and collision (2.01 against 1.93).
  Read as the plan reads, these development numbers would hold expectation 1 on ucf101 only; 2 on fluid and
  ucf101; 3 on fluid and collision; 4 for JetFlow's network on ucf101; 5 on ucf101; 6a on all three; 6b on
  collision; 6c and 6d nowhere. They are rehearsal numbers, not the result.

## Limits

- 16 x 16 pixels and six frames, and networks of 135,580 to 154,812 parameters: a test of the approaches at toy
  size, on CPU, for 20 minutes each. Nothing here says how they compare at production size.
- The Frechet distance in principal components of the training clips is not FVD (no I3D network); at
  this resolution an I3D network would see almost nothing. It measures how close the generated clips'
  distribution is to the held-out one in the directions the data varies most.
- One learning rate for every arm; MeanFlow's forward-mode product makes its step about 2.6 times dearer,
  so in equal time it takes fewer steps (reported).
- On the collision world the floor is close to every arm's score, so differences there may be small; on
  the fluid world the floor moved from 15.2 to 24.2 between seeds in the rehearsal from the choice of
  clips alone (now fixed: the same clips for every seed).
- JetFlow's latent velocity projection, with the wavelet setting the network uses, damps each frame's
  per-channel mean drift (by a quarter) rather than projecting onto divergence-free fields; it is scored
  as it is.

## Review (AGENTS.md 1.6)

Before freezing, a separate reviewer that did not take part checks this plan against the runner: the arms,
the margins, the seeds, what `expectations` in `scripts/run_prereg_video_flow.py` computes, and the
consequences. Its findings, and what was done about each, are written here. Before the result is recorded,
a reviewer checks every number of the record against the raw outputs. The rule on traps concerns blind
tests of a check; this compares training approaches on fixed data and tests no check, so it has no trap
arm, and says so.

**Findings of the review before freezing (26 November), and what was done:**

1. *Blocks.* JetFlow's Jet2 step has its curvature term reversed (-dv/dt), worse than Euler on a curved
   flow. *Done:* first scored as packaged and corrected side by side; then, on the owner's word, fixed in
   the package (see above), so the run scores the fixed step only.
2. *Blocks.* `meanflow-jetflow` could not learn the gap (zero-started layers on JetFlow's token path gave
   no gradient). *Done:* the gap reaches it as a channel shift; a test from the real start shows the gap
   matters after training.
3. The frozen list was too narrow. *Done:* the package files, the toy worlds, the split rule and the test
   file are frozen; each job records its commit.
4. One failed job would cancel every expectation. *Done:* each is read on its own.
5. "Solenoidal projection" was the wrong name for what the wavelet setting does. *Done:* renamed, and
   described under Limits.
6. The parameter match favoured the UNet (1.6 times JetFlow's backbone). *Done:* matched to the backbone.
7. The rehearsal's extra observation rested on one seed and did not hold over three. *Done:* replaced
   by the three-seed means.
8. The hybrid consequence was loose. *Done:* 6a and 6b together for `meanflow-jetflow`; each hybrid names
   what it would replace; "holds on ucf101 only" has a consequence.
9. JetFlow's sampler drew its own noise; flow matching trained on [0.05, 0.95] while samplers start at
   t = 1; MeanFlow's weight used the mean; the floor moved with the seed. *Done:* shared noise (a test
   shows it equals `generate_video`); t in (0, 1); the squared norm; the same floor clips for every seed.
10. MeanFlow, the metric and the expectations' arithmetic: no finding.
11. Data: no finding; the run now records the UCF101 groups it trained and scored on.
12. The changed code had not run on GitHub. *Done:* a second rehearsal before freezing.

**Second look at the fixes (26 November), and what was done:** (1) A new test (JetFlow's sampler against
`generate_video`) failed on the runners at a tolerance of 1e-5 (float32 rounding through the probe's
division by 0.001), which stopped five jobs of the second rehearsal before training. *Done:* compared at
1e-3; a third rehearsal. (2) The channel shift mixed the gap with each channel's mean, a weaker way in
than the UNet's. *Done:* the gap is added to JetFlow's time embedding through a forward hook, as for the
UNet; the test trains at the run's settings. (3) The starting weights were not seeded. *Done:* seeded
before the model is built. (4) Four imported package files were not frozen, and NumPy's and OpenCV's
versions not recorded. *Done:* frozen and recorded. (5) A stale parameter count under Limits. *Done.*
The reviewer found the corrected step right (sign, probe, the dt^2/2 factor; exact on dz/dt = t),
JetFlow's sampler from shared noise equal to `generate_video` up to rounding, the expectations, the
consequences and the grid matching the code.
