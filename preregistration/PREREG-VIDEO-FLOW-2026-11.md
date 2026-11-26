# PREREG-VIDEO-FLOW-2026-11: JetFlow, plain flow matching and MeanFlow at equal training time

**Status:** draft, not frozen. It will be frozen by `preregistration/DIGEST-VIDEO-FLOW.json`, which pins this
file, `scripts/run_prereg_video_flow.py`, `packages/jetflow-video/jetflow_video/head_to_head.py`,
`.github/workflows/video-flow.yml`, `tests/test_video_flow_head_to_head.py`, the JetFlow package files it uses
(`model.py`, `stepper.py`, `noether.py`, `physics.py`, `diffusion_baseline.py`, `bridge.py`, `tensor_train.py`)
and the toy worlds and split rule (`apedp_dyn/video/fluid_domain.py`, `collision_domain.py`, `eval_gate.py`),
after the independent review below; every job also records the commit it ran. Runs once, on GitHub, by pushing the
frozen commit to a branch `video-flow-run-1`; every job checks the digest before it reads anything.

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
| `fm-jetflow` | `JetFlowVideoModel`, hidden 48, two layers (210,637 parameters, of which its forward uses the backbone's 135,580) | flow matching, t uniform in (0, 1) | Euler 1, 2, 4, 8, 32; JetFlow's sampler (its Jet2 step and its latent velocity projection; its steering is the identity at default settings) as packaged, 1, 2, 4 steps = 2, 4, 8 calls; the same with the step's curvature sign corrected, 1, 2, 4 steps |
| `fm-unet` | the compact video UNet, width 40, matched to JetFlow's backbone (152,492) | flow matching, as above | Euler 1, 2, 4, 8, 32; the Jet2 step as packaged, 2 steps; corrected, 1, 2, 4 steps; **hybrids:** JetFlow's sampler on the UNet, as packaged and corrected, 2 steps = 4 calls |
| `meanflow-unet` | the same UNet with the gap t - r embedded beside t (154,812) | MeanFlow: u(z_t, r, t) against sg(v - (t - r) du/dt), du/dt by one forward-mode product along (v, 0, 1); (t, r) logit-normal(-0.4, 1), r = t on 75 % of steps; weight sg(1 / (squared norm of the error + 0.001)), the paper's adaptive weight with power 1 | MeanFlow 1, 2, 4 steps = 1, 2, 4 calls; Euler on u(z, t, t) 8, 32; **hybrid:** the corrected Jet2 step on u(z, t, t), 2 steps = 4 calls |
| `meanflow-jetflow` (**hybrid**) | JetFlow's network; the gap t - r reaches it as a learned shift of each latent channel (a Fourier embedding of the gap through a layer that starts at zero), so JetFlow's code is unchanged | MeanFlow as above, du/dt by a finite difference along (v, 0, 1) with step 0.001 (the target is not differentiated, so no forward-mode product is needed) | MeanFlow 1, 2, 4 steps; Euler 8, 32 |

**JetFlow's step as packaged has its curvature term reversed.** `JetFlowStepper` computes the acceleration
as (v - v_probe) / h, which is -dv/dt (found by this plan's independent review and pinned by
`tests/test_video_flow_head_to_head.py`: on dz/dt = z one step from t = 1 lands at -0.5 z where the exact
answer is z / e and Euler gives 0). The packaged step is scored as it is, because that is JetFlow as it
stands; the corrected step (`jet2_corrected_step`, the same two calls) is scored beside it, because that is
the method JetFlow means. JetFlow's package is not changed: a fix there waits for the owner's word.

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

1. **JetFlow beats plain flow matching at four calls**, read twice: (as packaged) `fm-jetflow` with its
   sampler as packaged at two steps (four calls) beats `fm-unet` with Euler at four steps; (corrected)
   the same with JetFlow's sampler corrected.
2. **MeanFlow beats plain flow matching at one call:** `meanflow-unet` at one step beats `fm-unet` with
   Euler at one step.
3. **MeanFlow at one call is no worse than JetFlow at four:** `meanflow-unet`'s mean FD at one step is at
   most 1/0.9 times the better of `fm-jetflow`'s two samplers (as packaged, corrected) at two steps.
4. **The corrected Jet2 step beats Euler at four calls on the same network:** on `fm-unet` (corrected
   Jet2, two steps, against Euler four) and on `fm-jetflow` (its corrected sampler, two steps, against
   Euler four), read separately. The packaged step is reported beside them.
5. **JetFlow's network beats the UNet with the same sampler:** `fm-jetflow` with Euler at 32 steps beats
   `fm-unet` with Euler at 32.
6. **Hybrids** (on the owner's word of 26 November, 20:32: "test hybrid approach in all the suggested
   solutions"), each read on its own:
   a. MeanFlow helps JetFlow's network: `meanflow-jetflow` at one step beats `fm-jetflow` with Euler at
      one step;
   b. MeanFlow on JetFlow's network beats MeanFlow on the UNet at one step;
   c. the corrected Jet2 step on MeanFlow's velocity, two steps (four calls), beats MeanFlow's own four
      steps;
   d. JetFlow's corrected sampler on the UNet, two steps (four calls), beats Euler at four steps on the
      UNet.

A miss is reported as a miss. A job that fails is reported as missing; each expectation that needs it is
then marked not read, and the others are read.

## What each outcome proposes (the owner decides; nothing goes in force, AGENTS.md 1.5)

"Holds where it counts" means on ucf101 and on at least one toy world. Something that holds on ucf101
alone is reported and proposes a larger test of it, not its adoption; something that holds on toy worlds
only is reported, not proposed.

- **1 (as packaged or corrected) holds where it counts:** propose continuing to build the video work on
  JetFlow, with its step corrected if only the corrected reading holds (the fix to `stepper.py` itself
  waits for the owner's word). **1 fails on ucf101 both ways:** propose not building the video work on
  JetFlow's approach; if 4 or 5 holds where it counts, propose keeping only that part (the corrected Jet2
  step, or the network).
- **2 and 3 hold where they count:** propose MeanFlow as the few-step training objective for the video
  package, under its own preregistration at a larger size.
- **6a and 6b both hold where they count:** propose `meanflow-jetflow` (JetFlow's network trained by
  MeanFlow) in place of both `fm-jetflow` and `meanflow-unet`. **6c:** propose the corrected Jet2 step in
  place of MeanFlow's own multi-step sampler at four calls. **6d:** propose JetFlow's corrected sampler in
  place of Euler for the UNet at four calls. Each under its own preregistration at a larger size.
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
  SHA-256 is pinned. A second rehearsal of all 36 jobs runs this code before freezing (`video-flow-smoke-2`).

## Limits

- 16 x 16 pixels and six frames, and models of about 210,000 parameters: a test of the approaches at toy
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
   flow. *Done:* the packaged step is scored as it is and a corrected step beside it (1, 3, 4 and 6 read
   on the corrected one, 1 on both); a test pins both; the package is not changed (owner's word).
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
