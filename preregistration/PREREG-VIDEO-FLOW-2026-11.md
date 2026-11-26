# PREREG-VIDEO-FLOW-2026-11: JetFlow, plain flow matching and MeanFlow at equal training time

**Status:** draft, not frozen. It will be frozen by `preregistration/DIGEST-VIDEO-FLOW.json`, which pins this
file, `scripts/run_prereg_video_flow.py`, `packages/jetflow-video/jetflow_video/head_to_head.py` and
`.github/workflows/video-flow.yml`, after the independent review below. Runs once, on GitHub, by pushing the
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
| `fm-jetflow` | `JetFlowVideoModel`, hidden 48, two layers (210,637 parameters) | flow matching | Euler 1, 2, 4, 8, 32; JetFlow's own `generate_video` (Jet2 step, solenoidal projection, steering) 1, 2, 4 steps = 2, 4, 8 calls |
| `fm-unet` | the compact video UNet, width matched to JetFlow (216,396) | flow matching | Euler 1, 2, 4, 8, 32; the Jet2 step alone 1, 2, 4 steps = 2, 4, 8 calls |
| `meanflow-unet` | the same UNet with the gap t - r embedded beside t (219,564) | MeanFlow: u(z_t, r, t) against sg(v - (t - r) du/dt), du/dt by one forward-mode product along (v, 0, 1); (t, r) logit-normal(-0.4, 1), r = t on 75 % of steps; adaptive weight 1 / (error + 0.001) | MeanFlow 1, 2, 4 steps = 1, 2, 4 calls; Euler on u(z, t, t) 8, 32 |

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

Each of the 27 jobs (three arms, three data sets, seeds 0, 1 and 2) runs alone on one GitHub runner:
train for 1,200 seconds, then generate 256 clips with every sampler of the arm's grid (the same noise
seeds for every arm except JetFlow's own sampler, which draws its own) and decode them.

**Quality** is the Frechet distance (FD) between the generated clips and the held-out real clips, in the
first 16 principal components of the training clips. Its **floor** is the FD of 256 training clips to the
held-out ones (what a model that reproduces its training distribution exactly would score). Reported
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
4. **The Jet2 step beats Euler at four calls on the same network:** on `fm-unet` (Jet2 two steps against
   Euler four) and on `fm-jetflow` (its own sampler two steps against Euler four), read separately.
5. **JetFlow's network beats the UNet with the same sampler:** `fm-jetflow` with Euler at 32 steps beats
   `fm-unet` with Euler at 32.

A miss is reported as a miss. A job that fails is reported as missing; the expectations that need it are
then not read.

## What each outcome proposes (the owner decides; nothing goes in force, AGENTS.md 1.5)

- **1 holds on ucf101 and on at least one toy world:** propose continuing to build the video work on
  JetFlow. **1 fails on ucf101:** propose not building the video work on JetFlow's approach; if 4 or 5
  holds there, propose keeping only that part (the Jet2 step or the network).
- **2 and 3 hold on ucf101:** propose MeanFlow as the few-step training objective for the video package,
  under its own preregistration at a larger size.
- **Neither:** plain flow matching with a many-step sampler stays the reference; nothing new is proposed.

## Disclosed before freezing

- **Development numbers (local, two cores, 15-second and 8-second runs, 16 to 24 clips per sampler).**
  Fluid, 80 training clips, 15 s: FD with JetFlow's network 116.5 (Euler 2) and 133.5 (its own sampler, 2
  steps); the UNet 92.3 (Euler 2) and 148.3 (Jet2, 1 step); MeanFlow 107.0 (1 step) and 70.9 (Euler 4).
  Collision, 400 clips, 8 s: the floor 1.20; JetFlow's network 1.58 to 1.75; the UNet's Jet2 4 steps 2.26.
  These are too short to say anything; they show the pipeline runs.
- **Choices made:** the clip shape, JetFlow's size and the UNet's matched width are those of the earlier
  matched benchmark; the learning rate is the repository's (1e-3 for every arm, not tuned); the MeanFlow
  settings are the paper's defaults; 16 principal components; 256 clips per sampler.
- **Rehearsal** on the development splits, every job, 120 seconds of training and 64 clips per sampler
  (run `video-flow-smoke-1`): *to be written here before freezing.*

## Limits

- 16 x 16 pixels and six frames, and models of about 210,000 parameters: a test of the approaches at toy
  size, on CPU, for 20 minutes each. Nothing here says how they compare at production size.
- The Frechet distance in principal components of the training clips is not FVD (no I3D network); at
  this resolution an I3D network would see almost nothing. It measures how close the generated clips'
  distribution is to the held-out one in the directions the data varies most.
- One learning rate for every arm; MeanFlow's forward-mode product makes its step about 2.6 times dearer,
  so in equal time it takes fewer steps (reported).
- On the collision world the floor is close to every arm's score, so differences there may be small.

## Review (AGENTS.md 1.6)

Before freezing, a separate reviewer that did not take part checks this plan against the runner: the arms,
the margins, the seeds, what `expectations` in `scripts/run_prereg_video_flow.py` computes, and the
consequences. Its findings, and what was done about each, are written here. Before the result is recorded,
a reviewer checks every number of the record against the raw outputs. The rule on traps concerns blind
tests of a check; this compares training approaches on fixed data and tests no check, so it has no trap
arm, and says so.

*Findings: to be written here.*
