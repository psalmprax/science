# PREREG-VIDEO-HYBRID-2026-11: MeanFlow on JetFlow's network against its parents, larger

**Status:** draft, not frozen. Written 27 November 2026 on the owner's word "go ahead to test all" (02:32
Berlin time), which answered the proposal of `RESULT-VIDEO-FLOW-2026-11`: a larger preregistered test of its
hybrid. It is frozen by `preregistration/DIGEST-VIDEO-HYBRID.json` after the independent review and a
rehearsal, and run once on a branch `video-hybrid-run-1`.

## Why

In the first comparison (`RESULT-VIDEO-FLOW-2026-11`) JetFlow's network trained by MeanFlow (the hybrid
`meanflow-jetflow`) beat both its parents at one network call on real footage (58.25 against 144.21 for
MeanFlow on the UNet and 142.41 for JetFlow's flow matching) and on the collision world. That run was small
(6 frames of 16 x 16, 20 minutes of training), its real-footage split shared actors and scenes between training
and scoring, one MeanFlow seed scored badly, and the hybrid's clips flickered (roughness 0.144 against 0.047 for
plain flow matching). This test asks whether the hybrid's lead survives a larger size, a clean split and a
flicker guard, and whether it can also match plain flow matching at four calls.

## What is compared

The models and samplers of `packages/jetflow-video/jetflow_video/head_to_head.py` (frozen by
DIGEST-VIDEO-FLOW, not edited). `scripts/run_prereg_video_hybrid.py` sets that module's clip size before it
builds anything: **8 frames of 32 x 32** (was 6 of 16) and **JetFlow's network 64 wide** (was 48); the UNet is
matched to JetFlow's backbone as before.

| Arm | What it is | Samplers (network calls per clip) |
| :--- | :--- | :--- |
| `meanflow-jetflow` | the hybrid: JetFlow's network trained by MeanFlow | MeanFlow 1, 2, 4 steps (1, 2, 4); Euler 32 |
| `meanflow-unet` | MeanFlow on the UNet (a parent) | MeanFlow 1, 2, 4; Euler 32 |
| `fm-jetflow` | flow matching on JetFlow's network (a parent) | Euler 1, 4, 32; JetFlow's sampler 2 steps (4) |
| `fm-unet` | plain flow matching on the UNet (the reference) | Euler 1, 4, 32 |

## Data

- `fluid` and `collision`, the repository's toy worlds: final clips 500 to 999, development 1000 to 1499, each
  split 80/20 by the repository's manifest split (seeds 15 and 16). Neither range was used by the first test.
- `ucf101`, the UCF101 subset on the Hugging Face Hub (SHA-256 pinned as before), every third frame, area
  resized to 32 x 32: both of its splits are pooled and split **by group**, so no actor or scene is on both
  sides. Final: train on groups 1 to 19, score on groups 20 to 25. Development: train on 1 to 15, score on 16 to
  19; development never reads groups 20 to 25.

## Protocol

Each runner trains one arm on one data set with one seed (0, 1 or 2) for **2,700 seconds** on four threads,
from weights drawn from the seed, then generates 256 clips with each sampler of its grid (noise seed 20,000
plus the seed) and scores them against the held-out real clips: Fréchet distance in the first 16 principal
components of the training clips (FD), with the copying ratio and temporal roughness beside it. The floor is the
FD of a fixed set of 256 training clips against the held-out clips. 36 jobs; a job that fails is missing and the
expectations that need it are marked not read.

## Expectations, fixed now

"A beats B": over the three seeds A's mean FD is at most 0.9 times B's, and A's FD is lower than B's on at
least two of the three seeds (paired by seed). Each expectation is read on each data set.

1. **H1: the hybrid beats MeanFlow on the UNet at one call** (MeanFlow 1 step against MeanFlow 1 step).
2. **H2: the hybrid beats JetFlow's flow matching at one call** (MeanFlow 1 step against Euler 1).
3. **H3: the hybrid at four calls beats plain flow matching at four calls** (MeanFlow 4 steps against the
   UNet's Euler 4).
4. **H4: the hybrid's clips at four calls are at most 1.5 times as rough as plain flow matching's at four
   calls** (mean temporal roughness over the seeds; a guard against flicker).

## What each outcome proposes (the owner decides; nothing goes in force, AGENTS.md 1.5)

"Holds where it counts" means on ucf101 and on at least one toy world; something that holds on ucf101 alone
proposes a further test, not adoption; something that holds on toy worlds only is reported.

- **H1 and H2 hold where they count, and H4 holds on ucf101:** propose the hybrid in place of both its parents
  in the video package (`packages/jetflow-video`), as its few-step model.
- **H3 also holds where it counts (with H4 on ucf101):** propose the hybrid as the video package's model at four
  calls, in place of plain flow matching.
- **H1 or H2 fails on ucf101:** the hybrid is retired, by the owner's standing word of 27 November ("retire
  anything that does not meet our expectation").
- **H4 fails on ucf101:** no adoption is proposed whatever H1 to H3 say; the flicker is reported.

Nothing else in APEDP-X uses the video package; no route of the referee changes either way.

## Limits

CPU only, small clips and short training: a result here says which training objective and network to carry
into a larger build, not that any of them makes good video. Equal wall-clock on runners of different speed
gives different numbers of steps; each job records its steps and its processor.

## Disclosed before freezing

- The comparison module's clip size is set by this runner at run time (its module-level `FRAMES`, `SIZE` and
  JetFlow's width are reassigned before anything is built); the module's file is not changed. The tests check
  that every arm builds, trains and samples at the new size.
- The noise seeds differ from the first test's (20,000 plus the seed, there 10,000 plus the seed), and the toy
  clips are new, so no clip of the first test is reused.
