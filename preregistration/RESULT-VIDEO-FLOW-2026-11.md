# Result: PREREG-VIDEO-FLOW-2026-11

**Run:** once, 2026-11-26 22:27 – 23:49 UTC, GitHub Actions run `45904875839` on the branch `video-flow-run-1`
at commit `2f38f99`, which froze the plan (`preregistration/DIGEST-VIDEO-FLOW.json`, 19 files). All 36 jobs and
the collect job succeeded on the first attempt; no job is missing; no Fréchet distance is NaN. Every job record
carries commit `2f38f99` and the 19 frozen digests, identical to the digest file. Final mode: 1,200 seconds of
training per job, 256 generated clips per sampler; toy worlds 400 training and 100 held-out clips (clips 0 to
499), the UCF101 subset 1,200 training clips (its train split) and 300 held-out clips (its test split).

Outputs, unedited, in [`RESULT-VIDEO-FLOW-2026-11/`](RESULT-VIDEO-FLOW-2026-11/): `results.json` (the collect
job's record, SHA-256 40f6759c...) and `jobs/` (the 36 job files `job_<data>_<arm>_s<seed>.json`, the bytes
each job printed gzipped in its log, decoded; each equal in content to its entry in `results.json`).

## Outcome

Fréchet distance in the first 16 principal components of the training clips (FD; lower is better), mean over
seeds 0, 1 and 2 (per-seed values in brackets where they matter). The floor is the FD of a fixed set of 256
training clips against the held-out clips: what real clips score.

| Model and sampler (network calls per clip) | fluid | collision | ucf101 |
| :--- | ---: | ---: | ---: |
| floor | 8.89 | 0.50 | 24.82 |
| flow matching, UNet, Euler 1 (1) | 101.22 | 1.775 | 97.78 |
| flow matching, UNet, Euler 4 (4) | 30.49 | 1.364 | 38.72 |
| flow matching, UNet, Euler 32 (32) | 27.82 | 0.958 | 32.20 |
| flow matching, UNet, Jet2 2 steps (4) | 32.23 | 1.475 | 38.83 |
| flow matching, UNet, JetFlow's sampler 2 steps (4) | 32.34 | 1.463 | 38.78 |
| JetFlow, Euler 1 (1) | 138.16 | 1.959 | 142.41 |
| JetFlow, Euler 4 (4) | 56.21 | 1.392 | 67.96 |
| JetFlow, its sampler 2 steps (4) | 51.39 | 1.218 | 57.97 |
| JetFlow, Euler 32 (32) | 39.76 | 0.953 | 53.95 |
| MeanFlow, UNet, 1 step (1) | 27.80 | 1.493 | 144.21 (121.2, 116.5, 195.0) |
| MeanFlow, UNet, 4 steps (4) | 18.27 | 0.877 | 92.20 (51.0, 52.9, 172.8) |
| MeanFlow, UNet, Jet2 2 steps (4) | 35.61 | 1.378 | 68.05 (84.8, 48.3, 71.0) |
| MeanFlow on JetFlow's network (hybrid), 1 step (1) | 42.59 | 1.116 | 58.25 |
| the hybrid, 4 steps (4) | 33.74 | 0.697 | 47.73 |
| the hybrid, Euler 32 (32) | 33.55 | 0.623 | 46.59 |

Per-seed values of every row are in `results.json`. On ucf101 one MeanFlow-UNet seed (seed 2) scored far worse
on its MeanFlow and Euler samplers (195.0 at one step, 172.8 at four, 163 to 166 with Euler 8 and 32; its Jet2
row, 71.0, is in line with the other seeds); it moves that arm's means. Its loss and step count look like the
others', so the records do not say why.

Reported beside FD, as the plan says (all in `results.json`): temporal roughness, against the held-out clips'
own (fluid 0.055, collision 0.029, ucf101 0.026). On ucf101 the hybrid is rough at one call (0.144) and at four
(0.091), where the UNet's Euler 4 is 0.047: its clips flicker more than real footage. The copying ratio (a generated
clip's mean distance to its nearest training clip, over the same for held-out real clips; well below 1 means
copying) averages 1.01, 0.87 and 0.95 on the three data sets; the lowest, 0.56 to 0.65, is JetFlow's Euler 1 on
ucf101. Generation took 0.0025 to
0.16 seconds per clip (one call to 32).

## Expectations

"A beats B": A's mean FD at most 0.9 times B's and lower on at least two of the three seeds. As the runner
computed them (`results.json`, `expectations`), per data set fluid / collision / ucf101:

1. **JetFlow beats flow matching at four calls** (its sampler 2 steps against the UNet's Euler 4):
   missed / **held** / missed (51.39 against 30.49; 1.218 against 1.364, 3 seeds; 57.97 against 38.72).
2. **MeanFlow beats flow matching at one call:** **held** / **held** / missed (27.80 against 101.22; 1.493 against
   1.775; 144.21 against 97.78, no seed lower).
3. **MeanFlow at one call no worse than JetFlow at four** (at most 1/0.9 of it): **held** / missed / missed
   (27.80 against 51.39; 1.493 against 1.218; 144.21 against 57.97).
4. **The Jet2 step beats Euler at four calls on the same network.** UNet: missed on all three (32.23 against
   30.49; 1.475 against 1.364; 38.83 against 38.72). JetFlow: missed / **held** / **held** (51.39 against 56.21,
   lower on all 3 seeds but not by 10 %; 1.218 against 1.392; 57.97 against 67.96).
5. **JetFlow's network beats the UNet, Euler 32:** missed on all three (39.76 against 27.82; 0.953 against 0.958,
   1 seed lower; 53.95 against 32.20).
6. Hybrids:
   a. **MeanFlow helps JetFlow's network at one call** (the hybrid against JetFlow's Euler 1): **held** on all
      three (42.59 against 138.16; 1.116 against 1.959; 58.25 against 142.41).
   b. **MeanFlow on JetFlow's network beats MeanFlow on the UNet at one call:** missed / **held** / **held**
      (42.59 against 27.80; 1.116 against 1.493; 58.25 against 144.21).
   c. **The Jet2 step on MeanFlow's velocity beats MeanFlow's own four steps:** missed / missed / **held**
      (35.61 against 18.27; 1.378 against 0.877; 68.05 against 92.20, lower on seeds 1 and 2: 48.3 against 52.9
      and 71.0 against 172.8, higher on seed 0, 84.8 against 51.0). It holds on ucf101 only because of seed 2:
      on seeds 0 and 1 the Jet2 step is worse in the mean, 66.6 against 51.9.
   d. **JetFlow's sampler on the UNet beats Euler at four calls:** missed on all three (32.34 against 30.49;
      1.463 against 1.364; 38.78 against 38.72).

## Proposal (the owner decides; nothing changes in force, AGENTS.md 1.5)

The plan's consequences, read as written ("holds where it counts": on ucf101 and on at least one toy world;
on ucf101 alone: a larger test, not adoption; on toy worlds only: reported, not proposed):

- **1 fails on ucf101**: propose **not building the video work on JetFlow's approach**. 4 holds where it counts
  for JetFlow's network (ucf101 and collision), so **propose keeping only that part: the Jet2 step**, as
  JetFlow's sampler on its own network. 5 fails everywhere, so not the network on its own.
- **2 and 3 fail on ucf101**: no MeanFlow proposal. 2 held on both toy worlds and 3 on fluid: reported only.
- **6a and 6b both hold where they count** (ucf101 and collision): **propose `meanflow-jetflow`, JetFlow's
  network trained by MeanFlow, in place of both `fm-jetflow` and `meanflow-unet`, under its own
  preregistration at a larger size.**
- **6c holds on ucf101 alone**: propose a larger test of the Jet2 step on MeanFlow's velocity, not its
  adoption. It held there only because of MeanFlow's one bad seed, which weakens even that.
- **6d fails everywhere**: JetFlow's sampler on the UNet is not proposed.

What the plan's consequences do not say, and should be read beside them: on the real footage (ucf101) plain
flow matching on the UNet with Euler at four calls (38.72) is better than every one-call model, the hybrid
included (58.25), and than the hybrid at four calls (47.73); at 32 calls it is the best model there (32.20). On
collision the hybrid is the best at every number of calls (1.116 at one, 0.697 at four, 0.623 at 32 against
flow matching's 1.775, 1.364, 0.958). On fluid MeanFlow on the UNet is the best at every number of calls
(27.80 at one, 18.07 at two, 18.27 at four).

## Deviations and disclosures

No deviation from the protocol. At equal wall-clock the number of training steps varied between jobs (flow
matching on the UNet 103,711 to 177,212; JetFlow 65,150 to 140,078; MeanFlow on the UNet 31,729 to 63,809; the
hybrid 51,824 to 90,395), recorded per job; the records do not name the processor (every job had four
threads), so the cause is not shown. The rehearsals used the development splits; these numbers are the first
on the final splits.

One limit the plan did not state: the UCF101 subset's own train and test splits share groups (the same actors
and scenes): the train split covers groups 1 to 25 and all 19 test-split groups are among them, unlike the
rehearsal's development split, which held groups 8 to 10 out. So on ucf101 the held-out clips are new clips of
familiar groups. This changes no verdict, but it bears on what the ucf101 FD and its floor (24.82) measure.

## Independent check (AGENTS.md 1.6)

A reviewer agent that took no part in the work checked this record cold against `results.json`, the job
records, the frozen plan and the runner. It found no wrong number: every table cell, floor, mean, seeds-lower
count and verdict matches its own recomputation; the runner's `beats()` applies the plan's rule; each proposal
follows the plan's consequences read literally; the run claims (commit, digests, no missing job, no NaN, the
times) hold. Its findings on the first draft, and what was done:

1. 6c was said to hold "through" MeanFlow's bad seed; it holds only because of it (seeds 0 and 1: 66.6 against
   51.9). *Done:* reworded in the expectation and the proposal.
2. "Trained badly" went beyond the records (its loss and steps look normal; only its scores are bad). *Done.*
3. The `jobs/` files had been re-written compactly and renamed, not kept as printed. *Done:* replaced by the
   decoded bytes under the jobs' own names.
4. The readings the plan reports beside FD were missing (roughness against the held-out clips', the copying
   ratio, seconds per clip). *Done:* added under Outcome, with the hybrid's roughness on ucf101.
5. The UCF101 subset's train and test splits share groups, which was not disclosed. *Done:* under
   Deviations and disclosures.
6. Smaller points: step ranges for all four arms, no claim about runners the records do not support, fluid's
   MeanFlow best at every number of calls, the floor's definition, the table caption. *Done.*
