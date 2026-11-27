# PREREG-VIDEO-HYBRID-2026-11: MeanFlow on JetFlow's network against its parents, larger

**Status:** frozen on 27 November 2026 by `preregistration/DIGEST-VIDEO-HYBRID.json`, after two independent
reviews, a third check of the fixes and two rehearsals; run once on a branch `video-hybrid-run-1`. Written on
27 November on the owner's word "go ahead to test all" (02:32 Berlin time), which answered the proposal of
`RESULT-VIDEO-FLOW-2026-11`: a larger preregistered test of its hybrid.

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
  resized to 32 x 32: its three splits (train, val, test) are pooled and split **by group**, so no actor or
  scene is on both sides. Final: train on groups 1 to 19, score on groups 20 to 25. Development: train on 1 to
  15, score on 16 to 19; development never reads groups 20 to 25. The split is clean but the footage is not
  fresh: the first test trained and scored on the same videos (its train and test splits, groups 20 to 25
  included), and that is where the hypothesis came from.

## Protocol

Each runner trains one arm on one data set with one seed (0, 1 or 2) for **2,700 seconds** on four threads,
from weights drawn from the seed, then generates 256 clips with each sampler of its grid (noise seed 20,000
plus the seed) and scores them against the held-out real clips: Fréchet distance in the first 16 principal
components of the training clips (FD), with the copying ratio and temporal roughness beside it. The floor is the
FD of a fixed set of 256 training clips against the held-out clips. 36 jobs, ordered seed by seed so that the
four arms of one data set and seed start together.

- **A breakdown is a loss for the arm that broke down, never a win for the other.** A breakdown is a numerical
  failure of the model: its weights are not finite, an eigen-solver does not converge, a score is not finite
  (any `FloatingPointError`, `ArithmeticError` or `LinAlgError`, and a `RuntimeError` or `ValueError` whose
  message names one). Another error (out of memory, a fault in the code) fails the job, which is then missing.
  If training breaks down, every sampler of that job is a breakdown. In a comparison, a seed on which the arm
  under test broke down and its comparator did not fails the expectation; a seed on which the comparator broke
  down is left out, and with fewer than two seeds left the expectation is not read. The record is written after
  training and after every sampler, so a job stopped part way keeps what it finished; Euler at 32 steps, the
  slowest sampler, is last in every grid.
- **Missing** means only that a job is absent, or that `collect` rejects it. In the final run every job must
  record the final mode, the frozen digests, 8 frames of 32 x 32, width 64 and four threads, 2,700 seconds and
  256 clips, 400 and 100 toy clips, the pinned archive digest for ucf101, and digests of every file of this
  repository it loaded that match the checkout (the frozen list cannot name all of them: importing the video
  package loads well over a hundred). All jobs must have run on one commit, the ucf101 jobs must agree on their
  clip counts, and a job with two records is rejected. A job stopped part way is read for the samplers it
  finished; the rest are missing. An expectation that needs something missing is marked not read.
- Each job records its training steps, its processor, a fixed CPU benchmark (seconds for 200 products of
  384 x 384 float32 matrices at four threads), and the versions of Python, PyTorch, NumPy and OpenCV. NumPy,
  OpenCV and the Hub client are pinned in the workflow (2.5.5, 5.1.0.95, 2.5.0).

## Expectations, fixed now

"A beats B": over the seeds where both are finite (at least two), A's mean FD is at most 0.9 times B's, and A's
FD is lower than B's on at least two seeds (paired by seed). "A is no worse than B" is the first test's form: A's
mean FD is at most 1/0.9 times B's over those seeds. Either fails if A broke down on a seed where B did not (see
Protocol). Each expectation is read on each data set.

1. **H1: the hybrid at one call beats MeanFlow on the UNet at one call** (MeanFlow 1 step against MeanFlow 1
   step).
2. **H2: the hybrid at one call is no worse than JetFlow's flow matching at its best sampler with at most four
   calls** (MeanFlow 1 step against whichever of `fm-jetflow`'s Euler 1, Euler 4 and its own sampler at 2 steps
   has the lowest mean FD over its finite seeds, among those finite on at least two seeds). In the draft this compared one Euler step, a known poor sampler; the
   reviewer called that an easy goalpost.
3. **H3: the hybrid at four calls beats plain flow matching at four calls** (MeanFlow 4 steps against the
   UNet's Euler 4).
4. **H4-1 and H4-4: the hybrid's clips at one call, and at four calls, are at most 1.5 times as rough as plain
   flow matching's at four calls** (mean temporal roughness over the seeds where both are finite; a guard against
   flicker, read for each sampler a proposal would name; not read where flow matching broke down on two seeds).

## What each outcome proposes (the owner decides; nothing goes in force, AGENTS.md 1.5)

The rules are read in this order and `collect` computes them (`outcome` in the runner, which records the rules
it applied). "Holds where it
counts" means on ucf101 and on at least one toy world; for the one-call reading, H1 and H2 must both hold on the
same toy world.

| Order | When | Proposal |
| :--- | :--- | :--- |
| 1 | H1 or H2 is read and fails on ucf101 | List the hybrid as retired (deprecated, kept for replication; nothing in force changes), under the owner's standing word of 27 November 00:41 ("retire anything that does not meet our expectation"). Nothing else is read. |
| 2 | Otherwise, H1 or H2 is not read on ucf101 | No one-call proposal; a rerun of the missing or broken-down jobs is put to the owner. Rule 4 is still read. This includes a comparator that broke down on two of the three seeds, even where the hybrid lost the one comparable seed: a comparator that unreliable does not decide the hybrid's fate either way. |
| 3 | Otherwise (H1 and H2 hold on ucf101) | If they also hold together on a toy world and H4-1 holds on ucf101: name the hybrid as the video package's few-step model at one call (its README and `docs/DOCUMENTATION.md` row, experimental). If they hold together on no toy world: a further test is proposed. If H4-1 does not hold: no one-call proposal, and the flicker is reported; if it is not read, that is said. |
| 4 | Unless rule 1 applied | If H3 holds on ucf101 and on a toy world and H4-4 holds on ucf101: name the hybrid as the package's model at four calls. H3 on ucf101 alone: a further test. Otherwise no four-call proposal, saying whether H3 or H4-4 failed or was not read. |

The video package (`packages/jetflow-video`) names no model of its own today, and both parents are already
listed as retired as approaches, so a proposal here changes only what the package's documentation names. Code
elsewhere imports parts of the package (`apedp_dyn/pde/higher_order_engine.py` uses its `tensor_train`), but no
route of the referee reads its models, and none changes either way.

## Limits

CPU only, small clips and short training: a result here says which training objective and network to carry
into a larger build, not that any of them makes good video.

- **Equal wall-clock on unequal runners.** The rehearsal's runners were seven processor models (AMD EPYC 7763,
  9V45 and 9V74; Intel Xeon 8370C, 8573C, 6973P-C and 6975P-C). In the same 240 seconds `fm-unet` made from
  7,535 to 20,339 steps (8,107 to 17,516 on fluid alone), `fm-jetflow` 4,976 to 8,411, `meanflow-unet` 3,457 to
  7,958 and the hybrid 3,842 to 7,463. The first 20 jobs start together in blocks of four arms; the last 16
  start as places free up, so they may not. Which processor a job gets is not controlled. Each job records its
  processor and the benchmark, so a reader can see whether a result follows the processor.
- **The step costs shifted with the size.** At 8 x 32 x 32 the reviewer measured the hybrid's training step
  about 5.7 times slower than at the first test's size and MeanFlow on the UNet's about 2.3 times (single-thread,
  rough), so equal wall-clock moves against the hybrid compared with the first test.
- **The derivatives differ.** MeanFlow's target needs the time derivative of the network's output: on the UNet it
  is exact (`torch.func.jvp`), on JetFlow's network a finite difference (step 0.001), as in the first test's
  frozen module.

## Disclosed before freezing

- The comparison module's clip size is set by this runner at run time (its module-level `FRAMES`, `SIZE` and
  JetFlow's width are reassigned before anything is built); the module's file is not changed. The tests run
  every arm and every sampler of its grid at the new size, and the reviewer checked that nothing captured the old
  values at import.
- The noise seeds differ from the first test's (20,000 plus the seed, there 10,000 plus the seed), and the toy
  clips are new; the real footage is not (see Data).
- **Independent review** (27 November, a reviewer that had not seen this plan written) found ten problems, all
  addressed before freezing: a seed that blows up was excused as missing (now a loss); the outcome rules had gaps
  and a conflict (now the ordered table, computed by `collect`); the flicker guard read four calls only (now one
  and four); H2 compared a known poor sampler (now JetFlow's best at four calls or fewer, no-worse-than form); the
  OpenCV version was not recorded (the reviewer also said the processor name was not, but the rehearsal's records
  name it, so on GitHub's runners the draft's fallback worked); the matrix ran all hybrid jobs last (now interleaved by
  seed, with a benchmark per job); UCF101's val split was left out and the reuse of its footage not said; the
  tests sampled one sampler per arm and `collect` checked nothing; the frozen list missed `apedp_dyn/video/`'s
  package files and `pyproject.toml`, and NumPy and OpenCV were unpinned; and the consequences were worded as acts.
- **The same reviewer checked the revision** and found eight more problems, all addressed before freezing:
  - a comparator's breakdown made its mean infinite and handed the hybrid H1, H2 or H4 (now a lost seed, never a
    win);
  - the outcome read "not read" before "failed", so a missing comparator could rescue a losing hybrid (now a read
    failure on ucf101 retires it first, and Euler 32 is last in every grid);
  - a training breakdown sampled the part-trained model, and every `RuntimeError` counted as a breakdown (now
    every sampler is a breakdown, and only numerical errors count);
  - the code and the table disagreed on when rule 4 is read, and "not read" was reported as "does not hold";
  - H2's new form was chosen after the rehearsal's expectations had been computed and seen (below);
  - the processor count and step spread here were understated;
  - the frozen list still missed files the run loads (now every loaded file's digest is recorded and checked);
  - `collect` did not check the commit, threads, clip size, ucf101 counts or duplicate records.
- **Rehearsal 1** (`video-hybrid-smoke-1`, run 45924026970, the draft runner, development splits, 240 seconds,
  64 clips): all 36 jobs and the collection finished, and the jobs fit the time limits. Its `collect` computed
  the draft's expectations and they were seen before this revision; they are development numbers (other clips,
  other groups, a tenth of the training), not a reading of this test. Two bear on choices made after them:
  - **H2's form.** The hybrid at one call against `fm-jetflow`: 507 against 711 (Euler 1), 505 (Euler 4) and 479
    (its sampler, 2 steps) on fluid; 9.6 against 12.5, 10.8 and 9.4 on collision; 652 against 931, 725 and 636
    on ucf101. Against its sampler the hybrid was slightly worse everywhere, so a "beats" form against
    JetFlow's best would have failed there and the no-worse form chosen holds. The no-worse form is the one the
    first test used, and the reviewer offered both.
  - **H4-1 cuts the other way.** The hybrid's roughness at one call was 0.114, 0.082 and 0.124 against flow
    matching's 0.057, 0.042 and 0.070 at four calls: about twice, so H4-1 failed on all three data sets.
  A second rehearsal runs the revised runner before freezing.
- **A third check by the same reviewer** found the round-two fixes sound and four minor points, fixed: the
  breakdown pattern matched words such as "Inference" (now whole words only); rule 2's wording and its case of
  a comparator broken on two seeds (above); a job stopped part way did not record its loaded files (now written
  after training too); and a record rejected whole for mixed commits or ucf101 counts now says why at the top.
- **Rehearsal 2** (`video-hybrid-smoke-2`, run 45932074892, the revised runner) stopped one job in its tests:
  the grid test counted 0 calls for `fm-jetflow` on one runner. The frozen comparison module keys its call
  counters by `id(model)`, and a freed model's id can be reused by the next model built in the same process,
  which then gets the dead model's counter. A job builds one model, so the run cannot meet it; the test now keeps
  every model alive. The module is frozen by DIGEST-VIDEO-FLOW and is not edited. The other 35 jobs and the
  collection finished. The stopped job (ucf101, the hybrid, seed 1) left H1 to H4 not read on ucf101, so
  `outcome` gave rules 2 and 4 (no one-call proposal, a rerun put to the owner; no four-call proposal, H3 not
  read): the machinery behaved as written. Every record named its processor and OpenCV 5.1.0 and listed 148
  loaded repository files. The benchmark took 0.18 to 0.33 seconds across runners, and the ucf101 jobs agreed on
  988 training and 244 held-out windows (development groups).
