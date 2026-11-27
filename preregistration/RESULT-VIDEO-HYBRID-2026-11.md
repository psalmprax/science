# RESULT-VIDEO-HYBRID-2026-11: MeanFlow on JetFlow's network against its parents, larger

**Status:** recorded on 27 November 2026, after an independent check of every number (below). The preregistered
run of `preregistration/PREREG-VIDEO-HYBRID-2026-11.md` (frozen by `DIGEST-VIDEO-HYBRID.json`) ran once: run
45936568502, branch `video-hybrid-run-1`, commit 5f71ac9. It is the only run on a `video-hybrid-run-*` branch. The
collection's output is kept unedited in
[`RESULT-VIDEO-HYBRID-2026-11/results.json`](RESULT-VIDEO-HYBRID-2026-11/results.json) (SHA-256 5746249d…), with
every job's record.

## What ran

- All 36 jobs finished: four arms, three data sets, three seeds, 2,700 seconds of training, 256 clips per sampler.
  No job is missing, rejected or partial, and no training or sampler broke down.
- Every record:
  - is in final mode;
  - names the frozen digests and commit 5f71ac9;
  - has the plan's clip size, width and threads;
  - passes the collection's checks, including the digest of every loaded repository file.
- The ucf101 jobs agree on 1,232 training windows (308 videos, groups 1 to 19) and 388 held-out windows (97
  videos, groups 20 to 25).

## The expectations, as `collect` read them

FD is the Fréchet distance in 16 principal components, mean over three seeds; lower is better. "Seeds lower" counts
the seeds on which the hybrid's FD was lower. Roughness is the mean absolute second difference in time (temporal
roughness), mean over seeds; the reference in H4 is plain flow matching (`fm-unet`) at four Euler calls.

| Data | H1: hybrid vs MeanFlow-UNet, one call | H2: hybrid vs JetFlow's best ≤ 4 calls | H3: hybrid four calls vs flow matching four | H4-1: roughness at one call (limit 1.5x) | H4-4: roughness at four calls |
| :--- | :--- | :--- | :--- | :--- | :--- |
| fluid | fails: 341.5 vs 270.3, 1 seed lower | holds: 341.5 vs 387.1 (its sampler, 2 steps) | fails: 318.6 vs 196.2, 0 seeds lower | fails: 0.106 vs 0.049 | holds: 0.052 vs 0.049 |
| collision | holds: 5.34 vs 7.56, 3 seeds lower | holds: 5.34 vs 6.21 | holds: 4.40 vs 9.15, 3 seeds lower | holds: 0.039 vs 0.032 | holds: 0.038 vs 0.032 |
| ucf101 | holds: 460.3 vs 552.7, 2 seeds lower | holds: 460.3 vs 573.6 | fails: 374.6 vs 384.2 (0.975 of it, limit 0.9), 2 seeds lower | **fails: 0.164 vs 0.053 (3.1x)** | **fails: 0.141 vs 0.053 (2.7x)** |

The plan's readings beside FD, from the records:

| Reading | fluid | collision | ucf101 |
| :--- | :--- | :--- | :--- |
| Floor (FD of 256 raw training clips against the held-out ones) | 43.2 | 2.60 | 81.2 |
| Held-out clips' own roughness | 0.0096 | 0.0216 | 0.0306 |
| Copying ratio, mean over every arm and sampler | 1.04 | 0.95 | 0.94 |
| Copying ratio, the hybrid at one call | 1.30 | 0.99 | 1.10 |

The lowest copying ratio was 0.57, JetFlow's flow matching at Euler 1 on ucf101. Generation took 0.006 to 0.37
seconds per clip.

## What the outcome proposes (the plan's rules, as `outcome` applied them: rules 3 and 4)

- **Rule 1 does not apply.** H1 and H2 both hold on ucf101. The plan's only reason to retire is H1 or H2 read and
  failed on ucf101.
- **Rule 3: no one-call proposal.** H1 and H2 hold where they count, on ucf101 and together on collision. But H4-1
  does not hold on ucf101, so the flicker is reported and the hybrid is not named the package's one-call model.
- **Rule 4: no four-call proposal.** H3 does not hold on ucf101: at four calls the hybrid is 2.5 % better than flow
  matching at four calls, short of the 10 % asked. H4-4 also fails on ucf101 (2.7 times), so even with H3 a
  four-call proposal would have been blocked.

Nothing goes in force and nothing in the video package's documentation changes.

**Listed as retired, under the owner's standing word of 27 November 00:41** ("retire anything that does not meet our
expectation"), as that word was applied to the earlier results that day. The plan's own table gives no proposal
and no retirement. The hybrid did not meet these expectations:
- H1 on fluid;
- H3 on fluid and ucf101;
- H4-1 on fluid and ucf101;
- H4-4 on ucf101.

It is deprecated and kept for replication. Its code (`head_to_head.py`, frozen) is unchanged, and nothing in force
changes. One word from the owner reverses this.

## What it means

- **On real footage the hybrid keeps a lower FD at one call, but its clips are much rougher.** At one call it
  scored 460 against 553 for MeanFlow on the UNet and 574 for JetFlow's own best sampler at up to four calls. That
  lead rests on the minimum two of three seeds: MeanFlow on the UNet ranged from 395.9 to 657.7, and its seed 0 was
  lower than two of the hybrid's three. The lead holds on collision as well, but not on fluid, where H1 fails.
  - The hybrid's clips are 5.3 times as rough as the held-out footage, against 1.7 times for the reference. That
    supports calling it flicker; the first test's flicker (0.144 against 0.047) is confirmed (0.164 against 0.053).
    The FD did not penalise the roughness enough to change the ranking.
- **The roughness goes with JetFlow's network, not only with MeanFlow or one-call sampling.** On ucf101,
  JetFlow's network under flow matching at Euler 32 is as rough (0.166) as the hybrid at one call. The hybrid at
  Euler 32 is 0.149. The reference, flow matching on the UNet at Euler 4 (0.053), is the smoothest of its arm's
  samplers; its Euler 1 and Euler 32 are both 0.081.
- **At 32 calls the UNet arms overtake it; JetFlow's own flow matching does not.** On ucf101:
  - at four calls the hybrid still led both UNet arms: 374.6 against 411.6 for MeanFlow on the UNet and 384.2 for
    flow matching;
  - at 32 calls the UNet arms led: 287.6 for MeanFlow on the UNet and 292.1 for flow matching, against 353.6 for
    the hybrid;
  - JetFlow's flow matching was best at Euler 32, 520.6, still worse than the hybrid at one call.
- **The toy worlds disagree.** On collision the hybrid leads at one and at four calls. On fluid it loses to
  MeanFlow on the UNet at one call and to flow matching at four.

## Limits

- **Scale.** CPU only, 8 frames of 32 x 32, 45 minutes of training. This says which objective and network to carry
  into a larger build, not that any of them makes good video.
- **The real footage is not fresh.** The first test trained and scored on these same videos, groups 20 to 25
  included, and that is where the hypothesis came from. This test splits them by group.
- **Equal wall-clock on unequal runners.** Seven processor models, and training steps ranged widely: for example
  123,609 to 230,346 for flow matching on fluid.
  - All three ucf101 jobs of flow matching on the UNet ran on the slowest model (benchmark 0.303 to 0.317 s).
  - H1 on ucf101 does not look processor-driven: seed 1 was won with half MeanFlow-UNet's steps (46,329 against
    90,265).
  - The hybrid's best ucf101 seed had the most steps (80,659) on the fastest runner (0.133 s).
  - 18 jobs, not the plan's 20, started together. The collision hybrid of seed 0 started 15 minutes late, and
    JetFlow's collision job of seed 1 started 40 minutes late.
- **Two disclosed asymmetries.** The step costs at this size shifted against the hybrid. MeanFlow's time
  derivative is a finite difference on JetFlow's network and exact on the UNet. H1 compares exactly these two.
- **Convergence is not visible.** MeanFlow's recorded final loss is 1.0 in all 18 MeanFlow jobs. That is its
  adaptively weighted loss, which sits near 1 by construction, so the records do not show whether the MeanFlow
  arms converged.
- **The floor is out of reach.** It uses raw training clips, not clips passed through the fixed latent decoder that
  every model's output passes through, so no model can reach it. The best ucf101 FD is 3.5 times the floor.

## Independent check (AGENTS.md 1.6)

A reviewer that took no part in the run recomputed every number from the job records with its own script. It
compared `results.json` with the record printed in the collect job's log (byte-identical) and each job's record with
the one printed in that job's log (all equal). It confirmed every job's final-mode checks, including 149 loaded
files against commit 5f71ac9. It found no wrong number and no flaw in the run, and confirmed that the outcome stated
is the plan's.

It found ten points of wording, omission and limits. Each is now in the text above:

- "The parents catch up" was inaccurate: flow matching on the UNet is the reference, and JetFlow's flow matching
  never catches up. It is reworded.
- The flicker reading left out JetFlow's own roughness, the reference's other samplers and the held-out roughness.
  These are added, and the reading is reworded to what the run shows.
- The copying ratio, held-out roughness and generation time were missing. They are added.
- The limits on reused footage, step costs and the finite-difference derivative were weakened or dropped. They
  are restored.
- What the processor records show was not said. It is added.
- The wording on retirement was narrow, and H4-4's failure was not named under rule 4. The failed expectations are
  now listed; the retirement follows the standing word, not the plan's table.
- The one-call lead's seed margin and its absence on fluid were not said. Both are added.
- The status was set before the check. It is set now.
- The MeanFlow loss and the floor's construction were not stated. Both are now limits.
- Two side notes on the frozen plan:
  - its rehearsal-2 note says every record listed 148 loaded files, but the ucf101 records list 13, because they
    never import `apedp_dyn`;
  - rehearsal 2 had six of the seven processor models.

  Both are recorded here; the frozen plan is not edited.
