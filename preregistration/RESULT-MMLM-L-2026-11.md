# Result: PREREG-MMLM-L-2026-11

**Run:** once, 2026-11-19, at commit `6cf4907`, which froze the preregistration, the script, the law
model, the corpus loader, the searches and the manifest (`preregistration/DIGEST-MMLM-L.json`; the
script checks it before running). Python 3.13.18, numpy 2.5.3, scipy 1.18.1, sympy 1.14.0. Two CPU
cores, about four minutes. No deviation.

Outputs, unedited, in [`RESULT-MMLM-L-2026-11/`](RESULT-MMLM-L-2026-11/): `results.json`, `runs.json`
(every fresh law's cost under every model and mode, and every search run) and `run.log`.

## Fresh laws

ODEBench has 117 component laws. 27 read the same as one of the 133 training laws after
canonicalisation, 34 the same as an earlier component, and **56** were kept. Those 56 were never used
to build either model.

## Part 1: language

Mean bits per fresh law (lower is better), all 133 laws as training:

| | Frozen | While learning |
| :--- | ---: | ---: |
| Law model (MMLM-L) | **22.61** | **19.27** |
| Earlier model (`SelfAdjustingEquationModel`) | 27.94 | 24.87 |
| Unigram | 29.29 | 28.32 |

Paired over the 56 laws, the law model is cheaper than the earlier model on 52 and dearer on 4 when
frozen (sign test p = 1e-11), and on 51 against 5 while learning (p = 1e-10).

Feynman laws only as training, so `dynamics` starts with no laws (no criterion was set):

| | Frozen | While learning |
| :--- | ---: | ---: |
| Law model (MMLM-L) | 31.51 | **20.08** |
| Earlier model | 30.97 | 26.22 |
| Unigram | **29.97** | 28.75 |

## Part 2: search

100 runs: 40 invented laws and the 10 fresh ODEBench laws the enumeration can express, at 1 % and
5 % noise. The plain exhaustive search recovered 93 in 192.1 s.

| Arm | Recovered | Wrong early stops | Median share of candidates fitted | Seconds |
| :--- | ---: | ---: | ---: | ---: |
| Law prior, confirmed stop | **96** | **4** | 3.2 % | 8.2 |
| Law prior, first stop | 95 | 5 | 0.1 % | 3.6 |
| Earlier prior, confirmed stop | 97 | 3 | 3.2 % | 8.6 |
| Earlier prior, first stop | 95 | 5 | 0.2 % | 3.6 |

Position of the true law in the search order, over the 50 laws: median 20 under the law prior and
24.5 under the earlier prior; the law prior ranks it earlier for 18 laws and later for 21 (sign test
p = 0.75).

## Against the expectations

1. **Better language model on fresh laws: held.** 22.61 against 27.94 bits per law, p = 1e-11.
2. **Learns faster: held.** 19.27 against 24.87 (earlier model) and 28.32 (unigram), p = 1e-10.
3. **Not a worse search order: held, and no more than that.** Median position 20 against 24.5; the
   paired comparison shows no difference (18 against 21).
4. **The confirmed stop is safe: held.** 4 wrong early stops in 100 runs (limit 5).
5. **No laws lost: held.** 96 recovered against the plain search's 93.
6. **Work is still saved: held.** Median 3.2 % of candidates fitted (limit 10 %); 8.2 s against
   192.1 s, 4.3 % of the plain search's time (limit 25 %).

All six held.

## What this means

- As a model of how laws are written, MMLM-L is clearly better than what it replaces: about a fifth
  fewer bits on laws from a source neither model was built on, and it keeps that lead while learning.
- That gain does not show up as a better search order. On these laws the two priors rank the true
  law equally well. The search improved because of the stop rule, not because of the prior: the
  confirmed stop works as well with the earlier prior (97 recovered, 3 wrong stops).
- The weakness disclosed before the run is confirmed. With no laws from the field, the law model is
  not better than a unigram (31.51 against 29.97 bits; dearer on 37 of 56 laws). It needs laws from a
  field before it helps there, and then it learns quickly (20.08 while learning).

## After the run (not preregistered)

Computed from `runs.json` after seeing the result.

- **The four wrong stops.** `invented_12` at 5 % noise (the plain search recovered it);
  `invented_15` at 5 % noise and `odebench_05_0` at both noise levels, which the plain search also
  failed to recover. So one of the four is a law the confirmed stop lost and the full search found.
- **Where the confirmed stop beat the plain search.** In 4 runs it recovered a law the plain search
  missed, and in 1 run the reverse. The plain search takes the shortest description over all
  candidates; the confirmed stop takes it only among candidates whose residuals show no structure.
- **The search set is small where it matters.** Only 10 of the 56 fresh ODEBench laws are within the
  enumerated space (the others have constants inside the expression). The search result rests mostly
  on the 40 invented laws.
- **Learning is not monotone along the stream.** The law model's cost is 18.95 bits on the first ten
  fresh laws and 27.45 on the last ten (the earlier model: 22.27 and 35.28). The file runs from
  one-dimensional systems to four-dimensional ones, so later laws are longer; the comparison between
  models at each point is the meaningful one.
