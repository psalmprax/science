# What the symbolic search engines could not reach, and the patent corpus

**Status:** measured record + module note · **Scope:** the beam, hierarchical and ensemble
search surfaces in `apedp/advanced_search.py` and `apedp/feature_selection.py`, the
evolutionary search's selection split in `apedp/evolutionary_search.py`, and the new patent
prior-art index · **Evidence:** `benchmarks/search_engine_gaps.json` (produced by
`scripts/measure_search_engine_gaps.py`), `tests/test_search_engines_real.py`,
`tests/test_evolutionary_search.py`, `tests/test_dyn_patent_prior_art.py`.

Three defects were measured, not inferred. Each is stated with the number that changed.

## 1. The ranker preferred the worse candidate

`ValidationPipeline` builds a composite `overall` score, and `HierarchicalSearch.search`
sorts candidates by it. The formula was

```
0.4 * max(0, cv_mean - cv_std) + 0.2 * ood + 0.2 * stability + 0.2 * 0.8
```

with `cv_mean` a **cross-validation error**. A perfect fit (error 0) therefore scored **0.56**
and a candidate that missed by 0.9 scored **0.83**: the search returned its worst candidate
first, and every result sorted by `overall_score` was worse than a random draw from its own
front. The Rust kernel carried the same sign.

Both are fixed: the score is now `0.4 * max(0, 1 - (cv_mean + cv_std)) + ...`, and the Python
path recomputes it from the returned components rather than trusting the kernel, so the two
paths cannot disagree. The fix is worth more than every template added below — it is what
lets any of them be found:

| engine | planted laws recovered, inverted score | with the sign fixed |
|---|---|---|
| `HierarchicalSearch` (classic) | 0/12 | 2/12 |

## 2. Reachability: what the template sets cannot express

`scripts/measure_search_engine_gaps.py` plants twelve laws, one per structural family,
searches on 200 rows and scores the returned candidate on 100 untouched rows, with the
candidate's scale and offset refitted on the training rows only. Recovery means holdout
`R^2 >= 0.99`.

Classic `HierarchicalSearch` templates are `x`, `x**2`, `sin(x)`, `exp(x)`, `x0+x1`,
`x0*x1`, expanded by `x**2`, `x*2`, `2*x`, `sin(x)`, `x+1`. A decay, a reciprocal, a square
root, a cubic, a logarithm, a cosine, a rational saturation and a ratio are not merely
unlikely: they are **unreachable at any budget**. `beam_search_expressions` atoms are the
variables plus `1` and `2`, so the same forms are unreachable there.

| engine | recovered |
|---|---|
| `HierarchicalSearch`, classic | **2/12** (`power_2_5`, `product`) |
| `HierarchicalSearch`, extended | **12/12** |
| `beam_search_expressions`, classic atoms | **1/12** (`linear`) |
| `beam_search_expressions(include_functions=True)` | **9/12** |

The extended hierarchical set adds reciprocal, square root, cubic, `x**0.5`, `exp(x)`,
`exp(-x)`, `log`, `cos`, rational saturation, pair ratios and a variable in an exponent, and
fits an exact least-squares scale and offset to every candidate before scoring it: a law
found up to a constant is a found law. The extended beam atoms add the unary forms the
default atom set cannot compose. Both extensions are opt-in-preserving: `include_functions`
defaults to `False` so existing atom counts and orderings do not move, and
`template_set="classic"` reproduces the old behaviour for comparison.

The benchmark's `sin(2x)` miss is now covered by bounded frequency-scaled sine and cosine
expansions (multipliers 0.5, 2 and 3) in the extended hierarchical mid phase. A regression
test searches noisy training observations, then applies the returned expression and
training-fitted scale/offset to an independent noisy holdout; the holdout is not passed to
search. The rerun of the 12-law protocol recovers the sine case at holdout `R^2 = 0.9999`,
bringing extended hierarchical recovery from 11/12 to 12/12. The regenerated
`benchmarks/search_engine_gaps.json` records both the measured result and the frequency
multipliers used. This closes the measured `sin(2x)` gap without fitting frequency from
confirmation data.

**Still limited:** these are discrete frequency multipliers, not continuous frequency or
phase estimation. Off-grid frequencies and phase-shifted sinusoids remain outside this
change's evidence; the added regression pin does not claim general periodic-law recovery.

## 3. The evolutionary search selected on the confirmation rows

`evolve(x_train, y_train, x_test, y_test)` chose `grouped_pick` by minimising error on
`y_test`. That makes the test rows a second training set: the reported holdout error of a
candidate selected for its holdout error is optimistic by construction, and the split stops
being a confirmation split.

Selection now runs a deterministic 5-fold inner cross-validation over the **training** rows
(`SearchConfig.selection_folds`), refitting the candidate's coefficients inside each fold.
The test rows are read once, after the pick is fixed, to report
`SearchResult.confirmation_mse`; `selection_method` says `inner_cv` or `train_error`, and
`selection_folds` says how many folds actually ran. Both the native and Python paths were
changed; the native path evaluates folds without a refit because the kernel resolves a
candidate's constants inside the expression it returns, which is stated in the code.

The pin is behavioural, not textual:
`test_grouped_pick_is_selected_without_reading_the_confirmation_rows` inverts the
confirmation targets and requires the pick to be identical while the confirmed error gets
worse. A selection that consulted the test rows cannot pass that.

## 4. Patent prior art

`PriorArtAuditor` searches 193 textbook laws. That answers "is this a known physical law",
not "is this claimed in a patent". `apedp/patent_prior_art.py` adds the second question over
a local corpus, with three rules:

1. **Provenance is mandatory.** A record is either a real patent carrying an http(s) source
   URL and a retrieval date, or it is explicitly marked `synthetic`. A record that claims a
   real patent without attribution is refused at load (`PatentProvenanceError`), so a
   hand-written formula cannot enter the corpus wearing a patent number. Duplicate ids are
   refused for the same reason.
2. **Claims are not formulas.** Only a record with a normalised `expression` can be matched
   symbolically. The rest are counted and reported: `NO_SEARCHABLE_CLAIMS` with
   `searched_patents` and `searchable_claims` is a different finding from
   `NOT_FOUND_IN_PATENT_CORPUS`, and collapsing the two would turn "no data" into "no prior
   art".
3. **A verdict is corpus-relative.** Every result carries the corpus size, the synthetic
   count, and a disclaimer that a patent-corpus verdict is not a novelty opinion and not a
   freedom-to-operate opinion.

Matching is rename-insensitive algebraic equivalence first (a patent's symbol names are its
draughtsman's notation: `a*x/(1+b*x)` and `alpha*T/(1+beta*T)` are the same claimed form),
then a shared structural signature (`RELATED_STRUCTURE_IN_PATENT_CORPUS`), otherwise
absence. Usage:

```python
from apedp.patent_prior_art import PatentPriorArtIndex

index = PatentPriorArtIndex.from_jsonl("data/patents/corpus.jsonl")
result = index.audit("a * x / (1 + b * x)")
```

**No corpus is bundled.** The loader, the record schema and the source options are; the data
is a deliberate gap, because a patent corpus is a licensed, large, periodically-refreshed
artefact and a token sample in this repository would be a claim of coverage the project does
not have. Candidate public sources are named in the module docstring (USPTO PatentsView,
USPTO Bulk Data, EPO OPS, Google Patents public datasets); whichever is used, every record
must carry its URL and retrieval date, because the loader requires them.

## 5. Pins

- `tests/test_search_engines_real.py` — the ranker prefers the smaller error, `search()`
  returns its best candidate, the extended set reaches a decay the classic set cannot,
  harmonic recovery transfers to an untouched holdout, a fitted coefficient is recovered
  (`7.5*x**2` → `x0**2` with scale 7.5), and the beam function atoms are present only when
  asked for.
- `tests/test_evolutionary_search.py` — the confirmation rows cannot move the pick, and the
  confirmed error is measured rather than chosen.
- `tests/test_dyn_patent_prior_art.py` — provenance refusal, duplicate refusal, unparseable
  expression refusal, exact/related/absent verdicts, the no-searchable-claims count, and the
  synthetic disclosure in every verdict.
- `benchmarks/search_engine_gaps.json` — the twelve-law table above, with the protocol
  (train/holdout sizes, noise, budget, criterion, leakage statement) recorded beside it.

## 6. Robustness across noise, range, and generated worlds

The original one-seed measurement is now complemented by
`scripts/measure_search_engine_gaps.py`'s multi-condition protocol and the
reproducible record in `benchmarks/search_engine_robustness.json`. The recorded
run crosses three synthetic-world seeds (7, 17, 27), two train-scaled noise
levels (1% and 5%), and two positive input ranges ([0.5, 3] and [0.1, 10]):
12 conditions and 144 law/condition trials across the same 12 planted families.
The extended hierarchical engine recovered 135/144 trials (9–12 of 12 per
condition); the extended beam recovered 102/144 (8–9 per condition).

For each condition, input rows are split before noise is added. Noise magnitude
uses only the training target's standard deviation; candidate search and
coefficient fitting see training rows only, and holdout rows score the fixed
candidate. This removes even the previous use of full-target scale for noise
generation.

These are multiple generated synthetic worlds, not independent real datasets.
Both train and holdout inputs within a condition are sampled from the same
declared interval; changing intervals between conditions is not an
out-of-distribution train/test experiment. The results show sensitivity to
seed, noise and domain width, but do not establish recovery on real scientific
data, arbitrary equations, extrapolation ranges or phase/off-grid periodic laws.
