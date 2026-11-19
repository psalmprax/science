# Patent corpus — provenance

Ten publications. Five were fetched from Google Patents on **2026-10-31** by
`scripts/fetch_patent_corpus.py`; five more were harvested on **2026-11-01** by
`scripts/grow_patent_corpus.py`. Nothing in `corpus.jsonl` was typed from memory:
every record descends from a page that was actually retrieved, and the retrieval is
reproducible.

## What is stored, and which layer is ground truth

| path | contents | status |
|---|---|---|
| `raw/<ID>.json` | source URL, UTC retrieval timestamp, HTTP status, byte count, SHA-256 of the HTML, the query that found it (harvested records only), and the **verbatim** title, abstract and claims extracted from it | ground truth |
| `normalised_claims.json` | the claim number used, the normalised `expression`, and a note quoting the sentence it was read from | **human reading** |
| `auto_expressions.json` | formulas pulled from claim text by the automatic extractor, each with the sentence and pattern that produced it | **automatic** (currently empty — see below) |
| `extraction_yield.json` | what the automatic extractor found across **every** claim of every artefact, counted | measurement |
| `corpus.jsonl` | the engine's input: verbatim claim + normalised expression (`null` where the claim states no relation) | derived |
| `SOURCES.md` | the measured HTTP status of every source that was tried, with credentials reported by presence only | measurement |

The formula is not in the patent's text as the engine reads it — it is my
normalisation of a relation the claim states. Keeping the layers in separate files is
what makes that visible: you can diff `corpus.jsonl`'s `claim` against `raw/` and see
it is byte-identical, and you can read the note to see exactly which words the
expression came from.

## The records

| patent | office | normalised claims | expressions | found by |
|---|---|---|---|---|
| US6285999B1 | US | 0 | — | curated set |
| US8965721B2 | US | 2 (claims 1, 8) | `Z_meas/Z_ref`, `I_lim*V_min` | curated set |
| US7120566B1 | US | 2 (claims 1, 18) | `AF*exp(t_A)` (claim 18 restates it) | curated set |
| US11488017B2 | US | 0 | — | curated set |
| EP4334731B1 | EP | 3 (claims 2, 7, 8) | `eta_C*i_ch`, `C_t_I/C_0_I`, `C_t_E/C_0_E` | curated set |
| US11474159B2 | US | 4 (claims 4, 5, 6, 7) | `1 - SS_res/SS_tot`, `k_SEI*SOH + b_SEI`, `-0.01*b_SEI*SOH + b_SEI`, `A_SEI*exp(B_SEI/T) + C_SEI` | `battery state of health estimation` |
| US20250321289A1 | US | 2 (claims 4, 11) | `s(t)**2`, `SoH_pred` | `battery state of health estimation` |
| US11921163B2 | US | 0 | — | `battery state of health estimation` |
| US12130335B2 | US | 0 | — | `battery state of health estimation` |
| US12372585B2 | US | 0 | — | `battery state of health estimation` |

Exact digests and byte counts are in `raw/<ID>.json` and are repeated per record in
`benchmarks/patent_prior_art/prior_art_run.json`. Re-check them with:

```bash
python3 scripts/fetch_patent_corpus.py --verify       # curated five
python3 scripts/grow_patent_corpus.py --scan-claims   # extractor yield, all artefacts
```

A changed digest means the page changed upstream, not that the record was forged —
either way the record is then stale and the run must be repeated.

## Growth, and what it did and did not buy

The brief was to grow the corpus from a bulk source so that presence and absence mean
something beyond five claims. Both named sources are implemented and both were
measured rather than assumed (`data/patents/SOURCES.md`):

| source | measured 2026-11-01 | can it be used here |
|---|---|---|
| PatentsView (`search.patentsview.org`) | name does not resolve from this host; the API also requires `PATENTSVIEW_API_KEY` | no — and it carries titles/abstracts, **not claims** |
| EPO OPS (`ops.epo.org`) | `403` without OAuth credentials | no — needs `EPO_OPS_KEY`/`EPO_OPS_SECRET` |
| Google Patents query endpoint | `400` for the bare probe; `503` "Sorry…" block page once queried, and `503` on individual publication pages too | partially — five publications were harvested before the block |
| USPTO legacy `ds-api` bulk search | `301` → `data.uspto.gov`, which serves an Angular portal and an API that needs a key | no — legacy host retired |
| USPTO Open Data Portal | key-gated search API | no — needs a USPTO API key |

HuggingFace's BIGPATENT view was also checked as a keyless bulk alternative: its
viewer serves `abstract` and `description` and **no claim text**, so it could not
produce a claim record, and it is not a patent office. HUPD (Harvard USPTO Patent
Dataset) carries claims but its viewer is unsupported for script-based datasets, so
no rows were retrievable without downloading the full corpus.

So the growth path that ran is the Google Patents query endpoint, on the same subject
vocabulary as the curated five, with the query recorded per record. That is a
**query-selected sample of a query endpoint**, not an office population, and the
record says so in `SOURCES.md`.

**What grew: the record layer, from five to ten.** Absence is now stated over ten
publications instead of five, each with a source URL, a retrieval date, a digest and
the query that found it.

**What did not grow from fetching: the searchable layer, which stayed at three normalised claims.**
The automatic extractor was run over every claim of every artefact — **501 claims,
0 parsed formulas**. The reason is in the text: patent claims state relations in prose
("determining a ratio of said measured DC impedance to a reference DC impedance"), and
where an equation does appear it is usually an image, which is why EP4334731B1 claim 1
had to be replaced by claim 2. `auto_expressions.json` is therefore empty, and
`corpus_status()` reports `claims_scanned_for_formulas: 501` and
`formulas_found_in_claim_text: 0` next to `records: 29`, `searchable_claims: 15` and
`decisive_claims: 8`, so
a `NOT_FOUND_IN_PATENT_CORPUS` verdict carries its own denominator.

**What did grow it was transcription, not fetching.** Every claim whose text states an
explicit relation was normalised by hand, one to four per record: **fifteen claims across
seven records**, so `searchable_claims` went 3 → 15 and every verdict now reports which
record *and which claim number and expression* matched
(`matched_claim_number`). Three records gained a second or third form this way:
EP4334731B1 (claims 2, 7, 8), US11474159B2 (claims 4, 5, 7) and US8965721B2 (claim 8
beside claim 1); US20250321289A1 and US7120566B1 hold one each. The work is visible per
record in `normalised_claims.json`, and five of them are marked **COARSE** because the
normalisation dropped structure (grouped sums in `1 - SS_res/SS_tot`, the integral in
`s(t)**2`). The remaining claims either state no relation or repeat one already
recorded; that reading is per record, not a claim of exhaustiveness.

**Retries are logged, not narrated.** `data/patents/harvest_attempts.jsonl` records every
`--retry-harvest` attempt — timestamp, endpoint, HTTP status, outcome (`open`,
`blocked`, `unreachable`) and how many records it actually added. Re-running the
harvest later therefore leaves evidence: the two attempts logged at 2026-11-01
(`01:22:13Z` and `01:23:45Z`) each returned HTTP `503` with `records_added: 0`, which is
why the corpus was then still at ten records — and they are distinguishable from an
attempt that ran against an open endpoint and found nothing. The third attempt
(`01:34:17Z`, HTTP `200`) succeeded: the block had lifted, 21 publications were fetched
(19 new records), and the corpus grew from 10 to 29 records with 15 normalised claims,
**8 decisive and 7 operand-only**, after the two newly reachable relations were
transcribed by hand (`US11496054B2` claim 1's volt-second relation, and
`US12021452B2` claim 1's 2:1 conversion ratio, which is a quotient and therefore
refused). The other 17 new records state no relation in any claim and are counted as
unsearchable, each with a `_note` recording that all of its claims were read. On the second attempt the block
was confirmed independently on four Google paths (the `xhr/query` endpoint, two
`/patent/…/en` pages and the search page): all `503`, all the same 1,103-byte block page.

**Each attempt also carries the split it left unchanged.** New entries in
`harvest_attempts.jsonl` include `corpus_split`: record count, searchable claims, how
many are decisive, how many are refused as operand-only, and the refused claim numbers
per record. A blocked attempt therefore states what the corpus still consists of —
15 claims, 8 decisive, 7 refused (EP4334731B1 2/7/8, US12021452B2 1,
US20250321289A1 11, US8965721B2 1/8) — instead of only reporting that nothing arrived.

**Recorded OPS fixtures, replayed rather than simulated.** `data/patents/ops_fixtures/`
holds two responses recorded from the live endpoints with no OAuth token:
`claims_EP4334731_403.xml` (HTTP 403, "violation of Fair Use policy") and
`token_401.xml` (HTTP 401, "Client identifier is required"), each with URL, timestamp,
byte count and SHA-256 in `PROVENANCE.json`. `tests/test_patent_bulk_sources.py`
replays the 403 body **through the adapter** — asserting the endpoint, the
`Authorization` and `Accept` headers, and that the status maps to
`SourceUnavailable(status=403)` — rather than through the parser alone. No
authenticated claims response could be recorded here, so that replay test skips with
the command that would produce one (`scripts/record_epo_ops_fixture.py --number …`);
a synthetic body is never passed off as a recording.

**Claim 1 of each harvested patent now carries its own reading.** All five were read:
**none states a relation**. Four state procedures only (a GNN pipeline, a QV→IC→ICA
pipeline, a time-series model input/output, and an ML-plus-Kalman fusion pipeline), and
`US11474159B2` claim 1 names three "relation functions" without writing them. So
curating claim 1 grew the set by exactly one form, not five: the pointer's simplified
second relation function (`US11474159B2` claim 6, `-0.01*b_SEI*SOH + b_SEI`, decisive
because the coefficient is the claim's). Claim 3's Arrhenius law was **not** counted a
second time — it is the same form as claim 7. The readings are stored as a `claim_1`
object per record in `normalised_claims.json`, so "claim 1 was read" is on disk rather
than implied.

## Records that were fetched and rejected

Four candidates produced no usable record and are not silently absent:

| patent | why rejected |
|---|---|
| US2237713A (Optical pyrometer, 1941) | the source exposes no claim markup for this publication — claims are page images |
| CN102403767B (Battery discharge control) | same: claim section present, no claim text |
| DE102011079469A1 (Charge acceptance) | same: zero claims parsed, so it cannot become a record |

This is a **selection bias**: the corpus holds publications whose claims the source
renders as text, which favours recent US and EP filings. It is a ten-record corpus of
held claims, not a sample of a patent office.

## Three limits that change what a verdict means

1. **The searchable set is the claims that state a relation, not every claim.** Fifteen
   claims across seven records were normalised, one to four per record (four for
   `US11474159B2`, whose claim 1 defers to three relation functions); twenty-two
   records state no relation in any claim. EP4334731B1's claim 1 renders its efficiency
   symbols as image placeholders (`Figure imgb02xx`), which is why claims 2, 7 and 8
   carry that record. So `NOT_FOUND_IN_PATENT_CORPUS` means "not among these fifteen
   normalised claims", not "not in these twenty-nine patents".
2. **An operands-and-one-operation claim cannot carry a verdict.** `US8965721B2`
   claim 1 names a *ratio of two impedances* and says nothing about the functional
   form, so its expression is `Z_meas/Z_ref` — which matches **any** quotient. The
   same is true of a bare product or sum (`eta_C*i_ch`, `I_lim*V_min`, `SoH_pred`).
   Those seven of the fifteen claims are therefore held apart as **operand-only**: they
   are counted (`operand_only_claims: 7`, `decisive_claims: 8`) and a match returns
   `CLAIM_STATES_OPERANDS_ONLY`, not `KNOWN_IN_PATENT_CORPUS`. The probes
   `generic_ratio` and `self_check_efficiency` exist to keep that refusal measured.
3. **The claim as written is not repaired.** `US7120566B1` claim 1 states
   `t_F = AF·exp(t_A)`, which exponentiates a time and is therefore dimensionally
   meaningless. It is recorded as written with the oddity noted; correcting a claim
   is not this corpus's job.

## What a verdict is not

Not a novelty opinion, not a freedom-to-operate opinion, not legal advice, and not
a search of any patent office's full text. A professional prior-art search is a
different activity with different tools.
