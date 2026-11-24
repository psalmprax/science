# Patent corpus data

This directory holds the bundled patent corpus the prior-art engine searches:
`corpus.jsonl` (29 publications), the raw fetched artefacts under `raw/`, the
hand-curated normalised claims in `normalised_claims.json`, and the refusal
listing `REFUSED_CLAIMS.md`. The package reaches it through
[`BUNDLED_CORPUS_PATH`](../../apedp/patent_prior_art.py), and `corpus_status()`
reports the record count, the searchable and decisive claim counts, the
operand-only refusals and the corpus digest.

The corpus is small on purpose. Fifteen of the 501 claims across the 29 records
carry a normalised relation, across seven records; twenty-two records state no
relation in any claim and are counted as unsearchable. Eight claims can carry a
verdict and seven are refused as operand-only, which is below the declared floor
of 50 decisive claims — so every verdict says the index is thin.

Do not treat an empty corpus, an unsearchable claim, or a corpus-relative
`NOT_FOUND_IN_PATENT_CORPUS` result as evidence of novelty or freedom to operate.
Each record carries its source URL, retrieval date and the SHA-256 of the fetched
HTML; adding records requires the same provenance, and the licences and access
terms of the source must be reviewed before redistribution. Refresh the coverage
record in `PROVENANCE.md` alongside any new dataset.
