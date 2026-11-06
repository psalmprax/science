# Patent corpus data

No patent corpus is bundled in this directory. The loader in
[`apedp/patent_prior_art.py`](../../apedp/patent_prior_art.py) supports a local
JSONL corpus, but adding records requires a source and retrieval date for each
real patent. Review the source's access terms, license, redistribution
permissions, and any restrictions on extracted claim text before adding or
publishing data.

Do not treat an empty corpus, an unsearchable claim, or a corpus-relative
`NOT_FOUND_IN_PATENT_CORPUS` result as evidence of novelty or freedom to operate.
Record corpus source, retrieval date, jurisdictional and temporal coverage, and
refresh method alongside any future dataset so that its coverage can be
assessed.
