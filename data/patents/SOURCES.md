# Patent corpus sources: what was attempted, and what was used

Probed on **2026-11-01** by `scripts/grow_patent_corpus.py --probe-sources`. Credentials are reported by presence only; no secret is read into this file.

| source | URL | HTTP | credentials present | note |
|---|---|---|---|---|
| `patentsview_api` | `https://search.patentsview.org/api/v1/patent/` | None | PATENTSVIEW_API_KEY=False, EPO_OPS_KEY=False | unreachable: [Errno -2] Name or service not known — search API; titles and abstracts only, no claim text; needs PATENTSVIEW_API_KEY |
| `patentsview_legacy` | `https://api.patentsview.org/patents/query` | 200 | PATENTSVIEW_API_KEY=False, EPO_OPS_KEY=False | other — retired API host; answers HTML, not JSON |
| `epo_ops` | `https://ops.epo.org/3.2/rest-services/published-data/search` | 403 | PATENTSVIEW_API_KEY=False, EPO_OPS_KEY=False | Forbidden — published-data search plus per-publication claims; needs OAuth (EPO_OPS_KEY/EPO_OPS_SECRET) |
| `google_patents_query` | `https://patents.google.com/xhr/query` | 503 | PATENTSVIEW_API_KEY=False, EPO_OPS_KEY=False | Service Unavailable — keyless query endpoint used for the bundled corpus; claims read from publication pages |
| `uspto_ds_api` | `https://developer.uspto.gov/ds-api/patents/grant/v1/records` | 200 | PATENTSVIEW_API_KEY=False, EPO_OPS_KEY=False | other — probe followed a 301 to data.uspto.gov and read the portal page, not an API response; the API needs a key |
| `uspto_open_data` | `https://data.uspto.gov/api/v1/patent/applications/search` | 200 | PATENTSVIEW_API_KEY=False, EPO_OPS_KEY=False | other — same: the 200 is the portal page, not an API response; the API needs a USPTO key |

**Corpus built from `google_patents_query`.** Six endpoints were probed and are recorded above. PatentsView's API host does not resolve, EPO OPS refuses without OAuth, both USPTO APIs need a key, and Google Patents serves a 503 block page — so the bundled corpus came from Google's query endpoint before that block. No bulk source could be used here, and that is the finding this table records.

The substitution is the finding, not a footnote. A bulk-source corpus built on
PatentsView full-text tables would support claims about an office's population;
this corpus does not. It is a query-selected sample, every record keeps the
query that found it, and a record count here is not a population size.

Both bulk paths are implemented in `scripts/grow_patent_corpus.py` and both
**exit 2 with the measured reason** when their credentials are absent:
`--source patentsview` (needs `PATENTSVIEW_API_KEY`, and yields titles and
abstracts but no claims, so claims must then come from EPO OPS) and
`--source epo-ops` (needs `EPO_OPS_KEY`/`EPO_OPS_SECRET` and serves claims
directly through `parse_epo_claims`). An empty result from a source that could
not be queried is refused rather than recorded as "no patents found".
