# PREREG-CRYPTO-PIT-2026-11 — Deviation 01: the data builder could not fetch a non-ASCII pair

**Date:** 2026-11-17. **Stage:** data build. No panel existed before or after; nothing was analysed.

## What happened

The first build ran on GitHub Actions (run 43993312069, commit `6d20a3f`, 01:09–01:19 UTC) and
stopped with:

```
UnicodeEncodeError: 'ascii' codec can't encode characters in position 30-33: ordinal not in range(128)
...
RuntimeError: GET https://data.binance.vision/data/spot/monthly/klines/币安人生USDT/1d/币安人生USDT-1d-2026-01.zip failed after 4 attempts: 'ascii' codec can't encode characters ...
```

The archive lists a USDT pair whose symbol is not ASCII. The builder put the object key into the
URL as it was, and an HTTP request line must be ASCII. The job failed before writing anything: the
`market-data` branch does not exist (checked through the GitHub API after the failure), and no
panel file, partial or complete, was downloaded or opened by anyone.

## What changed

`scripts/build_point_in_time_crypto_panel.py` gains `archive_url(key)`, which percent-encodes the
object path (`urllib.parse.quote(key, safe="/")`). Every ASCII key comes out unchanged, so every
other request is byte-for-byte what it was. Two tests pin it
(`tests/test_point_in_time_crypto_panel.py`): the fake archive now refuses non-ASCII request lines
the way urllib does, and a pair named `币安人生USDT` is fetched, verified and given columns. Both
fail against the old builder.

Nothing else changed. The preregistration text and the analysis script are byte-identical to the
frozen versions. The universe rule is unchanged: the pair was always meant to be in it ("every
Binance spot pair quoted in USDT"), and now it can be fetched.

| File | Frozen digest | Now |
| :--- | :--- | :--- |
| `preregistration/PREREG-CRYPTO-PIT-2026-11.md` | `7357ba5e…8e85` | unchanged |
| `scripts/build_point_in_time_crypto_panel.py` | `42a7f27d189dd2caef304002d60ea773a5002ae6677e692c90c16683ca8f285e` | `4d0b97a1abf0d9bbceb5df63349177f9f124abbe0b9da622755877a4fdc227d8` |
| `scripts/run_prereg_crypto_pit.py` | `098d32fe…d573` | unchanged |

## Why this is a deviation and not a new preregistration

The deviation rule says a failed build is re-run with the analysis unchanged, and that changing a
candidate, a window, a universe parameter or a threshold *after the data exists* requires a new
preregistration. This failure could not be cured by re-running, so the builder had to change; the
change is to how a URL is spelled, the data did not exist when it was made, and none of the four
things the rule protects moved. It is recorded here, with the old digest, so the result can be
labelled as built under Deviation 01.
