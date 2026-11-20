"""Entry point for the offline dry run: ``python -m mmlm_matter_001.pipeline``."""

from __future__ import annotations

from collections.abc import Sequence

from mmlm_matter_001.pipeline.dry_run import main


def run(argv: Sequence[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(run())
