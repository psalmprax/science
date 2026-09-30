"""Canonical locations of the frozen benchmark corpora.

The corpora themselves are checked in and checksummed, but the scripts that
originally generated them (``generate_synthesis_50_corpus.py`` and
``synthetic_synthesis_generator.py``) were never committed. Consumers that
only need to *read* a corpus should import its path from here rather than
depend on a generator that does not exist.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 50-problem autonomous model synthesis tournament (S01..S50).
CORPUS_50_DIR = REPO_ROOT / "benchmark_corpus_synthesis_50"

#: Single-problem synthetic synthesis smoke corpus (SYNTH_01).
CORPUS_SYNTHESIS_DIR = REPO_ROOT / "benchmark_corpus_synthesis"

#: 60-problem external benchmark corpus (P/E/R/O/C/X/A blocks).
CORPUS_EXTERNAL_60_DIR = REPO_ROOT / "benchmark_corpus_external_60"

__all__ = [
    "CORPUS_50_DIR",
    "CORPUS_EXTERNAL_60_DIR",
    "CORPUS_SYNTHESIS_DIR",
    "REPO_ROOT",
]
