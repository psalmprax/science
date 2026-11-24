"""docs/CHANGELOG_INDEX.md is the index scripts/changelog_index.py makes from CHANGELOG.md, not a hand-kept list."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _script() -> object:
    spec = importlib.util.spec_from_file_location("changelog_index", ROOT / "scripts" / "changelog_index.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["changelog_index"] = module
    spec.loader.exec_module(module)
    return module


index = _script()


def test_the_index_is_the_one_the_changelog_gives() -> None:
    wanted = index.build(index.CHANGELOG.read_text(encoding="utf-8"))
    assert index.INDEX.read_text(encoding="utf-8") == wanted, "run: python scripts/changelog_index.py --write"


def test_an_entry_is_listed_by_its_bold_title_or_its_first_words_under_its_section() -> None:
    text = "\n".join(["# Changelog", "", "## [Unreleased]", "", "### Added", "",
                      "- **A reading for small tables (2026-11-24;", "  development).** It reads tables.", "  - a sub-point",
                      "- Burn down the lint debt in one module.", "", "### Fixed", "", "- **One bug.** Details.", "",
                      "## [Unreleased] — an older block (2026-10-23)", "", "- **Not listed.** Older entries are not."])
    built = index.build(text)
    assert "### Added\n\n- A reading for small tables (2026-11-24; development)\n- Burn down the lint debt in one module\n" in built
    assert "### Fixed\n\n- One bug\n" in built and "a sub-point" not in built and "Not listed" not in built
    assert "## Earlier blocks\n\n- [Unreleased] — an older block (2026-10-23)\n" in built
    assert built.endswith("3 entries in the current block; 1 earlier blocks.\n")
    long_title = "- **" + "word " * 60 + "end.** Text."
    assert len(index.title_of([long_title], 0)) == index.TITLE_WIDTH and index.title_of([long_title], 0).endswith("…")
