"""The unreferenced-module register is a census with a ceiling, not a comment.

``tests/test_smoke_imports.py`` proves every module imports. It does not prove a
test ever exercises one, and the difference was invisible: 39% of ``apedp_dyn``
was named by no test at all. This register makes that set finite and gated --
the detector is proven on a synthetic tree so a rule that always answered
"referenced" would fail here, and the ceiling can only be raised deliberately.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_untested_modules import OUT, build, is_referenced, module_paths

ROOT = Path(__file__).resolve().parents[1]


def _recorded() -> dict:
    return json.loads(OUT.read_text(encoding="utf-8"))


def test_register_is_exactly_what_the_tree_contains():
    assert _recorded() == build()


def test_totals_are_internally_consistent_and_every_entry_exists():
    payload = _recorded()
    totals = payload["totals"]
    assert totals["referenced"] + totals["unreferenced"] == totals["modules"]
    assert totals["unreferenced"] == len(payload["unreferenced"])
    assert payload["pinned_ceiling"] == totals["unreferenced"]
    modules = module_paths(ROOT)
    assert totals["modules"] == len(modules)
    for path in payload["unreferenced"]:
        assert path in modules, path
        assert Path(path).is_file(), path


def test_referenced_modules_are_not_in_the_register():
    """A module a test names may not be counted as unreferenced."""
    payload = _recorded()
    corpus = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((ROOT / "tests").rglob("*.py"))
    )
    for path in payload["unreferenced"]:
        assert not is_referenced(path.removesuffix(".py").replace("/", "."), corpus), path


def test_detector_finds_a_dotted_path_and_a_from_import(tmp_path):
    package = tmp_path / "apedp_dyn" / "pkg"
    package.mkdir(parents=True)
    (package / "cited.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "named.py").write_text("VALUE = 2\n", encoding="utf-8")
    (package / "ignored.py").write_text("VALUE = 3\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_cited.py").write_text(
        "from apedp_dyn.pkg.cited import VALUE\n", encoding="utf-8")
    (tests / "test_named.py").write_text(
        "from apedp_dyn.pkg import named\n", encoding="utf-8")

    payload = build(tmp_path)
    assert payload["unreferenced"] == ["apedp_dyn/pkg/ignored.py"], payload["unreferenced"]
    assert payload["totals"] == {"modules": 3, "referenced": 2, "unreferenced": 1}


def test_detector_credits_a_parenthesised_multiline_from_import(tmp_path):
    """A names list broken over lines is still a reference the interpreter resolves.

    Missing this form scored every module imported that way as unnamed; this
    audit's own coverage file imports the PS20 problems in exactly that shape.
    """
    package = tmp_path / "apedp_dyn" / "pkg"
    package.mkdir(parents=True)
    (package / "cited.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "also.py").write_text("VALUE = 2\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_multi.py").write_text(
        "from apedp_dyn.pkg import (\n    cited,\n    also,\n)\n", encoding="utf-8")

    assert build(tmp_path)["unreferenced"] == []


def test_prose_mentioning_a_module_name_is_not_coverage(tmp_path):
    """The rule that made the first measurement wrong: `server` is not a test."""
    package = tmp_path / "apedp_dyn" / "pkg"
    package.mkdir(parents=True)
    (package / "server.py").write_text("VALUE = 1\n", encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_prose.py").write_text(
        "# the server module is not imported here\nPROSE = 'server'\n", encoding="utf-8")

    assert build(tmp_path)["unreferenced"] == ["apedp_dyn/pkg/server.py"]


def test_ceiling_is_a_ceiling_not_a_description():
    """The pinned number must bound the measured one, not merely repeat it."""
    payload = _recorded()
    assert len(payload["unreferenced"]) <= payload["pinned_ceiling"]
    # Recount from the tree rather than trusting the file it just read.
    assert build()["totals"]["unreferenced"] <= payload["pinned_ceiling"]
