"""Keep the scaffold inventory's source links and evidence tiers checkable."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs" / "SCAFFOLD_INVENTORY.md"
TIERS = {
    "computed",
    "rule_screen",
    "literature_lookup",
    "planning_template",
    "reference_spec",
    "refused",
}


def test_inventory_defines_each_evidence_tier():
    text = INVENTORY.read_text(encoding="utf-8")
    for tier in TIERS:
        assert f"`{tier}`" in text


def test_inventory_source_links_resolve():
    text = INVENTORY.read_text(encoding="utf-8")
    inventory = text.split("## Current-checkout inventory", 1)[1].split(
        "## Static-table and scope-scan findings", 1
    )[0]
    links = re.findall(r"\[[^\]]+\]\((\.\./[^)#]+)(?:#[^)]*)?\)", inventory)
    assert len(links) >= 20
    for link in links:
        target = (INVENTORY.parent / link).resolve()
        assert target.is_file(), f"Inventory source link does not exist: {link}"


def test_documentation_index_links_to_inventory():
    index = ROOT / "docs" / "DOCUMENTATION.md"
    text = index.read_text(encoding="utf-8")
    assert "`docs/SCAFFOLD_INVENTORY.md`" in text
    assert (index.parent / "SCAFFOLD_INVENTORY.md").is_file()


def test_inventory_states_the_known_scope_and_census_limits():
    text = INVENTORY.read_text(encoding="utf-8")
    assert "not a claim that every module" in text
    assert "scripts/static_table_census.json" in text
    assert "scripts/scope_scan_baseline.json" in text
    assert "reference_spec" in text
