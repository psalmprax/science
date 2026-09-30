"""Smoke tests for apedp modules no other test file exercises.

An audit found modules under ``apedp/`` that no test file names. Most were
never broken; they were simply never run, which is the same failure mode as a
check that reports nothing: the module could stop working tomorrow and the
suite would not notice.

A note on how this list was found, because the first method was wrong. The
audit grepped the tests for module *paths* (``apedp.api.routes.health``) and
concluded the HTTP API was untested. It is not -- ``tests/test_api_routes.py``
drives the whole app through ``TestClient`` and never imports a route module
directly, because it tests the app rather than the files. Path-grepping finds
modules no test mentions; it cannot tell whether they are covered some other
way. So this file covers what is genuinely unexercised, confirmed by running
it, and the API is covered by what it already has.

Scope is deliberately narrow: pure, offline, deterministic functions only.
Nothing here reaches the network, spawns a solver, or reads a corpus.
"""
from __future__ import annotations

import json

import numpy as np
import pytest


class TestDecisionPolicyBackends:
    """apedp/decision/policy.py -- the search-control backends."""

    def test_rule_based_backend_returns_a_decision(self) -> None:
        from apedp.decision.policy import RuleBasedBackend

        out = RuleBasedBackend().decide({"stagnation_steps": 12, "budget_left": 0.5})
        assert set(out) >= {"decision", "probabilities"}, out
        assert isinstance(out["decision"], str) and out["decision"], out

    def test_jev_backend_without_a_client_says_so(self) -> None:
        """The failure must be legible, not a crash deep in a search loop."""
        from apedp.decision.policy import JevBackend

        with pytest.raises(NotImplementedError, match="no client"):
            JevBackend().decide({"stagnation_steps": 1})

    def test_jev_backend_wraps_a_client_and_falls_back_on_garbage(self) -> None:
        from apedp.decision.policy import JevBackend

        class Good:
            def decide(self, state):
                return {"decision": "continue", "probabilities": {"continue": 1.0}}

        assert JevBackend(Good()).decide({})["decision"] == "continue"

        class Garbage:
            def decide(self, state):
                return "not a dict"

        with pytest.raises(TypeError):
            JevBackend(Garbage()).decide({})

    def test_local_search_controller_always_returns_an_action(self) -> None:
        from apedp.decision.policy import LocalSearchController, SearchAction

        for state in (
            {"stagnation_steps": 0, "budget_left": 1.0},
            {"stagnation_steps": 50, "budget_left": 0.99},
            {"stagnation_steps": 0, "budget_left": 0.0},
        ):
            d = LocalSearchController().decide(state)
            assert isinstance(d.action, SearchAction), d
            assert isinstance(d.reason, str) and d.reason, d

    def test_search_action_enum_covers_stop_and_handoff(self) -> None:
        from apedp.decision.policy import SearchAction

        names = {a.name for a in SearchAction}
        assert {"STOP_STAGNATION", "HUMAN_REVIEW", "CONTINUE"} <= names, names


class TestDocumentIngestion:
    """apedp/ingestion/ingestion_docs.py -- DOCX/PDF text extraction."""

    def test_docx_round_trip_through_a_real_zip(self, tmp_path) -> None:
        """Build a minimal valid DOCX and read it back."""
        import zipfile

        from apedp.ingestion.ingestion_docs import extract_text_from_docx

        docx = tmp_path / "sample.docx"
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>Hello ingestion</w:t></w:r></w:p></w:body></w:document>"
        )
        with zipfile.ZipFile(docx, "w") as z:
            z.writestr("word/document.xml", document_xml)

        text = extract_text_from_docx(str(docx))
        assert "Hello ingestion" in text, text

    def test_get_text_dispatches_on_extension_and_ignores_others(
        self, tmp_path,
    ) -> None:
        from apedp.ingestion.ingestion_docs import get_text

        assert get_text(str(tmp_path / "nothing.txt")) == ""

    def test_load_apedp_documents_skips_missing_files(self, tmp_path) -> None:
        """An empty directory yields {} -- absent, not an error, not empty text."""
        from apedp.ingestion.ingestion_docs import load_apedp_documents

        assert load_apedp_documents(str(tmp_path)) == {}


class TestExternalScienceNoiseLadder:
    """apedp/ingestion/external_science.py -- the reproducible noise ladder."""

    @staticmethod
    def _dataset():
        from apedp.ingestion.external_science import AdapterDataset

        return AdapterDataset(
            x_train={"x": np.array([1.0, 2.0, 3.0, 4.0])},
            y_train=np.array([2.0, 4.0, 6.0, 8.0]),
            x_holdout={"x": np.array([5.0, 6.0])},
            y_holdout=np.array([10.0, 12.0]),
            dataset_hash="abc123",
            provenance={"source": "unit-test"},
            firewall={},
        )

    def test_zero_noise_leaves_the_target_exactly_alone(self) -> None:
        from apedp.ingestion.external_science import apply_noise_ladder

        (variant,) = apply_noise_ladder(self._dataset(), (0.0,), seed=1)
        assert np.array_equal(variant.y_train, self._dataset().y_train)
        assert variant.provenance["clean_holdout_preserved"] is True

    def test_same_seed_reproduces_and_a_different_seed_does_not(self) -> None:
        from apedp.ingestion.external_science import apply_noise_ladder

        ds = self._dataset()
        a = apply_noise_ladder(ds, (0.01,), seed=7)[0]
        b = apply_noise_ladder(ds, (0.01,), seed=7)[0]
        c = apply_noise_ladder(ds, (0.01,), seed=8)[0]
        assert np.array_equal(a.y_train, b.y_train), "seed must pin the draw"
        assert not np.array_equal(a.y_train, c.y_train), "seed must actually matter"

    def test_holdout_stays_clean_at_every_noise_level(self) -> None:
        """The point of a noise ladder: train is degraded, holdout is not."""
        from apedp.ingestion.external_science import apply_noise_ladder

        ds = self._dataset()
        for variant in apply_noise_ladder(ds, (0.0, 0.05), seed=3):
            assert np.array_equal(variant.y_holdout, ds.y_holdout)

    def test_negative_noise_is_rejected(self) -> None:
        from apedp.ingestion.external_science import apply_noise_ladder

        with pytest.raises(ValueError, match="non-negative"):
            apply_noise_ladder(self._dataset(), (-0.1,))

    def test_manifest_declares_itself_infrastructure_only(self) -> None:
        """A blind-discovery manifest must not claim results it has not got."""
        from apedp.ingestion.external_science import NISTWebBookCase, build_nist_manifest

        base = self._dataset()
        ds = type(base)(
            x_train=base.x_train, y_train=base.y_train,
            x_holdout=base.x_holdout, y_holdout=base.y_holdout,
            dataset_hash=base.dataset_hash,
            provenance={**base.provenance, "file_sha256": "deadbeef",
                        "groups": ["a"], "holdout_groups": ["b"],
                        "train_rows": 4, "holdout_rows": 2},
            firewall=base.firewall,
        )
        case = NISTWebBookCase(case_id="c1", compound="Water", property_name="BoilingPoint")
        manifest = build_nist_manifest(
            ds, case, seeds=(1, 2), grammar={}, operator_limits={},
        )
        assert manifest["protocol_status"] == "INFRASTRUCTURE_ONLY", manifest
        assert manifest["source_sha256"] == "deadbeef"
        assert manifest["seeds"] == [1, 2]
        assert manifest["case_id"] == "c1", manifest


class TestApiHealthRoutes:
    """apedp/api/routes/health.py -- liveness, which must need no credential."""

    def test_health_reports_ok_and_a_version(self) -> None:
        import asyncio

        from apedp import __version__
        from apedp.api.routes.health import health

        out = asyncio.run(health())
        assert out["status"] == "ok", out
        assert out["version"] == __version__, out

    def test_health_paths_are_exempt_from_authentication(self) -> None:
        """A liveness check behind a credential cannot gate traffic."""
        from apedp.api import AUTH_EXEMPT_PATHS

        assert {"/health", "/docs", "/openapi.json"} <= AUTH_EXEMPT_PATHS

    def test_health_router_registers_its_routes(self) -> None:
        from apedp.api.routes.health import router

        paths = {getattr(r, "path", None) for r in router.routes}
        assert "/health" in paths, paths


class TestPersistenceSerializer:
    """apedp/persistence/serializer.py -- model (de)serialization."""

    def test_json_round_trip_preserves_a_plain_payload(self) -> None:
        from apedp.persistence.serializer import from_json, to_json

        payload = {"family": "power_law", "params": {"A": 2.5, "p": 1.5}}
        assert from_json(to_json(payload)) == payload

    def test_round_trip_is_stable_across_repeats(self) -> None:
        from apedp.persistence.serializer import from_json, to_json

        payload = {"nested": {"a": [1, 2, 3], "b": None}}
        first = to_json(payload)
        assert first == to_json(payload), "serialization must be deterministic"
        assert from_json(first) == from_json(to_json(from_json(first)))

    def test_malformed_json_raises_rather_than_returning_junk(self) -> None:
        from apedp.persistence.serializer import from_json

        with pytest.raises(json.JSONDecodeError):
            from_json("{not json")


class TestEquationSeedsAreStructured:
    """apedp/equations/seeds.py -- the discovery seed catalogue."""

    def test_seeds_are_a_nonempty_list_of_records(self) -> None:
        from apedp.equations.seeds import seed_pages

        pages = seed_pages()
        assert isinstance(pages, list) and pages, pages
        for entry in pages:
            assert isinstance(entry, dict), entry
            assert {"page", "domain", "license", "name"} <= set(entry), entry
            assert entry["name"], entry

    def test_every_seed_declares_a_licence(self) -> None:
        """A seed catalogue without provenance cannot be cited or checked."""
        from apedp.equations.seeds import seed_pages

        missing = [e["name"] for e in seed_pages() if not e.get("license")]
        assert not missing, f"seeds with no declared licence: {missing}"

    def test_seed_pages_are_serializable(self) -> None:
        """A catalogue that cannot cross a process boundary is not a catalogue."""
        from apedp.equations.seeds import seed_pages

        assert json.loads(json.dumps(seed_pages()))
