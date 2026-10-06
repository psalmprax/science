"""Tests for APEDP 1000+ Equations Catalog across 7 scientific domains.

Verifies:
  1. Total equation count exceeds 1,000 equations.
  2. All 7 domains (classical, feynman, kinetics, circuits, astrophysics, biology, nonlinear) are represented.
  3. Every catalog problem materializes a valid DynProblem / ProblemSpec with truth equations.
  4. Symbolic parameter integrity: equation symbols map to declared states, inputs, or parameters.
  5. Integration with _problem_modules() and FastAPI endpoints (/api/physics/systems, /api/discover/registry).
  6. High performance: cached listing, category filtering, search, and pagination.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from apedp_dyn.problems.base import DynProblem, ProblemSpec
from apedp_dyn.problems.catalog import (
    count_catalog_equations,
    get_catalog_modules,
    get_problem,
)
from apedp_dyn.ui.server import app

client = TestClient(app, raise_server_exceptions=False)


class Test1000EquationsCatalog:
    """Verifies the 1,000+ equations catalog and procedural generator."""

    def test_total_equation_count_exceeds_1000(self):
        """The library must scale up to >= 1,000 ground-truth dynamical equations."""
        cat = get_catalog_modules()
        total_equations = count_catalog_equations()
        assert total_equations >= 1000, f"Expected >= 1000 equations, found {total_equations}"
        assert len(cat) >= 300, f"Expected >= 300 dynamical systems, found {len(cat)}"

    def test_all_seven_domains_present(self):
        """Catalog must cover all 7 foundational scientific domains specified in the task."""
        cat = get_catalog_modules()
        categories = {m.category for m in cat.values()}
        expected = {
            "classical",
            "feynman",
            "kinetics",
            "circuits",
            "astrophysics",
            "biology",
            "nonlinear",
        }
        assert expected <= categories, f"Missing domains: {expected - categories}"

        for dom in expected:
            dom_mods = [m for m in cat.values() if m.category == dom]
            assert len(dom_mods) >= 20, f"Domain {dom} under-represented with {len(dom_mods)} systems"
            dom_eqs = sum(len(m.equation_strings) for m in dom_mods)
            assert dom_eqs >= 50, f"Domain {dom} has only {dom_eqs} equations"

    def test_problem_spec_alias_contract(self):
        """ProblemSpec must be an alias for DynProblem in base.py."""
        assert ProblemSpec is DynProblem

    def test_catalog_problem_spec_materialization(self):
        """Catalog module materializes valid DynProblem with states, parameters, and truth equations."""
        prob = get_problem("cat_phys_harm_001")
        assert prob is not None
        spec = prob.spec()
        assert isinstance(spec, DynProblem)
        assert len(spec.states) >= 1
        assert len(spec.truth_equations) == len(spec.states)
        assert len(spec.parameters) >= 1
        assert "t_span" in spec.train
        assert "x0" in spec.train
        assert "u_of_t" in spec.train

    def test_symbolic_parameter_integrity(self):
        """Equations must reference declared parameters symbolically rather than baking in float strings."""
        sample_ids = [
            "cat_phys_harm_001",
            "cat_phys_anharm_021",
            "cat_feynman_cyclo_025",
            "cat_circ_rlc_006",
            "cat_nlin_lorenz_001",
            "cat_clas_chain_osc_001",
            "cat_biol_seir_variant_001",
        ]
        cat = get_catalog_modules()
        for pid in sample_ids:
            mod = cat[pid]
            spec = mod.spec()
            param_names = {p.name for p in spec.parameters}
            state_names = {s.name for s in spec.states}
            input_names = {u.name for u in spec.inputs}
            allowed = param_names | state_names | input_names | {"t", "sin", "cos", "tanh", "exp", "sqrt"}
            for eq in spec.truth_equations:
                # Every equation should be non-empty and parsed by SymPy
                assert eq.sympy_expr is not None
                expr_symbols = {str(s) for s in eq.sympy_expr.free_symbols}
                assert expr_symbols <= allowed, f"{pid} equation {eq} has unexpected symbols: {expr_symbols - allowed}"

    def test_sprott_chaotic_systems_all_present(self):
        """Sprott A through S (19 minimal quadratic systems) must all be registered."""
        cat = get_catalog_modules()
        for letter in "abcdefghijklmnopqrs":
            sprott_id = f"cat_nlin_sprott_{letter}"
            matching = [k for k in cat if k.startswith(sprott_id)]
            assert len(matching) >= 1, f"Missing Sprott system {letter.upper()}"


class TestPhysicsSystemsAPI:
    """Verifies the /api/physics/systems endpoint with catalog, filtering, and pagination."""

    def test_catalog_endpoint_lists_all_systems(self):
        """Returns all systems (PS20 core + 1000+ catalog) when called without filters."""
        resp = client.get("/api/physics/systems")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 300
        ids = {it["id"] for it in items}
        # Core benchmark systems preserved
        assert {"pendulum", "duffing", "coupled", "lotka_volterra", "van_der_pol", "logistic"} <= ids
        # Catalog systems included
        assert any(it.startswith("cat_phys_") for it in ids)
        assert any(it.startswith("cat_feynman_") for it in ids)
        assert any(it.startswith("cat_chem_") for it in ids)

    def test_category_filtering(self):
        """Filters systems by scientific domain."""
        resp = client.get("/api/physics/systems?category=feynman")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 20
        assert all(it.get("category") == "feynman" for it in items)

    def test_search_filtering(self):
        """Searches systems by name, id, or equation substring."""
        resp = client.get("/api/physics/systems?search=lorenz")
        assert resp.status_code == 200
        items = resp.json()
        assert len(items) >= 1
        assert any("lorenz" in it["name"].lower() or "lorenz" in it["id"].lower() for it in items)

    def test_pagination_limit_and_offset(self):
        """Paginates the catalog cleanly."""
        resp1 = client.get("/api/physics/systems?limit=10&offset=0")
        assert resp1.status_code == 200
        items1 = resp1.json()
        assert len(items1) == 10

        resp2 = client.get("/api/physics/systems?limit=10&offset=10")
        assert resp2.status_code == 200
        items2 = resp2.json()
        assert len(items2) == 10

        ids1 = [it["id"] for it in items1]
        ids2 = [it["id"] for it in items2]
        assert set(ids1).isdisjoint(set(ids2)), "Pagination offset must advance cleanly"

    def test_discover_registry_on_catalog_system(self):
        """Runs Level 2 open discovery on a catalog problem module."""
        resp = client.post("/api/discover/registry", json={"system": "cat_phys_harm_001"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["system"] == "cat_phys_harm_001"
        assert data["verdict"] in ("unique_support", "multi_hypothesis", "insufficient_evidence", "no_valid_candidate")
        assert "pareto_candidates" in data
        assert "plot_series" in data
