# Databricks notebook source
"""APEDP ingestion subpackage for external data sources."""
from apedp.ingestion.external_benchmarks import (
    AIFeynmanBenchmarkSuite,
    FeynmanBenchmarkCase,
    NACA0012TransonicBenchmark,
    NanocrystallineMicrocompressionBenchmark,
)
from apedp.ingestion.external_science import (
    NISTWebBookAdapter,
    NISTWebBookCase,
    apply_noise_ladder,
    build_nist_manifest,
    run_nist_blind_discovery,
)
from apedp.ingestion.ingestion_academic_apis import INSTITUTIONAL_TARGETS, list_targets
from apedp.ingestion.ingestion_docs import load_apedp_documents
from apedp.ingestion.ingestion_web_equations import harvest_wikipedia_equations

__all__ = [
    "INSTITUTIONAL_TARGETS",
    "AIFeynmanBenchmarkSuite",
    "FeynmanBenchmarkCase",
    "NACA0012TransonicBenchmark",
    "NISTWebBookAdapter",
    "NISTWebBookCase",
    "NanocrystallineMicrocompressionBenchmark",
    "apply_noise_ladder",
    "build_nist_manifest",
    "harvest_wikipedia_equations",
    "list_targets",
    "load_apedp_documents",
    "run_nist_blind_discovery",
]
