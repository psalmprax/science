"""APEDP ingestion subpackage for external data sources."""
from apedp.ingestion.external_benchmarks import (
    AIFeynmanBenchmarkSuite,
    FeynmanBenchmarkCase,
    NACA0012TransonicBenchmark,
    NanocrystallineMicrocompressionBenchmark,
)
from apedp.ingestion.ingestion_academic_apis import INSTITUTIONAL_TARGETS, list_targets
from apedp.ingestion.ingestion_docs import load_apedp_documents
from apedp.ingestion.ingestion_web_equations import harvest_wikipedia_equations

__all__ = [
    "INSTITUTIONAL_TARGETS",
    "AIFeynmanBenchmarkSuite",
    "FeynmanBenchmarkCase",
    "NACA0012TransonicBenchmark",
    "NanocrystallineMicrocompressionBenchmark",
    "harvest_wikipedia_equations",
    "list_targets",
    "load_apedp_documents",
]
