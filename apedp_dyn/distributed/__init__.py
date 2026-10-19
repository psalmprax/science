"""Distributed execution and out-of-core chunking framework for APEDP-X.

Provides partitionable map-reduce sweeps, streaming generators for bounded memory footprints,
disk checkpointing for fault-tolerant recovery, and cluster hooks (Ray/multiprocessing).
"""
from __future__ import annotations

from apedp_dyn.distributed.chunked_executor import (
    DistributedChunkedExecutor,
    DistributedTask,
    PartitionSpec,
)

__all__ = [
    "DistributedChunkedExecutor",
    "DistributedTask",
    "PartitionSpec",
]
