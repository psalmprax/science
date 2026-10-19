"""APEDP-Distributed: Autonomous Distributed & Out-of-Core Chunked Sweep Execution Framework."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure root package can be imported if running within monorepo
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from apedp_dyn.distributed.chunked_executor import (
    DistributedChunkedExecutor,
    DistributedTask,
    PartitionSpec,
    square_worker,
)

__version__ = "0.1.0"

__all__ = [
    "DistributedChunkedExecutor",
    "DistributedTask",
    "PartitionSpec",
    "square_worker",
]
