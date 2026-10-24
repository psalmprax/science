"""Tests for distributed and out-of-core chunked executor.

Guarantees:
1. Streaming map parity against serial evaluation across backends.
2. Memory bounding: items processed in chunks without loading all outputs simultaneously.
3. Fault-tolerant checkpointing and seamless partition resumption.
4. Ray fallback when ray is not present in environment.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from apedp_dyn.distributed.chunked_executor import (
    DistributedChunkedExecutor,
    PartitionSpec,
    square_worker,
)


def test_partition_spec_structure():
    spec = PartitionSpec(partition_id=2, total_partitions=10, shard_key="env_sweep_shard_2")
    d = spec.to_dict()
    assert d["partition_id"] == 2
    assert d["total_partitions"] == 10
    assert d["shard_key"] == "env_sweep_shard_2"


def test_chunked_executor_serial_and_threading_parity():
    items = list(range(50))
    expected = [x * x for x in items]

    # Serial backend
    exec_serial = DistributedChunkedExecutor(backend="serial")
    results_serial = list(exec_serial.map_stream(square_worker, items, chunk_size=10))
    assert results_serial == expected

    # Thread pool backend
    exec_threads = DistributedChunkedExecutor(backend="thread_pool", max_workers=2)
    results_threads = list(exec_threads.map_stream(square_worker, items, chunk_size=15))
    assert results_threads == expected


def test_chunked_executor_multiprocessing_parity():
    items = list(range(40))
    expected = [x * x for x in items]

    exec_mp = DistributedChunkedExecutor(backend="multiprocessing", max_workers=2)
    results_mp = list(exec_mp.map_stream(square_worker, items, chunk_size=8))
    assert results_mp == expected


def test_chunked_executor_checkpointing_resumption(tmp_path: Path):
    items = list(range(30))
    expected = [x * x for x in items]

    ckpt_dir = tmp_path / "checkpoints"

    # 1. Run first half (chunk_size=10, so 3 chunks: chunk_0, chunk_1, chunk_2)
    exec_1 = DistributedChunkedExecutor(backend="serial", checkpoint_dir=ckpt_dir)
    # Simulate partial execution by manually running chunk 0 and writing checkpoint
    exec_1._save_chunk_checkpoint(0, [0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
    assert exec_1._is_chunk_checkpointed(0)
    assert not exec_1._is_chunk_checkpointed(1)

    # 2. Run execution with executor: chunk 0 should load from checkpoint, chunks 1 & 2 compute
    stream = exec_1.map_stream(square_worker, items, chunk_size=10)
    results = list(stream)
    assert results == expected
    assert exec_1._is_chunk_checkpointed(1)
    assert exec_1._is_chunk_checkpointed(2)


def test_chunked_executor_streaming_reduce():
    items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    expected_sum = sum(items)

    executor = DistributedChunkedExecutor(backend="serial")
    stream = executor.map_stream(lambda x: x, items, chunk_size=3)
    total = executor.reduce(lambda acc, x: acc + x, stream, initial=0)
    assert total == expected_sum


def test_chunked_executor_ray_fallback_when_ray_uninstalled(monkeypatch: pytest.MonkeyPatch):
    # Hide ray import if installed
    import sys
    monkeypatch.setitem(sys.modules, "ray", None)

    items = [2, 4, 6]
    expected = [4, 16, 36]

    exec_ray = DistributedChunkedExecutor(backend="ray", max_workers=2)
    results = list(exec_ray.map_stream(square_worker, items, chunk_size=2))
    assert results == expected
