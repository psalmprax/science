"""Distributed and out-of-core chunked executor for APEDP-X.

Enables scaling discovery sweeps and candidate evaluations beyond single-node
in-memory limits via:
1. Streaming chunked generators (O(chunk_size) peak RAM instead of O(N)).
2. Fault-tolerant checkpointing with automatic partition resumption.
3. Pluggable compute backends:
   - "serial": In-process deterministic execution for testing.
   - "thread_pool": Multi-threaded execution for I/O and Rust-released GIL kernels.
   - "multiprocessing": Multi-core process-isolated pool.
   - "ray": Remote cluster execution via Ray task graphs (falls back gracefully if ray not installed).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Generic, Iterable, Iterator, Literal, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class PartitionSpec:
    """Specification of an execution partition/shard."""
    partition_id: int
    total_partitions: int
    shard_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "total_partitions": self.total_partitions,
            "shard_key": self.shard_key,
        }


@dataclass
class DistributedTask(Generic[T, R]):
    """Atomic unit of distributed work."""
    task_id: str
    payload: T
    partition: PartitionSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def square_worker(x: int) -> int:
    """Serializable helper worker function for distributed task demonstration."""
    return x * x


class DistributedChunkedExecutor:
    """Manages partitionable, memory-bounded, out-of-core sweep executions."""

    def __init__(
        self,
        backend: Literal["serial", "thread_pool", "multiprocessing", "ray"] = "multiprocessing",
        max_workers: int | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        self.backend = backend
        self.max_workers = max_workers
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _chunk_items(self, items: Sequence[T], chunk_size: int) -> Iterator[list[T]]:
        """Slice a sequence into chunks of maximum length chunk_size."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        for i in range(0, len(items), chunk_size):
            yield list(items[i : i + chunk_size])

    def _checkpoint_path(self, chunk_idx: int) -> Path | None:
        if not self.checkpoint_dir:
            return None
        return self.checkpoint_dir / f"chunk_{chunk_idx:06d}.json"

    def _is_chunk_checkpointed(self, chunk_idx: int) -> bool:
        p = self._checkpoint_path(chunk_idx)
        return p is not None and p.is_file()

    def _save_chunk_checkpoint(self, chunk_idx: int, results: list[Any]) -> None:
        p = self._checkpoint_path(chunk_idx)
        if p:
            try:
                # Attempt json serialization if payload allows; otherwise mark completion marker
                p.write_text(json.dumps({"chunk_idx": chunk_idx, "results": results}, default=str), encoding="utf-8")
            except Exception:
                p.write_text(json.dumps({"chunk_idx": chunk_idx, "status": "completed"}), encoding="utf-8")

    def _load_chunk_checkpoint(self, chunk_idx: int) -> list[Any] | None:
        p = self._checkpoint_path(chunk_idx)
        if not p or not p.is_file():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data.get("results")
        except Exception:
            return None

    def map_stream(
        self,
        func: Callable[[T], R],
        items: Sequence[T],
        chunk_size: int = 100,
    ) -> Iterator[R]:
        """Streamingly execute func over items in chunks.

        Yields individual results as they are evaluated, ensuring that memory usage
        does not exceed O(chunk_size * element_size).
        """
        chunks = list(self._chunk_items(items, chunk_size))

        for chunk_idx, chunk in enumerate(chunks):
            # 1. Checkpoint resumption
            if self._is_chunk_checkpointed(chunk_idx):
                cached = self._load_chunk_checkpoint(chunk_idx)
                if cached is not None:
                    for res in cached:
                        yield res
                    continue

            # 2. Execution across requested backend
            if self.backend == "serial":
                chunk_results = [func(item) for item in chunk]
            elif self.backend == "thread_pool":
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    chunk_results = list(pool.map(func, chunk))
            elif self.backend == "multiprocessing":
                with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                    chunk_results = list(pool.map(func, chunk))
            elif self.backend == "ray":
                chunk_results = self._map_ray(func, chunk)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")

            # 3. Checkpoint write
            self._save_chunk_checkpoint(chunk_idx, chunk_results)

            # 4. Stream results
            for res in chunk_results:
                yield res

    def _map_ray(self, func: Callable[[T], R], chunk: list[T]) -> list[R]:
        """Dispatch task onto Ray cluster if available."""
        try:
            import ray
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            remote_fn = ray.remote(func)
            futures = [remote_fn.remote(item) for item in chunk]
            return ray.get(futures)
        except (ImportError, TypeError):
            # Fall back cleanly to multiprocessing pool if Ray is not installed
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as pool:
                return list(pool.map(func, chunk))

    def reduce(
        self,
        reducer: Callable[[R, R], R],
        stream: Iterable[R],
        initial: R | None = None,
    ) -> R:
        """Incrementally fold a streaming result set with constant auxiliary memory."""
        it = iter(stream)
        if initial is None:
            try:
                acc = next(it)
            except StopIteration:
                raise ValueError("reduce() of empty sequence with no initial value")
        else:
            acc = initial

        for val in it:
            acc = reducer(acc, val)
        return acc
