"""Durable local orchestration for complete discovery-pipeline jobs."""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

from apedp.pipeline import DiscoveryPipeline


class _SyncRedis(Protocol):
    """The synchronous ``redis.Redis`` surface :class:`RedisPipelineJobManager` relies on.

    ``redis.Redis`` is annotated for both the sync and async clients, so every
    command is typed as ``Awaitable | value`` and the variadic ``eval`` arguments
    are declared as ``*keys_and_args: list[Any]``. Pinning the eight commands
    actually used here keeps the manager type-checkable without asserts.
    """

    def ping(self) -> bool: ...

    def get(self, name: str) -> Any: ...

    def set(self, name: str, value: Any, **kwargs: Any) -> Any: ...

    def mget(self, keys: list[str], *args: Any) -> list[Any]: ...

    def rpush(self, name: str, *values: Any) -> int: ...

    def scan_iter(self, match: str | None = None, count: int | None = None) -> Iterator[str]: ...

    def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any: ...

    def blpop(self, keys: str, timeout: int = 0) -> tuple[Any, Any] | None: ...

logger = logging.getLogger(__name__)
_TERMINAL = frozenset({"SUCCEEDED", "REFUSED", "FAILED", "CANCELLED"})
_LEASE_SECONDS = 30.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stale(timestamp: str | None) -> bool:
    if not timestamp:
        return True
    try:
        value = datetime.fromisoformat(timestamp)
    except ValueError:
        return True
    return datetime.now(timezone.utc) - value > timedelta(seconds=_LEASE_SECONDS)


class PipelineJobManager:
    """Run pipeline jobs with durable state and shared-filesystem ownership.

    Multiple manager instances may point at the same state directory. Atomic
    claim directories prevent duplicate execution, while heartbeats allow a new
    worker to recover a job whose previous process died. This remains a local
    filesystem backend rather than a replacement for a distributed broker.
    """

    def __init__(self, state_dir: str = "pipeline_jobs", max_workers: int = 2):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="apedp-pipeline")
        self._lock = threading.RLock()
        self._futures: dict[str, Future] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._load_records()
        self._schedule_pending()

    def _path(self, job_id: str) -> Path:
        return self.state_dir / f"{job_id}.json"

    def _claim_path(self, job_id: str) -> Path:
        return self.state_dir / f".{job_id}.claim"

    def _load_records(self) -> None:
        for path in self.state_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                job_id = record.get("job_id")
                if not isinstance(job_id, str):
                    continue
                status = record.get("status")
                if status in {"RUNNING", "CANCELLATION_REQUESTED"} and _stale(record.get("heartbeat_at")):
                    # Rewind the in-flight attempt so recovery does not consume
                    # another configured retry merely because a worker died.
                    record["status"] = "PENDING"
                    record["attempt"] = max(0, int(record.get("attempt", 0)) - 1)
                    record["error"] = "requeued after stale worker lease"
                    record["progress"] = "restart recovery"
                    record["lease_owner"] = None
                    record["heartbeat_at"] = None
                    record["updated_at"] = _now()
                    self._write(record)
                self._jobs[job_id] = record
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Ignoring unreadable pipeline job record %s: %s", path, exc)

    def _schedule_pending(self) -> None:
        with self._lock:
            for job_id, record in self._jobs.items():
                if record.get("status") == "PENDING" and job_id not in self._futures:
                    self._futures[job_id] = self._executor.submit(self._execute, job_id)

    def _write(self, record: dict[str, Any]) -> None:
        path = self._path(str(record["job_id"]))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        os.replace(temporary, path)

    def _update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            record = dict(self._jobs[job_id])
            record.update(changes)
            record["updated_at"] = _now()
            self._jobs[job_id] = record
            self._write(record)
            return dict(record)

    def _try_claim(self, job_id: str) -> bool:
        try:
            self._claim_path(job_id).mkdir()
            self._update(job_id, lease_owner=self.worker_id, heartbeat_at=_now())
            return True
        except FileExistsError:
            return False

    def _release_claim(self, job_id: str) -> None:
        try:
            self._claim_path(job_id).rmdir()
        except OSError:
            pass

    def submit(self, parameters: dict[str, Any], max_attempts: int = 1) -> dict[str, Any]:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        missing = sorted({"csv_path", "target_column"} - set(parameters))
        if missing:
            raise ValueError(f"missing pipeline parameter(s): {missing}")
        job_id = uuid.uuid4().hex
        now = _now()
        record = {
            "job_id": job_id, "status": "PENDING", "progress": "queued",
            "parameters": dict(parameters), "attempt": 0, "max_attempts": int(max_attempts),
            "submitted_at": now, "updated_at": now, "result": None, "error": None,
            "cancel_requested": False, "lease_owner": None, "heartbeat_at": None,
        }
        with self._lock:
            self._jobs[job_id] = record
            self._write(record)
            self._futures[job_id] = self._executor.submit(self._execute, job_id)
        return dict(record)

    def _heartbeat(self, job_id: str, stop: threading.Event) -> None:
        while not stop.wait(max(1.0, _LEASE_SECONDS / 3.0)):
            try:
                self._update(job_id, heartbeat_at=_now())
            except (KeyError, OSError):
                return

    def _execute(self, job_id: str) -> None:
        if not self._try_claim(job_id):
            return
        heartbeat_stop = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        try:
            with self._lock:
                record = self._jobs.get(job_id)
                if record is None:
                    return
                if record.get("cancel_requested"):
                    self._update(job_id, status="CANCELLED", progress="cancelled before start", finished_at=_now())
                    return
                self._update(
                    job_id, status="RUNNING", progress="ingesting and discovering",
                    attempt=int(record["attempt"]) + 1, started_at=record.get("started_at", _now()),
                    error=None, lease_owner=self.worker_id, heartbeat_at=_now(),
                )
                parameters = dict(record["parameters"])
            heartbeat_thread = threading.Thread(target=self._heartbeat, args=(job_id, heartbeat_stop), daemon=True)
            heartbeat_thread.start()
            try:
                pipeline = DiscoveryPipeline(
                    output_dir=str(parameters.pop("output_dir", "pipeline_results")),
                    max_bad_row_fraction=float(parameters.pop("max_bad_row_fraction", 0.2)),
                    strict_units=bool(parameters.get("strict_units", False)),
                )
                result = pipeline.run(**parameters)
                status = "SUCCEEDED" if result.get("status") == "PERSISTED" else "REFUSED"
                with self._lock:
                    cancelled = bool(self._jobs[job_id].get("cancel_requested"))
                self._update(job_id, status="CANCELLED" if cancelled else status,
                             progress="cancelled after pipeline completion" if cancelled else "complete",
                             result=result, finished_at=_now())
            # Any failure must be persisted against the job, never propagated
            # out of a worker thread: BLE001 is deliberate here.
            except Exception as exc:  # noqa: BLE001 - worker boundary
                with self._lock:
                    current = self._jobs[job_id]
                    retry = int(current["attempt"]) < int(current["max_attempts"]) and not current.get("cancel_requested")
                if retry:
                    self._update(job_id, status="PENDING", progress="retry queued", error=str(exc),
                                 lease_owner=None, heartbeat_at=None)
                    with self._lock:
                        self._futures[job_id] = self._executor.submit(self._execute, job_id)
                else:
                    self._update(job_id, status="FAILED", progress="failed",
                                 error=f"{type(exc).__name__}: {exc}", finished_at=_now())
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)
            self._release_claim(job_id)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return dict(record) if record is not None else None

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda item: item.get("submitted_at", ""), reverse=True)
            return [dict(record) for record in records[:limit]]

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            record = self._jobs[job_id]
            if record["status"] in _TERMINAL:
                return dict(record)
            future = self._futures.get(job_id)
            if future is not None and future.cancel():
                return self._update(job_id, status="CANCELLED", progress="cancelled before start", finished_at=_now())
            return self._update(job_id, status="CANCELLATION_REQUESTED", cancel_requested=True,
                               progress="cancellation requested; active worker will finish safely")

    def retry(self, job_id: str) -> dict[str, Any]:
        record = self.get(job_id)
        if record is None:
            raise KeyError(job_id)
        if record["status"] not in {"FAILED", "REFUSED", "CANCELLED"}:
            raise ValueError(f"job '{job_id}' is not retryable from status {record['status']}")
        return self.submit(dict(record["parameters"]), max_attempts=int(record.get("max_attempts", 1)))

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=not wait)


class RedisPipelineJobManager:
    """Redis Streams-style queue backend for API/worker separation.

    The API process only enqueues and reads records. One or more separate worker
    processes call :meth:`run_worker`; Redis locks and expirations provide
    cross-process ownership and stale-worker recovery. Install the optional
    ``apedp[orchestration]`` extra and provide ``APEDP_REDIS_URL``.
    """

    def __init__(self, redis_url: str, queue_name: str = "apedp:pipeline", lease_seconds: int = 60):
        # ``import_module`` rather than a local ``import redis``: the redis client
        # is untyped at function scope, so mypy binds the name to ``None``.
        try:
            redis = import_module("redis")
        except ImportError as exc:
            raise RuntimeError("Redis backend requires optional dependency: pip install 'apedp[orchestration]'") from exc
        if lease_seconds < 5:
            raise ValueError("lease_seconds must be at least 5 seconds")
        self.redis: _SyncRedis = redis.Redis.from_url(redis_url, decode_responses=True)
        self.redis.ping()
        self.queue_name = queue_name
        self.prefix = f"{queue_name}:job:"
        self.lease_seconds = int(lease_seconds)
        self.worker_id = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"

    def _key(self, job_id: str) -> str:
        return f"{self.prefix}{job_id}"

    def _read(self, job_id: str) -> dict[str, Any] | None:
        value = self.redis.get(self._key(job_id))
        return json.loads(value) if value else None

    def _write(self, record: dict[str, Any]) -> dict[str, Any]:
        record = dict(record)
        record["updated_at"] = _now()
        self.redis.set(self._key(record["job_id"]), json.dumps(record, default=str))
        return record

    def _update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        record = self._read(job_id)
        if record is None:
            raise KeyError(job_id)
        record.update(changes)
        return self._write(record)

    def submit(self, parameters: dict[str, Any], max_attempts: int = 1) -> dict[str, Any]:
        missing = sorted({"csv_path", "target_column"} - set(parameters))
        if missing:
            raise ValueError(f"missing pipeline parameter(s): {missing}")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        now = _now()
        record = {
            "job_id": uuid.uuid4().hex, "status": "PENDING", "progress": "queued",
            "parameters": dict(parameters), "attempt": 0, "max_attempts": int(max_attempts),
            "submitted_at": now, "updated_at": now, "result": None, "error": None,
            "cancel_requested": False, "lease_owner": None, "heartbeat_at": None,
        }
        self._write(record)
        self.redis.rpush(self.queue_name, record["job_id"])
        return record

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self._read(job_id)

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        keys = list(self.redis.scan_iter(match=f"{self.prefix}*", count=limit * 2))
        records = [json.loads(value) for value in self.redis.mget(keys) if value]
        records.sort(key=lambda item: item.get("submitted_at", ""), reverse=True)
        return records[:limit]

    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self._read(job_id)
        if record is None:
            raise KeyError(job_id)
        if record["status"] in _TERMINAL:
            return record
        return self._update(job_id, status="CANCELLATION_REQUESTED", cancel_requested=True,
                            progress="cancellation requested")

    def retry(self, job_id: str) -> dict[str, Any]:
        record = self._read(job_id)
        if record is None:
            raise KeyError(job_id)
        if record["status"] not in {"FAILED", "REFUSED", "CANCELLED"}:
            raise ValueError(f"job '{job_id}' is not retryable from status {record['status']}")
        return self.submit(dict(record["parameters"]), int(record.get("max_attempts", 1)))

    def _run_one(self, job_id: str) -> None:
        record = self._read(job_id)
        if record is None or record.get("status") in _TERMINAL:
            return
        lock_key = f"{self._key(job_id)}:lease"
        lease_token = f"{self.worker_id}:{uuid.uuid4().hex}"
        if not self.redis.set(lock_key, lease_token, nx=True, ex=self.lease_seconds):
            return
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            renew_script = (
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('expire', KEYS[1], ARGV[2]) else return 0 end"
            )
            while not heartbeat_stop.wait(max(1.0, self.lease_seconds / 3.0)):
                renewed = self.redis.eval(renew_script, 1, lock_key, lease_token, self.lease_seconds)
                if not renewed:
                    return
                try:
                    self._update(job_id, heartbeat_at=_now(), lease_owner=self.worker_id)
                except KeyError:
                    return

        thread: threading.Thread | None = None
        try:
            if record.get("cancel_requested"):
                self._update(job_id, status="CANCELLED", progress="cancelled before start", finished_at=_now())
                return
            attempt = int(record.get("attempt", 0)) + 1
            self._update(job_id, status="RUNNING", progress="ingesting and discovering", attempt=attempt,
                         lease_owner=self.worker_id, heartbeat_at=_now(), error=None)
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            parameters = dict(record["parameters"])
            pipeline = DiscoveryPipeline(
                output_dir=str(parameters.pop("output_dir", "pipeline_results")),
                max_bad_row_fraction=float(parameters.pop("max_bad_row_fraction", 0.2)),
                strict_units=bool(parameters.get("strict_units", False)),
            )
            result = pipeline.run(**parameters)
            current = self._read(job_id) or {}
            cancelled = bool(current.get("cancel_requested"))
            self._update(job_id, status="CANCELLED" if cancelled else ("SUCCEEDED" if result.get("status") == "PERSISTED" else "REFUSED"),
                         progress="complete", result=result, finished_at=_now())
        # Any failure must be persisted and retried through the queue rather
        # than killing the worker: BLE001 is deliberate here.
        except Exception as exc:  # noqa: BLE001 - worker boundary
            current = self._read(job_id) or record
            if int(current.get("attempt", 0)) < int(current.get("max_attempts", 1)) and not current.get("cancel_requested"):
                self._update(job_id, status="PENDING", progress="retry queued", error=str(exc),
                             lease_owner=None, heartbeat_at=None)
                self.redis.rpush(self.queue_name, job_id)
            else:
                self._update(job_id, status="FAILED", progress="failed",
                             error=f"{type(exc).__name__}: {exc}", finished_at=_now())
        finally:
            heartbeat_stop.set()
            # Joined only when it was actually started.
            #
            # The thread used to be constructed before the ``try``, so a job
            # cancelled before it started reached this ``finally`` with an
            # unstarted thread and ``join`` raised RuntimeError -- which is not
            # caught here, so it propagated out of ``_run_one`` into
            # ``run_worker``'s loop and killed the worker process. Worse, the
            # lease release below the join never ran, so the job stayed locked
            # for a full lease period. PipelineJobManager._execute already
            # guards this with an ``is not None`` check on a thread it assigns
            # later; this now does the same rather than relying on a Thread
            # object that exists but was never run.
            if thread is not None:
                thread.join(timeout=1.0)
            release_script = (
                "if redis.call('get', KEYS[1]) == ARGV[1] then "
                "return redis.call('del', KEYS[1]) else return 0 end"
            )
            self.redis.eval(release_script, 1, lock_key, lease_token)

    def run_worker(self, stop_event: threading.Event | None = None, poll_timeout: int = 5) -> None:
        """Consume jobs until ``stop_event`` is set; intended for a worker process."""
        stop = stop_event or threading.Event()
        while not stop.is_set():
            item = self.redis.blpop(self.queue_name, timeout=poll_timeout)
            if item is not None:
                self._run_one(item[1])


def main() -> None:
    """Run a Redis worker: ``python -m apedp.orchestration``."""
    import argparse

    parser = argparse.ArgumentParser(description="APEDP durable pipeline worker")
    parser.add_argument("--redis-url", default=os.environ.get("APEDP_REDIS_URL"))
    parser.add_argument("--queue", default=os.environ.get("APEDP_REDIS_QUEUE", "apedp:pipeline"))
    parser.add_argument("--lease-seconds", type=int, default=60)
    args = parser.parse_args()
    if not args.redis_url:
        parser.error("--redis-url or APEDP_REDIS_URL is required")
    RedisPipelineJobManager(args.redis_url, args.queue, args.lease_seconds).run_worker()


if __name__ == "__main__":
    main()
