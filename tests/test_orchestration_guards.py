"""Exercise the guards in apedp/orchestration.py that nothing has ever run.

Why this file exists
--------------------
A branch-coverage run put ``apedp.orchestration`` at 17 of 80 branches with 63
untaken decision outcomes -- the worst module in the repo. Reading them showed
the untaken guards are almost entirely *defensive*: ``if max_workers < 1``,
``if max_attempts < 1``, ``if limit < 1``, ``if record is None``,
``if not isinstance(job_id, str)``, the cancel-before-start paths, the
retry-exhaustion path. None of them had ever executed.

That is worth acting on only if an unexercised guard is actually broken, so
these tests are written to *find out* rather than to raise a coverage number.
One of them found a real bug immediately:

``RedisPipelineJobManager._run_one`` constructed its heartbeat thread before
the ``try`` and started it after the cancel check. Cancelling a job that was
still queued therefore reached the ``finally`` with an unstarted thread, where
``thread.join(timeout=1.0)`` raises ``RuntimeError: cannot join thread before
it is started``. Nothing catches it there, so it escaped ``_run_one`` into
``run_worker``'s polling loop and terminated the worker process -- and because
the lease-release ``eval`` sits below the join, it never ran either, so the job
stayed locked for a full lease period. One operator cancelling one queued job
took down the worker.

The sibling ``PipelineJobManager._execute`` never had the bug: it assigns
``None`` and creates the thread later, then guards the join. These tests pin
both backends so the two cannot drift apart again.

No Redis server is required. ``RedisPipelineJobManager.__init__`` calls
``import_module("redis")`` and then ``ping()``, so the module import is
patched to return a stand-in whose ``Redis.from_url`` hands back an in-memory
fake. Everything under test is this module's own logic; the fake only has to be
faithful about the eight commands the ``_SyncRedis`` protocol declares.
"""
from __future__ import annotations

import fnmatch
import json
import threading
import types
from pathlib import Path

import pytest

import apedp.orchestration as orch
from apedp.orchestration import (
    PipelineJobManager,
    RedisPipelineJobManager,
    _stale,
)


class FakeRedis:
    """In-memory stand-in for the eight commands ``_SyncRedis`` declares.

    ``eval`` is a stub: the manager uses it for two Lua scripts (renew and
    release a lease) whose *server-side* semantics are redis's job, not this
    module's. What the tests assert about them is that they are issued.
    """

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.queues: dict[str, list[str]] = {}
        self.evals: list[tuple] = []
        self.set_calls: list[tuple] = []

    def ping(self) -> bool:
        return True

    def get(self, name: str):
        return self.kv.get(name)

    def set(self, name: str, value, **kwargs):
        self.set_calls.append((name, value, kwargs))
        self.kv[name] = str(value)
        return True

    def mget(self, keys, *args):
        return [self.kv.get(k) for k in keys]

    def rpush(self, name: str, *values) -> int:
        self.queues.setdefault(name, []).extend(values)
        return len(values)

    def scan_iter(self, match: str | None = None, count: int | None = None):
        for key in list(self.kv):
            if match is None or fnmatch.fnmatch(key, match):
                yield key

    def eval(self, script, numkeys, *keys_and_args):
        self.evals.append((script, numkeys, keys_and_args))
        return 1

    def blpop(self, keys, timeout: int = 0):
        return None


@pytest.fixture
def fake_redis(monkeypatch) -> FakeRedis:
    instance = FakeRedis()
    stand_in = types.SimpleNamespace(
        Redis=types.SimpleNamespace(from_url=lambda *a, **k: instance))
    monkeypatch.setattr(orch, "import_module", lambda name: stand_in)
    return instance


@pytest.fixture
def redis_manager(fake_redis) -> RedisPipelineJobManager:
    return RedisPipelineJobManager("redis://fake/0")


def _job_record(m: RedisPipelineJobManager, job_id: str) -> dict:
    return json.loads(m.redis.get(m._key(job_id)))


def _params() -> dict:
    return {"csv_path": "data.csv", "target_column": "y"}


class TestRedisCancelBeforeStart:
    """The regression: an unstarted thread in the ``finally``, and a stuck lease."""

    def test_cancelling_a_queued_job_does_not_raise(self, redis_manager):
        m = redis_manager
        record = m.submit(_params(), max_attempts=1)
        job_id = record["job_id"]
        # Operator cancels while the job is still on the queue; the worker then
        # picks it up and finds cancel_requested already set.
        m.redis.set(m._key(job_id),
                    json.dumps({**record, "cancel_requested": True}))

        m._run_one(job_id)  # must not raise

    def test_the_cancelled_job_actually_reaches_CANCELLED(self, redis_manager):
        """The RuntimeError previously masked the state transition's aftermath.

        The status was written before the failure, so this passed even while the
        call blew up -- which is why the bug could sit unnoticed behind a
        correct-looking record.
        """
        m = redis_manager
        record = m.submit(_params(), max_attempts=1)
        job_id = record["job_id"]
        m.redis.set(m._key(job_id),
                    json.dumps({**record, "cancel_requested": True}))

        m._run_one(job_id)

        assert _job_record(m, job_id)["status"] == "CANCELLED"

    def test_the_lease_is_released_after_an_early_return(self, redis_manager):
        """The damage the RuntimeError caused beyond the traceback.

        The release ``eval`` sits below the join in the same ``finally``, so a
        join that raised meant the lease was never dropped. The job would then
        be unprocessable by any other worker until the lease expired.
        """
        m = redis_manager
        record = m.submit(_params(), max_attempts=1)
        job_id = record["job_id"]
        m.redis.set(m._key(job_id),
                    json.dumps({**record, "cancel_requested": True}))

        m._run_one(job_id)

        lock_key = f"{m._key(job_id)}:lease"
        assert lock_key in m.redis.kv, "fixture drifted: no lease was ever taken"
        assert m.redis.evals, (
            "no lease-release script was issued; the job is still locked")

    def test_a_terminal_job_is_skipped_without_taking_a_lease(self, redis_manager):
        """``if record is None or record.get("status") in _TERMINAL`` -- untaken."""
        m = redis_manager
        record = m.submit(_params(), max_attempts=1)
        job_id = record["job_id"]
        m.redis.set(m._key(job_id), json.dumps({**record, "status": "SUCCEEDED"}))

        before = len(m.redis.set_calls)
        m._run_one(job_id)
        assert len(m.redis.set_calls) == before, (
            "a terminal job must not acquire a lease")

    def test_an_unknown_job_id_is_skipped(self, redis_manager):
        m = redis_manager
        m._run_one("no-such-job")  # must not raise
        assert not m.redis.evals

    def test_run_worker_survives_a_cancelled_job(self, redis_manager, monkeypatch):
        """The path that actually mattered: one cancel must not kill the worker.

        ``run_worker`` polls in a loop with no exception handling, so anything
        escaping ``_run_one`` terminates the process. This drives the loop once
        through the cancelling job and asserts the worker is still willing to
        pick up the next one.
        """
        m = redis_manager
        record = m.submit(_params(), max_attempts=1)
        job_id = record["job_id"]
        m.redis.set(m._key(job_id),
                    json.dumps({**record, "cancel_requested": True}))

        stop = threading.Event()
        processed: list[str] = []

        def fake_blpop(keys, timeout=0):
            if processed:
                stop.set()
                return None
            processed.append(job_id)
            return (keys, job_id)

        m.redis.blpop = fake_blpop  # type: ignore[method-assign]
        m.run_worker(stop, poll_timeout=0)

        assert processed == [job_id], "the worker never dequeued the job"
        assert _job_record(m, job_id)["status"] == "CANCELLED", (
            "the worker stopped mid-loop; a cancelled job took it down")


class TestHeartbeatThreadLifecycle:
    """The fix narrows when the join runs, so the started case needs pinning too.

    Guarding the join with ``is not None`` is only safe if the started path
    still joins. A change that skipped the join unconditionally would leave the
    heartbeat thread running past job completion -- invisible in the cancel
    tests, and leaving a thread that wakes up every lease/3 seconds to write to
    a job record that is already finished.
    """

    def test_a_job_that_actually_runs_starts_and_joins_its_heartbeat(
            self, redis_manager, monkeypatch):
        m = redis_manager
        joined: list[str] = []
        real_thread_cls = threading.Thread

        class _Thread(real_thread_cls):
            def join(self, timeout=None):  # type: ignore[override]
                joined.append(self.name or "heartbeat")
                super().join(timeout=timeout)

        monkeypatch.setattr(orch.threading, "Thread", _Thread)

        class _StubPipeline:
            def __init__(self, **_kwargs):
                pass

            def run(self, **kwargs):
                return {"status": "PERSISTED"}

        monkeypatch.setattr(orch, "DiscoveryPipeline", _StubPipeline)

        record = m.submit(_params(), max_attempts=1)
        m._run_one(record["job_id"])

        assert joined, (
            "the heartbeat thread was never joined on the normal path; the "
            "cancel-path fix must not have disabled the join entirely")
        assert _job_record(m, record["job_id"])["status"] == "SUCCEEDED", (
            "fixture drifted: a clean run should persist SUCCEEDED")

    def test_the_lease_is_released_on_the_normal_path_too(self, redis_manager,
                                                           monkeypatch):
        m = redis_manager

        class _StubPipeline:
            def __init__(self, **_kwargs):
                pass

            def run(self, **kwargs):
                return {"status": "PERSISTED"}

        monkeypatch.setattr(orch, "DiscoveryPipeline", _StubPipeline)
        record = m.submit(_params(), max_attempts=1)
        m._run_one(record["job_id"])

        assert m.redis.evals, "no lease-release script issued on the happy path"


class TestLocalBackendGuards:
    """``PipelineJobManager`` -- the guards that were never executed."""

    def test_max_workers_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError, match="max_workers must be positive"):
            PipelineJobManager(state_dir=str(tmp_path / "s"), max_workers=0)

    def test_submit_rejects_non_positive_max_attempts(self, tmp_path):
        m = PipelineJobManager(state_dir=str(tmp_path / "s"))
        try:
            for bad in (0, -1):
                with pytest.raises(ValueError, match="max_attempts must be positive"):
                    m.submit(_params(), max_attempts=bad)
        finally:
            m.shutdown()

    def test_submit_names_the_missing_parameters(self, tmp_path):
        m = PipelineJobManager(state_dir=str(tmp_path / "s"))
        try:
            with pytest.raises(ValueError) as exc:
                m.submit({"csv_path": "d.csv"})
            assert "target_column" in str(exc.value), str(exc.value)
        finally:
            m.shutdown()

    def test_list_rejects_a_non_positive_limit(self, tmp_path):
        m = PipelineJobManager(state_dir=str(tmp_path / "s"))
        try:
            with pytest.raises(ValueError, match="limit must be positive"):
                m.list(limit=0)
        finally:
            m.shutdown()

    def test_cancel_and_retry_reject_an_unknown_job(self, tmp_path):
        m = PipelineJobManager(state_dir=str(tmp_path / "s"))
        try:
            assert m.get("nope") is None
            with pytest.raises(KeyError):
                m.cancel("nope")
            with pytest.raises(KeyError):
                m.retry("nope")
        finally:
            m.shutdown()

    def test_retry_refuses_a_job_that_is_not_in_a_failed_state(self, tmp_path):
        m = PipelineJobManager(state_dir=str(tmp_path / "s"))
        try:
            record = m.submit(_params())
            with pytest.raises(ValueError, match="not retryable"):
                m.retry(record["job_id"])
        finally:
            m.shutdown(wait=False)

    def test_unreadable_records_are_skipped_not_fatal(self, tmp_path):
        """``if not isinstance(job_id, str): continue`` -- untaken.

        A stray .json in the state directory must not stop the manager from
        starting; it is logged and ignored.
        """
        state = tmp_path / "s"
        state.mkdir()
        (state / "junk.json").write_text("{not json", encoding="utf-8")
        (state / "no_id.json").write_text(json.dumps({"status": "PENDING"}),
                                           encoding="utf-8")
        (state / "bad_id.json").write_text(json.dumps({"job_id": 17,
                                                       "status": "PENDING"}),
                                            encoding="utf-8")
        (state / "good.json").write_text(json.dumps({"job_id": "good",
                                                     "status": "SUCCEEDED"}),
                                         encoding="utf-8")

        m = PipelineJobManager(state_dir=str(state))
        try:
            assert "good" in m._jobs
            assert 17 not in m._jobs
            assert list(m._jobs) == ["good"], (
                f"expected only the valid record to load, got {list(m._jobs)}")
        finally:
            m.shutdown()


class TestStalenessGuard:
    """``_stale`` -- both early returns were untaken."""

    def test_a_missing_timestamp_is_stale(self):
        assert _stale(None) is True
        assert _stale("") is True

    def test_an_unparseable_timestamp_is_stale(self):
        # A corrupt heartbeat must be treated as expired, not crash the loader
        # into the except clause that swallows the record entirely.
        assert _stale("not-a-timestamp") is True

    def test_a_recent_timestamp_is_not_stale(self):
        assert _stale(orch._now()) is False


class TestRedisBackendGuards:
    """``RedisPipelineJobManager`` validation, which is also untaken."""

    def test_lease_seconds_has_a_floor(self, fake_redis):
        with pytest.raises(ValueError, match="lease_seconds must be at least 5"):
            RedisPipelineJobManager("redis://fake/0", lease_seconds=4)

    def test_submit_names_missing_parameters(self, redis_manager):
        with pytest.raises(ValueError) as exc:
            redis_manager.submit({"csv_path": "d.csv"})
        assert "target_column" in str(exc.value)

    def test_submit_rejects_non_positive_max_attempts(self, redis_manager):
        with pytest.raises(ValueError, match="max_attempts must be positive"):
            redis_manager.submit(_params(), max_attempts=0)

    def test_list_rejects_a_non_positive_limit(self, redis_manager):
        with pytest.raises(ValueError, match="limit must be positive"):
            redis_manager.list(limit=0)

    def test_cancel_and_retry_reject_an_unknown_job(self, redis_manager):
        assert redis_manager.get("nope") is None
        with pytest.raises(KeyError):
            redis_manager.cancel("nope")
        with pytest.raises(KeyError):
            redis_manager.retry("nope")

    def test_retry_refuses_a_job_that_is_not_in_a_failed_state(self, redis_manager):
        record = redis_manager.submit(_params())
        with pytest.raises(ValueError, match="not retryable"):
            redis_manager.retry(record["job_id"])

    def test_list_only_returns_this_queue_prefix(self, redis_manager):
        """The ``scan_iter`` match pattern must not leak another queue's jobs."""
        redis_manager.submit(_params())
        redis_manager.redis.kv["other:job:xyz"] = json.dumps({"job_id": "xyz"})
        assert [r["job_id"] for r in redis_manager.list()] != ["xyz"]
        assert all(r["job_id"] != "xyz" for r in redis_manager.list())

    def test_missing_optional_dependency_is_named(self, monkeypatch):
        def boom(name):
            raise ImportError("no redis")

        monkeypatch.setattr(orch, "import_module", boom)
        with pytest.raises(RuntimeError, match="optional dependency"):
            RedisPipelineJobManager("redis://fake/0")


def test_state_dir_is_created(tmp_path):
    """The mkdir the constructor relies on, asserted rather than assumed."""
    target = tmp_path / "deep" / "nested" / "jobs"
    m = PipelineJobManager(state_dir=str(target))
    try:
        assert Path(target).is_dir()
    finally:
        m.shutdown()