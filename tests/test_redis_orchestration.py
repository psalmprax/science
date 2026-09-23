"""Live Redis orchestration checks.

The test is intentionally skipped unless a Redis instance is reachable. CI
provides one as a service; local runs can opt in with APEDP_REDIS_URL.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import pytest


@pytest.mark.integration
def test_redis_worker_executes_pipeline_job(monkeypatch):
    redis = pytest.importorskip("redis")
    redis_url = os.environ.get("APEDP_REDIS_URL", "redis://localhost:6379/15")
    client = redis.Redis.from_url(redis_url, decode_responses=True)
    try:
        client.ping()
    except redis.RedisError as exc:
        pytest.skip(f"Redis is unavailable at {redis_url}: {exc}")

    from apedp.orchestration import RedisPipelineJobManager

    class FakePipeline:
        def __init__(self, **_kwargs):
            pass

        def run(self, **_kwargs):
            return {"status": "PERSISTED", "model_id": "redis-test-model"}

    monkeypatch.setattr("apedp.orchestration.DiscoveryPipeline", FakePipeline)
    queue = f"apedp:test:{uuid.uuid4().hex}"
    manager = RedisPipelineJobManager(redis_url, queue_name=queue, lease_seconds=5)
    stop = threading.Event()
    worker = threading.Thread(target=manager.run_worker, args=(stop, 1), daemon=True)
    worker.start()
    try:
        submitted = manager.submit({"csv_path": "data.csv", "target_column": "y"})
        deadline = time.monotonic() + 5.0
        current = manager.get(submitted["job_id"])
        while current["status"] not in {"SUCCEEDED", "FAILED", "REFUSED", "CANCELLED"}:
            if time.monotonic() >= deadline:
                pytest.fail(f"Redis job did not finish: {current}")
            time.sleep(0.02)
            current = manager.get(submitted["job_id"])
        assert current["status"] == "SUCCEEDED"
        assert current["result"]["model_id"] == "redis-test-model"
        assert current["attempt"] == 1
    finally:
        stop.set()
        worker.join(timeout=3.0)
        client.delete(manager._key(submitted["job_id"]) if "submitted" in locals() else "")
        client.delete(queue)
        client.delete(f"{manager._key(submitted['job_id'])}:lease" if "submitted" in locals() else "")
