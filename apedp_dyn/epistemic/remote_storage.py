"""Remote & Distributed Object Storage Backend for Epistemic Ledgers.

Provides durable, cloud-agnostic persistence for hypothesis lineage records,
burned holdout registries, and trial ledgers across ephemeral execution runtimes
(Google Colab, Kaggle, ephemeral Docker containers, and HPC nodes).

Enforces AGENTS.md §1.2 ("Anti-Snooping Holdout Discipline"):
Confirmation holdouts burned during iterative falsification must be permanently
preserved to prevent data snooping even when compute nodes terminate abruptly.

Supported Backends:
- LocalStorageBackend: High-performance local filesystem with cross-platform atomic locking.
- MemoryStorageBackend: In-memory thread-safe backend for fast testing and ephemeral scoping.
- S3StorageBackend: Amazon S3 and S3-compatible object stores (MinIO, Ceph, Cloudflare R2).
- GCSStorageBackend: Google Cloud Storage with precondition generation checking.
"""
from __future__ import annotations

import abc
import contextlib
import datetime
import os
import threading
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any


class StorageLockError(RuntimeError):
    """Raised when an exclusive storage lock cannot be acquired within the timeout."""
    pass


class BaseStorageBackend(abc.ABC):
    """Abstract interface for epistemic ledger object storage backends."""

    @abc.abstractmethod
    def read_bytes(self, key: str) -> bytes:
        """Read binary data from the specified key."""
        raise NotImplementedError

    @abc.abstractmethod
    def write_bytes(self, key: str, data: bytes) -> None:
        """Write binary data to the specified key."""
        raise NotImplementedError

    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """Check whether the given key exists in storage."""
        raise NotImplementedError

    @abc.abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from storage. Returns True if deleted, False if non-existent."""
        raise NotImplementedError

    @abc.abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]:
        """List all object keys matching the given prefix."""
        raise NotImplementedError

    @abc.abstractmethod
    def acquire_lock(self, key: str, timeout: float = 30.0) -> str:
        """Acquire an exclusive lock for key. Returns lock token on success or raises StorageLockError."""
        raise NotImplementedError

    @abc.abstractmethod
    def release_lock(self, key: str, lock_token: str) -> bool:
        """Release an exclusive lock. Returns True if successfully released."""
        raise NotImplementedError

    @contextlib.contextmanager
    def lock(self, key: str, timeout: float = 30.0, retry_interval: float = 0.05) -> Generator[str, None, None]:
        """Context manager for distributed/file locking on an epistemic ledger resource."""
        token = self.acquire_lock(key, timeout=timeout)
        try:
            yield token
        finally:
            self.release_lock(key, token)

    def sync_to_remote(self, local_path: Path | str, remote_key: str) -> None:
        """Atomically upload a local ledger file to remote storage under a lock."""
        path = Path(local_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local file does not exist: {path}")

        with self.lock(remote_key):
            with open(path, "rb") as f:
                data = f.read()
            self.write_bytes(remote_key, data)

    def pull_from_remote(self, remote_key: str, local_path: Path | str) -> bool:
        """Download remote storage object into a local path atomically under lock."""
        path = Path(local_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        with self.lock(remote_key):
            if not self.exists(remote_key):
                return False
            data = self.read_bytes(remote_key)

        # Atomic local write
        temp_file = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
        try:
            with open(temp_file, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_file, path)
            return True
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    pass

    def export_snapshot(self, local_path: Path | str, snapshot_name: str | None = None) -> str:
        """Create an immutable, timestamped snapshot of the current ledger state in remote storage."""
        path = Path(local_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local file does not exist: {path}")

        now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if snapshot_name is None:
            snapshot_name = f"snapshots/{path.stem}_{now_str}_{uuid.uuid4().hex[:8]}{path.suffix}"
        else:
            snapshot_name = f"snapshots/{snapshot_name}"

        with open(path, "rb") as f:
            data = f.read()

        self.write_bytes(snapshot_name, data)
        return snapshot_name


class LocalStorageBackend(BaseStorageBackend):
    """Local filesystem storage backend with robust cross-platform lock files."""

    def __init__(self, base_path: Path | str):
        self.base_path = Path(base_path).expanduser().resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._held_locks: dict[str, int] = {}
        self._thread_lock = threading.RLock()

    def _resolve(self, key: str) -> Path:
        normalized = key.lstrip("/")
        return self.base_path / normalized

    def read_bytes(self, key: str) -> bytes:
        p = self._resolve(key)
        if not p.is_file():
            raise FileNotFoundError(f"Key not found in local storage: {key}")
        with open(p, "rb") as f:
            return f.read()

    def write_bytes(self, key: str, data: bytes) -> None:
        p = self._resolve(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        temp_p = p.with_name(f".{p.name}.tmp.{uuid.uuid4().hex}")
        try:
            with open(temp_p, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_p, p)
        finally:
            if temp_p.exists():
                try:
                    temp_p.unlink()
                except OSError:
                    pass

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def delete(self, key: str) -> bool:
        p = self._resolve(key)
        if p.is_file():
            p.unlink()
            return True
        return False

    def list_keys(self, prefix: str = "") -> list[str]:
        keys: list[str] = []
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.startswith(".lock_") or file.endswith(".lock"):
                    continue
                full = Path(root) / file
                rel = full.relative_to(self.base_path).as_posix()
                if rel.startswith(prefix):
                    keys.append(rel)
        return sorted(keys)

    def acquire_lock(self, key: str, timeout: float = 30.0, stale_timeout: float = 60.0) -> str:
        lock_file = self.base_path / f".lock_{key.replace('/', '_')}"
        token = uuid.uuid4().hex
        deadline = time.time() + timeout

        while True:
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                with self._thread_lock:
                    self._held_locks[token] = fd
                    payload = f"{token}:{time.time()}\n".encode()
                    os.write(fd, payload)
                return token
            except FileExistsError:
                # Check for stale lock
                try:
                    mtime = lock_file.stat().st_mtime
                    if (time.time() - mtime) > stale_timeout:
                        try:
                            lock_file.unlink()
                        except OSError:
                            pass
                except OSError:
                    pass

                if time.time() >= deadline:
                    raise StorageLockError(
                        f"Timed out after {timeout:.1f}s waiting to acquire lock on '{key}'."
                    )
                time.sleep(0.05)

    def release_lock(self, key: str, lock_token: str) -> bool:
        had_held_fd = False
        with self._thread_lock:
            fd = self._held_locks.pop(lock_token, None)
            if fd is not None:
                had_held_fd = True
                try:
                    os.close(fd)
                except OSError:
                    pass

        lock_file = self.base_path / f".lock_{key.replace('/', '_')}"
        if lock_file.is_file():
            try:
                content = lock_file.read_bytes().decode("utf-8")
                file_token = content.split(":", 1)[0].strip()
                if file_token == lock_token:
                    lock_file.unlink()
                    return True
                return False
            except OSError:
                return False
        return had_held_fd


class MemoryStorageBackend(BaseStorageBackend):
    """In-memory thread-safe storage backend ideal for testing and ephemeral scoping."""

    def __init__(self):
        self._data: dict[str, bytes] = {}
        self._locks: dict[str, str] = {}  # key -> token
        self._lock = threading.RLock()

    def read_bytes(self, key: str) -> bytes:
        with self._lock:
            if key not in self._data:
                raise FileNotFoundError(f"Key not found in memory storage: {key}")
            return self._data[key]

    def write_bytes(self, key: str, data: bytes) -> None:
        with self._lock:
            self._data[key] = bytes(data)

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    def list_keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            return sorted([k for k in self._data if k.startswith(prefix)])

    def acquire_lock(self, key: str, timeout: float = 30.0) -> str:
        token = uuid.uuid4().hex
        deadline = time.time() + timeout
        while True:
            with self._lock:
                if key not in self._locks:
                    self._locks[key] = token
                    return token
            if time.time() >= deadline:
                raise StorageLockError(
                    f"Timed out after {timeout:.1f}s waiting to acquire memory lock on '{key}'."
                )
            time.sleep(0.02)

    def release_lock(self, key: str, lock_token: str) -> bool:
        with self._lock:
            if self._locks.get(key) == lock_token:
                del self._locks[key]
                return True
            return False


class S3StorageBackend(BaseStorageBackend):
    """Amazon S3 / S3-compatible distributed object storage backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        s3_client: Any = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self._client = s3_client
        self._active_locks: dict[str, str] = {}

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3  # type: ignore[import-untyped]
                kwargs: dict[str, Any] = {}
                if self.region_name:
                    kwargs["region_name"] = self.region_name
                if self.endpoint_url:
                    kwargs["endpoint_url"] = self.endpoint_url
                self._client = boto3.client("s3", **kwargs)
            except ImportError as err:
                raise ImportError(
                    "boto3 is required for S3StorageBackend. "
                    "Install with: pip install boto3"
                ) from err
        return self._client

    def _full_key(self, key: str) -> str:
        clean = key.lstrip("/")
        return f"{self.prefix}/{clean}" if self.prefix else clean

    def read_bytes(self, key: str) -> bytes:
        full_key = self._full_key(key)
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=full_key)
            return resp["Body"].read()
        except Exception as exc:
            raise FileNotFoundError(f"Key {full_key} not found in bucket {self.bucket}: {exc}") from exc

    def write_bytes(self, key: str, data: bytes) -> None:
        full_key = self._full_key(key)
        self.client.put_object(Bucket=self.bucket, Key=full_key, Body=data)

    def exists(self, key: str) -> bool:
        full_key = self._full_key(key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        full_key = self._full_key(key)
        if self.exists(key):
            self.client.delete_object(Bucket=self.bucket, Key=full_key)
            return True
        return False

    def list_keys(self, prefix: str = "") -> list[str]:
        full_prefix = self._full_key(prefix)
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                if self.prefix and k.startswith(self.prefix + "/"):
                    k = k[len(self.prefix) + 1 :]
                keys.append(k)
        return sorted(keys)

    def acquire_lock(self, key: str, timeout: float = 30.0, stale_timeout: float = 60.0) -> str:
        lock_key = self._full_key(f"locks/{key}.lock")
        token = uuid.uuid4().hex
        deadline = time.time() + timeout

        while True:
            lock_is_active = False
            try:
                resp = self.client.get_object(Bucket=self.bucket, Key=lock_key)
                body = resp["Body"].read().decode("utf-8")
                parts = body.strip().split(":")
                lock_time = float(parts[1]) if len(parts) > 1 else 0.0
                if (time.time() - lock_time) <= stale_timeout:
                    lock_is_active = True
                else:
                    # Break expired lock
                    self.client.delete_object(Bucket=self.bucket, Key=lock_key)
            except Exception:
                # Lock does not exist
                lock_is_active = False

            if not lock_is_active:
                try:
                    payload = f"{token}:{time.time()}".encode()
                    self.client.put_object(Bucket=self.bucket, Key=lock_key, Body=payload)
                    self._active_locks[token] = lock_key
                    return token
                except Exception:
                    pass

            if time.time() >= deadline:
                raise StorageLockError(f"Timed out after {timeout:.1f}s acquiring S3 lock on {key}.")
            time.sleep(0.05)

    def release_lock(self, key: str, lock_token: str) -> bool:
        lock_key = self._active_locks.pop(lock_token, self._full_key(f"locks/{key}.lock"))
        try:
            resp = self.client.get_object(Bucket=self.bucket, Key=lock_key)
            body = resp["Body"].read().decode("utf-8")
            file_token = body.split(":", 1)[0].strip()
            if file_token == lock_token:
                self.client.delete_object(Bucket=self.bucket, Key=lock_key)
                return True
            return False
        except Exception:
            return False


class GCSStorageBackend(BaseStorageBackend):
    """Google Cloud Storage (GCS) distributed backend."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        client: Any = None,
        project: str | None = None,
    ):
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")
        self.project = project
        self._client = client
        self._bucket_obj: Any = None
        self._active_locks: dict[str, str] = {}

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from google.cloud import storage  # type: ignore[import-untyped]
                self._client = storage.Client(project=self.project)
            except ImportError as err:
                raise ImportError(
                    "google-cloud-storage is required for GCSStorageBackend. "
                    "Install with: pip install google-cloud-storage"
                ) from err
        return self._client

    @property
    def bucket(self) -> Any:
        if self._bucket_obj is None:
            self._bucket_obj = self.client.bucket(self.bucket_name)
        return self._bucket_obj

    def _full_key(self, key: str) -> str:
        clean = key.lstrip("/")
        return f"{self.prefix}/{clean}" if self.prefix else clean

    def read_bytes(self, key: str) -> bytes:
        blob = self.bucket.blob(self._full_key(key))
        if not blob.exists():
            raise FileNotFoundError(f"Key {key} not found in GCS bucket {self.bucket_name}")
        return blob.download_as_bytes()

    def write_bytes(self, key: str, data: bytes) -> None:
        blob = self.bucket.blob(self._full_key(key))
        blob.upload_from_string(data)

    def exists(self, key: str) -> bool:
        blob = self.bucket.blob(self._full_key(key))
        return blob.exists()

    def delete(self, key: str) -> bool:
        blob = self.bucket.blob(self._full_key(key))
        if blob.exists():
            blob.delete()
            return True
        return False

    def list_keys(self, prefix: str = "") -> list[str]:
        full_prefix = self._full_key(prefix)
        blobs = self.client.list_blobs(self.bucket_name, prefix=full_prefix)
        keys: list[str] = []
        for b in blobs:
            name = b.name
            if self.prefix and name.startswith(self.prefix + "/"):
                name = name[len(self.prefix) + 1 :]
            keys.append(name)
        return sorted(keys)

    def acquire_lock(self, key: str, timeout: float = 30.0, stale_timeout: float = 60.0) -> str:
        lock_key = self._full_key(f"locks/{key}.lock")
        token = uuid.uuid4().hex
        deadline = time.time() + timeout
        blob = self.bucket.blob(lock_key)

        while True:
            if blob.exists():
                try:
                    content = blob.download_as_bytes().decode("utf-8")
                    parts = content.strip().split(":")
                    lock_time = float(parts[1]) if len(parts) > 1 else 0.0
                    if (time.time() - lock_time) > stale_timeout:
                        blob.delete()
                except Exception:
                    pass

            try:
                payload = f"{token}:{time.time()}".encode()
                blob.upload_from_string(payload, if_generation_match=0)
                self._active_locks[token] = lock_key
                return token
            except Exception:
                pass

            if time.time() >= deadline:
                raise StorageLockError(f"Timed out after {timeout:.1f}s acquiring GCS lock on {key}.")
            time.sleep(0.1)

    def release_lock(self, key: str, lock_token: str) -> bool:
        lock_key = self._active_locks.pop(lock_token, self._full_key(f"locks/{key}.lock"))
        blob = self.bucket.blob(lock_key)
        try:
            if blob.exists():
                content = blob.download_as_bytes().decode("utf-8")
                file_token = content.split(":", 1)[0].strip()
                if file_token == lock_token:
                    blob.delete()
                    return True
                return False
            return True
        except Exception:
            return False


def get_storage_backend(
    backend_type: str | None = None,
    base_path: Path | str | None = None,
    bucket: str | None = None,
    prefix: str = "",
    **kwargs: Any,
) -> BaseStorageBackend:
    """Factory creating configured storage backend from explicit arguments or environment variables.

    Environment Variables:
        APEDP_STORAGE_BACKEND: 'local', 'memory', 's3', 'gcs'
        APEDP_STORAGE_BUCKET: S3/GCS bucket name
        APEDP_STORAGE_PREFIX: Object prefix / subpath
        APEDP_STORAGE_DIR: Local directory path
    """
    resolved_type = (backend_type or os.environ.get("APEDP_STORAGE_BACKEND") or "local").lower()

    if resolved_type == "memory":
        return MemoryStorageBackend()

    elif resolved_type == "s3":
        b_name = bucket or os.environ.get("APEDP_STORAGE_BUCKET")
        if not b_name:
            raise ValueError("S3 storage backend requires bucket name via 'bucket' argument or APEDP_STORAGE_BUCKET.")
        p_name = prefix or os.environ.get("APEDP_STORAGE_PREFIX", "")
        return S3StorageBackend(bucket=b_name, prefix=p_name, **kwargs)

    elif resolved_type == "gcs":
        b_name = bucket or os.environ.get("APEDP_STORAGE_BUCKET")
        if not b_name:
            raise ValueError("GCS storage backend requires bucket name via 'bucket' argument or APEDP_STORAGE_BUCKET.")
        p_name = prefix or os.environ.get("APEDP_STORAGE_PREFIX", "")
        return GCSStorageBackend(bucket=b_name, prefix=p_name, **kwargs)

    elif resolved_type == "local":
        p_dir = base_path or os.environ.get("APEDP_STORAGE_DIR")
        if p_dir is None:
            from apedp_dyn.notebook_adapter import configure_storage
            _, resolved_persist, _, _ = configure_storage()
            p_dir = resolved_persist
        return LocalStorageBackend(base_path=p_dir)

    else:
        raise ValueError(
            f"Unsupported storage backend type '{resolved_type}'. "
            "Supported backends: 'local', 'memory', 's3', 'gcs'."
        )
