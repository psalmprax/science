"""Full-Lineage Experiment & Candidate Ledger with Anti-Snooping Guard.

Maintains an immutable record of the complete hypothesis search process:
  - Lineage tracking: parent hypotheses, feature sets, parameter choices, and variants.
  - Multi-state lifecycle: PROPOSED, PRE_SCREEN_REJECTED, CALIBRATION_FAILED,
    DEVELOPMENT_PASSED, CONFIRMATION_REJECTED, ECONOMIC_EVALUATION_FAILED, CONFIRMED_VALID.
  - Dataset exposure tracking & Holdout Contamination Guard:
    Prevents data-snooping by permanently burning any confirmation holdout slice
    inspected by a failed candidate so it cannot be re-used by successor iterations.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from apedp_dyn.epistemic.remote_storage import BaseStorageBackend


class CandidateLifecycleState(str, Enum):
    """Explicit lifecycle states for candidate hypotheses."""
    PROPOSED = "proposed"
    PRE_SCREEN_REJECTED = "pre_screen_rejected"
    CALIBRATION_FAILED = "calibration_failed"
    DEVELOPMENT_PASSED = "development_passed"
    CONFIRMATION_REJECTED = "confirmation_rejected"
    ECONOMIC_EVALUATION_FAILED = "economic_evaluation_failed"
    CONFIRMED_VALID = "confirmed_valid"
    UNRESOLVED = "unresolved"


class DatasetRole(str, Enum):
    """Role of a dataset slice in the discovery and validation protocol."""
    TRAINING_DEVELOPMENT = "training_development"
    CONFIRMATION_HOLDOUT = "confirmation_holdout"
    PROSPECTIVE_LIVE = "prospective_live"


class HoldoutContaminationError(RuntimeError):
    """Raised when an iterative candidate attempts to inspect a burned/contaminated holdout slice."""
    pass


class LedgerIntegrityError(RuntimeError):
    """Raised when the on-disk ledger is damaged, edited, or truncated.

    The burned-holdout registry is rebuilt from this file, so a line that cannot
    be read or that breaks the hash chain is a lost burn, not a cosmetic defect.
    """


#: How many distinct candidates may inspect one confirmation-holdout slice.
#: A confirmation holdout is a single-use resource: every further candidate
#: scored on it turns it into a selection set, whether or not an earlier one
#: failed. ``None`` disables the budget (the pre-budget behaviour).
DEFAULT_MAX_HOLDOUT_INSPECTIONS: int | None = 1

_CHAIN_GENESIS = "GENESIS"
_LEGACY_PREFIX = "LEGACY:"


def normalize_holdout_key(dataset_id: str, slice_identifier: str) -> tuple[str, str]:
    """Canonical identity of a holdout slice for burn and budget lookups.

    Unicode-normalised, whitespace-collapsed and case-folded, so ``"SPY"`` /
    ``"spy"`` and ``"holdout_2024"`` / ``"holdout_2024 "`` are one slice. A
    spelling variant must not be a way to obtain a fresh, unburned holdout.
    """

    def _norm(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", str(value)).split()).casefold()

    return _norm(dataset_id), _norm(slice_identifier)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _entry_hash(prev_hash: str, payload: dict[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "entry_hash"}
    return hashlib.sha256(f"{prev_hash}\n{_canonical_json(body)}".encode()).hexdigest()


def _legacy_link(raw_line: str) -> str:
    return _LEGACY_PREFIX + hashlib.sha256(raw_line.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DatasetExposureRecord:
    """Audit record of a candidate inspecting a dataset slice."""
    candidate_id: str
    dataset_id: str
    slice_identifier: str
    role: DatasetRole
    inspected_at: str
    result_status: str  # "PASSED", "FAILED", "IN_PROGRESS"


@dataclass
class CandidateExperimentNode:
    """Full-lineage record for one candidate hypothesis."""
    candidate_id: str
    proposer_id: str
    formula: str
    version: int = 1
    parent_candidate_id: str | None = None
    feature_set: tuple[str, ...] = ()
    parameter_choices: dict[str, float] = field(default_factory=dict)
    attempted_variants: list[str] = field(default_factory=list)
    state: CandidateLifecycleState = CandidateLifecycleState.PROPOSED
    rejection_reason: str | None = None
    data_exposures: list[DatasetExposureRecord] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d["data_exposures"] = [
            {**asdict(exp), "role": exp.role.value} for exp in self.data_exposures
        ]
        return d


class FullLineageExperimentLedger:
    """Immutable experiment ledger managing hypothesis lineage and preventing holdout reuse."""

    def __init__(
        self,
        persistence_path: Path | str | None = None,
        storage_backend: BaseStorageBackend | None = None,
        remote_key: str = "experiment_ledger.jsonl",
        auto_sync: bool = False,
        max_holdout_inspections: int | None = DEFAULT_MAX_HOLDOUT_INSPECTIONS,
        head_marker: bool = True,
    ):
        if max_holdout_inspections is not None and max_holdout_inspections < 1:
            raise ValueError("max_holdout_inspections must be >= 1 or None")
        self._nodes: dict[str, CandidateExperimentNode] = {}
        # Tracks burned holdout slices: (dataset_id, slice_identifier) -> failing_candidate_id
        self._burned_holdouts: dict[tuple[str, str], str] = {}
        # Normalised slice identity -> the key it was first burned under. Lookups
        # go through this index so a respelled slice name finds the same burn.
        self._burned_index: dict[tuple[str, str], tuple[str, str]] = {}
        # Normalised slice identity -> candidates that inspected it, in order.
        self._holdout_inspectors: dict[tuple[str, str], list[str]] = {}
        self._max_holdout_inspections = max_holdout_inspections
        # ``<ledger>.head`` records the newest entry so a truncated file is
        # detected. Turn it off only for a ledger that is deliberately rewound
        # outside this class (a committed fixture restored with ``git checkout``),
        # where a shorter file is expected; the hash chain still applies.
        self._head_marker = head_marker
        self._storage_backend = storage_backend
        self._remote_key = remote_key
        self._auto_sync = auto_sync
        # FLAW-18: full-ledger remote re-upload amortized over this many
        # appends (each append is still durably journaled locally first).
        self._auto_sync_batch = 16
        self._remote_ops_pending = 0
        self._persistence_path = Path(persistence_path) if persistence_path else None

        if self._persistence_path is None and self._storage_backend is not None:
            from apedp_dyn.env import get_temp_storage_dir
            self._persistence_path = get_temp_storage_dir() / f"cached_{remote_key}"

        if self._storage_backend and self._persistence_path and not self._persistence_path.is_file():
            self.pull_from_remote()

        if self._persistence_path and self._persistence_path.is_file():
            self._load_from_disk()

    def flush(self) -> None:
        """Flush any pending amortized remote synchronization operations."""
        if self._auto_sync and self._storage_backend and self._remote_ops_pending > 0:
            self.sync_to_remote()

    def close(self) -> None:
        """Flush pending remote syncs on ledger shutdown."""
        self.flush()

    def __enter__(self) -> FullLineageExperimentLedger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @classmethod
    def from_env(
        cls,
        filename: str = "experiment_ledger.jsonl",
        storage_backend: BaseStorageBackend | None = None,
        auto_sync: bool | None = None,
    ) -> FullLineageExperimentLedger:
        """Initialize ledger using persistent storage path configured by runtime adapter or remote storage."""
        import os

        from apedp_dyn.env import get_ledger_path
        backend = storage_backend
        if backend is None and os.environ.get("APEDP_STORAGE_BACKEND"):
            from apedp_dyn.epistemic.remote_storage import get_storage_backend
            backend = get_storage_backend()
        sync_enabled = auto_sync if auto_sync is not None else (backend is not None)
        return cls(
            persistence_path=get_ledger_path(filename),
            storage_backend=backend,
            remote_key=filename,
            auto_sync=sync_enabled,
        )

    def register_candidate(
        self,
        candidate_id: str,
        proposer_id: str,
        formula: str,
        parent_candidate_id: str | None = None,
        feature_set: tuple[str, ...] = (),
        parameter_choices: dict[str, float] | None = None,
    ) -> CandidateExperimentNode:
        """Register a newly proposed candidate in the experiment ledger."""
        version = 1
        attempted_variants = []
        if parent_candidate_id and parent_candidate_id in self._nodes:
            parent = self._nodes[parent_candidate_id]
            version = parent.version + 1
            attempted_variants = [*list(parent.attempted_variants), parent.formula]

        node = CandidateExperimentNode(
            candidate_id=candidate_id,
            proposer_id=proposer_id,
            formula=formula,
            version=version,
            parent_candidate_id=parent_candidate_id,
            feature_set=tuple(feature_set),
            parameter_choices=parameter_choices or {},
            attempted_variants=attempted_variants,
            state=CandidateLifecycleState.PROPOSED,
        )
        self._nodes[candidate_id] = node
        self._persist_append(node)
        return node

    def record_data_exposure(
        self,
        candidate_id: str,
        dataset_id: str,
        slice_identifier: str,
        role: DatasetRole,
        outcome_status: str = "IN_PROGRESS",
    ) -> DatasetExposureRecord:
        """Record dataset exposure. Raises HoldoutContaminationError if holdout slice was previously burned."""
        if candidate_id not in self._nodes:
            raise KeyError(f"Candidate {candidate_id} not registered in ledger.")

        holdout_key = (dataset_id, slice_identifier)
        if role == DatasetRole.CONFIRMATION_HOLDOUT:
            self._assert_holdout_available(holdout_key, candidate_id)

        exposure = DatasetExposureRecord(
            candidate_id=candidate_id,
            dataset_id=dataset_id,
            slice_identifier=slice_identifier,
            role=role,
            inspected_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            result_status=outcome_status,
        )

        node = self._nodes[candidate_id]
        node.data_exposures.append(exposure)
        node.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        burn_key = holdout_key if role == DatasetRole.CONFIRMATION_HOLDOUT else None
        burn_now = (role == DatasetRole.CONFIRMATION_HOLDOUT and outcome_status == "FAILED")
        try:
            self._persist_append(
                node,
                check_holdout_key=burn_key,
                burn_holdout_for=candidate_id if burn_now else None,
                force_remote_sync=burn_now,
            )
        except HoldoutContaminationError:
            node.data_exposures.pop()
            raise
        if role == DatasetRole.CONFIRMATION_HOLDOUT:
            self._note_inspection(holdout_key, candidate_id)
        if burn_now:
            self._burn(holdout_key, candidate_id)
        return exposure

    def update_lifecycle_state(
        self,
        candidate_id: str,
        new_state: CandidateLifecycleState,
        rejection_reason: str | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> CandidateExperimentNode:
        """Transition candidate state and burn confirmation holdouts if rejected."""
        if candidate_id not in self._nodes:
            raise KeyError(f"Candidate {candidate_id} not registered.")

        node = self._nodes[candidate_id]
        node.state = new_state
        node.rejection_reason = rejection_reason
        if metrics:
            node.metrics.update(metrics)
        node.updated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # If rejected at confirmation or economic stage (or failed any gate), permanently burn all confirmation slices it inspected
        terminal_failure_states = (
            CandidateLifecycleState.CONFIRMATION_REJECTED,
            CandidateLifecycleState.ECONOMIC_EVALUATION_FAILED,
            CandidateLifecycleState.CALIBRATION_FAILED,
            CandidateLifecycleState.PRE_SCREEN_REJECTED,
            CandidateLifecycleState.UNRESOLVED,
        )
        burned_any = False
        if new_state in terminal_failure_states:
            for exp in node.data_exposures:
                if exp.role == DatasetRole.CONFIRMATION_HOLDOUT:
                    self._burn((exp.dataset_id, exp.slice_identifier), candidate_id)
                    burned_any = True

        self._persist_append(node, force_remote_sync=burned_any)
        return node

    def get_candidate(self, candidate_id: str) -> CandidateExperimentNode:
        return self._nodes[candidate_id]

    def get_all_candidates(self) -> list[CandidateExperimentNode]:
        return list(self._nodes.values())

    def get_burned_holdouts(self) -> dict[tuple[str, str], str]:
        return dict(self._burned_holdouts)

    def is_holdout_burned(self, dataset_id: str, slice_identifier: str) -> bool:
        """Whether a slice is burned, under any spelling of its identifiers."""
        return normalize_holdout_key(dataset_id, slice_identifier) in self._burned_index

    def get_holdout_inspection_counts(self) -> dict[tuple[str, str], int]:
        """Distinct candidates that have inspected each confirmation holdout.

        Keys are normalised slice identities (see :func:`normalize_holdout_key`).
        """
        return {key: len(cands) for key, cands in self._holdout_inspectors.items()}

    @property
    def max_holdout_inspections(self) -> int | None:
        return self._max_holdout_inspections

    def _burn(self, holdout_key: tuple[str, str], candidate_id: str) -> None:
        norm = normalize_holdout_key(*holdout_key)
        first_key = self._burned_index.setdefault(norm, holdout_key)
        self._burned_holdouts[first_key] = candidate_id

    def _note_inspection(self, holdout_key: tuple[str, str], candidate_id: str) -> None:
        inspectors = self._holdout_inspectors.setdefault(normalize_holdout_key(*holdout_key), [])
        if candidate_id not in inspectors:
            inspectors.append(candidate_id)

    def _assert_holdout_available(self, holdout_key: tuple[str, str], candidate_id: str) -> None:
        """Raise unless ``candidate_id`` may inspect this confirmation holdout."""
        norm = normalize_holdout_key(*holdout_key)
        label = f"{holdout_key[0]}:{holdout_key[1]}"
        burned_under = self._burned_index.get(norm)
        if burned_under is not None:
            failing_cand = self._burned_holdouts[burned_under]
            raise HoldoutContaminationError(
                f"DATA SNOOPING VIOLATION: Dataset slice '{label}' "
                f"was previously burned by failing candidate '{failing_cand}'. "
                f"Confirmation holdouts may not be reused for replacement hypotheses."
            )
        if self._max_holdout_inspections is None:
            return
        inspectors = self._holdout_inspectors.get(norm, [])
        if candidate_id not in inspectors and len(inspectors) >= self._max_holdout_inspections:
            raise HoldoutContaminationError(
                f"DATA SNOOPING VIOLATION: Dataset slice '{label}' has already been inspected by "
                f"{len(inspectors)} candidate(s) ({', '.join(inspectors)}); the confirmation-holdout "
                f"inspection budget is {self._max_holdout_inspections}. Scoring another candidate on "
                "it would turn the holdout into a selection set."
            )

    @property
    def storage_backend(self) -> BaseStorageBackend | None:
        return self._storage_backend

    @property
    def persistence_path(self) -> Path | None:
        """The ledger's own durable file path (or None for in-memory use).

        Consumed by the caller-layer protocol to derive its durable lead
        state location (restart-proof sealed-slice consumption)."""
        return self._persistence_path

    def sync_to_remote(self) -> None:
        """Synchronize local ledger state to the configured remote storage backend."""
        if self._storage_backend and self._persistence_path and self._persistence_path.is_file():
            self._storage_backend.sync_to_remote(self._persistence_path, self._remote_key)
            self._remote_ops_pending = 0

    def pull_from_remote(self) -> bool:
        """Pull remote ledger from storage backend, load candidates and burned holdouts."""
        if not self._storage_backend or not self._persistence_path:
            return False
        pulled = self._storage_backend.pull_from_remote(self._remote_key, self._persistence_path)
        if pulled and self._persistence_path.is_file():
            # The remote copy is the source of truth after a pull; a head marker
            # left by an older local file would describe a different ledger.
            self._head_path().unlink(missing_ok=True)
            self._load_from_disk()
            return True
        return False

    def export_snapshot(self, snapshot_name: str | None = None) -> str:
        """Export an immutable, timestamped snapshot of the current ledger state."""
        if self._storage_backend and self._persistence_path and self._persistence_path.is_file():
            # FLAW-18: a snapshot is an externally visible checkpoint — flush
            # any pending amortized syncs first so remote state is not stale
            # relative to the exported snapshot.
            if self._auto_sync and self._remote_ops_pending > 0:
                self.sync_to_remote()
            return self._storage_backend.export_snapshot(self._persistence_path, snapshot_name)
        elif self._persistence_path and self._persistence_path.is_file():
            import shutil
            import uuid
            now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            name = snapshot_name or f"{self._persistence_path.stem}_{now_str}_{uuid.uuid4().hex[:8]}{self._persistence_path.suffix}"
            snap_dir = self._persistence_path.parent / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_path = snap_dir / name
            shutil.copy2(self._persistence_path, snap_path)
            return str(snap_path)
        else:
            raise RuntimeError("Cannot export snapshot: ledger has no persistence path or data on disk.")

    def _persist_append(
        self,
        node: CandidateExperimentNode,
        *,
        check_holdout_key: tuple[str, str] | None = None,
        burn_holdout_for: str | None = None,
        force_remote_sync: bool = False,
    ) -> None:
        if self._persistence_path:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            # FLAW-18 (part 1): an OS advisory lock wraps the append so that
            # parallel discovery workers cannot interleave partial JSONL
            # records and corrupt the ledger. flock is POSIX advisory and is
            # transparent on single-writer flows; on platforms without
            # fcntl (Windows) the append degrades gracefully to the previous
            # single-open behavior rather than crashing a legitimate run.
            try:
                import fcntl
                assert fcntl.flock is not None  # resolved only to assert availability
            except ImportError:  # pragma: no cover - Windows-only path
                fcntl = None  # type: ignore[assignment]
            with open(self._persistence_path, "a+", encoding="utf-8") as f:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    if check_holdout_key is not None:
                        prev_hash = self._refresh_burned_holdouts_locked(f)
                        self._assert_holdout_available(check_holdout_key, node.candidate_id)
                        if burn_holdout_for is not None:
                            self._burn(check_holdout_key, burn_holdout_for)
                    else:
                        prev_hash = self._tail_link_locked(f)
                    payload = node.to_dict()
                    payload["prev_hash"] = prev_hash
                    payload["entry_hash"] = _entry_hash(prev_hash, payload)
                    f.seek(0, os.SEEK_END)
                    f.write(json.dumps(payload) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    self._write_head_locked(os.fstat(f.fileno()).st_size, payload["entry_hash"])
                finally:
                    if fcntl is not None:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            # FLAW-18 (part 2): full-ledger re-upload after EVERY append is
            # O(N²) network traffic on parallel discovery runs. Remote sync
            # is now amortized: pending appends accumulate locally (each one
            # is still durably journaled on disk first) and flush to remote
            # in batches, or immediately on holdout burn / context close / explicit sync.
            if self._auto_sync and self._storage_backend:
                self._remote_ops_pending = getattr(self, "_remote_ops_pending", 0) + 1
                if force_remote_sync or self._remote_ops_pending >= self._auto_sync_batch:
                    self.sync_to_remote()
                    self._remote_ops_pending = 0

    # ------------------------------------------------------------------
    # Tamper evidence. Every appended line carries ``prev_hash`` (the previous
    # line's ``entry_hash``) and its own ``entry_hash``, so an edited, reordered
    # or removed line breaks the chain at the next read. A chain cannot see a
    # removed *tail*, so the newest hash and the file size it was written at are
    # also kept in a ``<ledger>.head`` marker; a file that no longer reaches that
    # marker has been truncated. This is evidence against loss and casual
    # editing, not a defence against someone who rewrites both files -- for that,
    # anchor ``head_hash`` somewhere the writer cannot reach.
    # ------------------------------------------------------------------

    def _head_path(self) -> Path:
        assert self._persistence_path is not None
        return self._persistence_path.with_name(self._persistence_path.name + ".head")

    @property
    def head_hash(self) -> str | None:
        """Hash of the newest ledger entry, for anchoring outside the ledger."""
        if self._persistence_path is None or not self._persistence_path.is_file():
            return None
        with open(self._persistence_path, encoding="utf-8") as f:
            link = self._tail_link_locked(f)
        return None if link == _CHAIN_GENESIS else link

    def _write_head_locked(self, size_bytes: int, entry_hash: str) -> None:
        if not self._head_marker:
            return
        head_path = self._head_path()
        tmp_path = head_path.with_name(head_path.name + ".tmp")
        tmp_path.write_text(
            json.dumps({"size_bytes": int(size_bytes), "head_hash": entry_hash}), encoding="utf-8"
        )
        os.replace(tmp_path, head_path)

    def _tail_link_locked(self, file_obj: Any) -> str:
        """Chain link the next appended entry must point at (last line's hash)."""
        end = os.fstat(file_obj.fileno()).st_size
        if end == 0:
            return _CHAIN_GENESIS
        # Text-mode handles only seek to opaque offsets, so the tail is read
        # through a separate binary handle on the same (already locked) file.
        with open(file_obj.name, "rb") as raw:
            block = 4096
            data = b""
            pos = end
            while pos > 0:
                step = min(block, pos)
                pos -= step
                raw.seek(pos)
                data = raw.read(step) + data
                if data.rstrip(b"\n").count(b"\n") >= 1:
                    break
                block *= 2
        last_line = data.rstrip(b"\n").rsplit(b"\n", 1)[-1].decode("utf-8")
        if not last_line.strip():
            return _CHAIN_GENESIS
        try:
            entry = json.loads(last_line)
        except ValueError as exc:
            raise LedgerIntegrityError(
                f"Ledger {self._persistence_path} ends in an unreadable line; refusing to append."
            ) from exc
        tail_hash = entry.get("entry_hash") if isinstance(entry, dict) else None
        return str(tail_hash) if tail_hash else _legacy_link(last_line)

    def _read_verified_entries(self, file_obj: Any) -> tuple[list[dict[str, Any]], str]:
        """Parse every line, verify the hash chain, and return (entries, tail link).

        Lines written before the chain existed carry no hashes and are accepted
        until the first chained line; after that every line must chain.
        """
        entries: list[dict[str, Any]] = []
        link = _CHAIN_GENESIS
        chained = False
        offset = 0
        head_hash_at_marker: str | None = None
        marker = self._read_head_marker()
        file_obj.seek(0)
        for line_number, line in enumerate(file_obj, start=1):
            offset += len(line.encode("utf-8"))
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                entry = json.loads(raw)
            except ValueError as exc:
                raise LedgerIntegrityError(
                    f"Ledger {self._persistence_path} line {line_number} is not valid JSON; "
                    "a damaged line may hide a burned holdout."
                ) from exc
            if not isinstance(entry, dict):
                raise LedgerIntegrityError(
                    f"Ledger {self._persistence_path} line {line_number} is not a JSON object."
                )
            if "entry_hash" in entry:
                chained = True
                if entry.get("prev_hash") != link or _entry_hash(link, entry) != entry["entry_hash"]:
                    raise LedgerIntegrityError(
                        f"Ledger {self._persistence_path} hash chain breaks at line {line_number}: "
                        "an entry was edited, reordered, or removed."
                    )
                link = str(entry["entry_hash"])
            elif chained:
                raise LedgerIntegrityError(
                    f"Ledger {self._persistence_path} line {line_number} is unchained after the "
                    "chain began; an entry was inserted or stripped of its hashes."
                )
            else:
                link = _legacy_link(raw)
            if marker is not None and offset == marker[0]:
                head_hash_at_marker = link
            entries.append(entry)
        if marker is not None and head_hash_at_marker != marker[1]:
            raise LedgerIntegrityError(
                f"Ledger {self._persistence_path} does not reach its recorded head "
                f"({marker[1][:12]}… at byte {marker[0]}); the file was truncated or replaced. "
                f"If it was rewound on purpose, remove {self._head_path().name} to accept it."
            )
        return entries, link

    def _read_head_marker(self) -> tuple[int, str] | None:
        head_path = self._head_path()
        if not self._head_marker or not head_path.is_file():
            return None
        try:
            payload = json.loads(head_path.read_text(encoding="utf-8"))
            return int(payload["size_bytes"]), str(payload["head_hash"])
        except (ValueError, KeyError, TypeError) as exc:
            raise LedgerIntegrityError(f"Ledger head marker {head_path} is unreadable.") from exc

    def _absorb_holdout_state(self, entry: dict[str, Any]) -> None:
        """Fold one journaled node into the burned and inspected registries."""
        cid = str(entry.get("candidate_id", ""))
        is_terminal_fail = entry.get("state", "") in _TERMINAL_FAILURE_STATE_VALUES
        for exp in entry.get("data_exposures", []):
            if exp.get("role") != DatasetRole.CONFIRMATION_HOLDOUT.value:
                continue
            key = (exp["dataset_id"], exp["slice_identifier"])
            self._note_inspection(key, cid)
            if exp.get("result_status") == "FAILED" or is_terminal_fail:
                self._burn(key, cid)

    def _refresh_burned_holdouts_locked(self, file_obj: Any) -> str:
        """Refresh burned/inspected holdout state while holding the exclusive file lock.

        Returns the chain link the next appended entry must carry.
        """
        entries, link = self._read_verified_entries(file_obj)
        for entry in entries:
            self._absorb_holdout_state(entry)
        return link

    def _load_from_disk(self) -> None:
        assert self._persistence_path is not None
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows-only path
            fcntl = None  # type: ignore[assignment]
        with open(self._persistence_path, encoding="utf-8") as f:
            # Shared lock: a writer holds LOCK_EX across its append and head
            # update, so a reader never sees a half-written line.
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                entries, _link = self._read_verified_entries(f)
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        for d in entries:
            exposures = [
                DatasetExposureRecord(
                    candidate_id=e["candidate_id"],
                    dataset_id=e["dataset_id"],
                    slice_identifier=e["slice_identifier"],
                    role=DatasetRole(e["role"]),
                    inspected_at=e["inspected_at"],
                    result_status=e["result_status"],
                )
                for e in d.get("data_exposures", [])
            ]
            node = CandidateExperimentNode(
                candidate_id=d["candidate_id"],
                proposer_id=d["proposer_id"],
                formula=d["formula"],
                version=d.get("version", 1),
                parent_candidate_id=d.get("parent_candidate_id"),
                feature_set=tuple(d.get("feature_set", ())),
                parameter_choices=d.get("parameter_choices", {}),
                attempted_variants=d.get("attempted_variants", []),
                state=CandidateLifecycleState(d["state"]),
                rejection_reason=d.get("rejection_reason"),
                data_exposures=exposures,
                metrics=d.get("metrics", {}),
                created_at=d.get("created_at", ""),
                updated_at=d.get("updated_at", ""),
            )
            self._nodes[node.candidate_id] = node
            self._absorb_holdout_state(d)


_TERMINAL_FAILURE_STATE_VALUES = frozenset(
    {
        CandidateLifecycleState.CONFIRMATION_REJECTED.value,
        CandidateLifecycleState.ECONOMIC_EVALUATION_FAILED.value,
        CandidateLifecycleState.CALIBRATION_FAILED.value,
        CandidateLifecycleState.PRE_SCREEN_REJECTED.value,
        CandidateLifecycleState.UNRESOLVED.value,
    }
)
