"""Leakage-resistant, auditable controls for offline score experiments.

The types in this module deliberately keep protected-set evidence aggregate-only.
They are development controls and are not imported by the submission runtime.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

try:  # pragma: no cover - all supported challenge hosts are POSIX.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


GENESIS_HASH = "0" * 64
_CASE_ID_RE = re.compile(r"\bMIB-\d+\b", re.IGNORECASE)
_PDF_FILENAME_RE = re.compile(
    r"(?:^|[/\\])?[^/\\\n\r\t]+\.pdf(?:$|[\s\"'])",
    re.IGNORECASE,
)
_LOOKUP_NAME_RE = re.compile(
    r"(?:"
    r"(?:case|label).*(?:lookup|map|table|index)"
    r"|(?:lookup|map|table|index).*(?:case|label)"
    r"|labels?_by_case"
    r"|cases?_by_label"
    r"|case_labels?"
    r")",
    re.IGNORECASE,
)
_FILE_HASH_LOOKUP_NAME_RE = re.compile(
    r"(?:"
    r"(?:file(?:name)?|pdf|document|path).*(?:hash|sha(?:256)?|digest)"
    r"|(?:hash|sha(?:256)?|digest).*(?:file(?:name)?|pdf|document|path)"
    r")",
    re.IGNORECASE,
)
_FILE_COLLECTION_NAME_RE = re.compile(
    r"(?:^|_)(?:files?|filenames?|pdfs?|documents?|paths?)(?:_|$)",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
_COMMIT_OR_SHA256_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", re.IGNORECASE)
_SAFE_DIMENSION_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}")
_IDENTITY_DIMENSION_RE = re.compile(
    r"(?:^|_)(?:cases?|rows?|samples?|outcomes?|truth|pred(?:ictions?)?|"
    r"files?|filenames?|pdfs?|documents?)(?:_|$|\d)",
    re.IGNORECASE,
)
_AGGREGATE_SCALAR_KEYS = frozenset(
    {
        "accuracy",
        "access_authorized",
        "baseline_verified",
        "blank_records",
        "brier_score",
        "calibration_score",
        "candidate_score",
        "catastrophic_false_approvals",
        "classification_score",
        "count",
        "deterministic",
        "duplicate_records",
        "error_count",
        "extra_records",
        "extraction_score",
        "false_approvals",
        "fold_consistent",
        "fraction",
        "hard_gate_failure_count",
        "invalid_records",
        "leakage_clean",
        "leakage_finding_count",
        "mean_brier",
        "mean",
        "median",
        "min",
        "max",
        "missing_records",
        "peak_rss_bytes",
        "record_count",
        "regression_waiver_count",
        "repeat_count",
        "precision",
        "recall",
        "rate",
        "runtime_seconds",
        "score",
        "score_delta",
        "stddev",
        "total_count",
        "total_score",
        "value",
        "variance",
        "warning_count",
    }
)
_AGGREGATE_SCALAR_SUFFIXES = (
    "_accuracy",
    "_authorized",
    "_brier",
    "_bytes",
    "_clean",
    "_consistent",
    "_count",
    "_delta",
    "_deterministic",
    "_fraction",
    "_gap",
    "_loss",
    "_max",
    "_mean",
    "_median",
    "_min",
    "_percentage",
    "_rate",
    "_score",
    "_seconds",
    "_stddev",
    "_total",
    "_variance",
    "_verified",
)
_AGGREGATE_HASH_SUFFIXES = ("_sha256", "_sha", "_hash")
_AGGREGATE_CONTAINER_KEYS = frozenset(
    {
        "checks",
        "class_metrics",
        "confusion_counts",
        "counts",
        "field_metrics",
        "fold_metrics",
        "gate_results",
        "metrics",
        "per_field_metrics",
        "regression_counts",
        "regression_waivers",
        "score_components",
    }
)
_AGGREGATE_NESTED_METRIC_CONTAINERS = frozenset(
    {
        "class_metrics",
        "field_metrics",
        "fold_metrics",
        "per_field_metrics",
    }
)
_AGGREGATE_SEQUENCE_KEYS = frozenset(
    {
        "fold_deltas",
        "fold_scores",
        "repeat_scores",
    }
)
_AGGREGATE_STRING_KEYS = frozenset(
    {
        "evidence_label",
        "evaluation_mode",
        "release_tier",
        "status",
    }
)
_AGGREGATE_STRING_VALUES = frozenset(
    {
        "aggregate_only",
        "baseline",
        "blocked",
        "candidate",
        "failed",
        "local",
        "passed",
        "protected",
        "public_grouped_robustness_not_unseen",
        "verified",
    }
)


class ExperimentControlError(ValueError):
    """A persisted experiment-control contract was violated."""


class IntegrityError(ExperimentControlError):
    """A hash chain, frozen artifact, or immutable manifest is invalid."""


class CompareAndSwapError(ExperimentControlError):
    """The caller's expected ledger head does not match persisted state."""


class LeakageError(ExperimentControlError):
    """Protected evidence contains case-level identity."""


class BudgetExhaustedError(ExperimentControlError):
    """No protected-set accesses remain."""


def canonical_json(value: Any) -> str:
    """Return the only JSON serialization accepted by the append-only stores."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExperimentControlError("value is not canonical-JSON serializable") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _record_hash(sequence: int, previous_hash: str, payload: Mapping[str, Any]) -> str:
    unsigned = {
        "payload": payload,
        "previous_hash": previous_hash,
        "sequence": sequence,
    }
    return _sha256_bytes(canonical_json(unsigned).encode("utf-8"))


def _raise_aggregate_schema(path: str, reason: str) -> None:
    raise LeakageError(f"protected evidence is not aggregate-only at {path}: {reason}")


def _require_nonidentifying_control_text(name: str, value: str) -> None:
    if _CASE_ID_RE.search(value) or _PDF_FILENAME_RE.search(value.strip()):
        raise LeakageError(f"{name} must not contain case or PDF identity")


def _validate_aggregate_scalar(key: str, value: Any, *, path: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        canonical_json(value)
        return
    if value is None:
        return
    if isinstance(value, str):
        if _CASE_ID_RE.search(value) or _PDF_FILENAME_RE.search(value.strip()):
            _raise_aggregate_schema(path, "case or filename identity is forbidden")
        if key.endswith(_AGGREGATE_HASH_SUFFIXES):
            digest_pattern = (
                _SHA256_RE if key.endswith("_sha256") else _COMMIT_OR_SHA256_RE
            )
            if not digest_pattern.fullmatch(value):
                _raise_aggregate_schema(
                    path, "hash metrics must be a full commit or SHA-256 hex digest"
                )
            return
        if key in _AGGREGATE_STRING_KEYS and value.casefold() in _AGGREGATE_STRING_VALUES:
            return
    _raise_aggregate_schema(path, "only aggregate numbers, booleans, or allowed labels are valid")


def _is_aggregate_scalar_key(key: str) -> bool:
    return (
        key in _AGGREGATE_SCALAR_KEYS
        or key in _AGGREGATE_STRING_KEYS
        or key.endswith(_AGGREGATE_SCALAR_SUFFIXES)
        or key.endswith(_AGGREGATE_HASH_SUFFIXES)
    )


def _validate_dimension_name(key: str, *, path: str) -> None:
    if not _SAFE_DIMENSION_RE.fullmatch(key):
        _raise_aggregate_schema(path, "invalid aggregate dimension")
    if (
        _FILE_HASH_LOOKUP_NAME_RE.search(key)
        or _IDENTITY_DIMENSION_RE.search(key)
        or key.casefold().endswith(("_id", "_ids"))
    ):
        _raise_aggregate_schema(path, "identity and per-file dimensions are forbidden")


def _validate_aggregate_container(
    value: Any,
    *,
    container_key: str,
    path: str,
) -> None:
    if not isinstance(value, Mapping):
        _raise_aggregate_schema(path, "aggregate metric container must be an object")
    for raw_key, child in value.items():
        key = str(raw_key).strip()
        normalized = key.casefold()
        child_path = f"{path}.{key}"
        _validate_dimension_name(key, path=child_path)
        if container_key in _AGGREGATE_NESTED_METRIC_CONTAINERS:
            if not isinstance(child, Mapping):
                _raise_aggregate_schema(
                    child_path, "nested metric dimensions must contain metric objects"
                )
            for raw_metric_key, metric_value in child.items():
                metric_key = str(raw_metric_key).strip()
                metric_path = f"{child_path}.{metric_key}"
                if not _SAFE_DIMENSION_RE.fullmatch(metric_key):
                    _raise_aggregate_schema(metric_path, "invalid metric key")
                if not _is_aggregate_scalar_key(metric_key.casefold()):
                    _raise_aggregate_schema(
                        metric_path, "nested key is not an aggregate metric"
                    )
                if isinstance(metric_value, (Mapping, list, tuple)):
                    _raise_aggregate_schema(
                        metric_path, "nested record-shaped values are forbidden"
                    )
                _validate_aggregate_scalar(
                    metric_key.casefold(), metric_value, path=metric_path
                )
            continue
        if isinstance(child, (Mapping, list, tuple)):
            _raise_aggregate_schema(child_path, "record-shaped values are forbidden")
        if container_key == "regression_waivers":
            if not isinstance(child, str) or not _SAFE_DIMENSION_RE.fullmatch(child):
                _raise_aggregate_schema(
                    child_path, "waivers require a non-identifying token"
                )
        elif isinstance(child, str):
            _raise_aggregate_schema(child_path, "string-valued dimensions are forbidden")
        else:
            _validate_aggregate_scalar(key, child, path=child_path)


def require_aggregate_only(value: Any) -> None:
    """Validate evidence against a strict aggregate-only JSON schema.

    The root accepts only metric/check/hash keys, named aggregate containers,
    and short numeric fold vectors. Arbitrary objects and record arrays are
    rejected, so renaming ``rows`` or ``outcomes`` cannot bypass the contract.
    """

    if not isinstance(value, Mapping):
        _raise_aggregate_schema("$", "root must be an aggregate object")
    for raw_key, child in value.items():
        key = str(raw_key).strip()
        normalized = key.casefold()
        path = f"$.{key}"
        if not _SAFE_DIMENSION_RE.fullmatch(key):
            _raise_aggregate_schema(path, "invalid aggregate key")
        if _FILE_HASH_LOOKUP_NAME_RE.search(normalized) or normalized.endswith(("_id", "_ids")):
            _raise_aggregate_schema(path, "identity and per-file digest keys are forbidden")
        if normalized in _AGGREGATE_CONTAINER_KEYS:
            _validate_aggregate_container(
                child,
                container_key=normalized,
                path=path,
            )
            continue
        if normalized in _AGGREGATE_SEQUENCE_KEYS:
            if not isinstance(child, (list, tuple)) or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in child
            ):
                _raise_aggregate_schema(path, "fold vectors may contain numbers only")
            canonical_json(list(child))
            continue
        if not _is_aggregate_scalar_key(normalized):
            _raise_aggregate_schema(path, "key is not in the aggregate evidence schema")
        _validate_aggregate_scalar(normalized, child, path=path)


class CanonicalHashChainStore:
    """Canonical, hash-chained JSONL with locked append and head-based CAS.

    A valid prefix is indistinguishable from an intentionally shorter ledger, so
    callers that need truncation detection must retain an expected head and/or
    record count and pass it to :meth:`verify`.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @staticmethod
    def _parse(raw: str) -> tuple[dict[str, Any], ...]:
        records: list[dict[str, Any]] = []
        previous_hash = GENESIS_HASH
        if not raw:
            return ()
        if not raw.endswith("\n"):
            raise IntegrityError("ledger has a truncated final JSONL record")
        for index, line in enumerate(raw.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"ledger record {index} is invalid JSON") from exc
            if not isinstance(record, dict):
                raise IntegrityError(f"ledger record {index} is not an object")
            required = {"sequence", "previous_hash", "payload", "record_hash"}
            if set(record) != required:
                raise IntegrityError(f"ledger record {index} has an invalid schema")
            if record["sequence"] != index:
                raise IntegrityError(f"ledger sequence mismatch at record {index}")
            if record["previous_hash"] != previous_hash:
                raise IntegrityError(f"ledger chain mismatch at record {index}")
            if not isinstance(record["payload"], dict):
                raise IntegrityError(f"ledger payload {index} is not an object")
            expected_hash = _record_hash(index, previous_hash, record["payload"])
            if record["record_hash"] != expected_hash:
                raise IntegrityError(f"ledger hash mismatch at record {index}")
            if line != canonical_json(record):
                raise IntegrityError(f"ledger record {index} is not canonical JSON")
            records.append(record)
            previous_hash = expected_hash
        return tuple(records)

    @staticmethod
    def _head(records: Sequence[Mapping[str, Any]]) -> str:
        return str(records[-1]["record_hash"]) if records else GENESIS_HASH

    def read(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        return self._parse(self.path.read_text(encoding="utf-8"))

    @property
    def head(self) -> str:
        return self._head(self.read())

    @property
    def length(self) -> int:
        return len(self.read())

    def verify(
        self,
        *,
        expected_head: str | None = None,
        expected_length: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        records = self.read()
        actual_head = self._head(records)
        if expected_head is not None and actual_head != expected_head:
            raise IntegrityError(
                f"ledger head mismatch: expected {expected_head}, got {actual_head}"
            )
        if expected_length is not None and len(records) != expected_length:
            raise IntegrityError(
                f"ledger length mismatch: expected {expected_length}, got {len(records)}"
            )
        return records

    def append(
        self,
        payload: Mapping[str, Any],
        *,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        return self.append_transactional(payload, expected_head=expected_head)

    def append_transactional(
        self,
        payload: Mapping[str, Any],
        *,
        expected_head: str | None = None,
        locked_check: Callable[
            [tuple[dict[str, Any], ...], Mapping[str, Any]],
            Mapping[str, Any] | None,
        ]
        | None = None,
    ) -> dict[str, Any]:
        """Check state and append while holding one exclusive file lock.

        The callback may raise to reject the append or return an existing full
        record for an idempotent retry. Returning ``None`` authorizes one append.
        """

        if not isinstance(payload, Mapping):
            raise ExperimentControlError("ledger payload must be an object")
        # Round-trip once so custom Mapping implementations cannot mutate the
        # value between hashing and persistence.
        normalized_payload = json.loads(canonical_json(dict(payload)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                records = self._parse(handle.read())
                actual_head = self._head(records)
                if expected_head is not None and expected_head != actual_head:
                    raise CompareAndSwapError(
                        f"ledger CAS failed: expected {expected_head}, got {actual_head}"
                    )
                if locked_check is not None:
                    existing = locked_check(records, normalized_payload)
                    if existing is not None:
                        normalized_existing = json.loads(
                            canonical_json(dict(existing))
                        )
                        required = {
                            "sequence",
                            "previous_hash",
                            "payload",
                            "record_hash",
                        }
                        if set(normalized_existing) != required:
                            raise IntegrityError(
                                "transaction callback returned a non-record value"
                            )
                        if normalized_existing not in records:
                            raise IntegrityError(
                                "transaction callback returned a record outside the ledger"
                            )
                        return normalized_existing
                sequence = len(records) + 1
                record = {
                    "sequence": sequence,
                    "previous_hash": actual_head,
                    "payload": normalized_payload,
                    "record_hash": _record_hash(
                        sequence, actual_head, normalized_payload
                    ),
                }
                handle.seek(0, os.SEEK_END)
                handle.write(canonical_json(record) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return record
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ExperimentLedger:
    """Append aggregate-only experiment evidence under unique experiment IDs."""

    def __init__(self, path: Path | str) -> None:
        self.store = CanonicalHashChainStore(path)

    def experiments(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(record["payload"])
            for record in self.store.verify()
            if record["payload"].get("event") == "experiment"
        )

    def record(
        self,
        experiment_id: str,
        evidence: Mapping[str, Any],
        *,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        experiment_id = str(experiment_id).strip()
        if not experiment_id:
            raise ExperimentControlError("experiment_id is required")
        _require_nonidentifying_control_text("experiment_id", experiment_id)
        require_aggregate_only(evidence)
        payload = {
            "event": "experiment",
            "experiment_id": experiment_id,
            "evidence": dict(evidence),
        }

        def check(
            records: tuple[dict[str, Any], ...],
            requested: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            for record in records:
                existing = record["payload"]
                if (
                    existing.get("event") == "experiment"
                    and existing.get("experiment_id") == experiment_id
                ):
                    if existing != requested:
                        raise ExperimentControlError(
                            f"experiment_id retry does not match original: {experiment_id}"
                        )
                    return record
            return None

        return self.store.append_transactional(
            payload,
            expected_head=expected_head,
            locked_check=check,
        )


class TaintRegistry:
    """Append-only registry of groups that can never return to a holdout."""

    def __init__(self, path: Path | str) -> None:
        self.store = CanonicalHashChainStore(path)

    def events(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(record["payload"]) for record in self.store.verify())

    def tainted_groups(self) -> frozenset[str]:
        return frozenset(
            str(event["group_id"])
            for event in self.events()
            if event.get("event") == "taint"
        )

    def taint(
        self,
        group_id: str,
        *,
        reason: str,
        source: str,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        group_id = str(group_id).strip()
        reason = str(reason).strip()
        source = str(source).strip()
        if not group_id or not reason or not source:
            raise ExperimentControlError("group_id, reason, and source are required")
        payload = {
            "event": "taint",
            "group_id": group_id,
            "reason": reason,
            "source": source,
        }
        return self.store.append(payload, expected_head=expected_head)

    def untaint(self, group_id: str) -> None:
        del group_id
        raise ExperimentControlError("taint is permanent; untaint is not supported")


def _file_pin(logical_path: str, artifact_path: Path) -> dict[str, Any]:
    if not artifact_path.is_file():
        raise IntegrityError(f"frozen artifact is missing: {artifact_path}")
    content = artifact_path.read_bytes()
    return {
        "path": logical_path,
        "size_bytes": len(content),
        "sha256": _sha256_bytes(content),
    }


class FrozenBaselineManifest:
    """Create-once manifest that pins artifact path, byte size, and SHA-256."""

    SCHEMA = "mib-frozen-baseline/v1"

    def __init__(self, manifest_path: Path | str) -> None:
        self.path = Path(manifest_path)

    @staticmethod
    def _normalize_artifacts(
        artifacts: Mapping[str, Path | str] | Iterable[Path | str],
    ) -> dict[str, Path]:
        if isinstance(artifacts, Mapping):
            normalized = {
                str(logical_path): Path(artifact_path)
                for logical_path, artifact_path in artifacts.items()
            }
        else:
            paths = [Path(path) for path in artifacts]
            normalized = {str(path): path for path in paths}
        if not normalized or any(not name.strip() for name in normalized):
            raise ExperimentControlError("at least one named artifact is required")
        return normalized

    def create(
        self,
        artifacts: Mapping[str, Path | str] | Iterable[Path | str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        require_aggregate_only(metadata or {})
        normalized = self._normalize_artifacts(artifacts)
        manifest = {
            "schema": self.SCHEMA,
            "artifacts": [
                _file_pin(name, normalized[name]) for name in sorted(normalized)
            ],
            "metadata": dict(metadata or {}),
        }
        if self.path.exists():
            existing = self.load()
            if existing != manifest:
                raise IntegrityError("frozen baseline manifest is immutable")
            return existing
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(canonical_json(manifest) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return manifest

    def load(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
            manifest = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise IntegrityError("frozen baseline manifest is unreadable") from exc
        if raw != canonical_json(manifest) + "\n":
            raise IntegrityError("frozen baseline manifest is not canonical JSON")
        if not isinstance(manifest, dict) or manifest.get("schema") != self.SCHEMA:
            raise IntegrityError("frozen baseline manifest schema is invalid")
        return manifest

    def verify(
        self,
        artifacts: Mapping[str, Path | str] | Iterable[Path | str] | None = None,
    ) -> dict[str, Any]:
        manifest = self.load()
        supplied = (
            self._normalize_artifacts(artifacts) if artifacts is not None else None
        )
        for pin in manifest.get("artifacts", ()):
            logical_path = str(pin.get("path", ""))
            path = supplied.get(logical_path) if supplied is not None else Path(logical_path)
            if path is None:
                raise IntegrityError(
                    f"no artifact was supplied for frozen path: {logical_path}"
                )
            actual = _file_pin(logical_path, path)
            if actual != pin:
                raise IntegrityError(f"frozen artifact changed: {logical_path}")
        if supplied is not None:
            pinned_names = {str(pin["path"]) for pin in manifest["artifacts"]}
            if set(supplied) != pinned_names:
                raise IntegrityError("supplied artifact set differs from frozen manifest")
        return manifest


@dataclass(frozen=True)
class GroupedFold:
    repeat: int
    fold: int
    tuning_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    tuning_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeat": self.repeat,
            "fold": self.fold,
            "tuning_groups": list(self.tuning_groups),
            "validation_groups": list(self.validation_groups),
            "tuning_case_ids": list(self.tuning_case_ids),
            "validation_case_ids": list(self.validation_case_ids),
        }


class RepeatedGroupedSplitManager:
    """Deterministic repeated K-fold splits with whole-group exclusion."""

    def __init__(self, *, seed: str, repeats: int = 3, folds: int = 5) -> None:
        if not str(seed):
            raise ExperimentControlError("split seed is required")
        if repeats < 1:
            raise ExperimentControlError("repeats must be positive")
        if folds < 2:
            raise ExperimentControlError("folds must be at least two")
        self.seed = str(seed)
        self.repeats = repeats
        self.folds = folds

    def _group_key(self, repeat: int, group_id: str) -> str:
        return _sha256_bytes(
            f"{self.seed}\0{repeat}\0{group_id}".encode("utf-8")
        )

    def split_groups(
        self,
        groups: Mapping[str, Sequence[str]],
        *,
        tainted_groups: Iterable[str] = (),
    ) -> tuple[GroupedFold, ...]:
        normalized: dict[str, tuple[str, ...]] = {}
        case_owner: dict[str, str] = {}
        for raw_group_id, raw_case_ids in groups.items():
            group_id = str(raw_group_id).strip()
            case_ids = tuple(sorted(str(case_id).strip() for case_id in raw_case_ids))
            if not group_id or not case_ids or any(not case_id for case_id in case_ids):
                raise ExperimentControlError(
                    "groups and their case IDs must be non-empty"
                )
            if len(set(case_ids)) != len(case_ids):
                raise ExperimentControlError(f"duplicate case in group: {group_id}")
            for case_id in case_ids:
                previous = case_owner.setdefault(case_id, group_id)
                if previous != group_id:
                    raise ExperimentControlError(
                        f"case belongs to multiple groups: {case_id}"
                    )
            normalized[group_id] = case_ids

        tainted = {str(group).strip() for group in tainted_groups}
        eligible = sorted(set(normalized) - tainted)
        if len(eligible) < self.folds:
            raise ExperimentControlError(
                "eligible group count must be at least the number of folds"
            )

        result: list[GroupedFold] = []
        for repeat in range(self.repeats):
            ordered = sorted(
                eligible, key=lambda group_id: self._group_key(repeat, group_id)
            )
            buckets = [ordered[index :: self.folds] for index in range(self.folds)]
            for fold, validation_groups_raw in enumerate(buckets):
                validation_groups = tuple(sorted(validation_groups_raw))
                validation_set = set(validation_groups)
                tuning_groups = tuple(
                    sorted(group for group in eligible if group not in validation_set)
                )
                validation_case_ids = tuple(
                    sorted(
                        case_id
                        for group in validation_groups
                        for case_id in normalized[group]
                    )
                )
                tuning_case_ids = tuple(
                    sorted(
                        case_id
                        for group in tuning_groups
                        for case_id in normalized[group]
                    )
                )
                result.append(
                    GroupedFold(
                        repeat=repeat,
                        fold=fold,
                        tuning_groups=tuning_groups,
                        validation_groups=validation_groups,
                        tuning_case_ids=tuning_case_ids,
                        validation_case_ids=validation_case_ids,
                    )
                )
        return tuple(result)

    def split_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        group_key: str = "group_id",
        case_key: str = "case_id",
        tainted_groups: Iterable[str] = (),
    ) -> tuple[GroupedFold, ...]:
        groups: dict[str, list[str]] = {}
        for row in rows:
            group_id = str(row.get(group_key, "")).strip()
            case_id = str(row.get(case_key, "")).strip()
            if not group_id or not case_id:
                raise ExperimentControlError(
                    f"split rows require {group_key} and {case_key}"
                )
            groups.setdefault(group_id, []).append(case_id)
        return self.split_groups(groups, tainted_groups=tainted_groups)


class ProtectedAccessBudget:
    """Persist a small, aggregate-only protected evaluation access budget."""

    CONFIGURATION_EVENT = "protected_budget_configuration"

    def __init__(self, path: Path | str, *, maximum_accesses: int) -> None:
        if (
            isinstance(maximum_accesses, bool)
            or not isinstance(maximum_accesses, int)
            or maximum_accesses < 1
        ):
            raise ExperimentControlError("maximum_accesses must be positive")
        self.store = CanonicalHashChainStore(path)
        self.maximum_accesses = maximum_accesses
        self._ensure_configuration()

    @classmethod
    def _validate_configuration(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        maximum_accesses: int,
    ) -> Mapping[str, Any] | None:
        configurations = [
            record
            for record in records
            if record["payload"].get("event") == cls.CONFIGURATION_EVENT
        ]
        if not configurations:
            if records:
                raise IntegrityError(
                    "protected access ledger predates its immutable configuration"
                )
            return None
        if len(configurations) != 1 or configurations[0]["sequence"] != 1:
            raise IntegrityError(
                "protected access ledger configuration must be its first and only configuration"
            )
        configuration = configurations[0]["payload"]
        expected = {
            "event": cls.CONFIGURATION_EVENT,
            "maximum_accesses": maximum_accesses,
        }
        if configuration != expected:
            raise IntegrityError(
                "protected access budget configuration is immutable"
            )
        return configurations[0]

    def _ensure_configuration(self) -> None:
        payload = {
            "event": self.CONFIGURATION_EVENT,
            "maximum_accesses": self.maximum_accesses,
        }

        def check(
            records: tuple[dict[str, Any], ...],
            requested: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            del requested
            return self._validate_configuration(
                records,
                maximum_accesses=self.maximum_accesses,
            )

        self.store.append_transactional(payload, locked_check=check)

    def accesses(self) -> tuple[dict[str, Any], ...]:
        records = self.store.verify()
        self._validate_configuration(
            records,
            maximum_accesses=self.maximum_accesses,
        )
        return tuple(
            dict(record["payload"])
            for record in records
            if record["payload"].get("event") == "protected_access"
        )

    @property
    def used(self) -> int:
        return len(self.accesses())

    @property
    def remaining(self) -> int:
        return self.maximum_accesses - self.used

    def record_access(
        self,
        access_id: str,
        *,
        candidate_sha256: str,
        aggregate_result: Mapping[str, Any],
        purpose: str,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        access_id = str(access_id).strip()
        candidate_sha256 = str(candidate_sha256).strip().lower()
        purpose = str(purpose).strip()
        if not access_id or not purpose:
            raise ExperimentControlError("access_id and purpose are required")
        _require_nonidentifying_control_text("access_id", access_id)
        _require_nonidentifying_control_text("purpose", purpose)
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
            raise ExperimentControlError("candidate_sha256 must be a SHA-256 hex digest")
        require_aggregate_only(aggregate_result)
        requested = {
            "event": "protected_access",
            "access_id": access_id,
            "candidate_sha256": candidate_sha256,
            "aggregate_result": dict(aggregate_result),
            "purpose": purpose,
        }

        def check(
            records: tuple[dict[str, Any], ...],
            normalized_request: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            self._validate_configuration(
                records,
                maximum_accesses=self.maximum_accesses,
            )
            accesses = [
                record
                for record in records
                if record["payload"].get("event") == "protected_access"
            ]
            for record in accesses:
                existing = record["payload"]
                if existing.get("access_id") == access_id:
                    if existing != normalized_request:
                        raise ExperimentControlError(
                            f"access_id retry does not match original request: {access_id}"
                        )
                    return record
            if len(accesses) >= self.maximum_accesses:
                raise BudgetExhaustedError("protected access budget is exhausted")
            return None

        return dict(
            self.store.append_transactional(
                requested,
                expected_head=expected_head,
                locked_check=check,
            )["payload"]
        )


@dataclass(frozen=True)
class LeakageFinding:
    path: str
    line: int
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "line": self.line,
            "code": self.code,
            "message": self.message,
        }


class RuntimeLeakageScanner:
    """Scan runtime Python/JSON for embedded case identity and lookup tables.

    ``allowlist`` maps an exact file path to the exact finding codes permitted in
    that file. Directories, globs, and blanket suppressions are intentionally not
    supported.
    """

    SUPPORTED_SUFFIXES = frozenset({".py", ".json"})

    def __init__(
        self,
        *,
        allowlist: Mapping[Path | str, Iterable[str]] | None = None,
    ) -> None:
        self.allowlist = {
            str(Path(path).resolve()): frozenset(str(code) for code in codes)
            for path, codes in (allowlist or {}).items()
        }

    @staticmethod
    def _string_findings(
        *,
        path: Path,
        line: int,
        value: str,
    ) -> list[LeakageFinding]:
        findings: list[LeakageFinding] = []
        if _CASE_ID_RE.search(value):
            findings.append(
                LeakageFinding(
                    str(path), line, "MIB_CASE_ID", "embedded MIB case identifier"
                )
            )
        if _PDF_FILENAME_RE.search(value.strip()):
            findings.append(
                LeakageFinding(
                    str(path), line, "PDF_FILENAME", "embedded PDF filename"
                )
            )
        return findings

    @classmethod
    def _scan_python(cls, path: Path) -> list[LeakageFinding]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            return [
                LeakageFinding(
                    str(path),
                    getattr(exc, "lineno", 1) or 1,
                    "UNSCANNABLE",
                    "runtime Python could not be parsed",
                )
            ]
        findings: list[LeakageFinding] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                findings.extend(
                    cls._string_findings(
                        path=path,
                        line=getattr(node, "lineno", 1),
                        value=node.value,
                    )
                )
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                names = [
                    target.id
                    for target in targets
                    if isinstance(target, ast.Name)
                ]
                if (
                    isinstance(value, ast.Dict)
                    and any(_LOOKUP_NAME_RE.search(name) for name in names)
                ):
                    findings.append(
                        LeakageFinding(
                            str(path),
                            getattr(node, "lineno", 1),
                            "CASE_LABEL_LOOKUP",
                            "runtime case/label lookup map",
                        )
                    )
                if any(_FILE_HASH_LOOKUP_NAME_RE.search(name) for name in names):
                    findings.append(
                        LeakageFinding(
                            str(path),
                            getattr(node, "lineno", 1),
                            (
                                "FILE_HASH_LOOKUP"
                                if isinstance(value, ast.Dict)
                                else "PER_FILE_DIGEST_KEY"
                            ),
                            "runtime per-file/PDF digest data",
                        )
                    )
                if (
                    isinstance(value, ast.Dict)
                    and any(_FILE_COLLECTION_NAME_RE.search(name) for name in names)
                    and any(
                        isinstance(child, ast.Constant)
                        and isinstance(child.value, str)
                        and _SHA256_RE.fullmatch(child.value)
                        for child in ast.walk(value)
                    )
                ):
                    findings.append(
                        LeakageFinding(
                            str(path),
                            getattr(node, "lineno", 1),
                            "FILE_HASH_LOOKUP",
                            "runtime file collection contains digest values",
                        )
                    )
            if isinstance(node, ast.Dict):
                for key_node, value_node in zip(node.keys, node.values):
                    if not (
                        isinstance(key_node, ast.Constant)
                        and isinstance(key_node.value, str)
                    ):
                        continue
                    key = key_node.value
                    digest_value = (
                        value_node.value
                        if isinstance(value_node, ast.Constant)
                        and isinstance(value_node.value, str)
                        else ""
                    )
                    if _FILE_HASH_LOOKUP_NAME_RE.search(key):
                        findings.append(
                            LeakageFinding(
                                str(path),
                                getattr(key_node, "lineno", 1),
                                "PER_FILE_DIGEST_KEY",
                                "runtime per-file digest key",
                            )
                        )
                    if (
                        _PDF_FILENAME_RE.search(key.strip())
                        and _SHA256_RE.fullmatch(digest_value)
                    ):
                        findings.append(
                            LeakageFinding(
                                str(path),
                                getattr(key_node, "lineno", 1),
                                "FILE_HASH_LOOKUP",
                                "runtime PDF-to-digest lookup entry",
                            )
                        )
        return findings

    @classmethod
    def _scan_json_value(
        cls,
        value: Any,
        *,
        path: Path,
        json_path: str = "$",
    ) -> list[LeakageFinding]:
        findings: list[LeakageFinding] = []
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{json_path}.{key}"
                normalized_key = str(key).casefold()
                findings.extend(
                    cls._string_findings(path=path, line=1, value=str(key))
                )
                if _LOOKUP_NAME_RE.search(str(key)) and isinstance(child, Mapping):
                    findings.append(
                        LeakageFinding(
                            str(path),
                            1,
                            "CASE_LABEL_LOOKUP",
                            f"runtime case/label lookup map at {child_path}",
                        )
                    )
                if _FILE_HASH_LOOKUP_NAME_RE.search(str(key)):
                    findings.append(
                        LeakageFinding(
                            str(path),
                            1,
                            (
                                "FILE_HASH_LOOKUP"
                                if isinstance(child, Mapping)
                                else "PER_FILE_DIGEST_KEY"
                            ),
                            f"runtime per-file/PDF digest data at {child_path}",
                        )
                    )
                if (
                    _FILE_COLLECTION_NAME_RE.search(str(key))
                    and cls._json_contains_sha256(child)
                ):
                    findings.append(
                        LeakageFinding(
                            str(path),
                            1,
                            "FILE_HASH_LOOKUP",
                            f"runtime file collection contains digests at {child_path}",
                        )
                    )
                if (
                    normalized_key in {"digest", "hash", "sha", "sha256"}
                    and _FILE_COLLECTION_NAME_RE.search(json_path.replace(".", "_"))
                ):
                    findings.append(
                        LeakageFinding(
                            str(path),
                            1,
                            "PER_FILE_DIGEST_KEY",
                            f"runtime per-file digest key at {child_path}",
                        )
                    )
                if (
                    _PDF_FILENAME_RE.search(str(key).strip())
                    and isinstance(child, str)
                    and _SHA256_RE.fullmatch(child)
                ):
                    findings.append(
                        LeakageFinding(
                            str(path),
                            1,
                            "FILE_HASH_LOOKUP",
                            f"runtime PDF-to-digest lookup entry at {child_path}",
                        )
                    )
                findings.extend(
                    cls._scan_json_value(child, path=path, json_path=child_path)
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                findings.extend(
                    cls._scan_json_value(
                        child, path=path, json_path=f"{json_path}[{index}]"
                    )
                )
        elif isinstance(value, str):
            findings.extend(cls._string_findings(path=path, line=1, value=value))
        return findings

    @classmethod
    def _json_contains_sha256(cls, value: Any) -> bool:
        if isinstance(value, Mapping):
            return any(cls._json_contains_sha256(child) for child in value.values())
        if isinstance(value, list):
            return any(cls._json_contains_sha256(child) for child in value)
        return isinstance(value, str) and bool(_SHA256_RE.fullmatch(value))

    @classmethod
    def _scan_json(cls, path: Path) -> list[LeakageFinding]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return [
                LeakageFinding(
                    str(path), 1, "UNSCANNABLE", "runtime JSON could not be parsed"
                )
            ]
        return cls._scan_json_value(value, path=path)

    @staticmethod
    def _files(paths: Iterable[Path | str]) -> tuple[Path, ...]:
        files: set[Path] = set()
        for raw_path in paths:
            path = Path(raw_path)
            if path.is_dir():
                files.update(
                    child
                    for child in path.rglob("*")
                    if child.is_file()
                    and child.suffix.casefold()
                    in RuntimeLeakageScanner.SUPPORTED_SUFFIXES
                    and "__pycache__" not in child.parts
                )
            elif (
                path.is_file()
                and path.suffix.casefold()
                in RuntimeLeakageScanner.SUPPORTED_SUFFIXES
            ):
                files.add(path)
            elif not path.exists():
                raise ExperimentControlError(f"runtime scan path is missing: {path}")
        return tuple(sorted(files, key=lambda item: str(item.resolve())))

    def scan(self, paths: Iterable[Path | str]) -> tuple[LeakageFinding, ...]:
        findings: list[LeakageFinding] = []
        for path in self._files(paths):
            current = (
                self._scan_python(path)
                if path.suffix.casefold() == ".py"
                else self._scan_json(path)
            )
            allowed_codes = self.allowlist.get(str(path.resolve()), frozenset())
            findings.extend(
                finding for finding in current if finding.code not in allowed_codes
            )
        return tuple(
            sorted(findings, key=lambda item: (item.path, item.line, item.code))
        )

    def require_clean(self, paths: Iterable[Path | str]) -> None:
        findings = self.scan(paths)
        if findings:
            summary = "; ".join(
                f"{finding.path}:{finding.line} {finding.code}"
                for finding in findings
            )
            raise LeakageError("runtime leakage scan failed: " + summary)


_PROMOTION_GATE_AUTHORITY = object()


class CandidateStateStore:
    """Persist assessments while retaining the latest passing candidate."""

    VALID_DECISIONS = frozenset({"PASSED", "BLOCKED"})

    def __init__(self, path: Path | str) -> None:
        self.store = CanonicalHashChainStore(path)

    def assessments(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(record["payload"])
            for record in self.store.verify()
            if record["payload"].get("event") == "candidate_assessment"
        )

    def latest_passing(self) -> dict[str, Any] | None:
        for assessment in reversed(self.assessments()):
            if assessment.get("decision") == "PASSED":
                return assessment
        return None

    def assess(
        self,
        assessment_id: str,
        *,
        candidate_id: str,
        candidate_sha256: str,
        decision: str,
        aggregate_evidence: Mapping[str, Any],
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        if str(decision).strip().upper() == "PASSED":
            raise ExperimentControlError(
                "PASSED can only be persisted by CandidatePromotionGate"
            )
        return self._persist_assessment(
            assessment_id,
            candidate_id=candidate_id,
            candidate_sha256=candidate_sha256,
            decision=decision,
            aggregate_evidence=aggregate_evidence,
            expected_head=expected_head,
            authority=None,
        )

    def _persist_assessment(
        self,
        assessment_id: str,
        *,
        candidate_id: str,
        candidate_sha256: str,
        decision: str,
        aggregate_evidence: Mapping[str, Any],
        expected_head: str | None,
        authority: object | None,
    ) -> dict[str, Any]:
        assessment_id = str(assessment_id).strip()
        candidate_id = str(candidate_id).strip()
        candidate_sha256 = str(candidate_sha256).strip().lower()
        decision = str(decision).strip().upper()
        if not assessment_id or not candidate_id:
            raise ExperimentControlError("assessment_id and candidate_id are required")
        _require_nonidentifying_control_text("assessment_id", assessment_id)
        _require_nonidentifying_control_text("candidate_id", candidate_id)
        if decision not in self.VALID_DECISIONS:
            raise ExperimentControlError("decision must be PASSED or BLOCKED")
        if decision == "PASSED" and authority is not _PROMOTION_GATE_AUTHORITY:
            raise ExperimentControlError(
                "PASSED can only be persisted by CandidatePromotionGate"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
            raise ExperimentControlError("candidate_sha256 must be a SHA-256 hex digest")
        require_aggregate_only(aggregate_evidence)
        payload = {
            "event": "candidate_assessment",
            "assessment_id": assessment_id,
            "candidate_id": candidate_id,
            "candidate_sha256": candidate_sha256,
            "decision": decision,
            "aggregate_evidence": dict(aggregate_evidence),
        }

        def check(
            records: tuple[dict[str, Any], ...],
            requested: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            for record in records:
                existing = record["payload"]
                if (
                    existing.get("event") == "candidate_assessment"
                    and existing.get("assessment_id") == assessment_id
                ):
                    if existing != requested:
                        raise ExperimentControlError(
                            f"assessment retry does not match original: {assessment_id}"
                        )
                    return record
            return None

        return dict(
            self.store.append_transactional(
                payload,
                expected_head=expected_head,
                locked_check=check,
            )["payload"]
        )


class CandidatePromotionGate:
    """Evaluate every hard gate and atomically persist the resulting decision."""

    def __init__(
        self,
        state: CandidateStateStore,
        *,
        protected_budget: ProtectedAccessBudget | None = None,
    ) -> None:
        self.state = state
        self.protected_budget = protected_budget

    @staticmethod
    def _count(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExperimentControlError(f"{name} must be a non-negative integer")
        return value

    @staticmethod
    def _flag(name: str, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ExperimentControlError(f"{name} must be a boolean")
        return value

    def evaluate_and_record(
        self,
        assessment_id: str,
        *,
        candidate_id: str,
        candidate_sha256: str,
        baseline_verified: bool,
        leakage_finding_count: int,
        deterministic: bool,
        false_approvals: int,
        missing_records: int,
        invalid_records: int,
        regression_counts: Mapping[str, int],
        fold_consistent: bool,
        access_id: str | None = None,
        access_authorized: bool | None = None,
        regression_waivers: Mapping[str, str] | None = None,
        aggregate_evidence: Mapping[str, Any] | None = None,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        """Persist PASSED only if all required gates actually evaluate true.

        When a protected budget is supplied, authorization is derived from a
        recorded access for the same candidate digest. Otherwise an explicit
        boolean authorization is required. Positive regressions pass only when
        each has a named, non-identifying waiver token.
        """

        candidate_sha256 = str(candidate_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
            raise ExperimentControlError("candidate_sha256 must be a SHA-256 hex digest")
        baseline_ok = self._flag("baseline_verified", baseline_verified)
        deterministic_ok = self._flag("deterministic", deterministic)
        folds_ok = self._flag("fold_consistent", fold_consistent)
        leakage_count = self._count(
            "leakage_finding_count", leakage_finding_count
        )
        false_approval_count = self._count("false_approvals", false_approvals)
        missing_count = self._count("missing_records", missing_records)
        invalid_count = self._count("invalid_records", invalid_records)

        normalized_regressions: dict[str, int] = {}
        if not isinstance(regression_counts, Mapping):
            raise ExperimentControlError("regression_counts must be an object")
        for raw_name, raw_count in regression_counts.items():
            name = str(raw_name).strip()
            if not _SAFE_DIMENSION_RE.fullmatch(name):
                raise ExperimentControlError("regression names must be safe aggregate tokens")
            normalized_regressions[name] = self._count(
                f"regression_counts.{name}", raw_count
            )
        normalized_waivers: dict[str, str] = {}
        if regression_waivers is not None and not isinstance(
            regression_waivers, Mapping
        ):
            raise ExperimentControlError("regression_waivers must be an object")
        for raw_name, raw_token in (regression_waivers or {}).items():
            name = str(raw_name).strip()
            token = str(raw_token).strip()
            if (
                name not in normalized_regressions
                or not _SAFE_DIMENSION_RE.fullmatch(token)
            ):
                raise ExperimentControlError(
                    "waivers must name a regression and use a safe explicit token"
                )
            normalized_waivers[name] = token
        unwaived_regressions = {
            name: count
            for name, count in normalized_regressions.items()
            if count > 0 and name not in normalized_waivers
        }

        if self.protected_budget is not None:
            requested_access_id = str(access_id or "").strip()
            access_ok = any(
                access.get("access_id") == requested_access_id
                and access.get("candidate_sha256") == candidate_sha256
                for access in self.protected_budget.accesses()
            )
        else:
            access_ok = self._flag("access_authorized", access_authorized)

        gate_results = {
            "access_authorized": access_ok,
            "baseline_verified": baseline_ok,
            "deterministic": deterministic_ok,
            "fold_consistent": folds_ok,
            "no_false_approvals": false_approval_count == 0,
            "no_invalid_records": invalid_count == 0,
            "no_leakage": leakage_count == 0,
            "no_missing_records": missing_count == 0,
            "regressions_cleared": not unwaived_regressions,
        }
        decision = "PASSED" if all(gate_results.values()) else "BLOCKED"

        evidence = dict(aggregate_evidence or {})
        protected_keys = {
            "access_authorized",
            "baseline_verified",
            "deterministic",
            "false_approvals",
            "fold_consistent",
            "gate_results",
            "hard_gate_failure_count",
            "invalid_records",
            "leakage_finding_count",
            "missing_records",
            "promotion_gate_verified",
            "regression_counts",
            "regression_waiver_count",
        }
        collisions = protected_keys.intersection(evidence)
        if collisions:
            raise ExperimentControlError(
                "aggregate_evidence cannot override promotion gates: "
                + ", ".join(sorted(collisions))
            )
        evidence.update(
            {
                "access_authorized": access_ok,
                "baseline_verified": baseline_ok,
                "deterministic": deterministic_ok,
                "false_approvals": false_approval_count,
                "fold_consistent": folds_ok,
                "gate_results": gate_results,
                "hard_gate_failure_count": sum(
                    not result for result in gate_results.values()
                ),
                "invalid_records": invalid_count,
                "leakage_finding_count": leakage_count,
                "missing_records": missing_count,
                "promotion_gate_verified": True,
                "regression_counts": normalized_regressions,
                "regression_waiver_count": len(normalized_waivers),
            }
        )
        require_aggregate_only(evidence)
        return self.state._persist_assessment(
            assessment_id,
            candidate_id=candidate_id,
            candidate_sha256=candidate_sha256,
            decision=decision,
            aggregate_evidence=evidence,
            expected_head=expected_head,
            authority=_PROMOTION_GATE_AUTHORITY,
        )
