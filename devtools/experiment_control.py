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
import stat
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path, PurePosixPath
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
        "fold_weights",
        "repeat_scores",
    }
)
_AGGREGATE_STRING_KEYS = frozenset(
    {
        "comparison_scope",
        "evidence_label",
        "evaluation_mode",
        "invocation_scope",
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
        "fusion_vs_legacy_resolver_after_current_case_linking",
        "local",
        "accepted_final_fusion_result_per_case",
        "passed",
        "protected",
        "public_grouped_robustness_not_unseen",
        "verified",
    }
)
_EXPERIMENT_PLAN_KEYS = frozenset(
    {
        "changed_files",
        "evidence_label",
        "evaluator_sha256",
        "expected_record_count",
        "hypothesis_sha256",
        "input_tree_sha256",
        "parent_commit_sha",
        "primary_variable_sha256",
        "runtime_contract_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    }
)
_EXPERIMENT_EVIDENCE_LABELS = frozenset(
    {"aggregate_only", "protected", "public_grouped_robustness_not_unseen"}
)
_EXPERIMENT_RESULT_DECISIONS = frozenset({"adopt", "reject", "rollback"})
_EXPERIMENT_RESULT_REQUIRED_KEYS = frozenset(
    {
        "baseline_artifact_sha256",
        "candidate_artifact_sha256",
        "checks",
        "confusion_counts",
        "evaluator_sha256",
        "expected_record_count",
        "field_metrics",
        "input_tree_sha256",
        "metrics",
        "regression_counts",
        "runtime_contract_sha256",
        "runtime_evidence_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    }
)
_EXPERIMENT_RESULT_REQUIRED_CHECKS = frozenset(
    {
        "decision_freeze_verified",
        "deterministic",
        "fold_consistent",
        "runtime_limits_verified",
        "runtime_leakage_clean",
    }
)
_EXPERIMENT_RESULT_REQUIRED_METRICS = frozenset(
    {
        "calibration_score",
        "candidate_image_bytes",
        "candidate_max_model_artifact_bytes",
        "candidate_model_bytes",
        "catastrophic_false_approvals",
        "classification_score",
        "extraction_score",
        "invalid_records",
        "missing_records",
        "output_bytes",
        "peak_container_memory_bytes",
        "peak_rss_bytes",
        "process_cpu_seconds",
        "record_count",
        "runtime_seconds",
        "tmp_bytes",
        "total_score",
    }
)
_EXPERIMENT_RESULT_REQUIRED_REGRESSIONS = frozenset({"adversarial", "golden"})
_PROMOTION_POPULATION_KEYS = frozenset(
    {
        "evaluator_sha256",
        "expected_record_count",
        "input_tree_sha256",
        "runtime_contract_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    }
)
RETROSPECTIVE_RECONCILIATION_SCHEMA_VERSION = (
    "wo12-retrospective-reconciliation-v1"
)
_RECONCILIATION_LEDGER_NAMES = frozenset(
    {
        "candidate_state_ledger",
        "experiment_ledger",
        "protected_access_ledger",
        "taint_registry",
    }
)
_RECONCILIATION_EVIDENCE_FILENAMES = {
    "wo15": "WO15_GROUPED_RECOVERY_EVIDENCE.json",
    "wo16": "WO16_GROUPED_FUSION_EVIDENCE.json",
    "wo17": "WO17_POLICY_REVALIDATION_EVIDENCE.json",
    "wo18": "WO18_DECISION_RECOVERY_EVIDENCE.json",
    "wo19": "WO19_CONFIDENCE_REFIT_EVIDENCE.json",
}
_RECONCILIATION_EVIDENCE_WORK_ORDERS = frozenset(
    _RECONCILIATION_EVIDENCE_FILENAMES
)
_RECONCILIATION_LEDGER_FILENAMES = {
    "candidate_state_ledger": "candidate_state_ledger.jsonl",
    "experiment_ledger": "experiment_ledger.jsonl",
    "protected_access_ledger": "protected_access_ledger.jsonl",
    "taint_registry": "taint_registry.jsonl",
}
_RECONCILIATION_DISPOSITIONS = {
    "wo15": "rejected_single_fold_concentration",
    "wo16": "historical_candidate_rejected_negative_folds",
    "wo17": "historical_result_revalidation_pending",
    "wo18": "blocked_precondition_no_promotion",
    "wo19": "evaluated_no_promotion",
    "wo20": "pending_runtime_evidence",
    "wo21": "pending_adversarial_rerun",
}


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


def _normalize_experiment_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the immutable, pre-execution experiment contract."""

    if not isinstance(plan, Mapping) or set(plan) != _EXPERIMENT_PLAN_KEYS:
        raise ExperimentControlError(
            "experiment plan must contain exactly: "
            + ", ".join(sorted(_EXPERIMENT_PLAN_KEYS))
        )

    string_fields = tuple(
        _EXPERIMENT_PLAN_KEYS - {"changed_files", "expected_record_count"}
    )
    if any(not isinstance(plan[name], str) for name in string_fields):
        raise ExperimentControlError("experiment plan text and hashes must be strings")
    expected_record_count = plan["expected_record_count"]
    if (
        isinstance(expected_record_count, bool)
        or not isinstance(expected_record_count, int)
        or expected_record_count < 1
    ):
        raise ExperimentControlError(
            "expected_record_count must be a positive integer"
        )
    hypothesis_sha256 = plan["hypothesis_sha256"].strip().lower()
    primary_variable_sha256 = plan["primary_variable_sha256"].strip().lower()
    parent_commit_sha = plan["parent_commit_sha"].strip().lower()
    evidence_label = plan["evidence_label"].strip().casefold()
    bound_hashes = {
        name: plan[name].strip().lower()
        for name in (
            "evaluator_sha256",
            "input_tree_sha256",
            "runtime_contract_sha256",
            "split_manifest_sha256",
            "truth_sha256",
        )
    }
    if not _SHA256_RE.fullmatch(hypothesis_sha256):
        raise ExperimentControlError(
            "hypothesis_sha256 must bind one external non-repository hypothesis"
        )
    if not _SHA256_RE.fullmatch(primary_variable_sha256):
        raise ExperimentControlError(
            "primary_variable_sha256 must bind one external primary variable"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", parent_commit_sha):
        raise ExperimentControlError("parent_commit_sha must be a full Git SHA")
    if evidence_label not in _EXPERIMENT_EVIDENCE_LABELS:
        raise ExperimentControlError("experiment evidence_label is invalid")
    for name, value in bound_hashes.items():
        if not _SHA256_RE.fullmatch(value):
            raise ExperimentControlError(
                f"{name} must be a SHA-256 hex digest"
            )

    raw_changed_files = plan["changed_files"]
    if not isinstance(raw_changed_files, (list, tuple)):
        raise ExperimentControlError("changed_files must be a list of repository paths")
    if not raw_changed_files:
        raise ExperimentControlError("changed_files must name at least one changed file")
    changed_files: list[str] = []
    for raw_path in raw_changed_files:
        if not isinstance(raw_path, str):
            raise ExperimentControlError("changed_files entries must be strings")
        path = raw_path.strip()
        pure_path = PurePosixPath(path)
        if (
            not path
            or path != raw_path
            or "\\" in path
            or pure_path.is_absolute()
            or str(pure_path) != path
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise ExperimentControlError(
                "changed_files entries must be normalized repository-relative paths"
            )
        _require_nonidentifying_control_text("changed_files entry", path)
        changed_files.append(path)
    if len(set(changed_files)) != len(changed_files):
        raise ExperimentControlError("changed_files entries must be unique")

    normalized = {
        "changed_files": sorted(changed_files),
        "evidence_label": evidence_label,
        "expected_record_count": expected_record_count,
        "hypothesis_sha256": hypothesis_sha256,
        "parent_commit_sha": parent_commit_sha,
        "primary_variable_sha256": primary_variable_sha256,
        **bound_hashes,
    }
    canonical_json(normalized)
    return normalized


def _normalize_promotion_population(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize the immutable population used for milestone comparisons."""

    if not isinstance(value, Mapping) or set(value) != _PROMOTION_POPULATION_KEYS:
        raise ExperimentControlError(
            "promotion_population must contain the exact protected population tuple"
        )
    normalized: dict[str, Any] = {}
    for name in sorted(_PROMOTION_POPULATION_KEYS - {"expected_record_count"}):
        digest = value[name]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ExperimentControlError(
                f"promotion_population {name} must be a SHA-256 digest"
            )
        normalized[name] = digest.lower()
    expected_record_count = value["expected_record_count"]
    if (
        isinstance(expected_record_count, bool)
        or not isinstance(expected_record_count, int)
        or expected_record_count < 1
    ):
        raise ExperimentControlError(
            "promotion_population expected_record_count must be positive"
        )
    normalized["expected_record_count"] = expected_record_count
    canonical_json(normalized)
    return normalized


def _promotion_population_from(
    value: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a normalized population tuple when all fields are present."""

    if not _PROMOTION_POPULATION_KEYS.issubset(value):
        return None
    return _normalize_promotion_population(
        {name: value[name] for name in _PROMOTION_POPULATION_KEYS}
    )


def _normalize_experiment_result_evidence(
    evidence: Mapping[str, Any],
    *,
    decision: str,
    evidence_label: str,
) -> dict[str, Any]:
    """Require the complete aggregate experiment-result contract."""

    if not isinstance(evidence, Mapping):
        raise ExperimentControlError("experiment result evidence must be an object")
    require_aggregate_only(evidence)
    normalized = json.loads(canonical_json(dict(evidence)))
    missing_root = _EXPERIMENT_RESULT_REQUIRED_KEYS - set(normalized)
    if missing_root:
        raise ExperimentControlError(
            "experiment result is missing contract keys: "
            + ", ".join(sorted(missing_root))
        )

    for key in (
        "baseline_artifact_sha256",
        "candidate_artifact_sha256",
        "evaluator_sha256",
        "input_tree_sha256",
        "runtime_contract_sha256",
        "runtime_evidence_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    ):
        if not isinstance(normalized[key], str) or not _SHA256_RE.fullmatch(
            normalized[key]
        ):
            raise ExperimentControlError(f"{key} must be a SHA-256 hex digest")

    expected_record_count = normalized["expected_record_count"]
    if (
        isinstance(expected_record_count, bool)
        or not isinstance(expected_record_count, int)
        or expected_record_count < 1
    ):
        raise ExperimentControlError(
            "expected_record_count must be a positive integer"
        )

    checks = normalized["checks"]
    if not isinstance(checks, Mapping):
        raise ExperimentControlError("experiment result checks must be an object")
    missing_checks = _EXPERIMENT_RESULT_REQUIRED_CHECKS - set(checks)
    if missing_checks or any(
        not isinstance(value, bool) for value in checks.values()
    ):
        raise ExperimentControlError(
            "experiment result checks are incomplete or non-boolean"
        )

    metrics = normalized["metrics"]
    if not isinstance(metrics, Mapping):
        raise ExperimentControlError("experiment result metrics must be an object")
    missing_metrics = _EXPERIMENT_RESULT_REQUIRED_METRICS - set(metrics)
    if missing_metrics:
        raise ExperimentControlError(
            "experiment result is missing metrics: "
            + ", ".join(sorted(missing_metrics))
        )
    integer_metrics = {
        "candidate_image_bytes",
        "candidate_max_model_artifact_bytes",
        "candidate_model_bytes",
        "catastrophic_false_approvals",
        "invalid_records",
        "missing_records",
        "output_bytes",
        "peak_container_memory_bytes",
        "peak_rss_bytes",
        "record_count",
        "tmp_bytes",
    }
    for name in _EXPERIMENT_RESULT_REQUIRED_METRICS:
        metric = metrics[name]
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or metric < 0
            or (name in integer_metrics and not isinstance(metric, int))
        ):
            raise ExperimentControlError(
                f"experiment result metric must be non-negative: {name}"
            )
    score_limits = {
        "calibration_score": 20.0,
        "classification_score": 80.0,
        "extraction_score": 50.0,
        "total_score": 150.0,
    }
    if any(metrics[name] > maximum for name, maximum in score_limits.items()):
        raise ExperimentControlError("experiment component score is out of range")
    component_total = (
        metrics["extraction_score"]
        + metrics["classification_score"]
        + metrics["calibration_score"]
    )
    if abs(metrics["total_score"] - component_total) > 1e-9:
        raise ExperimentControlError(
            "experiment total_score must equal its component scores"
        )

    regressions = normalized["regression_counts"]
    if not isinstance(regressions, Mapping):
        raise ExperimentControlError(
            "experiment regression_counts must be an object"
        )
    missing_regressions = _EXPERIMENT_RESULT_REQUIRED_REGRESSIONS - set(
        regressions
    )
    if missing_regressions or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in regressions.values()
    ):
        raise ExperimentControlError(
            "experiment regression counts are incomplete or invalid"
        )

    for key in ("confusion_counts", "field_metrics"):
        if not isinstance(normalized[key], Mapping) or not normalized[key]:
            raise ExperimentControlError(
                f"experiment result {key} must contain aggregate metrics"
            )

    protected_hash = normalized.get("protected_access_record_hash")
    if evidence_label == "protected":
        if (
            not isinstance(protected_hash, str)
            or not _SHA256_RE.fullmatch(protected_hash)
        ):
            raise ExperimentControlError(
                "protected results require a protected access record hash"
            )
    elif protected_hash is not None:
        raise ExperimentControlError(
            "non-protected results may not claim protected access"
        )

    if decision == "adopt":
        if evidence_label in {
            "protected",
            "public_grouped_robustness_not_unseen",
        }:
            fold_count = normalized.get("fold_count")
            repeat_count = normalized.get("repeat_count")
            fold_deltas = normalized.get("fold_deltas")
            fold_weights = normalized.get("fold_weights")
            repeat_scores = normalized.get("repeat_scores")
            if (
                isinstance(fold_count, bool)
                or not isinstance(fold_count, int)
                or fold_count != 5
                or isinstance(repeat_count, bool)
                or not isinstance(repeat_count, int)
                or repeat_count != 3
                or not isinstance(fold_deltas, list)
                or len(fold_deltas) != fold_count * repeat_count
                or not isinstance(fold_weights, list)
                or len(fold_weights) != fold_count * repeat_count
                or not isinstance(repeat_scores, list)
                or len(repeat_scores) != repeat_count
                or any(
                    isinstance(delta, bool)
                    or not isinstance(delta, (int, float))
                    or delta < 0
                    for delta in fold_deltas
                )
                or any(
                    isinstance(weight, bool)
                    or not isinstance(weight, int)
                    or weight <= 0
                    for weight in fold_weights
                )
                or any(
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or score <= 0
                    for score in repeat_scores
                )
            ):
                raise ExperimentControlError(
                    "public-grouped adoption requires repeated non-negative fold evidence"
                )
            for repeat in range(repeat_count):
                offset = repeat * fold_count
                repeat_deltas = fold_deltas[offset : offset + fold_count]
                repeat_weights = fold_weights[offset : offset + fold_count]
                if sum(repeat_weights) != metrics["record_count"]:
                    raise ExperimentControlError(
                        "each repeat's fold weights must cover record_count"
                    )
                weighted_sum = sum(
                    delta * weight
                    for delta, weight in zip(repeat_deltas, repeat_weights)
                )
                total_weight = sum(repeat_weights)
                observed = weighted_sum / total_weight
                if abs(observed - repeat_scores[repeat]) > 1e-9:
                    raise ExperimentControlError(
                        "repeat_scores must equal their weighted repeated-fold means"
                    )
                leave_one_out = [
                    (weighted_sum - delta * weight) / (total_weight - weight)
                    for delta, weight in zip(repeat_deltas, repeat_weights)
                ]
                if min(leave_one_out) <= 1e-12:
                    raise ExperimentControlError(
                        "public-grouped adoption may not depend on one fold "
                        "within any repeat"
                    )
        unsafe = (
            metrics["catastrophic_false_approvals"] != 0
            or metrics["record_count"] <= 0
            or metrics["record_count"] != expected_record_count
            or metrics["missing_records"] != 0
            or metrics["invalid_records"] != 0
            or metrics["candidate_image_bytes"] <= 0
            or metrics["candidate_image_bytes"] > 4 * 1024**3
            or metrics["candidate_model_bytes"] > 1024**3
            or metrics["candidate_max_model_artifact_bytes"] > 250 * 1024**2
            or metrics["output_bytes"] <= 0
            or metrics["output_bytes"] > 25 * 1024**2
            or metrics["peak_container_memory_bytes"] <= 0
            or metrics["peak_container_memory_bytes"] > 8 * 1024**3
            or metrics["peak_rss_bytes"] <= 0
            or metrics["peak_rss_bytes"] > 8 * 1024**3
            or metrics["tmp_bytes"] > 2 * 1024**3
            or metrics["runtime_seconds"] <= 0
            or metrics["runtime_seconds"] > 30_000
            or (
                metrics["record_count"] > 0
                and metrics["runtime_seconds"] / metrics["record_count"] > 6
            )
            or metrics["process_cpu_seconds"] <= 0
            or metrics["process_cpu_seconds"] > 120_000
            or any(regressions.values())
            or any(not value for value in checks.values())
        )
        if unsafe:
            raise ExperimentControlError(
                "an experiment with failed hard gates cannot be adopted"
            )
    return normalized


def require_retrospective_reconciliation(value: Any) -> None:
    """Validate a non-authoritative WO15-WO21 reconciliation artifact.

    This deliberately separate artifact can bind historical evidence to the
    published ledger snapshot. It can never claim preregistration, mutate a
    published ledger, consume protected access, or authorize promotion.
    """

    root_keys = {
        "authority",
        "classification",
        "evidence_files",
        "integrity_heads",
        "latest_governed_passing_candidate",
        "ledger_anchors",
        "schema_version",
        "work_order_dispositions",
    }
    if not isinstance(value, Mapping) or set(value) != root_keys:
        raise ExperimentControlError(
            "retrospective reconciliation has an invalid root schema"
        )
    if (
        value["schema_version"] != RETROSPECTIVE_RECONCILIATION_SCHEMA_VERSION
        or value["classification"] != "retrospective_not_preregistered"
    ):
        raise ExperimentControlError(
            "retrospective reconciliation classification is invalid"
        )

    required_authority = {
        "baseline_state_changed": False,
        "candidate_promotion_recorded": False,
        "candidate_state_changed": False,
        "new_protected_access_recorded": False,
        "preregistered": False,
        "promotion_authority": False,
        "protected_access_consumed": False,
        "published_ledgers_mutated": False,
    }
    if value["authority"] != required_authority:
        raise ExperimentControlError(
            "retrospective reconciliation may not claim governance authority"
        )

    integrity_heads = value["integrity_heads"]
    if not isinstance(integrity_heads, Mapping) or set(integrity_heads) != {
        "path",
        "sha256",
    }:
        raise ExperimentControlError("integrity-heads binding is invalid")
    if (
        integrity_heads["path"] != "evaluation/program/integrity_heads.json"
        or not isinstance(integrity_heads["sha256"], str)
        or not _SHA256_RE.fullmatch(integrity_heads["sha256"])
    ):
        raise ExperimentControlError("integrity-heads values are invalid")

    ledger_anchors = value["ledger_anchors"]
    if (
        not isinstance(ledger_anchors, Mapping)
        or set(ledger_anchors) != _RECONCILIATION_LEDGER_NAMES
    ):
        raise ExperimentControlError(
            "retrospective reconciliation must bind every published ledger"
        )
    for name, anchor in ledger_anchors.items():
        if not isinstance(anchor, Mapping) or set(anchor) != {
            "expected_head",
            "expected_length",
            "path",
            "sha256",
        }:
            raise ExperimentControlError(f"{name} has an invalid ledger anchor")
        if (
            not isinstance(anchor["expected_head"], str)
            or not _SHA256_RE.fullmatch(anchor["expected_head"])
            or isinstance(anchor["expected_length"], bool)
            or not isinstance(anchor["expected_length"], int)
            or anchor["expected_length"] < 0
            or anchor["path"]
            != f"evaluation/program/{_RECONCILIATION_LEDGER_FILENAMES[name]}"
            or not isinstance(anchor["sha256"], str)
            or not _SHA256_RE.fullmatch(anchor["sha256"])
        ):
            raise ExperimentControlError(f"{name} ledger anchor is invalid")

    evidence_files = value["evidence_files"]
    if (
        not isinstance(evidence_files, Mapping)
        or set(evidence_files) != _RECONCILIATION_EVIDENCE_WORK_ORDERS
    ):
        raise ExperimentControlError(
            "retrospective reconciliation evidence bindings are invalid"
        )
    for work_order, binding in evidence_files.items():
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise ExperimentControlError(
                f"{work_order} reconciliation evidence binding is invalid"
            )
        path = binding["path"]
        if (
            not isinstance(path, str)
            or path
            != f"evaluation/{_RECONCILIATION_EVIDENCE_FILENAMES[work_order]}"
            or PurePosixPath(path).is_absolute()
            or str(PurePosixPath(path)) != path
            or any(part in {"", ".", ".."} for part in PurePosixPath(path).parts)
            or not isinstance(binding["sha256"], str)
            or not _SHA256_RE.fullmatch(binding["sha256"])
        ):
            raise ExperimentControlError(
                f"{work_order} reconciliation evidence values are invalid"
            )
    if value["work_order_dispositions"] != _RECONCILIATION_DISPOSITIONS:
        raise ExperimentControlError(
            "retrospective reconciliation dispositions are invalid"
        )

    candidate = value["latest_governed_passing_candidate"]
    if not isinstance(candidate, Mapping) or set(candidate) != {
        "candidate_sha256",
        "record_hash",
        "total_score",
    }:
        raise ExperimentControlError(
            "latest governed passing candidate binding is invalid"
        )
    if (
        not isinstance(candidate["candidate_sha256"], str)
        or not _SHA256_RE.fullmatch(candidate["candidate_sha256"])
        or not isinstance(candidate["record_hash"], str)
        or not _SHA256_RE.fullmatch(candidate["record_hash"])
        or isinstance(candidate["total_score"], bool)
        or not isinstance(candidate["total_score"], (int, float))
        or not 0.0 <= candidate["total_score"] <= 150.0
    ):
        raise ExperimentControlError(
            "latest governed passing candidate values are invalid"
        )
    canonical_json(value)


def build_retrospective_reconciliation(
    *,
    integrity_heads_path: Path | str,
    evidence_paths: Mapping[str, Path | str],
) -> dict[str, Any]:
    """Verify real files and build the non-authoritative reconciliation shape."""

    integrity_path = Path(integrity_heads_path)
    try:
        integrity_bytes = integrity_path.read_bytes()
        integrity = json.loads(integrity_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError("integrity-heads artifact is unreadable") from exc
    if (
        not isinstance(integrity, Mapping)
        or not isinstance(integrity.get("stores"), Mapping)
        or set(integrity["stores"]) != _RECONCILIATION_LEDGER_NAMES
    ):
        raise IntegrityError("integrity-heads store schema is invalid")
    try:
        repository_root = integrity_path.resolve().parents[2]
        integrity_relative_path = integrity_path.resolve().relative_to(
            repository_root
        ).as_posix()
    except (IndexError, ValueError) as exc:
        raise IntegrityError(
            "integrity-heads path must be inside a repository layout"
        ) from exc
    if integrity_relative_path != "evaluation/program/integrity_heads.json":
        raise IntegrityError(
            "integrity-heads path must be evaluation/program/integrity_heads.json"
        )

    ledger_anchors: dict[str, dict[str, Any]] = {}
    ledger_records: dict[str, tuple[dict[str, Any], ...]] = {}
    for name, filename in _RECONCILIATION_LEDGER_FILENAMES.items():
        anchor = integrity["stores"][name]
        if not isinstance(anchor, Mapping):
            raise IntegrityError(f"{name} integrity anchor is invalid")
        ledger_path = integrity_path.parent / filename
        try:
            ledger_bytes = ledger_path.read_bytes()
        except OSError as exc:
            raise IntegrityError(f"{name} ledger is unreadable") from exc
        if _sha256_bytes(ledger_bytes) != anchor.get("sha256"):
            raise IntegrityError(f"{name} ledger file hash does not match")
        store = CanonicalHashChainStore(ledger_path)
        ledger_records[name] = store.verify(
            expected_head=anchor.get("expected_head"),
            expected_length=anchor.get("expected_length"),
        )
        ledger_anchors[name] = {
            "expected_head": anchor.get("expected_head"),
            "expected_length": anchor.get("expected_length"),
            "path": ledger_path.resolve().relative_to(repository_root).as_posix(),
            "sha256": anchor.get("sha256"),
        }

    if (
        not isinstance(evidence_paths, Mapping)
        or set(evidence_paths) != _RECONCILIATION_EVIDENCE_WORK_ORDERS
    ):
        raise ExperimentControlError(
            "reconciliation must name the WO15-WO19 evidence files"
        )
    evidence_files: dict[str, dict[str, str]] = {}
    for work_order, raw_path in evidence_paths.items():
        path = Path(raw_path)
        try:
            relative_path = path.resolve().relative_to(repository_root).as_posix()
            evidence_files[str(work_order)] = {
                "path": relative_path,
                "sha256": _sha256_bytes(path.read_bytes()),
            }
        except (OSError, ValueError) as exc:
            raise IntegrityError(
                f"{work_order} reconciliation evidence is unreadable or external"
            ) from exc

    passing_records = [
        record
        for record in ledger_records["candidate_state_ledger"]
        if record["payload"].get("event") == "candidate_assessment"
        and record["payload"].get("decision") == "PASSED"
    ]
    if not passing_records:
        raise IntegrityError("candidate-state ledger has no passing candidate")
    latest_passing = passing_records[-1]["payload"]
    try:
        latest_candidate_sha256 = latest_passing["candidate_sha256"]
        latest_total_score = latest_passing["aggregate_evidence"]["metrics"][
            "total_score"
        ]
    except (KeyError, TypeError) as exc:
        raise IntegrityError(
            "latest passing candidate lacks its aggregate score binding"
        ) from exc

    artifact = {
        "authority": {
            "baseline_state_changed": False,
            "candidate_promotion_recorded": False,
            "candidate_state_changed": False,
            "new_protected_access_recorded": False,
            "preregistered": False,
            "promotion_authority": False,
            "protected_access_consumed": False,
            "published_ledgers_mutated": False,
        },
        "classification": "retrospective_not_preregistered",
        "evidence_files": evidence_files,
        "integrity_heads": {
            "path": integrity_relative_path,
            "sha256": _sha256_bytes(integrity_bytes),
        },
        "latest_governed_passing_candidate": {
            "candidate_sha256": latest_candidate_sha256,
            "record_hash": passing_records[-1]["record_hash"],
            "total_score": latest_total_score,
        },
        "ledger_anchors": ledger_anchors,
        "schema_version": RETROSPECTIVE_RECONCILIATION_SCHEMA_VERSION,
        "work_order_dispositions": dict(_RECONCILIATION_DISPOSITIONS),
    }
    require_retrospective_reconciliation(artifact)
    normalized = json.loads(canonical_json(artifact))
    verify_retrospective_reconciliation(
        normalized,
        repository_root=repository_root,
    )
    return normalized


def verify_retrospective_reconciliation(
    value: Mapping[str, Any],
    *,
    repository_root: Path | str,
) -> None:
    """Verify every reconciliation path, digest, ledger head, and candidate pin."""

    require_retrospective_reconciliation(value)
    root = Path(repository_root).resolve()

    def bound_path(relative_path: str) -> Path:
        candidate = (root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise IntegrityError(
                "reconciliation binding escapes the repository"
            ) from exc
        return candidate

    integrity_binding = value["integrity_heads"]
    integrity_path = bound_path(integrity_binding["path"])
    try:
        integrity_bytes = integrity_path.read_bytes()
        integrity = json.loads(integrity_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError("bound integrity-heads artifact is unreadable") from exc
    if _sha256_bytes(integrity_bytes) != integrity_binding["sha256"]:
        raise IntegrityError("bound integrity-heads artifact hash changed")
    if (
        not isinstance(integrity, Mapping)
        or not isinstance(integrity.get("stores"), Mapping)
        or set(integrity["stores"]) != _RECONCILIATION_LEDGER_NAMES
    ):
        raise IntegrityError("bound integrity-heads store schema is invalid")

    for name, anchor in value["ledger_anchors"].items():
        expected_anchor = {
            "expected_head": anchor["expected_head"],
            "expected_length": anchor["expected_length"],
            "sha256": anchor["sha256"],
        }
        if integrity["stores"].get(name) != expected_anchor:
            raise IntegrityError(
                f"bound {name} ledger anchor contradicts integrity-heads"
            )
        ledger_path = bound_path(anchor["path"])
        try:
            ledger_bytes = ledger_path.read_bytes()
        except OSError as exc:
            raise IntegrityError(f"bound {name} ledger is unreadable") from exc
        if _sha256_bytes(ledger_bytes) != anchor["sha256"]:
            raise IntegrityError(f"bound {name} ledger file hash changed")
        CanonicalHashChainStore(ledger_path).verify(
            expected_head=anchor["expected_head"],
            expected_length=anchor["expected_length"],
        )

    for work_order, binding in value["evidence_files"].items():
        evidence_path = bound_path(binding["path"])
        try:
            evidence_bytes = evidence_path.read_bytes()
        except OSError as exc:
            raise IntegrityError(
                f"bound {work_order} evidence is unreadable"
            ) from exc
        if _sha256_bytes(evidence_bytes) != binding["sha256"]:
            raise IntegrityError(f"bound {work_order} evidence hash changed")

    candidate_binding = value["latest_governed_passing_candidate"]
    candidate_records = CanonicalHashChainStore(
        bound_path(value["ledger_anchors"]["candidate_state_ledger"]["path"])
    ).verify()
    passing_candidates = [
        record
        for record in candidate_records
        if record["payload"].get("event") == "candidate_assessment"
        and record["payload"].get("decision") == "PASSED"
    ]
    if (
        not passing_candidates
        or passing_candidates[-1]["record_hash"] != candidate_binding["record_hash"]
    ):
        raise IntegrityError(
            "bound candidate is not the latest governed passing candidate"
        )
    matching_candidates = [
        record
        for record in candidate_records
        if record["record_hash"] == candidate_binding["record_hash"]
        and record["payload"].get("event") == "candidate_assessment"
        and record["payload"].get("decision") == "PASSED"
    ]
    if len(matching_candidates) != 1:
        raise IntegrityError(
            "bound latest governed candidate is not a passing ledger record"
        )
    payload = matching_candidates[0]["payload"]
    if (
        payload.get("candidate_sha256") != candidate_binding["candidate_sha256"]
        or payload.get("aggregate_evidence", {}).get("metrics", {}).get(
            "total_score"
        )
        != candidate_binding["total_score"]
    ):
        raise IntegrityError("bound latest governed candidate values changed")


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


def _normalize_program_integrity_stores(
    stores: Mapping[str, CanonicalHashChainStore],
) -> dict[str, CanonicalHashChainStore]:
    if not isinstance(stores, Mapping) or set(stores) != _RECONCILIATION_LEDGER_NAMES:
        raise ExperimentControlError(
            "program integrity checkpoint requires all four governed stores"
        )
    normalized: dict[str, CanonicalHashChainStore] = {}
    resolved_paths: set[Path] = set()
    for name in sorted(_RECONCILIATION_LEDGER_NAMES):
        store = stores[name]
        if not isinstance(store, CanonicalHashChainStore):
            raise ExperimentControlError(
                f"{name} must be a CanonicalHashChainStore"
            )
        resolved_path = store.path.resolve()
        if resolved_path in resolved_paths:
            raise ExperimentControlError(
                "program integrity stores must use distinct ledger paths"
            )
        resolved_paths.add(resolved_path)
        normalized[name] = store
    return normalized


def _program_store_snapshot(
    store: CanonicalHashChainStore,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    try:
        raw = store.path.read_bytes() if store.path.exists() else b""
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise IntegrityError(
            f"governed ledger is unreadable: {store.path}"
        ) from exc
    records = store._parse(text)
    return (
        {
            "expected_head": store._head(records),
            "expected_length": len(records),
            "sha256": _sha256_bytes(raw),
        },
        records,
    )


def _validate_program_integrity_checkpoint(value: Any) -> dict[str, Any]:
    legacy_root_keys = {
        "baseline_manifest_sha256",
        "runtime_leakage_finding_count",
        "stores",
    }
    current_root_keys = legacy_root_keys | {"promotion_population"}
    if (
        not isinstance(value, Mapping)
        or set(value) not in (legacy_root_keys, current_root_keys)
    ):
        raise IntegrityError("program integrity checkpoint root schema is invalid")
    baseline_sha256 = value["baseline_manifest_sha256"]
    finding_count = value["runtime_leakage_finding_count"]
    stores = value["stores"]
    if (
        not isinstance(baseline_sha256, str)
        or not _SHA256_RE.fullmatch(baseline_sha256)
        or isinstance(finding_count, bool)
        or not isinstance(finding_count, int)
        or finding_count < 0
        or not isinstance(stores, Mapping)
        or set(stores) != _RECONCILIATION_LEDGER_NAMES
    ):
        raise IntegrityError("program integrity checkpoint values are invalid")
    if "promotion_population" in value:
        try:
            _normalize_promotion_population(value["promotion_population"])
        except ExperimentControlError as exc:
            raise IntegrityError(
                "program integrity checkpoint promotion population is invalid"
            ) from exc
    anchor_shapes: set[frozenset[str]] = set()
    for name in sorted(_RECONCILIATION_LEDGER_NAMES):
        anchor = stores[name]
        legacy_keys = {
            "expected_head",
            "expected_length",
            "sha256",
        }
        current_keys = legacy_keys | {"path"}
        if (
            not isinstance(anchor, Mapping)
            or (
                set(anchor) != legacy_keys
                and set(anchor) != current_keys
            )
        ):
            raise IntegrityError(f"{name} checkpoint anchor schema is invalid")
        anchor_shapes.add(frozenset(anchor))
        if (
            not isinstance(anchor["expected_head"], str)
            or not _SHA256_RE.fullmatch(anchor["expected_head"])
            or isinstance(anchor["expected_length"], bool)
            or not isinstance(anchor["expected_length"], int)
            or anchor["expected_length"] < 0
            or not isinstance(anchor["sha256"], str)
            or not _SHA256_RE.fullmatch(anchor["sha256"])
        ):
            raise IntegrityError(f"{name} checkpoint anchor values are invalid")
        if "path" in anchor:
            raw_path = anchor["path"]
            pure_path = PurePosixPath(raw_path) if isinstance(raw_path, str) else None
            if (
                pure_path is None
                or not raw_path
                or "\\" in raw_path
                or pure_path.is_absolute()
                or str(pure_path) != raw_path
                or any(part in {"", "."} for part in pure_path.parts)
            ):
                raise IntegrityError(f"{name} checkpoint path is invalid")
    if len(anchor_shapes) != 1:
        raise IntegrityError(
            "program integrity checkpoint may not mix legacy and path-bound anchors"
        )
    return json.loads(canonical_json(dict(value)))


def build_program_integrity_checkpoint(
    *,
    stores: Mapping[str, CanonicalHashChainStore],
    baseline_manifest_sha256: str,
    checkpoint_directory: Path | str,
    runtime_leakage_finding_count: int,
    promotion_population: Mapping[str, Any] | None = None,
) -> tuple[bytes, str]:
    """Build one canonical exact snapshot without publishing it as trusted.

    The returned digest becomes authoritative only after an external system
    such as Git or the Factory publishes it as the single current checkpoint.
    """

    normalized_stores = _normalize_program_integrity_stores(stores)
    checkpoint_root = Path(checkpoint_directory).resolve()
    anchors: dict[str, dict[str, Any]] = {}
    for name, store in normalized_stores.items():
        anchor = _program_store_snapshot(store)[0]
        anchor["path"] = Path(
            os.path.relpath(store.path.resolve(), checkpoint_root)
        ).as_posix()
        anchors[name] = anchor
    checkpoint = {
        "baseline_manifest_sha256": str(
            baseline_manifest_sha256
        ).strip().lower(),
        "runtime_leakage_finding_count": runtime_leakage_finding_count,
        "stores": anchors,
    }
    if promotion_population is not None:
        checkpoint["promotion_population"] = _normalize_promotion_population(
            promotion_population
        )
    normalized = _validate_program_integrity_checkpoint(checkpoint)
    raw = (canonical_json(normalized) + "\n").encode("utf-8")
    return raw, _sha256_bytes(raw)


@dataclass(frozen=True)
class ProgramIntegritySuccessor:
    """Unpublished checkpoint produced after one verified governed mutation."""

    previous_checkpoint_sha256: str
    checkpoint_bytes: bytes
    checkpoint_sha256: str
    mutated_store: str
    record_hash: str
    mutated: bool


@dataclass(frozen=True)
class PublishedCheckpointReference:
    """One current checkpoint reference returned by Git/Factory authority."""

    path: Path | str
    sha256: str


_CHECKPOINT_AUTHORITY = object()
_PROGRAM_LOCK_REGISTRY_GUARD = threading.Lock()
_PROGRAM_LOCK_REGISTRY: dict[str, threading.RLock] = {}
_PROGRAM_LOCK_STATE = threading.local()


def _program_thread_lock(lock_path: Path) -> threading.RLock:
    key = str(lock_path.resolve())
    with _PROGRAM_LOCK_REGISTRY_GUARD:
        lock = _PROGRAM_LOCK_REGISTRY.get(key)
        if lock is None:
            lock = threading.RLock()
            _PROGRAM_LOCK_REGISTRY[key] = lock
        return lock


class CheckpointAuthorityResolver:
    """Resolve the sole current checkpoint through an external authority.

    The callback must query an authenticated Git/Factory control plane. It must
    not derive a digest from the mutable ledgers that the caller is asking to
    change.
    """

    def __init__(
        self,
        resolve_current: Callable[[], PublishedCheckpointReference],
        *,
        trusted_checkpoint_root: Path | str,
    ) -> None:
        if not callable(resolve_current):
            raise ExperimentControlError(
                "checkpoint authority resolver must be callable"
            )
        self._resolve_current = resolve_current
        self._trusted_root = Path(trusted_checkpoint_root).resolve()

    def resolve(
        self,
        *,
        stores: Mapping[str, CanonicalHashChainStore],
    ) -> ProgramIntegrityCheckpoint:
        reference = self._resolve_current()
        if not isinstance(reference, PublishedCheckpointReference):
            raise IntegrityError(
                "checkpoint authority returned an invalid current reference"
            )
        checkpoint_path = Path(reference.path).resolve()
        try:
            checkpoint_path.relative_to(self._trusted_root)
        except ValueError as exc:
            raise IntegrityError(
                "checkpoint authority escaped its trusted checkpoint root"
            ) from exc
        checkpoint = ProgramIntegrityCheckpoint(
            checkpoint_path,
            expected_sha256=reference.sha256,
            stores=stores,
            _authority=_CHECKPOINT_AUTHORITY,
        )
        checkpoint.verify()
        return checkpoint

    @staticmethod
    def validate_successor(
        current: ProgramIntegrityCheckpoint,
        successor: ProgramIntegritySuccessor,
    ) -> None:
        """Validate the transition before an external authority publishes it."""

        current._require_authority()
        if not isinstance(successor, ProgramIntegritySuccessor):
            raise IntegrityError(
                "checkpoint successor has an invalid transition envelope"
            )
        with current.locked():
            try:
                expected = current.successor(
                    mutated_store=successor.mutated_store,
                    record_hash=successor.record_hash,
                )
            except ExperimentControlError as exc:
                raise IntegrityError(
                    "checkpoint successor is not the exact current transition"
                ) from exc
            actual_transition = (
                successor.previous_checkpoint_sha256,
                successor.checkpoint_bytes,
                successor.checkpoint_sha256,
                successor.mutated_store,
                successor.record_hash,
                successor.mutated,
            )
            expected_transition = (
                expected.previous_checkpoint_sha256,
                expected.checkpoint_bytes,
                expected.checkpoint_sha256,
                expected.mutated_store,
                expected.record_hash,
                expected.mutated,
            )
            if actual_transition != expected_transition:
                raise IntegrityError(
                    "checkpoint successor is not the exact recomputed transition"
                )


@dataclass(frozen=True, eq=False)
class GovernedMutationReceipt(Mapping[str, Any]):
    """Prior API mapping plus its unpublished exact successor checkpoint."""

    value: dict[str, Any]
    record_hash: str
    integrity: ProgramIntegritySuccessor

    def __getitem__(self, key: str) -> Any:
        return self.value[key]

    def __iter__(self):
        return iter(self.value)

    def __len__(self) -> int:
        return len(self.value)

    @property
    def assessment(self) -> dict[str, Any]:
        return self.value

    @property
    def assessment_record_hash(self) -> str:
        return self.record_hash

    @property
    def next_checkpoint_bytes(self) -> bytes:
        return self.integrity.checkpoint_bytes

    @property
    def next_checkpoint_sha256(self) -> str:
        return self.integrity.checkpoint_sha256

    @property
    def previous_checkpoint_sha256(self) -> str:
        return self.integrity.previous_checkpoint_sha256

    @property
    def mutated(self) -> bool:
        return self.integrity.mutated


class ProgramIntegrityCheckpoint:
    """An exact four-ledger checkpoint whose digest is supplied externally."""

    def __init__(
        self,
        path: Path | str,
        *,
        expected_sha256: str,
        stores: Mapping[str, CanonicalHashChainStore],
        _authority: object | None = None,
    ) -> None:
        normalized_digest = str(expected_sha256).strip().lower()
        if not _SHA256_RE.fullmatch(normalized_digest):
            raise ExperimentControlError(
                "expected checkpoint SHA-256 must be a full digest"
            )
        self.path = Path(path)
        self.expected_sha256 = normalized_digest
        self._stores = _normalize_program_integrity_stores(stores)
        self._authority = _authority

    @property
    def stores(self) -> Mapping[str, CanonicalHashChainStore]:
        return dict(self._stores)

    @property
    def authority_verified(self) -> bool:
        return self._authority is _CHECKPOINT_AUTHORITY

    def _require_authority(self) -> None:
        if not self.authority_verified:
            raise IntegrityError(
                "governed mutation requires the authority-resolved current checkpoint"
            )

    @contextmanager
    def locked(self):
        """Hold the program-wide mutation lock, re-entrantly in one thread."""

        self._require_authority()
        lock_path = self.path.parent / ".program-integrity.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        key = str(lock_path.resolve())
        thread_lock = _program_thread_lock(lock_path)
        with thread_lock:
            depths = getattr(_PROGRAM_LOCK_STATE, "depths", None)
            if depths is None:
                depths = {}
                _PROGRAM_LOCK_STATE.depths = depths
            entry = depths.get(key)
            if entry is not None:
                entry["depth"] += 1
                try:
                    yield
                finally:
                    entry["depth"] -= 1
                return

            flags = os.O_RDWR | os.O_CREAT
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(lock_path, flags, 0o600)
            except OSError as exc:
                raise IntegrityError(
                    "program-integrity lock cannot be opened safely"
                ) from exc
            handle = os.fdopen(descriptor, "a+", encoding="utf-8")

            def require_same_regular_lock() -> None:
                descriptor_stat = os.fstat(handle.fileno())
                try:
                    path_stat = os.stat(lock_path, follow_symlinks=False)
                except OSError as exc:
                    raise IntegrityError(
                        "program-integrity lock path was replaced"
                    ) from exc
                if (
                    not stat.S_ISREG(descriptor_stat.st_mode)
                    or not stat.S_ISREG(path_stat.st_mode)
                    or descriptor_stat.st_nlink != 1
                    or path_stat.st_nlink != 1
                    or descriptor_stat.st_dev != path_stat.st_dev
                    or descriptor_stat.st_ino != path_stat.st_ino
                    or (
                        hasattr(os, "geteuid")
                        and descriptor_stat.st_uid != os.geteuid()
                    )
                ):
                    raise IntegrityError(
                        "program-integrity lock is not one owned regular file"
                    )

            try:
                require_same_regular_lock()
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                require_same_regular_lock()
            except BaseException:
                handle.close()
                raise
            depths[key] = {"depth": 1, "handle": handle}
            try:
                yield
            finally:
                try:
                    require_same_regular_lock()
                finally:
                    del depths[key]
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    handle.close()

    def _load(self) -> tuple[dict[str, Any], bytes]:
        try:
            raw = self.path.read_bytes()
        except OSError as exc:
            raise IntegrityError("program integrity checkpoint is unreadable") from exc
        if _sha256_bytes(raw) != self.expected_sha256:
            raise IntegrityError(
                "program integrity checkpoint does not match the externally pinned digest"
            )
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError("program integrity checkpoint is invalid JSON") from exc
        normalized = _validate_program_integrity_checkpoint(value)
        if raw != (canonical_json(normalized) + "\n").encode("utf-8"):
            raise IntegrityError("program integrity checkpoint is not canonical JSON")
        for name, anchor in normalized["stores"].items():
            relative_path = anchor.get(
                "path", _RECONCILIATION_LEDGER_FILENAMES[name]
            )
            expected_path = (self.path.parent / relative_path).resolve()
            if self._stores[name].path.resolve() != expected_path:
                raise IntegrityError(
                    f"{name} path does not match the externally pinned checkpoint"
                )
        return normalized, raw

    def assert_store(
        self,
        name: str,
        store: CanonicalHashChainStore,
    ) -> None:
        if name not in self._stores:
            raise ExperimentControlError(f"unknown governed store: {name}")
        if not isinstance(store, CanonicalHashChainStore) or (
            self._stores[name].path.resolve() != store.path.resolve()
        ):
            raise IntegrityError(
                f"{name} does not match the externally checkpointed store"
            )

    def verify(
        self,
        *,
        required_stores: Mapping[str, CanonicalHashChainStore] | None = None,
    ) -> dict[str, Any]:
        if required_stores is not None:
            for name, store in required_stores.items():
                self.assert_store(name, store)
        checkpoint, _ = self._load()
        for name, store in self._stores.items():
            actual, _ = _program_store_snapshot(store)
            expected = {
                key: value
                for key, value in checkpoint["stores"][name].items()
                if key != "path"
            }
            if actual != expected:
                raise IntegrityError(
                    f"{name} does not match the externally pinned current checkpoint"
                )
        return checkpoint

    def expected_head(
        self,
        name: str,
        store: CanonicalHashChainStore,
        *,
        caller_expected_head: str | None = None,
    ) -> str:
        self.assert_store(name, store)
        checkpoint = self.verify()
        checkpoint_head = checkpoint["stores"][name]["expected_head"]
        if (
            caller_expected_head is not None
            and caller_expected_head != checkpoint_head
        ):
            raise CompareAndSwapError(
                f"{name} expected head contradicts the current checkpoint"
            )
        return str(checkpoint_head)

    def successor(
        self,
        *,
        mutated_store: str,
        record_hash: str,
    ) -> ProgramIntegritySuccessor:
        """Validate exactly zero/one target append and build its next snapshot."""

        with self.locked():
            return self._successor_locked(
                mutated_store=mutated_store,
                record_hash=record_hash,
            )

    def _successor_locked(
        self,
        *,
        mutated_store: str,
        record_hash: str,
    ) -> ProgramIntegritySuccessor:
        if mutated_store not in self._stores:
            raise ExperimentControlError(
                f"unknown governed store: {mutated_store}"
            )
        normalized_record_hash = str(record_hash).strip().lower()
        if not _SHA256_RE.fullmatch(normalized_record_hash):
            raise ExperimentControlError(
                "governed mutation record hash must be a SHA-256 digest"
            )
        previous, previous_raw = self._load()
        next_anchors: dict[str, dict[str, Any]] = {}
        mutated = False
        for name, store in self._stores.items():
            actual, records = _program_store_snapshot(store)
            before = previous["stores"][name]
            before_values = {
                key: value for key, value in before.items() if key != "path"
            }
            if name != mutated_store:
                if actual != before_values:
                    raise IntegrityError(
                        f"{name} changed during a different governed mutation"
                    )
            elif actual == before_values:
                if not any(
                    record["record_hash"] == normalized_record_hash
                    for record in records
                ):
                    raise IntegrityError(
                        "idempotent governed mutation record is not checkpointed"
                    )
            else:
                if (
                    actual["expected_length"] != before["expected_length"] + 1
                    or not records
                    or records[-1]["record_hash"] != normalized_record_hash
                    or records[-1]["previous_hash"] != before["expected_head"]
                    or actual["expected_head"] != normalized_record_hash
                ):
                    raise IntegrityError(
                        "governed mutation must append exactly one target record"
                    )
                mutated = True
            actual["path"] = before.get(
                "path",
                Path(
                    os.path.relpath(
                        store.path.resolve(),
                        self.path.parent.resolve(),
                    )
                ).as_posix(),
            )
            next_anchors[name] = actual
        successor = {
            "baseline_manifest_sha256": previous[
                "baseline_manifest_sha256"
            ],
            "runtime_leakage_finding_count": previous[
                "runtime_leakage_finding_count"
            ],
            "stores": next_anchors,
        }
        if "promotion_population" in previous:
            successor["promotion_population"] = previous[
                "promotion_population"
            ]
        normalized = _validate_program_integrity_checkpoint(successor)
        raw = (canonical_json(normalized) + "\n").encode("utf-8")
        if not mutated and raw != previous_raw:
            raise IntegrityError("idempotent checkpoint transition changed bytes")
        return ProgramIntegritySuccessor(
            previous_checkpoint_sha256=self.expected_sha256,
            checkpoint_bytes=raw,
            checkpoint_sha256=_sha256_bytes(raw),
            mutated_store=mutated_store,
            record_hash=normalized_record_hash,
            mutated=mutated,
        )


def _governed_mutation(method):
    """Serialize one supported mutation across the complete program snapshot."""

    @wraps(method)
    def wrapped(*args, **kwargs):
        if "integrity_checkpoint" not in kwargs:
            return method(*args, **kwargs)
        checkpoint = kwargs["integrity_checkpoint"]
        if not isinstance(checkpoint, ProgramIntegrityCheckpoint):
            raise ExperimentControlError(
                "integrity_checkpoint must be a ProgramIntegrityCheckpoint"
            )
        with checkpoint.locked():
            return method(*args, **kwargs)

    return wrapped


class ExperimentLedger:
    """Append legacy evidence or immutable two-stage experiment contracts.

    ``record`` remains available to read and reproduce the already-published
    legacy ledger. New experiments use ``preregister`` before execution and
    ``record_result`` after execution. Plans and results are separate hash-chain
    events, and a result is cryptographically bound to its exact plan record.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        protected_access_path: Path | str | None = None,
    ) -> None:
        self.store = CanonicalHashChainStore(path)
        self.protected_access_store = (
            CanonicalHashChainStore(protected_access_path)
            if protected_access_path is not None
            else None
        )

    def experiments(self) -> tuple[dict[str, Any], ...]:
        """Return legacy one-stage experiment payloads."""

        return tuple(
            dict(record["payload"])
            for record in self.store.verify()
            if record["payload"].get("event") == "experiment"
        )

    def plans(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(record["payload"])
            for record in self.store.verify()
            if record["payload"].get("event") == "experiment_plan"
        )

    def results(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            dict(record["payload"])
            for record in self.store.verify()
            if record["payload"].get("event") == "experiment_result"
        )

    def record(
        self,
        experiment_id: str,
        evidence: Mapping[str, Any],
        *,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        """Retry an already-published legacy record without creating a new one."""

        experiment_id = str(experiment_id).strip()
        if not _SAFE_DIMENSION_RE.fullmatch(experiment_id):
            raise ExperimentControlError(
                "experiment_id must be a non-identifying token"
            )
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
                if existing.get("experiment_id") != experiment_id:
                    continue
                if existing.get("event") == "experiment":
                    if existing != requested:
                        raise ExperimentControlError(
                            f"experiment_id retry does not match original: {experiment_id}"
                        )
                    return record
                raise ExperimentControlError(
                    f"experiment_id is already used by a two-stage event: {experiment_id}"
                )
            raise ExperimentControlError(
                "new one-stage experiment records are forbidden; preregister first"
            )

        return self.store.append_transactional(
            payload,
            expected_head=expected_head,
            locked_check=check,
        )

    @_governed_mutation
    def preregister(
        self,
        experiment_id: str,
        plan: Mapping[str, Any],
        *,
        integrity_checkpoint: ProgramIntegrityCheckpoint,
        expected_head: str | None = None,
    ) -> GovernedMutationReceipt:
        """Append an immutable experiment plan before candidate execution."""

        experiment_id = str(experiment_id).strip()
        if not _SAFE_DIMENSION_RE.fullmatch(experiment_id):
            raise ExperimentControlError(
                "experiment_id must be a non-identifying token"
            )
        _require_nonidentifying_control_text("experiment_id", experiment_id)
        normalized_plan = _normalize_experiment_plan(plan)
        payload = {
            "event": "experiment_plan",
            "experiment_id": experiment_id,
            "plan": normalized_plan,
        }
        integrity_checkpoint.expected_head(
            "experiment_ledger",
            self.store,
            caller_expected_head=expected_head,
        )

        def check(
            records: tuple[dict[str, Any], ...],
            requested: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            for record in records:
                existing = record["payload"]
                if existing.get("experiment_id") != experiment_id:
                    continue
                if existing.get("event") == "experiment_plan":
                    if existing != requested:
                        raise ExperimentControlError(
                            "experiment plan conflicts with immutable "
                            f"preregistration: {experiment_id}"
                        )
                    return record
                raise ExperimentControlError(
                    f"experiment_id is already used by another event: {experiment_id}"
                )
            return None

        checkpoint_head = integrity_checkpoint.expected_head(
            "experiment_ledger",
            self.store,
            caller_expected_head=expected_head,
        )
        record = self.store.append_transactional(
            payload,
            expected_head=checkpoint_head,
            locked_check=check,
        )
        return GovernedMutationReceipt(
            value=dict(record),
            record_hash=record["record_hash"],
            integrity=integrity_checkpoint.successor(
                mutated_store="experiment_ledger",
                record_hash=record["record_hash"],
            ),
        )

    @_governed_mutation
    def record_result(
        self,
        experiment_id: str,
        evidence: Mapping[str, Any],
        *,
        decision: str,
        rationale: str,
        integrity_checkpoint: ProgramIntegrityCheckpoint,
        expected_head: str | None = None,
    ) -> GovernedMutationReceipt:
        """Append one aggregate-only outcome bound to a prior immutable plan."""

        experiment_id = str(experiment_id).strip()
        if not _SAFE_DIMENSION_RE.fullmatch(experiment_id):
            raise ExperimentControlError(
                "experiment_id must be a non-identifying token"
            )
        _require_nonidentifying_control_text("experiment_id", experiment_id)
        normalized_decision = str(decision).strip().casefold()
        if normalized_decision not in _EXPERIMENT_RESULT_DECISIONS:
            raise ExperimentControlError(
                "experiment result decision must be adopt, reject, or rollback"
            )
        normalized_rationale = str(rationale).strip()
        if not _SAFE_DIMENSION_RE.fullmatch(normalized_rationale):
            raise ExperimentControlError(
                "experiment result rationale must be a non-identifying token"
            )
        _require_nonidentifying_control_text("rationale", normalized_rationale)

        integrity_checkpoint.expected_head(
            "experiment_ledger",
            self.store,
            caller_expected_head=expected_head,
        )
        records = self.store.verify()
        matching_plans = [
            record
            for record in records
            if record["payload"].get("event") == "experiment_plan"
            and record["payload"].get("experiment_id") == experiment_id
        ]
        if not matching_plans:
            raise ExperimentControlError(
                f"experiment result requires prior preregistration: {experiment_id}"
            )
        if len(matching_plans) != 1:
            raise IntegrityError(
                f"experiment has multiple preregistrations: {experiment_id}"
            )
        plan_record = matching_plans[0]
        plan_record_hash = plan_record["record_hash"]
        evidence_label = plan_record["payload"]["plan"]["evidence_label"]
        normalized_evidence = _normalize_experiment_result_evidence(
            evidence,
            decision=normalized_decision,
            evidence_label=evidence_label,
        )
        for binding in (
            "evaluator_sha256",
            "input_tree_sha256",
            "runtime_contract_sha256",
            "split_manifest_sha256",
            "truth_sha256",
        ):
            if (
                normalized_evidence[binding]
                != plan_record["payload"]["plan"][binding]
            ):
                raise ExperimentControlError(
                    f"experiment result {binding} does not match its plan"
                )
        if (
            normalized_evidence["expected_record_count"]
            != plan_record["payload"]["plan"]["expected_record_count"]
        ):
            raise ExperimentControlError(
                "experiment result expected_record_count does not match its plan"
            )
        if evidence_label == "protected":
            if self.protected_access_store is None:
                raise ExperimentControlError(
                    "protected results require the protected-access ledger"
                )
            integrity_checkpoint.assert_store(
                "protected_access_ledger",
                self.protected_access_store,
            )
            protected_records = self.protected_access_store.verify()
            ProtectedAccessBudget.validate_store_records(protected_records)
            protected_record_hash = normalized_evidence[
                "protected_access_record_hash"
            ]
            matching_accesses = [
                record
                for record in protected_records
                if record["record_hash"] == protected_record_hash
                and record["payload"].get("event") == "protected_access"
            ]
            if len(matching_accesses) != 1:
                raise ExperimentControlError(
                    "protected result does not match a recorded access"
                )
            access = matching_accesses[0]["payload"]
            if (
                access.get("candidate_sha256")
                != normalized_evidence["candidate_artifact_sha256"]
            ):
                raise ExperimentControlError(
                    "protected access candidate does not match experiment result"
                )
            if access.get("experiment_plan_record_hash") != plan_record_hash:
                raise ExperimentControlError(
                    "protected access does not match experiment plan"
                )
            access_experiment_head = access.get(
                "experiment_ledger_head_sha256"
            )
            access_head_records = [
                record
                for record in records
                if record["record_hash"] == access_experiment_head
            ]
            if (
                len(access_head_records) != 1
                or plan_record["sequence"]
                > access_head_records[0]["sequence"]
            ):
                raise ExperimentControlError(
                    "protected access predates experiment preregistration"
                )
            protected_result = dict(normalized_evidence)
            del protected_result["protected_access_record_hash"]
            if canonical_json(access.get("aggregate_result")) != canonical_json(
                protected_result
            ):
                raise ExperimentControlError(
                    "protected result must exactly match its recorded access aggregates"
                )
        payload = {
            "decision": normalized_decision,
            "event": "experiment_result",
            "evidence": normalized_evidence,
            "experiment_id": experiment_id,
            "plan_record_hash": plan_record_hash,
            "rationale": normalized_rationale,
        }

        def check(
            locked_records: tuple[dict[str, Any], ...],
            requested: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            del requested
            locked_plans = [
                record
                for record in locked_records
                if record["payload"].get("event") == "experiment_plan"
                and record["payload"].get("experiment_id") == experiment_id
            ]
            if len(locked_plans) != 1:
                raise ExperimentControlError(
                    f"experiment result requires one preregistration: {experiment_id}"
                )
            if locked_plans[0]["record_hash"] != plan_record_hash:
                raise IntegrityError(
                    f"experiment plan binding changed: {experiment_id}"
                )
            if any(
                record["payload"].get("event") == "experiment_result"
                and record["payload"].get("experiment_id") == experiment_id
                for record in locked_records
            ):
                raise ExperimentControlError(
                    f"experiment result is already recorded: {experiment_id}"
                )
            return None

        checkpoint_head = integrity_checkpoint.expected_head(
            "experiment_ledger",
            self.store,
            caller_expected_head=expected_head,
        )
        record = self.store.append_transactional(
            payload,
            expected_head=checkpoint_head,
            locked_check=check,
        )
        return GovernedMutationReceipt(
            value=dict(record),
            record_hash=record["record_hash"],
            integrity=integrity_checkpoint.successor(
                mutated_store="experiment_ledger",
                record_hash=record["record_hash"],
            ),
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

    @_governed_mutation
    def taint(
        self,
        group_id: str,
        *,
        reason: str,
        source: str,
        integrity_checkpoint: ProgramIntegrityCheckpoint,
        expected_head: str | None = None,
    ) -> GovernedMutationReceipt:
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
        checkpoint_head = integrity_checkpoint.expected_head(
            "taint_registry",
            self.store,
            caller_expected_head=expected_head,
        )
        record = self.store.append(payload, expected_head=checkpoint_head)
        return GovernedMutationReceipt(
            value=dict(record),
            record_hash=record["record_hash"],
            integrity=integrity_checkpoint.successor(
                mutated_store="taint_registry",
                record_hash=record["record_hash"],
            ),
        )

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

    def __init__(
        self,
        path: Path | str,
        *,
        maximum_accesses: int,
        integrity_checkpoint: ProgramIntegrityCheckpoint | None = None,
    ) -> None:
        if (
            isinstance(maximum_accesses, bool)
            or not isinstance(maximum_accesses, int)
            or maximum_accesses < 1
        ):
            raise ExperimentControlError("maximum_accesses must be positive")
        self.store = CanonicalHashChainStore(path)
        self.maximum_accesses = maximum_accesses
        self.initialization_receipt = self._ensure_configuration(
            integrity_checkpoint=integrity_checkpoint
        )
        self.initialization_integrity = (
            self.initialization_receipt.integrity
            if self.initialization_receipt is not None
            else None
        )

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

    @classmethod
    def validate_store_records(
        cls,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        """Validate configuration and every event in a protected-access ledger."""

        if not records:
            raise IntegrityError(
                "protected access ledger has no immutable configuration"
            )
        configuration = records[0]["payload"]
        if (
            set(configuration) != {"event", "maximum_accesses"}
            or configuration.get("event") != cls.CONFIGURATION_EVENT
            or isinstance(configuration.get("maximum_accesses"), bool)
            or not isinstance(configuration.get("maximum_accesses"), int)
            or configuration["maximum_accesses"] < 1
        ):
            raise IntegrityError(
                "protected access ledger configuration is invalid"
            )
        maximum_accesses = configuration["maximum_accesses"]
        cls._validate_configuration(
            records,
            maximum_accesses=maximum_accesses,
        )

        accesses: list[Mapping[str, Any]] = []
        access_ids: set[str] = set()
        for record in records[1:]:
            payload = record["payload"]
            legacy_keys = {
                "access_id",
                "aggregate_result",
                "candidate_sha256",
                "event",
                "purpose",
            }
            bound_keys = legacy_keys | {
                "experiment_ledger_head_sha256",
                "experiment_plan_record_hash",
            }
            payload_keys = frozenset(payload)
            legacy_baseline = (
                payload_keys == frozenset(legacy_keys)
                and record["sequence"] == 2
                and payload.get("access_id") == "baseline-establishment-v1"
            )
            if (
                payload_keys
                not in {frozenset(legacy_keys), frozenset(bound_keys)}
                or (
                    payload_keys == frozenset(legacy_keys)
                    and not legacy_baseline
                )
                or payload.get("event") != "protected_access"
            ):
                raise IntegrityError(
                    "protected access ledger contains an invalid event"
                )
            access_id = payload["access_id"]
            purpose = payload["purpose"]
            candidate_sha256 = payload["candidate_sha256"]
            if (
                not isinstance(access_id, str)
                or not access_id
                or not _SAFE_DIMENSION_RE.fullmatch(access_id)
                or not isinstance(purpose, str)
                or not purpose
                or _CASE_ID_RE.search(purpose)
                or _PDF_FILENAME_RE.search(purpose)
                or not isinstance(candidate_sha256, str)
                or not _SHA256_RE.fullmatch(candidate_sha256)
                or access_id in access_ids
                or (
                    "experiment_plan_record_hash" in payload
                    and (
                        not isinstance(
                            payload["experiment_plan_record_hash"], str
                        )
                        or not _SHA256_RE.fullmatch(
                            payload["experiment_plan_record_hash"]
                        )
                    )
                )
                or (
                    "experiment_ledger_head_sha256" in payload
                    and (
                        not isinstance(
                            payload["experiment_ledger_head_sha256"], str
                        )
                        or not _SHA256_RE.fullmatch(
                            payload["experiment_ledger_head_sha256"]
                        )
                    )
                )
            ):
                raise IntegrityError(
                    "protected access ledger contains invalid access metadata"
                )
            try:
                require_aggregate_only(payload["aggregate_result"])
            except (ExperimentControlError, TypeError) as exc:
                raise IntegrityError(
                    "protected access ledger contains invalid aggregate evidence"
                ) from exc
            access_ids.add(access_id)
            accesses.append(record)
        if len(accesses) > maximum_accesses:
            raise IntegrityError("protected access ledger exceeds its budget")
        return maximum_accesses

    def _ensure_configuration(
        self,
        *,
        integrity_checkpoint: ProgramIntegrityCheckpoint | None,
    ) -> GovernedMutationReceipt | None:
        payload = {
            "event": self.CONFIGURATION_EVENT,
            "maximum_accesses": self.maximum_accesses,
        }
        existing_records = self.store.verify()
        existing = self._validate_configuration(
            existing_records,
            maximum_accesses=self.maximum_accesses,
        )
        if existing is not None:
            return None
        if integrity_checkpoint is None:
            raise ExperimentControlError(
                "initializing a protected access budget requires "
                "the externally pinned current checkpoint"
            )
        with integrity_checkpoint.locked():
            checkpoint_head = integrity_checkpoint.expected_head(
                "protected_access_ledger",
                self.store,
            )

            def check(
                records: tuple[dict[str, Any], ...],
                requested: Mapping[str, Any],
            ) -> Mapping[str, Any] | None:
                del requested
                return self._validate_configuration(
                    records,
                    maximum_accesses=self.maximum_accesses,
                )

            record = self.store.append_transactional(
                payload,
                expected_head=checkpoint_head,
                locked_check=check,
            )
            return GovernedMutationReceipt(
                value=dict(record),
                record_hash=record["record_hash"],
                integrity=integrity_checkpoint.successor(
                    mutated_store="protected_access_ledger",
                    record_hash=record["record_hash"],
                ),
            )

    def accesses(self) -> tuple[dict[str, Any], ...]:
        records = self.store.verify()
        configured_maximum = self.validate_store_records(records)
        if configured_maximum != self.maximum_accesses:
            raise IntegrityError(
                "protected access budget configuration is immutable"
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

    @_governed_mutation
    def record_access(
        self,
        access_id: str,
        *,
        candidate_sha256: str,
        aggregate_result: Mapping[str, Any],
        experiment_ledger: ExperimentLedger,
        experiment_plan_record_hash: str,
        integrity_checkpoint: ProgramIntegrityCheckpoint,
        purpose: str,
        expected_head: str | None = None,
    ) -> GovernedMutationReceipt:
        access_id = str(access_id).strip()
        candidate_sha256 = str(candidate_sha256).strip().lower()
        experiment_plan_record_hash = str(
            experiment_plan_record_hash
        ).strip().lower()
        purpose = str(purpose).strip()
        if (
            not access_id
            or not _SAFE_DIMENSION_RE.fullmatch(access_id)
            or not purpose
        ):
            raise ExperimentControlError("access_id and purpose are required")
        _require_nonidentifying_control_text("access_id", access_id)
        _require_nonidentifying_control_text("purpose", purpose)
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
            raise ExperimentControlError("candidate_sha256 must be a SHA-256 hex digest")
        if not _SHA256_RE.fullmatch(experiment_plan_record_hash):
            raise ExperimentControlError(
                "experiment_plan_record_hash must be a SHA-256 hex digest"
            )
        if not isinstance(experiment_ledger, ExperimentLedger):
            raise ExperimentControlError(
                "experiment_ledger must be an ExperimentLedger"
            )
        integrity_checkpoint.assert_store(
            "experiment_ledger",
            experiment_ledger.store,
        )
        integrity_checkpoint.expected_head(
            "protected_access_ledger",
            self.store,
            caller_expected_head=expected_head,
        )
        experiment_records = experiment_ledger.store.verify()
        matching_plans = [
            record
            for record in experiment_records
            if record["record_hash"] == experiment_plan_record_hash
            and record["payload"].get("event") == "experiment_plan"
        ]
        if len(matching_plans) != 1:
            raise ExperimentControlError(
                "protected access requires an existing experiment plan"
            )
        experiment_ledger_head_sha256 = experiment_ledger.store._head(
            experiment_records
        )
        require_aggregate_only(aggregate_result)
        requested = {
            "event": "protected_access",
            "access_id": access_id,
            "candidate_sha256": candidate_sha256,
            "aggregate_result": dict(aggregate_result),
            "experiment_ledger_head_sha256": experiment_ledger_head_sha256,
            "experiment_plan_record_hash": experiment_plan_record_hash,
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
                    if canonical_json(existing) != canonical_json(
                        normalized_request
                    ):
                        raise ExperimentControlError(
                            f"access_id retry does not match original request: {access_id}"
                        )
                    return record
            if len(accesses) >= self.maximum_accesses:
                raise BudgetExhaustedError("protected access budget is exhausted")
            return None

        checkpoint_head = integrity_checkpoint.expected_head(
            "protected_access_ledger",
            self.store,
            caller_expected_head=expected_head,
        )
        record = self.store.append_transactional(
            requested,
            expected_head=checkpoint_head,
            locked_check=check,
        )
        return GovernedMutationReceipt(
            value=dict(record["payload"]),
            record_hash=record["record_hash"],
            integrity=integrity_checkpoint.successor(
                mutated_store="protected_access_ledger",
                record_hash=record["record_hash"],
            ),
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
_PROMOTION_MILESTONE_THRESHOLDS = {
    "milestone-136": 136.0,
    "milestone-142": 142.0,
    "milestone-146": 146.0,
    "milestone-148": 148.0,
}


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

    def latest_passing(
        self,
        *,
        integrity_checkpoint: ProgramIntegrityCheckpoint,
    ) -> dict[str, Any] | None:
        """Return only state finalized by the externally current checkpoint."""

        if not isinstance(integrity_checkpoint, ProgramIntegrityCheckpoint):
            raise ExperimentControlError(
                "integrity_checkpoint must be a ProgramIntegrityCheckpoint"
            )
        with integrity_checkpoint.locked():
            integrity_checkpoint.verify(
                required_stores={"candidate_state_ledger": self.store}
            )
            for assessment in reversed(self.assessments()):
                if assessment.get("decision") == "PASSED":
                    return assessment
        return None

    @_governed_mutation
    def assess(
        self,
        assessment_id: str,
        *,
        candidate_id: str,
        candidate_sha256: str,
        decision: str,
        aggregate_evidence: Mapping[str, Any],
        integrity_checkpoint: ProgramIntegrityCheckpoint,
        expected_head: str | None = None,
    ) -> GovernedMutationReceipt:
        if str(decision).strip().upper() == "PASSED":
            raise ExperimentControlError(
                "PASSED can only be persisted by CandidatePromotionGate"
            )
        record = self._persist_assessment_record(
            assessment_id,
            candidate_id=candidate_id,
            candidate_sha256=candidate_sha256,
            decision=decision,
            aggregate_evidence=aggregate_evidence,
            integrity_checkpoint=integrity_checkpoint,
            expected_head=expected_head,
            authority=None,
        )
        return GovernedMutationReceipt(
            value=dict(record["payload"]),
            record_hash=record["record_hash"],
            integrity=integrity_checkpoint.successor(
                mutated_store="candidate_state_ledger",
                record_hash=record["record_hash"],
            ),
        )

    def _persist_assessment_record(
        self,
        assessment_id: str,
        *,
        candidate_id: str,
        candidate_sha256: str,
        decision: str,
        aggregate_evidence: Mapping[str, Any],
        integrity_checkpoint: ProgramIntegrityCheckpoint,
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
        integrity_checkpoint.expected_head(
            "candidate_state_ledger",
            self.store,
            caller_expected_head=expected_head,
        )

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
                    if canonical_json(existing) != canonical_json(requested):
                        raise ExperimentControlError(
                            f"assessment retry does not match original: {assessment_id}"
                        )
                    return record
            return None

        checkpoint_head = integrity_checkpoint.expected_head(
            "candidate_state_ledger",
            self.store,
            caller_expected_head=expected_head,
        )
        return self.store.append_transactional(
            payload,
            expected_head=checkpoint_head,
            locked_check=check,
        )


class CandidatePromotionGate:
    """Promote only a protected, plan-bound, milestone-qualified result."""

    def __init__(
        self,
        state: CandidateStateStore,
        *,
        experiment_ledger: ExperimentLedger,
        taint_registry: TaintRegistry,
    ) -> None:
        self.state = state
        self.experiment_ledger = experiment_ledger
        self.taint_registry = taint_registry

    def _verified_result(
        self,
        record_hash: str,
    ) -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        str,
    ]:
        normalized_hash = str(record_hash).strip().lower()
        if not _SHA256_RE.fullmatch(normalized_hash):
            raise ExperimentControlError(
                "experiment_result_record_hash must be a SHA-256 hex digest"
            )
        records = self.experiment_ledger.store.verify()
        matches = [
            record for record in records if record["record_hash"] == normalized_hash
        ]
        if len(matches) != 1:
            raise ExperimentControlError(
                "promotion requires one bound experiment-result record"
            )
        result_record = matches[0]
        result = result_record["payload"]
        if set(result) != {
            "decision",
            "event",
            "evidence",
            "experiment_id",
            "plan_record_hash",
            "rationale",
        } or result.get("event") != "experiment_result":
            raise ExperimentControlError(
                "promotion record is not a canonical experiment result"
            )
        experiment_id = result.get("experiment_id")
        rationale = result.get("rationale")
        decision = result.get("decision")
        if (
            not isinstance(experiment_id, str)
            or not _SAFE_DIMENSION_RE.fullmatch(experiment_id)
            or not isinstance(rationale, str)
            or not _SAFE_DIMENSION_RE.fullmatch(rationale)
            or decision not in _EXPERIMENT_RESULT_DECISIONS
        ):
            raise ExperimentControlError(
                "promotion experiment-result metadata is invalid"
            )
        _require_nonidentifying_control_text("experiment_id", experiment_id)
        _require_nonidentifying_control_text("rationale", rationale)

        plan_hash = result.get("plan_record_hash")
        if not isinstance(plan_hash, str) or not _SHA256_RE.fullmatch(plan_hash):
            raise ExperimentControlError(
                "promotion experiment result has no valid plan binding"
            )
        experiment_plans = [
            record
            for record in records
            if record["payload"].get("event") == "experiment_plan"
            and record["payload"].get("experiment_id") == experiment_id
        ]
        experiment_results = [
            record
            for record in records
            if record["payload"].get("event") == "experiment_result"
            and record["payload"].get("experiment_id") == experiment_id
        ]
        if (
            len(experiment_plans) != 1
            or len(experiment_results) != 1
            or experiment_results[0]["record_hash"] != result_record["record_hash"]
        ):
            raise ExperimentControlError(
                "promotion requires exactly one plan and one result per experiment"
            )
        plan_record = experiment_plans[0]
        if (
            plan_record["record_hash"] != plan_hash
            or plan_record["sequence"] >= result_record["sequence"]
        ):
            raise ExperimentControlError(
                "promotion result does not bind one earlier experiment plan"
            )
        plan_payload = plan_record["payload"]
        if set(plan_payload) != {"event", "experiment_id", "plan"} or (
            plan_payload.get("event") != "experiment_plan"
            or plan_payload.get("experiment_id") != experiment_id
        ):
            raise ExperimentControlError(
                "promotion result and experiment plan do not match"
            )
        normalized_plan = _normalize_experiment_plan(plan_payload["plan"])
        if canonical_json(normalized_plan) != canonical_json(plan_payload["plan"]):
            raise IntegrityError("promotion experiment plan is not canonical")
        evidence_label = normalized_plan["evidence_label"]
        normalized_evidence = _normalize_experiment_result_evidence(
            result["evidence"],
            decision=decision,
            evidence_label=evidence_label,
        )
        if canonical_json(normalized_evidence) != canonical_json(
            result["evidence"]
        ):
            raise IntegrityError("promotion experiment result is not canonical")
        return (
            result_record,
            plan_record,
            normalized_evidence,
            normalized_plan,
            self.experiment_ledger.store._head(records),
        )

    def _verified_protected_access(
        self,
        evidence: Mapping[str, Any],
        *,
        candidate_sha256: str,
        plan_record: Mapping[str, Any],
        result_record: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        store = self.experiment_ledger.protected_access_store
        if store is None:
            raise ExperimentControlError(
                "promotion requires the protected-access ledger"
            )
        records = store.verify()
        ProtectedAccessBudget.validate_store_records(records)
        access_hash = evidence.get("protected_access_record_hash")
        if not isinstance(access_hash, str) or not _SHA256_RE.fullmatch(access_hash):
            raise ExperimentControlError(
                "promotion result has no protected-access binding"
            )
        matches = [
            record
            for record in records
            if record["record_hash"] == access_hash
            and record["payload"].get("event") == "protected_access"
        ]
        if len(matches) != 1:
            raise ExperimentControlError(
                "promotion result does not bind one protected access"
            )
        access_record = matches[0]
        access = access_record["payload"]
        if access.get("candidate_sha256") != candidate_sha256:
            raise ExperimentControlError(
                "protected access candidate does not match promotion candidate"
            )
        if (
            access.get("experiment_plan_record_hash")
            != plan_record["record_hash"]
        ):
            raise ExperimentControlError(
                "protected access does not match promotion experiment plan"
            )
        experiment_records = self.experiment_ledger.store.verify()
        access_head_matches = [
            record
            for record in experiment_records
            if record["record_hash"]
            == access.get("experiment_ledger_head_sha256")
        ]
        if (
            len(access_head_matches) != 1
            or plan_record["sequence"] > access_head_matches[0]["sequence"]
            or access_head_matches[0]["sequence"] >= result_record["sequence"]
        ):
            raise ExperimentControlError(
                "protected access must follow preregistration and precede the result"
            )
        protected_aggregate = dict(evidence)
        del protected_aggregate["protected_access_record_hash"]
        if canonical_json(access.get("aggregate_result")) != canonical_json(
            protected_aggregate
        ):
            raise ExperimentControlError(
                "promotion result does not exactly match protected aggregates"
            )
        return access_record, store._head(records)

    @_governed_mutation
    def evaluate_and_record(
        self,
        assessment_id: str,
        *,
        candidate_id: str,
        candidate_sha256: str,
        experiment_result_record_hash: str,
        integrity_checkpoint: ProgramIntegrityCheckpoint,
        expected_head: str | None = None,
    ) -> GovernedMutationReceipt:
        """Derive every promotion gate from one immutable protected result."""

        protected_store = self.experiment_ledger.protected_access_store
        if protected_store is None:
            raise ExperimentControlError(
                "promotion requires the protected-access ledger"
            )
        checkpoint_value = integrity_checkpoint.verify(
            required_stores={
                "candidate_state_ledger": self.state.store,
                "experiment_ledger": self.experiment_ledger.store,
                "protected_access_ledger": protected_store,
                "taint_registry": self.taint_registry.store,
            }
        )
        checkpoint_population_raw = checkpoint_value.get(
            "promotion_population"
        )
        if checkpoint_population_raw is None:
            raise IntegrityError(
                "promotion requires an externally checkpointed population tuple"
            )
        checkpoint_population = _normalize_promotion_population(
            checkpoint_population_raw
        )
        checkpoint_runtime_leakage_finding_count = checkpoint_value[
            "runtime_leakage_finding_count"
        ]
        taint_head = checkpoint_value["stores"]["taint_registry"][
            "expected_head"
        ]
        candidate_sha256 = str(candidate_sha256).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", candidate_sha256):
            raise ExperimentControlError("candidate_sha256 must be a SHA-256 hex digest")
        (
            result_record,
            plan_record,
            result_evidence,
            result_plan,
            experiment_head,
        ) = self._verified_result(experiment_result_record_hash)
        result_payload = result_record["payload"]
        if result_evidence["candidate_artifact_sha256"] != candidate_sha256:
            raise ExperimentControlError(
                "experiment result candidate does not match promotion candidate"
            )
        evidence_label = result_plan["evidence_label"]
        if evidence_label != "protected":
            raise ExperimentControlError(
                "candidate promotion requires protected evidence"
            )
        access_record, protected_head = self._verified_protected_access(
            result_evidence,
            candidate_sha256=candidate_sha256,
            plan_record=plan_record,
            result_record=result_record,
        )
        access_purpose = access_record["payload"]["purpose"]
        milestone_total_score = _PROMOTION_MILESTONE_THRESHOLDS.get(
            access_purpose
        )
        if milestone_total_score is None:
            raise ExperimentControlError(
                "protected access purpose is not a promotion milestone"
            )

        state_records = self.state.store.verify()
        state_head = self.state.store._head(state_records)
        if expected_head is not None and expected_head != state_head:
            raise CompareAndSwapError(
                f"candidate-state CAS failed: expected {expected_head}, got {state_head}"
            )
        existing = [
            record
            for record in state_records
            if record["payload"].get("event") == "candidate_assessment"
            and record["payload"].get("assessment_id") == str(assessment_id).strip()
        ]
        if existing:
            if len(existing) != 1:
                raise IntegrityError(
                    f"candidate assessment is duplicated: {assessment_id}"
                )
            prior_record = existing[0]
            prior = prior_record["payload"]
            prior_evidence = prior.get("aggregate_evidence", {})
            if (
                prior.get("candidate_id") != str(candidate_id).strip()
                or prior.get("candidate_sha256") != candidate_sha256
                or prior_evidence.get("experiment_result_record_hash")
                != result_record["record_hash"]
            ):
                raise ExperimentControlError(
                    f"assessment retry does not match original: {assessment_id}"
                )
            if prior.get("decision") == "PASSED":
                prior_gates = prior_evidence.get("gate_results")
                prior_population = _promotion_population_from(
                    prior_evidence
                )
                if (
                    prior_evidence.get("promotion_gate_verified") is not True
                    or prior_evidence.get("hard_gate_failure_count") != 0
                    or not isinstance(prior_gates, Mapping)
                    or not prior_gates
                    or any(value is not True for value in prior_gates.values())
                    or prior_population != checkpoint_population
                    or checkpoint_runtime_leakage_finding_count != 0
                ):
                    raise IntegrityError(
                        "idempotent PASSED receipt is not valid under "
                        "the current checkpoint"
                    )
            successor = integrity_checkpoint.successor(
                mutated_store="candidate_state_ledger",
                record_hash=prior_record["record_hash"],
            )
            return GovernedMutationReceipt(
                value=dict(prior),
                record_hash=prior_record["record_hash"],
                integrity=successor,
            )

        passing = [
            record["payload"]
            for record in state_records
            if record["payload"].get("event") == "candidate_assessment"
            and record["payload"].get("decision") == "PASSED"
        ]
        latest_passing = passing[-1] if passing else None
        latest_score: float | None = None
        latest_milestone: float | None = None
        latest_population: dict[str, Any] | None = None
        milestone_thresholds = tuple(
            sorted(_PROMOTION_MILESTONE_THRESHOLDS.values())
        )
        if latest_passing is not None:
            try:
                latest_evidence = latest_passing["aggregate_evidence"]
                raw_latest_score = latest_evidence["metrics"]["total_score"]
            except (KeyError, TypeError) as exc:
                raise IntegrityError(
                    "latest passing candidate lacks its bound total score"
                ) from exc
            if (
                isinstance(raw_latest_score, bool)
                or not isinstance(raw_latest_score, (int, float))
                or not 0 <= raw_latest_score <= 150
            ):
                raise IntegrityError(
                    "latest passing candidate total score is invalid"
                )
            latest_score = float(raw_latest_score)
            latest_population = _promotion_population_from(
                latest_evidence
            )
            if (
                latest_evidence.get("experiment_result_record_hash")
                is not None
                and latest_population is None
            ):
                raise IntegrityError(
                    "latest governed passing candidate lacks its population tuple"
                )
            if (
                latest_population is not None
                and latest_population != checkpoint_population
            ):
                raise IntegrityError(
                    "latest passing candidate population contradicts "
                    "the current checkpoint"
                )
            raw_latest_milestone = latest_evidence.get(
                "milestone_total_score"
            )
            if raw_latest_milestone is None:
                if (
                    latest_score >= milestone_thresholds[0]
                    or latest_evidence.get("experiment_result_record_hash")
                    is not None
                ):
                    raise IntegrityError(
                        "latest passing candidate lacks its achieved milestone"
                    )
            elif (
                isinstance(raw_latest_milestone, bool)
                or not isinstance(raw_latest_milestone, (int, float))
                or float(raw_latest_milestone) not in milestone_thresholds
                or float(raw_latest_milestone) > latest_score
            ):
                raise IntegrityError(
                    "latest passing candidate milestone is invalid"
                )
            else:
                latest_milestone = float(raw_latest_milestone)
        baseline_ok = (
            latest_passing is not None
            and latest_passing.get("candidate_sha256")
            == result_evidence["baseline_artifact_sha256"]
        )
        passing_candidate_sha256 = {
            record.get("candidate_sha256") for record in passing
        }
        candidate_changed = (
            candidate_sha256
            != result_evidence["baseline_artifact_sha256"]
        )
        candidate_digest_new = (
            candidate_sha256 not in passing_candidate_sha256
        )
        checks = result_evidence["checks"]
        metrics = result_evidence["metrics"]
        regressions = result_evidence["regression_counts"]
        next_milestone = next(
            (
                threshold
                for threshold in milestone_thresholds
                if latest_milestone is None
                or threshold > latest_milestone
            ),
            None,
        )
        result_population = _promotion_population_from(result_evidence)
        plan_population = _promotion_population_from(result_plan)
        comparison_population = (
            latest_population
            if latest_population is not None
            else checkpoint_population
        )
        population_ok = (
            result_population is not None
            and plan_population is not None
            and result_population == plan_population
            and result_population == checkpoint_population
            and result_population == comparison_population
            and result_population["expected_record_count"]
            == metrics["record_count"]
        )

        gate_results = {
            "baseline_verified": baseline_ok,
            "candidate_changed_verified": candidate_changed,
            "candidate_digest_unique_verified": candidate_digest_new,
            "checkpoint_runtime_leakage_clean": (
                checkpoint_runtime_leakage_finding_count == 0
            ),
            "evidence_class_verified": evidence_label == "protected",
            "experiment_result_adopted": (
                result_payload["decision"] == "adopt"
            ),
            "population_binding_verified": population_ok,
            "protected_aggregate_verified": True,
            "milestone_sequence_verified": (
                next_milestone is not None
                and milestone_total_score == next_milestone
            ),
            "milestone_score_reached": (
                metrics["total_score"] >= milestone_total_score
            ),
            "score_improved": (
                latest_score is not None
                and metrics["total_score"] > latest_score
            ),
            "decision_freeze_verified": checks[
                "decision_freeze_verified"
            ],
            "runtime_limits_verified": checks["runtime_limits_verified"],
            "deterministic": checks["deterministic"],
            "fold_consistent": checks["fold_consistent"],
            "no_false_approvals": (
                metrics["catastrophic_false_approvals"] == 0
            ),
            "no_invalid_records": metrics["invalid_records"] == 0,
            "no_leakage": checks["runtime_leakage_clean"],
            "no_missing_records": metrics["missing_records"] == 0,
            "regressions_cleared": not any(regressions.values()),
        }
        decision = "PASSED" if all(gate_results.values()) else "BLOCKED"

        evidence = dict(result_evidence)
        evidence.update(
            {
                "baseline_verified": baseline_ok,
                "experiment_ledger_head_sha256": experiment_head,
                "experiment_plan_record_hash": plan_record["record_hash"],
                "experiment_result_record_hash": result_record["record_hash"],
                "gate_results": gate_results,
                "hard_gate_failure_count": sum(
                    not result for result in gate_results.values()
                ),
                "checkpoint_runtime_leakage_finding_count": (
                    checkpoint_runtime_leakage_finding_count
                ),
                "integrity_checkpoint_sha256": (
                    integrity_checkpoint.expected_sha256
                ),
                "milestone_total_score": milestone_total_score,
                "promotion_gate_verified": True,
                "protected_access_ledger_head_sha256": protected_head,
                "taint_registry_head_sha256": taint_head,
            }
        )
        require_aggregate_only(evidence)
        assessment_record = self.state._persist_assessment_record(
            assessment_id,
            candidate_id=candidate_id,
            candidate_sha256=candidate_sha256,
            decision=decision,
            aggregate_evidence=evidence,
            integrity_checkpoint=integrity_checkpoint,
            expected_head=state_head,
            authority=_PROMOTION_GATE_AUTHORITY,
        )
        successor = integrity_checkpoint.successor(
            mutated_store="candidate_state_ledger",
            record_hash=assessment_record["record_hash"],
        )
        return GovernedMutationReceipt(
            value=dict(assessment_record["payload"]),
            record_hash=assessment_record["record_hash"],
            integrity=successor,
        )
