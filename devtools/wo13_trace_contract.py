"""Strict, label-blind contract for the external WO13 runtime trace.

The per-case rows produced under this contract are development evidence.  They
must stay outside the repository.  Only aggregate, K-safe atlas output may be
committed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping


TRACE_SCHEMA_VERSION = "mib-wo13-truth-blind-trace/v1"
CAPTURE_MODES = frozenset(
    {"authoritative_production", "test"}
)
TRACE_DIMENSIONS = (
    "provenance_route",
    "applicant_linking_state",
    "evidence_conflict",
    "ocr_recovery_path",
    "policy_trace",
)
TRACE_CATEGORY_ENUMS = {
    "provenance_route": frozenset(
        {
            "visible_ocr",
            "authoritative_source",
            "mixed_visible_sources",
            "no_accepted_provenance",
            "unknown",
        }
    ),
    "applicant_linking_state": frozenset(
        {
            "linked_unique",
            "linked_ambiguous",
            "unlinked",
            "authoritative_scope",
            "unknown",
        }
    ),
    "evidence_conflict": frozenset(
        {
            "none",
            "field_conflict",
            "identity_conflict",
            "authority_conflict",
            "multiple_conflicts",
            "unknown",
        }
    ),
    "ocr_recovery_path": frozenset(
        {
            "primary",
            "consensus_retry",
            "sparse_intake_retry",
            "orientation_retry",
            "risk_flag_retry",
            "targeted_rapidocr",
            "multiple_recovery_paths",
            "no_visible_ocr",
            "unknown",
        }
    ),
    "policy_trace": frozenset(
        {
            "binding_authority",
            "deterministic_policy",
            "revalidated_policy",
            "recovery_review",
            "recovery_denial",
            "needs_review_conflict",
            "unknown",
        }
    ),
}

_CASE_ID_RE = re.compile(r"^MIB-[0-9]{6}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROW_KEYS = frozenset({"case_id", *TRACE_DIMENSIONS, "runtime_seconds"})
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "capture_mode",
        "source_revision_sha",
        "checkout_revision_sha",
        "capture_source_revision_sha",
        "input_tree_sha256",
        "processing_snapshot_input_tree_sha256",
        "layout_manifest_sha256",
        "dataset_archive_sha256",
        "runtime_contract_sha256",
        "frozen_baseline_manifest_sha256",
        "baseline_predictions_sha256",
        "runtime_graph_sha256",
        "trace_tool_sha256",
        "container_graph_sha256",
        "source_snapshot_sha256",
        "authority_manifest_sha256",
        "runtime_identity_sha256",
        "dependency_identity_sha256",
        "python_executable_sha256",
        "predictions_sha256",
        "production_tree_verified",
        "runtime_contract_verified",
        "runtime_environment_verified",
        "runtime_interface_verified",
        "container_limits_verified",
        "processing_snapshot_verified",
        "stability_checks",
        "case_count",
        "attempted",
        "answered",
        "omitted",
        "max_workers",
        "retry_missing_attempts",
        "retry_passes_used",
        "batch_wall_seconds",
        "rows",
    }
)


class TraceContractError(ValueError):
    """The trace is incomplete, unbound, or outside the allowlisted schema."""


def sha256_path(path: Path) -> str:
    """Return the lowercase SHA-256 of one file without interpreting it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for evidence hashing and persistence."""

    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TraceContractError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceContractError(f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceContractError(f"{label} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise TraceContractError(f"{label} must be finite and non-negative")
    return normalized


def _digest(value: Any, *, label: str) -> str:
    normalized = str(value).casefold()
    if not _SHA256_RE.fullmatch(normalized):
        raise TraceContractError(f"{label} must be a lowercase SHA-256")
    return normalized


def validate_trace_row(value: Any) -> dict[str, Any]:
    """Validate and normalize one strictly allowlisted trace row."""

    if not isinstance(value, Mapping) or set(value) != _ROW_KEYS:
        raise TraceContractError(
            f"trace rows must contain exactly {sorted(_ROW_KEYS)}"
        )
    case_id = value["case_id"]
    if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
        raise TraceContractError("trace case_id must match MIB-NNNNNN")
    row: dict[str, Any] = {"case_id": case_id}
    for dimension in TRACE_DIMENSIONS:
        category = value[dimension]
        if (
            not isinstance(category, str)
            or category not in TRACE_CATEGORY_ENUMS[dimension]
        ):
            raise TraceContractError(
                f"{dimension} must be one of "
                f"{sorted(TRACE_CATEGORY_ENUMS[dimension])}"
            )
        row[dimension] = category
    row["runtime_seconds"] = _finite_nonnegative(
        value["runtime_seconds"],
        label="runtime_seconds",
    )
    return row


def validate_trace_capture(value: Any) -> dict[str, Any]:
    """Fail closed unless the complete capture obeys the exact v1 contract."""

    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_KEYS:
        raise TraceContractError(
            "trace capture must use the exact versioned top-level schema"
        )
    if value["schema_version"] != TRACE_SCHEMA_VERSION:
        raise TraceContractError("trace capture schema_version is unsupported")
    capture_mode = value["capture_mode"]
    if not isinstance(capture_mode, str) or capture_mode not in CAPTURE_MODES:
        raise TraceContractError(
            f"capture_mode must be one of {sorted(CAPTURE_MODES)}"
        )

    revision = str(value["source_revision_sha"]).casefold()
    if not _REVISION_RE.fullmatch(revision):
        raise TraceContractError(
            "source_revision_sha must be a lowercase 40-character Git SHA"
        )
    normalized: dict[str, Any] = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "capture_mode": capture_mode,
        "source_revision_sha": revision,
    }
    checkout_revision = str(value["checkout_revision_sha"]).casefold()
    if not _REVISION_RE.fullmatch(checkout_revision):
        raise TraceContractError(
            "checkout_revision_sha must be a lowercase 40-character Git SHA"
        )
    normalized["checkout_revision_sha"] = checkout_revision
    capture_source_revision = str(
        value["capture_source_revision_sha"]
    ).casefold()
    if not _REVISION_RE.fullmatch(capture_source_revision):
        raise TraceContractError(
            "capture_source_revision_sha must be a lowercase Git SHA"
        )
    normalized["capture_source_revision_sha"] = capture_source_revision
    for label in (
        "input_tree_sha256",
        "processing_snapshot_input_tree_sha256",
        "layout_manifest_sha256",
        "dataset_archive_sha256",
        "runtime_contract_sha256",
        "frozen_baseline_manifest_sha256",
        "baseline_predictions_sha256",
        "runtime_graph_sha256",
        "trace_tool_sha256",
        "container_graph_sha256",
        "source_snapshot_sha256",
        "authority_manifest_sha256",
        "runtime_identity_sha256",
        "dependency_identity_sha256",
        "python_executable_sha256",
        "predictions_sha256",
    ):
        normalized[label] = _digest(value[label], label=label)
    verification_keys = (
        "production_tree_verified",
        "runtime_contract_verified",
        "runtime_environment_verified",
        "runtime_interface_verified",
        "container_limits_verified",
        "processing_snapshot_verified",
    )
    verification = {
        key: value[key]
        for key in verification_keys
    }
    if any(not isinstance(flag, bool) for flag in verification.values()):
        raise TraceContractError("capture verification flags must be boolean")
    stability_checks = value["stability_checks"]
    stability_keys = {
        "input_tree_unchanged",
        "layout_manifest_unchanged",
        "dataset_archive_unchanged",
        "runtime_contract_unchanged",
        "frozen_baseline_manifest_unchanged",
        "baseline_predictions_unchanged",
        "runtime_graph_unchanged",
        "trace_tool_unchanged",
        "container_graph_unchanged",
        "source_snapshot_unchanged",
        "authority_manifest_unchanged",
        "runtime_identity_unchanged",
        "checkout_revision_unchanged",
        "capture_source_revision_unchanged",
        "production_tree_unchanged",
    }
    if (
        not isinstance(stability_checks, Mapping)
        or set(stability_checks) != stability_keys
        or any(
            not isinstance(stability_checks[key], bool)
            for key in stability_keys
        )
    ):
        raise TraceContractError(
            "stability_checks must use the exact boolean schema"
        )
    if capture_mode == "authoritative_production":
        if not all(verification.values()):
            raise TraceContractError(
                "authoritative capture requires every verification gate"
            )
        if not all(stability_checks.values()):
            raise TraceContractError(
                "authoritative capture requires every stability check"
            )
        if (
            normalized["predictions_sha256"]
            != normalized["baseline_predictions_sha256"]
        ):
            raise TraceContractError(
                "authoritative predictions must match baseline bytes"
            )
        if (
            normalized["processing_snapshot_input_tree_sha256"]
            != normalized["input_tree_sha256"]
        ):
            raise TraceContractError(
                "authoritative processing snapshot must match input authority"
            )
        if checkout_revision != capture_source_revision:
            raise TraceContractError(
                "authoritative checkout must equal capture source revision"
            )
    elif any(verification.values()):
        raise TraceContractError(
            "test capture cannot assert authoritative verification gates"
        )
    normalized.update(verification)
    normalized["stability_checks"] = {
        key: stability_checks[key] for key in sorted(stability_keys)
    }

    case_count = _positive_int(value["case_count"], label="case_count")
    attempted = _nonnegative_int(value["attempted"], label="attempted")
    answered = _nonnegative_int(value["answered"], label="answered")
    omitted = _nonnegative_int(value["omitted"], label="omitted")
    max_workers = _positive_int(value["max_workers"], label="max_workers")
    if max_workers > 4:
        raise TraceContractError("max_workers must not exceed four")
    if capture_mode == "authoritative_production" and max_workers != 4:
        raise TraceContractError(
            "authoritative runtime capture requires exactly four workers"
        )
    retry_missing_attempts = _nonnegative_int(
        value["retry_missing_attempts"],
        label="retry_missing_attempts",
    )
    if retry_missing_attempts > 3:
        raise TraceContractError("retry_missing_attempts must not exceed three")
    retry_passes_used = _nonnegative_int(
        value["retry_passes_used"],
        label="retry_passes_used",
    )
    if retry_passes_used > retry_missing_attempts:
        raise TraceContractError(
            "retry_passes_used cannot exceed retry_missing_attempts"
        )
    if attempted != answered + omitted:
        raise TraceContractError("attempted must equal answered plus omitted")
    if case_count != answered:
        raise TraceContractError("case_count must equal answered")
    if omitted:
        raise TraceContractError("a finalized trace capture cannot omit cases")

    raw_rows = value["rows"]
    if not isinstance(raw_rows, list):
        raise TraceContractError("trace rows must be an array")
    rows = [validate_trace_row(row) for row in raw_rows]
    if len(rows) != case_count:
        raise TraceContractError("case_count does not match trace rows")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise TraceContractError("trace rows contain duplicate case_id values")
    if case_ids != sorted(case_ids):
        raise TraceContractError("trace rows must be sorted by case_id")

    normalized.update(
        {
            "case_count": case_count,
            "attempted": attempted,
            "answered": answered,
            "omitted": omitted,
            "max_workers": max_workers,
            "retry_missing_attempts": retry_missing_attempts,
            "retry_passes_used": retry_passes_used,
            "batch_wall_seconds": _finite_nonnegative(
                value["batch_wall_seconds"],
                label="batch_wall_seconds",
            ),
            "rows": rows,
        }
    )
    return normalized


def atomic_write_trace(path: Path, value: Any) -> None:
    """Validate and atomically finalize one external trace file."""

    normalized = validate_trace_capture(value)
    destination = Path(path)
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"trace output directory does not exist: {destination.parent}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(canonical_json_bytes(normalized))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
