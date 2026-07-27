"""Pure, truth-safe contracts shared by WO-17 capture and evidence.

This module deliberately imports only the Python standard library.  In
particular it must never import ``mib_pipeline``: the evidence builder loads
this contract while the truth path is present in its argument vector.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import stat
import tarfile
from datetime import date
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


CAPTURE_REPEAT_COUNT = 2
MAX_WORKERS = 4
SANDBOX_BACKEND = "macos_sandbox_exec_v1"
FORBIDDEN_ARCHIVE_PATHS = frozenset({"data/train_labels.csv"})

FIELD_NAMES = (
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
    "confidence",
)
POLICY_AUDIT_COUNT_NAMES = (
    "late_recovery_before_revalidation_count",
    "contradicted_synthetic_reason_removed_count",
    "independent_denial_reason_retained_count",
    "review_confidence_restored_count",
    "normal_policy_rerun_count",
    "signed_late_authority_recovery_count",
    "late_adjudication_evidence_preserved_count",
    "late_biohazard_evidence_preserved_count",
    "forced_approval_count",
    "serialization_default_used_as_policy_evidence_count",
    "sentinel_value_used_as_policy_evidence_count",
    "placeholder_value_used_as_evidence_count",
    "stale_threshold_mismatch_count",
    "contradicted_synthetic_reason_left_active_count",
)
MATCHER_COUNT_NAMES = (
    "eligible_guarded_initial_count",
    "guarded_initial_approval_count",
    "unguarded_initial_approval_count",
    "late_revalidation_approval_count",
)
CONTRACT_CHECK_NAMES = (
    "authoritative_review_veto",
    "explicit_denial_veto",
    "late_revalidation_false_preserves_review",
    "matcher_positive_control",
    "placeholder_veto",
    "sentinel_veto",
    "serialization_default_veto",
    "strikethrough_veto",
    "superseded_veto",
    "text_layer_veto",
    "wrong_applicant_scope_veto",
    "wrong_record_scope_veto",
)
AUDIT_ROOT_KEYS = frozenset(
    {
        "checks",
        "counts",
        "evidence_label",
        "evaluation_mode",
        "input_tree_sha256",
        "layout_manifest_sha256",
        "predictions_sha256",
        "producer_graph_sha256",
        "source_revision_sha",
        "status",
    }
)
OBSERVATION_ROOT_KEYS = frozenset(
    {
        "capture_tool_sha256",
        "checks",
        "counts",
        "deterministic",
        "evidence_label",
        "evaluation_mode",
        "first_audit_sha256",
        "first_predictions_sha256",
        "input_tree_sha256",
        "layout_manifest_sha256",
        "max_worker_count",
        "metrics",
        "producer_graph_sha256",
        "record_count",
        "repeat_count",
        "sandbox_backend_sha256",
        "sandbox_policy_sha256",
        "second_audit_sha256",
        "second_predictions_sha256",
        "source_archive_sha256",
        "source_revision_sha",
        "source_tree_sha256",
        "status",
    }
)
OBSERVATION_CHECKS = frozenset(
    {
        "archive_source_bound",
        "audit_deterministic",
        "byte_deterministic",
        "input_recomputed",
        "producer_graph_stable",
        "source_clean",
        "label_access_absent",
        "network_access_denied",
        "runtime_environment_exact",
        "sandbox_enforced",
        "sensitive_access_denied",
    }
)
OBSERVATION_COUNTS = frozenset(
    {
        "first_answered_count",
        "first_attempted_count",
        "first_omitted_count",
        "second_answered_count",
        "second_attempted_count",
        "second_omitted_count",
        "label_access_count",
    }
)
OBSERVATION_METRICS = frozenset(
    {
        "first_output_bytes",
        "first_peak_rss_bytes",
        "first_process_cpu_seconds",
        "first_runtime_seconds",
        "second_output_bytes",
        "second_peak_rss_bytes",
        "second_process_cpu_seconds",
        "second_runtime_seconds",
    }
)

_CASE_ID_PATTERN = re.compile(r"^MIB-[0-9]{6}$")
_SPONSOR_ID_PATTERN = re.compile(r"^SPN-[0-9]{4}$")
_FEE_VALUES = frozenset({"paid", "waived", "unpaid", "unknown"})
_ADJUDICATION_VALUES = frozenset(
    {"APPROVED", "DENIED", "NEEDS_REVIEW"}
)


class PredictionContractError(ValueError):
    """Prediction bytes do not satisfy the frozen public JSONL contract."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PredictionContractError(
            "value is not canonical-JSON serializable"
        ) from exc


def _safe_string(value: Any, fallback: str = "unknown") -> str:
    if not isinstance(value, str):
        return fallback
    normalized = value.strip()
    return normalized or fallback


def _safe_arrival_date(value: Any) -> str:
    if not isinstance(value, str):
        return "1900-01-01"
    normalized = value.strip()
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError:
        return "1900-01-01"
    return normalized if parsed.isoformat() == normalized else "1900-01-01"


def _safe_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        return 0.0
    return normalized


def _normalized_prediction(value: Mapping[str, Any]) -> dict[str, Any]:
    raw_case_id = value.get("case_id")
    case_id = raw_case_id.strip() if isinstance(raw_case_id, str) else ""
    if not _CASE_ID_PATTERN.fullmatch(case_id):
        raise PredictionContractError(
            "production prediction has no valid case_id"
        )

    raw_sponsor_id = value.get("sponsor_id")
    sponsor_id = (
        raw_sponsor_id.strip()
        if isinstance(raw_sponsor_id, str)
        else ""
    )
    if not _SPONSOR_ID_PATTERN.fullmatch(sponsor_id):
        sponsor_id = "SPN-0000"

    fee_status = _safe_string(value.get("fee_status"))
    if fee_status not in _FEE_VALUES:
        fee_status = "unknown"
    adjudication = _safe_string(
        value.get("adjudication"), "NEEDS_REVIEW"
    )
    if adjudication not in _ADJUDICATION_VALUES:
        adjudication = "NEEDS_REVIEW"

    normalized = {
        "case_id": case_id,
        "applicant_name": _safe_string(value.get("applicant_name")),
        "species_code": _safe_string(value.get("species_code")),
        "home_world": _safe_string(value.get("home_world")),
        "visa_class": _safe_string(value.get("visa_class")),
        "sponsor_id": sponsor_id,
        "arrival_date": _safe_arrival_date(value.get("arrival_date")),
        "declared_purpose": _safe_string(
            value.get("declared_purpose")
        ),
        "risk_flags": _safe_string(value.get("risk_flags"), "none"),
        "fee_status": fee_status,
        "adjudication": adjudication,
        "confidence": _safe_confidence(value.get("confidence")),
    }
    if tuple(normalized) != FIELD_NAMES:
        raise PredictionContractError(
            "internal prediction contract field order is invalid"
        )
    return normalized


def validate_prediction_bytes(
    content: bytes,
    *,
    expected_count: int,
) -> tuple[Mapping[str, Any], ...]:
    """Validate canonical sorted JSONL without importing candidate models."""

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise PredictionContractError(
            "expected prediction count must be positive"
        )
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PredictionContractError(
            "production predictions are not UTF-8"
        ) from exc

    rows: list[dict[str, Any]] = []
    try:
        for line in text.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise PredictionContractError(
                    "production prediction line is not an object"
                )
            rows.append(_normalized_prediction(value))
    except json.JSONDecodeError as exc:
        raise PredictionContractError(
            "production prediction is not canonically schema-valid"
        ) from exc

    case_ids = [str(row["case_id"]) for row in rows]
    if (
        len(rows) != expected_count
        or len(case_ids) != len(set(case_ids))
    ):
        raise PredictionContractError(
            "production predictions are incomplete or duplicated"
        )
    ordered = tuple(sorted(rows, key=lambda row: str(row["case_id"])))
    expected = "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for row in ordered
    ).encode("utf-8")
    if content != expected:
        raise PredictionContractError(
            "production prediction bytes are not canonical sorted JSONL"
        )
    return ordered


def archive_tree_sha256(raw: bytes) -> str:
    """Hash the ordinary extracted Git-tree bytes without extracting them."""

    files: list[tuple[bytes, bytes]] = []
    seen: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as bundle:
        for member in bundle.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or "\\" in member.name
                or any(part in {"", ".", ".."} for part in pure.parts)
                or member.name in seen
                or not (member.isdir() or member.isfile())
            ):
                raise RuntimeError("Git archive contains an unsafe entry")
            seen.add(member.name)
            if (
                member.isdir()
                or "__pycache__" in pure.parts
                or pure.as_posix() in FORBIDDEN_ARCHIVE_PATHS
            ):
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(
                    "Git archive regular file has no bytes"
                )
            files.append(
                (pure.as_posix().encode("utf-8"), source.read())
            )
    digest = hashlib.sha256()
    for relative, content in sorted(files):
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(content)
    return digest.hexdigest()


def grouped_producer_graph_sha256(
    repo_root: Path | None = None,
) -> str:
    """Hash production files plus this transitive capture contract."""

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]
    package_root = repo_root / "mib_pipeline"
    graph_paths = {
        path
        for path in package_root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    }
    artifact_root = package_root / "artifacts"
    if artifact_root.is_dir():
        graph_paths.update(
            path
            for path in artifact_root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    required = (
        repo_root / "requirements.lock",
        repo_root / "Dockerfile",
        repo_root / "run.sh",
        repo_root / "devtools" / "policy_grouped_capture_contract.py",
    )
    if any(
        not path.is_file()
        or stat.S_ISLNK(path.lstat().st_mode)
        for path in required
    ):
        raise RuntimeError(
            "grouped producer graph lacks a required regular file"
        )
    graph_paths.update(required)
    if not graph_paths:
        raise RuntimeError("grouped producer graph contains no source files")

    entries = []
    for path in sorted(graph_paths):
        raw = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
    return hashlib.sha256(
        (_canonical_json(entries) + "\n").encode("utf-8")
    ).hexdigest()
