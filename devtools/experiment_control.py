"""Leakage-resistant, auditable controls for offline score experiments.

The types in this module deliberately keep protected-set evidence aggregate-only.
They are development controls and are not imported by the submission runtime.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import weakref
from dataclasses import dataclass
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
        "baseline_score",
        "baseline_total_score",
        "baseline_verified",
        "blank_records",
        "brier_score",
        "calibration_score",
        "candidate_image_bytes",
        "candidate_max_model_artifact_bytes",
        "candidate_model_bytes",
        "candidate_score",
        "catastrophic_false_approvals",
        "classification_score",
        "count",
        "deterministic",
        "duplicate_records",
        "error_count",
        "expected_record_count",
        "extra_records",
        "extraction_score",
        "false_approvals",
        "fold_count",
        "fold_consistent",
        "fraction",
        "group_count",
        "hard_gate_failure_count",
        "input_pdf_count",
        "invalid_records",
        "largest_group_count",
        "layout_group_count",
        "leakage_clean",
        "leakage_finding_count",
        "matching_layout_taint_group_count",
        "matching_layout_taint_record_count",
        "mean_brier",
        "mean",
        "median",
        "min",
        "max",
        "missing_records",
        "output_bytes",
        "peak_container_memory_bytes",
        "peak_rss_bytes",
        "positive_target_count",
        "record_count",
        "regression_waiver_count",
        "repeat_count",
        "precision",
        "process_cpu_seconds",
        "promotion_gate_verified",
        "recall",
        "rate",
        "runtime_seconds",
        "score",
        "score_delta",
        "singleton_group_count",
        "smallest_group_count",
        "split_count",
        "stddev",
        "synthetic_exclusion_group_count",
        "synthetic_exclusion_record_count",
        "taint_event_count",
        "total_count",
        "total_score",
        "training_case_count",
        "tmp_bytes",
        "tuning_group_count",
        "tuning_record_count",
        "value",
        "validation_group_assignment_count",
        "validation_group_count",
        "validation_record_assignment_count",
        "validation_record_count",
        "variance",
        "warning_count",
        "whole_public_cohort_taint_token_event_count",
    }
)
_AGGREGATE_HASH_KEYS = frozenset(
    {
        "artifact_sha256",
        "baseline_artifact_sha256",
        "baseline_manifest_sha256",
        "candidate_artifact_sha256",
        "candidate_sha256",
        "candidate_state_record_hash",
        "dataset_archive_sha256",
        "dimension_source_sha256",
        "evaluator_sha256",
        "expected_input_tree_sha256",
        "expected_layout_manifest_sha256",
        "expected_taint_registry_sha256",
        "expected_tool_source_sha256",
        "file_sha256",
        "frozen_baseline_manifest_sha256",
        "hypothesis_sha256",
        "input_tree_sha256",
        "layout_manifest_sha256",
        "manifest_sha256",
        "plan_record_hash",
        "predictions_sha256",
        "primary_variable_sha256",
        "protected_access_record_hash",
        "record_hash",
        "runtime_contract_sha256",
        "runtime_evidence_sha256",
        "runtime_graph_sha256",
        "source_sha256",
        "split_manifest_sha256",
        "submission_sha256",
        "taint_registry_head_sha256",
        "taint_registry_sha256",
        "tool_source_sha256",
        "trace_tool_sha256",
        "truth_sha256",
    }
)
_AGGREGATE_REVISION_KEYS = frozenset(
    {
        "baseline_commit_sha",
        "checkout_revision_sha",
        "parent_commit_sha",
        "source_revision_sha",
    }
)
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
_AGGREGATE_FIELD_DIMENSIONS = frozenset(
    {
        "adjudication",
        "applicant_name",
        "arrival_date",
        "confidence",
        "declared_purpose",
        "fee_status",
        "home_world",
        "risk",
        "risk_flags",
        "species_code",
        "sponsor_id",
        "visa_class",
    }
)
_AGGREGATE_CLASS_DIMENSIONS = frozenset(
    {
        "approved",
        "denied",
        "needs_review",
        "negative",
        "positive",
        "unknown",
    }
)
_AGGREGATE_CONFUSION_DIMENSIONS = frozenset(
    f"{truth}_to_{prediction}"
    for truth in ("approved", "denied", "needs_review")
    for prediction in ("approved", "denied", "needs_review")
)
_AGGREGATE_CHECK_DIMENSIONS = frozenset(
    {
        "baseline_verified",
        "candidate_verified",
        "decision_freeze_verified",
        "deterministic",
        "fold_consistent",
        "group_exclusive",
        "input_population_matches_manifest",
        "input_tree_digest_verified",
        "leakage_clean",
        "manifest_canonical_freezer_output",
        "manifest_declares_label_blind_construction",
        "manifest_digest_verified",
        "no_unseen_or_protected_claim",
        "population_coverage_once_per_repeat",
        "public_robustness_not_unseen",
        "runtime_leakage_clean",
        "runtime_limits_verified",
        "source_revision_verified",
        "split_deterministic",
        "synthetic_exclusion_mechanics_verified",
        "tool_source_verified",
        "whole_public_cohort_taint_token_present",
    }
)
_AGGREGATE_REGRESSION_DIMENSIONS = frozenset(
    {
        "adversarial",
        "confidence",
        "decision",
        "extraction",
        "golden",
        "runtime",
        "schema",
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
        "protected_access_binding_sha256",
        "runtime_contract_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    }
)
_EXPERIMENT_PLAN_SHA256_KEYS = frozenset(
    {
        "evaluator_sha256",
        "hypothesis_sha256",
        "input_tree_sha256",
        "primary_variable_sha256",
        "runtime_contract_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    }
)
_EXPERIMENT_EVIDENCE_LABELS = frozenset(
    {
        "aggregate_only",
        "protected",
        "public_grouped_robustness_not_unseen",
    }
)
_EXPERIMENT_RESULT_DECISIONS = frozenset({"adopt", "reject", "rollback"})
_EXPERIMENT_RESULT_KEYS = frozenset(
    {
        "baseline_artifact_sha256",
        "candidate_artifact_sha256",
        "candidate_state_record_hash",
        "checks",
        "evaluator_sha256",
        "evidence_label",
        "expected_record_count",
        "fold_metrics",
        "input_tree_sha256",
        "metrics",
        "protected_access_record_hash",
        "runtime_contract_sha256",
        "runtime_evidence_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    }
)
_EXPERIMENT_RESULT_CHECKS = frozenset(
    {
        "baseline_verified",
        "candidate_verified",
        "decision_freeze_verified",
        "deterministic",
        "fold_consistent",
        "runtime_leakage_clean",
        "runtime_limits_verified",
    }
)
_EXPERIMENT_RESULT_METRICS = frozenset(
    {
        "baseline_total_score",
        "calibration_score",
        "candidate_image_bytes",
        "candidate_max_model_artifact_bytes",
        "candidate_model_bytes",
        "catastrophic_false_approvals",
        "classification_score",
        "duplicate_records",
        "extra_records",
        "extraction_score",
        "invalid_records",
        "missing_records",
        "output_bytes",
        "peak_container_memory_bytes",
        "peak_rss_bytes",
        "process_cpu_seconds",
        "record_count",
        "runtime_seconds",
        "score_delta",
        "tmp_bytes",
        "total_score",
    }
)
_EXPERIMENT_RESULT_INTEGER_METRICS = frozenset(
    {
        "candidate_image_bytes",
        "candidate_max_model_artifact_bytes",
        "candidate_model_bytes",
        "catastrophic_false_approvals",
        "duplicate_records",
        "extra_records",
        "invalid_records",
        "missing_records",
        "output_bytes",
        "peak_container_memory_bytes",
        "peak_rss_bytes",
        "record_count",
        "tmp_bytes",
    }
)
_EXPERIMENT_FOLD_KEYS = frozenset(
    f"repeat_{repeat}_fold_{fold}"
    for repeat in range(1, 4)
    for fold in range(1, 6)
)
_EXPERIMENT_FOLD_METRICS = frozenset(
    {
        "baseline_score",
        "candidate_score",
        "catastrophic_false_approvals",
        "invalid_records",
        "missing_records",
        "record_count",
        "score_delta",
        "validation_group_count",
    }
)
_WO12_MANIFEST_ROOT_KEYS = frozenset(
    {
        "cases",
        "folds",
        "label_blind_construction",
        "layout_signature",
        "repeats",
        "schema",
        "split_seed",
    }
)
_WO12_LAYOUT_SIGNATURE_KEYS = frozenset(
    {
        "first_page_grayscale_ink_bucket_width",
        "first_page_grayscale_ink_pixel_threshold_exclusive",
        "first_page_render_height",
        "first_page_render_width",
        "inputs",
        "pillow_version",
        "pdfium_version",
        "pypdfium2_version",
        "version",
    }
)
_WO12_CASE_ID_RE = re.compile(r"MIB-[0-9]{6}")
_WO12_LAYOUT_GROUP_RE = re.compile(
    r"page-count-[0-9]{2,}__ink-bucket-[0-9]{2,}"
)
_CANDIDATE_ASSESSMENT_KEYS = frozenset(
    {
        "aggregate_evidence",
        "assessment_id",
        "candidate_id",
        "candidate_sha256",
        "decision",
        "event",
    }
)
_PROMOTION_GATE_RESULT_KEYS = frozenset(
    {
        "access_authorized",
        "baseline_verified",
        "deterministic",
        "fold_consistent",
        "no_false_approvals",
        "no_invalid_records",
        "no_leakage",
        "no_missing_records",
        "regressions_cleared",
    }
)
_PROTECTED_ACCESS_KEYS = frozenset(
    {
        "access_id",
        "aggregate_result",
        "candidate_sha256",
        "event",
        "purpose",
    }
)
_MAX_AGGREGATE_ROOT_ENTRIES = 32
_MAX_AGGREGATE_CONTAINER_ENTRIES = 32
_MAX_AGGREGATE_NESTED_METRICS = 32
_MAX_AGGREGATE_SEQUENCE_VALUES = 15
_MAX_AGGREGATE_SCALAR_VALUES = 192
_MAX_AGGREGATE_ABSOLUTE_VALUE = 1_000_000_000_000.0
_MAX_AGGREGATE_COUNT = 10_000_000
_MAX_AGGREGATE_BYTES = 1024**4
_MAX_AGGREGATE_SECONDS = 31.0 * 24.0 * 60.0 * 60.0
_MAX_SPLIT_MANIFEST_BYTES = 4 * 1024**2
_MAX_EXPERIMENT_RECORD_COUNT = 5_000
_MAX_RUNTIME_SECONDS = 4.0 * 60.0 * 60.0
_MAX_RUNTIME_SECONDS_PER_RECORD = 6.0
_MAX_PROCESS_CPU_SECONDS = 120_000.0
_MAX_IMAGE_BYTES = 4 * 1024**3
_MAX_MODEL_BYTES = 1024**3
_MAX_MODEL_ARTIFACT_BYTES = 250 * 1024**2
_MAX_OUTPUT_BYTES = 25 * 1024**2
_MAX_MEMORY_BYTES = 8 * 1024**3
_MAX_TMP_BYTES = 2 * 1024**3
_MAX_RESOURCE_REPRESENTATION_BYTES = 1024**4
_MAX_RESOURCE_REPRESENTATION_SECONDS = 31.0 * 24.0 * 60.0 * 60.0


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


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ExperimentControlError(f"{name} must be a SHA-256 hex digest")
    normalized = value.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ExperimentControlError(f"{name} must be a SHA-256 hex digest")
    return normalized


def _normalize_experiment_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable, non-identifying pre-execution contract."""

    if not isinstance(plan, Mapping) or set(plan) != _EXPERIMENT_PLAN_KEYS:
        raise ExperimentControlError(
            "experiment plan must contain the exact governed schema"
        )
    raw_changed_files = plan["changed_files"]
    if (
        not isinstance(raw_changed_files, (list, tuple))
        or not raw_changed_files
        or len(raw_changed_files) > 64
    ):
        raise ExperimentControlError(
            "experiment plan changed_files must be a non-empty bounded list"
        )
    changed_files: list[str] = []
    for raw_path in raw_changed_files:
        if not isinstance(raw_path, str):
            raise ExperimentControlError(
                "experiment plan changed_files entries must be strings"
            )
        path = raw_path.strip()
        pure_path = PurePosixPath(path)
        _require_nonidentifying_control_text("changed_files", path)
        if (
            not path
            or path != raw_path
            or "\\" in path
            or pure_path.is_absolute()
            or str(pure_path) != path
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise ExperimentControlError(
                "experiment plan changed_files must be relative POSIX paths"
            )
        changed_files.append(path)
    if len(set(changed_files)) != len(changed_files):
        raise ExperimentControlError(
            "experiment plan changed_files must not contain duplicates"
        )

    if not isinstance(plan["evidence_label"], str):
        raise ExperimentControlError(
            "experiment plan evidence_label must be a string"
        )
    evidence_label = plan["evidence_label"].strip().casefold()
    if evidence_label not in _EXPERIMENT_EVIDENCE_LABELS:
        raise ExperimentControlError(
            "experiment plan evidence_label is not governed"
        )
    expected_record_count = plan["expected_record_count"]
    if (
        isinstance(expected_record_count, bool)
        or not isinstance(expected_record_count, int)
        or not 5 <= expected_record_count <= _MAX_EXPERIMENT_RECORD_COUNT
    ):
        raise ExperimentControlError(
            "experiment plan expected_record_count must be between 5 and 5000"
        )
    if not isinstance(plan["parent_commit_sha"], str):
        raise ExperimentControlError(
            "experiment plan parent_commit_sha must be a Git SHA"
        )
    parent_commit_sha = plan["parent_commit_sha"].strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", parent_commit_sha):
        raise ExperimentControlError(
            "experiment plan parent_commit_sha must be a Git SHA"
        )

    normalized = {
        "changed_files": sorted(changed_files),
        "evidence_label": evidence_label,
        "expected_record_count": expected_record_count,
        "parent_commit_sha": parent_commit_sha,
    }
    for key in sorted(_EXPERIMENT_PLAN_SHA256_KEYS):
        normalized[key] = _require_sha256(key, plan[key])
    protected_binding = plan["protected_access_binding_sha256"]
    if evidence_label == "protected":
        normalized["protected_access_binding_sha256"] = _require_sha256(
            "protected_access_binding_sha256",
            protected_binding,
        )
    elif protected_binding is not None:
        raise ExperimentControlError(
            "only protected plans may bind a protected access budget"
        )
    else:
        normalized["protected_access_binding_sha256"] = None
    return normalized


def _read_external_split_manifest(path: Path | str) -> bytes:
    """Read one bounded, regular manifest snapshot outside the repository."""

    raw_path = Path(path)
    if not raw_path.is_absolute():
        raise IntegrityError("split manifest path must be absolute")
    try:
        resolved = raw_path.resolve(strict=True)
        repository_root = Path(__file__).resolve().parents[1]
        if resolved == repository_root or repository_root in resolved.parents:
            raise IntegrityError(
                "split manifest must remain outside the repository"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(raw_path, flags)
    except IntegrityError:
        raise
    except OSError as exc:
        raise IntegrityError("split manifest is not a readable regular file") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_SPLIT_MANIFEST_BYTES
        ):
            raise IntegrityError("split manifest is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise IntegrityError("split manifest changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise IntegrityError("split manifest changed while being read")
        after = os.fstat(descriptor)
        current = raw_path.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        if identity_before != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or (before.st_dev, before.st_ino) != (
            current.st_dev,
            current.st_ino,
        ):
            raise IntegrityError("split manifest changed while being read")
        return b"".join(chunks)
    except OSError as exc:
        raise IntegrityError("split manifest could not be read atomically") from exc
    finally:
        os.close(descriptor)


def _verify_experiment_split_manifest(
    path: Path | str,
    *,
    input_dir: Path | str,
    expected_input_tree_sha256: str,
    expected_sha256: str,
    expected_record_count: int,
) -> dict[str, dict[str, int]]:
    """Bind exact 3x5 aggregate rows to a canonical external WO-12 manifest."""

    content = _read_external_split_manifest(path)
    if _sha256_bytes(content) != _require_sha256(
        "split_manifest_sha256", expected_sha256
    ):
        raise IntegrityError(
            "split manifest does not match the preregistered SHA-256"
        )
    try:
        raw = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(
            "split manifest must be canonical UTF-8 JSON"
        ) from exc
    if (
        not isinstance(raw, dict)
        or set(raw) != _WO12_MANIFEST_ROOT_KEYS
        or content != (canonical_json(raw) + "\n").encode("utf-8")
    ):
        raise IntegrityError(
            "split manifest does not match canonical WO-12 v2 schema"
        )
    split_seed = raw["split_seed"]
    if (
        raw["schema"] != "mib-wo12-layout-groups/v2"
        or raw["repeats"] != 3
        or raw["folds"] != 5
        or raw["label_blind_construction"] is not True
        or not isinstance(split_seed, str)
        or not split_seed
        or split_seed != split_seed.strip()
        or len(split_seed) > 256
    ):
        raise IntegrityError("split manifest has an invalid WO-12 contract")

    signature = raw["layout_signature"]
    expected_signature = {
        "first_page_grayscale_ink_bucket_width": 0.03,
        "first_page_grayscale_ink_pixel_threshold_exclusive": 210,
        "first_page_render_height": 166,
        "first_page_render_width": 128,
        "inputs": ["pdf_page_count", "first_page_rendered_pixels"],
        "version": "page-count-plus-first-page-ink-v1",
    }
    if (
        not isinstance(signature, dict)
        or set(signature) != _WO12_LAYOUT_SIGNATURE_KEYS
        or any(signature.get(key) != value for key, value in expected_signature.items())
        or any(
            not isinstance(signature.get(key), str)
            or not signature[key].strip()
            or len(signature[key]) > 128
            for key in (
                "pillow_version",
                "pdfium_version",
                "pypdfium2_version",
            )
        )
    ):
        raise IntegrityError(
            "split manifest has invalid label-blind signature metadata"
        )

    rows = raw["cases"]
    if (
        not isinstance(rows, list)
        or len(rows) != expected_record_count
        or any(
            not isinstance(row, dict)
            or set(row) != {"case_id", "layout_group"}
            or not isinstance(row["case_id"], str)
            or _WO12_CASE_ID_RE.fullmatch(row["case_id"]) is None
            or not isinstance(row["layout_group"], str)
            or len(row["layout_group"]) > 80
            or _WO12_LAYOUT_GROUP_RE.fullmatch(row["layout_group"]) is None
            for row in rows
        )
        or [row["case_id"] for row in rows]
        != sorted(row["case_id"] for row in rows)
    ):
        raise IntegrityError("split manifest case rows are invalid")

    groups: dict[str, list[str]] = {}
    seen_case_ids: set[str] = set()
    for row in rows:
        case_id = row["case_id"]
        if case_id in seen_case_ids:
            raise IntegrityError(
                "split manifest case identities must occur exactly once"
            )
        seen_case_ids.add(case_id)
        groups.setdefault(row["layout_group"], []).append(case_id)
    if len(groups) < 5:
        raise IntegrityError(
            "split manifest must contain at least five layout groups"
        )

    try:
        splits = RepeatedGroupedSplitManager(
            seed=split_seed,
            repeats=3,
            folds=5,
        ).split_groups(groups)
    except ExperimentControlError as exc:
        raise IntegrityError("split manifest cannot produce the governed layout") from exc
    expected_folds: dict[str, dict[str, int]] = {}
    repeat_assignments: list[tuple[tuple[str, int], ...]] = []
    all_groups = set(groups)
    for repeat in range(3):
        repeat_splits = tuple(
            split for split in splits if split.repeat == repeat
        )
        if len(repeat_splits) != 5:
            raise IntegrityError("split manifest did not produce exact 3x5 folds")
        covered_groups: set[str] = set()
        covered_cases: set[str] = set()
        assignment: list[tuple[str, int]] = []
        for split in repeat_splits:
            validation_groups = set(split.validation_groups)
            validation_cases = set(split.validation_case_ids)
            if (
                validation_groups & set(split.tuning_groups)
                or validation_cases & set(split.tuning_case_ids)
                or covered_groups & validation_groups
                or covered_cases & validation_cases
            ):
                raise IntegrityError(
                    "split manifest violates whole-group exclusivity"
                )
            covered_groups.update(validation_groups)
            covered_cases.update(validation_cases)
            assignment.extend(
                (group_id, split.fold) for group_id in validation_groups
            )
            expected_folds[
                f"repeat_{repeat + 1}_fold_{split.fold + 1}"
            ] = {
                "record_count": len(validation_cases),
                "validation_group_count": len(validation_groups),
            }
        if covered_groups != all_groups or covered_cases != seen_case_ids:
            raise IntegrityError(
                "split manifest does not cover the full population per repeat"
            )
        repeat_assignments.append(tuple(sorted(assignment)))
    if len(set(repeat_assignments)) != 3:
        raise IntegrityError(
            "split manifest repeats must produce distinct group assignments"
        )
    if set(expected_folds) != _EXPERIMENT_FOLD_KEYS:
        raise IntegrityError("split manifest did not produce exact 3x5 folds")

    try:
        from devtools import grouped_split_evidence

        snapshot = grouped_split_evidence._strict_freezer_manifest(
            path,
            expected_sha256=expected_sha256,
        )
        actual_input_tree_sha256 = (
            grouped_split_evidence._verify_input_tree_and_recomputed_manifest(
                input_dir,
                snapshot,
                expected_sha256=expected_input_tree_sha256,
            )
        )
    except Exception as exc:
        if isinstance(exc, IntegrityError):
            raise
        raise IntegrityError(
            "split manifest is not derived from the bound label-blind PDF tree"
        ) from exc
    if actual_input_tree_sha256 != _require_sha256(
        "input_tree_sha256", expected_input_tree_sha256
    ):
        raise IntegrityError(
            "recomputed split input tree does not match its plan"
        )
    return expected_folds


def _normalize_experiment_result(
    evidence: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    decision: str,
    expected_folds: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    """Validate one exact, bounded 3x5 result bound to its immutable plan."""

    if not isinstance(evidence, Mapping):
        raise ExperimentControlError("experiment result evidence is required")
    require_aggregate_only(evidence)
    normalized = json.loads(canonical_json(dict(evidence)))
    if set(normalized) != _EXPERIMENT_RESULT_KEYS:
        raise ExperimentControlError(
            "experiment result must contain the exact governed result schema"
        )

    evidence_label = normalized["evidence_label"]
    if (
        not isinstance(evidence_label, str)
        or evidence_label != plan["evidence_label"]
    ):
        raise ExperimentControlError(
            "experiment result evidence_label does not match its plan"
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
        normalized[key] = _require_sha256(key, normalized[key])
    for key in (
        "evaluator_sha256",
        "input_tree_sha256",
        "runtime_contract_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    ):
        if normalized[key] != plan[key]:
            raise ExperimentControlError(
                f"experiment result {key} does not match its plan"
            )
    protected_access_record_hash = normalized["protected_access_record_hash"]
    if evidence_label == "protected":
        normalized["protected_access_record_hash"] = _require_sha256(
            "protected_access_record_hash",
            protected_access_record_hash,
        )
    elif protected_access_record_hash is not None:
        raise ExperimentControlError(
            "non-protected results may not claim protected access"
        )
    candidate_state_record_hash = normalized["candidate_state_record_hash"]
    if decision == "adopt":
        normalized["candidate_state_record_hash"] = _require_sha256(
            "candidate_state_record_hash",
            candidate_state_record_hash,
        )
        if (
            normalized["candidate_artifact_sha256"]
            == normalized["baseline_artifact_sha256"]
        ):
            raise ExperimentControlError(
                "an adopted candidate must differ from the frozen baseline"
            )
    elif candidate_state_record_hash is not None:
        raise ExperimentControlError(
            "non-adopted results may not claim candidate promotion state"
        )

    expected_record_count = normalized["expected_record_count"]
    if (
        isinstance(expected_record_count, bool)
        or not isinstance(expected_record_count, int)
        or expected_record_count != plan["expected_record_count"]
    ):
        raise ExperimentControlError(
            "experiment result record count does not match its plan"
        )

    checks = normalized["checks"]
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _EXPERIMENT_RESULT_CHECKS
        or any(not isinstance(value, bool) for value in checks.values())
    ):
        raise ExperimentControlError(
            "experiment result checks must contain the exact boolean gate schema"
        )

    metrics = normalized["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != _EXPERIMENT_RESULT_METRICS:
        raise ExperimentControlError(
            "experiment result metrics must contain the exact metric schema"
        )
    for name in _EXPERIMENT_RESULT_METRICS:
        value = metrics[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (
                name not in {"score_delta"}
                and float(value) < 0.0
            )
            or (
                name in _EXPERIMENT_RESULT_INTEGER_METRICS
                and not isinstance(value, int)
            )
        ):
            raise ExperimentControlError(
                f"experiment result metric is invalid: {name}"
            )
    if any(
        int(metrics[name]) > expected_record_count
        for name in (
            "catastrophic_false_approvals",
            "duplicate_records",
            "extra_records",
            "invalid_records",
            "missing_records",
            "record_count",
        )
    ):
        raise ExperimentControlError(
            "experiment result record metrics exceed the planned population"
        )
    score_limits = {
        "baseline_total_score": 150.0,
        "calibration_score": 20.0,
        "classification_score": 80.0,
        "extraction_score": 50.0,
        "total_score": 150.0,
    }
    if any(float(metrics[name]) > limit for name, limit in score_limits.items()):
        raise ExperimentControlError("experiment result score is out of range")
    byte_metrics = (
        "candidate_image_bytes",
        "candidate_max_model_artifact_bytes",
        "candidate_model_bytes",
        "output_bytes",
        "peak_container_memory_bytes",
        "peak_rss_bytes",
        "tmp_bytes",
    )
    if any(
        int(metrics[name]) > _MAX_RESOURCE_REPRESENTATION_BYTES
        for name in byte_metrics
    ) or any(
        float(metrics[name]) > _MAX_RESOURCE_REPRESENTATION_SECONDS
        for name in ("process_cpu_seconds", "runtime_seconds")
    ):
        raise ExperimentControlError(
            "experiment resource metric exceeds its representation cap"
        )
    component_total = (
        float(metrics["extraction_score"])
        + float(metrics["classification_score"])
        + float(metrics["calibration_score"])
    )
    if not math.isclose(
        float(metrics["total_score"]),
        component_total,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ExperimentControlError(
            "experiment total_score must equal its component scores"
        )
    expected_delta = (
        float(metrics["total_score"])
        - float(metrics["baseline_total_score"])
    )
    if not math.isclose(
        float(metrics["score_delta"]),
        expected_delta,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ExperimentControlError(
            "experiment score_delta must equal candidate minus baseline"
        )

    fold_metrics = normalized["fold_metrics"]
    if not isinstance(fold_metrics, Mapping) or set(fold_metrics) != _EXPERIMENT_FOLD_KEYS:
        raise ExperimentControlError(
            "experiment result must contain exactly three repeats of five folds"
        )
    if set(expected_folds) != _EXPERIMENT_FOLD_KEYS:
        raise IntegrityError("verified split layout is not exact 3x5 evidence")
    repeat_deltas: list[float] = []
    for repeat in range(1, 4):
        repeat_rows: list[Mapping[str, Any]] = []
        for fold in range(1, 6):
            fold_key = f"repeat_{repeat}_fold_{fold}"
            fold_row = fold_metrics[fold_key]
            if (
                not isinstance(fold_row, Mapping)
                or set(fold_row) != _EXPERIMENT_FOLD_METRICS
            ):
                raise ExperimentControlError(
                    f"experiment fold has an invalid schema: {fold_key}"
                )
            for name in ("baseline_score", "candidate_score", "score_delta"):
                value = fold_row[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ExperimentControlError(
                        f"experiment fold metric is invalid: {fold_key}.{name}"
                    )
            if not (
                0.0 <= float(fold_row["baseline_score"]) <= 150.0
                and 0.0 <= float(fold_row["candidate_score"]) <= 150.0
            ):
                raise ExperimentControlError(
                    f"experiment fold score is out of range: {fold_key}"
                )
            if not math.isclose(
                float(fold_row["score_delta"]),
                float(fold_row["candidate_score"])
                - float(fold_row["baseline_score"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ExperimentControlError(
                    f"experiment fold score_delta is inconsistent: {fold_key}"
                )
            for name in (
                "catastrophic_false_approvals",
                "invalid_records",
                "missing_records",
                "record_count",
                "validation_group_count",
            ):
                value = fold_row[name]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value
                    < (
                        1
                        if name in {"record_count", "validation_group_count"}
                        else 0
                    )
                ):
                    raise ExperimentControlError(
                        f"experiment fold count is invalid: {fold_key}.{name}"
                    )
            if any(
                int(fold_row[name]) != int(expected_folds[fold_key][name])
                for name in ("record_count", "validation_group_count")
            ):
                raise ExperimentControlError(
                    f"experiment fold population does not match the frozen "
                    f"layout: {fold_key}"
                )
            if any(
                int(fold_row[name]) > int(fold_row["record_count"])
                for name in (
                    "catastrophic_false_approvals",
                    "invalid_records",
                    "missing_records",
                )
            ):
                raise ExperimentControlError(
                    f"experiment fold failures exceed its population: {fold_key}"
                )
            repeat_rows.append(fold_row)
        if sum(int(row["record_count"]) for row in repeat_rows) != expected_record_count:
            raise ExperimentControlError(
                f"experiment repeat {repeat} does not cover the planned population"
            )
        weighted_baseline = sum(
            float(row["baseline_score"]) * int(row["record_count"])
            for row in repeat_rows
        ) / expected_record_count
        weighted_candidate = sum(
            float(row["candidate_score"]) * int(row["record_count"])
            for row in repeat_rows
        ) / expected_record_count
        if not math.isclose(
            weighted_baseline,
            float(metrics["baseline_total_score"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            weighted_candidate,
            float(metrics["total_score"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ExperimentControlError(
                "each repeat's weighted baseline and candidate scores must "
                "equal their aggregate totals"
            )
        weighted_delta = sum(
            float(row["score_delta"]) * int(row["record_count"])
            for row in repeat_rows
        )
        repeat_deltas.append(weighted_delta / expected_record_count)
    if any(
        not math.isclose(
            delta,
            float(metrics["score_delta"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        for delta in repeat_deltas
    ):
        raise ExperimentControlError(
            "each repeat's weighted fold delta must equal aggregate score_delta"
        )
    if (
        int(metrics["candidate_max_model_artifact_bytes"])
        > int(metrics["candidate_model_bytes"])
    ):
        raise ExperimentControlError(
            "largest model artifact cannot exceed total model bytes"
        )

    if decision == "adopt":
        hard_gate_failed = (
            not all(checks.values())
            or int(metrics["record_count"]) != expected_record_count
            or int(metrics["catastrophic_false_approvals"]) != 0
            or int(metrics["duplicate_records"]) != 0
            or int(metrics["extra_records"]) != 0
            or int(metrics["invalid_records"]) != 0
            or int(metrics["missing_records"]) != 0
            or float(metrics["score_delta"]) <= 0.0
            or int(metrics["candidate_image_bytes"]) <= 0
            or int(metrics["candidate_image_bytes"]) > _MAX_IMAGE_BYTES
            or int(metrics["candidate_model_bytes"]) > _MAX_MODEL_BYTES
            or int(metrics["candidate_max_model_artifact_bytes"])
            > _MAX_MODEL_ARTIFACT_BYTES
            or int(metrics["output_bytes"]) <= 0
            or int(metrics["output_bytes"]) > _MAX_OUTPUT_BYTES
            or int(metrics["peak_container_memory_bytes"]) <= 0
            or int(metrics["peak_container_memory_bytes"]) > _MAX_MEMORY_BYTES
            or int(metrics["peak_rss_bytes"]) <= 0
            or int(metrics["peak_rss_bytes"]) > _MAX_MEMORY_BYTES
            or float(metrics["process_cpu_seconds"]) <= 0.0
            or float(metrics["process_cpu_seconds"]) > _MAX_PROCESS_CPU_SECONDS
            or float(metrics["runtime_seconds"]) <= 0.0
            or float(metrics["runtime_seconds"]) > _MAX_RUNTIME_SECONDS
            or float(metrics["runtime_seconds"]) / expected_record_count
            > _MAX_RUNTIME_SECONDS_PER_RECORD
            or int(metrics["tmp_bytes"]) > _MAX_TMP_BYTES
            or any(delta <= 1e-12 for delta in repeat_deltas)
            or any(
                float(row["score_delta"]) < -1e-12
                for row in fold_metrics.values()
            )
        )
        if not hard_gate_failed:
            for repeat in range(1, 4):
                rows = [
                    fold_metrics[f"repeat_{repeat}_fold_{fold}"]
                    for fold in range(1, 6)
                ]
                weighted_sum = sum(
                    float(row["score_delta"]) * int(row["record_count"])
                    for row in rows
                )
                total_weight = sum(int(row["record_count"]) for row in rows)
                if any(
                    (
                        weighted_sum
                        - float(row["score_delta"]) * int(row["record_count"])
                    )
                    / (total_weight - int(row["record_count"]))
                    <= 1e-12
                    for row in rows
                ):
                    hard_gate_failed = True
                    break
        if not hard_gate_failed and any(
            int(row[name]) != 0
            for row in fold_metrics.values()
            for name in (
                "catastrophic_false_approvals",
                "invalid_records",
                "missing_records",
            )
        ):
            hard_gate_failed = True
        if hard_gate_failed:
            raise ExperimentControlError(
                "an experiment with failed hard gates cannot be adopted"
            )
    return normalized


def _raise_aggregate_schema(path: str, reason: str) -> None:
    raise LeakageError(f"protected evidence is not aggregate-only at {path}: {reason}")


def _require_nonidentifying_control_text(name: str, value: str) -> None:
    if _CASE_ID_RE.search(value) or _PDF_FILENAME_RE.search(value.strip()):
        raise LeakageError(f"{name} must not contain case or PDF identity")


def _validate_aggregate_scalar(key: str, value: Any, *, path: str) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            numeric_value = float(value)
        except (OverflowError, ValueError) as exc:
            raise LeakageError(
                f"protected evidence is not aggregate-only at {path}: "
                "numeric metric is not finitely representable"
            ) from exc
        if (
            not math.isfinite(numeric_value)
            or abs(numeric_value) > _MAX_AGGREGATE_ABSOLUTE_VALUE
        ):
            _raise_aggregate_schema(
                path, "numeric metric exceeds its finite representation cap"
            )
        if key == "count" or key.endswith("_count"):
            if (
                not isinstance(value, int)
                or value < 0
                or value > _MAX_AGGREGATE_COUNT
            ):
                _raise_aggregate_schema(
                    path, "count metrics must be bounded non-negative integers"
                )
        if key.endswith("_bytes"):
            if (
                not isinstance(value, int)
                or value < 0
                or value > _MAX_AGGREGATE_BYTES
            ):
                _raise_aggregate_schema(
                    path, "resource bytes must be bounded non-negative integers"
                )
        if key.endswith("_seconds") and (
            numeric_value < 0.0
            or numeric_value > _MAX_AGGREGATE_SECONDS
        ):
            _raise_aggregate_schema(
                path, "resource seconds must be bounded and non-negative"
            )
        canonical_json(value)
        return
    if value is None:
        return
    if isinstance(value, str):
        if _CASE_ID_RE.search(value) or _PDF_FILENAME_RE.search(value.strip()):
            _raise_aggregate_schema(path, "case or filename identity is forbidden")
        if key in _AGGREGATE_HASH_KEYS | _AGGREGATE_REVISION_KEYS:
            digest_pattern = (
                _COMMIT_OR_SHA256_RE
                if key in _AGGREGATE_REVISION_KEYS
                else _SHA256_RE
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
        or key in _AGGREGATE_HASH_KEYS
        or key in _AGGREGATE_REVISION_KEYS
    )


def _aggregate_scalar_value_count(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_aggregate_scalar_value_count(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_aggregate_scalar_value_count(child) for child in value)
    return 1


def _validate_dimension_name(key: str, *, path: str) -> None:
    if not _SAFE_DIMENSION_RE.fullmatch(key):
        _raise_aggregate_schema(path, "invalid aggregate dimension")
    if (
        _CASE_ID_RE.search(key)
        or _PDF_FILENAME_RE.search(key)
        or
        _FILE_HASH_LOOKUP_NAME_RE.search(key)
        or _IDENTITY_DIMENSION_RE.search(key)
        or key.casefold().endswith(("_id", "_ids"))
    ):
        _raise_aggregate_schema(path, "identity and per-file dimensions are forbidden")


def _dimension_is_semantic(container_key: str, key: str) -> bool:
    if container_key in {"counts", "metrics", "score_components"}:
        return key in _AGGREGATE_SCALAR_KEYS
    if container_key == "checks":
        return key in _AGGREGATE_CHECK_DIMENSIONS
    if container_key == "confusion_counts":
        return key in _AGGREGATE_CONFUSION_DIMENSIONS
    if container_key == "gate_results":
        return key in _PROMOTION_GATE_RESULT_KEYS
    if container_key in {"regression_counts", "regression_waivers"}:
        return key in _AGGREGATE_REGRESSION_DIMENSIONS
    if container_key == "fold_metrics":
        return key in _EXPERIMENT_FOLD_KEYS
    if container_key == "class_metrics":
        return key in _AGGREGATE_CLASS_DIMENSIONS
    if container_key in {"field_metrics", "per_field_metrics"}:
        return key in _AGGREGATE_FIELD_DIMENSIONS
    return False


def _validate_aggregate_container(
    value: Any,
    *,
    container_key: str,
    path: str,
) -> None:
    if not isinstance(value, Mapping):
        _raise_aggregate_schema(path, "aggregate metric container must be an object")
    if len(value) > _MAX_AGGREGATE_CONTAINER_ENTRIES:
        _raise_aggregate_schema(path, "aggregate metric container is too large")
    for raw_key, child in value.items():
        key = str(raw_key).strip()
        normalized = key.casefold()
        child_path = f"{path}.{key}"
        _validate_dimension_name(key, path=child_path)
        if not _dimension_is_semantic(container_key, normalized):
            _raise_aggregate_schema(
                child_path,
                "dimension is not in the semantic aggregate schema",
            )
        if container_key in _AGGREGATE_NESTED_METRIC_CONTAINERS:
            if not isinstance(child, Mapping):
                _raise_aggregate_schema(
                    child_path, "nested metric dimensions must contain metric objects"
                )
            if len(child) > _MAX_AGGREGATE_NESTED_METRICS:
                _raise_aggregate_schema(
                    child_path, "nested metric container is too large"
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
    if len(value) > _MAX_AGGREGATE_ROOT_ENTRIES:
        _raise_aggregate_schema("$", "aggregate root contains too many entries")
    if _aggregate_scalar_value_count(value) > _MAX_AGGREGATE_SCALAR_VALUES:
        _raise_aggregate_schema(
            "$", "aggregate evidence contains too many scalar values"
        )
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
            if not child or len(child) > _MAX_AGGREGATE_SEQUENCE_VALUES:
                _raise_aggregate_schema(
                    path, "fold vectors must be non-empty and bounded"
                )
            for item in child:
                _validate_aggregate_scalar(normalized, item, path=path)
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


class _CandidateStateAuthorization:
    """Opaque, identity-checked one-process adoption capability."""

    __slots__ = ("__weakref__",)


_CANDIDATE_STATE_AUTHORIZATIONS: weakref.WeakKeyDictionary[
    _CandidateStateAuthorization,
    tuple[str, str, str],
] = weakref.WeakKeyDictionary()


class _ProtectedAccessAuthorization:
    """Opaque, identity-checked one-process protected-access capability."""

    __slots__ = ("__weakref__",)


_PROTECTED_ACCESS_AUTHORIZATIONS: weakref.WeakKeyDictionary[
    _ProtectedAccessAuthorization,
    tuple[str, int, str, str, str],
] = weakref.WeakKeyDictionary()


def _verify_candidate_state_record(
    ledger_path: Path | str,
    *,
    authorization: object,
    record_hash: str,
    candidate_sha256: str,
) -> None:
    """Verify the exact PASSED promotion-gate record used for adoption."""

    normalized_hash = _require_sha256(
        "candidate_state_record_hash", record_hash
    )
    try:
        resolved_ledger_path = str(Path(ledger_path).resolve(strict=True))
    except (OSError, RuntimeError) as exc:
        raise IntegrityError(
            "candidate state ledger path could not be resolved"
        ) from exc
    bound = (
        _CANDIDATE_STATE_AUTHORIZATIONS.get(authorization)
        if isinstance(authorization, _CandidateStateAuthorization)
        else None
    )
    if bound != (
        candidate_sha256,
        resolved_ledger_path,
        normalized_hash,
    ):
        raise IntegrityError(
            "candidate adoption lacks a live CandidatePromotionGate capability"
        )
    records = CanonicalHashChainStore(ledger_path).verify()
    matching = [
        record for record in records if record["record_hash"] == normalized_hash
    ]
    if len(matching) != 1:
        raise IntegrityError(
            "candidate state record hash is not present exactly once"
        )
    payload = matching[0]["payload"]
    if (
        set(payload) != _CANDIDATE_ASSESSMENT_KEYS
        or payload.get("event") != "candidate_assessment"
        or payload.get("decision") != "PASSED"
        or payload.get("candidate_sha256") != candidate_sha256
        or not isinstance(payload.get("assessment_id"), str)
        or not _SAFE_DIMENSION_RE.fullmatch(payload["assessment_id"])
        or not isinstance(payload.get("candidate_id"), str)
        or not _SAFE_DIMENSION_RE.fullmatch(payload["candidate_id"])
    ):
        raise IntegrityError(
            "candidate state record is not a bound PASSED assessment"
        )
    aggregate_evidence = payload["aggregate_evidence"]
    if not isinstance(aggregate_evidence, Mapping):
        raise IntegrityError(
            "candidate state record lacks aggregate promotion evidence"
        )
    try:
        require_aggregate_only(aggregate_evidence)
    except LeakageError as exc:
        raise IntegrityError(
            "candidate state promotion evidence is not aggregate-only"
        ) from exc
    gate_results = aggregate_evidence.get("gate_results")
    if (
        aggregate_evidence.get("promotion_gate_verified") is not True
        or not isinstance(gate_results, Mapping)
        or set(gate_results) != _PROMOTION_GATE_RESULT_KEYS
        or any(value is not True for value in gate_results.values())
        or any(
            aggregate_evidence.get(name) is not True
            for name in (
                "access_authorized",
                "baseline_verified",
                "deterministic",
                "fold_consistent",
            )
        )
        or any(
            aggregate_evidence.get(name) != 0
            for name in (
                "false_approvals",
                "hard_gate_failure_count",
                "invalid_records",
                "leakage_finding_count",
                "missing_records",
                "regression_waiver_count",
            )
        )
    ):
        raise IntegrityError(
            "candidate state record did not pass every promotion gate"
        )
    regression_counts = aggregate_evidence.get("regression_counts")
    if (
        not isinstance(regression_counts, Mapping)
        or regression_counts.get("golden") != 0
        or regression_counts.get("adversarial") != 0
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != 0
            for value in regression_counts.values()
        )
    ):
        raise IntegrityError(
            "candidate state record must clear golden and adversarial regressions"
        )


def _verify_protected_access_record(
    ledger_path: Path | str,
    *,
    authorization: object,
    expected_binding_sha256: str,
    record_hash: str,
    candidate_sha256: str,
    expected_aggregate_result: Mapping[str, Any],
) -> None:
    """Verify one exact budgeted protected access for the candidate digest."""

    normalized_hash = _require_sha256(
        "protected_access_record_hash", record_hash
    )
    normalized_binding = _require_sha256(
        "protected_access_binding_sha256",
        expected_binding_sha256,
    )
    try:
        resolved_ledger_path = str(Path(ledger_path).resolve(strict=True))
    except (OSError, RuntimeError) as exc:
        raise IntegrityError(
            "protected access ledger path could not be resolved"
        ) from exc
    records = CanonicalHashChainStore(ledger_path).verify()
    if not records:
        raise IntegrityError("protected access ledger is empty")
    configuration = records[0]["payload"]
    maximum_accesses = configuration.get("maximum_accesses")
    if (
        configuration.get("event")
        != ProtectedAccessBudget.CONFIGURATION_EVENT
        or set(configuration) != {"event", "maximum_accesses"}
        or isinstance(maximum_accesses, bool)
        or not isinstance(maximum_accesses, int)
        or not 1 <= maximum_accesses <= _MAX_AGGREGATE_COUNT
    ):
        raise IntegrityError(
            "protected access ledger configuration is invalid"
        )
    actual_binding = protected_access_binding(
        ledger_path,
        maximum_accesses=maximum_accesses,
    )
    bound = (
        _PROTECTED_ACCESS_AUTHORIZATIONS.get(authorization)
        if isinstance(authorization, _ProtectedAccessAuthorization)
        else None
    )
    if bound != (
        resolved_ledger_path,
        maximum_accesses,
        actual_binding,
        normalized_hash,
        candidate_sha256,
    ) or actual_binding != normalized_binding:
        raise IntegrityError(
            "protected result lacks its preregistered live budget capability"
        )
    access_records = records[1:]
    if len(access_records) > maximum_accesses:
        raise IntegrityError("protected access ledger exceeds its budget")
    for record in access_records:
        payload = record["payload"]
        if (
            set(payload) != _PROTECTED_ACCESS_KEYS
            or payload.get("event") != "protected_access"
            or not isinstance(payload.get("access_id"), str)
            or not _SAFE_DIMENSION_RE.fullmatch(payload["access_id"])
            or not isinstance(payload.get("purpose"), str)
            or not _SAFE_DIMENSION_RE.fullmatch(payload["purpose"])
            or not isinstance(payload.get("candidate_sha256"), str)
            or _SHA256_RE.fullmatch(payload["candidate_sha256"]) is None
            or not isinstance(payload.get("aggregate_result"), Mapping)
        ):
            raise IntegrityError("protected access ledger record is invalid")
        try:
            require_aggregate_only(payload["aggregate_result"])
        except LeakageError as exc:
            raise IntegrityError(
                "protected access record is not aggregate-only"
            ) from exc
    matching = [
        record
        for record in access_records
        if record["record_hash"] == normalized_hash
    ]
    if (
        len(matching) != 1
        or matching[0]["payload"]["candidate_sha256"] != candidate_sha256
        or matching[0]["payload"]["aggregate_result"]
        != dict(expected_aggregate_result)
    ):
        raise IntegrityError(
            "protected access record is not bound to the candidate result"
        )


def protected_access_summary(
    experiment_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact aggregate projection authorized by protected access."""

    if not isinstance(experiment_evidence, Mapping):
        raise ExperimentControlError(
            "protected experiment evidence must be an object"
        )
    metrics = experiment_evidence.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ExperimentControlError(
            "protected experiment evidence lacks metrics"
        )
    metric_keys = (
        "baseline_total_score",
        "calibration_score",
        "catastrophic_false_approvals",
        "classification_score",
        "duplicate_records",
        "extra_records",
        "extraction_score",
        "invalid_records",
        "missing_records",
        "record_count",
        "score_delta",
        "total_score",
    )
    try:
        summary = {
            "baseline_artifact_sha256": experiment_evidence[
                "baseline_artifact_sha256"
            ],
            "candidate_artifact_sha256": experiment_evidence[
                "candidate_artifact_sha256"
            ],
            "evidence_label": experiment_evidence["evidence_label"],
            "metrics": {key: metrics[key] for key in metric_keys},
            "runtime_evidence_sha256": experiment_evidence[
                "runtime_evidence_sha256"
            ],
            "split_manifest_sha256": experiment_evidence[
                "split_manifest_sha256"
            ],
        }
    except KeyError as exc:
        raise ExperimentControlError(
            "protected experiment evidence is incomplete"
        ) from exc
    require_aggregate_only(summary)
    return json.loads(canonical_json(summary))


class ExperimentLedger:
    """Append aggregate-only evidence under immutable two-stage contracts."""

    def __init__(self, path: Path | str) -> None:
        self.store = CanonicalHashChainStore(path)

    def experiments(self) -> tuple[dict[str, Any], ...]:
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
        del experiment_id, evidence, expected_head
        raise ExperimentControlError(
            "legacy one-stage experiment writes are disabled; "
            "use preregister() and record_result()"
        )

    def preregister(
        self,
        experiment_id: str,
        plan: Mapping[str, Any],
        *,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        """Persist the exact plan before any candidate execution."""

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

        def check(
            records: tuple[dict[str, Any], ...],
            requested: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            matching = [
                record
                for record in records
                if record["payload"].get("experiment_id") == experiment_id
            ]
            plans = [
                record
                for record in matching
                if record["payload"].get("event") == "experiment_plan"
            ]
            results = [
                record
                for record in matching
                if record["payload"].get("event") == "experiment_result"
            ]
            unexpected = [
                record
                for record in matching
                if record["payload"].get("event")
                not in {"experiment_plan", "experiment_result"}
            ]
            if unexpected or len(plans) > 1 or len(results) > 1:
                raise IntegrityError(
                    f"experiment_id is not globally unique: {experiment_id}"
                )
            if plans:
                if plans[0]["payload"] != requested:
                    raise ExperimentControlError(
                        "experiment plan conflicts with immutable "
                        f"preregistration: {experiment_id}"
                    )
                return plans[0]
            if matching:
                raise IntegrityError(
                    f"experiment result exists without its plan: {experiment_id}"
                )
            if (
                expected_head is not None
                and self.store._head(records) != expected_head
            ):
                raise CompareAndSwapError(
                    "experiment plan CAS failed: "
                    f"expected {expected_head}, got {self.store._head(records)}"
                )
            return None

        return self.store.append_transactional(
            payload,
            expected_head=None,
            locked_check=check,
        )

    def record_result(
        self,
        experiment_id: str,
        evidence: Mapping[str, Any],
        *,
        decision: str,
        rationale: str,
        input_dir: Path | str,
        split_manifest_path: Path | str,
        candidate_state_ledger_path: Path | str | None = None,
        candidate_state_authorization: object | None = None,
        protected_access_ledger_path: Path | str | None = None,
        protected_access_authorization: object | None = None,
        expected_head: str | None = None,
    ) -> dict[str, Any]:
        """Persist one aggregate result bound to a prior immutable plan."""

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

        records = self.store.verify()
        plans = [
            record
            for record in records
            if record["payload"].get("event") == "experiment_plan"
            and record["payload"].get("experiment_id") == experiment_id
        ]
        if len(plans) != 1:
            raise ExperimentControlError(
                "experiment result requires exactly one prior plan"
            )
        plan_record = plans[0]
        plan = plan_record["payload"]["plan"]
        expected_folds = _verify_experiment_split_manifest(
            split_manifest_path,
            input_dir=input_dir,
            expected_input_tree_sha256=plan["input_tree_sha256"],
            expected_sha256=plan["split_manifest_sha256"],
            expected_record_count=plan["expected_record_count"],
        )
        normalized_evidence = _normalize_experiment_result(
            evidence,
            plan=plan,
            decision=normalized_decision,
            expected_folds=expected_folds,
        )
        if normalized_decision == "adopt":
            if candidate_state_ledger_path is None:
                raise ExperimentControlError(
                    "adoption requires a candidate state ledger path"
                )
            _verify_candidate_state_record(
                candidate_state_ledger_path,
                authorization=candidate_state_authorization,
                record_hash=normalized_evidence[
                    "candidate_state_record_hash"
                ],
                candidate_sha256=normalized_evidence[
                    "candidate_artifact_sha256"
                ],
            )
        elif (
            candidate_state_ledger_path is not None
            or candidate_state_authorization is not None
        ):
            raise ExperimentControlError(
                "candidate state authorization is valid only for adoption"
            )
        if normalized_evidence["evidence_label"] == "protected":
            if protected_access_ledger_path is None:
                raise ExperimentControlError(
                    "protected evidence requires an access ledger path"
                )
            _verify_protected_access_record(
                protected_access_ledger_path,
                authorization=protected_access_authorization,
                expected_binding_sha256=plan[
                    "protected_access_binding_sha256"
                ],
                record_hash=normalized_evidence[
                    "protected_access_record_hash"
                ],
                candidate_sha256=normalized_evidence[
                    "candidate_artifact_sha256"
                ],
                expected_aggregate_result=protected_access_summary(
                    normalized_evidence
                ),
            )
        elif (
            protected_access_ledger_path is not None
            or protected_access_authorization is not None
        ):
            raise ExperimentControlError(
                "protected access authorization requires protected evidence"
            )
        payload = {
            "decision": normalized_decision,
            "event": "experiment_result",
            "evidence": normalized_evidence,
            "experiment_id": experiment_id,
            "plan_record_hash": plan_record["record_hash"],
            "rationale": normalized_rationale,
        }

        def check(
            locked_records: tuple[dict[str, Any], ...],
            requested: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            matching = [
                record
                for record in locked_records
                if record["payload"].get("experiment_id") == experiment_id
            ]
            locked_plans = [
                record
                for record in matching
                if record["payload"].get("event") == "experiment_plan"
            ]
            locked_results = [
                record
                for record in matching
                if record["payload"].get("event") == "experiment_result"
            ]
            unexpected = [
                record
                for record in matching
                if record["payload"].get("event")
                not in {"experiment_plan", "experiment_result"}
            ]
            if (
                len(locked_plans) != 1
                or locked_plans[0]["record_hash"]
                != requested["plan_record_hash"]
                or unexpected
            ):
                raise IntegrityError(
                    f"experiment plan binding changed: {experiment_id}"
                )
            if len(locked_results) > 1:
                raise IntegrityError(
                    f"experiment has multiple results: {experiment_id}"
                )
            if locked_results:
                if locked_results[0]["payload"] != requested:
                    raise ExperimentControlError(
                        f"experiment result conflicts with recorded result: {experiment_id}"
                    )
                return locked_results[0]
            if (
                expected_head is not None
                and self.store._head(locked_records) != expected_head
            ):
                raise CompareAndSwapError(
                    "experiment result CAS failed: "
                    f"expected {expected_head}, got {self.store._head(locked_records)}"
                )
            return None

        return self.store.append_transactional(
            payload,
            expected_head=None,
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
        self._issued_authorizations: dict[
            str, _ProtectedAccessAuthorization
        ] = {}
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

        record = self.store.append_transactional(
            requested,
            expected_head=expected_head,
            locked_check=check,
        )
        authorization = _ProtectedAccessAuthorization()
        resolved_path = str(self.store.path.resolve(strict=True))
        binding = protected_access_binding(
            self.store.path,
            maximum_accesses=self.maximum_accesses,
        )
        _PROTECTED_ACCESS_AUTHORIZATIONS[authorization] = (
            resolved_path,
            self.maximum_accesses,
            binding,
            record["record_hash"],
            candidate_sha256,
        )
        self._issued_authorizations[access_id] = authorization
        return dict(record["payload"])

    def authorization_for(self, access_id: str) -> object:
        """Return the live capability issued with one recorded access."""

        normalized = str(access_id).strip()
        authorization = self._issued_authorizations.get(normalized)
        if authorization is None:
            raise ExperimentControlError(
                "no live protected access authorization exists for this access"
            )
        return authorization


def protected_access_binding(
    ledger_path: Path | str,
    *,
    maximum_accesses: int,
) -> str:
    """Bind an immutable protected budget to its exact canonical path."""

    try:
        resolved_path = str(Path(ledger_path).resolve(strict=True))
    except (OSError, RuntimeError) as exc:
        raise IntegrityError(
            "protected access ledger path could not be resolved"
        ) from exc
    records = CanonicalHashChainStore(ledger_path).verify()
    configuration = ProtectedAccessBudget._validate_configuration(
        records,
        maximum_accesses=maximum_accesses,
    )
    if configuration is None:
        raise IntegrityError(
            "protected access ledger lacks its configuration"
        )
    payload = {
        "configuration_record_hash": configuration["record_hash"],
        "ledger_path": resolved_path,
        "maximum_accesses": maximum_accesses,
    }
    return hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()


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
        self._issued_authorizations: dict[
            str, _CandidateStateAuthorization
        ] = {}

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
        persisted = self.state._persist_assessment(
            assessment_id,
            candidate_id=candidate_id,
            candidate_sha256=candidate_sha256,
            decision=decision,
            aggregate_evidence=evidence,
            expected_head=expected_head,
            authority=_PROMOTION_GATE_AUTHORITY,
        )
        if persisted["decision"] == "PASSED":
            matching_records = [
                record
                for record in self.state.store.verify()
                if record["payload"] == persisted
                and record["payload"].get("assessment_id")
                == str(assessment_id).strip()
            ]
            if len(matching_records) != 1:
                raise IntegrityError(
                    "PASSED assessment could not be bound to its exact record"
                )
            try:
                resolved_path = str(
                    self.state.store.path.resolve(strict=True)
                )
            except (OSError, RuntimeError) as exc:
                raise IntegrityError(
                    "candidate state ledger could not be bound"
                ) from exc
            authorization = _CandidateStateAuthorization()
            _CANDIDATE_STATE_AUTHORIZATIONS[authorization] = (
                candidate_sha256,
                resolved_path,
                matching_records[0]["record_hash"],
            )
            self._issued_authorizations[candidate_sha256] = authorization
        return persisted

    def authorization_for(self, candidate_sha256: str) -> object:
        """Return the live capability issued by this gate for one PASSED digest."""

        normalized = _require_sha256(
            "candidate_sha256", candidate_sha256
        )
        authorization = self._issued_authorizations.get(normalized)
        if authorization is None:
            raise ExperimentControlError(
                "no live PASSED authorization exists for this candidate"
            )
        return authorization
