#!/usr/bin/env python3
"""Build revision-bound, aggregate-only WO-18 comparison evidence.

Identity-bearing truth, role membership, layout groups, and out-of-fold
predictions remain external.  Each of the exact four comparison arms supplies
three complete out-of-fold prediction files and a fit attestation for all
three-by-five grouped folds.  This builder verifies every byte binding, uses
the official evaluator for every repeat/fold/protected-role slice, scans the
runtime for identity leakage, and emits only the aggregate promotion result.

The local 32-case corpus is public-exposed robustness evidence, not an unseen
holdout and not the unavailable full-public 1,000-case evaluation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.decision_recovery_gate import (  # noqa: E402
    APPROACHES,
    CANDIDATE_APPROACH,
    CONTROL_APPROACH,
    PROTECTED_ROLES,
    REQUIRED_FOLDS,
    REQUIRED_REPEATS,
    ApproachRunAggregate,
    DecisionRecoveryContractAudit,
    DecisionRecoveryEvidence,
    DecisionRecoveryGate,
    ProtectedRoleScore,
    ScoreSlice,
)
from devtools.decision_recovery_cv import (  # noqa: E402
    ARM_MANIFEST_SCHEMA,
    ComparisonExecution,
    approach_graph_sha256,
    execute_grouped_oof,
    execution_fingerprint,
    load_frozen_feature_rows,
    runner_source_sha256,
)
from devtools.decision_recovery_contract_probe import (  # noqa: E402
    contract_fixture_payload,
    run_contract_probes,
)
from devtools.experiment_control import (  # noqa: E402
    ExperimentControlError,
    RepeatedGroupedSplitManager,
    RuntimeLeakageScanner,
    canonical_json,
    require_aggregate_only,
)
from devtools.grouped_fusion_evidence import (  # noqa: E402
    _safety_event_case_ids,
    verify_clean_candidate_checkout,
)
from devtools.fusion_audit_run import case_id_set_sha256  # noqa: E402
from devtools.grouped_recovery_evidence import (  # noqa: E402
    FrozenLayoutManifest,
    _filter_rows,
    _invalid_record_count,
    _read_json_object,
    _read_submission,
    _read_truth_subset,
    load_layout_manifest,
)
from devtools.ocr_ablation import _input_tree_sha256  # noqa: E402
from devtools.wo18_production_capture import (  # noqa: E402
    CAPTURE_OBSERVATION_SCHEMA,
    producer_graph_sha256,
)
from scripts import evaluate as official_evaluate  # noqa: E402


PROTECTED_ROLE_MANIFEST_SCHEMA = "mib-wo18-protected-roles/v1"
CONTRACT_AUDIT_SCHEMA = "mib-wo18-contract-audit/v1"
REQUIRED_COHORT_RECORDS = 32
PROTECTED_ROLE_TAXONOMY = {
    "binding_authority": "binding_authoritative_signed_decision",
    "visible_disqualifier": "high_precision_visible_disqualifying_evidence",
    "approval_guard": "complete_clean_visible_approval_recovery_eligible",
    "denial_guard": "explicit_visible_violation_or_binding_denial_eligible",
    "uncertainty": "residual_uncertainty_requires_needs_review",
}
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_FEATURE_RE = re.compile(r"[a-z][a-z0-9_]{0,79}")
_FORBIDDEN_FEATURE_PATTERNS = (
    re.compile(r"(?:^|_)case_(?:id|number|numeric)(?:_|$)"),
    re.compile(r"(?:^|_)(?:file|filename|path|pdf)(?:_|$)"),
    re.compile(r"(?:^|_)(?:hash|digest|sha256?)(?:_|$)"),
    re.compile(r"(?:^|_)applicant_(?:name|identity)(?:_|$)"),
    re.compile(r"(?:^|_)(?:raw_)?(?:label|truth)(?:_|$)"),
    re.compile(r"(?:^|_)(?:lookup|memorized|per_case)(?:_|$)"),
    re.compile(r"(?:^|_)(?:leaderboard|social)(?:_|$)"),
    re.compile(r"(?:^|_)hidden_(?:text|content)(?:_|$)"),
    re.compile(r"(?:^|_)(?:document_)?order(?:_|$)"),
)


class DecisionRecoveryEvidenceBuildError(ExperimentControlError):
    """An external WO-18 evidence input is incomplete or malformed."""


@dataclass(frozen=True)
class FrozenProtectedRoles:
    by_role: Mapping[str, tuple[str, ...]]
    sha256: str
    frozen_before_scoring: bool


@dataclass(frozen=True)
class ArmRun:
    repeat: int
    rows: tuple[Mapping[str, Any], ...]
    path: Path
    sha256: str
    rerun_path: Path
    rerun_sha256: str
    deterministic: bool


@dataclass(frozen=True)
class ArmEvidence:
    approach: str
    runs: tuple[ArmRun, ...]
    manifest_sha256: str
    model_artifact_paths: tuple[Path, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    normalized = str(value).strip().casefold()
    if not _SHA256_RE.fullmatch(normalized):
        raise DecisionRecoveryEvidenceBuildError(
            f"{name} must be a full SHA-256 digest"
        )
    return normalized


def _require_revision(value: Any) -> str:
    normalized = str(value).strip().casefold()
    if not _GIT_COMMIT_RE.fullmatch(normalized):
        raise DecisionRecoveryEvidenceBuildError(
            "source revision must be a full Git commit SHA"
        )
    return normalized


def _hash_string_set(values: Sequence[str]) -> str:
    return _sha256_bytes(
        canonical_json(sorted(str(value) for value in values)).encode("utf-8")
    )


def _feature_schema_hash(feature_names: Sequence[str]) -> str:
    return _sha256_bytes(
        canonical_json(list(feature_names)).encode("utf-8")
    )


PROTECTED_ROLE_TAXONOMY_SHA256 = _sha256_bytes(
    canonical_json(PROTECTED_ROLE_TAXONOMY).encode("utf-8")
)


def _runtime_feature_names() -> tuple[str, ...]:
    """Load the fixed production schema without importing any model artifact."""

    try:
        module = importlib.import_module("mib_pipeline.model_recovery")
        raw = getattr(module, "FEATURE_NAMES")
    except (ImportError, AttributeError) as exc:
        raise DecisionRecoveryEvidenceBuildError(
            "production FEATURE_NAMES could not be loaded"
        ) from exc
    if not isinstance(raw, (tuple, list)):
        raise DecisionRecoveryEvidenceBuildError(
            "production FEATURE_NAMES must be an ordered sequence"
        )
    names = tuple(str(value).strip() for value in raw)
    if not names or len(set(names)) != len(names):
        raise DecisionRecoveryEvidenceBuildError(
            "production FEATURE_NAMES must be non-empty and unique"
        )
    return names


def _feature_schema_findings(feature_names: Sequence[str]) -> tuple[str, ...]:
    findings: list[str] = []
    for name in feature_names:
        if not _SAFE_FEATURE_RE.fullmatch(name):
            findings.append("invalid_feature_name")
            continue
        if any(pattern.search(name) for pattern in _FORBIDDEN_FEATURE_PATTERNS):
            findings.append("forbidden_feature_name")
    return tuple(findings)


def _model_specific_source_findings(path: Path) -> tuple[str, ...]:
    """Reject direct identity/document-key reads missed by the generic scan."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return ("unscannable_model_source",)
    always_forbidden_attributes = {
        "applicant_name",
        "filename",
        "stem",
        "path",
        "pdf_hash",
        "document_order",
    }
    scope_only_attributes = {
        "case_id",
        "active_applicant",
    }
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def scope_use_allowed(node: ast.Attribute) -> bool:
        current: ast.AST = node
        for _ in range(5):
            parent = parents.get(current)
            if parent is None:
                return False
            if isinstance(parent, ast.Compare):
                return True
            if isinstance(parent, ast.keyword) and parent.arg in {
                "expected_case_id",
                "active_applicant",
            }:
                return True
            if isinstance(
                parent,
                (ast.Dict, ast.List, ast.Tuple, ast.Return, ast.Assign),
            ):
                return False
            current = parent
        return False

    findings: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in always_forbidden_attributes
        ):
            findings.append(f"forbidden_attribute_{node.attr}")
        elif (
            isinstance(node, ast.Attribute)
            and node.attr in scope_only_attributes
            and not scope_use_allowed(node)
        ):
            findings.append(f"predictive_identity_access_{node.attr}")
    return tuple(findings)


def load_protected_role_manifest(
    path: Path | str,
    *,
    expected_layout_manifest_sha256: str,
    expected_case_ids: Sequence[str],
    expected_group_by_case: Mapping[str, str],
) -> FrozenProtectedRoles:
    """Validate a pre-scoring, identity-bearing role manifest."""

    manifest_path = Path(path)
    raw = _read_json_object(manifest_path, label="protected role manifest")
    if set(raw) != {
        "schema_version",
        "frozen_before_scoring",
        "layout_manifest_sha256",
        "role_taxonomy_sha256",
        "cases",
    }:
        raise DecisionRecoveryEvidenceBuildError(
            "protected role manifest must contain the exact contract keys"
        )
    if raw.get("schema_version") != PROTECTED_ROLE_MANIFEST_SCHEMA:
        raise DecisionRecoveryEvidenceBuildError(
            "protected role manifest schema is unsupported"
        )
    if raw.get("frozen_before_scoring") is not True:
        raise DecisionRecoveryEvidenceBuildError(
            "protected roles must be frozen before scoring"
        )
    if (
        _require_sha256(
            "protected role taxonomy", raw.get("role_taxonomy_sha256")
        )
        != PROTECTED_ROLE_TAXONOMY_SHA256
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "protected roles do not use the fixed WO-18 taxonomy"
        )
    if (
        _require_sha256(
            "protected role layout binding",
            raw.get("layout_manifest_sha256"),
        )
        != _require_sha256(
            "expected layout manifest",
            expected_layout_manifest_sha256,
        )
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "protected roles do not bind the frozen layout manifest"
        )
    rows = raw.get("cases")
    if not isinstance(rows, list) or not rows:
        raise DecisionRecoveryEvidenceBuildError(
            "protected role manifest cases must be a non-empty list"
        )
    by_role: dict[str, list[str]] = {role: [] for role in PROTECTED_ROLES}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "case_id",
            "protected_roles",
        }:
            raise DecisionRecoveryEvidenceBuildError(
                f"protected role row {index} has an invalid schema"
            )
        case_id = str(row.get("case_id", "")).strip()
        roles = row.get("protected_roles")
        if (
            not case_id
            or case_id in seen
            or not isinstance(roles, list)
            or not roles
            or len(roles) != 1
            or any(role not in PROTECTED_ROLES for role in roles)
        ):
            raise DecisionRecoveryEvidenceBuildError(
                "every case needs unique, recognized protected roles"
            )
        seen.add(case_id)
        for role in roles:
            by_role[str(role)].append(case_id)
    if seen != set(expected_case_ids):
        raise DecisionRecoveryEvidenceBuildError(
            "protected role membership must cover the exact frozen cohort"
        )
    if any(len(case_ids) < 2 for case_ids in by_role.values()):
        raise DecisionRecoveryEvidenceBuildError(
            "every required protected role must have non-vacuous support"
        )
    if any(
        len(
            {
                expected_group_by_case[case_id]
                for case_id in case_ids
            }
        )
        < 2
        for case_ids in by_role.values()
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "every protected role must span at least two layout groups"
        )
    return FrozenProtectedRoles(
        by_role={
            role: tuple(sorted(case_ids))
            for role, case_ids in by_role.items()
        },
        sha256=_sha256_file(manifest_path),
        frozen_before_scoring=True,
    )


def _validate_protected_role_semantics(
    *,
    protected_roles: FrozenProtectedRoles,
    frozen_features: Any,
    truth: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require every manually frozen role to satisfy its fixed semantics."""

    role_by_case = {
        case_id: role
        for role, case_ids in protected_roles.by_role.items()
        for case_id in case_ids
    }
    for case_id, role in role_by_case.items():
        values = frozen_features.cases[case_id].features.values
        truth_decision = str(
            truth[case_id].get("adjudication", "")
        ).strip().upper()
        binding = any(
            values[name] == 1.0
            for name in (
                "binding_approval",
                "binding_denial",
                "binding_review",
            )
        )
        explicit_violation = values["policy_explicit_violation"] == 1.0
        approval_eligible = (
            truth_decision == "APPROVED"
            and all(
                values[name] == 1.0
                for name in (
                    "resolved_fraction",
                    "visible_fraction",
                    "exact_case_scope_fraction",
                    "exact_subject_scope_fraction",
                    "clean_fraction",
                    "link_confidence",
                )
            )
            and all(
                values[name] == 0.0
                for name in (
                    "unresolved_linkage",
                    "packet_conflict",
                    "packet_watermark",
                    "policy_explicit_violation",
                    "policy_review_gap",
                    "policy_review_conflict",
                    "policy_review_visibility",
                    "policy_review_waiver",
                    "policy_review_other",
                )
            )
        )
        denial_eligible = truth_decision == "DENIED" and (
            explicit_violation or values["binding_denial"] == 1.0
        )
        uncertain = (
            truth_decision == "NEEDS_REVIEW"
            or values["unresolved_linkage"] == 1.0
            or values["packet_conflict"] == 1.0
            or values["packet_watermark"] == 1.0
            or values["unknown_fraction"] > 0.0
            or values["contested_fraction"] > 0.0
        )
        valid = {
            "binding_authority": binding,
            "visible_disqualifier": explicit_violation,
            "approval_guard": approval_eligible,
            "denial_guard": denial_eligible,
            "uncertainty": uncertain,
        }[role]
        if not valid:
            raise DecisionRecoveryEvidenceBuildError(
                "protected role assignment does not match freshly bound "
                "feature/baseline/truth semantics"
            )


def _validate_prediction_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    expected_case_ids: Sequence[str],
    label: str,
) -> None:
    predictions, duplicates, blank_rows = official_evaluate.index_submission(
        rows
    )
    expected = set(expected_case_ids)
    if duplicates or blank_rows:
        raise DecisionRecoveryEvidenceBuildError(
            f"{label} contains duplicate or blank case identity"
        )
    if set(predictions) != expected:
        raise DecisionRecoveryEvidenceBuildError(
            f"{label} must cover the exact frozen cohort"
        )


def _expected_fold_attestations(
    manifest: FrozenLayoutManifest,
) -> Mapping[tuple[int, int], Mapping[str, Any]]:
    manager = RepeatedGroupedSplitManager(
        seed=manifest.split_seed,
        repeats=REQUIRED_REPEATS,
        folds=REQUIRED_FOLDS,
    )
    splits = manager.split_groups(manifest.groups)
    return {
        (split.repeat, split.fold): {
            "fold_index": split.fold,
            "training_groups_sha256": _hash_string_set(
                split.tuning_groups
            ),
            "validation_groups_sha256": _hash_string_set(
                split.validation_groups
            ),
            "training_case_set_sha256": _hash_string_set(
                split.tuning_case_ids
            ),
            "validation_case_set_sha256": _hash_string_set(
                split.validation_case_ids
            ),
        }
        for split in splits
    }


def load_arm_manifest(
    path: Path | str,
    *,
    expected_approach: str,
    expected_source_revision_sha: str,
    layout_manifest: FrozenLayoutManifest,
    expected_role_manifest_sha256: str,
    expected_truth_sha256: str,
    expected_input_tree_sha256: str,
    expected_feature_schema_sha256: str,
    expected_feature_rows_sha256: str,
    expected_evaluator_sha256: str,
    expected_execution: ComparisonExecution,
) -> ArmEvidence:
    """Load one exact, three-repeat OOF arm and verify every fit/file pin."""

    manifest_path = Path(path).resolve()
    raw = _read_json_object(manifest_path, label=f"{expected_approach} arm")
    if set(raw) != {
        "schema_version",
        "approach",
        "source_revision_sha",
        "layout_manifest_sha256",
        "protected_role_manifest_sha256",
        "truth_sha256",
        "input_tree_sha256",
        "feature_schema_sha256",
        "frozen_feature_rows_sha256",
        "official_evaluator_sha256",
        "runner_source_sha256",
        "approach_graph_sha256",
        "runs",
    }:
        raise DecisionRecoveryEvidenceBuildError(
            f"{expected_approach} arm must contain the exact contract keys"
        )
    if raw.get("schema_version") != ARM_MANIFEST_SCHEMA:
        raise DecisionRecoveryEvidenceBuildError(
            f"{expected_approach} arm schema is unsupported"
        )
    if raw.get("approach") != expected_approach:
        raise DecisionRecoveryEvidenceBuildError(
            "arm approach does not match its CLI assignment"
        )
    bindings = (
        (
            "source revision",
            _require_revision(raw.get("source_revision_sha")),
            _require_revision(expected_source_revision_sha),
        ),
        (
            "layout manifest",
            _require_sha256(
                "arm layout manifest", raw.get("layout_manifest_sha256")
            ),
            layout_manifest.sha256,
        ),
        (
            "protected role manifest",
            _require_sha256(
                "arm protected roles",
                raw.get("protected_role_manifest_sha256"),
            ),
            _require_sha256(
                "expected protected roles",
                expected_role_manifest_sha256,
            ),
        ),
        (
            "truth",
            _require_sha256("arm truth", raw.get("truth_sha256")),
            _require_sha256("expected truth", expected_truth_sha256),
        ),
        (
            "input tree",
            _require_sha256(
                "arm input tree", raw.get("input_tree_sha256")
            ),
            _require_sha256(
                "expected input tree", expected_input_tree_sha256
            ),
        ),
        (
            "feature schema",
            _require_sha256(
                "arm feature schema", raw.get("feature_schema_sha256")
            ),
            _require_sha256(
                "expected feature schema",
                expected_feature_schema_sha256,
            ),
        ),
        (
            "frozen feature rows",
            _require_sha256(
                "arm frozen feature rows",
                raw.get("frozen_feature_rows_sha256"),
            ),
            _require_sha256(
                "expected frozen feature rows",
                expected_feature_rows_sha256,
            ),
        ),
        (
            "official evaluator",
            _require_sha256(
                "arm official evaluator",
                raw.get("official_evaluator_sha256"),
            ),
            _require_sha256(
                "expected official evaluator",
                expected_evaluator_sha256,
            ),
        ),
        (
            "runner source",
            _require_sha256(
                "arm runner source", raw.get("runner_source_sha256")
            ),
            runner_source_sha256(),
        ),
        (
            "approach graph",
            _require_sha256(
                "arm approach graph", raw.get("approach_graph_sha256")
            ),
            approach_graph_sha256(expected_approach),
        ),
    )
    for label, actual, expected in bindings:
        if actual != expected:
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} arm {label} binding mismatch"
            )

    raw_runs = raw.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != REQUIRED_REPEATS:
        raise DecisionRecoveryEvidenceBuildError(
            f"{expected_approach} must contain exactly three OOF runs"
        )
    expected_folds = _expected_fold_attestations(layout_manifest)
    runs: list[ArmRun] = []
    seen_repeats: set[int] = set()
    seen_prediction_paths: set[Path] = set()
    model_artifact_paths: set[Path] = set()
    subset_paths: set[Path] = set()
    expected_folds_by_key = {
        (fold.repeat, fold.fold): fold for fold in expected_execution.folds
    }
    for raw_run in raw_runs:
        if not isinstance(raw_run, Mapping) or set(raw_run) != {
            "repeat_index",
            "predictions_path",
            "predictions_sha256",
            "rerun_predictions_path",
            "rerun_predictions_sha256",
            "folds",
        }:
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} run has an invalid schema"
            )
        repeat = raw_run.get("repeat_index")
        if (
            isinstance(repeat, bool)
            or not isinstance(repeat, int)
            or repeat not in range(REQUIRED_REPEATS)
            or repeat in seen_repeats
        ):
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} repeat indexes must be 0, 1, and 2"
            )
        seen_repeats.add(repeat)
        prediction_path = Path(
            str(raw_run.get("predictions_path", ""))
        ).resolve()
        if (
            prediction_path in seen_prediction_paths
            or not prediction_path.is_file()
        ):
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} prediction artifacts must be distinct files"
            )
        seen_prediction_paths.add(prediction_path)
        prediction_sha = _require_sha256(
            "prediction SHA", raw_run.get("predictions_sha256")
        )
        if _sha256_file(prediction_path) != prediction_sha:
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} prediction byte binding mismatch"
            )
        rows = _read_submission(prediction_path)
        _validate_prediction_rows(
            rows=rows,
            expected_case_ids=layout_manifest.case_ids,
            label=f"{expected_approach} repeat {repeat}",
        )
        expected_run_rows = expected_execution.rows[expected_approach][repeat]
        if canonical_json([dict(row) for row in rows]) != canonical_json(
            [dict(row) for row in expected_run_rows]
        ):
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} repeat {repeat} is not the recomputed "
                "fixed-runner OOF output"
            )
        rerun_path = Path(
            str(raw_run.get("rerun_predictions_path", ""))
        ).resolve()
        if (
            rerun_path in seen_prediction_paths
            or not rerun_path.is_file()
            or rerun_path == prediction_path
        ):
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} rerun artifacts must be distinct files"
            )
        seen_prediction_paths.add(rerun_path)
        rerun_sha = _require_sha256(
            "rerun prediction SHA",
            raw_run.get("rerun_predictions_sha256"),
        )
        if _sha256_file(rerun_path) != rerun_sha:
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} rerun byte binding mismatch"
            )
        if prediction_sha != rerun_sha:
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} repeat {repeat} is not byte-deterministic"
            )
        rerun_rows = _read_submission(rerun_path)
        _validate_prediction_rows(
            rows=rerun_rows,
            expected_case_ids=layout_manifest.case_ids,
            label=f"{expected_approach} repeat {repeat} rerun",
        )
        fold_rows = raw_run.get("folds")
        if not isinstance(fold_rows, list) or len(fold_rows) != REQUIRED_FOLDS:
            raise DecisionRecoveryEvidenceBuildError(
                f"{expected_approach} repeat {repeat} needs five fit attestations"
            )
        normalized_folds: dict[int, Mapping[str, Any]] = {}
        for fold_row in fold_rows:
            if not isinstance(fold_row, Mapping) or set(fold_row) != {
                "fold_index",
                "training_groups_sha256",
                "validation_groups_sha256",
                "training_case_set_sha256",
                "validation_case_set_sha256",
                "fold_model_artifact_path",
                "fold_model_artifact_sha256",
                "validation_predictions_path",
                "validation_predictions_sha256",
            }:
                raise DecisionRecoveryEvidenceBuildError(
                    "fit attestation must be an object"
                )
            fold_index = fold_row.get("fold_index")
            if (
                isinstance(fold_index, bool)
                or not isinstance(fold_index, int)
                or fold_index in normalized_folds
            ):
                raise DecisionRecoveryEvidenceBuildError(
                    "fit attestation fold indexes must be unique integers"
                )
            normalized_folds[fold_index] = dict(fold_row)
        for fold in range(REQUIRED_FOLDS):
            fold_row = normalized_folds.get(fold)
            if fold_row is None:
                raise DecisionRecoveryEvidenceBuildError(
                    "fit attestation is missing a grouped fold"
                )
            expected_common = expected_folds[(repeat, fold)]
            if any(
                fold_row.get(name) != value
                for name, value in expected_common.items()
            ):
                raise DecisionRecoveryEvidenceBuildError(
                    f"{expected_approach} repeat {repeat} fold {fold} "
                    "does not prove group-exclusive fitting"
                )
            expected_fold = expected_folds_by_key[(repeat, fold)]
            model_path = Path(
                str(fold_row.get("fold_model_artifact_path", ""))
            ).resolve()
            subset_path = Path(
                str(fold_row.get("validation_predictions_path", ""))
            ).resolve()
            if (
                not model_path.is_file()
                or not subset_path.is_file()
                or subset_path in subset_paths
            ):
                raise DecisionRecoveryEvidenceBuildError(
                    "fold artifacts must be readable and validation subsets "
                    "must be distinct"
                )
            subset_paths.add(subset_path)
            model_sha = _require_sha256(
                "fold model artifact",
                fold_row.get("fold_model_artifact_sha256"),
            )
            subset_sha = _require_sha256(
                "fold validation predictions",
                fold_row.get("validation_predictions_sha256"),
            )
            if (
                _sha256_file(model_path) != model_sha
                or _sha256_file(subset_path) != subset_sha
            ):
                raise DecisionRecoveryEvidenceBuildError(
                    "fold artifact byte binding mismatch"
                )
            model_payload = _read_json_object(
                model_path, label="fold compact model"
            )
            if canonical_json(model_payload) != canonical_json(
                dict(expected_fold.model_artifact)
            ):
                raise DecisionRecoveryEvidenceBuildError(
                    "fold compact model was not fit from the frozen "
                    "training groups"
                )
            subset_rows = _read_submission(subset_path)
            expected_subset = expected_fold.validation_rows[
                expected_approach
            ]
            if canonical_json([dict(row) for row in subset_rows]) != (
                canonical_json([dict(row) for row in expected_subset])
            ):
                raise DecisionRecoveryEvidenceBuildError(
                    "fold validation subset is not the recomputed OOF output"
                )
            model_artifact_paths.add(model_path)
        runs.append(
            ArmRun(
                repeat=repeat,
                rows=tuple(rows),
                path=prediction_path,
                sha256=prediction_sha,
                rerun_path=rerun_path,
                rerun_sha256=rerun_sha,
                deterministic=(
                    prediction_path.read_bytes() == rerun_path.read_bytes()
                ),
            )
        )
    return ArmEvidence(
        approach=expected_approach,
        runs=tuple(sorted(runs, key=lambda value: value.repeat)),
        manifest_sha256=_sha256_file(manifest_path),
        model_artifact_paths=tuple(sorted(model_artifact_paths)),
    )


def _false_positive_denial_case_ids(
    truth: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    predictions, _, _ = official_evaluate.index_submission(rows)
    return frozenset(
        case_id
        for case_id, truth_row in truth.items()
        if case_id in predictions
        and str(
            predictions[case_id].get("adjudication", "")
        ).strip().upper()
        == "DENIED"
        and str(truth_row.get("adjudication", "")).strip().upper()
        != "DENIED"
    )


def _aggregate_run(
    truth: Mapping[str, Mapping[str, Any]],
    runs: Sequence[ArmRun],
) -> ApproachRunAggregate:
    aggregates = [
        official_evaluate.build_results(truth, run.rows)[0] for run in runs
    ]
    return ApproachRunAggregate(
        total_score=sum(
            float(value["scores"]["total_score"]) for value in aggregates
        )
        / len(aggregates),
        classification_score=sum(
            float(value["scores"]["classification_score"])
            for value in aggregates
        )
        / len(aggregates),
        record_count=min(
            int(value["counts"]["scored_predictions"])
            for value in aggregates
        ),
        catastrophic_false_approvals=max(
            int(value["raw"]["catastrophic_false_approvals"])
            for value in aggregates
        ),
        false_positive_denials=max(
            len(_false_positive_denial_case_ids(truth, run.rows))
            for run in runs
        ),
        missing_records=max(
            int(value["counts"]["missing_cases"]) for value in aggregates
        ),
        invalid_records=max(
            _invalid_record_count(value) for value in aggregates
        ),
        duplicate_records=max(
            int(value["counts"]["duplicate_case_ids"])
            for value in aggregates
        ),
        extra_records=max(
            int(value["counts"]["extra_cases"]) for value in aggregates
        ),
        deterministic=all(run.deterministic for run in runs),
    )


def _slice_score(
    *,
    truth: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
) -> float:
    subset_truth = {case_id: truth[case_id] for case_id in case_ids}
    subset_rows = _filter_rows(rows, case_ids)
    return float(
        official_evaluate.build_results(subset_truth, subset_rows)[0][
            "scores"
        ]["total_score"]
    )


def _comparison_slices(
    *,
    manifest: FrozenLayoutManifest,
    protected_roles: FrozenProtectedRoles,
    truth: Mapping[str, Mapping[str, Any]],
    arms: Mapping[str, ArmEvidence],
) -> tuple[
    tuple[ScoreSlice, ...],
    tuple[ScoreSlice, ...],
    tuple[ProtectedRoleScore, ...],
    bool,
    bool,
]:
    manager = RepeatedGroupedSplitManager(
        seed=manifest.split_seed,
        repeats=REQUIRED_REPEATS,
        folds=REQUIRED_FOLDS,
    )
    splits = manager.split_groups(manifest.groups)
    repeated = manager.split_groups(manifest.groups)
    group_exclusive = splits == repeated and all(
        not (set(split.tuning_groups) & set(split.validation_groups))
        and not (set(split.tuning_case_ids) & set(split.validation_case_ids))
        for split in splits
    )
    paired = all(
        set(split.validation_case_ids).issubset(truth) for split in splits
    )
    runs_by_arm = {
        approach: {run.repeat: run for run in arm.runs}
        for approach, arm in arms.items()
    }
    repeat_scores: list[ScoreSlice] = []
    for repeat in range(REQUIRED_REPEATS):
        repeat_scores.append(
            ScoreSlice(
                repeat=repeat,
                fold=None,
                record_count=len(manifest.case_ids),
                layout_group_count=len(manifest.groups),
                scores={
                    approach: _slice_score(
                        truth=truth,
                        rows=runs_by_arm[approach][repeat].rows,
                        case_ids=manifest.case_ids,
                    )
                    for approach in APPROACHES
                },
            )
        )
    fold_scores: list[ScoreSlice] = []
    for split in splits:
        fold_scores.append(
            ScoreSlice(
                repeat=split.repeat,
                fold=split.fold,
                record_count=len(split.validation_case_ids),
                layout_group_count=len(split.validation_groups),
                scores={
                    approach: _slice_score(
                        truth=truth,
                        rows=runs_by_arm[approach][split.repeat].rows,
                        case_ids=split.validation_case_ids,
                    )
                    for approach in APPROACHES
                },
            )
        )
    case_to_group = {
        case_id: group
        for group, case_ids in manifest.groups.items()
        for case_id in case_ids
    }
    role_scores: list[ProtectedRoleScore] = []
    for role in PROTECTED_ROLES:
        role_case_ids = protected_roles.by_role[role]
        role_group_count = len(
            {case_to_group[case_id] for case_id in role_case_ids}
        )
        for repeat in range(REQUIRED_REPEATS):
            role_scores.append(
                ProtectedRoleScore(
                    role=role,
                    repeat=repeat,
                    record_count=len(role_case_ids),
                    layout_group_count=role_group_count,
                    scores={
                        approach: _slice_score(
                            truth=truth,
                            rows=runs_by_arm[approach][repeat].rows,
                            case_ids=role_case_ids,
                        )
                        for approach in APPROACHES
                    },
                )
            )
    return (
        tuple(repeat_scores),
        tuple(fold_scores),
        tuple(role_scores),
        group_exclusive,
        paired,
    )


def _load_one_contract_audit(
    path: Path | str,
    *,
    expected_repeat_index: int,
    expected_source_revision_sha: str,
    expected_layout_manifest_sha256: str,
    expected_role_manifest_sha256: str,
    expected_input_tree_sha256: str,
    expected_contract_fixture_sha256: str,
    runtime_paths: Sequence[Path | str],
    feature_name_loader: Callable[[], tuple[str, ...]],
) -> tuple[DecisionRecoveryContractAudit, str]:
    payload = _read_json_object(Path(path), label="decision recovery audit")
    if set(payload) != {
        "schema_version",
        "repeat_index",
        "source_revision_sha",
        "layout_manifest_sha256",
        "protected_role_manifest_sha256",
        "input_tree_sha256",
        "feature_schema_sha256",
        "feature_names",
        "contract_fixture_sha256",
        "counts",
        "deterministic",
    }:
        raise DecisionRecoveryEvidenceBuildError(
            "decision recovery audit must contain the exact contract keys"
        )
    if payload.get("schema_version") != CONTRACT_AUDIT_SCHEMA:
        raise DecisionRecoveryEvidenceBuildError(
            "decision recovery audit schema is unsupported"
        )
    if payload.get("repeat_index") != expected_repeat_index:
        raise DecisionRecoveryEvidenceBuildError(
            "contract audit repeat_index does not match its run"
        )
    bindings = (
        (
            _require_revision(payload.get("source_revision_sha")),
            _require_revision(expected_source_revision_sha),
        ),
        (
            _require_sha256(
                "audit layout manifest",
                payload.get("layout_manifest_sha256"),
            ),
            _require_sha256(
                "expected layout manifest",
                expected_layout_manifest_sha256,
            ),
        ),
        (
            _require_sha256(
                "audit protected roles",
                payload.get("protected_role_manifest_sha256"),
            ),
            _require_sha256(
                "expected protected roles",
                expected_role_manifest_sha256,
            ),
        ),
        (
            _require_sha256(
                "audit input tree", payload.get("input_tree_sha256")
            ),
            _require_sha256(
                "expected input tree", expected_input_tree_sha256
            ),
        ),
    )
    if any(actual != expected for actual, expected in bindings):
        raise DecisionRecoveryEvidenceBuildError(
            "decision recovery audit binding mismatch"
        )
    raw_names = payload.get("feature_names")
    if not isinstance(raw_names, list) or not all(
        isinstance(value, str) for value in raw_names
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "audit feature_names must be an ordered string list"
        )
    audited_names = tuple(value.strip() for value in raw_names)
    runtime_names = feature_name_loader()
    schema_findings = _feature_schema_findings(runtime_names)
    feature_schema_exact = (
        audited_names == runtime_names
        and _feature_schema_hash(runtime_names)
        == _require_sha256(
            "audit feature schema", payload.get("feature_schema_sha256")
        )
        and not schema_findings
    )
    runtime_findings = RuntimeLeakageScanner().scan(runtime_paths)
    raw_counts = payload.get("counts")
    if not isinstance(raw_counts, Mapping):
        raise DecisionRecoveryEvidenceBuildError(
            "decision recovery audit counts must be an object"
        )
    counts = dict(raw_counts)
    expected_finding_count = len(runtime_findings) + len(schema_findings)
    if counts.get("forbidden_feature_finding_count") != expected_finding_count:
        raise DecisionRecoveryEvidenceBuildError(
            "audit leakage count does not match the fresh runtime/schema scan"
        )
    audit = DecisionRecoveryContractAudit(
        counts=counts,
        leakage_clean=expected_finding_count == 0,
        feature_schema_exact=feature_schema_exact,
        deterministic=payload.get("deterministic"),
    )
    fixture_sha = _require_sha256(
        "contract fixture", payload.get("contract_fixture_sha256")
    )
    if fixture_sha != expected_contract_fixture_sha256:
        raise DecisionRecoveryEvidenceBuildError(
            "contract audit does not bind the supplied fixture bytes"
        )
    return audit, fixture_sha


def _load_contract_audits(
    paths: Sequence[Path | str],
    *,
    expected_source_revision_sha: str,
    expected_layout_manifest_sha256: str,
    expected_role_manifest_sha256: str,
    expected_input_tree_sha256: str,
    contract_fixture_path: Path | str,
    model_artifact_paths: Sequence[Path | str],
    feature_name_loader: Callable[[], tuple[str, ...]],
) -> tuple[DecisionRecoveryContractAudit, str]:
    if len(paths) != 2 or len({Path(path).resolve() for path in paths}) != 2:
        raise DecisionRecoveryEvidenceBuildError(
            "exactly two distinct contract audit reruns are required"
        )
    fixture_path = Path(contract_fixture_path).resolve()
    if not fixture_path.is_file():
        raise DecisionRecoveryEvidenceBuildError(
            "contract fixture must be a readable file"
        )
    fixture_payload = _read_json_object(
        fixture_path, label="decision recovery contract fixture"
    )
    if canonical_json(fixture_payload) != canonical_json(
        contract_fixture_payload()
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "contract fixture does not match the executable WO-18 probes"
        )
    artifacts = tuple(Path(path).resolve() for path in model_artifact_paths)
    if not artifacts or len(set(artifacts)) != len(artifacts) or any(
        not path.is_file() for path in artifacts
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "at least one distinct fitted model artifact is required"
        )
    model_source = REPO_ROOT / "mib_pipeline" / "model_recovery.py"
    if not model_source.is_file():
        raise DecisionRecoveryEvidenceBuildError(
            "fixed production model_recovery.py is missing"
        )
    source_findings = _model_specific_source_findings(model_source)
    runtime_paths = (model_source, fixture_path, *artifacts)
    fixture_sha = _sha256_file(fixture_path)
    results = tuple(
        _load_one_contract_audit(
            path,
            expected_repeat_index=repeat,
            expected_source_revision_sha=expected_source_revision_sha,
            expected_layout_manifest_sha256=(
                expected_layout_manifest_sha256
            ),
            expected_role_manifest_sha256=(
                expected_role_manifest_sha256
            ),
            expected_input_tree_sha256=expected_input_tree_sha256,
            expected_contract_fixture_sha256=fixture_sha,
            runtime_paths=runtime_paths,
            feature_name_loader=feature_name_loader,
        )
        for repeat, path in enumerate(paths, start=1)
    )
    first, second = results[0][0], results[1][0]
    if first != second:
        raise DecisionRecoveryEvidenceBuildError(
            "contract probe reruns are not deterministic"
        )
    if source_findings:
        raise DecisionRecoveryEvidenceBuildError(
            "model-specific identity access scan is not clean"
        )
    expected_counts = run_contract_probes()
    if (
        dict(first.counts) != expected_counts
        or dict(second.counts) != expected_counts
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "contract audit counts do not match freshly executed probes"
        )
    return first, fixture_sha


def _artifact_set_sha256(
    *,
    arm_paths: Mapping[str, Path | str],
    arms: Mapping[str, ArmEvidence],
    feature_rows_path: Path | str,
    rerun_feature_rows_path: Path | str,
    capture_observation_path: Path | str,
    contract_audit_paths: Sequence[Path | str],
    contract_fixture_path: Path | str,
    model_artifact_paths: Sequence[Path | str],
) -> str:
    pins: list[dict[str, str]] = [
        {
            "logical_name": "production_feature_rows",
            "sha256": _sha256_file(feature_rows_path),
        },
        {
            "logical_name": "production_feature_rows_rerun",
            "sha256": _sha256_file(rerun_feature_rows_path),
        },
        {
            "logical_name": "production_capture_observation",
            "sha256": _sha256_file(capture_observation_path),
        },
    ]
    for approach in APPROACHES:
        pins.append(
            {
                "logical_name": f"{approach}_manifest",
                "sha256": arms[approach].manifest_sha256,
            }
        )
        for run in arms[approach].runs:
            pins.append(
                {
                    "logical_name": (
                        f"{approach}_repeat_{run.repeat}_predictions"
                    ),
                    "sha256": run.sha256,
                }
            )
            pins.append(
                {
                    "logical_name": (
                        f"{approach}_repeat_{run.repeat}_rerun_predictions"
                    ),
                    "sha256": run.rerun_sha256,
                }
            )
        # Bind the supplied path's bytes even if a future ArmEvidence parser
        # stores additional normalized state.
        if _sha256_file(arm_paths[approach]) != arms[approach].manifest_sha256:
            raise DecisionRecoveryEvidenceBuildError(
                "arm manifest changed during evidence construction"
            )
    for repeat, path in enumerate(contract_audit_paths, start=1):
        pins.append(
            {
                "logical_name": f"contract_audit_repeat_{repeat}",
                "sha256": _sha256_file(path),
            }
        )
    pins.append(
        {
            "logical_name": "contract_fixture",
            "sha256": _sha256_file(contract_fixture_path),
        }
    )
    for index, path in enumerate(model_artifact_paths, start=1):
        pins.append(
            {
                "logical_name": f"fitted_model_artifact_{index}",
                "sha256": _sha256_file(path),
            }
        )
    return _sha256_bytes(canonical_json(pins).encode("utf-8"))


def _validate_production_capture(
    *,
    feature_rows_path: Path | str,
    rerun_feature_rows_path: Path | str,
    capture_observation_path: Path | str,
    source_revision_sha: str,
    layout_manifest_sha256: str,
    input_tree_sha256: str,
    feature_schema_sha256: str,
    expected_record_count: int,
) -> Mapping[str, Any]:
    """Verify the label-free producer, its graph, and two exact capture runs."""

    observation = _read_json_object(
        Path(capture_observation_path),
        label="production capture observation",
    )
    expected_keys = {
        "schema_version",
        "source_revision_sha",
        "producer_source_sha256",
        "producer_graph_sha256",
        "layout_manifest_sha256",
        "input_tree_sha256",
        "feature_schema_sha256",
        "record_count",
        "capture_run_count",
        "feature_rows_sha256",
        "rerun_feature_rows_sha256",
        "byte_deterministic",
        "truth_or_role_input_count",
    }
    if set(observation) != expected_keys:
        raise DecisionRecoveryEvidenceBuildError(
            "production capture observation has an invalid schema"
        )
    if observation.get("schema_version") != CAPTURE_OBSERVATION_SCHEMA:
        raise DecisionRecoveryEvidenceBuildError(
            "production capture observation schema is unsupported"
        )
    primary_path = Path(feature_rows_path)
    rerun_path = Path(rerun_feature_rows_path)
    if (
        not primary_path.is_file()
        or not rerun_path.is_file()
        or primary_path.resolve() == rerun_path.resolve()
        or primary_path.read_bytes() != rerun_path.read_bytes()
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "production feature capture reruns are not byte-identical"
        )
    feature_rows_sha = _sha256_file(primary_path)
    bindings = (
        (
            "source revision",
            _require_revision(observation.get("source_revision_sha")),
            _require_revision(source_revision_sha),
        ),
        (
            "layout manifest",
            _require_sha256(
                "capture layout manifest",
                observation.get("layout_manifest_sha256"),
            ),
            _require_sha256(
                "expected layout manifest",
                layout_manifest_sha256,
            ),
        ),
        (
            "input tree",
            _require_sha256(
                "capture input tree",
                observation.get("input_tree_sha256"),
            ),
            _require_sha256("expected input tree", input_tree_sha256),
        ),
        (
            "feature schema",
            _require_sha256(
                "capture feature schema",
                observation.get("feature_schema_sha256"),
            ),
            _require_sha256(
                "expected feature schema",
                feature_schema_sha256,
            ),
        ),
        (
            "feature rows",
            _require_sha256(
                "capture feature rows",
                observation.get("feature_rows_sha256"),
            ),
            feature_rows_sha,
        ),
        (
            "rerun feature rows",
            _require_sha256(
                "capture rerun feature rows",
                observation.get("rerun_feature_rows_sha256"),
            ),
            _sha256_file(rerun_path),
        ),
        (
            "producer source",
            _require_sha256(
                "capture producer source",
                observation.get("producer_source_sha256"),
            ),
            _sha256_file(
                REPO_ROOT / "devtools" / "wo18_production_capture.py"
            ),
        ),
        (
            "producer graph",
            _require_sha256(
                "capture producer graph",
                observation.get("producer_graph_sha256"),
            ),
            producer_graph_sha256(),
        ),
    )
    if any(actual != expected for _label, actual, expected in bindings):
        mismatched = next(
            label
            for label, actual, expected in bindings
            if actual != expected
        )
        raise DecisionRecoveryEvidenceBuildError(
            f"production capture {mismatched} binding mismatch"
        )
    if (
        observation.get("record_count") != expected_record_count
        or observation.get("capture_run_count") != 2
        or observation.get("byte_deterministic") is not True
        or observation.get("truth_or_role_input_count") != 0
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "production capture is incomplete, non-deterministic, or label-aware"
        )
    return observation


def build_aggregate_evidence(
    *,
    layout_manifest_path: Path | str,
    expected_layout_manifest_sha256: str,
    protected_role_manifest_path: Path | str,
    expected_protected_role_manifest_sha256: str,
    expected_input_tree_sha256: str,
    input_dir: Path | str,
    feature_rows_path: Path | str,
    rerun_feature_rows_path: Path | str,
    capture_observation_path: Path | str,
    truth_path: Path | str,
    arm_manifest_paths: Mapping[str, Path | str],
    contract_audit_paths: Sequence[Path | str],
    contract_fixture_path: Path | str,
    source_revision_sha: str,
    checkout_verifier: Callable[[Path, str], None] = (
        verify_clean_candidate_checkout
    ),
    feature_name_loader: Callable[[], tuple[str, ...]] = (
        _runtime_feature_names
    ),
) -> dict[str, Any]:
    """Build the only repository-safe WO-18 evidence representation."""

    source_revision = _require_revision(source_revision_sha)
    checkout_verifier(REPO_ROOT, source_revision)
    if set(arm_manifest_paths) != set(APPROACHES):
        raise DecisionRecoveryEvidenceBuildError(
            "exactly the four named WO-18 arm manifests are required"
        )
    manifest = load_layout_manifest(layout_manifest_path)
    if manifest.sha256 != _require_sha256(
        "expected layout manifest", expected_layout_manifest_sha256
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "layout manifest byte digest mismatch"
        )
    if len(manifest.case_ids) != REQUIRED_COHORT_RECORDS:
        raise DecisionRecoveryEvidenceBuildError(
            "WO-18 local evidence requires the exact frozen 32-case cohort"
        )
    input_tree_sha, input_pdf_count = _input_tree_sha256(Path(input_dir))
    expected_tree_sha = _require_sha256(
        "expected input tree", expected_input_tree_sha256
    )
    if input_tree_sha != expected_tree_sha or input_pdf_count != len(
        manifest.case_ids
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "input directory does not match the frozen tree/count binding"
        )
    pdf_case_ids = tuple(
        sorted(
            path.stem
            for path in Path(input_dir).iterdir()
            if path.is_file() and path.suffix.casefold() == ".pdf"
        )
    )
    cohort_case_set_sha = case_id_set_sha256(manifest.case_ids)
    if case_id_set_sha256(pdf_case_ids) != cohort_case_set_sha:
        raise DecisionRecoveryEvidenceBuildError(
            "input PDF stems do not match the frozen cohort case set"
        )
    roles = load_protected_role_manifest(
        protected_role_manifest_path,
        expected_layout_manifest_sha256=manifest.sha256,
        expected_case_ids=manifest.case_ids,
        expected_group_by_case={
            case_id: group_id
            for group_id, case_ids in manifest.groups.items()
            for case_id in case_ids
        },
    )
    if roles.sha256 != _require_sha256(
        "expected protected role manifest",
        expected_protected_role_manifest_sha256,
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "protected role manifest byte digest mismatch"
        )
    truth_sha = _sha256_file(truth_path)
    truth = _read_truth_subset(truth_path, manifest.case_ids)
    evaluator_sha = _sha256_file(Path(official_evaluate.__file__))
    feature_names = feature_name_loader()
    feature_schema_sha = _feature_schema_hash(feature_names)
    capture_observation = _validate_production_capture(
        feature_rows_path=feature_rows_path,
        rerun_feature_rows_path=rerun_feature_rows_path,
        capture_observation_path=capture_observation_path,
        source_revision_sha=source_revision,
        layout_manifest_sha256=manifest.sha256,
        input_tree_sha256=expected_tree_sha,
        feature_schema_sha256=feature_schema_sha,
        expected_record_count=len(manifest.case_ids),
    )
    frozen_features = load_frozen_feature_rows(
        feature_rows_path,
        layout_manifest=manifest,
        expected_source_revision_sha=source_revision,
        expected_input_tree_sha256=expected_tree_sha,
    )
    _validate_protected_role_semantics(
        protected_roles=roles,
        frozen_features=frozen_features,
        truth=truth,
    )
    expected_execution = execute_grouped_oof(
        layout_manifest=manifest,
        truth=truth,
        features=frozen_features,
    )
    repeated_execution = execute_grouped_oof(
        layout_manifest=manifest,
        truth=truth,
        features=frozen_features,
    )
    if execution_fingerprint(expected_execution) != execution_fingerprint(
        repeated_execution
    ):
        raise DecisionRecoveryEvidenceBuildError(
            "fixed four-arm grouped OOF runner is not deterministic"
        )
    arms = {
        approach: load_arm_manifest(
            arm_manifest_paths[approach],
            expected_approach=approach,
            expected_source_revision_sha=source_revision,
            layout_manifest=manifest,
            expected_role_manifest_sha256=roles.sha256,
            expected_truth_sha256=truth_sha,
            expected_input_tree_sha256=expected_input_tree_sha256,
            expected_feature_schema_sha256=feature_schema_sha,
            expected_feature_rows_sha256=frozen_features.sha256,
            expected_evaluator_sha256=evaluator_sha,
            expected_execution=expected_execution,
        )
        for approach in APPROACHES
    }
    fitted_model_artifacts = tuple(
        sorted(
            {
                path
                for arm in arms.values()
                for path in arm.model_artifact_paths
            }
        )
    )
    contract_audit, contract_fixture_sha = _load_contract_audits(
        contract_audit_paths,
        expected_source_revision_sha=source_revision,
        expected_layout_manifest_sha256=manifest.sha256,
        expected_role_manifest_sha256=roles.sha256,
        expected_input_tree_sha256=expected_input_tree_sha256,
        contract_fixture_path=contract_fixture_path,
        model_artifact_paths=fitted_model_artifacts,
        feature_name_loader=feature_name_loader,
    )
    (
        repeat_scores,
        fold_scores,
        protected_role_scores,
        group_exclusive,
        paired_fold_members,
    ) = _comparison_slices(
        manifest=manifest,
        protected_roles=roles,
        truth=truth,
        arms=arms,
    )
    full_runs = {
        approach: _aggregate_run(truth, arms[approach].runs)
        for approach in APPROACHES
    }
    new_catastrophic = 0
    new_false_denial = 0
    for repeat in range(REQUIRED_REPEATS):
        control_rows = arms[CONTROL_APPROACH].runs[repeat].rows
        candidate_rows = arms[CANDIDATE_APPROACH].runs[repeat].rows
        control_catastrophic, _ = _safety_event_case_ids(
            truth, control_rows
        )
        candidate_catastrophic, _ = _safety_event_case_ids(
            truth, candidate_rows
        )
        new_catastrophic += len(
            candidate_catastrophic - control_catastrophic
        )
        new_false_denial += len(
            _false_positive_denial_case_ids(truth, candidate_rows)
            - _false_positive_denial_case_ids(truth, control_rows)
        )
    artifact_set_sha = _artifact_set_sha256(
        arm_paths=arm_manifest_paths,
        arms=arms,
        feature_rows_path=feature_rows_path,
        rerun_feature_rows_path=rerun_feature_rows_path,
        capture_observation_path=capture_observation_path,
        contract_audit_paths=contract_audit_paths,
        contract_fixture_path=contract_fixture_path,
        model_artifact_paths=fitted_model_artifacts,
    )
    evidence = DecisionRecoveryEvidence(
        source_revision_sha=source_revision,
        layout_manifest_sha256=manifest.sha256,
        protected_role_manifest_sha256=roles.sha256,
        input_tree_sha256=_require_sha256(
            "expected input tree", expected_input_tree_sha256
        ),
        cohort_set_sha256=cohort_case_set_sha,
        truth_sha256=truth_sha,
        official_evaluator_sha256=evaluator_sha,
        feature_schema_sha256=feature_schema_sha,
        artifact_set_sha256=artifact_set_sha,
        expected_record_count=len(manifest.case_ids),
        expected_layout_group_count=len(manifest.groups),
        full_runs=full_runs,
        repeat_scores=repeat_scores,
        fold_scores=fold_scores,
        protected_role_scores=protected_role_scores,
        contract_audit=contract_audit,
        new_catastrophic_false_approval_count=new_catastrophic,
        new_false_positive_denial_count=new_false_denial,
        manifest_frozen_before_scoring=manifest.frozen_before_scoring,
        roles_frozen_before_scoring=roles.frozen_before_scoring,
        group_exclusive=group_exclusive,
        paired_fold_members=paired_fold_members,
    )
    aggregate = DecisionRecoveryGate().evaluate(
        evidence
    ).to_aggregate_evidence()
    # The fixture hash is useful proof but intentionally has no path or cases.
    aggregate["contract_fixture_sha256"] = contract_fixture_sha
    aggregate["production_capture_observation_sha256"] = _sha256_file(
        capture_observation_path
    )
    aggregate["production_capture_graph_sha256"] = str(
        capture_observation["producer_graph_sha256"]
    )
    aggregate["production_capture_run_count"] = int(
        capture_observation["capture_run_count"]
    )
    aggregate["production_capture_byte_deterministic"] = bool(
        capture_observation["byte_deterministic"]
    )
    aggregate["production_capture_truth_or_role_input_count"] = int(
        capture_observation["truth_or_role_input_count"]
    )
    require_aggregate_only(aggregate)
    return aggregate


def render_aggregate_markdown(evidence: Mapping[str, Any]) -> str:
    """Render a compact identity-free four-arm comparison."""

    require_aggregate_only(evidence)
    status = str(evidence["status"]).upper()
    class_metrics = evidence["class_metrics"]
    role_metrics = evidence["field_metrics"]
    gates = evidence["gate_results"]
    lines = [
        "# WO-18 Identity-Free Decision Recovery",
        "",
        f"**Promotion gate: {status}**",
        "",
        (
            "Scope: public-exposed grouped robustness evidence; this is not "
            "an unseen holdout or the full-public 1,000-case evaluation."
        ),
        "",
        "## Four-arm comparison",
        "",
        "| Approach | Mean official score | Classification | Catastrophic approvals | False-positive denials |",
        "|---|---:|---:|---:|---:|",
    ]
    for approach in APPROACHES:
        value = class_metrics[approach]
        lines.append(
            f"| `{approach}` | {value['total_score']:.6f} | "
            f"{value['classification_score']:.6f} | "
            f"{value['catastrophic_false_approvals']} | "
            f"{value['false_positive_denial_count']} |"
        )
    lines.extend(
        [
            "",
            "## Generalization requirements",
            "",
            f"- Full candidate delta: {evidence['score_delta']:+.6f}",
            (
                "- Minimum repeated-run delta: "
                f"{evidence['repeat_score_delta_min']:+.6f}"
            ),
            (
                "- Minimum grouped-fold delta: "
                f"{evidence['fold_score_delta_min']:+.6f}"
            ),
            (
                "- Minimum protected-role delta: "
                f"{evidence['protected_role_score_delta_min']:+.6f}"
            ),
            "",
            "## Protected roles",
            "",
            "| Role | Support | Minimum delta | Mean delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for role in PROTECTED_ROLES:
        value = role_metrics[role]
        lines.append(
            f"| `{role}` | {value['record_count']} | "
            f"{value['score_delta_min']:+.6f} | "
            f"{value['score_delta_mean']:+.6f} |"
        )
    lines.extend(["", "## Hard gates", ""])
    for name, passed in gates.items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "All committed content is aggregate-only. Identity-bearing truth, "
            "roles, fold membership, predictions, and fit attestations remain "
            "external and are represented only by immutable digests.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--arm must use APPROACH=/absolute/manifest.json"
        )
    approach, raw_path = value.split("=", 1)
    if approach not in APPROACHES or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "--arm approach must be one of the exact WO-18 approaches"
        )
    return approach, Path(raw_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-layout-manifest-sha256", required=True
    )
    parser.add_argument("--protected-role-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-protected-role-manifest-sha256", required=True
    )
    parser.add_argument("--expected-input-tree-sha256", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--feature-rows", type=Path, required=True)
    parser.add_argument("--rerun-feature-rows", type=Path, required=True)
    parser.add_argument("--capture-observation", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        type=_parse_arm,
        required=True,
        help="repeat exactly four times as APPROACH=/path/to/arm.json",
    )
    parser.add_argument(
        "--contract-audit",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly twice for deterministic contract-probe reruns",
    )
    parser.add_argument("--contract-fixture", type=Path, required=True)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args(argv)
    arm_paths = dict(arguments.arm)
    if len(arguments.arm) != len(APPROACHES) or set(arm_paths) != set(
        APPROACHES
    ):
        parser.error("--arm must assign each exact WO-18 approach once")
    if len(arguments.contract_audit) != 2:
        parser.error("--contract-audit must be supplied exactly twice")
    try:
        aggregate = build_aggregate_evidence(
            layout_manifest_path=arguments.layout_manifest,
            expected_layout_manifest_sha256=(
                arguments.expected_layout_manifest_sha256
            ),
            protected_role_manifest_path=(
                arguments.protected_role_manifest
            ),
            expected_protected_role_manifest_sha256=(
                arguments.expected_protected_role_manifest_sha256
            ),
            expected_input_tree_sha256=(
                arguments.expected_input_tree_sha256
            ),
            input_dir=arguments.input_dir,
            feature_rows_path=arguments.feature_rows,
            rerun_feature_rows_path=arguments.rerun_feature_rows,
            capture_observation_path=arguments.capture_observation,
            truth_path=arguments.truth,
            arm_manifest_paths=arm_paths,
            contract_audit_paths=arguments.contract_audit,
            contract_fixture_path=arguments.contract_fixture,
            source_revision_sha=arguments.source_revision_sha,
        )
    except ExperimentControlError as exc:
        parser.error(str(exc))
    _atomic_write(
        arguments.output_json,
        canonical_json(aggregate) + "\n",
    )
    _atomic_write(
        arguments.output_markdown,
        render_aggregate_markdown(aggregate),
    )
    print(f"WO-18 promotion gate: {str(aggregate['status']).upper()}")
    return 0 if aggregate["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
