#!/usr/bin/env python3
"""Close WO-18 honestly when its empirical promotion gates cannot be met.

This builder consumes the external, identity-bearing WO-18 capture and
three-by-five grouped OOF artifacts.  It verifies their byte bindings and
official-evaluator results, then emits a repository-safe aggregate report.
It is intentionally incapable of promoting or enabling the candidate model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.decision_recovery_cv import (  # noqa: E402
    ARM_MANIFEST_SCHEMA,
    FEATURE_ROWS_SCHEMA,
    approach_graph_sha256,
    execute_grouped_oof,
    execution_fingerprint,
    load_frozen_feature_rows,
    runner_source_sha256,
)
from devtools.decision_recovery_evidence import (  # noqa: E402
    CONTRACT_AUDIT_SCHEMA,
    PROTECTED_ROLE_TAXONOMY_SHA256,
    _feature_schema_findings,
    _feature_schema_hash,
    _model_specific_source_findings,
    _runtime_feature_names,
)
from devtools.decision_recovery_contract_probe import (  # noqa: E402
    contract_fixture_payload,
    run_contract_probes,
)
from devtools.decision_recovery_gate import (  # noqa: E402
    APPROACHES,
    CANDIDATE_APPROACH,
    CONTROL_APPROACH,
    PROTECTED_ROLES,
    REQUIRED_FOLDS,
    REQUIRED_REPEATS,
    DecisionRecoveryContractAudit,
)
from devtools.experiment_control import (  # noqa: E402
    ExperimentControlError,
    RepeatedGroupedSplitManager,
    RuntimeLeakageScanner,
    canonical_json,
    require_aggregate_only,
)
from devtools.grouped_recovery_evidence import (  # noqa: E402
    load_layout_manifest,
)
from devtools.wo18_production_capture import (  # noqa: E402
    CAPTURE_OBSERVATION_SCHEMA,
)
from scripts import evaluate as official_evaluate  # noqa: E402


COVERAGE_GAP_SCHEMA = "mib-wo18-protected-role-coverage-gap/v1"
EXPECTED_MISSING_ROLES = ("binding_authority", "approval_guard")
EXPECTED_RECORD_COUNT = 32
EXPECTED_SOURCE_REVISION = "e4330940414c382dcab89816110ab2be2ff238cf"
OUTPUT_SCHEMA = "mib-wo18-blocked-evidence/v1"
_SHA256_LENGTH = 64
_REVISION_LENGTH = 40
_CASE_ID_RE = re.compile(r"\bMIB-\d{6}\b", re.IGNORECASE)


class WO18BlockedEvidenceError(ExperimentControlError):
    """The external WO-18 blocked-evidence set is incomplete or inconsistent."""


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WO18BlockedEvidenceError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise WO18BlockedEvidenceError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise WO18BlockedEvidenceError(f"required artifact is unreadable") from exc
    return digest.hexdigest()


def _digest(name: str, value: Any) -> str:
    normalized = str(value).strip().casefold()
    if (
        len(normalized) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise WO18BlockedEvidenceError(f"{name} must be a full SHA-256 digest")
    return normalized


def _revision(value: Any) -> str:
    normalized = str(value).strip().casefold()
    if (
        len(normalized) != _REVISION_LENGTH
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise WO18BlockedEvidenceError(
            "source revision must be a full Git commit SHA"
        )
    if normalized != EXPECTED_SOURCE_REVISION:
        raise WO18BlockedEvidenceError(
            "this closure builder is bound to the evaluated e433 revision"
        )
    return normalized


def _same(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise WO18BlockedEvidenceError(f"{name} binding mismatch")


def _git_bytes(revision: str, relative_path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise WO18BlockedEvidenceError(
            "evaluated revision or required evaluated blob is unavailable"
        )
    return result.stdout


def _evaluated_graph(revision: str) -> tuple[str, str, str, bool]:
    """Recompute capture pins from Git bytes, independent of a dirty checkout."""

    listing = subprocess.run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            revision,
            "--",
            "mib_pipeline",
            "requirements.lock",
            "Dockerfile",
            "run.sh",
        ],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if listing.returncode != 0:
        raise WO18BlockedEvidenceError("evaluated revision is unavailable")
    paths = []
    for value in listing.stdout.splitlines():
        path = value.strip()
        if (
            (path.startswith("mib_pipeline/") and path.endswith(".py"))
            or path.startswith("mib_pipeline/artifacts/")
            or path in {"requirements.lock", "Dockerfile", "run.sh"}
        ):
            paths.append(path)
    if not paths:
        raise WO18BlockedEvidenceError("evaluated production graph is empty")
    entries = []
    candidate_references = []
    for path in sorted(paths):
        content = _git_bytes(revision, path)
        entries.append(
            {
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        )
        if (
            path.endswith(".py")
            and path
            not in {"mib_pipeline/model_recovery.py", "mib_pipeline/__init__.py"}
            and b"GatedHybridDecisionRecoveryAdjudicator" in content
        ):
            candidate_references.append(path)
    graph_sha = hashlib.sha256(
        (canonical_json(entries) + "\n").encode("utf-8")
    ).hexdigest()
    producer_source_sha = hashlib.sha256(
        _git_bytes(revision, "devtools/wo18_production_capture.py")
    ).hexdigest()
    runner_sha = hashlib.sha256(
        _git_bytes(revision, "devtools/decision_recovery_cv.py")
    ).hexdigest()
    return graph_sha, producer_source_sha, runner_sha, not candidate_references


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(value) != expected:
        raise WO18BlockedEvidenceError(f"{label} has an invalid schema")


def _validate_capture(
    *,
    root: Path,
    source_revision: str,
    layout_sha: str,
) -> dict[str, Any]:
    feature_path = root / "feature-rows.json"
    rerun_path = root / "feature-rows-rerun.json"
    observation_path = root / "capture-observation.json"
    if feature_path.read_bytes() != rerun_path.read_bytes():
        raise WO18BlockedEvidenceError(
            "production capture and rerun must be byte-identical"
        )
    feature_sha = _sha256_file(feature_path)
    feature = _read_object(feature_path, label="frozen feature rows")
    _require_exact_keys(
        feature,
        {
            "schema_version",
            "frozen_before_fit",
            "source_revision_sha",
            "layout_manifest_sha256",
            "input_tree_sha256",
            "feature_schema_sha256",
            "rows",
        },
        label="frozen feature rows",
    )
    if (
        feature["schema_version"] != FEATURE_ROWS_SCHEMA
        or feature["frozen_before_fit"] is not True
    ):
        raise WO18BlockedEvidenceError("feature capture is not frozen")
    _same("feature source", feature["source_revision_sha"], source_revision)
    _same("feature layout", feature["layout_manifest_sha256"], layout_sha)
    rows = feature["rows"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_RECORD_COUNT:
        raise WO18BlockedEvidenceError(
            "feature capture must contain exactly 32 records"
        )
    runtime_schema_sha = _feature_schema_hash(_runtime_feature_names())
    _same(
        "feature schema",
        _digest("feature schema", feature["feature_schema_sha256"]),
        runtime_schema_sha,
    )

    observation = _read_object(
        observation_path, label="production capture observation"
    )
    _require_exact_keys(
        observation,
        {
            "schema_version",
            "source_revision_sha",
            "producer_source_sha256",
            "producer_graph_sha256",
            "layout_manifest_sha256",
            "input_tree_sha256",
            "feature_schema_sha256",
            "feature_rows_sha256",
            "rerun_feature_rows_sha256",
            "record_count",
            "capture_run_count",
            "truth_or_role_input_count",
            "byte_deterministic",
        },
        label="production capture observation",
    )
    if observation["schema_version"] != CAPTURE_OBSERVATION_SCHEMA:
        raise WO18BlockedEvidenceError("capture observation schema is unsupported")
    bindings = {
        "source_revision_sha": source_revision,
        "layout_manifest_sha256": layout_sha,
        "input_tree_sha256": feature["input_tree_sha256"],
        "feature_schema_sha256": runtime_schema_sha,
        "feature_rows_sha256": feature_sha,
        "rerun_feature_rows_sha256": feature_sha,
        "record_count": EXPECTED_RECORD_COUNT,
        "capture_run_count": 2,
        "truth_or_role_input_count": 0,
        "byte_deterministic": True,
    }
    for name, expected in bindings.items():
        _same(f"capture observation {name}", observation[name], expected)
    (
        evaluated_graph_sha,
        evaluated_producer_source_sha,
        evaluated_runner_source_sha,
        candidate_not_composed,
    ) = _evaluated_graph(source_revision)
    _same(
        "capture evaluated producer source",
        _digest("producer source", observation["producer_source_sha256"]),
        evaluated_producer_source_sha,
    )
    _same(
        "capture evaluated producer graph",
        _digest("producer graph", observation["producer_graph_sha256"]),
        evaluated_graph_sha,
    )
    return {
        "feature_rows_sha256": feature_sha,
        "feature_schema_sha256": runtime_schema_sha,
        "input_tree_sha256": _digest(
            "input tree", feature["input_tree_sha256"]
        ),
        "observation_sha256": _sha256_file(observation_path),
        "producer_graph_sha256": observation["producer_graph_sha256"],
        "producer_source_sha256": observation["producer_source_sha256"],
        "evaluated_runner_source_sha256": evaluated_runner_source_sha,
        "candidate_runtime_composed": not candidate_not_composed,
        "feature_rows": rows,
    }


def _fresh_role_coverage(
    *,
    feature_rows: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Mapping[str, Any]],
    layout: Any,
) -> tuple[dict[str, dict[str, int]], int, int]:
    group_by_case = {
        case_id: group
        for group, case_ids in layout.groups.items()
        for case_id in case_ids
    }
    eligible: dict[str, set[str]] = {role: set() for role in PROTECTED_ROLES}
    seen: set[str] = set()
    for row in feature_rows:
        if not isinstance(row, Mapping):
            raise WO18BlockedEvidenceError("feature row must be an object")
        case_id = str(row.get("case_id", "")).strip()
        values = row.get("features")
        if (
            not case_id
            or case_id in seen
            or case_id not in truth
            or case_id not in group_by_case
            or not isinstance(values, Mapping)
        ):
            raise WO18BlockedEvidenceError(
                "feature rows do not cover the bound cohort exactly"
            )
        seen.add(case_id)
        truth_decision = str(
            truth[case_id].get("adjudication", "")
        ).strip().upper()
        binding = any(
            values[name] == 1.0
            for name in ("binding_approval", "binding_denial", "binding_review")
        )
        visible_violation = values["policy_explicit_violation"] == 1.0
        approval = (
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
        denial = truth_decision == "DENIED" and (
            visible_violation or values["binding_denial"] == 1.0
        )
        uncertainty = (
            truth_decision == "NEEDS_REVIEW"
            or values["unresolved_linkage"] == 1.0
            or values["packet_conflict"] == 1.0
            or values["packet_watermark"] == 1.0
            or values["unknown_fraction"] > 0.0
            or values["contested_fraction"] > 0.0
        )
        for role, valid in {
            "binding_authority": binding,
            "visible_disqualifier": visible_violation,
            "approval_guard": approval,
            "denial_guard": denial,
            "uncertainty": uncertainty,
        }.items():
            if valid:
                eligible[role].add(case_id)
    if seen != set(layout.case_ids):
        raise WO18BlockedEvidenceError(
            "feature rows do not cover the bound cohort exactly"
        )
    union = set().union(*eligible.values())
    return (
        {
            role: {
                "record_count": len(case_ids),
                "layout_group_count": len(
                    {group_by_case[case_id] for case_id in case_ids}
                ),
            }
            for role, case_ids in eligible.items()
        },
        len(union),
        len(seen - union),
    )


def _validate_coverage_gap(
    *,
    root: Path,
    source_revision: str,
    layout_sha: str,
    capture: Mapping[str, Any],
    truth_sha: str,
    truth: Mapping[str, Mapping[str, Any]] | None = None,
    layout: Any | None = None,
) -> dict[str, Any]:
    path = root / "protected-role-coverage-gap.json"
    payload = _read_object(path, label="protected-role coverage gap")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "frozen_before_scoring",
            "source_revision_sha",
            "layout_manifest_sha256",
            "role_taxonomy_sha256",
            "feature_rows_sha256",
            "truth_sha256",
            "cohort_record_count",
            "eligible_case_count",
            "unassigned_case_count",
            "role_coverage",
            "missing_required_roles",
            "exact_one_role_full_cohort_manifest_possible",
            "promotion_blocked",
            "candidate_runtime_enabled",
        },
        label="protected-role coverage gap",
    )
    if (
        payload["schema_version"] != COVERAGE_GAP_SCHEMA
        or payload["frozen_before_scoring"] is not True
    ):
        raise WO18BlockedEvidenceError("coverage gap is not a frozen v1 artifact")
    expected_bindings = {
        "source_revision_sha": source_revision,
        "layout_manifest_sha256": layout_sha,
        "role_taxonomy_sha256": PROTECTED_ROLE_TAXONOMY_SHA256,
        "feature_rows_sha256": capture["feature_rows_sha256"],
        "truth_sha256": truth_sha,
        "cohort_record_count": EXPECTED_RECORD_COUNT,
        "missing_required_roles": list(EXPECTED_MISSING_ROLES),
        "exact_one_role_full_cohort_manifest_possible": False,
        "promotion_blocked": True,
        "candidate_runtime_enabled": False,
    }
    for name, expected in expected_bindings.items():
        _same(f"coverage gap {name}", payload[name], expected)
    eligible = payload["eligible_case_count"]
    unassigned = payload["unassigned_case_count"]
    if (
        isinstance(eligible, bool)
        or isinstance(unassigned, bool)
        or not isinstance(eligible, int)
        or not isinstance(unassigned, int)
        or eligible + unassigned != EXPECTED_RECORD_COUNT
        or unassigned <= 0
    ):
        raise WO18BlockedEvidenceError("coverage support totals are invalid")
    coverage = payload["role_coverage"]
    if not isinstance(coverage, Mapping) or set(coverage) != set(PROTECTED_ROLES):
        raise WO18BlockedEvidenceError("coverage must contain the exact five roles")
    safe_coverage: dict[str, dict[str, int]] = {}
    for role in PROTECTED_ROLES:
        values = coverage[role]
        if not isinstance(values, Mapping):
            raise WO18BlockedEvidenceError("role coverage must be an object")
        _require_exact_keys(
            values,
            {"eligible_case_count", "eligible_layout_group_count"},
            label="role coverage",
        )
        case_count = values["eligible_case_count"]
        group_count = values["eligible_layout_group_count"]
        if (
            isinstance(case_count, bool)
            or isinstance(group_count, bool)
            or not isinstance(case_count, int)
            or not isinstance(group_count, int)
            or case_count < 0
            or group_count < 0
            or group_count > case_count
        ):
            raise WO18BlockedEvidenceError("role coverage counts are invalid")
        should_be_missing = role in EXPECTED_MISSING_ROLES
        if should_be_missing and (case_count != 0 or group_count != 0):
            raise WO18BlockedEvidenceError("a declared missing role has support")
        if not should_be_missing and (case_count < 2 or group_count < 2):
            raise WO18BlockedEvidenceError(
                "a non-missing protected role lacks non-vacuous support"
            )
        safe_coverage[role] = {
            "record_count": case_count,
            "layout_group_count": group_count,
        }
    if truth is not None and layout is not None:
        fresh_coverage, fresh_eligible, fresh_unassigned = _fresh_role_coverage(
            feature_rows=capture["feature_rows"],
            truth=truth,
            layout=layout,
        )
        _same("fresh protected-role coverage", safe_coverage, fresh_coverage)
        _same("fresh eligible support", eligible, fresh_eligible)
        _same("fresh unassigned support", unassigned, fresh_unassigned)
    return {
        "sha256": _sha256_file(path),
        "coverage": safe_coverage,
        "eligible_count": eligible,
        "unassigned_count": unassigned,
    }


def _load_truth(path: Path, expected_sha: str | None = None) -> tuple[dict[str, Any], str]:
    truth_sha = _sha256_file(path)
    if expected_sha is not None:
        _same("truth", truth_sha, expected_sha)
    truth = official_evaluate.read_truth(path)
    if len(truth) != EXPECTED_RECORD_COUNT:
        raise WO18BlockedEvidenceError("truth must contain exactly 32 cases")
    return truth, truth_sha


def _validate_file_binding(path_value: Any, digest_value: Any, *, label: str) -> Path:
    path = Path(str(path_value)).resolve()
    if not path.is_file():
        raise WO18BlockedEvidenceError(f"{label} file is missing")
    _same(label, _sha256_file(path), _digest(label, digest_value))
    return path


def _expected_splits(layout: Any) -> Mapping[tuple[int, int], Any]:
    manager = RepeatedGroupedSplitManager(
        seed=layout.split_seed,
        repeats=REQUIRED_REPEATS,
        folds=REQUIRED_FOLDS,
    )
    splits = manager.split_groups(layout.groups)
    if len(splits) != REQUIRED_REPEATS * REQUIRED_FOLDS:
        raise WO18BlockedEvidenceError("layout did not produce exact 3x5 splits")
    return {(split.repeat, split.fold): split for split in splits}


def _hash_set(values: Sequence[str]) -> str:
    return hashlib.sha256(
        canonical_json(sorted(str(value) for value in values)).encode("utf-8")
    ).hexdigest()


def _validate_score(
    *,
    score_path: Path,
    truth: Mapping[str, Any],
    prediction_path: Path,
) -> dict[str, Any]:
    recorded = _read_object(score_path, label="official evaluator aggregate")
    predictions = official_evaluate.read_submission(prediction_path)
    computed, _ = official_evaluate.build_results(truth, predictions)
    if canonical_json(recorded) != canonical_json(computed):
        raise WO18BlockedEvidenceError(
            "recorded score does not match the official evaluator"
        )
    counts = computed["counts"]
    if (
        counts["truth_cases"] != EXPECTED_RECORD_COUNT
        or counts["submitted_records"] != EXPECTED_RECORD_COUNT
        or counts["scored_predictions"] != EXPECTED_RECORD_COUNT
        or any(
            counts[name] != 0
            for name in (
                "missing_cases",
                "extra_cases",
                "duplicate_case_ids",
                "blank_case_rows",
                "invalid_adjudication_records",
                "invalid_confidence_records",
                "invalid_fee_status_records",
            )
        )
    ):
        raise WO18BlockedEvidenceError("official evaluator aggregate is incomplete")
    return computed


def _validate_arms(
    *,
    root: Path,
    source_revision: str,
    layout: Any,
    capture: Mapping[str, Any],
    coverage_sha: str,
    truth: Mapping[str, Any],
    truth_sha: str,
) -> dict[str, Any]:
    manifest_dir = root / "cv" / "manifests"
    expected_splits = _expected_splits(layout)
    evaluator_sha = _sha256_file(Path(official_evaluate.__file__).resolve())
    for relative in (
        "mib_pipeline/model_recovery.py",
        "devtools/decision_recovery_cv.py",
    ):
        if (REPO_ROOT / relative).read_bytes() != _git_bytes(
            source_revision, relative
        ):
            raise WO18BlockedEvidenceError(
                "live OOF callable graph differs from evaluated source"
            )
    _same(
        "evaluated runner source",
        runner_source_sha256(),
        capture["evaluated_runner_source_sha256"],
    )
    frozen_features = load_frozen_feature_rows(
        root / "feature-rows.json",
        layout_manifest=layout,
        expected_source_revision_sha=source_revision,
        expected_input_tree_sha256=capture["input_tree_sha256"],
    )
    fresh_execution = execute_grouped_oof(
        layout_manifest=layout,
        truth=truth,
        features=frozen_features,
    )
    fresh_rerun = execute_grouped_oof(
        layout_manifest=layout,
        truth=truth,
        features=frozen_features,
    )
    if execution_fingerprint(fresh_execution) != execution_fingerprint(
        fresh_rerun
    ):
        raise WO18BlockedEvidenceError(
            "fresh grouped OOF execution is not deterministic"
        )
    summaries: dict[str, Any] = {}
    artifact_hashes: list[str] = []
    fold_scores_by_approach: dict[str, list[float]] = {
        approach: [] for approach in APPROACHES
    }
    for approach in APPROACHES:
        manifest_path = manifest_dir / f"{approach}.json"
        manifest = _read_object(manifest_path, label=f"{approach} arm manifest")
        _require_exact_keys(
            manifest,
            {
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
            },
            label=f"{approach} arm manifest",
        )
        bindings = {
            "schema_version": ARM_MANIFEST_SCHEMA,
            "approach": approach,
            "source_revision_sha": source_revision,
            "layout_manifest_sha256": layout.sha256,
            "protected_role_manifest_sha256": coverage_sha,
            "truth_sha256": truth_sha,
            "input_tree_sha256": capture["input_tree_sha256"],
            "feature_schema_sha256": capture["feature_schema_sha256"],
            "frozen_feature_rows_sha256": capture["feature_rows_sha256"],
            "official_evaluator_sha256": evaluator_sha,
        }
        for name, expected in bindings.items():
            _same(f"{approach} {name}", manifest[name], expected)
        _same(
            f"{approach} evaluated runner source",
            _digest("runner source", manifest["runner_source_sha256"]),
            capture["evaluated_runner_source_sha256"],
        )
        _same(
            f"{approach} callable graph",
            _digest("approach graph", manifest["approach_graph_sha256"]),
            approach_graph_sha256(approach),
        )
        runs = manifest["runs"]
        if (
            not isinstance(runs, list)
            or len(runs) != REQUIRED_REPEATS
            or {run.get("repeat_index") for run in runs}
            != set(range(REQUIRED_REPEATS))
        ):
            raise WO18BlockedEvidenceError(
                f"{approach} must contain exact repeats 0, 1, and 2"
            )
        repeat_scores: list[float] = []
        for run in sorted(runs, key=lambda value: value["repeat_index"]):
            _require_exact_keys(
                run,
                {
                    "repeat_index",
                    "predictions_path",
                    "predictions_sha256",
                    "rerun_predictions_path",
                    "rerun_predictions_sha256",
                    "folds",
                },
                label=f"{approach} repeat",
            )
            repeat = run["repeat_index"]
            prediction_path = _validate_file_binding(
                run["predictions_path"],
                run["predictions_sha256"],
                label=f"{approach} repeat prediction",
            )
            rerun_path = _validate_file_binding(
                run["rerun_predictions_path"],
                run["rerun_predictions_sha256"],
                label=f"{approach} repeat rerun",
            )
            if prediction_path.read_bytes() != rerun_path.read_bytes():
                raise WO18BlockedEvidenceError(
                    f"{approach} repeat is not byte-deterministic"
                )
            external_rows = official_evaluate.read_submission(prediction_path)
            if canonical_json(external_rows) != canonical_json(
                list(fresh_execution.rows[approach][repeat])
            ):
                raise WO18BlockedEvidenceError(
                    f"{approach} OOF rows differ from fresh execution"
                )
            folds = run["folds"]
            if (
                not isinstance(folds, list)
                or len(folds) != REQUIRED_FOLDS
                or {fold.get("fold_index") for fold in folds}
                != set(range(REQUIRED_FOLDS))
            ):
                raise WO18BlockedEvidenceError(
                    f"{approach} repeat must contain exact folds 0 through 4"
                )
            for fold in folds:
                _require_exact_keys(
                    fold,
                    {
                        "fold_index",
                        "training_groups_sha256",
                        "validation_groups_sha256",
                        "training_case_set_sha256",
                        "validation_case_set_sha256",
                        "fold_model_artifact_path",
                        "fold_model_artifact_sha256",
                        "validation_predictions_path",
                        "validation_predictions_sha256",
                    },
                    label=f"{approach} fold",
                )
                split = expected_splits[(repeat, fold["fold_index"])]
                expected_hashes = {
                    "training_groups_sha256": _hash_set(split.tuning_groups),
                    "validation_groups_sha256": _hash_set(
                        split.validation_groups
                    ),
                    "training_case_set_sha256": _hash_set(
                        split.tuning_case_ids
                    ),
                    "validation_case_set_sha256": _hash_set(
                        split.validation_case_ids
                    ),
                }
                for name, expected in expected_hashes.items():
                    _same(f"{approach} fold {name}", fold[name], expected)
                model_path = _validate_file_binding(
                    fold["fold_model_artifact_path"],
                    fold["fold_model_artifact_sha256"],
                    label=f"{approach} fold model",
                )
                fold_prediction_path = _validate_file_binding(
                    fold["validation_predictions_path"],
                    fold["validation_predictions_sha256"],
                    label=f"{approach} fold prediction",
                )
                fold_rows = official_evaluate.read_submission(
                    fold_prediction_path
                )
                indexed, duplicates, blanks = official_evaluate.index_submission(
                    fold_rows
                )
                if (
                    duplicates
                    or blanks
                    or set(indexed) != set(split.validation_case_ids)
                ):
                    raise WO18BlockedEvidenceError(
                        f"{approach} fold membership is not grouped-OOF exact"
                    )
                fresh_fold = next(
                    item
                    for item in fresh_execution.folds
                    if item.repeat == repeat
                    and item.fold == fold["fold_index"]
                )
                model_payload = _read_object(
                    model_path,
                    label=f"{approach} fold model",
                )
                if canonical_json(model_payload) != canonical_json(
                    dict(fresh_fold.model_artifact)
                ):
                    raise WO18BlockedEvidenceError(
                        f"{approach} fold model was not fit from the "
                        "frozen training groups"
                    )
                if canonical_json(fold_rows) != canonical_json(
                    list(fresh_fold.validation_rows[approach])
                ):
                    raise WO18BlockedEvidenceError(
                        f"{approach} fold rows differ from fresh execution"
                    )
                subset_truth = {
                    case_id: truth[case_id]
                    for case_id in split.validation_case_ids
                }
                fold_score = official_evaluate.build_results(
                    subset_truth, fold_rows
                )[0]
                fold_scores_by_approach[approach].append(
                    float(fold_score["scores"]["total_score"])
                )
                artifact_hashes.extend(
                    (
                        _sha256_file(model_path),
                        _sha256_file(fold_prediction_path),
                    )
                )
            score = _validate_score(
                score_path=root / f"{approach}-repeat-{repeat}-score.json",
                truth=truth,
                prediction_path=prediction_path,
            )
            repeat_scores.append(float(score["scores"]["total_score"]))
            artifact_hashes.extend(
                (
                    _sha256_file(prediction_path),
                    _sha256_file(rerun_path),
                    _sha256_file(root / f"{approach}-repeat-{repeat}-score.json"),
                )
            )
        summaries[approach] = {
            "total_score": statistics.mean(repeat_scores),
            "score_min": min(repeat_scores),
            "score_max": max(repeat_scores),
            "repeat_scores": repeat_scores,
            "repeat_count": REQUIRED_REPEATS,
            "fold_count": REQUIRED_REPEATS * REQUIRED_FOLDS,
        }
        artifact_hashes.append(_sha256_file(manifest_path))
    candidate_scores = summaries[CANDIDATE_APPROACH]["repeat_scores"]
    control_scores = summaries[CONTROL_APPROACH]["repeat_scores"]
    deltas = [
        candidate - control
        for candidate, control in zip(candidate_scores, control_scores)
    ]
    if any(not math.isclose(delta, 0.0, abs_tol=1e-12) for delta in deltas):
        raise WO18BlockedEvidenceError(
            "this closure is only valid for the observed zero hybrid delta"
        )
    fold_deltas = [
        candidate - control
        for candidate, control in zip(
            fold_scores_by_approach[CANDIDATE_APPROACH],
            fold_scores_by_approach[CONTROL_APPROACH],
        )
    ]
    if len(fold_deltas) != REQUIRED_REPEATS * REQUIRED_FOLDS or any(
        not math.isclose(delta, 0.0, abs_tol=1e-12)
        for delta in fold_deltas
    ):
        raise WO18BlockedEvidenceError(
            "the observed hybrid must have zero delta in all 15 folds"
        )
    return {
        "summaries": summaries,
        "repeat_deltas": deltas,
        "fold_deltas": fold_deltas,
        "evaluator_sha256": evaluator_sha,
        "artifact_hashes": artifact_hashes,
    }


def _validate_contract(
    *,
    root: Path,
    source_revision: str,
    layout_sha: str,
    coverage_sha: str,
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    for relative in (
        "mib_pipeline/model_recovery.py",
        "devtools/decision_recovery_contract_probe.py",
    ):
        if (REPO_ROOT / relative).read_bytes() != _git_bytes(
            source_revision, relative
        ):
            raise WO18BlockedEvidenceError(
                "live contract probe graph differs from evaluated source"
            )
    fixture_path = root / "contract" / "contract-fixture.json"
    fixture = _read_object(fixture_path, label="contract fixture")
    if canonical_json(fixture) != canonical_json(contract_fixture_payload()):
        raise WO18BlockedEvidenceError(
            "contract fixture does not match the executable probes"
        )
    fixture_sha = _sha256_file(fixture_path)
    runtime_names = _runtime_feature_names()
    schema_findings = _feature_schema_findings(runtime_names)
    model_source = REPO_ROOT / "mib_pipeline" / "model_recovery.py"
    source_findings = _model_specific_source_findings(model_source)
    model_paths = sorted((root / "cv" / "models").glob("*.json"))
    if len(model_paths) != REQUIRED_REPEATS * REQUIRED_FOLDS:
        raise WO18BlockedEvidenceError(
            "contract scan requires the exact 15 fitted fold models"
        )
    runtime_findings = RuntimeLeakageScanner().scan(
        (model_source, fixture_path, *model_paths)
    )
    finding_count = (
        len(schema_findings) + len(source_findings) + len(runtime_findings)
    )
    audits: list[dict[str, Any]] = []
    audit_hashes: list[str] = []
    for repeat_index in (1, 2):
        path = root / "contract" / f"contract-audit-{repeat_index}.json"
        payload = _read_object(path, label="contract audit")
        _require_exact_keys(
            payload,
            {
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
            },
            label="contract audit",
        )
        bindings = {
            "schema_version": CONTRACT_AUDIT_SCHEMA,
            "repeat_index": repeat_index,
            "source_revision_sha": source_revision,
            "layout_manifest_sha256": layout_sha,
            "protected_role_manifest_sha256": coverage_sha,
            "input_tree_sha256": capture["input_tree_sha256"],
            "feature_schema_sha256": capture["feature_schema_sha256"],
            "feature_names": list(runtime_names),
            "contract_fixture_sha256": fixture_sha,
            "deterministic": True,
        }
        for name, expected in bindings.items():
            _same(f"contract audit {name}", payload[name], expected)
        if payload["counts"].get("forbidden_feature_finding_count") != finding_count:
            raise WO18BlockedEvidenceError(
                "contract leakage count does not match fresh scans"
            )
        audit = DecisionRecoveryContractAudit(
            counts=payload["counts"],
            leakage_clean=finding_count == 0,
            feature_schema_exact=not schema_findings,
            deterministic=payload["deterministic"],
        )
        if (
            not audit.probes_exercised
            or not audit.unsafe_counts_zero
            or not audit.leakage_clean
            or not audit.feature_schema_exact
        ):
            raise WO18BlockedEvidenceError("contract audit did not pass")
        audits.append(payload)
        audit_hashes.append(_sha256_file(path))
    normalized = []
    for audit in audits:
        value = dict(audit)
        value.pop("repeat_index")
        normalized.append(canonical_json(value))
    if normalized[0] != normalized[1]:
        raise WO18BlockedEvidenceError(
            "contract audit reruns are not fixture-deterministic"
        )
    fresh_counts_one = run_contract_probes()
    fresh_counts_two = run_contract_probes()
    if (
        canonical_json(fresh_counts_one) != canonical_json(fresh_counts_two)
        or canonical_json(fresh_counts_one)
        != canonical_json(audits[0]["counts"])
    ):
        raise WO18BlockedEvidenceError(
            "fresh contract probe counts do not match both audits"
        )
    return {
        "fixture_sha256": fixture_sha,
        "audit_hashes": audit_hashes,
        "finding_count": finding_count,
        "probe_count": sum(
            audits[0]["counts"][name]
            for name in DecisionRecoveryContractAudit.REQUIRED_POSITIVE_COUNTS
        ),
    }


def _require_repository_safe(value: Any, *, key: str = "$") -> None:
    """Reject identity, filesystem, and record-shaped data in committed output."""

    forbidden_keys = {
        "case_id",
        "case_ids",
        "filename",
        "file_path",
        "predictions_path",
        "truth_rows",
        "feature_rows",
    }
    if isinstance(value, Mapping):
        for raw_name, child in value.items():
            name = str(raw_name).strip().casefold()
            if name in forbidden_keys or name.endswith(("_path", "_paths")):
                raise WO18BlockedEvidenceError(
                    "repository evidence contains an identity/path dimension"
                )
            _require_repository_safe(child, key=name)
        return
    if isinstance(value, (list, tuple)):
        if any(isinstance(child, Mapping) for child in value):
            raise WO18BlockedEvidenceError(
                "repository evidence contains record-shaped rows"
            )
        for child in value:
            _require_repository_safe(child, key=key)
        return
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.startswith(("/", "~")) or _CASE_ID_RE.search(normalized):
            raise WO18BlockedEvidenceError(
                "repository evidence contains identity or filesystem data"
            )
    canonical_json(value)


def build_blocked_evidence(
    *,
    artifact_root: Path,
    layout_manifest_path: Path,
    truth_path: Path,
    source_revision_sha: str,
) -> dict[str, Any]:
    """Validate the external run and return aggregate-only blocked evidence."""

    source_revision = _revision(source_revision_sha)
    layout = load_layout_manifest(layout_manifest_path)
    if (
        not layout.frozen_before_scoring
        or len(layout.case_ids) != EXPECTED_RECORD_COUNT
        or len(layout.groups) < REQUIRED_FOLDS
    ):
        raise WO18BlockedEvidenceError(
            "layout manifest must be frozen with 32 cases and at least 5 groups"
        )
    capture = _validate_capture(
        root=artifact_root,
        source_revision=source_revision,
        layout_sha=layout.sha256,
    )
    if capture["candidate_runtime_composed"]:
        raise WO18BlockedEvidenceError(
            "blocked closure requires the candidate to remain outside "
            "the evaluated production composition"
        )
    truth, truth_sha = _load_truth(truth_path)
    coverage = _validate_coverage_gap(
        root=artifact_root,
        source_revision=source_revision,
        layout_sha=layout.sha256,
        capture=capture,
        truth_sha=truth_sha,
        truth=truth,
        layout=layout,
    )
    arms = _validate_arms(
        root=artifact_root,
        source_revision=source_revision,
        layout=layout,
        capture=capture,
        coverage_sha=coverage["sha256"],
        truth=truth,
        truth_sha=truth_sha,
    )
    contract = _validate_contract(
        root=artifact_root,
        source_revision=source_revision,
        layout_sha=layout.sha256,
        coverage_sha=coverage["sha256"],
        capture=capture,
    )
    all_hashes = sorted(
        {
            layout.sha256,
            truth_sha,
            capture["feature_rows_sha256"],
            capture["observation_sha256"],
            coverage["sha256"],
            contract["fixture_sha256"],
            *contract["audit_hashes"],
            *arms["artifact_hashes"],
        }
    )
    artifact_set_sha = hashlib.sha256(
        canonical_json(all_hashes).encode("utf-8")
    ).hexdigest()
    class_metrics = {
        approach: {
            "total_score": values["total_score"],
            "score_min": values["score_min"],
            "score_max": values["score_max"],
            "repeat_count": values["repeat_count"],
            "fold_count": values["fold_count"],
        }
        for approach, values in arms["summaries"].items()
    }
    aggregate = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "blocked_precondition",
        "comparison_scope": "public_grouped_robustness_not_unseen",
        "source_revision_sha": source_revision,
        "layout_manifest_sha256": layout.sha256,
        "truth_sha256": truth_sha,
        "input_tree_sha256": capture["input_tree_sha256"],
        "feature_schema_sha256": capture["feature_schema_sha256"],
        "frozen_feature_rows_sha256": capture["feature_rows_sha256"],
        "protected_role_coverage_gap_sha256": coverage["sha256"],
        "legacy_misnamed_coverage_gap_binding_sha256": coverage["sha256"],
        "official_evaluator_sha256": arms["evaluator_sha256"],
        "contract_fixture_sha256": contract["fixture_sha256"],
        "production_capture_observation_sha256": capture[
            "observation_sha256"
        ],
        "production_capture_producer_graph_sha256": capture[
            "producer_graph_sha256"
        ],
        "production_capture_producer_source_sha256": capture[
            "producer_source_sha256"
        ],
        "artifact_set_sha256": artifact_set_sha,
        "score_delta": 0.0,
        "repeat_scores": arms["summaries"][CANDIDATE_APPROACH][
            "repeat_scores"
        ],
        "repeat_score_deltas": arms["repeat_deltas"],
        "fold_score_deltas": arms["fold_deltas"],
        "counts": {
            "record_count": EXPECTED_RECORD_COUNT,
            "layout_group_count": len(layout.groups),
            "approach_count": len(APPROACHES),
            "repeat_count": REQUIRED_REPEATS,
            "fold_count": REQUIRED_REPEATS * REQUIRED_FOLDS,
            "eligible_role_record_count": coverage["eligible_count"],
            "unassigned_role_record_count": coverage["unassigned_count"],
            "missing_required_role_count": len(EXPECTED_MISSING_ROLES),
            "contract_audit_count": 2,
            "contract_probe_count": contract["probe_count"],
            "leakage_finding_count": contract["finding_count"],
        },
        "checks": {
            "capture_byte_deterministic": True,
            "capture_binding_verified": True,
            "grouped_oof_manifest_verified": True,
            "official_evaluator_verified": True,
            "contract_fixture_deterministic": True,
            "contract_audits_clean": True,
            "identity_leakage_clean": contract["finding_count"] == 0,
            "standard_promotion_gate_evaluated": False,
            "evaluated_revision_context_verified": True,
            "candidate_runtime_composed": capture[
                "candidate_runtime_composed"
            ],
        },
        "gate_results": {
            "protected_role_coverage_complete": False,
            "hybrid_delta_strictly_positive": False,
            "candidate_disabled": True,
            "candidate_runtime_enabled": False,
            "model_promoted": False,
        },
        "class_metrics": class_metrics,
        "field_metrics": coverage["coverage"],
    }
    _require_repository_safe(aggregate)
    return aggregate


def render_markdown(evidence: Mapping[str, Any]) -> str:
    """Render the identity-free closure rationale."""

    _require_repository_safe(evidence)
    metrics = evidence["class_metrics"]
    lines = [
        "# WO-18 Identity-Free Decision Recovery",
        "",
        (
            "**Status: BLOCKED AT PRECONDITION — candidate disabled, no "
            "model promoted.**"
        ),
        "",
        (
            "This is public-exposed grouped robustness evidence, not an "
            "unseen holdout and not the full-public 1,000-case evaluation."
        ),
        "",
        "## Why promotion is blocked",
        "",
        (
            "1. **Protected-role coverage is impossible on the frozen cohort.** "
            "The `binding_authority` and `approval_guard` roles have zero "
            "eligible cases and zero eligible layout groups. Assigning either "
            "role would fabricate evidence, so the required five-role gate "
            "cannot be evaluated honestly. Consequently, the standard "
            "`DecisionRecoveryGate` was not evaluated."
        ),
        (
            "2. **The gated hybrid adds no value.** Across all three grouped "
            "OOF repeats its official-evaluator score is byte-for-byte "
            "equivalent to the deterministic engine, producing a delta of "
            "`0.000000` in every repeat. The gate requires a strictly positive "
            "delta."
        ),
        "",
        "## Four-arm official-evaluator results",
        "",
        "| Approach | Mean score | Minimum | Maximum |",
        "|---|---:|---:|---:|",
    ]
    for approach in APPROACHES:
        value = metrics[approach]
        lines.append(
            f"| `{approach}` | {value['total_score']:.6f} | "
            f"{value['score_min']:.6f} | {value['score_max']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Verified controls",
            "",
            "- Two production captures are byte-identical and fully bound.",
            "- All four arms contain exact 3×5 grouped OOF manifests.",
            "- Every repeat score was recomputed with the official evaluator.",
            "- Two contract audits match the executable fixture deterministically.",
            "- Contract, feature-schema, model-source, and runtime leakage scans are clean.",
            (
                "- The historical CV/audit field named "
                "`protected_role_manifest_sha256` is treated only as a "
                "legacy-misnamed coverage-gap binding; it is not evidence of "
                "a valid protected-role manifest."
            ),
            "- The evaluated production composition does not include the candidate adjudicator.",
            "- Identity-bearing cases, paths, truth rows, predictions, and folds remain external.",
            "",
            f"Evaluated source: `{evidence['source_revision_sha']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        evidence = build_blocked_evidence(
            artifact_root=arguments.artifact_root,
            layout_manifest_path=arguments.layout_manifest,
            truth_path=arguments.truth,
            source_revision_sha=arguments.source_revision_sha,
        )
    except ExperimentControlError as exc:
        parser.error(str(exc))
    _atomic_write(arguments.output_json, canonical_json(evidence) + "\n")
    _atomic_write(arguments.output_markdown, render_markdown(evidence))
    print("WO-18 promotion gate: BLOCKED (candidate disabled)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
