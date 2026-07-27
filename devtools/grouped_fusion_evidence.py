#!/usr/bin/env python3
"""Build aggregate-only WO-16 grouped evidence-fusion evidence.

Identity-bearing layout groups, truth, and predictions remain external.  The
builder uses the official evaluator for the full comparison and every
group-exclusive fold, validates repeated legacy-control and candidate runs,
and emits only the aggregate gate decision bound to a source revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.experiment_control import (  # noqa: E402
    ExperimentControlError,
    RepeatedGroupedSplitManager,
    canonical_json,
    require_aggregate_only,
)
from devtools.grouped_fusion_gate import (  # noqa: E402
    REQUIRED_FOLDS,
    REQUIRED_REPEATS,
    FoldScorePair,
    FusionAuditAggregate,
    FusionRunAggregate,
    GroupedFusionEvidence,
    GroupedFusionGate,
)
from devtools.fusion_audit_contract import (  # noqa: E402
    FUSION_AUDIT_COMPARISON_SCOPE,
    FUSION_AUDIT_INVOCATION_SCOPE,
)
from devtools.ocr_ablation import (  # noqa: E402
    AblationConfigurationError,
    _read_json as _read_ablation_json,
    _validate_observation as _validate_ablation_observation,
)
from devtools.grouped_recovery_evidence import (  # noqa: E402
    LAYOUT_MANIFEST_SCHEMA,
    FrozenLayoutManifest,
    _filter_rows,
    _invalid_record_count,
    _read_distinct_runs,
    _read_json_object,
    _read_truth_subset,
    _rows_fingerprint,
    load_layout_manifest,
)
from scripts import evaluate as official_evaluate  # noqa: E402


_SOURCE_DIGEST_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class GroupedFusionEvidenceBuildError(ExperimentControlError):
    """An external WO-16 evidence input is incomplete or malformed."""


@dataclass(frozen=True)
class PredictionRuns:
    """One deterministic prediction arm plus its official full aggregate."""

    rows: tuple[Mapping[str, Any], ...]
    full: FusionRunAggregate


def verify_clean_candidate_checkout(
    repo_root: Path, expected_source_revision_sha: str
) -> None:
    """Require the CLI candidate revision to be the fully clean checkout."""

    expected = str(expected_source_revision_sha).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise GroupedFusionEvidenceBuildError(
            "candidate checkout proof requires a full Git commit SHA"
        )
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise GroupedFusionEvidenceBuildError(
            "candidate checkout could not be verified"
        ) from exc
    if head.returncode != 0 or status.returncode != 0:
        raise GroupedFusionEvidenceBuildError(
            "candidate checkout could not be verified"
        )
    if head.stdout.strip().lower() != expected:
        raise GroupedFusionEvidenceBuildError(
            "candidate source revision does not match checkout HEAD"
        )
    if status.stdout.strip():
        raise GroupedFusionEvidenceBuildError(
            "candidate checkout has working tree modifications"
        )


def verify_legacy_commit(
    repo_root: Path, expected_source_revision_sha: str
) -> None:
    """Require the named legacy control revision to exist as a local commit."""

    expected = str(expected_source_revision_sha).strip().lower()
    if not _GIT_COMMIT_RE.fullmatch(expected):
        raise GroupedFusionEvidenceBuildError(
            "legacy control requires a full Git commit SHA"
        )
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "cat-file",
                "-e",
                f"{expected}^{{commit}}",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise GroupedFusionEvidenceBuildError(
            "legacy control commit could not be verified"
        ) from exc
    if completed.returncode != 0:
        raise GroupedFusionEvidenceBuildError(
            "legacy control source revision is not a local Git commit"
        )


def _load_arm_observations(
    *,
    arm: str,
    observation_paths: Sequence[Path | str],
    prediction_paths: Sequence[Path | str],
    expected_source_revision_sha: str,
) -> tuple[dict[str, Any], ...]:
    """Validate two source- and file-bound production observations."""

    if len(observation_paths) != 2 or len(prediction_paths) != 2:
        raise GroupedFusionEvidenceBuildError(
            f"{arm} requires exactly two observations and prediction files"
        )
    resolved_observation_paths = tuple(
        Path(path).resolve() for path in observation_paths
    )
    if len(set(resolved_observation_paths)) != 2:
        raise GroupedFusionEvidenceBuildError(
            f"{arm} requires two distinct observation artifacts"
        )
    observations: list[dict[str, Any]] = []
    for path in resolved_observation_paths:
        try:
            observations.append(
                _validate_ablation_observation(
                    path, _read_ablation_json(path)
                )
            )
        except AblationConfigurationError as exc:
            raise GroupedFusionEvidenceBuildError(
                f"{arm} observation is invalid: {exc}"
            ) from exc

    expected_revision = str(expected_source_revision_sha).strip().lower()
    if any(
        str(observation["source_revision"]).strip().lower()
        != expected_revision
        for observation in observations
    ):
        raise GroupedFusionEvidenceBuildError(
            f"{arm} observation source revision does not match its arm"
        )
    if any(
        str(observation["variant_id"]).strip() != "baseline"
        for observation in observations
    ):
        raise GroupedFusionEvidenceBuildError(
            f"{arm} observations must use the production baseline config"
        )
    repeat_indexes: set[int] = set()
    for observation in observations:
        repeat_index = observation["repeat_index"]
        if isinstance(repeat_index, bool) or not isinstance(
            repeat_index, int
        ):
            raise GroupedFusionEvidenceBuildError(
                f"{arm} observation repeat_index must be an integer"
            )
        repeat_indexes.add(repeat_index)
    if repeat_indexes != {1, 2}:
        raise GroupedFusionEvidenceBuildError(
            f"{arm} observations must contain repeats 1 and 2"
        )

    supplied_predictions = {
        Path(path).resolve() for path in prediction_paths
    }
    if len(supplied_predictions) != 2:
        raise GroupedFusionEvidenceBuildError(
            f"{arm} determinism evidence requires distinct prediction "
            "artifacts"
        )
    observed_predictions = {
        Path(str(observation["predictions_path"])).resolve()
        for observation in observations
    }
    if (
        observed_predictions != supplied_predictions
    ):
        raise GroupedFusionEvidenceBuildError(
            f"{arm} observations do not bind the supplied prediction files"
        )

    consistency_keys = (
        "benchmark_id",
        "config_sha256",
        "input_tree_sha256",
        "input_pdf_count",
        "max_workers",
        "metrics_source",
    )
    for key in consistency_keys:
        if len(
            {
                canonical_json(observation.get(key))
                for observation in observations
            }
        ) != 1:
            raise GroupedFusionEvidenceBuildError(
                f"{arm} observations disagree on {key}"
            )

    input_tree_sha256 = str(
        observations[0]["input_tree_sha256"]
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", input_tree_sha256):
        raise GroupedFusionEvidenceBuildError(
            f"{arm} input_tree_sha256 must be a full SHA-256 digest"
        )
    input_pdf_count = observations[0]["input_pdf_count"]
    if (
        isinstance(input_pdf_count, bool)
        or not isinstance(input_pdf_count, int)
        or input_pdf_count < 1
    ):
        raise GroupedFusionEvidenceBuildError(
            f"{arm} input_pdf_count must be a positive integer"
        )
    max_workers = observations[0]["max_workers"]
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= 4
    ):
        raise GroupedFusionEvidenceBuildError(
            f"{arm} max_workers must be between 1 and 4"
        )
    if not str(observations[0]["benchmark_id"]).strip():
        raise GroupedFusionEvidenceBuildError(
            f"{arm} benchmark_id must be non-empty"
        )
    for observation in observations:
        for name in ("attempted", "answered", "omitted"):
            value = observation[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise GroupedFusionEvidenceBuildError(
                    f"{arm} observation {name} must be an integer"
                )
        if (
            observation["attempted"] != input_pdf_count
            or observation["answered"] != input_pdf_count
            or observation["omitted"] != 0
        ):
            raise GroupedFusionEvidenceBuildError(
                f"{arm} observation run is incomplete"
            )
    return tuple(
        sorted(observations, key=lambda value: int(value["repeat_index"]))
    )


def _observation_set_sha256(
    observations: Sequence[Mapping[str, Any]],
) -> str:
    """Hash only non-path observation bindings for aggregate publication."""

    bound = [
        {
            key: observation[key]
            for key in (
                "benchmark_id",
                "variant_id",
                "repeat_index",
                "source_revision",
                "config_sha256",
                "input_tree_sha256",
                "input_pdf_count",
                "max_workers",
                "predictions_sha256",
                "attempted",
                "answered",
                "omitted",
                "metrics_source",
            )
        }
        for observation in observations
    ]
    return hashlib.sha256(canonical_json(bound).encode("utf-8")).hexdigest()


def _safety_event_case_ids(
    truth: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[frozenset[str], frozenset[str]]:
    """Return internal paired safety-event sets; callers emit counts only."""

    predictions, _, _ = official_evaluate.index_submission(rows)
    catastrophic = frozenset(
        case_id
        for case_id, truth_row in truth.items()
        if case_id in predictions
        and official_evaluate.score_case(
            case_id, truth_row, predictions[case_id]
        )["catastrophic_false_approval"]
    )
    false_positive_denials = frozenset(
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
    return catastrophic, false_positive_denials


def _full_run(
    truth: Mapping[str, Mapping[str, Any]],
    runs: Sequence[tuple[Mapping[str, Any], ...]],
) -> PredictionRuns:
    aggregates = [official_evaluate.build_results(truth, rows)[0] for rows in runs]
    deterministic = (
        len({_rows_fingerprint(rows) for rows in runs}) == 1
        and len({canonical_json(aggregate) for aggregate in aggregates}) == 1
    )
    rows = runs[0]
    aggregate = aggregates[0]
    counts = aggregate["counts"]
    _, false_positive_denials = _safety_event_case_ids(truth, rows)
    return PredictionRuns(
        rows=rows,
        full=FusionRunAggregate(
            total_score=float(aggregate["scores"]["total_score"]),
            record_count=int(counts["scored_predictions"]),
            catastrophic_false_approvals=int(
                aggregate["raw"]["catastrophic_false_approvals"]
            ),
            false_positive_denials=len(false_positive_denials),
            missing_records=int(counts["missing_cases"]),
            invalid_records=_invalid_record_count(aggregate),
            deterministic=deterministic,
            duplicate_records=int(counts["duplicate_case_ids"]),
            extra_records=int(counts["extra_cases"]),
        ),
    )


def load_fusion_audit(
    path: Path | str,
    *,
    expected_source_revision_sha: str,
    expected_input_tree_sha256: str | None = None,
    expected_input_pdf_count: int | None = None,
    expected_case_id_set_sha256: str | None = None,
) -> FusionAuditAggregate:
    """Load counters and verify their source, input, and identity-free cohort."""

    payload = _read_json_object(Path(path), label="fusion audit")
    if payload.get("comparison_scope") != FUSION_AUDIT_COMPARISON_SCOPE:
        raise GroupedFusionEvidenceBuildError(
            "fusion audit comparison_scope is missing or unsupported"
        )
    if payload.get("invocation_scope") != FUSION_AUDIT_INVOCATION_SCOPE:
        raise GroupedFusionEvidenceBuildError(
            "fusion audit invocation_scope is missing or unsupported"
        )
    require_aggregate_only(payload)
    expected_digest = str(expected_source_revision_sha).strip().lower()
    if not _SOURCE_DIGEST_RE.fullmatch(expected_digest):
        raise GroupedFusionEvidenceBuildError(
            "expected fusion audit source_revision_sha must be a full "
            "Git commit or SHA-256 digest"
        )
    audit_digest = str(payload.get("source_revision_sha", "")).strip().lower()
    if not _SOURCE_DIGEST_RE.fullmatch(audit_digest):
        raise GroupedFusionEvidenceBuildError(
            "fusion audit must include a full source_revision_sha"
        )
    if audit_digest != expected_digest:
        raise GroupedFusionEvidenceBuildError(
            "fusion audit source_revision_sha does not match the "
            "evidence source revision"
        )
    audit_input_tree_sha256 = str(
        payload.get("input_tree_sha256", "")
    ).strip().lower()
    audit_case_id_set_sha256 = str(
        payload.get("case_id_set_sha256", "")
    ).strip().lower()
    audit_input_pdf_count = payload.get("input_pdf_count")
    if not re.fullmatch(r"[0-9a-f]{64}", audit_input_tree_sha256):
        raise GroupedFusionEvidenceBuildError(
            "fusion audit requires a full input_tree_sha256"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", audit_case_id_set_sha256):
        raise GroupedFusionEvidenceBuildError(
            "fusion audit requires a full case_id_set_sha256"
        )
    if (
        isinstance(audit_input_pdf_count, bool)
        or not isinstance(audit_input_pdf_count, int)
        or audit_input_pdf_count < 1
    ):
        raise GroupedFusionEvidenceBuildError(
            "fusion audit requires a positive input_pdf_count"
        )
    if (
        expected_input_tree_sha256 is not None
        and audit_input_tree_sha256
        != str(expected_input_tree_sha256).strip().lower()
    ):
        raise GroupedFusionEvidenceBuildError(
            "fusion audit input tree does not match prediction observations"
        )
    if (
        expected_input_pdf_count is not None
        and audit_input_pdf_count != expected_input_pdf_count
    ):
        raise GroupedFusionEvidenceBuildError(
            "fusion audit input count does not match the frozen cohort"
        )
    if (
        expected_case_id_set_sha256 is not None
        and audit_case_id_set_sha256
        != str(expected_case_id_set_sha256).strip().lower()
    ):
        raise GroupedFusionEvidenceBuildError(
            "fusion audit case cohort does not match the frozen manifest"
        )

    nested = payload.get("counts")
    if nested is None:
        counts: Mapping[str, Any] = payload
    elif isinstance(nested, Mapping):
        counts = nested
    else:
        raise GroupedFusionEvidenceBuildError(
            "fusion audit counts must be an object"
        )
    required = (
        "changed_field_count",
        "changed_field_complete_provenance_count",
        "clean_higher_authority_override_count",
        "binding_authority_override_count",
        "text_layer_winner_count",
        "serialization_default_used_as_evidence_count",
        "correlated_views_collapsed",
        "independent_agreement_resolutions",
        "same_rank_contested_count",
        "cross_applicant_candidates_excluded",
    )
    missing = tuple(name for name in required if name not in counts)
    if missing:
        raise GroupedFusionEvidenceBuildError(
            "fusion audit is missing required aggregate counters: "
            + ", ".join(missing)
        )
    return FusionAuditAggregate(
        changed_field_count=counts["changed_field_count"],
        changed_field_complete_provenance_count=counts[
            "changed_field_complete_provenance_count"
        ],
        clean_higher_authority_override_count=counts[
            "clean_higher_authority_override_count"
        ],
        binding_authority_override_count=counts[
            "binding_authority_override_count"
        ],
        text_layer_winner_count=counts["text_layer_winner_count"],
        serialization_default_used_as_evidence_count=counts[
            "serialization_default_used_as_evidence_count"
        ],
        correlated_views_collapsed=counts["correlated_views_collapsed"],
        independent_agreement_resolutions=counts[
            "independent_agreement_resolutions"
        ],
        same_rank_contested_count=counts["same_rank_contested_count"],
        cross_applicant_candidates_excluded=counts[
            "cross_applicant_candidates_excluded"
        ],
    )


def _case_id_set_sha256(case_ids: Sequence[str]) -> str:
    normalized = sorted(str(case_id).strip() for case_id in case_ids)
    return hashlib.sha256(
        canonical_json(normalized).encode("utf-8")
    ).hexdigest()


def _fold_pairs(
    *,
    manifest: FrozenLayoutManifest,
    truth: Mapping[str, Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[FoldScorePair, ...], bool, bool, bool]:
    manager = RepeatedGroupedSplitManager(
        seed=manifest.split_seed,
        repeats=REQUIRED_REPEATS,
        folds=REQUIRED_FOLDS,
    )
    splits = manager.split_groups(manifest.groups)
    split_deterministic = splits == manager.split_groups(manifest.groups)
    group_exclusive = all(
        not (set(split.tuning_groups) & set(split.validation_groups))
        and not (set(split.tuning_case_ids) & set(split.validation_case_ids))
        for split in splits
    )
    paired_fold_members = all(
        set(split.validation_case_ids).issubset(truth) for split in splits
    )

    pairs: list[FoldScorePair] = []
    for split in splits:
        fold_truth = {
            case_id: truth[case_id] for case_id in split.validation_case_ids
        }
        control_aggregate, _ = official_evaluate.build_results(
            fold_truth,
            _filter_rows(control_rows, split.validation_case_ids),
        )
        candidate_aggregate, _ = official_evaluate.build_results(
            fold_truth,
            _filter_rows(candidate_rows, split.validation_case_ids),
        )
        member_count = len(split.validation_case_ids)
        pairs.append(
            FoldScorePair(
                repeat=split.repeat,
                fold=split.fold,
                control_score=float(control_aggregate["scores"]["total_score"]),
                candidate_score=float(
                    candidate_aggregate["scores"]["total_score"]
                ),
                control_record_count=member_count,
                candidate_record_count=member_count,
                layout_group_count=len(split.validation_groups),
            )
        )
    return tuple(pairs), group_exclusive, paired_fold_members, split_deterministic


def build_aggregate_evidence(
    *,
    layout_manifest_path: Path | str,
    expected_layout_manifest_sha256: str,
    expected_input_tree_sha256: str,
    truth_path: Path | str,
    legacy_control_prediction_paths: Sequence[Path | str],
    candidate_prediction_paths: Sequence[Path | str],
    legacy_control_observation_paths: Sequence[Path | str],
    candidate_observation_paths: Sequence[Path | str],
    fusion_audit_path: Path | str,
    legacy_control_source_revision_sha: str,
    source_revision_sha: str,
    checkout_verifier: Callable[[Path, str], None] | None = None,
    legacy_revision_verifier: Callable[[Path, str], None] | None = None,
) -> dict[str, Any]:
    """Build the only public, repository-safe form of WO-16 evidence."""

    legacy_source_digest = str(
        legacy_control_source_revision_sha
    ).strip().lower()
    if not _GIT_COMMIT_RE.fullmatch(legacy_source_digest):
        raise GroupedFusionEvidenceBuildError(
            "legacy_control_source_revision_sha must be a full Git commit SHA"
        )
    source_digest = str(source_revision_sha).strip().lower()
    if not _GIT_COMMIT_RE.fullmatch(source_digest):
        raise GroupedFusionEvidenceBuildError(
            "source_revision_sha must be a full Git commit SHA"
        )
    expected_input_digest = str(
        expected_input_tree_sha256
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_input_digest):
        raise GroupedFusionEvidenceBuildError(
            "expected_input_tree_sha256 must be a full SHA-256 digest"
        )
    verifier = checkout_verifier or verify_clean_candidate_checkout
    verifier(REPO_ROOT, source_digest)
    legacy_verifier = legacy_revision_verifier or verify_legacy_commit
    legacy_verifier(REPO_ROOT, legacy_source_digest)
    manifest = load_layout_manifest(layout_manifest_path)
    expected_manifest_digest = str(expected_layout_manifest_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_digest):
        raise GroupedFusionEvidenceBuildError(
            "expected_layout_manifest_sha256 must be a full SHA-256 digest"
        )
    if manifest.sha256 != expected_manifest_digest:
        raise GroupedFusionEvidenceBuildError(
            "layout manifest bytes do not match the pre-recorded frozen digest"
        )

    legacy_observations = _load_arm_observations(
        arm="legacy control",
        observation_paths=legacy_control_observation_paths,
        prediction_paths=legacy_control_prediction_paths,
        expected_source_revision_sha=legacy_source_digest,
    )
    candidate_observations = _load_arm_observations(
        arm="candidate",
        observation_paths=candidate_observation_paths,
        prediction_paths=candidate_prediction_paths,
        expected_source_revision_sha=source_digest,
    )
    all_observations = legacy_observations + candidate_observations
    for key in (
        "benchmark_id",
        "config_sha256",
        "input_tree_sha256",
        "input_pdf_count",
        "max_workers",
        "metrics_source",
    ):
        if len(
            {
                canonical_json(observation.get(key))
                for observation in all_observations
            }
        ) != 1:
            raise GroupedFusionEvidenceBuildError(
                f"control and candidate observations disagree on {key}"
            )
    input_pdf_count = candidate_observations[0]["input_pdf_count"]
    input_tree_sha256 = str(
        candidate_observations[0]["input_tree_sha256"]
    ).strip().lower()
    if input_tree_sha256 != expected_input_digest:
        raise GroupedFusionEvidenceBuildError(
            "prediction observations do not match the pre-recorded input "
            "tree digest"
        )
    if input_pdf_count != len(manifest.case_ids):
        raise GroupedFusionEvidenceBuildError(
            "prediction observation count does not match the frozen cohort"
        )
    case_id_set_sha256 = _case_id_set_sha256(manifest.case_ids)
    audit = load_fusion_audit(
        fusion_audit_path,
        expected_source_revision_sha=source_digest,
        expected_input_tree_sha256=expected_input_digest,
        expected_input_pdf_count=input_pdf_count,
        expected_case_id_set_sha256=case_id_set_sha256,
    )
    truth = _read_truth_subset(truth_path, manifest.case_ids)
    legacy_control = _full_run(
        truth,
        _read_distinct_runs(
            legacy_control_prediction_paths, arm="legacy control"
        ),
    )
    candidate = _full_run(
        truth,
        _read_distinct_runs(candidate_prediction_paths, arm="candidate"),
    )
    legacy_catastrophic, legacy_false_positive_denials = (
        _safety_event_case_ids(truth, legacy_control.rows)
    )
    candidate_catastrophic, candidate_false_positive_denials = (
        _safety_event_case_ids(truth, candidate.rows)
    )
    folds, group_exclusive, paired, split_deterministic = _fold_pairs(
        manifest=manifest,
        truth=truth,
        control_rows=legacy_control.rows,
        candidate_rows=candidate.rows,
    )
    evidence = GroupedFusionEvidence(
        layout_manifest_sha256=manifest.sha256,
        expected_record_count=len(manifest.case_ids),
        expected_layout_group_count=len(manifest.groups),
        legacy_control_full=legacy_control.full,
        candidate_full=candidate.full,
        candidate_fusion_audit=audit,
        new_catastrophic_false_approval_count=len(
            candidate_catastrophic - legacy_catastrophic
        ),
        new_false_positive_denial_count=len(
            candidate_false_positive_denials
            - legacy_false_positive_denials
        ),
        folds=folds,
        manifest_frozen_before_scoring=manifest.frozen_before_scoring,
        group_exclusive=group_exclusive,
        paired_fold_members=paired,
        split_deterministic=split_deterministic,
    )
    aggregate = GroupedFusionGate().evaluate(evidence).to_aggregate_evidence()
    aggregate["legacy_control_source_revision_sha"] = legacy_source_digest
    aggregate["source_revision_sha"] = source_digest
    aggregate["input_pdf_count"] = input_pdf_count
    aggregate["input_tree_sha256"] = input_tree_sha256
    aggregate["case_id_set_sha256"] = case_id_set_sha256
    aggregate["legacy_control_observation_set_sha256"] = (
        _observation_set_sha256(legacy_observations)
    )
    aggregate["candidate_observation_set_sha256"] = (
        _observation_set_sha256(candidate_observations)
    )
    require_aggregate_only(aggregate)
    return aggregate


def render_aggregate_markdown(aggregate: Mapping[str, Any]) -> str:
    """Render sanitized evidence without exposing any input identity."""

    require_aggregate_only(aggregate)
    gates = aggregate["gate_results"]
    checks = aggregate["checks"]
    metrics = aggregate["metrics"]
    counts = aggregate["counts"]
    acceptance = "PASS" if aggregate["status"] == "passed" else "FAIL"
    lines = [
        "# WO-16 grouped applicant-aware evidence fusion",
        "",
        f"- Evidence class: `{aggregate['evaluation_mode']}`",
        f"- Status: **{str(aggregate['status']).upper()}**",
        f"- Work Order acceptance: **{acceptance}**",
        f"- Source revision: `{aggregate['source_revision_sha']}`",
        f"- Frozen layout manifest: `{aggregate['layout_manifest_sha256']}`",
        f"- Records / layout groups: {aggregate['expected_record_count']} / "
        f"{aggregate['layout_group_count']}",
        f"- Input PDFs / tree: {aggregate['input_pdf_count']} / "
        f"`{aggregate['input_tree_sha256']}`",
        f"- Identity-free case cohort: "
        f"`{aggregate['case_id_set_sha256']}`",
        f"- Candidate observation binding: "
        f"`{aggregate['candidate_observation_set_sha256']}`",
        f"- Legacy-control observation binding: "
        f"`{aggregate['legacy_control_observation_set_sha256']}`",
        "",
        "## Legacy-control comparison",
        "",
        f"- Legacy-control revision: "
        f"`{aggregate['legacy_control_source_revision_sha']}`",
        f"- Legacy control: {float(aggregate['full_control_score']):.9f}",
        f"- Candidate: {float(aggregate['full_candidate_score']):.9f}",
        f"- Delta: {float(aggregate['score_delta']):+.9f}",
        "",
        "## Repeated grouped robustness",
        "",
        "| Repeat | Weighted delta | Positive folds | Leave-best-fold-out delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for repeat in range(REQUIRED_REPEATS):
        lines.append(
            "| "
            f"{repeat + 1} | "
            f"{float(metrics[f'repeat_{repeat}_weighted_score_delta']):+.9f} | "
            f"{int(metrics[f'repeat_{repeat}_positive_fold_count'])}/"
            f"{REQUIRED_FOLDS} | "
            f"{float(metrics[f'repeat_{repeat}_leave_best_fold_out_delta']):+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Fusion safety and audit",
            "",
            f"- Catastrophic false-approval delta: "
            f"{aggregate['catastrophic_false_approvals_delta']:+d}",
            f"- False-positive denial delta: "
            f"{aggregate['false_positive_denials_delta']:+d}",
            f"- Newly introduced catastrophic false approvals: "
            f"{aggregate['new_catastrophic_false_approval_count']}",
            f"- Newly introduced false-positive denials: "
            f"{aggregate['new_false_positive_denial_count']}",
            f"- Missing / invalid / duplicate / extra records: "
            f"{aggregate['missing_records']} / {aggregate['invalid_records']} / "
            f"{aggregate['duplicate_records']} / {aggregate['extra_records']}",
        f"- Changed fields with complete provenance: "
        f"(resolver-level fusion vs legacy after current linkage; "
        f"one accepted final resolver result per case): "
        f"{counts['changed_field_complete_provenance_count']} / "
        f"{counts['changed_field_count']}",
            f"- Correlated views collapsed: "
            f"{counts['correlated_views_collapsed']}",
            f"- Independent-agreement resolutions: "
            f"{counts['independent_agreement_resolutions']}",
            f"- Same-rank contested outcomes: "
            f"{counts['same_rank_contested_count']}",
            f"- Cross-applicant candidates excluded: "
            f"{counts['cross_applicant_candidates_excluded']}",
            f"- Clean higher-authority / binding-authority overrides: "
            f"{counts['clean_higher_authority_override_count']} / "
            f"{counts['binding_authority_override_count']}",
            f"- Text-layer winners: {counts['text_layer_winner_count']}",
            f"- Serialization defaults used as evidence: "
            f"{counts['serialization_default_used_as_evidence_count']}",
            "",
            "## Robustness checks",
            "",
            "| Diagnostic | Result |",
            "| --- | :---: |",
            f"| `no_negative_folds` (hard) | "
            f"{'PASS' if checks['no_negative_folds'] else 'FAIL'} |",
            f"| `leave_best_fold_out_nonnegative` | "
            f"{'PASS' if checks['leave_best_fold_out_nonnegative'] else 'WARN'} |",
            f"| `fold_majority_positive` | "
            f"{'PASS' if checks['fold_majority_positive'] else 'WARN'} |",
            f"| `leave_best_fold_out_positive` (hard) | "
            f"{'PASS' if checks['leave_best_fold_out_positive'] else 'FAIL'} |",
            "",
        ]
    )
    if checks["score_gain_concentration_warning"]:
        lines.extend(
            [
                "> **Robustness warning:** the aggregate repeated-CV gain is "
                "positive, but it is not uniform across layout folds and/or "
                "becomes non-positive when the strongest fold is omitted. "
                "Negative folds or a non-positive leave-best-fold-out result "
                "block adoption. Fold-majority remains a diagnostic.",
                "",
            ]
        )
    else:
        lines.extend(["- No score-gain concentration warning.", ""])
    lines.extend(["## Hard gates", "", "| Gate | Result |", "| --- | :---: |"])
    for name, passed in gates.items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            "> Public-label-exposed grouped robustness evidence; this is not an "
            "unseen holdout result.",
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
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build aggregate-only WO-16 repeated grouped evidence."
    )
    parser.add_argument("--layout-manifest", required=True)
    parser.add_argument("--expected-layout-manifest-sha256", required=True)
    parser.add_argument("--expected-input-tree-sha256", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument(
        "--legacy-control-prediction",
        action="append",
        required=True,
        help="repeat for each legacy-control run (exactly two)",
    )
    parser.add_argument(
        "--candidate-prediction",
        action="append",
        required=True,
        help="repeat for each candidate run (exactly two)",
    )
    parser.add_argument(
        "--legacy-control-observation",
        action="append",
        required=True,
        help="OCR-ablation observation for each legacy-control prediction",
    )
    parser.add_argument(
        "--candidate-observation",
        action="append",
        required=True,
        help="OCR-ablation observation for each candidate prediction",
    )
    parser.add_argument("--fusion-audit", required=True)
    parser.add_argument("--legacy-control-source-revision-sha", required=True)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        aggregate = build_aggregate_evidence(
            layout_manifest_path=args.layout_manifest,
            expected_layout_manifest_sha256=(
                args.expected_layout_manifest_sha256
            ),
            expected_input_tree_sha256=args.expected_input_tree_sha256,
            truth_path=args.truth,
            legacy_control_prediction_paths=(
                args.legacy_control_prediction
            ),
            candidate_prediction_paths=args.candidate_prediction,
            legacy_control_observation_paths=(
                args.legacy_control_observation
            ),
            candidate_observation_paths=args.candidate_observation,
            fusion_audit_path=args.fusion_audit,
            legacy_control_source_revision_sha=(
                args.legacy_control_source_revision_sha
            ),
            source_revision_sha=args.source_revision_sha,
        )
        _atomic_write(Path(args.output_json), canonical_json(aggregate) + "\n")
        _atomic_write(
            Path(args.output_markdown), render_aggregate_markdown(aggregate)
        )
    except (ExperimentControlError, OSError) as exc:
        print(f"grouped fusion evidence error: {exc}", file=sys.stderr)
        return 1
    print(
        f"WO-16 aggregate evidence: {aggregate['status']} "
        f"(delta {float(aggregate['score_delta']):+.9f})"
    )
    return 0 if aggregate["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
