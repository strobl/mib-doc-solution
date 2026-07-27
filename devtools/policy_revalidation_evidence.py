#!/usr/bin/env python3
"""Build aggregate-only, revision-bound WO-17 acceptance evidence.

Identity-bearing truth, layout membership, and predictions remain external.
The committed artifact contains official-evaluator aggregates, paired
confusion deltas, immutable artifact hashes, and two kinds of execution audit:

* production-cohort occurrence counters (which may legitimately be zero), and
* non-vacuous contract probes that exercise every required before/after path.

Both prediction arms and both candidate audit artifacts must be byte-bound to
two deterministic runs over the same frozen 32-case input tree.
"""

from __future__ import annotations

import argparse
import hashlib
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

from devtools.experiment_control import (  # noqa: E402
    ExperimentControlError,
    canonical_json,
    require_aggregate_only,
)
from devtools.grouped_fusion_evidence import (  # noqa: E402
    _case_id_set_sha256,
    _load_arm_observations,
    _observation_set_sha256,
    _safety_event_case_ids,
    verify_clean_candidate_checkout,
    verify_legacy_commit,
)
from devtools.grouped_recovery_evidence import (  # noqa: E402
    _invalid_record_count,
    _read_distinct_runs,
    _read_json_object,
    _read_truth_subset,
    _rows_fingerprint,
    load_layout_manifest,
)
from devtools.policy_revalidation_audit_contract import (  # noqa: E402
    CONTRACT_AUDIT_COUNTS,
    COHORT_AUDIT_COUNTS,
    POLICY_REVALIDATION_AUDIT_SCHEMA,
)
from devtools.policy_revalidation_gate import (  # noqa: E402
    PolicyExecutionAudit,
    PolicyRevalidationEvidence,
    PolicyRevalidationGate,
    PolicyRunAggregate,
)
from scripts import evaluate as official_evaluate  # noqa: E402


REQUIRED_COHORT_RECORDS = 32
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_AUDIT_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "source_revision_sha",
        "input_tree_sha256",
        "input_pdf_count",
        "case_id_set_sha256",
        "repeat_index",
        "predictions_sha256",
        "contract_fixture_sha256",
        "cohort_counts",
        "contract_counts",
    }
)


class PolicyRevalidationEvidenceBuildError(ExperimentControlError):
    """An external WO-17 evidence input is incomplete or malformed."""


@dataclass(frozen=True)
class PredictionArm:
    rows: tuple[Mapping[str, Any], ...]
    aggregate: PolicyRunAggregate
    missing_case_ids: frozenset[str]
    invalid_case_ids: frozenset[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _full_run(
    truth: Mapping[str, Mapping[str, Any]],
    runs: Sequence[tuple[Mapping[str, Any], ...]],
) -> PredictionArm:
    aggregates = [
        official_evaluate.build_results(truth, rows)[0] for rows in runs
    ]
    deterministic = (
        len({_rows_fingerprint(rows) for rows in runs}) == 1
        and len({canonical_json(value) for value in aggregates}) == 1
    )
    rows = tuple(runs[0])
    aggregate = aggregates[0]
    predictions, _, _ = official_evaluate.index_submission(rows)
    missing_case_ids = frozenset(set(truth) - set(predictions))
    invalid_case_ids = frozenset(
        case_id
        for case_id, truth_row in truth.items()
        if case_id in predictions
        and (
            not (
                scored := official_evaluate.score_case(
                    case_id, truth_row, predictions[case_id]
                )
            )["adjudication_valid"]
            or not scored["confidence_valid"]
            or not scored["fee_status_valid"]
        )
    )
    _, false_positive_denials = _safety_event_case_ids(truth, rows)
    counts = aggregate["counts"]
    scores = aggregate["scores"]
    confusion = aggregate["confusion"]
    return PredictionArm(
        rows=rows,
        aggregate=PolicyRunAggregate(
            total_score=float(scores["total_score"]),
            extraction_score=float(scores["extraction_score"]),
            classification_score=float(scores["classification_score"]),
            calibration_score=float(scores["calibration_score"]),
            record_count=int(counts["scored_predictions"]),
            catastrophic_false_approvals=int(
                aggregate["raw"]["catastrophic_false_approvals"]
            ),
            false_positive_denials=len(false_positive_denials),
            missing_records=int(counts["missing_cases"]),
            invalid_records=_invalid_record_count(aggregate),
            duplicate_records=int(counts["duplicate_case_ids"]),
            extra_records=int(counts["extra_cases"]),
            approved_to_needs_review_count=int(
                confusion.get("APPROVED->NEEDS_REVIEW", 0)
            ),
            denied_to_needs_review_count=int(
                confusion.get("DENIED->NEEDS_REVIEW", 0)
            ),
            deterministic=deterministic,
        ),
        missing_case_ids=missing_case_ids,
        invalid_case_ids=invalid_case_ids,
    )


def _non_policy_field_change_count(
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
) -> int:
    """Count paired extraction-row changes without exposing their identities."""

    control, _, _ = official_evaluate.index_submission(control_rows)
    candidate, _, _ = official_evaluate.index_submission(candidate_rows)
    return sum(
        any(
            control.get(case_id, {}).get(field_name)
            != candidate.get(case_id, {}).get(field_name)
            for field_name in official_evaluate.FIELDS
        )
        for case_id in case_ids
    )


def _validated_count_mapping(
    value: Any,
    *,
    label: str,
    required: tuple[str, ...],
) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise PolicyRevalidationEvidenceBuildError(
            f"{label} must contain the exact required counters"
        )
    validated: dict[str, int] = {}
    for name in required:
        count = value[name]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise PolicyRevalidationEvidenceBuildError(
                f"{label}.{name} must be a non-negative integer"
            )
        validated[name] = count
    return validated


def _load_policy_audits(
    *,
    paths: Sequence[Path | str],
    candidate_observations: Sequence[Mapping[str, Any]],
    expected_source_revision_sha: str,
    expected_input_tree_sha256: str,
    expected_input_pdf_count: int,
    expected_case_id_set_sha256: str,
) -> tuple[PolicyExecutionAudit, str, str]:
    """Load two path-free candidate audits and verify every byte binding."""

    if len(paths) != 2:
        raise PolicyRevalidationEvidenceBuildError(
            "candidate requires exactly two policy audit artifacts"
        )
    resolved_paths = tuple(Path(path).resolve() for path in paths)
    if len(set(resolved_paths)) != 2:
        raise PolicyRevalidationEvidenceBuildError(
            "policy audit artifacts must be distinct files"
        )
    by_repeat: dict[int, Mapping[str, Any]] = {}
    raw_bytes: list[bytes] = []
    expected_prediction_sha_by_repeat = {
        int(observation["repeat_index"]): str(
            observation["predictions_sha256"]
        ).strip().casefold()
        for observation in candidate_observations
    }
    for path in resolved_paths:
        payload = _read_json_object(path, label="policy audit")
        if set(payload) != _AUDIT_TOP_LEVEL_KEYS:
            raise PolicyRevalidationEvidenceBuildError(
                "policy audit must contain only the exact contract keys"
            )
        if (
            payload.get("schema_version")
            != POLICY_REVALIDATION_AUDIT_SCHEMA
        ):
            raise PolicyRevalidationEvidenceBuildError(
                "policy audit schema_version is unsupported"
            )
        repeat_index = payload.get("repeat_index")
        if (
            isinstance(repeat_index, bool)
            or not isinstance(repeat_index, int)
            or repeat_index not in {1, 2}
            or repeat_index in by_repeat
        ):
            raise PolicyRevalidationEvidenceBuildError(
                "policy audits require unique repeat indexes 1 and 2"
            )
        source_revision = str(
            payload.get("source_revision_sha", "")
        ).strip().casefold()
        input_tree = str(
            payload.get("input_tree_sha256", "")
        ).strip().casefold()
        case_set = str(
            payload.get("case_id_set_sha256", "")
        ).strip().casefold()
        prediction_sha = str(
            payload.get("predictions_sha256", "")
        ).strip().casefold()
        fixture_sha = str(
            payload.get("contract_fixture_sha256", "")
        ).strip().casefold()
        if source_revision != expected_source_revision_sha:
            raise PolicyRevalidationEvidenceBuildError(
                "policy audit source revision does not match the candidate"
            )
        if input_tree != expected_input_tree_sha256:
            raise PolicyRevalidationEvidenceBuildError(
                "policy audit input tree does not match the frozen cohort"
            )
        if payload.get("input_pdf_count") != expected_input_pdf_count:
            raise PolicyRevalidationEvidenceBuildError(
                "policy audit input count does not match the frozen cohort"
            )
        if case_set != expected_case_id_set_sha256:
            raise PolicyRevalidationEvidenceBuildError(
                "policy audit case-set digest does not match the manifest"
            )
        if not _SHA256_RE.fullmatch(prediction_sha) or prediction_sha != (
            expected_prediction_sha_by_repeat.get(repeat_index)
        ):
            raise PolicyRevalidationEvidenceBuildError(
                "policy audit is not bound to its candidate prediction bytes"
            )
        if not _SHA256_RE.fullmatch(fixture_sha):
            raise PolicyRevalidationEvidenceBuildError(
                "policy audit requires a full contract_fixture_sha256"
            )
        cohort = _validated_count_mapping(
            payload.get("cohort_counts"),
            label="cohort_counts",
            required=COHORT_AUDIT_COUNTS,
        )
        contract = _validated_count_mapping(
            payload.get("contract_counts"),
            label="contract_counts",
            required=CONTRACT_AUDIT_COUNTS,
        )
        by_repeat[repeat_index] = {
            "cohort_counts": cohort,
            "contract_counts": contract,
            "contract_fixture_sha256": fixture_sha,
        }
        raw_bytes.append(path.read_bytes())

    if set(by_repeat) != {1, 2}:
        raise PolicyRevalidationEvidenceBuildError(
            "policy audits must cover repeats 1 and 2"
        )
    ordered = tuple(by_repeat[index] for index in (1, 2))
    fixture_hashes = {
        str(value["contract_fixture_sha256"]) for value in ordered
    }
    if len(fixture_hashes) != 1:
        raise PolicyRevalidationEvidenceBuildError(
            "policy audits disagree on the contract fixture binding"
        )
    deterministic = (
        canonical_json(ordered[0]["cohort_counts"])
        == canonical_json(ordered[1]["cohort_counts"])
        and canonical_json(ordered[0]["contract_counts"])
        == canonical_json(ordered[1]["contract_counts"])
    )
    cohort_counts = {
        name: max(
            int(value["cohort_counts"][name])  # type: ignore[index]
            for value in ordered
        )
        for name in COHORT_AUDIT_COUNTS
    }
    contract_counts = {
        name: max(
            int(value["contract_counts"][name])  # type: ignore[index]
            for value in ordered
        )
        for name in CONTRACT_AUDIT_COUNTS
    }
    audit_set_sha256 = hashlib.sha256(
        canonical_json(
            sorted(hashlib.sha256(value).hexdigest() for value in raw_bytes)
        ).encode("utf-8")
    ).hexdigest()
    return (
        PolicyExecutionAudit(
            cohort_counts=cohort_counts,
            contract_counts=contract_counts,
            deterministic=deterministic,
        ),
        next(iter(fixture_hashes)),
        audit_set_sha256,
    )


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
    candidate_policy_audit_paths: Sequence[Path | str],
    legacy_control_source_revision_sha: str,
    source_revision_sha: str,
    checkout_verifier: Callable[[Path, str], None] | None = None,
    legacy_revision_verifier: Callable[[Path, str], None] | None = None,
) -> dict[str, Any]:
    """Build the only repository-safe WO-17 acceptance artifact."""

    legacy_revision = str(
        legacy_control_source_revision_sha
    ).strip().casefold()
    source_revision = str(source_revision_sha).strip().casefold()
    expected_manifest_sha = str(
        expected_layout_manifest_sha256
    ).strip().casefold()
    expected_input_tree = str(
        expected_input_tree_sha256
    ).strip().casefold()
    if not _GIT_COMMIT_RE.fullmatch(legacy_revision):
        raise PolicyRevalidationEvidenceBuildError(
            "legacy control revision must be a full Git commit SHA"
        )
    if not _GIT_COMMIT_RE.fullmatch(source_revision):
        raise PolicyRevalidationEvidenceBuildError(
            "candidate revision must be a full Git commit SHA"
        )
    if not _SHA256_RE.fullmatch(expected_manifest_sha):
        raise PolicyRevalidationEvidenceBuildError(
            "expected layout manifest digest must be a full SHA-256"
        )
    if not _SHA256_RE.fullmatch(expected_input_tree):
        raise PolicyRevalidationEvidenceBuildError(
            "expected input tree digest must be a full SHA-256"
        )
    (checkout_verifier or verify_clean_candidate_checkout)(
        REPO_ROOT, source_revision
    )
    (legacy_revision_verifier or verify_legacy_commit)(
        REPO_ROOT, legacy_revision
    )

    manifest_path = Path(layout_manifest_path)
    manifest = load_layout_manifest(manifest_path)
    if _sha256_file(manifest_path) != expected_manifest_sha:
        raise PolicyRevalidationEvidenceBuildError(
            "layout manifest bytes do not match the pre-recorded digest"
        )
    if len(manifest.case_ids) != REQUIRED_COHORT_RECORDS:
        raise PolicyRevalidationEvidenceBuildError(
            "WO-17 must use the frozen 32-case cohort"
        )

    legacy_observations = _load_arm_observations(
        arm="legacy control",
        observation_paths=legacy_control_observation_paths,
        prediction_paths=legacy_control_prediction_paths,
        expected_source_revision_sha=legacy_revision,
    )
    candidate_observations = _load_arm_observations(
        arm="candidate",
        observation_paths=candidate_observation_paths,
        prediction_paths=candidate_prediction_paths,
        expected_source_revision_sha=source_revision,
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
            raise PolicyRevalidationEvidenceBuildError(
                f"control and candidate observations disagree on {key}"
            )
    input_pdf_count = candidate_observations[0]["input_pdf_count"]
    input_tree_sha256 = str(
        candidate_observations[0]["input_tree_sha256"]
    ).strip().casefold()
    if (
        input_pdf_count != REQUIRED_COHORT_RECORDS
        or input_tree_sha256 != expected_input_tree
    ):
        raise PolicyRevalidationEvidenceBuildError(
            "observations do not match the frozen 32-case input tree"
        )

    case_id_set_sha256 = _case_id_set_sha256(manifest.case_ids)
    audit, contract_fixture_sha256, audit_set_sha256 = _load_policy_audits(
        paths=candidate_policy_audit_paths,
        candidate_observations=candidate_observations,
        expected_source_revision_sha=source_revision,
        expected_input_tree_sha256=input_tree_sha256,
        expected_input_pdf_count=REQUIRED_COHORT_RECORDS,
        expected_case_id_set_sha256=case_id_set_sha256,
    )
    truth = _read_truth_subset(truth_path, manifest.case_ids)
    control = _full_run(
        truth,
        _read_distinct_runs(
            legacy_control_prediction_paths, arm="legacy control"
        ),
    )
    candidate = _full_run(
        truth,
        _read_distinct_runs(
            candidate_prediction_paths, arm="candidate"
        ),
    )
    legacy_catastrophic, legacy_false_denials = _safety_event_case_ids(
        truth, control.rows
    )
    candidate_catastrophic, candidate_false_denials = (
        _safety_event_case_ids(truth, candidate.rows)
    )
    evidence = PolicyRevalidationEvidence(
        layout_manifest_sha256=manifest.sha256,
        expected_record_count=REQUIRED_COHORT_RECORDS,
        layout_group_count=len(manifest.groups),
        legacy_control=control.aggregate,
        candidate=candidate.aggregate,
        audit=audit,
        new_catastrophic_false_approval_count=len(
            candidate_catastrophic - legacy_catastrophic
        ),
        new_false_positive_denial_count=len(
            candidate_false_denials - legacy_false_denials
        ),
        new_missing_record_count=len(
            candidate.missing_case_ids - control.missing_case_ids
        ),
        new_invalid_record_count=len(
            candidate.invalid_case_ids - control.invalid_case_ids
        ),
        non_policy_field_change_count=_non_policy_field_change_count(
            control.rows, candidate.rows, manifest.case_ids
        ),
        manifest_frozen_before_scoring=manifest.frozen_before_scoring,
    )
    aggregate = PolicyRevalidationGate().evaluate(
        evidence
    ).to_aggregate_evidence()
    aggregate.update(
        {
            "legacy_control_source_revision_sha": legacy_revision,
            "source_revision_sha": source_revision,
            "input_pdf_count": input_pdf_count,
            "input_tree_sha256": input_tree_sha256,
            "case_id_set_sha256": case_id_set_sha256,
            "legacy_control_observation_set_sha256": (
                _observation_set_sha256(legacy_observations)
            ),
            "candidate_observation_set_sha256": (
                _observation_set_sha256(candidate_observations)
            ),
            "policy_execution_audit_set_sha256": audit_set_sha256,
            "contract_fixture_sha256": contract_fixture_sha256,
        }
    )
    require_aggregate_only(aggregate)
    return aggregate


def render_aggregate_markdown(aggregate: Mapping[str, Any]) -> str:
    """Render the sanitized before/after and acceptance evidence."""

    require_aggregate_only(aggregate)
    confusion = aggregate["confusion_counts"]
    counts = aggregate["counts"]
    gates = aggregate["gate_results"]
    lines = [
        "# WO-17 late-recovery policy revalidation",
        "",
        f"- Evidence class: `{aggregate['evaluation_mode']}`",
        f"- Status: **{str(aggregate['status']).upper()}**",
        f"- Source revision: `{aggregate['source_revision_sha']}`",
        f"- Legacy-control revision: "
        f"`{aggregate['legacy_control_source_revision_sha']}`",
        f"- Frozen layout manifest: "
        f"`{aggregate['layout_manifest_sha256']}`",
        f"- Input PDFs / tree: {aggregate['input_pdf_count']} / "
        f"`{aggregate['input_tree_sha256']}`",
        f"- Identity-free cohort binding: "
        f"`{aggregate['case_id_set_sha256']}`",
        f"- Candidate observations: "
        f"`{aggregate['candidate_observation_set_sha256']}`",
        f"- Legacy observations: "
        f"`{aggregate['legacy_control_observation_set_sha256']}`",
        f"- Execution audits: "
        f"`{aggregate['policy_execution_audit_set_sha256']}`",
        f"- Contract fixtures: `{aggregate['contract_fixture_sha256']}`",
        "",
        "## Official evaluator",
        "",
        f"- Legacy control: {float(aggregate['full_control_score']):.9f}",
        f"- Candidate: {float(aggregate['full_candidate_score']):.9f}",
        f"- Delta: {float(aggregate['score_delta']):+.9f}",
        f"- Extraction / classification / calibration deltas: "
        f"{float(aggregate['extraction_score_delta']):+.9f} / "
        f"{float(aggregate['classification_score_delta']):+.9f} / "
        f"{float(aggregate['calibration_score_delta']):+.9f}",
        "",
        "## Targeted confusion deltas",
        "",
        "| Truth → prediction | Control | Candidate | Delta |",
        "| --- | ---: | ---: | ---: |",
        "| APPROVED → NEEDS_REVIEW | "
        f"{confusion['approved_to_needs_review_control_count']} | "
        f"{confusion['approved_to_needs_review_candidate_count']} | "
        f"{confusion['approved_to_needs_review_delta']:+d} |",
        "| DENIED → NEEDS_REVIEW | "
        f"{confusion['denied_to_needs_review_control_count']} | "
        f"{confusion['denied_to_needs_review_candidate_count']} | "
        f"{confusion['denied_to_needs_review_delta']:+d} |",
        "",
        "## Execution-order trace",
        "",
        "| Aggregate trace | Frozen cohort | Contract probes |",
        "| --- | ---: | ---: |",
        "| Legacy synthetic decision before late recovery | — | "
        f"{counts['contract_legacy_synthetic_before_late_recovery_count']} |",
        "| Late recovery before revalidation | "
        f"{counts['cohort_late_recovery_before_revalidation_count']} | "
        f"{counts['contract_candidate_late_recovery_before_revalidation_count']} |",
        "| Revalidation after late recovery | "
        f"{counts['cohort_revalidation_after_late_recovery_count']} | "
        f"{counts['contract_candidate_revalidation_after_late_recovery_count']} |",
        "| Contradicted synthetic reasons removed | "
        f"{counts['cohort_contradicted_synthetic_reason_removed_count']} | "
        f"{counts['contract_contradicted_synthetic_reason_removed_count']} |",
        "| Independent denial reasons retained | "
        f"{counts['cohort_independent_denial_reason_retained_count']} | "
        f"{counts['contract_independent_denial_reason_retained_count']} |",
        "| Original review confidence restored | "
        f"{counts['cohort_review_confidence_restored_count']} | "
        f"{counts['contract_review_confidence_restored_count']} |",
        "| Signed late authority recovered | "
        f"{counts['cohort_signed_late_authority_recovery_count']} | "
        f"{counts['contract_signed_late_authority_recovery_count']} |",
        "| Late adjudication evidence preserved | "
        f"{counts['cohort_late_adjudication_evidence_preserved_count']} | "
        f"{counts['contract_late_adjudication_evidence_preserved_count']} |",
        "| Late biohazard evidence preserved | "
        f"{counts['cohort_late_biohazard_evidence_preserved_count']} | "
        f"{counts['contract_late_biohazard_evidence_preserved_count']} |",
        "",
        "> Cohort occurrence counters may be zero. The contract-probe column "
        "must be non-vacuous for every required branch.",
        "",
        "## Safety",
        "",
        f"- Newly introduced catastrophic false approvals: "
        f"{aggregate['new_catastrophic_false_approval_count']}",
        f"- Newly introduced false-positive denials: "
        f"{aggregate['new_false_positive_denial_count']}",
        f"- New missing / invalid records: "
        f"{aggregate['new_missing_record_count']} / "
        f"{aggregate['new_invalid_record_count']}",
        f"- Candidate missing / invalid / duplicate / extra records: "
        f"{aggregate['missing_records']} / {aggregate['invalid_records']} / "
        f"{aggregate['duplicate_records']} / {aggregate['extra_records']}",
        f"- Non-policy field changes: "
        f"{aggregate['non_policy_field_change_count']}",
        "",
        "## Hard gates",
        "",
        "| Gate | Result |",
        "| --- | :---: |",
    ]
    lines.extend(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in sorted(gates.items())
    )
    lines.extend(
        [
            "",
            "> Public-label-exposed 32-case robustness evidence; this is not "
            "an unseen holdout result. Identity-bearing inputs and traces "
            "remain external.",
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
        description="Build aggregate-only WO-17 policy evidence."
    )
    parser.add_argument("--layout-manifest", required=True)
    parser.add_argument("--expected-layout-manifest-sha256", required=True)
    parser.add_argument("--expected-input-tree-sha256", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument(
        "--legacy-control-prediction", action="append", required=True
    )
    parser.add_argument(
        "--candidate-prediction", action="append", required=True
    )
    parser.add_argument(
        "--legacy-control-observation", action="append", required=True
    )
    parser.add_argument(
        "--candidate-observation", action="append", required=True
    )
    parser.add_argument(
        "--candidate-policy-audit", action="append", required=True
    )
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
            candidate_policy_audit_paths=args.candidate_policy_audit,
            legacy_control_source_revision_sha=(
                args.legacy_control_source_revision_sha
            ),
            source_revision_sha=args.source_revision_sha,
        )
        _atomic_write(
            Path(args.output_json), canonical_json(aggregate) + "\n"
        )
        _atomic_write(
            Path(args.output_markdown),
            render_aggregate_markdown(aggregate),
        )
    except (ExperimentControlError, OSError) as exc:
        print(f"policy revalidation evidence error: {exc}", file=sys.stderr)
        return 1
    print(
        f"WO-17 aggregate evidence: {aggregate['status']} "
        f"(delta {float(aggregate['score_delta']):+.9f})"
    )
    return 0 if aggregate["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
