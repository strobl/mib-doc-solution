#!/usr/bin/env python3
"""Build aggregate-only WO-15 grouped recovery evidence.

All identity-bearing inputs stay outside the repository evidence artifact:

* a layout-group manifest frozen before scoring,
* public truth rows,
* repeated control and candidate predictions, and
* an aggregate candidate recovery-audit JSON file.

The builder uses the official evaluator in :mod:`scripts.evaluate` for the
full comparison and every validation fold.  Only the strict aggregate view
from :class:`devtools.grouped_recovery_gate.GroupedRecoveryGate`, plus the
source revision digest, can be serialized by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.experiment_control import (  # noqa: E402
    ExperimentControlError,
    RepeatedGroupedSplitManager,
    canonical_json,
    require_aggregate_only,
)
from devtools.grouped_recovery_gate import (  # noqa: E402
    REQUIRED_FOLDS,
    REQUIRED_REPEATS,
    FoldScorePair,
    FullRunAggregate,
    GroupedRecoveryEvidence,
    GroupedRecoveryGate,
    RecoveryAuditAggregate,
)
from scripts import evaluate as official_evaluate  # noqa: E402


LAYOUT_MANIFEST_SCHEMA = "mib-wo15-layout-groups/v1"
_SOURCE_DIGEST_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class GroupedRecoveryEvidenceBuildError(ExperimentControlError):
    """An external WO-15 evidence input is incomplete or malformed."""


@dataclass(frozen=True)
class FrozenLayoutManifest:
    """Validated private layout groups and their raw frozen-file digest."""

    groups: Mapping[str, tuple[str, ...]]
    split_seed: str
    sha256: str
    frozen_before_scoring: bool

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(case_id for case_ids in self.groups.values() for case_id in case_ids)
        )


@dataclass(frozen=True)
class PredictionRuns:
    """Repeated predictions plus the aggregate metrics used by the gate."""

    rows: tuple[Mapping[str, Any], ...]
    full: FullRunAggregate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GroupedRecoveryEvidenceBuildError(
            f"{label} must be a readable JSON object"
        ) from exc
    if not isinstance(value, Mapping):
        raise GroupedRecoveryEvidenceBuildError(f"{label} must be a JSON object")
    return value


def load_layout_manifest(path: Path | str) -> FrozenLayoutManifest:
    """Load the private identity-bearing manifest and bind its exact bytes."""

    manifest_path = Path(path)
    raw = _read_json_object(manifest_path, label="layout manifest")
    if raw.get("schema") != LAYOUT_MANIFEST_SCHEMA:
        raise GroupedRecoveryEvidenceBuildError(
            f"layout manifest schema must be {LAYOUT_MANIFEST_SCHEMA!r}"
        )
    if raw.get("frozen_before_scoring") is not True:
        raise GroupedRecoveryEvidenceBuildError(
            "layout manifest must explicitly attest frozen_before_scoring=true"
        )
    if raw.get("repeats") != REQUIRED_REPEATS or raw.get("folds") != REQUIRED_FOLDS:
        raise GroupedRecoveryEvidenceBuildError(
            "layout manifest must freeze exactly three repeats of five folds"
        )
    split_seed = str(raw.get("split_seed", "")).strip()
    if not split_seed:
        raise GroupedRecoveryEvidenceBuildError(
            "layout manifest requires a non-empty split_seed"
        )

    rows = raw.get("cases")
    if not isinstance(rows, list) or not rows:
        raise GroupedRecoveryEvidenceBuildError(
            "layout manifest cases must be a non-empty list"
        )
    groups: dict[str, list[str]] = {}
    case_owner: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise GroupedRecoveryEvidenceBuildError(
                f"layout manifest case row {index} must be an object"
            )
        case_id = str(row.get("case_id", "")).strip()
        group_id = str(row.get("layout_group", "")).strip()
        if not case_id or not group_id:
            raise GroupedRecoveryEvidenceBuildError(
                f"layout manifest case row {index} requires case_id and layout_group"
            )
        previous = case_owner.setdefault(case_id, group_id)
        if previous != group_id or case_id in groups.setdefault(group_id, []):
            raise GroupedRecoveryEvidenceBuildError(
                "layout manifest case IDs must occur exactly once"
            )
        groups[group_id].append(case_id)
    if len(groups) < REQUIRED_FOLDS:
        raise GroupedRecoveryEvidenceBuildError(
            "layout manifest must contain at least five layout groups"
        )

    normalized = {
        group_id: tuple(sorted(case_ids))
        for group_id, case_ids in sorted(groups.items())
    }
    return FrozenLayoutManifest(
        groups=normalized,
        split_seed=split_seed,
        sha256=_sha256_file(manifest_path),
        frozen_before_scoring=True,
    )


def _read_truth_subset(
    path: Path | str, case_ids: Sequence[str]
) -> dict[str, Mapping[str, Any]]:
    try:
        rows = official_evaluate.read_csv_rows(Path(path))
    except OSError as exc:
        raise GroupedRecoveryEvidenceBuildError(
            "truth must be a readable CSV file"
        ) from exc
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        if case_id in indexed:
            raise GroupedRecoveryEvidenceBuildError(
                "truth contains duplicate case IDs"
            )
        indexed[case_id] = row
    missing = sorted(set(case_ids) - set(indexed))
    if missing:
        raise GroupedRecoveryEvidenceBuildError(
            f"truth is missing {len(missing)} frozen manifest cases"
        )
    return {case_id: indexed[case_id] for case_id in sorted(case_ids)}


def _read_submission(path: Path | str) -> tuple[Mapping[str, Any], ...]:
    try:
        rows = official_evaluate.read_submission(Path(path))
    except (OSError, SystemExit) as exc:
        raise GroupedRecoveryEvidenceBuildError(
            "prediction input must be a readable evaluator-compatible submission"
        ) from exc
    return tuple(rows)


def _read_distinct_runs(
    paths: Sequence[Path | str], *, arm: str
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    if len(paths) < 2:
        raise GroupedRecoveryEvidenceBuildError(
            "at least two repeated prediction files are required per arm"
        )
    resolved = tuple(Path(path).resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise GroupedRecoveryEvidenceBuildError(
            f"{arm} determinism evidence requires distinct prediction artifacts"
        )
    return tuple(_read_submission(path) for path in resolved)


def _rows_fingerprint(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash semantic rows internally; this digest is never emitted."""

    canonical_rows = sorted(canonical_json(dict(row)) for row in rows)
    return hashlib.sha256(
        canonical_json(canonical_rows).encode("utf-8")
    ).hexdigest()


def _invalid_record_count(aggregate: Mapping[str, Any]) -> int:
    counts = aggregate["counts"]
    return sum(
        int(counts.get(name, 0))
        for name in (
            "blank_case_rows",
            "invalid_adjudication_records",
            "invalid_confidence_records",
            "invalid_fee_status_records",
        )
    )


def _false_positive_denial_count(
    truth: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> int:
    predictions, _, _ = official_evaluate.index_submission(rows)
    return sum(
        str(predictions[case_id].get("adjudication", "")).strip().upper()
        == "DENIED"
        and str(truth_row.get("adjudication", "")).strip().upper() != "DENIED"
        for case_id, truth_row in truth.items()
        if case_id in predictions
    )


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
    full = FullRunAggregate(
        total_score=float(aggregate["scores"]["total_score"]),
        record_count=int(counts["scored_predictions"]),
        catastrophic_false_approvals=int(
            aggregate["raw"]["catastrophic_false_approvals"]
        ),
        missing_records=int(counts["missing_cases"]),
        invalid_records=_invalid_record_count(aggregate),
        false_positive_denial_recoveries=_false_positive_denial_count(
            truth, rows
        ),
        deterministic=deterministic,
        duplicate_records=int(counts["duplicate_case_ids"]),
        extra_records=int(counts["extra_cases"]),
    )
    return PredictionRuns(rows=rows, full=full)


def load_recovery_audit(
    path: Path | str,
    *,
    expected_source_revision_sha: str,
) -> RecoveryAuditAggregate:
    """Accept only revision-bound, broker-safe recovery-audit counters."""

    payload = _read_json_object(Path(path), label="recovery audit")
    require_aggregate_only(payload)
    expected_source_digest = str(expected_source_revision_sha).strip().lower()
    if not _SOURCE_DIGEST_RE.fullmatch(expected_source_digest):
        raise GroupedRecoveryEvidenceBuildError(
            "expected recovery audit source_revision_sha must be a full "
            "Git commit or SHA-256 digest"
        )
    audit_source_digest = str(payload.get("source_revision_sha", "")).strip().lower()
    if not _SOURCE_DIGEST_RE.fullmatch(audit_source_digest):
        raise GroupedRecoveryEvidenceBuildError(
            "recovery audit must include a full source_revision_sha"
        )
    if audit_source_digest != expected_source_digest:
        raise GroupedRecoveryEvidenceBuildError(
            "recovery audit source_revision_sha does not match the "
            "evidence source revision"
        )
    counts: Mapping[str, Any]
    nested = payload.get("counts")
    if nested is None:
        counts = payload
    elif isinstance(nested, Mapping):
        counts = nested
    else:
        raise GroupedRecoveryEvidenceBuildError(
            "recovery audit counts must be an object"
        )
    required = (
        "recovered_field_count",
        "recovered_field_complete_provenance_count",
        "serialization_default_used_as_evidence_count",
    )
    if any(name not in counts for name in required):
        raise GroupedRecoveryEvidenceBuildError(
            "recovery audit is missing required aggregate counters"
        )
    return RecoveryAuditAggregate(
        recovered_field_count=counts["recovered_field_count"],
        recovered_field_complete_provenance_count=counts[
            "recovered_field_complete_provenance_count"
        ],
        serialization_default_used_as_evidence_count=counts[
            "serialization_default_used_as_evidence_count"
        ],
    )


def _filter_rows(
    rows: Sequence[Mapping[str, Any]], case_ids: Sequence[str]
) -> tuple[Mapping[str, Any], ...]:
    allowed = set(case_ids)
    return tuple(
        row
        for row in rows
        if str(row.get("case_id", "")).strip() in allowed
    )


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
    return (
        tuple(pairs),
        group_exclusive,
        paired_fold_members,
        split_deterministic,
    )


def build_aggregate_evidence(
    *,
    layout_manifest_path: Path | str,
    expected_layout_manifest_sha256: str,
    truth_path: Path | str,
    control_prediction_paths: Sequence[Path | str],
    candidate_prediction_paths: Sequence[Path | str],
    recovery_audit_path: Path | str,
    source_revision_sha: str,
) -> dict[str, Any]:
    """Build the only public, repository-safe form of the WO-15 evidence."""

    source_digest = str(source_revision_sha).strip().lower()
    if not _SOURCE_DIGEST_RE.fullmatch(source_digest):
        raise GroupedRecoveryEvidenceBuildError(
            "source_revision_sha must be a full Git commit or SHA-256 digest"
        )
    manifest = load_layout_manifest(layout_manifest_path)
    expected_manifest_digest = str(expected_layout_manifest_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_digest):
        raise GroupedRecoveryEvidenceBuildError(
            "expected_layout_manifest_sha256 must be a full SHA-256 digest"
        )
    if manifest.sha256 != expected_manifest_digest:
        raise GroupedRecoveryEvidenceBuildError(
            "layout manifest bytes do not match the pre-recorded frozen digest"
        )
    audit = load_recovery_audit(
        recovery_audit_path,
        expected_source_revision_sha=source_digest,
    )
    truth = _read_truth_subset(truth_path, manifest.case_ids)
    control = _full_run(
        truth, _read_distinct_runs(control_prediction_paths, arm="control")
    )
    candidate = _full_run(
        truth, _read_distinct_runs(candidate_prediction_paths, arm="candidate")
    )
    folds, group_exclusive, paired, split_deterministic = _fold_pairs(
        manifest=manifest,
        truth=truth,
        control_rows=control.rows,
        candidate_rows=candidate.rows,
    )
    evidence = GroupedRecoveryEvidence(
        layout_manifest_sha256=manifest.sha256,
        expected_record_count=len(manifest.case_ids),
        expected_layout_group_count=len(manifest.groups),
        control_full=control.full,
        candidate_full=candidate.full,
        candidate_recovery_audit=audit,
        folds=folds,
        manifest_frozen_before_scoring=manifest.frozen_before_scoring,
        group_exclusive=group_exclusive,
        paired_fold_members=paired,
        split_deterministic=split_deterministic,
    )
    aggregate = GroupedRecoveryGate().evaluate(evidence).to_aggregate_evidence()
    aggregate["source_revision_sha"] = source_digest
    require_aggregate_only(aggregate)
    return aggregate


def render_aggregate_markdown(aggregate: Mapping[str, Any]) -> str:
    """Render the already-sanitized aggregate object without input identities."""

    require_aggregate_only(aggregate)
    gates = aggregate["gate_results"]
    checks = aggregate["checks"]
    repeat_metrics = aggregate["metrics"]
    acceptance = "PASS" if aggregate["status"] == "passed" else "FAIL"
    lines = [
        "# WO-15 grouped visible-evidence recovery",
        "",
        f"- Evidence class: `{aggregate['evaluation_mode']}`",
        f"- Status: **{str(aggregate['status']).upper()}**",
        f"- Work Order acceptance: **{acceptance}**",
        f"- Source revision: `{aggregate['source_revision_sha']}`",
        f"- Frozen layout manifest: `{aggregate['layout_manifest_sha256']}`",
        f"- Records / layout groups: {aggregate['expected_record_count']} / "
        f"{aggregate['layout_group_count']}",
        "",
        "## Paired score result",
        "",
        f"- Control: {float(aggregate['full_control_score']):.9f}",
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
            f"{float(repeat_metrics[f'repeat_{repeat}_weighted_score_delta']):+.9f} | "
            f"{int(repeat_metrics[f'repeat_{repeat}_positive_fold_count'])}/"
            f"{REQUIRED_FOLDS} | "
            f"{float(repeat_metrics[f'repeat_{repeat}_leave_best_fold_out_delta']):+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Safety and provenance",
            "",
            f"- Catastrophic false approvals: "
            f"{aggregate['catastrophic_false_approvals']}",
            f"- False-positive denial recovery delta: "
            f"{aggregate['false_positive_denial_recoveries_delta']:+d}",
            f"- Missing / invalid / duplicate / extra records: "
            f"{aggregate['missing_records']} / {aggregate['invalid_records']} / "
            f"{aggregate['duplicate_records']} / {aggregate['extra_records']}",
            f"- Recovered fields with complete provenance: "
            f"{aggregate['recovered_field_complete_provenance_count']} / "
            f"{aggregate['recovered_field_count']}",
            f"- Serialization defaults used as evidence: "
            f"{aggregate['serialization_default_used_as_evidence_count']}",
            "",
            "## Concentration diagnostics (non-hard)",
            "",
            "| Diagnostic | Result |",
            "| --- | :---: |",
            f"| `fold_majority_positive` | "
            f"{'PASS' if checks['fold_majority_positive'] else 'WARN'} |",
            f"| `leave_best_fold_out_positive` | "
            f"{'PASS' if checks['leave_best_fold_out_positive'] else 'WARN'} |",
            "",
        ]
    )
    if checks["score_gain_concentration_warning"]:
        lines.extend(
            [
                "> **Concentration warning:** the positive score gain is "
                "concentrated in a minority of folds or becomes zero when the "
                "strongest fold is omitted. This is disclosed as a non-hard "
                "robustness warning; it does not change the Work Order "
                "acceptance result.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "- No score-gain concentration warning.",
                "",
            ]
        )
    lines.extend(
        [
            "## Hard gates",
            "",
            "| Gate | Result |",
            "| --- | :---: |",
        ]
    )
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
        description="Build aggregate-only WO-15 repeated grouped evidence."
    )
    parser.add_argument("--layout-manifest", required=True)
    parser.add_argument(
        "--expected-layout-manifest-sha256",
        required=True,
        help="digest recorded when the identity-bearing manifest was frozen",
    )
    parser.add_argument("--truth", required=True)
    parser.add_argument(
        "--control-prediction",
        action="append",
        required=True,
        help="repeat once per deterministic control run (minimum two)",
    )
    parser.add_argument(
        "--candidate-prediction",
        action="append",
        required=True,
        help="repeat once per deterministic candidate run (minimum two)",
    )
    parser.add_argument("--recovery-audit", required=True)
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
            truth_path=args.truth,
            control_prediction_paths=args.control_prediction,
            candidate_prediction_paths=args.candidate_prediction,
            recovery_audit_path=args.recovery_audit,
            source_revision_sha=args.source_revision_sha,
        )
        _atomic_write(
            Path(args.output_json), canonical_json(aggregate) + "\n"
        )
        _atomic_write(
            Path(args.output_markdown), render_aggregate_markdown(aggregate)
        )
    except (GroupedRecoveryEvidenceBuildError, ExperimentControlError) as exc:
        print(f"grouped recovery evidence error: {exc}", file=sys.stderr)
        return 1
    print(
        f"WO-15 aggregate evidence: {aggregate['status']} "
        f"(delta {float(aggregate['score_delta']):+.9f})"
    )
    return 0 if aggregate["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
