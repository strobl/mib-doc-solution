#!/usr/bin/env python3
"""Capture WO-18 identity-free features from the accepted production graph.

This executable deliberately accepts no truth labels, role assignments, case
features, or prediction overrides.  It runs the exact production composition
root, records the accepted resolver/adjudicator state, derives the frozen
numeric feature vector, and requires a second byte-identical capture before
writing the observation manifest consumed by the WO-18 evidence process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.decision_recovery_cv import (  # noqa: E402
    FEATURE_ROWS_SCHEMA,
    feature_schema_sha256,
)
from devtools.experiment_control import canonical_json  # noqa: E402
from devtools.grouped_recovery_evidence import (  # noqa: E402
    FrozenLayoutManifest,
    load_layout_manifest,
)
from devtools.ocr_ablation import _input_tree_sha256  # noqa: E402
from mib_pipeline.adjudication import (  # noqa: E402
    AdjudicationOutcome,
    DecisionTrace,
)
from mib_pipeline.decision_recovery import (  # noqa: E402
    RevalidatedAdjudication,
    StagedAdjudication,
)
from mib_pipeline.model_recovery import (  # noqa: E402
    IdentityFreeDecisionFeatures,
    IdentityFreeFeatureBuilder,
)
from mib_pipeline.models import PredictionRow  # noqa: E402
from mib_pipeline.production import build_production_processor  # noqa: E402
from mib_pipeline.rapid_recovery import (  # noqa: E402
    RAPID_RECOVERY_ROUTE_ID,
)
from mib_pipeline.recovery_audit import (  # noqa: E402
    RecoveryAuditOverlay,
    SerializationOrigin,
)
from mib_pipeline.resolution import (  # noqa: E402
    ResolvedCase,
    ResolvedField,
)


CAPTURE_OBSERVATION_SCHEMA = "mib-wo18-production-capture/v1"
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_OUTPUT_EVIDENCE_FIELDS = (
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
)
class ProductionCaptureError(RuntimeError):
    """The production graph could not yield one uniquely accepted feature row."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def producer_graph_sha256(repo_root: Path = REPO_ROOT) -> str:
    """Hash the full package closure, artifacts, and offline envelope files."""

    entries: list[dict[str, object]] = []
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
    graph_paths.update(
        path
        for path in (
            repo_root / "requirements.lock",
            repo_root / "Dockerfile",
            repo_root / "run.sh",
        )
        if path.is_file()
    )
    if not graph_paths:
        raise ProductionCaptureError("production graph contains no source files")
    for path in sorted(graph_paths):
        relative = path.relative_to(repo_root).as_posix()
        entries.append(
            {
                "path": relative,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return _sha256_bytes(_canonical_bytes(entries))


class _RecordingResolver:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.records: list[ResolvedCase] = []

    @property
    def fusion_enabled(self) -> bool:
        return bool(getattr(self.delegate, "fusion_enabled", False))

    def resolve(self, linked: Any) -> ResolvedCase:
        resolved = self.delegate.resolve(linked)
        if not isinstance(resolved, ResolvedCase):
            raise ProductionCaptureError(
                "production resolver returned a non-ResolvedCase value"
            )
        self.records.append(resolved)
        return resolved


class _RecordingAdjudicator:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.records: list[tuple[ResolvedCase, AdjudicationOutcome]] = []

    def adjudicate_case(
        self,
        resolved_case: ResolvedCase,
    ) -> AdjudicationOutcome:
        outcome = self.delegate.adjudicate_case(resolved_case)
        self.records.append((resolved_case, outcome))
        return outcome

    def adjudicate_staged(
        self,
        resolved_case: ResolvedCase,
    ) -> StagedAdjudication:
        staged = self.delegate.adjudicate_staged(resolved_case)
        if not isinstance(staged, StagedAdjudication):
            raise ProductionCaptureError(
                "production adjudicator returned an invalid staged result"
            )
        self.records.append((resolved_case, staged.outcome))
        return staged

    def revalidate_after_recovery(
        self,
        resolved_case: ResolvedCase,
        *,
        original: StagedAdjudication,
    ) -> RevalidatedAdjudication:
        result = self.delegate.revalidate_after_recovery(
            resolved_case,
            original=original,
        )
        if not isinstance(result, RevalidatedAdjudication):
            raise ProductionCaptureError(
                "production adjudicator returned an invalid revalidation"
            )
        self.records.append((resolved_case, result.outcome))
        return result


@dataclass(frozen=True)
class CapturedProductionCase:
    row: PredictionRow
    resolved_case: ResolvedCase
    outcome: AdjudicationOutcome
    recovery_route: str
    features: IdentityFreeDecisionFeatures


def _overlay_audit_fields(
    resolved_case: ResolvedCase,
    audit: RecoveryAuditOverlay,
) -> ResolvedCase:
    """Rebuild scored fields from the accepted immutable recovery overlay."""

    fields = dict(resolved_case.fields)
    for field_name, field_audit in audit.fields.items():
        original = fields.get(field_name)
        considered = (
            original.considered
            if original is not None
            else ()
        )
        considered = (
            *considered,
            *(
                candidate
                for candidate in (
                    field_audit.primary_winning_evidence,
                    field_audit.winning_evidence,
                )
                if candidate is not None
            ),
        )
        deduplicated_list = []
        for candidate in considered:
            if candidate not in deduplicated_list:
                deduplicated_list.append(candidate)
        deduplicated = tuple(deduplicated_list)
        fields[field_name] = (
            replace(
                original,
                state=field_audit.final_evidence_state,
                value=field_audit.final_evidence_value,
                winning_evidence=field_audit.winning_evidence,
                considered=deduplicated,
                reason="accepted production recovery audit",
            )
            if original is not None
            else ResolvedField(
                field_name=field_name,
                state=field_audit.final_evidence_state,
                value=field_audit.final_evidence_value,
                winning_evidence=field_audit.winning_evidence,
                considered=deduplicated,
                reason="accepted production recovery audit",
            )
        )
    return replace(resolved_case, fields=MappingProxyType(fields))


def _resolved_policy_fingerprint(resolved_case: ResolvedCase) -> str:
    """Compare non-output policy state without serializing case identity."""

    payload = {
        "active_scope_present": resolved_case.active_applicant is not None,
        "unresolved_linkage": resolved_case.unresolved_linkage,
        "unresolved_reasons": sorted(resolved_case.unresolved_reasons),
        "rescinded_decision": resolved_case.rescinded_decision,
        "policy_fields": {
            field_name: {
                "state": field.state.value,
                "value": field.value,
                "winner_type": (
                    field.winning_evidence.evidence_type.value
                    if field.winning_evidence is not None
                    else None
                ),
                "winner_source": (
                    field.winning_evidence.source
                    if field.winning_evidence is not None
                    else None
                ),
            }
            for field_name, field in sorted(resolved_case.fields.items())
            if field_name not in _OUTPUT_EVIDENCE_FIELDS
        },
    }
    return _sha256_bytes(_canonical_bytes(payload))


def _select_accepted_resolved(
    records: Sequence[ResolvedCase],
    *,
    audit: RecoveryAuditOverlay,
    fusion_audit_counts: Mapping[str, int],
) -> ResolvedCase:
    candidates = [
        value
        for value in records
        if value.case_id == audit.case_id
        and dict(value.fusion_audit_counts) == dict(fusion_audit_counts)
    ]
    if not candidates:
        raise ProductionCaptureError(
            "no recorded resolved state matches the accepted fusion audit"
        )
    fingerprints = {
        _resolved_policy_fingerprint(candidate) for candidate in candidates
    }
    if len(fingerprints) != 1:
        raise ProductionCaptureError(
            "accepted resolver state is ambiguous across policy evidence"
        )
    return _overlay_audit_fields(candidates[-1], audit)


def _final_outcome(
    row: PredictionRow,
    resolved_case: ResolvedCase,
    records: Sequence[tuple[ResolvedCase, AdjudicationOutcome]],
) -> AdjudicationOutcome:
    matching = [
        outcome
        for recorded, outcome in records
        if recorded.case_id == resolved_case.case_id
        and dict(recorded.fusion_audit_counts)
        == dict(resolved_case.fusion_audit_counts)
        and _resolved_policy_fingerprint(recorded)
        == _resolved_policy_fingerprint(resolved_case)
        and outcome.row.adjudication == row.adjudication
        and outcome.trace.decision == row.adjudication
    ]
    if matching:
        trace_fingerprints = {
            _sha256_bytes(
                _canonical_bytes(
                    {
                        "decision": outcome.trace.decision,
                        "authoritative_source": (
                            outcome.trace.authoritative_source
                        ),
                        "denial_reasons": list(
                            outcome.trace.denial_reasons
                        ),
                        "review_reasons": list(
                            outcome.trace.review_reasons
                        ),
                        "approval_facts": list(
                            outcome.trace.approval_facts
                        ),
                        "exception_ids": list(
                            outcome.trace.exception_ids
                        ),
                    }
                )
            )
            for outcome in matching
        }
        if len(trace_fingerprints) != 1:
            raise ProductionCaptureError(
                "accepted adjudication trace is ambiguous"
            )
        return AdjudicationOutcome(
            row=row,
            trace=matching[-1].trace,
        )

    decision = row.adjudication
    reason = "accepted_runtime_recovery"
    return AdjudicationOutcome(
        row=row,
        trace=DecisionTrace(
            decision=decision,
            authoritative_source=False,
            denial_reasons=(reason,) if decision == "DENIED" else (),
            review_reasons=(
                (reason,) if decision == "NEEDS_REVIEW" else ()
            ),
            approval_facts=(reason,) if decision == "APPROVED" else (),
            exception_ids=(),
        ),
    )


def _recovery_route(
    audit: RecoveryAuditOverlay,
    policy_audit_counts: Mapping[str, int],
) -> str:
    recovered_sources = {
        field.recovery_source
        for field in audit.fields.values()
        if field.serialization_after_origin
        is SerializationOrigin.RECOVERED_VISIBLE_EVIDENCE
    }
    if None in recovered_sources:
        raise ProductionCaptureError(
            "recovered visible evidence has no auditable route"
        )
    if RAPID_RECOVERY_ROUTE_ID in recovered_sources:
        return "rapid_visible"
    if (
        policy_audit_counts.get(
            "late_recovery_before_revalidation_count",
            0,
        )
        > 0
    ):
        return "late_visible"
    return "primary"


def capture_production_case(
    pdf_path: Path,
    *,
    processor_factory: Callable[[], Any] = build_production_processor,
) -> CapturedProductionCase:
    """Run one PDF through the exact production graph and derive features."""

    outer = processor_factory()
    rapid = getattr(outer, "processor", None)
    recalibrator = getattr(outer, "recalibrator", None)
    if (
        rapid is None
        or recalibrator is None
        or not hasattr(rapid, "process_case_with_audit")
        or not hasattr(rapid, "_resolver")
        or not hasattr(rapid, "_adjudicator")
    ):
        raise ProductionCaptureError(
            "production composition root has an unsupported graph shape"
        )
    recording_resolver = _RecordingResolver(rapid._resolver)
    recording_adjudicator = _RecordingAdjudicator(rapid._adjudicator)
    rapid._resolver = recording_resolver
    rapid._adjudicator = recording_adjudicator

    result = rapid.process_case_with_audit(pdf_path)
    row = recalibrator.recalibrate(result.row)
    if not isinstance(row, PredictionRow):
        raise ProductionCaptureError(
            "production confidence stage returned a non-PredictionRow"
        )
    resolved = _select_accepted_resolved(
        recording_resolver.records,
        audit=result.audit,
        fusion_audit_counts=result.fusion_audit_counts,
    )
    outcome = _final_outcome(
        row,
        resolved,
        recording_adjudicator.records,
    )
    route = _recovery_route(
        result.audit,
        result.policy_audit_counts,
    )
    features = IdentityFreeFeatureBuilder().build(
        resolved,
        outcome,
        recovery_route=route,
    )
    return CapturedProductionCase(
        row=row,
        resolved_case=resolved,
        outcome=outcome,
        recovery_route=route,
        features=features,
    )


def build_feature_payload(
    *,
    input_dir: Path,
    layout_manifest: FrozenLayoutManifest,
    source_revision_sha: str,
    expected_input_tree_sha256: str,
    capture_case: Callable[[Path], CapturedProductionCase] = (
        capture_production_case
    ),
) -> Mapping[str, object]:
    """Capture every frozen PDF exactly once without consulting truth labels."""

    revision = str(source_revision_sha).strip().casefold()
    if not _GIT_COMMIT_RE.fullmatch(revision):
        raise ProductionCaptureError(
            "source_revision_sha must be a full Git commit SHA"
        )
    expected_tree = str(expected_input_tree_sha256).strip().casefold()
    if not _SHA256_RE.fullmatch(expected_tree):
        raise ProductionCaptureError(
            "expected_input_tree_sha256 must be a full SHA-256 digest"
        )
    actual_tree, pdf_count = _input_tree_sha256(input_dir)
    if (
        actual_tree != expected_tree
        or pdf_count != len(layout_manifest.case_ids)
    ):
        raise ProductionCaptureError(
            "input tree does not match the frozen cohort binding"
        )

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for pdf_path in sorted(input_dir.glob("*.pdf"), key=lambda path: path.name):
        captured = capture_case(pdf_path)
        case_id = captured.row.case_id
        if case_id in seen or case_id not in set(layout_manifest.case_ids):
            raise ProductionCaptureError(
                "production capture returned duplicate or unexpected case identity"
            )
        seen.add(case_id)
        rows.append(
            {
                "case_id": case_id,
                "baseline_prediction": captured.row.to_dict(),
                "baseline_decision": captured.row.adjudication,
                "recovery_route": captured.recovery_route,
                "features": captured.features.to_dict(),
            }
        )
    if seen != set(layout_manifest.case_ids):
        raise ProductionCaptureError(
            "production capture did not cover the exact frozen cohort"
        )
    return {
        "schema_version": FEATURE_ROWS_SCHEMA,
        "frozen_before_fit": True,
        "source_revision_sha": revision,
        "layout_manifest_sha256": layout_manifest.sha256,
        "input_tree_sha256": actual_tree,
        "feature_schema_sha256": feature_schema_sha256(),
        "rows": sorted(rows, key=lambda row: str(row["case_id"])),
    }


def _verify_clean_revision(source_revision_sha: str) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != source_revision_sha:
        raise ProductionCaptureError(
            "capture source revision does not match repository HEAD"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ProductionCaptureError(
            "production capture requires a clean committed checkout"
        )


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument("--expected-input-tree-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerun-output", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    arguments = parser.parse_args(argv)

    revision = arguments.source_revision_sha.strip().casefold()
    _verify_clean_revision(revision)
    manifest = load_layout_manifest(arguments.layout_manifest)
    first = build_feature_payload(
        input_dir=arguments.input_dir,
        layout_manifest=manifest,
        source_revision_sha=revision,
        expected_input_tree_sha256=(
            arguments.expected_input_tree_sha256
        ),
    )
    second = build_feature_payload(
        input_dir=arguments.input_dir,
        layout_manifest=manifest,
        source_revision_sha=revision,
        expected_input_tree_sha256=(
            arguments.expected_input_tree_sha256
        ),
    )
    first_bytes = _canonical_bytes(first)
    second_bytes = _canonical_bytes(second)
    if first_bytes != second_bytes:
        raise ProductionCaptureError(
            "production feature capture is not byte-deterministic"
        )
    _write_bytes(arguments.output, first_bytes)
    _write_bytes(arguments.rerun_output, second_bytes)
    output_sha = _sha256_bytes(first_bytes)
    observation = {
        "schema_version": CAPTURE_OBSERVATION_SCHEMA,
        "source_revision_sha": revision,
        "producer_source_sha256": _sha256_file(Path(__file__)),
        "producer_graph_sha256": producer_graph_sha256(),
        "layout_manifest_sha256": manifest.sha256,
        "input_tree_sha256": (
            arguments.expected_input_tree_sha256.strip().casefold()
        ),
        "feature_schema_sha256": feature_schema_sha256(),
        "record_count": len(manifest.case_ids),
        "capture_run_count": 2,
        "feature_rows_sha256": output_sha,
        "rerun_feature_rows_sha256": output_sha,
        "byte_deterministic": True,
        "truth_or_role_input_count": 0,
    }
    _write_bytes(arguments.observation, _canonical_bytes(observation))
    print(canonical_json(observation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
