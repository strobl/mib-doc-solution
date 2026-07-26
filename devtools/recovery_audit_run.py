#!/usr/bin/env python3
"""Run the production WO-15 recovery audit and emit aggregate counters only.

The production composition root wraps visible recovery with final confidence
recalibration.  Confidence does not alter the recovery overlay, so this tool
uses the exact wrapped production recovery processor and calls its explicit
``process_case_with_audit`` API.  Per-case rows and audit details remain in
memory and are never serialized.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
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
from mib_pipeline import build_production_processor  # noqa: E402
from mib_pipeline.batch import discover_case_pdfs  # noqa: E402
from mib_pipeline.extraction import EvidenceType  # noqa: E402
from mib_pipeline.recovery_audit import (  # noqa: E402
    RecoveryFieldAudit,
    SerializationOrigin,
    VisibleRecoveryResult,
)
from mib_pipeline.resolution import FieldState  # noqa: E402


_SOURCE_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


class RecoveryAuditRunError(ValueError):
    """The production audit could not be proven complete and aggregate-only."""


def _validate_source_revision(source_revision: str) -> str:
    normalized = str(source_revision).strip().casefold()
    if not _SOURCE_REVISION_RE.fullmatch(normalized):
        raise RecoveryAuditRunError(
            "source revision must be a full Git commit or SHA-256 digest"
        )
    return normalized


def _production_audit_processor(
    processor_factory: Callable[[], Any],
) -> Any:
    """Resolve the audited stage from the exact production composition root."""

    production = processor_factory()
    audited = getattr(production, "processor", None)
    if audited is None or not callable(
        getattr(audited, "process_case_with_audit", None)
    ):
        raise RecoveryAuditRunError(
            "production composition root does not expose an audited recovery stage"
        )
    return audited


def _same_scope(left: str, right: str) -> bool:
    return " ".join(left.split()).casefold() == " ".join(right.split()).casefold()


def _rect_key(box: Any) -> tuple[float, float, float, float]:
    return tuple(
        round(float(value), 6)
        for value in (box.left, box.bottom, box.right, box.top)
    )


def _has_complete_recovery_provenance(field: RecoveryFieldAudit) -> bool:
    """Recheck the evidence contract without serializing its identifying data."""

    candidate = field.winning_evidence
    recovery_source = field.recovery_source
    linked_scope = field.linked_recovery_scope
    if (
        candidate is None
        or recovery_source is None
        or linked_scope is None
        or candidate.field_name != field.field_name
        or candidate.value != field.final_evidence_value
        or candidate.value != field.serialization_after
        or candidate.source != "visible_ocr"
        or candidate.evidence_type is EvidenceType.TEXT_LAYER
        or not candidate.legible
        or candidate.superseded
        or not candidate.ocr_provenance
    ):
        return False

    provenance = candidate.ocr_provenance
    source_digests = {
        item.observation.source_sha256 for item in provenance
    }
    if len(source_digests) != 1:
        return False
    if any(
        item.observation.page_index != candidate.page_index
        for item in provenance
    ):
        return False
    if not any(
        item.route_id == recovery_source
        and (
            _rect_key(item.view_box) == _rect_key(candidate.box)
            or _rect_key(item.observation.box) == _rect_key(candidate.box)
        )
        for item in provenance
    ):
        return False

    observed_scopes = tuple(field.observed_applicant_scopes)
    if any(not _same_scope(scope, linked_scope) for scope in observed_scopes):
        return False
    if any(
        item.observation.applicant_scope is not None
        and not _same_scope(item.observation.applicant_scope, linked_scope)
        for item in provenance
    ):
        return False
    return True


def _default_used_as_evidence(field: RecoveryFieldAudit) -> bool:
    """Detect an output default that is incorrectly represented as evidence."""

    return (
        field.serialization_after_origin
        is SerializationOrigin.OUTPUT_DEFAULT
        and (
            field.final_evidence_state is FieldState.RESOLVED
            or field.final_evidence_value is not None
            or field.recovery_source is not None
            or field.winning_evidence is not None
        )
    )


def aggregate_recovery_results(
    results: Sequence[VisibleRecoveryResult],
) -> dict[str, int]:
    """Reduce valid in-memory audits to the three broker-safe counters."""

    recovered_field_count = 0
    recovered_field_complete_provenance_count = 0
    serialization_default_used_as_evidence_count = 0

    for result in results:
        if not isinstance(result, VisibleRecoveryResult):
            raise RecoveryAuditRunError(
                "production audit returned an invalid result type"
            )
        if not result.audit.fields:
            raise RecoveryAuditRunError(
                "production audit returned no field audits"
            )
        for field_name, field in result.audit.fields.items():
            if (
                not isinstance(field, RecoveryFieldAudit)
                or field_name != field.field_name
                or not hasattr(result.row, field_name)
                or getattr(result.row, field_name)
                != field.serialization_after
            ):
                raise RecoveryAuditRunError(
                    "production audit contains an invalid field record"
                )

            default_used = _default_used_as_evidence(field)
            serialization_default_used_as_evidence_count += int(default_used)
            if default_used:
                raise RecoveryAuditRunError(
                    "production audit treated a serialization default as evidence"
                )

            if (
                field.serialization_after_origin
                is SerializationOrigin.RECOVERED_VISIBLE_EVIDENCE
            ):
                recovered_field_count += 1
                complete = _has_complete_recovery_provenance(field)
                recovered_field_complete_provenance_count += int(complete)
                if not complete:
                    raise RecoveryAuditRunError(
                        "recovered field has incomplete provenance"
                    )
            elif field.recovery_source is not None:
                raise RecoveryAuditRunError(
                    "non-recovered field carries a recovery source"
                )

    return {
        "recovered_field_count": recovered_field_count,
        "recovered_field_complete_provenance_count": (
            recovered_field_complete_provenance_count
        ),
        "serialization_default_used_as_evidence_count": (
            serialization_default_used_as_evidence_count
        ),
    }


def run_recovery_audit(
    *,
    input_dir: Path | str,
    source_revision: str,
    max_workers: int = 4,
    processor_factory: Callable[[], Any] | None = None,
) -> dict[str, object]:
    """Run every top-level PDF and return one identity-free aggregate object."""

    source_digest = _validate_source_revision(source_revision)
    if not 1 <= max_workers <= 4:
        raise RecoveryAuditRunError("max_workers must be between 1 and 4")
    directory = Path(input_dir)
    if not directory.is_dir():
        raise RecoveryAuditRunError("input directory must exist")
    pdf_paths = discover_case_pdfs(directory)
    if not pdf_paths:
        raise RecoveryAuditRunError("input directory contains no PDF cases")

    factory = processor_factory or build_production_processor
    processor = _production_audit_processor(factory)

    def process_one(pdf_path: Path) -> VisibleRecoveryResult:
        try:
            result = processor.process_case_with_audit(pdf_path)
        except Exception as exc:
            raise RecoveryAuditRunError(
                "production audit processing failed"
            ) from exc
        if not isinstance(result, VisibleRecoveryResult):
            raise RecoveryAuditRunError(
                "production audit returned an invalid result type"
            )
        return result

    if len(pdf_paths) == 1 or max_workers == 1:
        results = tuple(process_one(path) for path in pdf_paths)
    else:
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="mib-recovery-audit",
        ) as executor:
            results = tuple(executor.map(process_one, pdf_paths))

    payload: dict[str, object] = {
        "counts": aggregate_recovery_results(results),
        "source_revision_sha": source_digest,
    }
    require_aggregate_only(payload)
    return payload


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
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
        description="Run aggregate-only WO-15 production recovery auditing."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        aggregate = run_recovery_audit(
            input_dir=args.input_dir,
            source_revision=args.source_revision,
            max_workers=args.max_workers,
        )
        _atomic_write(
            Path(args.output),
            canonical_json(aggregate) + "\n",
        )
    except (OSError, RecoveryAuditRunError, ExperimentControlError) as exc:
        print(f"recovery audit error: {exc}", file=sys.stderr)
        return 1
    print("WO-15 aggregate recovery audit complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
