#!/usr/bin/env python3
"""Run the production WO-16 resolver and emit aggregate fusion counters.

The exact production composition root remains responsible for rendering,
extraction, linking, resolution, adjudication, recovery, and final confidence
recalibration.  This tool changes no decisions: it temporarily wraps the
inner fusion resolver with a thread-safe observer, delegates every call, and
reduces only the identity-free ``ResolvedCase.fusion_audit_counts`` mappings.
The comparison is deliberately resolver-local: fusion versus the legacy
resolver after the current case linker. Exactly one accepted final resolver
result is counted per output case; optional recovery candidates that later
fail closed are excluded. Final-row score and safety deltas are measured
separately by the official evaluator. Prediction rows are written only to an
ephemeral directory and are never included in the public audit artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
import threading
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
from devtools.fusion_audit_contract import (  # noqa: E402
    FUSION_AUDIT_COMPARISON_SCOPE,
    FUSION_AUDIT_INVOCATION_SCOPE,
)
from mib_pipeline import BatchRunner, build_production_processor  # noqa: E402
from mib_pipeline.batch import discover_case_pdfs  # noqa: E402


SOURCE_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
REQUIRED_FUSION_COUNTS = (
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
UNSAFE_FUSION_COUNTS = (
    "clean_higher_authority_override_count",
    "binding_authority_override_count",
    "text_layer_winner_count",
    "serialization_default_used_as_evidence_count",
)


class FusionAuditRunError(ValueError):
    """The production fusion audit could not be proven complete and safe."""


def cohort_tree_sha256(pdf_paths: Sequence[Path]) -> str:
    """Hash the ordered input filenames and bytes without exposing identity."""

    digest = hashlib.sha256()
    for path in sorted(
        (Path(path) for path in pdf_paths),
        key=lambda item: (item.name.casefold(), item.name),
    ):
        name_bytes = path.name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        digest.update(file_digest.digest())
    return digest.hexdigest()


def case_id_set_sha256(case_ids: Sequence[str]) -> str:
    """Hash a canonical unique case-ID set without publishing its members."""

    normalized = tuple(sorted({str(case_id) for case_id in case_ids}))
    return hashlib.sha256(
        canonical_json(normalized).encode("utf-8")
    ).hexdigest()


def _validate_source_revision(source_revision: str) -> str:
    normalized = str(source_revision).strip().casefold()
    if not SOURCE_REVISION_RE.fullmatch(normalized):
        raise FusionAuditRunError(
            "source revision must be a full Git commit or SHA-256 digest"
        )
    return normalized


def _validate_final_counts(counts: Any) -> dict[str, int]:
    if not isinstance(counts, Mapping):
        raise FusionAuditRunError(
            "accepted final result did not expose fusion audit counters"
        )
    missing = tuple(
        name for name in REQUIRED_FUSION_COUNTS if name not in counts
    )
    if missing:
        raise FusionAuditRunError(
            "accepted final result is missing fusion audit counters: "
            + ", ".join(missing)
        )

    validated: dict[str, int] = {}
    for name in REQUIRED_FUSION_COUNTS:
        value = counts[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FusionAuditRunError(
                f"fusion audit counter {name} must be a non-negative integer"
            )
        validated[name] = value
    if (
        validated["changed_field_complete_provenance_count"]
        > validated["changed_field_count"]
    ):
        raise FusionAuditRunError(
            "complete provenance count cannot exceed changed field count"
        )
    return validated


class _FinalFusionAuditCollector:
    """Thread-safe collector for the one accepted fusion result per case."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepted: list[dict[str, int]] = []
        self._errors: list[str] = []

    def record(self, counts: Any) -> None:
        """Retain one already-selected final resolver audit mapping."""

        try:
            validated = _validate_final_counts(counts)
        except FusionAuditRunError as exc:
            with self._lock:
                self._errors.append(str(exc))
            return
        with self._lock:
            self._accepted.append(validated)

    def aggregate(self, *, expected_count: int) -> dict[str, int]:
        with self._lock:
            errors = tuple(self._errors)
            accepted = tuple(dict(counts) for counts in self._accepted)
        if errors:
            raise FusionAuditRunError(errors[0])
        if not accepted:
            raise FusionAuditRunError(
                "production run recorded no accepted final fusion results"
            )
        if len(accepted) != expected_count:
            raise FusionAuditRunError(
                "accepted final fusion result count does not match "
                "the output cohort"
            )

        aggregate = {
            name: sum(counts[name] for counts in accepted)
            for name in REQUIRED_FUSION_COUNTS
        }
        if (
            aggregate["changed_field_complete_provenance_count"]
            > aggregate["changed_field_count"]
        ):
            raise FusionAuditRunError(
                "complete provenance count cannot exceed changed field count"
            )
        unsafe = tuple(
            name for name in UNSAFE_FUSION_COUNTS if aggregate[name] != 0
        )
        if unsafe:
            raise FusionAuditRunError(
                "unsafe fusion audit counters must remain zero: "
                + ", ".join(unsafe)
            )
        return aggregate


def _instrument_production_processor(
    processor_factory: Callable[[], Any],
) -> tuple[Any, object, bool, _FinalFusionAuditCollector]:
    production = processor_factory()
    inner = getattr(production, "processor", None)
    if inner is None or not callable(getattr(inner, "process_case", None)):
        raise FusionAuditRunError(
            "production composition root does not expose its inner processor"
        )
    if not hasattr(inner, "_resolver"):
        raise FusionAuditRunError(
            "production inner processor does not expose its resolver"
        )
    resolver = getattr(inner, "_resolver")
    if getattr(resolver, "fusion_enabled", None) is not True:
        raise FusionAuditRunError(
            "production resolver is not fusion-enabled"
        )
    collector = _FinalFusionAuditCollector()
    observer_existed = hasattr(inner, "_fusion_audit_observer")
    original_observer = getattr(inner, "_fusion_audit_observer", None)
    try:
        setattr(inner, "_fusion_audit_observer", collector.record)
    except (AttributeError, TypeError) as exc:
        raise FusionAuditRunError(
            "production final fusion result cannot be instrumented"
        ) from exc
    if getattr(inner, "_fusion_audit_observer", None) != collector.record:
        raise FusionAuditRunError(
            "production final fusion instrumentation was not retained"
        )
    return production, original_observer, observer_existed, collector


def run_fusion_audit(
    *,
    input_dir: Path | str,
    source_revision: str,
    max_workers: int = 4,
    processor_factory: Callable[[], Any] | None = None,
) -> dict[str, object]:
    """Run all top-level PDFs and return only revision-bound aggregate counts."""

    source_digest = _validate_source_revision(source_revision)
    if not 1 <= max_workers <= 4:
        raise FusionAuditRunError("max_workers must be between 1 and 4")
    directory = Path(input_dir)
    if not directory.is_dir():
        raise FusionAuditRunError("input directory must exist")
    pdf_paths = discover_case_pdfs(directory)
    if not pdf_paths:
        raise FusionAuditRunError("input directory contains no PDF cases")

    factory = processor_factory or build_production_processor
    production, original_observer, observer_existed, collector = (
        _instrument_production_processor(factory)
    )
    inner = production.processor
    try:
        with tempfile.TemporaryDirectory(
            prefix="mib-wo16-fusion-audit-"
        ) as temporary_directory:
            prediction_path = Path(temporary_directory) / "predictions.jsonl"
            report = BatchRunner(
                production,
                max_workers=max_workers,
            ).run(directory, prediction_path)
            line_count = sum(
                1
                for line in prediction_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
    finally:
        if observer_existed:
            setattr(inner, "_fusion_audit_observer", original_observer)
        else:
            try:
                delattr(inner, "_fusion_audit_observer")
            except AttributeError:
                pass

    if (
        report.attempted != len(pdf_paths)
        or report.answered != len(pdf_paths)
        or report.omitted != 0
        or report.failures
        or line_count != len(pdf_paths)
    ):
        raise FusionAuditRunError(
            "production output is incomplete for the top-level PDF cohort"
        )

    payload: dict[str, object] = {
        "case_id_set_sha256": case_id_set_sha256(
            tuple(path.stem for path in pdf_paths)
        ),
        "comparison_scope": FUSION_AUDIT_COMPARISON_SCOPE,
        "counts": collector.aggregate(expected_count=len(pdf_paths)),
        "input_pdf_count": len(pdf_paths),
        "input_tree_sha256": cohort_tree_sha256(pdf_paths),
        "invocation_scope": FUSION_AUDIT_INVOCATION_SCOPE,
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
        description="Run aggregate-only WO-16 production fusion auditing."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        aggregate = run_fusion_audit(
            input_dir=args.input_dir,
            source_revision=args.source_revision,
            max_workers=args.max_workers,
        )
        _atomic_write(
            Path(args.output),
            canonical_json(aggregate) + "\n",
        )
    except (OSError, FusionAuditRunError, ExperimentControlError) as exc:
        print(f"fusion audit error: {exc}", file=sys.stderr)
        return 1
    print("WO-16 aggregate fusion audit complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
