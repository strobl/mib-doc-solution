#!/usr/bin/env python3
"""Capture truth-blind WO13 runtime dimensions without changing predictions.

This development-only runner constructs the production processor graph and
adds delegating observers around it.  Observers never alter arguments or
return values.  The per-case trace is finalized only when the canonical
prediction bytes match the explicitly pinned baseline SHA-256.
"""

from __future__ import annotations

import argparse
import contextlib
import contextvars
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
while str(_REPOSITORY_ROOT) in sys.path:
    sys.path.remove(str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_REPOSITORY_ROOT))

from mib_pipeline import (
    AdjudicationEngine,
    BatchRunner,
    CanonicalJsonlWriter,
    CaseLinker,
    ConfidenceCalibrator,
    DocumentRenderer,
    EvidencePrecedenceResolver,
    GeneralizablePolicyExceptionStore,
    OutputConfidenceRecalibrationProcessor,
    OutputConfidenceRecalibrator,
    PredictionRow,
    RapidOutputRecoveryProcessor,
    ReviewDenialRecoveryAdjudicator,
    VisibleEvidenceExtractor,
    build_rapid_extractor,
    discover_case_pdfs,
)
from mib_pipeline.batch import BatchRunReport

from devtools.wo13_trace_contract import (
    TRACE_SCHEMA_VERSION,
    atomic_write_trace,
    canonical_json_bytes,
    sha256_path,
    validate_trace_capture,
)
from devtools.grouped_split_evidence import (
    GroupedSplitEvidenceBuildError,
    _strict_freezer_manifest,
    _verify_input_tree_and_recomputed_manifest,
)
from devtools import layout_manifest_freezer


_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_BASELINE_PREDICTIONS_SHA256 = (
    "d6e23641a4e4c7a5517c2b691791146177665f5c297667adae17565f6918a42d"
)
_FROZEN_BASELINE_SCHEMA = "mib-frozen-baseline/v1"
_RUNTIME_CONTRACT_SCHEMA = "mib-wo17-runtime-contract/v1"
_AUTHORITY_SCHEMA = "mib-wo13-capture-authority/v1"
_GIT_EXECUTABLE = Path("/usr/bin/git")
_PRODUCTION_GIT_PATHS = (
    "Dockerfile",
    "run.sh",
    "solution.py",
    "requirements.lock",
    "mib_pipeline",
    "third_party_licenses",
)
_TRACE_GIT_PATHS = (
    "devtools/__init__.py",
    "devtools/wo13_trace_capture.py",
    "devtools/wo13_trace_contract.py",
    "devtools/grouped_split_evidence.py",
    "devtools/layout_manifest_freezer.py",
    "devtools/experiment_control.py",
    "scripts/score_loss_atlas.py",
)
_CONTAINER_GRAPH_PATHS = (
    Path("Dockerfile"),
    Path("run.sh"),
    Path("requirements.lock"),
)
_PINNED_ARTIFACT_PATHS = (
    Path("mib_pipeline/artifacts/confidence_calibration.json"),
    Path("mib_pipeline/artifacts/policy_exceptions.json"),
    Path("mib_pipeline/artifacts/output_confidence_recalibration.json"),
)
_RUNTIME_SOURCE_PATHS = (
    Path("solution.py"),
    *tuple(
        sorted(
            (
                path.relative_to(_REPOSITORY_ROOT)
                for path in (_REPOSITORY_ROOT / "mib_pipeline").glob("*.py")
            ),
            key=lambda path: path.as_posix(),
        )
    ),
    *tuple(
        sorted(
            (
                path.relative_to(_REPOSITORY_ROOT)
                for path in (
                    _REPOSITORY_ROOT / "mib_pipeline" / "artifacts"
                ).glob("*.json")
            ),
            key=lambda path: path.as_posix(),
        )
    ),
)


_AUTHORITY_PATH_KEYS = frozenset(
    {
        "input_dir",
        "layout_manifest",
        "dataset_archive",
        "runtime_contract",
        "frozen_baseline_manifest",
        "baseline_predictions",
    }
)
_AUTHORITY_HASH_KEYS = frozenset(
    {
        "input_tree_sha256",
        "layout_manifest_sha256",
        "dataset_archive_sha256",
        "runtime_contract_sha256",
        "frozen_baseline_manifest_sha256",
        "baseline_predictions_sha256",
        "runtime_graph_sha256",
        "trace_tool_sha256",
        "container_graph_sha256",
        "source_snapshot_sha256",
    }
)
_RUNTIME_IDENTITY_KEYS = frozenset(
    {
        "python_version",
        "python_implementation",
        "python_isolated",
        "python_dont_write_bytecode",
        "python_executable_sha256",
        "dependency_versions",
        "dependency_identity_sha256",
        "environment",
        "max_workers",
        "interface",
        "container_limits",
        "runtime_identity_sha256",
    }
)
_AUTHORITY_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "source_revision_sha",
        "approved_paths",
        "expected_hashes",
        "expected_record_count",
        "retry_missing_attempts",
        "runtime_identity",
    }
)
_RUNTIME_CONTRACT_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "capture",
        "container_limits",
        "environment",
        "evaluation",
        "interface",
    }
)
_RUNTIME_CAPTURE_KEYS = frozenset(
    {
        "arm_repeat_count",
        "execution",
        "max_workers",
        "metrics_source",
        "required_byte_determinism",
    }
)
_RUNTIME_LIMIT_KEYS = frozenset(
    {
        "image_bytes",
        "max_model_artifact_bytes",
        "model_bytes",
        "network",
        "output_bytes",
        "peak_memory_bytes",
        "per_record_runtime_seconds",
        "runtime_seconds",
        "tmp_bytes",
    }
)
_RUNTIME_EVALUATION_KEYS = frozenset(
    {
        "evaluator_sha256",
        "evidence_label",
        "expected_record_count",
        "input_tree_sha256",
        "layout_manifest_sha256",
        "truth_sha256",
    }
)
_RUNTIME_INTERFACE_KEYS = frozenset(
    {"entrypoint", "input", "output", "runner"}
)
_EXPECTED_RUNTIME_INTERFACE = {
    "entrypoint": "solution.py",
    "input": "directory containing canonical PDF cases",
    "output": "canonical twelve-field JSONL",
    "runner": "run.sh",
}
_EXPECTED_RUNTIME_ENVIRONMENT_KEYS = frozenset(
    {
        "BLIS_NUM_THREADS",
        "HOME",
        "MALLOC_ARENA_MAX",
        "MIB_MAX_WORKERS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OC_DISABLE_DOT_ACCESS_WARNING",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "TMPDIR",
        "TOKENIZERS_PARALLELISM",
        "VECLIB_MAXIMUM_THREADS",
    }
)
_EXPECTED_RUNTIME_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "4",
    "HOME": "/tmp",
    "MALLOC_ARENA_MAX": "4",
    "MIB_MAX_WORKERS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OC_DISABLE_DOT_ACCESS_WARNING": "1",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "TMPDIR": "/tmp",
    "TOKENIZERS_PARALLELISM": "false",
    "VECLIB_MAXIMUM_THREADS": "4",
}
_EXPECTED_CONTAINER_LIMITS = {
    "image_bytes": 4_294_967_296,
    "max_model_artifact_bytes": 262_144_000,
    "model_bytes": 1_073_741_824,
    "network": "none",
    "output_bytes": 26_214_400,
    "peak_memory_bytes": 8_589_934_592,
    "per_record_runtime_seconds": 6,
    "runtime_seconds": 14_400,
    "tmp_bytes": 2_147_483_648,
}
_CASE_PDF_NAME_RE = re.compile(r"^MIB-[0-9]{6}\.pdf$")
_MAX_DATASET_ARCHIVE_BYTES = 2_147_483_648
_MAX_DATASET_UNCOMPRESSED_BYTES = 2_147_483_648
_ALLOWED_ZIP_COMPRESSION = frozenset(
    {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
)


class TraceCaptureError(RuntimeError):
    """The capture cannot be finalized as exact-baseline evidence."""


@dataclass(frozen=True)
class CaptureAuthority:
    """One canonical preregistration plus all recomputed live bindings."""

    manifest_path: Path
    manifest_sha256: str
    payload: Mapping[str, Any]
    approved_paths: Mapping[str, Path]
    expected_hashes: Mapping[str, str]
    runtime_identity: Mapping[str, Any]

    @property
    def capture_source_revision_sha(self) -> str:
        return str(self.payload["source_revision_sha"])

    @property
    def expected_record_count(self) -> int:
        return int(self.payload["expected_record_count"])

    @property
    def max_workers(self) -> int:
        return int(self.runtime_identity["max_workers"])

    @property
    def retry_missing_attempts(self) -> int:
        return int(self.payload["retry_missing_attempts"])


@dataclass
class TraceSignals:
    """Mutable, context-local observations for one case invocation."""

    primary_candidates: Any = None
    rapid_candidates: Any = None
    linked_cases: list[Any] = field(default_factory=list)
    resolved_cases: list[Any] = field(default_factory=list)
    primary_outcome: Any = None
    ocr_paths: set[str] = field(default_factory=set)
    rapid_recovery_attempted: bool = False
    rapid_output_changed: bool = False
    rapid_authority_applied: bool = False
    semantic_denial_applied: bool = False
    applicant_repaired: bool = False
    source_fields_repaired: bool = False
    review_head_changed: bool = False


_CURRENT_SIGNALS: contextvars.ContextVar[TraceSignals | None] = (
    contextvars.ContextVar("wo13_current_trace_signals", default=None)
)


def current_trace_signals() -> TraceSignals | None:
    """Return the current case's observer state, if capture is active."""

    return _CURRENT_SIGNALS.get()


def _observe(callback: Callable[[TraceSignals], None]) -> None:
    signals = _CURRENT_SIGNALS.get()
    if signals is not None:
        callback(signals)


class TraceCollector:
    """Lock-protected collection with one immutable aggregate row per case."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._failed_attempt_seconds: dict[str, float] = {}
        self._lock = threading.Lock()

    def add_failed_attempt(self, case_id: str, runtime_seconds: float) -> None:
        with self._lock:
            self._failed_attempt_seconds[case_id] = (
                self._failed_attempt_seconds.get(case_id, 0.0)
                + runtime_seconds
            )

    def add(self, row: Mapping[str, Any]) -> None:
        case_id = str(row["case_id"])
        with self._lock:
            if case_id in self._rows:
                raise TraceCaptureError(
                    f"duplicate trace row for case_id {case_id}"
                )
            normalized = dict(row)
            normalized["runtime_seconds"] = float(
                normalized["runtime_seconds"]
            ) + self._failed_attempt_seconds.pop(case_id, 0.0)
            self._rows[case_id] = normalized

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(self._rows[case_id])
                for case_id in sorted(self._rows)
            ]


class DelegatingExtractorObserver:
    """Delegate extraction exactly while observing accepted retry paths."""

    _RETRY_METHODS = {
        "_fee_receipt_retry_evidence": "consensus_retry",
        "_consensus_retry_evidence": "consensus_retry",
        "_sparse_intake_retry_evidence": "sparse_intake_retry",
        "_orientation_retry_evidence": "orientation_retry",
        "_risk_flag_retry_evidence": "risk_flag_retry",
    }

    def __init__(self, delegate: Any, *, role: str) -> None:
        if role not in {"primary", "rapid"}:
            raise ValueError("extractor observer role must be primary or rapid")
        self._delegate = delegate
        self._role = role
        if role == "primary":
            self._instrument_retry_methods()

    def _instrument_retry_methods(self) -> None:
        for method_name, category in self._RETRY_METHODS.items():
            original = getattr(self._delegate, method_name)

            def observed(
                *args: Any,
                _original: Callable[..., Any] = original,
                _category: str = category,
                **kwargs: Any,
            ) -> Any:
                result = _original(*args, **kwargs)
                if result:
                    _observe(lambda signals: signals.ocr_paths.add(_category))
                return result

            setattr(self._delegate, method_name, observed)

    def extract(self, rendered_case: Any) -> Any:
        result = self._delegate.extract(rendered_case)
        if self._role == "primary":
            _observe(
                lambda signals: setattr(
                    signals,
                    "primary_candidates",
                    result,
                )
            )
        else:
            def record_rapid(signals: TraceSignals) -> None:
                signals.rapid_candidates = result
                signals.ocr_paths.add("targeted_rapidocr")

            _observe(record_rapid)
        return result


class DelegatingLinkerObserver:
    """Delegate case linking and retain only the typed result out of band."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def link(self, expected_case_id: str | None, candidates: Iterable[Any]) -> Any:
        result = self._delegate.link(expected_case_id, candidates)
        _observe(lambda signals: signals.linked_cases.append(result))
        return result


class DelegatingResolverObserver:
    """Delegate evidence resolution and retain only the typed result."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def resolve(self, linked_case: Any) -> Any:
        result = self._delegate.resolve(linked_case)
        _observe(lambda signals: signals.resolved_cases.append(result))
        return result


class DelegatingAdjudicatorObserver:
    """Delegate primary adjudication and retain its auditable trace."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def adjudicate_case(self, resolved_case: Any) -> Any:
        result = self._delegate.adjudicate_case(resolved_case)
        _observe(lambda signals: setattr(signals, "primary_outcome", result))
        return result


def _instrument_rapid_processor(processor: Any) -> Any:
    """Attach return-preserving observers to one dedicated processor instance."""

    original_recover = processor._recover

    def observed_recover(*args: Any, **kwargs: Any) -> Any:
        signals = current_trace_signals()
        provisional_snapshot = None
        if signals is not None:
            provisional_snapshot = (
                signals.rapid_authority_applied,
                signals.semantic_denial_applied,
                signals.rapid_output_changed,
                signals.applicant_repaired,
                signals.source_fields_repaired,
                signals.review_head_changed,
            )
            signals.rapid_recovery_attempted = True
        try:
            result = original_recover(*args, **kwargs)
        except BaseException:
            if signals is not None and provisional_snapshot is not None:
                (
                    signals.rapid_authority_applied,
                    signals.semantic_denial_applied,
                    signals.rapid_output_changed,
                    signals.applicant_repaired,
                    signals.source_fields_repaired,
                    signals.review_head_changed,
                ) = provisional_snapshot
            raise
        primary_row = kwargs.get("primary_row")
        if (
            isinstance(primary_row, PredictionRow)
            and isinstance(result, PredictionRow)
            and primary_row != result
        ):
            _observe(
                lambda signals: setattr(
                    signals,
                    "rapid_output_changed",
                    True,
                )
            )
        return result

    processor._recover = observed_recover

    original_authority = processor._authoritative_rapid_decision

    def observed_authority(*args: Any, **kwargs: Any) -> Any:
        result = original_authority(*args, **kwargs)
        if result is not None:
            _observe(
                lambda signals: setattr(
                    signals,
                    "rapid_authority_applied",
                    True,
                )
            )
        return result

    processor._authoritative_rapid_decision = observed_authority

    original_semantic = processor._semantic_denial_rules

    def observed_semantic(*args: Any, **kwargs: Any) -> Any:
        result = original_semantic(*args, **kwargs)
        if result:
            _observe(
                lambda signals: setattr(
                    signals,
                    "semantic_denial_applied",
                    True,
                )
            )
        return result

    processor._semantic_denial_rules = observed_semantic

    original_biometric = processor._repair_biometric_applicant

    def observed_biometric(*args: Any, **kwargs: Any) -> Any:
        result = original_biometric(*args, **kwargs)
        if result[1]:
            _observe(
                lambda signals: setattr(
                    signals,
                    "applicant_repaired",
                    True,
                )
            )
        return result

    processor._repair_biometric_applicant = observed_biometric

    original_source_repair = processor._repair_source_priority_fields

    def observed_source_repair(*args: Any, **kwargs: Any) -> Any:
        result = original_source_repair(*args, **kwargs)
        if result[1]:
            _observe(
                lambda signals: setattr(
                    signals,
                    "source_fields_repaired",
                    True,
                )
            )
        return result

    processor._repair_source_priority_fields = observed_source_repair

    original_review_heads = processor._apply_review_approval_heads

    def observed_review_heads(*args: Any, **kwargs: Any) -> Any:
        before = kwargs.get("final_row")
        result = original_review_heads(*args, **kwargs)
        if (
            isinstance(before, PredictionRow)
            and isinstance(result, PredictionRow)
            and before.adjudication != result.adjudication
        ):
            _observe(
                lambda signals: setattr(
                    signals,
                    "review_head_changed",
                    True,
                )
            )
        return result

    processor._apply_review_approval_heads = observed_review_heads
    return processor


def _items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    try:
        return tuple(value)
    except TypeError:
        return ()


def classify_provenance(signals: TraceSignals) -> str:
    outcome = signals.primary_outcome
    trace = getattr(outcome, "trace", None)
    if (
        signals.rapid_authority_applied
        or bool(getattr(trace, "authoritative_source", False))
    ):
        return "authoritative_source"
    primary_resolved = (
        signals.resolved_cases[0] if signals.resolved_cases else None
    )
    primary_fields = getattr(primary_resolved, "fields", {}) or {}
    primary_accepted = bool(
        signals.applicant_repaired
        or signals.source_fields_repaired
        or any(
            getattr(field, "winning_evidence", None) is not None
            for field in primary_fields.values()
        )
    )
    rapid_accepted = bool(signals.rapid_output_changed)
    if primary_accepted and rapid_accepted:
        return "mixed_visible_sources"
    if primary_accepted or rapid_accepted:
        return "visible_ocr"
    if _items(signals.primary_candidates) and (
        not signals.resolved_cases
        or not hasattr(primary_resolved, "fields")
    ):
        # Test doubles and pre-resolution observers can still establish that
        # visible OCR, rather than an implicit/default source, was accepted.
        return "visible_ocr"
    return "no_accepted_provenance"


def classify_linking(signals: TraceSignals) -> str:
    outcome = signals.primary_outcome
    trace = getattr(outcome, "trace", None)
    if (
        signals.rapid_authority_applied
        or bool(getattr(trace, "authoritative_source", False))
    ):
        return "authoritative_scope"
    if not signals.linked_cases:
        return "unknown"
    linked = signals.linked_cases[0]
    active_applicant = getattr(linked, "active_applicant", None)
    unresolved = bool(getattr(linked, "unresolved", False))
    if active_applicant is not None and not unresolved:
        return "linked_unique"
    if active_applicant is not None:
        return "linked_ambiguous"
    return "unlinked"


def classify_conflict(signals: TraceSignals) -> str:
    if not signals.resolved_cases and not signals.linked_cases:
        return "unknown"
    primary_resolved = (
        signals.resolved_cases[0] if signals.resolved_cases else None
    )
    contested = set(getattr(primary_resolved, "contested_fields", ()) or ())
    reasons: set[str] = set()
    if signals.linked_cases:
        reasons.update(
            str(reason).casefold()
            for reason in (
                getattr(signals.linked_cases[0], "unresolved_reasons", ()) or ()
            )
        )
    outcome = signals.primary_outcome
    trace = getattr(outcome, "trace", None)
    review_reasons = {
        str(reason).casefold()
        for reason in (getattr(trace, "review_reasons", ()) or ())
    }
    identity = bool(
        contested.intersection({"applicant_name", "case_id"})
        or any(
            "identity" in reason
            or "applicant" in reason
            or "case_id" in reason
            for reason in reasons | review_reasons
        )
    )
    authority = bool(
        "adjudication" in contested
        or "authoritative_visible_decision" in review_reasons
    )
    field_conflict = bool(contested - {"applicant_name", "case_id", "adjudication"})
    kinds = sum((identity, authority, field_conflict))
    if kinds > 1:
        return "multiple_conflicts"
    if identity:
        return "identity_conflict"
    if authority:
        return "authority_conflict"
    if field_conflict:
        return "field_conflict"
    return "none"


def classify_ocr_path(signals: TraceSignals) -> str:
    paths = set(signals.ocr_paths)
    if len(paths) > 1:
        return "multiple_recovery_paths"
    if paths:
        return next(iter(paths))
    if _items(signals.primary_candidates):
        return "primary"
    return "no_visible_ocr"


def classify_policy(signals: TraceSignals, row: PredictionRow) -> str:
    outcome = signals.primary_outcome
    trace = getattr(outcome, "trace", None)
    if (
        signals.rapid_authority_applied
        or bool(getattr(trace, "authoritative_source", False))
    ):
        return "binding_authority"
    denial_reasons = tuple(getattr(trace, "denial_reasons", ()) or ())
    approval_facts = tuple(getattr(trace, "approval_facts", ()) or ())
    if signals.semantic_denial_applied or any(
        str(reason).startswith("review_denial_") for reason in denial_reasons
    ):
        return "recovery_denial"
    if (
        signals.review_head_changed
        or any(
            str(fact).startswith(("review_", "xw1_"))
            for fact in approval_facts
        )
    ):
        return "revalidated_policy"
    if row.adjudication == "NEEDS_REVIEW":
        if classify_conflict(signals) not in {"none", "unknown"}:
            return "needs_review_conflict"
        if signals.rapid_recovery_attempted:
            return "recovery_review"
    return "deterministic_policy"


class TracingCaseProcessor:
    """Outermost context boundary that returns the delegate row unchanged."""

    def __init__(self, delegate: Any, collector: TraceCollector) -> None:
        self._delegate = delegate
        self._collector = collector

    def process_case(self, pdf_path: Path) -> PredictionRow | None:
        signals = TraceSignals()
        token = _CURRENT_SIGNALS.set(signals)
        started = time.perf_counter()
        try:
            try:
                row = self._delegate.process_case(pdf_path)
            except BaseException:
                self._collector.add_failed_attempt(
                    pdf_path.stem,
                    time.perf_counter() - started,
                )
                raise
            if row is None:
                self._collector.add_failed_attempt(
                    pdf_path.stem,
                    time.perf_counter() - started,
                )
                return None
            try:
                if not isinstance(row, PredictionRow):
                    row = PredictionRow.from_mapping(
                        row,
                        fallback_case_id=pdf_path.stem,
                    )
                trace_row = {
                    "case_id": row.case_id,
                    "provenance_route": classify_provenance(signals),
                    "applicant_linking_state": classify_linking(signals),
                    "evidence_conflict": classify_conflict(signals),
                    "ocr_recovery_path": classify_ocr_path(signals),
                    "policy_trace": classify_policy(signals, row),
                    "runtime_seconds": time.perf_counter() - started,
                }
            except BaseException:
                self._collector.add_failed_attempt(
                    pdf_path.stem,
                    time.perf_counter() - started,
                )
                raise
            self._collector.add(trace_row)
            return row
        finally:
            _CURRENT_SIGNALS.reset(token)


def build_observed_production_processor(
    *,
    artifact_root: Path | None = None,
) -> Any:
    """Construct the exact production graph with return-preserving observers.

    ``artifact_root`` is used by the authoritative path only.  It points to a
    private byte-verified snapshot of the three checked-in policy artifacts,
    preventing a transient worktree mutation from influencing a long capture.
    """

    if artifact_root is None:
        confidence_artifact = None
        policy_artifact = None
        output_confidence_artifact = None
    else:
        artifact_root = Path(artifact_root)
        confidence_artifact = (
            artifact_root / "confidence_calibration.json"
        )
        policy_artifact = artifact_root / "policy_exceptions.json"
        output_confidence_artifact = (
            artifact_root / "output_confidence_recalibration.json"
        )
        for path in (
            confidence_artifact,
            policy_artifact,
            output_confidence_artifact,
        ):
            if not path.is_file():
                raise TraceCaptureError(
                    "private policy-artifact snapshot is incomplete"
                )

    primary_extractor = DelegatingExtractorObserver(
        VisibleEvidenceExtractor(packet_page_type_markers=True),
        role="primary",
    )
    linker = DelegatingLinkerObserver(CaseLinker())
    resolver = DelegatingResolverObserver(EvidencePrecedenceResolver())
    adjudicator = DelegatingAdjudicatorObserver(
        ReviewDenialRecoveryAdjudicator(
            AdjudicationEngine(
                calibrator=(
                    ConfidenceCalibrator.from_pinned_artifact(
                        confidence_artifact
                    )
                    if confidence_artifact is not None
                    else ConfidenceCalibrator.from_pinned_artifact()
                ),
                exceptions=(
                    GeneralizablePolicyExceptionStore.from_pinned_artifact(
                        policy_artifact
                    )
                    if policy_artifact is not None
                    else GeneralizablePolicyExceptionStore.from_pinned_artifact()
                ),
            )
        )
    )

    def rapid_extractor_factory() -> DelegatingExtractorObserver:
        return DelegatingExtractorObserver(
            build_rapid_extractor(),
            role="rapid",
        )

    rapid = RapidOutputRecoveryProcessor(
        renderer=DocumentRenderer(),
        primary_extractor=primary_extractor,
        linker=linker,
        resolver=resolver,
        adjudicator=adjudicator,
        rapid_extractor_factory=rapid_extractor_factory,
    )
    return OutputConfidenceRecalibrationProcessor(
        processor=_instrument_rapid_processor(rapid),
        recalibrator=(
            OutputConfidenceRecalibrator.from_pinned_artifact(
                output_confidence_artifact
            )
            if output_confidence_artifact is not None
            else OutputConfidenceRecalibrator.from_pinned_artifact()
        ),
    )


def _graph_sha256(
    relative_paths: Sequence[Path],
    *,
    repository_root: Path = _REPOSITORY_ROOT,
) -> str:
    manifest = []
    for relative_path in relative_paths:
        path = Path(repository_root) / relative_path
        manifest.append(
            {
                "path": relative_path.as_posix(),
                "sha256": sha256_path(path),
            }
        )
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runtime_graph_sha256(repository_root: Path = _REPOSITORY_ROOT) -> str:
    """Hash the exact production code/artifact graph observed by this tool."""

    return _graph_sha256(
        _RUNTIME_SOURCE_PATHS,
        repository_root=repository_root,
    )


def container_graph_sha256(
    repository_root: Path = _REPOSITORY_ROOT,
) -> str:
    """Bind the Docker image recipe, entrypoint, and dependency lock."""

    return _graph_sha256(
        _CONTAINER_GRAPH_PATHS,
        repository_root=repository_root,
    )


def compute_input_tree_sha256(input_dir: Path) -> str:
    """Stream the canonical filename/content digest for the actual PDFs."""

    digest = hashlib.sha256()
    for path in discover_case_pdfs(Path(input_dir)):
        name_bytes = path.name.encode("utf-8")
        digest.update(len(name_bytes).to_bytes(4, "big"))
        digest.update(name_bytes)
        digest.update(bytes.fromhex(sha256_path(path)))
    return digest.hexdigest()


def trace_tool_graph_sha256(
    repository_root: Path = _REPOSITORY_ROOT,
) -> str:
    """Hash capture, contract, atlas, and canonical WO12 freezer sources."""

    return _graph_sha256(
        tuple(Path(path) for path in _TRACE_GIT_PATHS),
        repository_root=repository_root,
    )


def _capture_source_paths(
    repository_root: Path = _REPOSITORY_ROOT,
) -> tuple[Path, ...]:
    root = Path(repository_root)
    paths = {
        *_RUNTIME_SOURCE_PATHS,
        *_CONTAINER_GRAPH_PATHS,
        *(Path(path) for path in _TRACE_GIT_PATHS),
    }
    license_root = root / "third_party_licenses"
    if license_root.is_dir():
        paths.update(
            path.relative_to(root)
            for path in license_root.rglob("*")
            if path.is_file()
        )
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def source_snapshot_sha256(
    repository_root: Path = _REPOSITORY_ROOT,
) -> str:
    """Hash every file whose bytes define capture or container execution."""

    return _graph_sha256(
        _capture_source_paths(repository_root),
        repository_root=repository_root,
    )


@dataclass(frozen=True)
class TraceCaptureResult:
    report: BatchRunReport
    predictions_sha256: str
    trace_sha256: str
    batch_wall_seconds: float
    retry_passes_used: int


def _require_digest(value: str, *, label: str) -> str:
    normalized = str(value).casefold()
    if not _SHA256_RE.fullmatch(normalized):
        raise TraceCaptureError(f"{label} must be a lowercase SHA-256")
    return normalized


def _read_json_object(
    path: Path,
    *,
    label: str,
) -> tuple[Mapping[str, Any], str]:
    try:
        content = _read_stable_regular_file(Path(path), label=label)
        value = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceCaptureError(f"{label} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise TraceCaptureError(f"{label} must be a JSON object")
    return value, hashlib.sha256(content).hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _read_stable_regular_file(path: Path, *, label: str) -> bytes:
    """Read a non-symlink regular file and detect replacement during read."""

    requested = Path(path)
    descriptor = -1
    try:
        if requested.is_symlink():
            raise TraceCaptureError(f"{label} must not be a symlink")
        resolved = requested.resolve(strict=True)
        descriptor = os.open(
            str(resolved),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        initial = os.fstat(descriptor)
        entry = os.stat(resolved, follow_symlinks=False)
        if (
            not stat.S_ISREG(initial.st_mode)
            or (initial.st_dev, initial.st_ino)
            != (entry.st_dev, entry.st_ino)
        ):
            raise TraceCaptureError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final = os.fstat(descriptor)
        final_entry = os.stat(resolved, follow_symlinks=False)
        if (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
            initial.st_ctime_ns,
        ) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) or (final.st_dev, final.st_ino) != (
            final_entry.st_dev,
            final_entry.st_ino,
        ):
            raise TraceCaptureError(f"{label} changed while being read")
        return b"".join(chunks)
    except TraceCaptureError:
        raise
    except OSError as exc:
        raise TraceCaptureError(f"{label} could not be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def verify_dataset_archive_authority(
    path: Path,
    *,
    expected_record_count: int,
    expected_input_tree_sha256: str,
) -> dict[str, Any]:
    """Stream one stable canonical ZIP and bind it to the exact PDF tree."""

    expected_count = _require_positive_int(
        expected_record_count,
        label="dataset archive expected_record_count",
    )
    expected_tree = _require_digest(
        expected_input_tree_sha256,
        label="dataset archive expected_input_tree_sha256",
    )
    requested = Path(path)
    descriptor = -1
    handle = None
    try:
        if requested.is_symlink():
            raise TraceCaptureError(
                "dataset archive must not be a symlink"
            )
        resolved = requested.resolve(strict=True)
        descriptor = os.open(
            str(resolved),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        initial = os.fstat(descriptor)
        entry = os.stat(resolved, follow_symlinks=False)
        if (
            not stat.S_ISREG(initial.st_mode)
            or (initial.st_dev, initial.st_ino)
            != (entry.st_dev, entry.st_ino)
        ):
            raise TraceCaptureError(
                "dataset archive must be a stable regular file"
            )
        if initial.st_size > _MAX_DATASET_ARCHIVE_BYTES:
            raise TraceCaptureError(
                "dataset archive exceeds the bounded archive size"
            )
        handle = os.fdopen(descriptor, "rb", closefd=False)
        archive_digest = hashlib.sha256()
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            archive_digest.update(chunk)
        handle.seek(0)
        try:
            archive = zipfile.ZipFile(handle, mode="r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise TraceCaptureError(
                "dataset archive is not a valid canonical ZIP"
            ) from exc
        with archive:
            infos = archive.infolist()
            if len(infos) != expected_count:
                raise TraceCaptureError(
                    "dataset archive member count is incorrect"
                )
            seen_names: set[str] = set()
            total_uncompressed = 0
            for info in infos:
                name = info.filename
                unix_mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    not isinstance(name, str)
                    or _CASE_PDF_NAME_RE.fullmatch(name) is None
                    or info.is_dir()
                    or "/" in name
                    or "\\" in name
                    or stat.S_ISLNK(unix_mode)
                ):
                    raise TraceCaptureError(
                        "dataset archive members must be root MIB PDFs"
                    )
                if name in seen_names:
                    raise TraceCaptureError(
                        "dataset archive contains duplicate members"
                    )
                seen_names.add(name)
                if info.flag_bits & 0x1:
                    raise TraceCaptureError(
                        "dataset archive members must not be encrypted"
                    )
                if info.compress_type not in _ALLOWED_ZIP_COMPRESSION:
                    raise TraceCaptureError(
                        "dataset archive compression is unsupported"
                    )
                if info.file_size < 1:
                    raise TraceCaptureError(
                        "dataset archive PDF members must be non-empty"
                    )
                total_uncompressed += info.file_size
                if total_uncompressed > _MAX_DATASET_UNCOMPRESSED_BYTES:
                    raise TraceCaptureError(
                        "dataset archive exceeds the uncompressed size bound"
                    )

            tree_digest = hashlib.sha256()
            for info in sorted(
                infos,
                key=lambda item: (
                    item.filename.casefold(),
                    item.filename,
                ),
            ):
                member_digest = hashlib.sha256()
                member_size = 0
                try:
                    with archive.open(info, mode="r") as member:
                        while True:
                            chunk = member.read(1024 * 1024)
                            if not chunk:
                                break
                            member_size += len(chunk)
                            if member_size > info.file_size:
                                raise TraceCaptureError(
                                    "dataset archive member size is invalid"
                                )
                            member_digest.update(chunk)
                except (
                    OSError,
                    RuntimeError,
                    NotImplementedError,
                    zipfile.BadZipFile,
                ) as exc:
                    raise TraceCaptureError(
                        "dataset archive member could not be verified"
                    ) from exc
                if member_size != info.file_size:
                    raise TraceCaptureError(
                        "dataset archive member size is invalid"
                    )
                name_bytes = info.filename.encode("utf-8")
                tree_digest.update(
                    len(name_bytes).to_bytes(4, "big")
                )
                tree_digest.update(name_bytes)
                tree_digest.update(member_digest.digest())
        final = os.fstat(descriptor)
        final_entry = os.stat(resolved, follow_symlinks=False)
        if (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
            initial.st_ctime_ns,
        ) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) or (final.st_dev, final.st_ino) != (
            final_entry.st_dev,
            final_entry.st_ino,
        ):
            raise TraceCaptureError(
                "dataset archive changed while being verified"
            )
        observed_tree = tree_digest.hexdigest()
        if observed_tree != expected_tree:
            raise TraceCaptureError(
                "dataset archive PDF tree differs from input authority"
            )
        return {
            "archive_sha256": archive_digest.hexdigest(),
            "input_tree_sha256": observed_tree,
            "record_count": expected_count,
            "uncompressed_bytes": total_uncompressed,
        }
    except TraceCaptureError:
        raise
    except (OSError, RuntimeError) as exc:
        raise TraceCaptureError(
            "dataset archive could not be read safely"
        ) from exc
    finally:
        if handle is not None:
            handle.close()
        if descriptor >= 0:
            os.close(descriptor)


def _canonical_absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TraceCaptureError(f"{label} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise TraceCaptureError(f"{label} must be an absolute path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TraceCaptureError(f"{label} does not exist") from exc
    if resolved.as_posix() != value:
        raise TraceCaptureError(
            f"{label} must be its canonical resolved absolute path"
        )
    return resolved


def _require_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TraceCaptureError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TraceCaptureError(f"{label} must be a non-negative integer")
    return value


def load_runtime_contract_authority(path: Path) -> dict[str, Any]:
    """Validate the complete runtime contract, not only evaluation hashes."""

    payload, file_sha256 = _read_json_object(
        path,
        label="runtime contract",
    )
    if (
        set(payload) != _RUNTIME_CONTRACT_ROOT_KEYS
        or payload.get("schema_version") != _RUNTIME_CONTRACT_SCHEMA
    ):
        raise TraceCaptureError("runtime contract schema is unsupported")
    capture = payload.get("capture")
    container_limits = payload.get("container_limits")
    environment = payload.get("environment")
    evaluation = payload.get("evaluation")
    interface = payload.get("interface")
    if (
        not isinstance(capture, Mapping)
        or set(capture) != _RUNTIME_CAPTURE_KEYS
        or not isinstance(container_limits, Mapping)
        or set(container_limits) != _RUNTIME_LIMIT_KEYS
        or not isinstance(environment, Mapping)
        or set(environment) != _EXPECTED_RUNTIME_ENVIRONMENT_KEYS
        or not isinstance(evaluation, Mapping)
        or set(evaluation) != _RUNTIME_EVALUATION_KEYS
        or not isinstance(interface, Mapping)
        or set(interface) != _RUNTIME_INTERFACE_KEYS
    ):
        raise TraceCaptureError(
            "runtime contract sections do not match the exact schema"
        )
    max_workers = _require_positive_int(
        capture.get("max_workers"),
        label="runtime contract max_workers",
    )
    if (
        max_workers != 4
        or capture.get("arm_repeat_count") != 2
        or capture.get("execution") != "sequential"
        or capture.get("metrics_source")
        != "fresh_process_rusage_self_plus_waited_children_and_monotonic_wall"
        or capture.get("required_byte_determinism") is not True
    ):
        raise TraceCaptureError("runtime capture contract is unsupported")
    if dict(interface) != _EXPECTED_RUNTIME_INTERFACE:
        raise TraceCaptureError("runtime interface contract is unsupported")
    normalized_environment: dict[str, str] = {}
    for name in sorted(_EXPECTED_RUNTIME_ENVIRONMENT_KEYS):
        value = environment.get(name)
        if (
            not isinstance(value, str)
            or value != _EXPECTED_RUNTIME_ENVIRONMENT[name]
        ):
            raise TraceCaptureError(
                "runtime environment does not match the container contract"
            )
        normalized_environment[name] = value
    normalized_limits: dict[str, Any] = {}
    for name in sorted(_RUNTIME_LIMIT_KEYS):
        value = container_limits.get(name)
        if name == "network":
            if value != "none":
                raise TraceCaptureError(
                    "runtime container network must be disabled"
                )
        else:
            _require_positive_int(
                value,
                label=f"runtime container limit {name}",
            )
        normalized_limits[name] = value
    if normalized_limits != _EXPECTED_CONTAINER_LIMITS:
        raise TraceCaptureError(
            "runtime container limits do not match the four-hour contract"
        )
    input_tree = _require_digest(
        str(evaluation.get("input_tree_sha256", "")),
        label="runtime contract input_tree_sha256",
    )
    layout_manifest = _require_digest(
        str(evaluation.get("layout_manifest_sha256", "")),
        label="runtime contract layout_manifest_sha256",
    )
    expected_count = evaluation.get("expected_record_count")
    if (
        _require_positive_int(
            expected_count,
            label="runtime contract expected_record_count",
        )
        != expected_count
    ):
        raise AssertionError("unreachable")
    for label in ("evaluator_sha256", "truth_sha256"):
        _require_digest(
            str(evaluation.get(label, "")),
            label=f"runtime contract {label}",
        )
    if (
        evaluation.get("evidence_label")
        != "public_grouped_robustness_not_unseen"
    ):
        raise TraceCaptureError("runtime evidence label is unsupported")
    return {
        "file_sha256": file_sha256,
        "input_tree_sha256": input_tree,
        "layout_manifest_sha256": layout_manifest,
        "expected_record_count": expected_count,
        "max_workers": max_workers,
        "environment": normalized_environment,
        "interface": dict(interface),
        "container_limits": normalized_limits,
        "interface_sha256": _canonical_digest(dict(interface)),
        "environment_sha256": _canonical_digest(normalized_environment),
        "container_limits_sha256": _canonical_digest(normalized_limits),
    }


def _locked_dependency_names(
    requirements_path: Path = _REPOSITORY_ROOT / "requirements.lock",
) -> tuple[str, ...]:
    names: set[str] = set()
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("#")
            or line.startswith("--")
            or line.startswith("\\")
        ):
            continue
        match = re.match(r"^([A-Za-z0-9_.-]+)==", line)
        if match is not None:
            names.add(match.group(1))
    if not names:
        raise TraceCaptureError("requirements.lock contains no pinned packages")
    return tuple(sorted(names, key=lambda item: item.casefold()))


def dependency_versions(
    repository_root: Path = _REPOSITORY_ROOT,
) -> dict[str, str]:
    """Resolve every direct locked distribution in the active interpreter."""

    versions: dict[str, str] = {}
    for name in _locked_dependency_names(
        Path(repository_root) / "requirements.lock"
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TraceCaptureError(
                f"locked dependency is not installed: {name}"
            ) from exc
    return versions


def _installed_dependency_graph_sha256(
    dependency_names: Sequence[str],
) -> str:
    """Hash every installed file declared by each locked distribution."""

    distributions: list[dict[str, Any]] = []
    for name in dependency_names:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise TraceCaptureError(
                f"locked dependency is not installed: {name}"
            ) from exc
        files = distribution.files
        if files is None:
            raise TraceCaptureError(
                f"dependency has no installed-file manifest: {name}"
            )
        file_rows: list[dict[str, str]] = []
        for relative in sorted(files, key=lambda item: str(item)):
            located = Path(distribution.locate_file(relative))
            if not located.is_file():
                continue
            file_rows.append(
                {
                    "path": str(relative),
                    "sha256": sha256_path(located),
                }
            )
        if not file_rows:
            raise TraceCaptureError(
                f"dependency has no hashable installed files: {name}"
            )
        distributions.append(
            {
                "name": name,
                "version": distribution.version,
                "files_sha256": _canonical_digest(file_rows),
            }
        )
    return _canonical_digest(distributions)


def runtime_identity(
    runtime_contract: Mapping[str, Any],
    *,
    repository_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Return the exact interpreter/dependency/execution identity in use."""

    expected_environment = runtime_contract["environment"]
    observed_environment = {
        name: os.environ.get(name)
        for name in sorted(expected_environment)
    }
    if observed_environment != expected_environment:
        raise TraceCaptureError(
            "process environment does not match the runtime contract"
        )
    executable = Path(sys.executable).resolve(strict=True)
    versions = dependency_versions(repository_root)
    dependency_payload = {
        "requirements_lock_sha256": sha256_path(
            Path(repository_root) / "requirements.lock"
        ),
        "versions": versions,
        "installed_distribution_graph_sha256": (
            _installed_dependency_graph_sha256(tuple(versions))
        ),
    }
    identity: dict[str, Any] = {
        "python_version": sys.version,
        "python_implementation": sys.implementation.name,
        "python_isolated": bool(sys.flags.isolated),
        "python_dont_write_bytecode": bool(
            sys.flags.dont_write_bytecode
        ),
        "python_executable_sha256": sha256_path(executable),
        "dependency_versions": versions,
        "dependency_identity_sha256": _canonical_digest(dependency_payload),
        "environment": dict(expected_environment),
        "max_workers": runtime_contract["max_workers"],
        "interface": dict(runtime_contract["interface"]),
        "container_limits": dict(runtime_contract["container_limits"]),
    }
    identity["runtime_identity_sha256"] = _canonical_digest(identity)
    return identity


def _validate_runtime_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RUNTIME_IDENTITY_KEYS:
        raise TraceCaptureError(
            "authority runtime_identity does not match the exact schema"
        )
    identity = dict(value)
    if (
        not isinstance(identity["python_version"], str)
        or not identity["python_version"]
        or identity["python_implementation"] != "cpython"
        or identity["python_isolated"] is not True
        or identity["python_dont_write_bytecode"] is not True
        or not isinstance(identity["dependency_versions"], Mapping)
        or not identity["dependency_versions"]
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in identity["dependency_versions"].items()
        )
        or not isinstance(identity["environment"], Mapping)
        or set(identity["environment"])
        != _EXPECTED_RUNTIME_ENVIRONMENT_KEYS
        or not isinstance(identity["interface"], Mapping)
        or set(identity["interface"]) != _RUNTIME_INTERFACE_KEYS
        or not isinstance(identity["container_limits"], Mapping)
        or set(identity["container_limits"]) != _RUNTIME_LIMIT_KEYS
    ):
        raise TraceCaptureError("authority runtime identity is invalid")
    _require_digest(
        identity["python_executable_sha256"],
        label="authority python executable sha256",
    )
    _require_digest(
        identity["dependency_identity_sha256"],
        label="authority dependency identity sha256",
    )
    declared_digest = _require_digest(
        identity["runtime_identity_sha256"],
        label="authority runtime identity sha256",
    )
    unsigned = dict(identity)
    unsigned.pop("runtime_identity_sha256")
    if _canonical_digest(unsigned) != declared_digest:
        raise TraceCaptureError(
            "authority runtime identity digest does not match its content"
        )
    _require_positive_int(
        identity["max_workers"],
        label="authority max_workers",
    )
    return identity


def load_frozen_baseline_authority(
    manifest_path: Path,
    baseline_predictions_path: Path,
) -> dict[str, Any]:
    """Verify actual baseline bytes against both the manifest and pinned hash."""

    payload, manifest_sha256 = _read_json_object(
        manifest_path,
        label="frozen baseline manifest",
    )
    if payload.get("schema") != _FROZEN_BASELINE_SCHEMA:
        raise TraceCaptureError("frozen baseline manifest schema is unsupported")
    metadata = payload.get("metadata")
    artifacts = payload.get("artifacts")
    if not isinstance(metadata, Mapping) or not isinstance(artifacts, list):
        raise TraceCaptureError("frozen baseline manifest sections are invalid")
    source_revision = str(metadata.get("baseline_commit_sha", "")).casefold()
    if not _REVISION_RE.fullmatch(source_revision):
        raise TraceCaptureError("frozen baseline source revision is invalid")
    expected_count = metadata.get("count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise TraceCaptureError("frozen baseline count must be positive")
    declared_predictions: str | None = None
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise TraceCaptureError("frozen baseline artifact is invalid")
        if artifact.get("path") == "external/full1000_predictions.jsonl":
            if declared_predictions is not None:
                raise TraceCaptureError(
                    "frozen baseline declares predictions more than once"
                )
            declared_predictions = _require_digest(
                str(artifact.get("sha256", "")),
                label="frozen baseline predictions sha256",
            )
    if declared_predictions != EXPECTED_BASELINE_PREDICTIONS_SHA256:
        raise TraceCaptureError(
            "frozen baseline does not declare the pinned d6e236 prediction bytes"
        )
    observed_prediction_bytes = _read_stable_regular_file(
        baseline_predictions_path,
        label="frozen baseline predictions",
    )
    observed_predictions = hashlib.sha256(
        observed_prediction_bytes
    ).hexdigest()
    if observed_predictions != EXPECTED_BASELINE_PREDICTIONS_SHA256:
        raise TraceCaptureError(
            "baseline prediction file does not match pinned d6e236 bytes"
        )
    return {
        "manifest_sha256": manifest_sha256,
        "predictions_sha256": observed_predictions,
        "source_revision_sha": source_revision,
        "expected_record_count": expected_count,
    }


def authoritative_git_binding(
    baseline_source_revision_sha: str,
    *,
    capture_source_revision_sha: str | None = None,
    repository_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Require clean exact HEAD tools and baseline-identical production bytes."""

    baseline_revision = str(baseline_source_revision_sha).casefold()
    if not _REVISION_RE.fullmatch(baseline_revision):
        raise TraceCaptureError(
            "baseline source revision must be a full Git SHA"
        )
    capture_revision = (
        str(capture_source_revision_sha).casefold()
        if capture_source_revision_sha is not None
        else None
    )
    if capture_revision is not None and not _REVISION_RE.fullmatch(
        capture_revision
    ):
        raise TraceCaptureError(
            "capture source revision must be a full Git SHA"
        )
    if not _GIT_EXECUTABLE.is_file():
        raise TraceCaptureError("required Git executable is unavailable")
    sanitized_environment = {
        name: value
        for name, value in os.environ.items()
        if not name.upper().startswith("GIT_")
    }

    def run_git(
        arguments: Sequence[str],
        *,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [str(_GIT_EXECUTABLE), *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            env=sanitized_environment,
        )
        if check and completed.returncode != 0:
            raise TraceCaptureError(
                f"Git authority check failed: {' '.join(arguments)}"
            )
        return completed

    checkout = run_git(["rev-parse", "--verify", "HEAD"]).stdout.strip()
    if not _REVISION_RE.fullmatch(checkout):
        raise TraceCaptureError("checkout revision could not be resolved")
    if capture_revision is not None and checkout != capture_revision:
        raise TraceCaptureError(
            "checkout HEAD does not match preregistered capture source"
        )
    resolved_baseline = run_git(
        ["rev-parse", "--verify", f"{baseline_revision}^{{commit}}"]
    ).stdout.strip()
    if resolved_baseline != baseline_revision:
        raise TraceCaptureError(
            "frozen baseline revision did not resolve exactly"
        )
    full_status = run_git(
        [
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ]
    ).stdout
    if full_status.strip():
        raise TraceCaptureError(
            "capture source has modified or untracked files"
        )
    difference = run_git(
        [
            "diff",
            "--quiet",
            baseline_revision,
            "--",
            *_PRODUCTION_GIT_PATHS,
        ],
        check=False,
    )
    if difference.returncode != 0:
        raise TraceCaptureError(
            "production files do not match the frozen source revision"
        )
    for tool_path in _TRACE_GIT_PATHS:
        committed = run_git(
            ["cat-file", "-e", f"{checkout}:{tool_path}"],
            check=False,
        )
        if committed.returncode != 0:
            raise TraceCaptureError(
                f"trace authority source is not committed: {tool_path}"
            )
    return {
        "checkout_revision_sha": checkout,
        "source_revision_sha": baseline_revision,
        "capture_source_revision_sha": checkout,
        "runtime_graph_sha256": runtime_graph_sha256(repository_root),
        "trace_tool_sha256": trace_tool_graph_sha256(repository_root),
        "container_graph_sha256": container_graph_sha256(repository_root),
        "production_tree_verified": True,
        "capture_tree_verified": True,
    }


def _verify_layout_and_input(
    *,
    input_dir: Path,
    layout_manifest_path: Path,
    expected_layout_manifest_sha256: str,
    expected_input_tree_sha256: str,
) -> str:
    """Recompute canonical WO12 groups from the bound PDF bytes."""

    try:
        snapshot = _strict_freezer_manifest(
            layout_manifest_path,
            expected_sha256=expected_layout_manifest_sha256,
        )
        return _verify_input_tree_and_recomputed_manifest(
            input_dir,
            snapshot,
            expected_sha256=expected_input_tree_sha256,
        )
    except GroupedSplitEvidenceBuildError as exc:
        raise TraceCaptureError(
            f"canonical layout/input verification failed: {exc}"
        ) from exc


def _atomic_write_canonical(path: Path, value: Any) -> None:
    destination = Path(path)
    if destination.exists():
        raise TraceCaptureError(
            "authority manifest output already exists; preregistration "
            "is append-only"
        )
    if not destination.parent.is_dir():
        raise TraceCaptureError(
            "authority manifest output parent does not exist"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o400)
        os.replace(temporary_path, destination)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def prepare_capture_authority_manifest(
    *,
    output_path: Path,
    input_dir: Path,
    layout_manifest_path: Path,
    dataset_archive_path: Path,
    runtime_contract_path: Path,
    frozen_baseline_manifest_path: Path,
    baseline_predictions_path: Path,
    retry_missing_attempts: int = 1,
    repository_root: Path = _REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Create the canonical preregistration required by an authoritative run."""

    _require_external_output_path(
        Path(output_path),
        label="capture authority manifest",
    )
    approved = {
        "input_dir": _canonical_absolute_path(
            input_dir,
            label="input_dir",
        ),
        "layout_manifest": _canonical_absolute_path(
            layout_manifest_path,
            label="layout_manifest",
        ),
        "dataset_archive": _canonical_absolute_path(
            dataset_archive_path,
            label="dataset_archive",
        ),
        "runtime_contract": _canonical_absolute_path(
            runtime_contract_path,
            label="runtime_contract",
        ),
        "frozen_baseline_manifest": _canonical_absolute_path(
            frozen_baseline_manifest_path,
            label="frozen_baseline_manifest",
        ),
        "baseline_predictions": _canonical_absolute_path(
            baseline_predictions_path,
            label="baseline_predictions",
        ),
    }
    if len(set(approved.values())) != len(approved):
        raise TraceCaptureError(
            "authority inputs must use distinct canonical paths"
        )
    if not approved["input_dir"].is_dir():
        raise TraceCaptureError("input_dir must be a directory")
    for name, path in approved.items():
        if name not in {"input_dir", "dataset_archive"}:
            _read_stable_regular_file(path, label=name)
    retry_count = _require_nonnegative_int(
        retry_missing_attempts,
        label="retry_missing_attempts",
    )
    if retry_count > 3:
        raise TraceCaptureError(
            "retry_missing_attempts must not exceed three"
        )
    runtime_contract = load_runtime_contract_authority(
        approved["runtime_contract"]
    )
    baseline = load_frozen_baseline_authority(
        approved["frozen_baseline_manifest"],
        approved["baseline_predictions"],
    )
    git_binding = authoritative_git_binding(
        baseline["source_revision_sha"],
        repository_root=repository_root,
    )
    if (
        runtime_contract["expected_record_count"]
        != baseline["expected_record_count"]
    ):
        raise TraceCaptureError(
            "runtime contract and frozen baseline counts disagree"
        )
    archive_binding = verify_dataset_archive_authority(
        approved["dataset_archive"],
        expected_record_count=runtime_contract[
            "expected_record_count"
        ],
        expected_input_tree_sha256=runtime_contract[
            "input_tree_sha256"
        ],
    )
    _verify_layout_and_input(
        input_dir=approved["input_dir"],
        layout_manifest_path=approved["layout_manifest"],
        expected_layout_manifest_sha256=runtime_contract[
            "layout_manifest_sha256"
        ],
        expected_input_tree_sha256=runtime_contract[
            "input_tree_sha256"
        ],
    )
    identity = runtime_identity(
        runtime_contract,
        repository_root=repository_root,
    )
    payload = {
        "schema_version": _AUTHORITY_SCHEMA,
        "source_revision_sha": git_binding[
            "capture_source_revision_sha"
        ],
        "approved_paths": {
            name: path.as_posix()
            for name, path in sorted(approved.items())
        },
        "expected_hashes": {
            "input_tree_sha256": runtime_contract[
                "input_tree_sha256"
            ],
            "layout_manifest_sha256": runtime_contract[
                "layout_manifest_sha256"
            ],
            "dataset_archive_sha256": archive_binding[
                "archive_sha256"
            ],
            "runtime_contract_sha256": runtime_contract["file_sha256"],
            "frozen_baseline_manifest_sha256": baseline[
                "manifest_sha256"
            ],
            "baseline_predictions_sha256": baseline[
                "predictions_sha256"
            ],
            "runtime_graph_sha256": git_binding[
                "runtime_graph_sha256"
            ],
            "trace_tool_sha256": git_binding["trace_tool_sha256"],
            "container_graph_sha256": git_binding[
                "container_graph_sha256"
            ],
            "source_snapshot_sha256": source_snapshot_sha256(
                repository_root
            ),
        },
        "expected_record_count": runtime_contract[
            "expected_record_count"
        ],
        "retry_missing_attempts": retry_count,
        "runtime_identity": identity,
    }
    _atomic_write_canonical(Path(output_path), payload)
    return payload


def load_capture_authority(
    path: Path,
    *,
    repository_root: Path = _REPOSITORY_ROOT,
) -> CaptureAuthority:
    """Recompute every preregistered path, hash, source, and runtime fact."""

    content = _read_stable_regular_file(
        Path(path),
        label="capture authority manifest",
    )
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceCaptureError(
            "capture authority manifest is not valid UTF-8 JSON"
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or set(payload) != _AUTHORITY_ROOT_KEYS
        or payload.get("schema_version") != _AUTHORITY_SCHEMA
        or content != canonical_json_bytes(payload)
    ):
        raise TraceCaptureError(
            "capture authority manifest is not canonical exact-schema JSON"
        )
    source_revision = str(payload["source_revision_sha"]).casefold()
    if not _REVISION_RE.fullmatch(source_revision):
        raise TraceCaptureError(
            "authority source_revision_sha must be a full Git SHA"
        )
    raw_paths = payload["approved_paths"]
    raw_hashes = payload["expected_hashes"]
    if (
        not isinstance(raw_paths, Mapping)
        or set(raw_paths) != _AUTHORITY_PATH_KEYS
        or not isinstance(raw_hashes, Mapping)
        or set(raw_hashes) != _AUTHORITY_HASH_KEYS
    ):
        raise TraceCaptureError(
            "capture authority path/hash schema is invalid"
        )
    approved = {
        name: _canonical_absolute_path(
            value,
            label=f"authority {name}",
        )
        for name, value in raw_paths.items()
    }
    if len(set(approved.values())) != len(approved):
        raise TraceCaptureError(
            "authority inputs must use distinct canonical paths"
        )
    if not approved["input_dir"].is_dir():
        raise TraceCaptureError("authority input_dir must be a directory")
    expected_hashes = {
        name: _require_digest(
            value,
            label=f"authority {name}",
        )
        for name, value in raw_hashes.items()
    }
    expected_count = _require_positive_int(
        payload["expected_record_count"],
        label="authority expected_record_count",
    )
    retry_count = _require_nonnegative_int(
        payload["retry_missing_attempts"],
        label="authority retry_missing_attempts",
    )
    if retry_count > 3:
        raise TraceCaptureError(
            "authority retry_missing_attempts must not exceed three"
        )
    declared_identity = _validate_runtime_identity(
        payload["runtime_identity"]
    )
    runtime_contract = load_runtime_contract_authority(
        approved["runtime_contract"]
    )
    baseline = load_frozen_baseline_authority(
        approved["frozen_baseline_manifest"],
        approved["baseline_predictions"],
    )
    git_binding = authoritative_git_binding(
        baseline["source_revision_sha"],
        capture_source_revision_sha=source_revision,
        repository_root=repository_root,
    )
    observed_identity = runtime_identity(
        runtime_contract,
        repository_root=repository_root,
    )
    if observed_identity != declared_identity:
        raise TraceCaptureError(
            "active Python/dependency/runtime identity differs from authority"
        )
    archive_binding = verify_dataset_archive_authority(
        approved["dataset_archive"],
        expected_record_count=expected_count,
        expected_input_tree_sha256=expected_hashes[
            "input_tree_sha256"
        ],
    )
    observed_hashes = {
        "input_tree_sha256": runtime_contract["input_tree_sha256"],
        "layout_manifest_sha256": hashlib.sha256(
            _read_stable_regular_file(
                approved["layout_manifest"],
                label="authority layout_manifest",
            )
        ).hexdigest(),
        "dataset_archive_sha256": archive_binding["archive_sha256"],
        "runtime_contract_sha256": runtime_contract["file_sha256"],
        "frozen_baseline_manifest_sha256": baseline[
            "manifest_sha256"
        ],
        "baseline_predictions_sha256": baseline[
            "predictions_sha256"
        ],
        "runtime_graph_sha256": git_binding[
            "runtime_graph_sha256"
        ],
        "trace_tool_sha256": git_binding["trace_tool_sha256"],
        "container_graph_sha256": git_binding[
            "container_graph_sha256"
        ],
        "source_snapshot_sha256": source_snapshot_sha256(
            repository_root
        ),
    }
    if observed_hashes != expected_hashes:
        raise TraceCaptureError(
            "live authority hashes differ from preregistration"
        )
    if (
        runtime_contract["layout_manifest_sha256"]
        != expected_hashes["layout_manifest_sha256"]
        or runtime_contract["expected_record_count"] != expected_count
        or baseline["expected_record_count"] != expected_count
    ):
        raise TraceCaptureError(
            "authority record/layout facts disagree"
        )
    if len(discover_case_pdfs(approved["input_dir"])) != expected_count:
        raise TraceCaptureError(
            "authority input PDF count is incorrect"
        )
    _verify_layout_and_input(
        input_dir=approved["input_dir"],
        layout_manifest_path=approved["layout_manifest"],
        expected_layout_manifest_sha256=expected_hashes[
            "layout_manifest_sha256"
        ],
        expected_input_tree_sha256=expected_hashes[
            "input_tree_sha256"
        ],
    )
    return CaptureAuthority(
        manifest_path=Path(path).resolve(),
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        payload=dict(payload),
        approved_paths=approved,
        expected_hashes=expected_hashes,
        runtime_identity=declared_identity,
    )


def _require_external_output_path(path: Path, *, label: str) -> None:
    try:
        Path(path).resolve().relative_to(_REPOSITORY_ROOT)
    except ValueError:
        return
    raise TraceCaptureError(
        f"{label} must stay outside the repository"
    )


def _run_with_missing_retries(
    *,
    processor: Any,
    input_dir: Path,
    predictions_output: Path,
    max_workers: int,
    retry_missing_attempts: int,
) -> tuple[BatchRunReport, int]:
    """Run the normal bounded pool, retrying only omitted source cases."""

    if (
        isinstance(retry_missing_attempts, bool)
        or not isinstance(retry_missing_attempts, int)
        or not 0 <= retry_missing_attempts <= 3
    ):
        raise TraceCaptureError(
            "retry_missing_attempts must be an integer between zero and three"
        )
    paths = discover_case_pdfs(input_dir)
    runner = BatchRunner(processor, max_workers=max_workers)
    pending = paths
    successful: dict[str, PredictionRow] = {}
    final_failures: dict[str, Any] = {}
    retry_passes_used = 0
    for pass_index in range(retry_missing_attempts + 1):
        if not pending:
            break
        if pass_index:
            retry_passes_used += 1
        results = runner._process_all(pending)
        next_pending: list[Path] = []
        for path, result in zip(pending, results):
            if result.row is not None:
                successful[path.name] = result.row
                final_failures.pop(path.name, None)
            else:
                next_pending.append(path)
                final_failures[path.name] = result.failure
        pending = tuple(next_pending)

    CanonicalJsonlWriter().write(
        predictions_output,
        successful.values(),
    )
    failures = tuple(
        final_failures[path.name]
        for path in pending
        if final_failures.get(path.name) is not None
    )
    return (
        BatchRunReport(
            attempted=len(paths),
            answered=len(successful),
            omitted=len(failures),
            failures=failures,
        ),
        retry_passes_used,
    )


_STABILITY_BINDING_KEYS = {
    "input_tree_sha256": "input_tree_unchanged",
    "layout_manifest_sha256": "layout_manifest_unchanged",
    "dataset_archive_sha256": "dataset_archive_unchanged",
    "runtime_contract_sha256": "runtime_contract_unchanged",
    "frozen_baseline_manifest_sha256": (
        "frozen_baseline_manifest_unchanged"
    ),
    "baseline_predictions_sha256": "baseline_predictions_unchanged",
    "runtime_graph_sha256": "runtime_graph_unchanged",
    "trace_tool_sha256": "trace_tool_unchanged",
    "container_graph_sha256": "container_graph_unchanged",
    "source_snapshot_sha256": "source_snapshot_unchanged",
    "authority_manifest_sha256": "authority_manifest_unchanged",
    "runtime_identity_sha256": "runtime_identity_unchanged",
    "checkout_revision_sha": "checkout_revision_unchanged",
    "capture_source_revision_sha": (
        "capture_source_revision_unchanged"
    ),
    "production_tree_verified": "production_tree_unchanged",
}


@dataclass(frozen=True)
class PrivateCaptureSnapshot:
    input_dir: Path
    source_root: Path
    artifact_root: Path
    input_tree_sha256: str
    source_snapshot_sha256: str


@contextlib.contextmanager
def private_capture_snapshot(
    authority: CaptureAuthority,
    *,
    repository_root: Path = _REPOSITORY_ROOT,
) -> Iterable[PrivateCaptureSnapshot]:
    """Copy all consumed input/source bytes into one verified private tree."""

    external_root = layout_manifest_freezer._external_temporary_root()
    with tempfile.TemporaryDirectory(
        prefix="mib-wo13-authoritative-snapshot-",
        dir=str(external_root),
    ) as temporary_name:
        root = Path(temporary_name)
        input_snapshot = root / "input"
        source_snapshot = root / "source"
        input_snapshot.mkdir(mode=0o700)
        source_snapshot.mkdir(mode=0o700)
        pdfs = discover_case_pdfs(authority.approved_paths["input_dir"])
        if len(pdfs) != authority.expected_record_count:
            raise TraceCaptureError(
                "origin PDF count changed before snapshot creation"
            )
        for source_path in pdfs:
            target = input_snapshot / source_path.name
            try:
                layout_manifest_freezer._copy_pdf_snapshot(
                    source_path,
                    target,
                )
                os.chmod(target, 0o400)
            except Exception as exc:
                raise TraceCaptureError(
                    "could not create immutable PDF processing snapshot"
                ) from exc
        input_hash = compute_input_tree_sha256(input_snapshot)
        if input_hash != authority.expected_hashes["input_tree_sha256"]:
            raise TraceCaptureError(
                "processing snapshot input hash differs from authority"
            )
        archive_snapshot_binding = verify_dataset_archive_authority(
            authority.approved_paths["dataset_archive"],
            expected_record_count=authority.expected_record_count,
            expected_input_tree_sha256=input_hash,
        )
        if (
            archive_snapshot_binding["archive_sha256"]
            != authority.expected_hashes["dataset_archive_sha256"]
        ):
            raise TraceCaptureError(
                "dataset archive differs from snapshot authority"
            )
        _verify_layout_and_input(
            input_dir=input_snapshot,
            layout_manifest_path=authority.approved_paths[
                "layout_manifest"
            ],
            expected_layout_manifest_sha256=authority.expected_hashes[
                "layout_manifest_sha256"
            ],
            expected_input_tree_sha256=input_hash,
        )
        source_paths = _capture_source_paths(repository_root)
        for relative_path in source_paths:
            source = Path(repository_root) / relative_path
            target = source_snapshot / relative_path
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            target.write_bytes(
                _read_stable_regular_file(
                    source,
                    label=f"capture source {relative_path.as_posix()}",
                )
            )
            os.chmod(target, 0o400)
        source_hash = _graph_sha256(
            source_paths,
            repository_root=source_snapshot,
        )
        if (
            source_hash
            != authority.expected_hashes["source_snapshot_sha256"]
        ):
            raise TraceCaptureError(
                "private source snapshot differs from authority"
            )
        snapshot = PrivateCaptureSnapshot(
            input_dir=input_snapshot,
            source_root=source_snapshot,
            artifact_root=source_snapshot / "mib_pipeline" / "artifacts",
            input_tree_sha256=input_hash,
            source_snapshot_sha256=source_hash,
        )
        try:
            yield snapshot
        finally:
            post_archive_binding = verify_dataset_archive_authority(
                authority.approved_paths["dataset_archive"],
                expected_record_count=authority.expected_record_count,
                expected_input_tree_sha256=input_hash,
            )
            if (
                compute_input_tree_sha256(input_snapshot) != input_hash
                or _graph_sha256(
                    source_paths,
                    repository_root=source_snapshot,
                )
                != source_hash
                or post_archive_binding
                != archive_snapshot_binding
            ):
                raise TraceCaptureError(
                    "private processing snapshot changed during capture"
                )


def _make_authority_gate() -> tuple[
    Callable[[object | None], bool],
    Callable[[Callable[..., TraceCaptureResult]], Callable[..., TraceCaptureResult]],
]:
    capability = object()

    def verify(candidate: object | None) -> bool:
        return candidate is capability

    def bind(
        callback: Callable[..., TraceCaptureResult],
    ) -> Callable[..., TraceCaptureResult]:
        def gated(**kwargs: Any) -> TraceCaptureResult:
            return callback(_authority_capability=capability, **kwargs)

        return gated

    return verify, bind


_VERIFY_AUTHORITATIVE_CAPABILITY, _BIND_AUTHORITATIVE_ENTRYPOINT = (
    _make_authority_gate()
)


def _run_trace_capture_core_impl(
    authority_verifier: Callable[[object | None], bool],
    *,
    input_dir: Path,
    predictions_output: Path,
    trace_output: Path,
    layout_manifest_path: Path,
    binding: Mapping[str, Any],
    processor: Any,
    max_workers: int = 4,
    retry_missing_attempts: int = 1,
    stability_probe: Callable[[], Mapping[str, Any]] | None = None,
    protected_input_paths: Sequence[Path] = (),
    _authority_capability: object | None = None,
) -> TraceCaptureResult:
    """Run one capture; only the closure-gated path can mark it authoritative."""

    input_dir = Path(input_dir)
    predictions_output = Path(predictions_output)
    trace_output = Path(trace_output)
    layout_manifest_path = Path(layout_manifest_path)
    if not input_dir.is_dir():
        raise TraceCaptureError(f"input directory does not exist: {input_dir}")
    if not predictions_output.parent.is_dir():
        raise TraceCaptureError(
            f"prediction output directory does not exist: "
            f"{predictions_output.parent}"
        )
    if not trace_output.parent.is_dir():
        raise TraceCaptureError(
            f"trace output directory does not exist: {trace_output.parent}"
        )
    if not layout_manifest_path.is_file():
        raise TraceCaptureError(
            f"layout manifest does not exist: {layout_manifest_path}"
        )
    _require_external_output_path(
        predictions_output,
        label="raw prediction output",
    )
    _require_external_output_path(
        trace_output,
        label="per-case trace output",
    )
    resolved_outputs = {
        predictions_output.resolve(),
        trace_output.resolve(),
        layout_manifest_path.resolve(),
        *(Path(path).resolve() for path in protected_input_paths),
    }
    if len(resolved_outputs) != 3 + len(protected_input_paths):
        raise TraceCaptureError(
            "outputs and authority inputs must all use distinct paths"
        )
    for label, output_path in (
        ("prediction output", predictions_output),
        ("trace output", trace_output),
    ):
        try:
            output_path.resolve().relative_to(input_dir.resolve())
        except ValueError:
            continue
        raise TraceCaptureError(f"{label} must not be inside the input tree")
    if trace_output.exists():
        raise TraceCaptureError(
            "trace output already exists; choose a fresh external path"
        )
    observed_input_hash = compute_input_tree_sha256(input_dir)
    if observed_input_hash != binding["input_tree_sha256"]:
        raise TraceCaptureError(
            "input tree does not match the expected SHA-256"
        )
    authoritative = authority_verifier(_authority_capability)
    if authoritative:
        if stability_probe is None:
            raise TraceCaptureError(
                "authoritative capture requires a live stability probe"
            )
        expected_workers = binding.get("max_workers")
        if max_workers != expected_workers:
            raise TraceCaptureError(
                "max_workers differs from the preregistered runtime contract"
            )
        if retry_missing_attempts != binding.get(
            "retry_missing_attempts"
        ):
            raise TraceCaptureError(
                "retry policy differs from the preregistered authority"
            )

    collector = TraceCollector()
    observed = TracingCaseProcessor(
        processor,
        collector,
    )
    started = time.perf_counter()
    report, retry_passes_used = _run_with_missing_retries(
        processor=observed,
        input_dir=input_dir,
        predictions_output=predictions_output,
        max_workers=max_workers,
        retry_missing_attempts=retry_missing_attempts,
    )
    wall_seconds = time.perf_counter() - started
    if stability_probe is None:
        stability_checks = {
            check_name: False
            for check_name in _STABILITY_BINDING_KEYS.values()
        }
        if compute_input_tree_sha256(input_dir) != observed_input_hash:
            raise TraceCaptureError(
                "input tree changed during capture; trace was not finalized"
            )
    else:
        post_binding = stability_probe()
        stability_checks = {
            check_name: post_binding.get(key) == binding.get(key)
            for key, check_name in _STABILITY_BINDING_KEYS.items()
        }
        if not all(stability_checks.values()):
            failed = sorted(
                key for key, passed in stability_checks.items() if not passed
            )
            raise TraceCaptureError(
                "capture authority changed during execution: "
                + ", ".join(failed)
            )
    predictions_hash = sha256_path(predictions_output)
    if predictions_hash != binding["baseline_predictions_sha256"]:
        raise TraceCaptureError(
            "prediction-byte parity failed: "
            f"expected {binding['baseline_predictions_sha256']}, "
            f"observed {predictions_hash}; "
            "trace was not finalized"
        )
    expected_record_count = binding.get("expected_record_count")
    if (
        expected_record_count is not None
        and report.attempted != expected_record_count
    ):
        raise TraceCaptureError(
            "attempted case count changed from the frozen authority"
        )

    verified_flags = {
        "production_tree_verified": bool(
            authoritative and binding.get("production_tree_verified")
        ),
        "runtime_contract_verified": bool(
            authoritative and binding.get("runtime_contract_verified")
        ),
        "runtime_environment_verified": bool(
            authoritative and binding.get("runtime_environment_verified")
        ),
        "runtime_interface_verified": bool(
            authoritative and binding.get("runtime_interface_verified")
        ),
        "container_limits_verified": bool(
            authoritative and binding.get("container_limits_verified")
        ),
        "processing_snapshot_verified": bool(
            authoritative and binding.get("processing_snapshot_verified")
        ),
    }
    payload = validate_trace_capture(
        {
            "schema_version": TRACE_SCHEMA_VERSION,
            "capture_mode": (
                "authoritative_production" if authoritative else "test"
            ),
            "source_revision_sha": binding["source_revision_sha"],
            "checkout_revision_sha": binding["checkout_revision_sha"],
            "capture_source_revision_sha": binding.get(
                "capture_source_revision_sha",
                binding["checkout_revision_sha"],
            ),
            "input_tree_sha256": observed_input_hash,
            "processing_snapshot_input_tree_sha256": binding.get(
                "processing_snapshot_input_tree_sha256",
                observed_input_hash,
            ),
            "layout_manifest_sha256": binding[
                "layout_manifest_sha256"
            ],
            "dataset_archive_sha256": binding[
                "dataset_archive_sha256"
            ],
            "runtime_contract_sha256": binding[
                "runtime_contract_sha256"
            ],
            "frozen_baseline_manifest_sha256": binding[
                "frozen_baseline_manifest_sha256"
            ],
            "baseline_predictions_sha256": binding[
                "baseline_predictions_sha256"
            ],
            "runtime_graph_sha256": binding["runtime_graph_sha256"],
            "trace_tool_sha256": binding["trace_tool_sha256"],
            "container_graph_sha256": binding.get(
                "container_graph_sha256",
                "0" * 64,
            ),
            "source_snapshot_sha256": binding.get(
                "source_snapshot_sha256",
                "0" * 64,
            ),
            "authority_manifest_sha256": binding.get(
                "authority_manifest_sha256",
                "0" * 64,
            ),
            "runtime_identity_sha256": binding.get(
                "runtime_identity_sha256",
                "0" * 64,
            ),
            "dependency_identity_sha256": binding.get(
                "dependency_identity_sha256",
                "0" * 64,
            ),
            "python_executable_sha256": binding.get(
                "python_executable_sha256",
                "0" * 64,
            ),
            "predictions_sha256": predictions_hash,
            **verified_flags,
            "stability_checks": stability_checks,
            "case_count": report.answered,
            "attempted": report.attempted,
            "answered": report.answered,
            "omitted": report.omitted,
            "max_workers": max_workers,
            "retry_missing_attempts": retry_missing_attempts,
            "retry_passes_used": retry_passes_used,
            "batch_wall_seconds": wall_seconds,
            "rows": collector.rows(),
        }
    )
    atomic_write_trace(trace_output, payload)
    return TraceCaptureResult(
        report=report,
        predictions_sha256=predictions_hash,
        trace_sha256=sha256_path(trace_output),
        batch_wall_seconds=wall_seconds,
        retry_passes_used=retry_passes_used,
    )


def _bind_trace_core(
    implementation: Callable[..., TraceCaptureResult],
    verifier: Callable[[object | None], bool],
) -> Callable[..., TraceCaptureResult]:
    def bound(**kwargs: Any) -> TraceCaptureResult:
        return implementation(verifier, **kwargs)

    return bound


_run_trace_capture_core = _bind_trace_core(
    _run_trace_capture_core_impl,
    _VERIFY_AUTHORITATIVE_CAPABILITY,
)
del _bind_trace_core
del _run_trace_capture_core_impl
del _VERIFY_AUTHORITATIVE_CAPABILITY


def run_non_authoritative_test_capture(
    *,
    input_dir: Path,
    predictions_output: Path,
    trace_output: Path,
    source_revision_sha: str,
    input_tree_sha256: str,
    dataset_archive_sha256: str,
    layout_manifest_path: Path,
    expected_predictions_sha256: str,
    processor: Any,
    max_workers: int = 4,
    retry_missing_attempts: int = 1,
    runtime_sha256: str = "0" * 64,
    trace_tool_sha256: str = "0" * 64,
) -> TraceCaptureResult:
    """Explicit test-only injection path that can never be authoritative."""

    revision = str(source_revision_sha).casefold()
    if not _REVISION_RE.fullmatch(revision):
        raise TraceCaptureError("test source revision is invalid")
    binding = {
        "capture_mode": "test",
        "source_revision_sha": revision,
        "checkout_revision_sha": revision,
        "input_tree_sha256": _require_digest(
            input_tree_sha256,
            label="input_tree_sha256",
        ),
        "layout_manifest_sha256": sha256_path(layout_manifest_path),
        "dataset_archive_sha256": _require_digest(
            dataset_archive_sha256,
            label="dataset_archive_sha256",
        ),
        "runtime_contract_sha256": "0" * 64,
        "frozen_baseline_manifest_sha256": "0" * 64,
        "baseline_predictions_sha256": _require_digest(
            expected_predictions_sha256,
            label="expected_predictions_sha256",
        ),
        "runtime_graph_sha256": _require_digest(
            runtime_sha256,
            label="runtime_graph_sha256",
        ),
        "trace_tool_sha256": _require_digest(
            trace_tool_sha256,
            label="trace_tool_sha256",
        ),
        "production_tree_verified": False,
    }
    return _run_trace_capture_core(
        input_dir=input_dir,
        predictions_output=predictions_output,
        trace_output=trace_output,
        layout_manifest_path=layout_manifest_path,
        binding=binding,
        processor=processor,
        max_workers=max_workers,
        retry_missing_attempts=retry_missing_attempts,
    )


def _binding_from_capture_authority(
    authority: CaptureAuthority,
    snapshot: PrivateCaptureSnapshot,
) -> dict[str, Any]:
    baseline = load_frozen_baseline_authority(
        authority.approved_paths["frozen_baseline_manifest"],
        authority.approved_paths["baseline_predictions"],
    )
    return {
        "source_revision_sha": baseline["source_revision_sha"],
        "checkout_revision_sha": authority.capture_source_revision_sha,
        "capture_source_revision_sha": (
            authority.capture_source_revision_sha
        ),
        "input_tree_sha256": authority.expected_hashes[
            "input_tree_sha256"
        ],
        "processing_snapshot_input_tree_sha256": (
            snapshot.input_tree_sha256
        ),
        "layout_manifest_sha256": authority.expected_hashes[
            "layout_manifest_sha256"
        ],
        "dataset_archive_sha256": authority.expected_hashes[
            "dataset_archive_sha256"
        ],
        "runtime_contract_sha256": authority.expected_hashes[
            "runtime_contract_sha256"
        ],
        "frozen_baseline_manifest_sha256": authority.expected_hashes[
            "frozen_baseline_manifest_sha256"
        ],
        "baseline_predictions_sha256": authority.expected_hashes[
            "baseline_predictions_sha256"
        ],
        "runtime_graph_sha256": authority.expected_hashes[
            "runtime_graph_sha256"
        ],
        "trace_tool_sha256": authority.expected_hashes[
            "trace_tool_sha256"
        ],
        "container_graph_sha256": authority.expected_hashes[
            "container_graph_sha256"
        ],
        "source_snapshot_sha256": snapshot.source_snapshot_sha256,
        "authority_manifest_sha256": authority.manifest_sha256,
        "runtime_identity_sha256": authority.runtime_identity[
            "runtime_identity_sha256"
        ],
        "dependency_identity_sha256": authority.runtime_identity[
            "dependency_identity_sha256"
        ],
        "python_executable_sha256": authority.runtime_identity[
            "python_executable_sha256"
        ],
        "production_tree_verified": True,
        "runtime_contract_verified": True,
        "runtime_environment_verified": True,
        "runtime_interface_verified": True,
        "container_limits_verified": True,
        "processing_snapshot_verified": bool(
            snapshot.input_tree_sha256
            == authority.expected_hashes["input_tree_sha256"]
            and snapshot.source_snapshot_sha256
            == authority.expected_hashes["source_snapshot_sha256"]
        ),
        "expected_record_count": authority.expected_record_count,
        "max_workers": authority.max_workers,
        "retry_missing_attempts": authority.retry_missing_attempts,
    }


def _make_authoritative_entrypoint(
    builder: Callable[..., Any],
) -> Callable[..., TraceCaptureResult]:
    def implementation(
        *,
        _authority_capability: object,
        authority_manifest_path: Path,
        predictions_output: Path,
        trace_output: Path,
        max_workers: int | None = None,
        retry_missing_attempts: int | None = None,
    ) -> TraceCaptureResult:
        authority = load_capture_authority(authority_manifest_path)
        actual_workers = (
            authority.max_workers
            if max_workers is None
            else max_workers
        )
        actual_retries = (
            authority.retry_missing_attempts
            if retry_missing_attempts is None
            else retry_missing_attempts
        )
        if actual_workers != authority.max_workers:
            raise TraceCaptureError(
                "requested workers differ from preregistered authority"
            )
        if actual_retries != authority.retry_missing_attempts:
            raise TraceCaptureError(
                "requested retry policy differs from preregistered authority"
            )
        origin_input = authority.approved_paths["input_dir"]
        for label, output in (
            ("prediction output", Path(predictions_output)),
            ("trace output", Path(trace_output)),
        ):
            try:
                output.resolve().relative_to(origin_input)
            except ValueError:
                continue
            raise TraceCaptureError(
                f"{label} must not be inside the authority input tree"
            )
        trace_path = Path(trace_output)
        try:
            with private_capture_snapshot(authority) as snapshot:
                binding = _binding_from_capture_authority(
                    authority,
                    snapshot,
                )

                def stability_probe() -> Mapping[str, Any]:
                    post_authority = load_capture_authority(
                        authority_manifest_path
                    )
                    return _binding_from_capture_authority(
                        post_authority,
                        snapshot,
                    )

                protected = tuple(
                    path
                    for name, path in authority.approved_paths.items()
                    if name not in {"input_dir", "layout_manifest"}
                ) + (authority.manifest_path,)
                return _run_trace_capture_core(
                    input_dir=snapshot.input_dir,
                    predictions_output=predictions_output,
                    trace_output=trace_output,
                    layout_manifest_path=authority.approved_paths[
                        "layout_manifest"
                    ],
                    binding=binding,
                    processor=builder(
                        artifact_root=snapshot.artifact_root
                    ),
                    max_workers=actual_workers,
                    retry_missing_attempts=actual_retries,
                    stability_probe=stability_probe,
                    protected_input_paths=protected,
                    _authority_capability=_authority_capability,
                )
        except BaseException:
            try:
                trace_path.unlink()
            except FileNotFoundError:
                pass
            raise

    return _BIND_AUTHORITATIVE_ENTRYPOINT(implementation)


run_authoritative_trace_capture = _make_authoritative_entrypoint(
    build_observed_production_processor
)
del _BIND_AUTHORITATIVE_ENTRYPOINT
del _make_authoritative_entrypoint


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture external WO13 dimensions from the exact production "
            "runtime with a prediction-byte parity gate."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-authority",
        help="preregister exact committed source, paths, hashes, and runtime",
    )
    prepare.add_argument("--authority-output", required=True)
    prepare.add_argument("--input-dir", required=True)
    prepare.add_argument("--layout-manifest", required=True)
    prepare.add_argument("--dataset-archive", required=True)
    prepare.add_argument("--runtime-contract", required=True)
    prepare.add_argument("--frozen-baseline-manifest", required=True)
    prepare.add_argument("--baseline-predictions", required=True)
    prepare.add_argument(
        "--retry-missing-attempts",
        type=int,
        default=1,
        help="bounded retries for omitted PDFs only (0-3)",
    )
    capture = commands.add_parser(
        "capture",
        help="run only from a canonical preregistered authority manifest",
    )
    capture.add_argument("--authority-manifest", required=True)
    capture.add_argument("--predictions-output", required=True)
    capture.add_argument("--trace-output", required=True)
    capture.add_argument("--max-workers", type=int)
    capture.add_argument("--retry-missing-attempts", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare-authority":
            payload = prepare_capture_authority_manifest(
                output_path=Path(args.authority_output),
                input_dir=Path(args.input_dir),
                layout_manifest_path=Path(args.layout_manifest),
                dataset_archive_path=Path(args.dataset_archive),
                runtime_contract_path=Path(args.runtime_contract),
                frozen_baseline_manifest_path=Path(
                    args.frozen_baseline_manifest
                ),
                baseline_predictions_path=Path(
                    args.baseline_predictions
                ),
                retry_missing_attempts=args.retry_missing_attempts,
            )
            print(
                "authority_manifest_sha256="
                f"{sha256_path(Path(args.authority_output))} "
                f"source_revision_sha={payload['source_revision_sha']}",
                file=sys.stderr,
            )
            return 0
        result = run_authoritative_trace_capture(
            authority_manifest_path=Path(args.authority_manifest),
            predictions_output=Path(args.predictions_output),
            trace_output=Path(args.trace_output),
            max_workers=args.max_workers,
            retry_missing_attempts=args.retry_missing_attempts,
        )
    except (OSError, TraceCaptureError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 64
    print(
        f"attempted={result.report.attempted} "
        f"answered={result.report.answered} "
        f"omitted={result.report.omitted} "
        f"predictions_sha256={result.predictions_sha256} "
        f"trace_sha256={result.trace_sha256}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
