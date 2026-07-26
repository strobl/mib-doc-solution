"""Immutable, evidence-preserving audit records for visible-value recovery.

This module is deliberately an overlay on top of resolution.  It records what
the primary resolver knew, what the schema-safe row serialized, and whether a
later visible OCR observation recovered a value.  It does not mutate
``ResolvedCase`` or combine evidence; applicant linking and evidence fusion
remain separate policy boundaries.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Union

from .extraction import CandidateEvidence, EvidenceType
from .ingestion import Rect
from .models import PredictionRow
from .provenance import OcrProvenance
from .resolution import FieldState, ResolvedField


SerializedValue = Union[str, float]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_CHARACTER_RE = re.compile(r"[\x00-\x1f\x7f]")
_DEFAULT_FORBIDDEN_CUES = (
    "sample_denial_watermark",
    "strikethrough",
)


def _stable_text(value: str, name: str) -> str:
    normalized = value.strip()
    if not normalized or _CONTROL_CHARACTER_RE.search(normalized):
        raise ValueError(f"{name} must be a stable non-empty string")
    return normalized


def _optional_stable_text(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    return _stable_text(value, name)


def _scope_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _rect_key(box: Rect) -> tuple[float, float, float, float]:
    return tuple(
        round(float(value), 6)
        for value in (box.left, box.bottom, box.right, box.top)
    )


def _serialized_value(value: SerializedValue, name: str) -> SerializedValue:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a string or finite float")
    if isinstance(value, str):
        return value
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def observed_applicant_scopes(
    candidate: CandidateEvidence | None,
) -> tuple[str, ...]:
    """Return every distinct applicant scope carried by visible evidence.

    The candidate hint and every physical OCR observation are retained.  A
    disagreement is intentionally not collapsed to one "winner"; validation
    can therefore reject ambiguous scope rather than hiding it.
    """

    if candidate is None:
        return ()
    by_key: dict[str, str] = {}
    values = (
        (candidate.applicant_hint,)
        + tuple(
            item.observation.applicant_scope
            for item in candidate.ocr_provenance
        )
    )
    for value in values:
        if value is None:
            continue
        normalized = _stable_text(value, "observed applicant scope")
        by_key.setdefault(_scope_key(normalized), normalized)
    return tuple(by_key[key] for key in sorted(by_key))


class SerializationOrigin(str, Enum):
    """Why a schema-safe serialized value exists."""

    PRIMARY_VISIBLE_EVIDENCE = "primary_visible_evidence"
    PRIMARY_RESOLVED_VALUE = "primary_resolved_value"
    RECOVERED_VISIBLE_EVIDENCE = "recovered_visible_evidence"
    OUTPUT_DEFAULT = "output_default"


class CandidateValidationFailure(str, Enum):
    """Stable, machine-readable reasons why recovery must fail closed."""

    MISSING_CANDIDATE = "missing_candidate"
    FIELD_MISMATCH = "field_mismatch"
    MISSING_VALUE = "missing_value"
    ILLEGIBLE = "illegible"
    SUPERSEDED = "superseded"
    FORBIDDEN_VISUAL_CUE = "forbidden_visual_cue"
    LOW_CONFIDENCE = "low_confidence"
    MISSING_PROVENANCE = "missing_provenance"
    INCOMPLETE_PROVENANCE = "incomplete_provenance"
    SOURCE_MISMATCH = "source_mismatch"
    PAGE_MISMATCH = "page_mismatch"
    APPLICANT_SCOPE_MISSING = "applicant_scope_missing"
    APPLICANT_SCOPE_MISMATCH = "applicant_scope_mismatch"
    CANDIDATE_BOX_NOT_PROVENANCED = "candidate_box_not_provenanced"
    NON_OCR_SOURCE = "non_ocr_source"
    NON_OCR_EVIDENCE_TYPE = "non_ocr_evidence_type"
    PRIMARY_STATE_NOT_UNKNOWN = "primary_state_not_unknown"
    SERIALIZATION_VALUE_MISMATCH = "serialization_value_mismatch"
    RECOVERY_ROUTE_MISMATCH = "recovery_route_mismatch"


@dataclass(frozen=True)
class CandidateValidationPolicy:
    """Expected physical scope for one high-precision recovery candidate."""

    expected_field_name: str
    expected_source_sha256: str
    expected_page_index: int
    expected_applicant_scope: str | None
    minimum_confidence: float
    forbidden_visual_cues: tuple[str, ...] = _DEFAULT_FORBIDDEN_CUES
    require_candidate_box_observation: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_field_name",
            _stable_text(self.expected_field_name, "expected_field_name"),
        )
        digest = self.expected_source_sha256.strip().casefold()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(
                "expected_source_sha256 must be a 64-character hex digest"
            )
        object.__setattr__(self, "expected_source_sha256", digest)
        if self.expected_page_index < 0:
            raise ValueError("expected_page_index must be non-negative")
        object.__setattr__(
            self,
            "expected_applicant_scope",
            _optional_stable_text(
                self.expected_applicant_scope,
                "expected_applicant_scope",
            ),
        )
        confidence = float(self.minimum_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("minimum_confidence must be between 0 and 1")
        object.__setattr__(self, "minimum_confidence", confidence)
        cues = tuple(
            sorted(
                {
                    _stable_text(cue, "forbidden_visual_cue")
                    for cue in self.forbidden_visual_cues
                }
            )
        )
        object.__setattr__(self, "forbidden_visual_cues", cues)


@dataclass(frozen=True)
class CandidateValidationResult:
    """A deterministic validation decision that never treats doubt as valid."""

    failures: tuple[CandidateValidationFailure, ...]
    observed_applicant_scopes: tuple[str, ...]
    provenance_fingerprints: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.failures


def validate_recovered_candidate(
    candidate: CandidateEvidence | None,
    policy: CandidateValidationPolicy,
) -> CandidateValidationResult:
    """Validate a recovered candidate against immutable physical expectations.

    Any absent or conflicting field, source, page, applicant scope, confidence,
    or OCR provenance makes the result invalid.  A caller can therefore use
    ``accepted`` directly as a fail-closed gate.
    """

    if candidate is None:
        return CandidateValidationResult(
            failures=(CandidateValidationFailure.MISSING_CANDIDATE,),
            observed_applicant_scopes=(),
            provenance_fingerprints=(),
        )

    failures: set[CandidateValidationFailure] = set()
    if candidate.field_name != policy.expected_field_name:
        failures.add(CandidateValidationFailure.FIELD_MISMATCH)
    if candidate.source != "visible_ocr":
        failures.add(CandidateValidationFailure.NON_OCR_SOURCE)
    if candidate.evidence_type is EvidenceType.TEXT_LAYER:
        failures.add(CandidateValidationFailure.NON_OCR_EVIDENCE_TYPE)
    if not isinstance(candidate.value, str) or not candidate.value.strip():
        failures.add(CandidateValidationFailure.MISSING_VALUE)
    if not candidate.legible:
        failures.add(CandidateValidationFailure.ILLEGIBLE)
    if candidate.superseded:
        failures.add(CandidateValidationFailure.SUPERSEDED)
    if set(candidate.visual_cues).intersection(policy.forbidden_visual_cues):
        failures.add(CandidateValidationFailure.FORBIDDEN_VISUAL_CUE)
    confidence = candidate.ocr_confidence
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or float(confidence) < policy.minimum_confidence
    ):
        failures.add(CandidateValidationFailure.LOW_CONFIDENCE)

    provenance = candidate.ocr_provenance
    scopes = observed_applicant_scopes(candidate)
    fingerprints: tuple[str, ...] = ()
    if not provenance:
        failures.add(CandidateValidationFailure.MISSING_PROVENANCE)
    else:
        if not all(isinstance(item, OcrProvenance) for item in provenance):
            failures.add(CandidateValidationFailure.INCOMPLETE_PROVENANCE)
        else:
            fingerprints = tuple(item.fingerprint for item in provenance)
            if any(
                item.observation.source_sha256
                != policy.expected_source_sha256
                for item in provenance
            ):
                failures.add(CandidateValidationFailure.SOURCE_MISMATCH)
            if (
                candidate.page_index != policy.expected_page_index
                or any(
                    item.observation.page_index
                    != policy.expected_page_index
                    for item in provenance
                )
            ):
                failures.add(CandidateValidationFailure.PAGE_MISMATCH)
            if (
                policy.require_candidate_box_observation
                and not any(
                    _rect_key(item.view_box) == _rect_key(candidate.box)
                    or _rect_key(item.observation.box)
                    == _rect_key(candidate.box)
                    for item in provenance
                )
            ):
                failures.add(
                    CandidateValidationFailure.CANDIDATE_BOX_NOT_PROVENANCED
                )

    expected_scope = policy.expected_applicant_scope
    if expected_scope is None:
        if scopes:
            failures.add(CandidateValidationFailure.APPLICANT_SCOPE_MISMATCH)
        if any(
            item.observation.applicant_scope is not None
            for item in provenance
        ):
            failures.add(CandidateValidationFailure.APPLICANT_SCOPE_MISMATCH)
    else:
        expected_key = _scope_key(expected_scope)
        if not scopes:
            failures.add(CandidateValidationFailure.APPLICANT_SCOPE_MISSING)
        elif any(_scope_key(scope) != expected_key for scope in scopes):
            failures.add(CandidateValidationFailure.APPLICANT_SCOPE_MISMATCH)
        if any(
            item.observation.applicant_scope is None
            or _scope_key(item.observation.applicant_scope) != expected_key
            for item in provenance
        ):
            failures.add(CandidateValidationFailure.APPLICANT_SCOPE_MISMATCH)

    return CandidateValidationResult(
        failures=tuple(sorted(failures, key=lambda item: item.value)),
        observed_applicant_scopes=scopes,
        provenance_fingerprints=fingerprints,
    )


def _primary_origin(primary: ResolvedField) -> SerializationOrigin:
    if primary.state is not FieldState.RESOLVED:
        return SerializationOrigin.OUTPUT_DEFAULT
    winner = primary.winning_evidence
    if (
        winner is not None
        and winner.value == primary.value
        and winner.source == "visible_ocr"
        and winner.evidence_type is not EvidenceType.TEXT_LAYER
        and winner.legible
        and not winner.superseded
        and bool(winner.ocr_provenance)
    ):
        return SerializationOrigin.PRIMARY_VISIBLE_EVIDENCE
    return SerializationOrigin.PRIMARY_RESOLVED_VALUE


@dataclass(frozen=True)
class RecoveryFieldAudit:
    """Immutable before/after record for one output field."""

    field_name: str
    primary_state: FieldState
    primary_evidence_value: str | None
    serialization_before: SerializedValue
    serialization_before_origin: SerializationOrigin
    final_evidence_state: FieldState
    final_evidence_value: str | None
    serialization_after: SerializedValue
    serialization_after_origin: SerializationOrigin
    recovery_source: str | None
    primary_winning_evidence: CandidateEvidence | None
    winning_evidence: CandidateEvidence | None
    observed_applicant_scopes: tuple[str, ...]
    linked_recovery_scope: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "field_name",
            _stable_text(self.field_name, "field_name"),
        )
        object.__setattr__(
            self,
            "serialization_before",
            _serialized_value(
                self.serialization_before,
                "serialization_before",
            ),
        )
        object.__setattr__(
            self,
            "serialization_after",
            _serialized_value(
                self.serialization_after,
                "serialization_after",
            ),
        )
        object.__setattr__(
            self,
            "recovery_source",
            _optional_stable_text(self.recovery_source, "recovery_source"),
        )
        object.__setattr__(
            self,
            "linked_recovery_scope",
            _optional_stable_text(
                self.linked_recovery_scope,
                "linked_recovery_scope",
            ),
        )
        normalized_scopes = tuple(
            sorted(
                {
                    _stable_text(scope, "observed_applicant_scope")
                    for scope in self.observed_applicant_scopes
                },
                key=_scope_key,
            )
        )
        object.__setattr__(
            self,
            "observed_applicant_scopes",
            normalized_scopes,
        )

        if (
            self.primary_state is FieldState.RESOLVED
        ) != (self.primary_evidence_value is not None):
            raise ValueError(
                "primary evidence value must exist exactly when primary state "
                "is resolved"
            )
        if (
            self.final_evidence_state is FieldState.RESOLVED
        ) != (self.final_evidence_value is not None):
            raise ValueError(
                "final evidence value must exist exactly when final state is "
                "resolved"
            )
        if (
            self.primary_winning_evidence is not None
            and self.primary_winning_evidence.field_name != self.field_name
        ):
            raise ValueError("primary winning evidence field does not match")
        if (
            self.winning_evidence is not None
            and self.winning_evidence.field_name != self.field_name
        ):
            raise ValueError("winning evidence field does not match")

        if self.recovery_source is None:
            if (
                self.final_evidence_state is not self.primary_state
                or self.final_evidence_value != self.primary_evidence_value
            ):
                raise ValueError(
                    "changed evidence requires an explicit recovery_source"
                )
            if (
                self.serialization_after_origin
                is SerializationOrigin.RECOVERED_VISIBLE_EVIDENCE
            ):
                raise ValueError(
                    "recovered origin requires an explicit recovery_source"
                )
        else:
            if self.final_evidence_state is not FieldState.RESOLVED:
                raise ValueError("recovery must produce resolved evidence")
            if self.winning_evidence is None:
                raise ValueError("recovery must retain its winning evidence")
            if self.winning_evidence.value != self.final_evidence_value:
                raise ValueError(
                    "winning evidence value does not match final evidence"
                )
            if (
                self.serialization_after_origin
                is not SerializationOrigin.RECOVERED_VISIBLE_EVIDENCE
            ):
                raise ValueError(
                    "recovery must be marked recovered visible evidence"
                )
            if not self.winning_evidence.ocr_provenance:
                raise ValueError("recovery must retain complete OCR provenance")

        if (
            self.serialization_before_origin
            is SerializationOrigin.OUTPUT_DEFAULT
            and self.primary_state is FieldState.RESOLVED
        ):
            raise ValueError(
                "resolved primary evidence cannot be marked as output default"
            )
        if (
            self.serialization_after_origin
            is SerializationOrigin.OUTPUT_DEFAULT
            and self.final_evidence_state is FieldState.RESOLVED
        ):
            raise ValueError(
                "resolved final evidence cannot be marked as output default"
            )

    @property
    def ocr_provenance(self) -> tuple[OcrProvenance, ...]:
        if self.winning_evidence is None:
            return ()
        return self.winning_evidence.ocr_provenance

    @property
    def before_is_explicit_visible_value(self) -> bool:
        return (
            self.serialization_before_origin
            is SerializationOrigin.PRIMARY_VISIBLE_EVIDENCE
        )

    @property
    def after_is_explicit_visible_value(self) -> bool:
        return self.serialization_after_origin in {
            SerializationOrigin.PRIMARY_VISIBLE_EVIDENCE,
            SerializationOrigin.RECOVERED_VISIBLE_EVIDENCE,
        }

    @property
    def before_is_output_default(self) -> bool:
        return (
            self.serialization_before_origin
            is SerializationOrigin.OUTPUT_DEFAULT
        )

    @property
    def after_is_output_default(self) -> bool:
        return (
            self.serialization_after_origin
            is SerializationOrigin.OUTPUT_DEFAULT
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "primary_state": self.primary_state.value,
            "primary_evidence_value": self.primary_evidence_value,
            "serialization_before": self.serialization_before,
            "serialization_before_origin": (
                self.serialization_before_origin.value
            ),
            "final_evidence_state": self.final_evidence_state.value,
            "final_evidence_value": self.final_evidence_value,
            "serialization_after": self.serialization_after,
            "serialization_after_origin": (
                self.serialization_after_origin.value
            ),
            "recovery_source": self.recovery_source,
            "primary_winning_evidence": _candidate_to_dict(
                self.primary_winning_evidence
            ),
            "winning_evidence": _candidate_to_dict(self.winning_evidence),
            "observed_applicant_scopes": list(
                self.observed_applicant_scopes
            ),
            "linked_recovery_scope": self.linked_recovery_scope,
            "ocr_provenance": [
                item.to_dict() for item in self.ocr_provenance
            ],
        }


def unchanged_field_audit(
    *,
    primary: ResolvedField,
    serialized_value: SerializedValue,
    linked_recovery_scope: str | None,
) -> RecoveryFieldAudit:
    """Record a field for which no valid visible recovery was accepted."""

    origin = _primary_origin(primary)
    winner = primary.winning_evidence
    return RecoveryFieldAudit(
        field_name=primary.field_name,
        primary_state=primary.state,
        primary_evidence_value=primary.value,
        serialization_before=serialized_value,
        serialization_before_origin=origin,
        final_evidence_state=primary.state,
        final_evidence_value=primary.value,
        serialization_after=serialized_value,
        serialization_after_origin=origin,
        recovery_source=None,
        primary_winning_evidence=winner,
        winning_evidence=winner,
        observed_applicant_scopes=observed_applicant_scopes(winner),
        linked_recovery_scope=linked_recovery_scope,
    )


def recovered_field_audit(
    *,
    primary: ResolvedField,
    serialization_before: SerializedValue,
    serialization_after: SerializedValue,
    candidate: CandidateEvidence | None,
    recovery_source: str,
    linked_recovery_scope: str | None,
    validation_policy: CandidateValidationPolicy,
) -> tuple[RecoveryFieldAudit | None, CandidateValidationResult]:
    """Build an accepted recovery record, or return no record on any doubt."""

    return _visible_change_field_audit(
        primary=primary,
        serialization_before=serialization_before,
        serialization_after=serialization_after,
        candidate=candidate,
        recovery_source=recovery_source,
        linked_recovery_scope=linked_recovery_scope,
        validation_policy=validation_policy,
        require_primary_unknown=True,
    )


def visible_repair_field_audit(
    *,
    primary: ResolvedField,
    serialization_before: SerializedValue,
    serialization_after: SerializedValue,
    candidate: CandidateEvidence | None,
    recovery_source: str,
    linked_recovery_scope: str | None,
    validation_policy: CandidateValidationPolicy,
) -> tuple[RecoveryFieldAudit | None, CandidateValidationResult]:
    """Audit a validated visible repair of an already-serialized value.

    Unlike ``recovered_field_audit``, a biometric or source-priority repair may
    replace a primary ``RESOLVED`` value.  It is still bound to the same source,
    page, applicant scope, OCR route, exact serialized value, and complete
    provenance before it can be recorded as visible evidence.
    """

    return _visible_change_field_audit(
        primary=primary,
        serialization_before=serialization_before,
        serialization_after=serialization_after,
        candidate=candidate,
        recovery_source=recovery_source,
        linked_recovery_scope=linked_recovery_scope,
        validation_policy=validation_policy,
        require_primary_unknown=False,
    )


def _visible_change_field_audit(
    *,
    primary: ResolvedField,
    serialization_before: SerializedValue,
    serialization_after: SerializedValue,
    candidate: CandidateEvidence | None,
    recovery_source: str,
    linked_recovery_scope: str | None,
    validation_policy: CandidateValidationPolicy,
    require_primary_unknown: bool,
) -> tuple[RecoveryFieldAudit | None, CandidateValidationResult]:
    validation = validate_recovered_candidate(candidate, validation_policy)
    failures = set(validation.failures)
    if primary.field_name != validation_policy.expected_field_name:
        failures.add(CandidateValidationFailure.FIELD_MISMATCH)
    if (
        require_primary_unknown
        and primary.state is not FieldState.UNKNOWN
    ):
        failures.add(CandidateValidationFailure.PRIMARY_STATE_NOT_UNKNOWN)
    if (
        candidate is not None
        and serialization_after != candidate.value
    ):
        failures.add(
            CandidateValidationFailure.SERIALIZATION_VALUE_MISMATCH
        )
    normalized_recovery_source = _stable_text(
        recovery_source,
        "recovery_source",
    )
    if (
        candidate is not None
        and candidate.ocr_provenance
        and not any(
            item.route_id == normalized_recovery_source
            and (
                _rect_key(item.view_box) == _rect_key(candidate.box)
                or _rect_key(item.observation.box)
                == _rect_key(candidate.box)
            )
            for item in candidate.ocr_provenance
        )
    ):
        failures.add(CandidateValidationFailure.RECOVERY_ROUTE_MISMATCH)
    if failures != set(validation.failures):
        validation = CandidateValidationResult(
            failures=tuple(sorted(failures, key=lambda item: item.value)),
            observed_applicant_scopes=validation.observed_applicant_scopes,
            provenance_fingerprints=validation.provenance_fingerprints,
        )
    if candidate is None or not validation.accepted:
        return None, validation
    if (
        validation_policy.expected_applicant_scope is not None
        and (
            linked_recovery_scope is None
            or _scope_key(linked_recovery_scope)
            != _scope_key(validation_policy.expected_applicant_scope)
        )
    ):
        return (
            None,
            CandidateValidationResult(
                failures=(
                    CandidateValidationFailure.APPLICANT_SCOPE_MISMATCH,
                ),
                observed_applicant_scopes=validation.observed_applicant_scopes,
                provenance_fingerprints=validation.provenance_fingerprints,
            ),
        )

    return (
        RecoveryFieldAudit(
            field_name=primary.field_name,
            primary_state=primary.state,
            primary_evidence_value=primary.value,
            serialization_before=serialization_before,
            serialization_before_origin=_primary_origin(primary),
            final_evidence_state=FieldState.RESOLVED,
            final_evidence_value=candidate.value,
            serialization_after=serialization_after,
            serialization_after_origin=(
                SerializationOrigin.RECOVERED_VISIBLE_EVIDENCE
            ),
            recovery_source=normalized_recovery_source,
            primary_winning_evidence=primary.winning_evidence,
            winning_evidence=candidate,
            observed_applicant_scopes=validation.observed_applicant_scopes,
            linked_recovery_scope=linked_recovery_scope,
        ),
        validation,
    )


@dataclass(frozen=True)
class RecoveryAuditOverlay:
    """Read-only per-field recovery history for one case output row."""

    case_id: str
    fields: Mapping[str, RecoveryFieldAudit]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _stable_text(self.case_id, "case_id"))
        copied = dict(self.fields)
        if any(name != audit.field_name for name, audit in copied.items()):
            raise ValueError("audit mapping keys must match field_name")
        object.__setattr__(self, "fields", MappingProxyType(copied))

    @classmethod
    def from_fields(
        cls,
        *,
        case_id: str,
        fields: Iterable[RecoveryFieldAudit],
    ) -> "RecoveryAuditOverlay":
        indexed: dict[str, RecoveryFieldAudit] = {}
        for audit in fields:
            if audit.field_name in indexed:
                raise ValueError(
                    f"duplicate recovery audit field: {audit.field_name}"
                )
            indexed[audit.field_name] = audit
        return cls(case_id=case_id, fields=indexed)

    def field(self, field_name: str) -> RecoveryFieldAudit:
        return self.fields[field_name]

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "fields": {
                field_name: self.fields[field_name].to_dict()
                for field_name in sorted(self.fields)
            },
        }


@dataclass(frozen=True)
class VisibleRecoveryResult:
    """Schema-safe row plus its immutable, non-fusing recovery overlay."""

    row: PredictionRow
    audit: RecoveryAuditOverlay

    def __post_init__(self) -> None:
        if self.row.case_id != self.audit.case_id:
            raise ValueError("row and recovery audit case IDs must match")
        for field_name, field_audit in self.audit.fields.items():
            if not hasattr(self.row, field_name):
                raise ValueError(
                    f"recovery audit field is absent from row: {field_name}"
                )
            if (
                getattr(self.row, field_name)
                != field_audit.serialization_after
            ):
                raise ValueError(
                    "row value and recovery audit serialization_after "
                    f"must match for {field_name}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "row": self.row.to_dict(),
            "audit": self.audit.to_dict(),
        }


def _candidate_to_dict(
    candidate: CandidateEvidence | None,
) -> dict[str, object] | None:
    if candidate is None:
        return None
    return {
        "field_name": candidate.field_name,
        "value": candidate.value,
        "evidence_type": candidate.evidence_type.value,
        "page_index": candidate.page_index,
        "box": list(_rect_key(candidate.box)),
        "legible": candidate.legible,
        "superseded": candidate.superseded,
        "ocr_confidence": candidate.ocr_confidence,
        "visual_cues": list(candidate.visual_cues),
        "source": candidate.source,
        "case_id_hint": candidate.case_id_hint,
        "applicant_hint": candidate.applicant_hint,
        "ocr_provenance": [
            item.to_dict() for item in candidate.ocr_provenance
        ],
    }
