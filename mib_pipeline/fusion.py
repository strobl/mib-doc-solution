"""Deterministic, provenance-aware fusion of visible field evidence.

This module is deliberately independent of case linking and output
serialization.  It consumes already-extracted :class:`CandidateEvidence`
records, applies the binding evidence hierarchy supplied by the caller, and
returns an immutable decision plus an aggregateable safety trace.

OCR confidence is only a tie-breaker between equivalent visible observations.
It never changes authority rank and repeated OCR views of the same physical
pixels never become repeated votes.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterable, Protocol

from .extraction import CandidateEvidence, EvidenceType
from .models import (
    ADJUDICATION_VALUES,
    CASE_ID_PATTERN,
    FEE_VALUES,
    SPONSOR_ID_PATTERN,
)


class EvidenceRanker(Protocol):
    """Structural type accepted by :meth:`EvidenceFuser.resolve`."""

    def rank(self, evidence_type: EvidenceType) -> int:
        """Return the binding one-based authority rank."""


Ranker = EvidenceRanker | Callable[[EvidenceType], int]


@dataclass(frozen=True)
class FusionTrace:
    """Identity-free diagnostics for one field-level fusion decision."""

    winning_rank: int | None
    candidate_count: int
    eligible_candidate_count: int
    winning_rank_candidate_count: int
    observation_count: int
    independent_evidence_count: int
    independent_agreement_count: int
    correlated_candidate_count: int
    independent_evidence_type_count: int
    independent_page_count: int
    disagreement_count: int
    entropy_bits: float
    disagreement_ratio: float
    provenance_complete_count: int
    provenance_completeness: float
    value_counts: tuple[tuple[str, int], ...]
    veto_reasons: tuple[tuple[str, int], ...]
    safety_counters: tuple[tuple[str, int], ...]

    def veto_count(self, reason: str) -> int:
        """Return one deterministic veto count for audit aggregation."""

        return dict(self.veto_reasons).get(reason, 0)

    def safety_count(self, name: str) -> int:
        """Return one deterministic safety counter for audit aggregation."""

        return dict(self.safety_counters).get(name, 0)


@dataclass(frozen=True)
class FusionDecision:
    """Immutable result that ``EvidencePrecedenceResolver`` can delegate to."""

    field_name: str
    state: str
    value: str | None
    winner: CandidateEvidence | None
    considered: tuple[CandidateEvidence, ...]
    trace: FusionTrace
    reason: str

    def __post_init__(self) -> None:
        if self.state not in {"resolved", "unknown", "contested"}:
            raise ValueError("fusion state must be resolved, unknown, or contested")
        if self.state == "resolved":
            if self.value is None or self.winner is None:
                raise ValueError("resolved fusion decisions require a value and winner")
        elif self.value is not None or self.winner is not None:
            raise ValueError("non-resolved fusion decisions cannot expose a winner")

    @property
    def winning_evidence(self) -> CandidateEvidence | None:
        """Compatibility alias for the existing resolution model."""

        return self.winner


@dataclass(frozen=True)
class _NormalizedCandidate:
    candidate: CandidateEvidence
    value_key: str
    output_value: str
    rank: int
    observation_keys: tuple[str, ...]
    correlation_keys: tuple[str, ...]
    provenance_complete: bool
    correction: bool


_STRIKE_CUES = frozenset(
    {
        "crossed_out",
        "strike",
        "strikethrough",
        "struck",
        "struck_out",
        "struck_through",
    }
)
_CORRECTION_CUES = frozenset(
    {"amended", "correction", "corrected", "override", "replacement"}
)
_DEFAULT_SOURCES = frozenset(
    {
        "default",
        "output_default",
        "schema_default",
        "serialization_default",
        "synthetic_default",
    }
)
_PRODUCTION_VISIBLE_OCR_SOURCE = "visible_ocr"
_NEAR_OBSERVATION_OVERLAP = 0.90
_NEAR_OBSERVATION_AREA_RATIO = 0.80
_LOCAL_RECORD_CUE = re.compile(
    r"^(?:local_)?record(?:_id)?\s*[:=]\s*([A-Za-z0-9._-]+)$",
    re.IGNORECASE,
)
_DATE_FIELDS = frozenset({"arrival_date", "packet_receipt_date"})
_FREE_TEXT_FIELDS = frozenset({"declared_purpose", "diplomatic_note"})
_LOWERCASE_CATEGORICAL_FIELDS = frozenset(
    {
        "biohazard_check",
        "diplomatic_waiver_code",
        "fee_status",
        "hardship_waiver",
        "minimal_diplomatic_packet",
        "page_type_present_fee_receipt",
        "page_type_present_other",
        "page_type_present_sponsor_attestation",
        "work_permit_requested",
    }
)


def _compact(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def _cue_key(value: str) -> str:
    return re.sub(r"[\s-]+", "_", _compact(value).casefold())


def _name_key(value: str) -> str:
    """Conservative identity comparison: Unicode, whitespace, and case only."""

    return _compact(value).casefold()


def _rect_key(rect: object) -> tuple[float, float, float, float]:
    return tuple(
        round(float(value), 6)
        for value in (
            rect.left,  # type: ignore[attr-defined]
            rect.bottom,  # type: ignore[attr-defined]
            rect.right,  # type: ignore[attr-defined]
            rect.top,  # type: ignore[attr-defined]
        )
    )


def _bound_provenance(candidate: CandidateEvidence) -> tuple[object, ...]:
    """Return provenance witnesses that actually cover the candidate box."""

    candidate_box = _rect_key(candidate.box)
    return tuple(
        item
        for item in candidate.ocr_provenance
        if item.observation.page_index == candidate.page_index
        and (
            _rect_key(item.view_box) == candidate_box
            or _rect_key(item.observation.box) == candidate_box
        )
    )


def candidate_has_complete_provenance(
    candidate: CandidateEvidence,
    *,
    expected_case_id: str | None,
    active_applicant: str | None,
    active_applicant_aliases: Iterable[str] = (),
) -> bool:
    """Prove that a visible winner is bound to one packet and physical box."""

    if candidate.source != _PRODUCTION_VISIBLE_OCR_SOURCE:
        return False
    expected = (
        _compact(expected_case_id).upper()
        if isinstance(expected_case_id, str) and _compact(expected_case_id)
        else None
    )
    if expected is not None and (
        candidate.case_id_hint is None
        or _compact(candidate.case_id_hint).upper() != expected
    ):
        return False
    provenance = candidate.ocr_provenance
    bound = _bound_provenance(candidate)
    if not provenance or not bound:
        return False
    if len({item.observation.source_sha256 for item in provenance}) != 1:
        return False
    if any(
        item.observation.page_index != candidate.page_index
        for item in provenance
    ):
        return False

    valid_scopes = {
        _name_key(value)
        for value in (active_applicant, *active_applicant_aliases)
        if isinstance(value, str) and _compact(value)
    }
    observed_scopes = tuple(
        item.observation.applicant_scope for item in provenance
    )
    if valid_scopes:
        if any(
            scope is None or _name_key(scope) not in valid_scopes
            for scope in observed_scopes
        ):
            return False
    elif any(scope is not None for scope in observed_scopes):
        return False
    return True


def _normalize_risk_set(value: str) -> str | None:
    compact = _compact(value)
    if not compact:
        return None
    raw_items = re.split(r"[,;|]+", compact)
    items = {
        re.sub(r"[\s-]+", "_", item.strip().casefold())
        for item in raw_items
        if item.strip()
    }
    if not items:
        return None
    if items <= {"none", "clear", "no_flags"}:
        return "none"
    if items & {"none", "clear", "no_flags"}:
        # A single observation cannot simultaneously assert no risk and risk.
        return None
    return "|".join(sorted(items))


def _normalize_value(field_name: str, value: str) -> tuple[str, str] | None:
    """Return an exact comparison key and conservative canonical output.

    Extraction may repair OCR before this boundary.  Fusion itself performs no
    edit-distance, vocabulary, or semantic matching.
    """

    compact = _compact(value)
    if not compact:
        return None

    if field_name == "risk_flags":
        normalized = _normalize_risk_set(compact)
        return (normalized, normalized) if normalized is not None else None

    if field_name == "case_id":
        normalized = compact.upper()
        return (
            (normalized, normalized)
            if CASE_ID_PATTERN.fullmatch(normalized)
            else None
        )

    if field_name == "sponsor_id":
        normalized = compact.upper()
        return (
            (normalized, normalized)
            if SPONSOR_ID_PATTERN.fullmatch(normalized)
            else None
        )

    if field_name in _DATE_FIELDS:
        try:
            parsed = date.fromisoformat(compact)
        except ValueError:
            return None
        normalized = parsed.isoformat()
        return (normalized, normalized) if normalized == compact else None

    if field_name == "fee_status":
        normalized = compact.casefold()
        return (
            (normalized, normalized)
            if normalized in FEE_VALUES
            else None
        )

    if field_name == "adjudication":
        normalized = re.sub(r"[\s-]+", "_", compact.upper())
        return (
            (normalized, normalized)
            if normalized in ADJUDICATION_VALUES
            else None
        )

    if field_name == "visa_class":
        normalized = compact.upper()
        return (
            (normalized, normalized)
            if re.fullmatch(r"[A-Z][A-Z0-9]*-[0-9]+", normalized)
            else None
        )

    if field_name == "species_code":
        normalized = re.sub(r"\s+", "_", compact.upper())
        return (
            (normalized, normalized)
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", normalized)
            else None
        )

    if field_name == "applicant_name":
        # Do not remove punctuation, reorder tokens, or use edit distance.
        return _name_key(compact), compact

    if field_name in _FREE_TEXT_FIELDS:
        normalized = compact.casefold()
        return normalized, normalized

    if field_name in _LOWERCASE_CATEGORICAL_FIELDS:
        normalized = compact.casefold()
        return normalized, normalized

    # Opaque exact categories (for example home_world) agree across case and
    # whitespace differences only.  Preserve the winner's canonical spelling.
    return compact.casefold(), compact


def _candidate_sort_key(candidate: CandidateEvidence) -> tuple[object, ...]:
    confidence = (
        float(candidate.ocr_confidence)
        if math.isfinite(float(candidate.ocr_confidence))
        else -1.0
    )
    return (
        candidate.field_name,
        candidate.evidence_type.value,
        candidate.page_index,
        round(float(candidate.box.left), 6),
        round(float(candidate.box.bottom), 6),
        round(float(candidate.box.right), 6),
        round(float(candidate.box.top), 6),
        "" if candidate.value is None else candidate.value,
        candidate.case_id_hint or "",
        candidate.applicant_hint or "",
        candidate.source,
        tuple(sorted(candidate.visual_cues)),
        confidence,
        tuple(item.fingerprint for item in candidate.ocr_provenance),
    )


def _confidence(candidate: CandidateEvidence) -> float:
    value = float(candidate.ocr_confidence)
    return value if math.isfinite(value) else -1.0


def _rank(ranker: Ranker, evidence_type: EvidenceType) -> int:
    rank_method = getattr(ranker, "rank", None)
    raw_rank = (
        rank_method(evidence_type)
        if callable(rank_method)
        else ranker(evidence_type)  # type: ignore[operator]
    )
    if isinstance(raw_rank, bool):
        raise ValueError("evidence rank must be a positive integer")
    numeric = int(raw_rank)
    if numeric != raw_rank or numeric < 1:
        raise ValueError("evidence rank must be a positive integer")
    return numeric


def _fallback_observation_key(candidate: CandidateEvidence) -> str:
    """Conservatively collapse identical boxes when provenance is absent."""

    payload = (
        candidate.page_index,
        round(float(candidate.box.left), 6),
        round(float(candidate.box.bottom), 6),
        round(float(candidate.box.right), 6),
        round(float(candidate.box.top), 6),
        candidate.case_id_hint or "",
        candidate.applicant_hint or "",
    )
    return "fallback:" + "|".join(str(item) for item in payload)


def _observation_keys(candidate: CandidateEvidence) -> tuple[str, ...]:
    physical = {
        "physical:" + item.observation.observation_id
        for item in _bound_provenance(candidate)
    }
    if not physical:
        physical.add(_fallback_observation_key(candidate))
    return tuple(sorted(physical))


def _overlap_fraction(left: object, right: object) -> float:
    """High smaller-box overlap, gated by broadly comparable box areas."""

    intersection_width = max(
        0.0,
        min(left.right, right.right)  # type: ignore[attr-defined]
        - max(left.left, right.left),  # type: ignore[attr-defined]
    )
    intersection_height = max(
        0.0,
        min(left.top, right.top)  # type: ignore[attr-defined]
        - max(left.bottom, right.bottom),  # type: ignore[attr-defined]
    )
    intersection = intersection_width * intersection_height
    left_area = float(left.width * left.height)  # type: ignore[attr-defined]
    right_area = float(right.width * right.height)  # type: ignore[attr-defined]
    smaller = min(left_area, right_area)
    larger = max(left_area, right_area)
    if (
        smaller <= 0.0
        or larger <= 0.0
        or smaller / larger < _NEAR_OBSERVATION_AREA_RATIO
    ):
        return 0.0
    return intersection / smaller


def _near_identical_physical_observation(
    left: CandidateEvidence,
    right: CandidateEvidence,
) -> bool:
    """Collapse high-overlap boxes from the same immutable packet page."""

    for left_item in _bound_provenance(left):
        left_observation = left_item.observation
        for right_item in _bound_provenance(right):
            right_observation = right_item.observation
            if (
                left_observation.source_sha256
                != right_observation.source_sha256
                or left_observation.page_index
                != right_observation.page_index
            ):
                continue
            if (
                _overlap_fraction(
                    left_observation.box,
                    right_observation.box,
                )
                >= _NEAR_OBSERVATION_OVERLAP
            ):
                return True
    return False


def _local_record_ids(candidate: CandidateEvidence) -> frozenset[str]:
    return frozenset(
        match.group(1).casefold()
        for cue in candidate.visual_cues
        if (match := _LOCAL_RECORD_CUE.match(_compact(cue))) is not None
    )


def _share_local_record(
    left: CandidateEvidence,
    right: CandidateEvidence,
) -> bool:
    """Match explicit records across complete or partial provenance."""

    shared_ids = _local_record_ids(left).intersection(
        _local_record_ids(right)
    )
    if (
        not shared_ids
        or left.source != _PRODUCTION_VISIBLE_OCR_SOURCE
        or right.source != _PRODUCTION_VISIBLE_OCR_SOURCE
        or left.page_index != right.page_index
        or (
            left.case_id_hint
            and right.case_id_hint
            and left.case_id_hint != right.case_id_hint
        )
    ):
        return False
    left_sources = {
        item.observation.source_sha256 for item in left.ocr_provenance
    }
    right_sources = {
        item.observation.source_sha256 for item in right.ocr_provenance
    }
    return not (
        left_sources
        and right_sources
        and left_sources.isdisjoint(right_sources)
    )


def candidates_share_physical_observation(
    left: CandidateEvidence,
    right: CandidateEvidence,
) -> bool:
    """Return whether two reads are correlated views of one visible region.

    Exact provenance/local-record identities bind directly.  High-overlap
    geometry is accepted only within the same source/page and non-conflicting
    case anchors. A missing case hint is a wildcard; applicant hints are also
    extracted metadata, not physical discriminators. Neither can turn the same
    page pixels into independent votes. A one-pixel OCR-route shift collapses
    without merging neighboring regions.
    """

    if set(_observation_keys(left)).intersection(_observation_keys(right)):
        return True
    if _share_local_record(left, right):
        return True
    if _near_identical_physical_observation(left, right):
        return True
    if _bound_provenance(left) and _bound_provenance(right):
        return False
    return (
        left.source == right.source == _PRODUCTION_VISIBLE_OCR_SOURCE
        and left.page_index == right.page_index
        and not (
            left.case_id_hint
            and right.case_id_hint
            and left.case_id_hint != right.case_id_hint
        )
        and _overlap_fraction(left.box, right.box)
        >= _NEAR_OBSERVATION_OVERLAP
    )


def _correlated_groups(
    candidates: tuple[_NormalizedCandidate, ...],
) -> tuple[tuple[_NormalizedCandidate, ...], ...]:
    """Return deterministic complete-link physical/local record groups.

    Pairwise geometric overlap is not transitive.  Requiring every member of a
    merged group to correlate with every member prevents a drift chain from
    joining distant page regions.
    """

    if not candidates:
        return ()
    groups = [
        (candidate,)
        for candidate in sorted(
            candidates,
            key=lambda item: _candidate_sort_key(item.candidate),
        )
    ]

    def group_key(
        group: tuple[_NormalizedCandidate, ...],
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            _candidate_sort_key(item.candidate)
            for item in sorted(
                group,
                key=lambda item: _candidate_sort_key(item.candidate),
            )
        )

    while True:
        mergeable: list[
            tuple[
                tuple[tuple[object, ...], ...],
                int,
                int,
            ]
        ] = []
        for left_index, left in enumerate(groups):
            for right_index in range(left_index + 1, len(groups)):
                right = groups[right_index]
                if all(
                    candidates_share_physical_observation(
                        left_item.candidate,
                        right_item.candidate,
                    )
                    for left_item in left
                    for right_item in right
                ):
                    mergeable.append(
                        (
                            group_key((*left, *right)),
                            left_index,
                            right_index,
                        )
                    )
        if not mergeable:
            break
        _, left_index, right_index = min(mergeable)
        groups[left_index] = tuple(
            sorted(
                (*groups[left_index], *groups[right_index]),
                key=lambda item: _candidate_sort_key(item.candidate),
            )
        )
        del groups[right_index]
        groups.sort(key=group_key)
    return tuple(groups)


def _entropy(value_counts: Counter[str]) -> float:
    total = sum(value_counts.values())
    if total <= 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total)
        for count in value_counts.values()
        if count
    )


class EvidenceFuser:
    """Fuse one field while preserving authority and physical independence."""

    @classmethod
    def resolve(
        cls,
        field_name: str,
        candidates: Iterable[CandidateEvidence],
        *,
        ranker: Ranker,
        expected_case_id: str | None,
        active_applicant: str | None,
        active_applicant_aliases: Iterable[str] = (),
    ) -> FusionDecision:
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError("field_name must be a non-empty string")
        expected = None
        if isinstance(expected_case_id, str) and _compact(expected_case_id):
            expected = _compact(expected_case_id).upper()
            if not CASE_ID_PATTERN.fullmatch(expected):
                raise ValueError("expected_case_id must be a valid MIB case ID")
        elif expected_case_id is not None and not isinstance(
            expected_case_id, str
        ):
            raise ValueError("expected_case_id must be a string or None")
        active_applicant_aliases = tuple(active_applicant_aliases)
        valid_applicant_keys: set[str] = set()
        if isinstance(active_applicant, str) and _compact(active_applicant):
            valid_applicant_keys.add(_name_key(active_applicant))
        for alias in active_applicant_aliases:
            if not isinstance(alias, str) or not _compact(alias):
                raise ValueError(
                    "active_applicant_aliases must contain non-empty strings"
                )
            valid_applicant_keys.add(_name_key(alias))

        considered = tuple(
            sorted(
                (
                    candidate
                    for candidate in candidates
                    if candidate.field_name == field_name
                ),
                key=_candidate_sort_key,
            )
        )
        veto_counts: Counter[str] = Counter()
        vetoed_candidates = 0
        eligible: list[_NormalizedCandidate] = []

        for candidate in considered:
            reasons: set[str] = set()
            cues = {_cue_key(cue) for cue in candidate.visual_cues}

            if candidate.evidence_type is EvidenceType.TEXT_LAYER:
                reasons.add("text_layer_diagnostic_only")
            if not candidate.legible:
                reasons.add("illegible")
            if candidate.value is None:
                reasons.add("missing_value")
            if candidate.superseded:
                reasons.add("superseded")
            if cues & _STRIKE_CUES:
                reasons.add("struck")
            if any("watermark" in cue for cue in cues):
                reasons.add("watermarked")
            if _cue_key(candidate.source) in _DEFAULT_SOURCES:
                reasons.add("serialization_default")
            if candidate.source != _PRODUCTION_VISIBLE_OCR_SOURCE:
                reasons.add("non_production_ocr_source")

            if expected is None and field_name == "case_id":
                # Case identity is established by CaseLinker/source filename.
                reasons.add("case_identity_not_linked")
            elif expected is None and candidate.case_id_hint:
                reasons.add("orphan_case_scope")
            if expected is not None and candidate.case_id_hint:
                if _compact(candidate.case_id_hint).upper() != expected:
                    reasons.add("wrong_case")
            if (
                expected is not None
                and field_name == "case_id"
                and isinstance(candidate.value, str)
                and _compact(candidate.value).upper() != expected
            ):
                reasons.add("wrong_case")

            applicant_scopes = {
                _name_key(item.observation.applicant_scope)
                for item in candidate.ocr_provenance
                if item.observation.applicant_scope is not None
            }
            if valid_applicant_keys:
                if (
                    candidate.applicant_hint is not None
                    and _name_key(candidate.applicant_hint)
                    not in valid_applicant_keys
                ):
                    reasons.add("wrong_applicant")
                if applicant_scopes - valid_applicant_keys:
                    reasons.add("wrong_applicant")
            elif field_name == "applicant_name":
                # Applicant identity is established by CaseLinker.  Fusion
                # must not independently manufacture an active identity.
                reasons.add("applicant_identity_not_linked")
            elif (
                candidate.applicant_hint is not None
                or applicant_scopes
            ):
                reasons.add("orphan_applicant_scope")

            normalized = (
                _normalize_value(field_name, candidate.value)
                if isinstance(candidate.value, str)
                else None
            )
            if candidate.value is not None and normalized is None:
                reasons.add("invalid_value")

            if reasons:
                vetoed_candidates += 1
                veto_counts.update(reasons)
                continue

            assert normalized is not None
            observation_keys = _observation_keys(candidate)
            eligible.append(
                _NormalizedCandidate(
                    candidate=candidate,
                    value_key=normalized[0],
                    output_value=normalized[1],
                    rank=_rank(ranker, candidate.evidence_type),
                    observation_keys=observation_keys,
                    correlation_keys=observation_keys,
                    provenance_complete=candidate_has_complete_provenance(
                        candidate,
                        expected_case_id=expected,
                        active_applicant=active_applicant,
                        active_applicant_aliases=active_applicant_aliases,
                    ),
                    correction=bool(cues & _CORRECTION_CUES),
                )
            )

        if not eligible:
            return cls._decision(
                field_name=field_name,
                state="unknown",
                value=None,
                winner=None,
                considered=considered,
                reason="no eligible visible evidence",
                winning_rank=None,
                eligible=(),
                top_rank=(),
                groups=(),
                group_value_sets=(),
                value_counts=Counter(),
                veto_counts=veto_counts,
                vetoed_candidates=vetoed_candidates,
                correction_override_count=0,
            )

        winning_rank = min(item.rank for item in eligible)
        top_rank = tuple(
            sorted(
                (item for item in eligible if item.rank == winning_rank),
                key=lambda item: _candidate_sort_key(item.candidate),
            )
        )
        groups = _correlated_groups(top_rank)

        group_value_sets: list[frozenset[str]] = []
        correction_override_count = 0
        group_finalists: list[tuple[_NormalizedCandidate, ...]] = []
        for group in groups:
            corrections = tuple(item for item in group if item.correction)
            finalists = corrections or group
            if corrections:
                correction_override_count += len(group) - len(corrections)
            group_finalists.append(finalists)
            group_value_sets.append(
                frozenset(item.value_key for item in finalists)
            )

        flattened_values: Counter[str] = Counter()
        singleton_votes: Counter[str] = Counter()
        for values in group_value_sets:
            flattened_values.update(values)
            if len(values) == 1:
                singleton_votes.update(values)

        all_values = set(flattened_values)
        contested = any(len(values) != 1 for values in group_value_sets)
        contested = contested or len(all_values) != 1
        if contested:
            return cls._decision(
                field_name=field_name,
                state="contested",
                value=None,
                winner=None,
                considered=considered,
                reason=f"same-rank conflict at precedence rank {winning_rank}",
                winning_rank=winning_rank,
                eligible=tuple(eligible),
                top_rank=top_rank,
                groups=groups,
                group_value_sets=tuple(group_value_sets),
                value_counts=flattened_values,
                veto_counts=veto_counts,
                vetoed_candidates=vetoed_candidates,
                correction_override_count=correction_override_count,
            )

        value_key = next(iter(all_values))
        finalists = tuple(
            item
            for group in group_finalists
            for item in group
            if item.value_key == value_key
        )
        winner_item = min(
            finalists,
            key=lambda item: (
                not item.provenance_complete,
                -_confidence(item.candidate),
                _candidate_sort_key(item.candidate),
            ),
        )
        return cls._decision(
            field_name=field_name,
            state="resolved",
            value=winner_item.output_value,
            winner=winner_item.candidate,
            considered=considered,
            reason=(
                f"resolved at precedence rank {winning_rank} from "
                f"{len(groups)} independent observation(s)"
            ),
            winning_rank=winning_rank,
            eligible=tuple(eligible),
            top_rank=top_rank,
            groups=groups,
            group_value_sets=tuple(group_value_sets),
            value_counts=flattened_values,
            veto_counts=veto_counts,
            vetoed_candidates=vetoed_candidates,
            correction_override_count=correction_override_count,
        )

    @staticmethod
    def _decision(
        *,
        field_name: str,
        state: str,
        value: str | None,
        winner: CandidateEvidence | None,
        considered: tuple[CandidateEvidence, ...],
        reason: str,
        winning_rank: int | None,
        eligible: tuple[_NormalizedCandidate, ...],
        top_rank: tuple[_NormalizedCandidate, ...],
        groups: tuple[tuple[_NormalizedCandidate, ...], ...],
        group_value_sets: tuple[frozenset[str], ...],
        value_counts: Counter[str],
        veto_counts: Counter[str],
        vetoed_candidates: int,
        correction_override_count: int,
    ) -> FusionDecision:
        physical_observations = {
            key
            for item in top_rank
            for key in item.observation_keys
        }
        evidence_types = {
            item.candidate.evidence_type for item in top_rank
        }
        pages = {item.candidate.page_index for item in top_rank}
        complete_groups = sum(
            bool(group) and all(item.provenance_complete for item in group)
            for group in groups
        )
        completeness = (
            complete_groups / len(groups) if groups else 0.0
        )

        singleton_votes: Counter[str] = Counter()
        local_conflicts = 0
        for values in group_value_sets:
            if len(values) == 1:
                singleton_votes.update(values)
            else:
                local_conflicts += 1
        agreement = max(singleton_votes.values(), default=0)
        if state == "resolved":
            disagreement_count = 0
        else:
            disagreement_count = local_conflicts + max(
                0,
                sum(singleton_votes.values()) - agreement,
            )
        disagreement_ratio = (
            disagreement_count / len(groups) if groups else 0.0
        )
        correlated_count = max(0, len(top_rank) - len(groups))
        lower_authority_count = max(0, len(eligible) - len(top_rank))

        safety = Counter(
            {
                "candidate_count": len(considered),
                "eligible_candidate_count": len(eligible),
                "vetoed_candidate_count": vetoed_candidates,
                "binding_rank_candidate_count": (
                    len(top_rank) if winning_rank == 1 else 0
                ),
                "lower_authority_ignored_count": lower_authority_count,
                "physical_observation_count": len(physical_observations),
                "independent_evidence_count": len(groups),
                "correlated_candidate_count": correlated_count,
                "local_correction_override_count": correction_override_count,
                "same_rank_conflict_count": int(state == "contested"),
                "serialization_default_used_as_evidence_count": 0,
                "resolved_count": int(state == "resolved"),
                "unknown_count": int(state == "unknown"),
                "contested_count": int(state == "contested"),
            }
        )
        for veto_name, count in veto_counts.items():
            safety[f"veto_{veto_name}_count"] = count

        trace = FusionTrace(
            winning_rank=winning_rank,
            candidate_count=len(considered),
            eligible_candidate_count=len(eligible),
            winning_rank_candidate_count=len(top_rank),
            observation_count=len(physical_observations),
            independent_evidence_count=len(groups),
            independent_agreement_count=agreement,
            correlated_candidate_count=correlated_count,
            independent_evidence_type_count=len(evidence_types),
            independent_page_count=len(pages),
            disagreement_count=disagreement_count,
            entropy_bits=round(_entropy(value_counts), 12),
            disagreement_ratio=round(disagreement_ratio, 12),
            provenance_complete_count=complete_groups,
            provenance_completeness=round(completeness, 12),
            value_counts=tuple(sorted(value_counts.items())),
            veto_reasons=tuple(sorted(veto_counts.items())),
            safety_counters=tuple(sorted(safety.items())),
        )
        return FusionDecision(
            field_name=field_name,
            state=state,
            value=value,
            winner=winner,
            considered=considered,
            trace=trace,
            reason=reason,
        )
