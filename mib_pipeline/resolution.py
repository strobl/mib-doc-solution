"""Active-case linking and deterministic evidence precedence resolution."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping

from .extraction import CandidateEvidence, EvidenceType
from .fusion import (
    EvidenceFuser,
    FusionTrace,
    candidate_has_complete_provenance,
    candidates_share_physical_observation,
)
from .models import CASE_ID_PATTERN, FIELD_NAMES


POLICY_ONLY_FIELDS = (
    "stay_duration_days",
    "packet_receipt_date",
    "biohazard_check",
    "hardship_waiver",
    "diplomatic_waiver_code",
    "diplomatic_note",
    "minimal_diplomatic_packet",
    "work_permit_requested",
    "page_type_present_fee_receipt",
    "page_type_present_other",
    "page_type_present_sponsor_attestation",
)

RESOLVABLE_FIELDS = (
    tuple(field for field in FIELD_NAMES if field != "confidence")
    + POLICY_ONLY_FIELDS
)


class FieldState(str, Enum):
    RESOLVED = "resolved"
    UNKNOWN = "unknown"
    CONTESTED = "contested"


@dataclass(frozen=True)
class LinkedCase:
    case_id: str
    active_applicant: str | None
    evidence: tuple[CandidateEvidence, ...]
    unresolved: bool
    unresolved_reasons: tuple[str, ...] = ()
    active_applicant_aliases: tuple[str, ...] = ()
    cross_applicant_candidates_excluded: int = 0


@dataclass(frozen=True)
class ResolvedField:
    field_name: str
    state: FieldState
    value: str | None
    winning_evidence: CandidateEvidence | None
    considered: tuple[CandidateEvidence, ...]
    reason: str
    fusion_trace: FusionTrace | None = None


@dataclass(frozen=True)
class ResolvedCase:
    case_id: str
    active_applicant: str | None
    fields: Mapping[str, ResolvedField]
    unresolved_linkage: bool
    unresolved_reasons: tuple[str, ...]
    rescinded_decision: bool = False
    fusion_audit_counts: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def value(self, field_name: str) -> str | None:
        field = self.fields[field_name]
        return field.value if field.state is FieldState.RESOLVED else None

    @property
    def contested_fields(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name, field in self.fields.items()
            if field.state is FieldState.CONTESTED
        )

    @property
    def unknown_fields(self) -> tuple[str, ...]:
        return tuple(
            field_name
            for field_name, field in self.fields.items()
            if field.state is FieldState.UNKNOWN
        )


class EvidencePrecedenceHierarchy:
    """Binding six-level hierarchy from the MIB field manual."""

    _RANKS = {
        EvidenceType.ADJUDICATOR_STAMP: 1,
        EvidenceType.SIGNED_MANUAL_NOTE: 1,
        EvidenceType.INTAKE_FORM: 2,
        EvidenceType.BIOMETRIC_SLIP: 3,
        EvidenceType.SPONSOR_ATTESTATION: 4,
        EvidenceType.REGISTRY_EXTRACT: 5,
        EvidenceType.TEXT_LAYER: 6,
    }

    @classmethod
    def rank(cls, evidence_type: EvidenceType) -> int:
        return cls._RANKS[evidence_type]


class CaseLinker:
    """Scope evidence to the case filename and its reliably-linked applicant."""

    _NAME_CLUSTER_SIMILARITY = 0.80
    _SPONSOR_CORROBORATION_SIMILARITY = 0.65
    _SPONSOR_MINIMUM_CONFIDENCE = 0.90
    _SUPPORT_MINIMUM_CONFIDENCE = 0.75
    _LOW_INTAKE_CONFIDENCE = 0.72
    _CONFLICTING_NAME_SIMILARITY = 0.90

    @staticmethod
    def _is_clean_visible_candidate(candidate: CandidateEvidence) -> bool:
        cues = {
            re.sub(r"[\s-]+", "_", cue.strip().casefold())
            for cue in candidate.visual_cues
        }
        return (
            candidate.source == "visible_ocr"
            and candidate.evidence_type is not EvidenceType.TEXT_LAYER
            and candidate.legible
            and candidate.value is not None
            and not candidate.superseded
            and not cues.intersection(
                {
                    "crossed_out",
                    "strike",
                    "strikethrough",
                    "struck",
                    "struck_out",
                    "struck_through",
                }
            )
            and not any("watermark" in cue for cue in cues)
        )

    @staticmethod
    def _normalized_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())

    @classmethod
    def _name_similarity(cls, left: str, right: str) -> float:
        left_key = cls._normalized_name(left)
        right_key = cls._normalized_name(right)
        if not left_key or not right_key:
            return 0.0
        return difflib.SequenceMatcher(None, left_key, right_key).ratio()

    @classmethod
    def _candidate_sort_key(
        cls,
        candidate: CandidateEvidence,
    ) -> tuple[object, ...]:
        """Provide a content-only tie-break independent of extraction order."""

        return (
            cls._normalized_name(candidate.value or ""),
            candidate.value or "",
            EvidencePrecedenceHierarchy.rank(candidate.evidence_type),
            candidate.evidence_type.value,
            candidate.page_index,
            candidate.box.left,
            candidate.box.bottom,
            candidate.box.right,
            candidate.box.top,
            -candidate.ocr_confidence,
            candidate.case_id_hint or "",
            candidate.applicant_hint or "",
            candidate.source,
            candidate.visual_cues,
            tuple(item.fingerprint for item in candidate.ocr_provenance),
        )

    @classmethod
    def _cluster_sort_key(
        cls,
        cluster: Iterable[CandidateEvidence],
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            cls._candidate_sort_key(candidate)
            for candidate in sorted(cluster, key=cls._candidate_sort_key)
        )

    @classmethod
    def _independent_observation_groups(
        cls,
        candidates: Iterable[CandidateEvidence],
    ) -> tuple[tuple[CandidateEvidence, ...], ...]:
        """Collapse correlated OCR routes before measuring identity support."""

        ordered = tuple(sorted(candidates, key=cls._candidate_sort_key))
        groups = [(candidate,) for candidate in ordered]
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
                            left_candidate,
                            right_candidate,
                        )
                        for left_candidate in left
                        for right_candidate in right
                    ):
                        mergeable.append(
                            (
                                cls._cluster_sort_key((*left, *right)),
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
                    key=cls._candidate_sort_key,
                )
            )
            del groups[right_index]
            groups.sort(key=cls._cluster_sort_key)
        return tuple(groups)

    @classmethod
    def _cluster_applicants(
        cls,
        applicant_evidence: Iterable[CandidateEvidence],
    ) -> list[list[CandidateEvidence]]:
        """Build deterministic complete-link name clusters.

        Every two names in one cluster must independently meet the similarity
        threshold.  This prevents a fuzzy A~B~C chain from merging A and C
        when those endpoints are not compatible.  At each agglomerative step,
        the strongest complete-link pair wins with a content-only tie-break,
        so input order cannot change the partition.
        """

        clusters = [
            [candidate]
            for candidate in sorted(
                applicant_evidence,
                key=cls._candidate_sort_key,
            )
        ]
        while True:
            eligible_pairs: list[
                tuple[
                    float,
                    tuple[tuple[object, ...], ...],
                    int,
                    int,
                ]
            ] = []
            for left_index, left in enumerate(clusters):
                for right_index in range(left_index + 1, len(clusters)):
                    right = clusters[right_index]
                    complete_link_similarity = min(
                        cls._name_similarity(
                            left_candidate.value or "",
                            right_candidate.value or "",
                        )
                        for left_candidate in left
                        for right_candidate in right
                    )
                    if (
                        complete_link_similarity
                        < cls._NAME_CLUSTER_SIMILARITY
                    ):
                        continue
                    merged_key = cls._cluster_sort_key((*left, *right))
                    eligible_pairs.append(
                        (
                            complete_link_similarity,
                            merged_key,
                            left_index,
                            right_index,
                        )
                    )
            if not eligible_pairs:
                break
            _, _, left_index, right_index = min(
                eligible_pairs,
                key=lambda pair: (-pair[0], pair[1]),
            )
            clusters[left_index] = sorted(
                (*clusters[left_index], *clusters[right_index]),
                key=cls._candidate_sort_key,
            )
            del clusters[right_index]
            clusters.sort(key=cls._cluster_sort_key)
        return clusters

    @classmethod
    def _corroborated_sponsor_applicant(
        cls,
        applicant_evidence: Iterable[CandidateEvidence],
    ) -> tuple[str, set[str]] | None:
        """Prefer a structured sponsor name only over a damaged intake read.

        The override is deliberately narrow: a high-confidence applicant from
        the repeated sponsor sentence must be independently corroborated by a
        registry or biometric applicant. Every conflicting intake applicant
        must be below the refinement confidence gate. The returned aliases omit
        those damaged intake readings so their page-scoped values cannot regain
        precedence over the corroborated sources.
        """

        applicant_evidence = tuple(applicant_evidence)
        structured_sponsors = tuple(
            candidate
            for candidate in applicant_evidence
            if candidate.evidence_type is EvidenceType.SPONSOR_ATTESTATION
            and cls._is_clean_visible_candidate(candidate)
            and "structured_sponsor_narrative" in candidate.visual_cues
            and candidate.value is not None
            and candidate.ocr_confidence >= cls._SPONSOR_MINIMUM_CONFIDENCE
        )
        sponsor_values = {
            candidate.value for candidate in structured_sponsors if candidate.value
        }
        if len(sponsor_values) != 1:
            return None
        sponsor_value = next(iter(sponsor_values))
        supporting = tuple(
            candidate
            for candidate in applicant_evidence
            if candidate.evidence_type
            in {EvidenceType.BIOMETRIC_SLIP, EvidenceType.REGISTRY_EXTRACT}
            and cls._is_clean_visible_candidate(candidate)
            and candidate.value is not None
            and candidate.ocr_confidence >= cls._SUPPORT_MINIMUM_CONFIDENCE
            and cls._name_similarity(sponsor_value, candidate.value)
            >= cls._SPONSOR_CORROBORATION_SIMILARITY
            and not any(
                candidates_share_physical_observation(
                    candidate,
                    sponsor,
                )
                for sponsor in structured_sponsors
            )
        )
        if not supporting:
            return None
        conflicting_intake = tuple(
            candidate
            for candidate in applicant_evidence
            if candidate.evidence_type is EvidenceType.INTAKE_FORM
            and cls._is_clean_visible_candidate(candidate)
            and candidate.value is not None
            and cls._name_similarity(sponsor_value, candidate.value)
            < cls._CONFLICTING_NAME_SIMILARITY
        )
        if not conflicting_intake or any(
            candidate.ocr_confidence >= cls._LOW_INTAKE_CONFIDENCE
            for candidate in conflicting_intake
        ):
            return None
        aliases = {sponsor_value}
        aliases.update(
            candidate.value for candidate in supporting if candidate.value is not None
        )
        return sponsor_value, aliases

    @staticmethod
    def _clean_scoped_candidates(
        evidence: Iterable[CandidateEvidence],
        field_name: str,
    ) -> tuple[CandidateEvidence, ...]:
        return tuple(
            candidate
            for candidate in evidence
            if candidate.field_name == field_name
            and CaseLinker._is_clean_visible_candidate(candidate)
        )

    @staticmethod
    def _physical_source_signature(
        candidate: CandidateEvidence,
    ) -> tuple[object, ...]:
        """Identify one visible fact while ignoring its sequential name hint."""

        return (
            candidate.value,
            candidate.evidence_type,
            candidate.page_index,
            candidate.box,
            candidate.ocr_confidence,
            candidate.visual_cues,
            candidate.source,
        )

    @classmethod
    def _non_identity_scope_is_stable(
        cls,
        ordinary: tuple[CandidateEvidence, ...],
        alternative: tuple[CandidateEvidence, ...],
    ) -> bool:
        """Allow a corroborated name only when every other visible fact is stable."""

        field_names = {
            candidate.field_name
            for evidence in (ordinary, alternative)
            for candidate in evidence
        } - {"case_id", "applicant_name"}
        source_stable_fields = {
            "adjudication",
            "risk_flags",
            *POLICY_ONLY_FIELDS,
        }
        for field_name in field_names:
            old = cls._clean_scoped_candidates(ordinary, field_name)
            new = cls._clean_scoped_candidates(alternative, field_name)
            if {candidate.value for candidate in old} != {
                candidate.value for candidate in new
            }:
                return False
            if field_name in source_stable_fields and {
                cls._physical_source_signature(candidate) for candidate in old
            } != {
                cls._physical_source_signature(candidate) for candidate in new
            }:
                return False
        return True

    @classmethod
    def _case_pages_are_separable(
        cls,
        expected_case_id: str,
        candidates: tuple[CandidateEvidence, ...],
        conflicting_case_ids: set[str],
    ) -> bool:
        """Return whether every visible case occupies an exact, disjoint page.

        A packet can contain an archival page for another case without making
        the filename-selected case ambiguous.  That is safe only when the
        expected case is itself visibly anchored and every extracted fact on
        every case-bearing page carries the same exact case hint as that
        page's single clean case marker.  Unhinted facts, mixed case markers,
        or orphan pages keep the conservative unresolved result.
        """

        visible_ids = {expected_case_id, *conflicting_case_ids}
        page_case_ids: dict[int, set[str]] = {}
        for candidate in candidates:
            if (
                candidate.field_name == "case_id"
                and cls._is_clean_visible_candidate(candidate)
                and candidate.value in visible_ids
            ):
                page_case_ids.setdefault(candidate.page_index, set()).add(
                    candidate.value
                )

        if not any(
            page_ids == {expected_case_id}
            for page_ids in page_case_ids.values()
        ):
            return False
        if (
            any(len(page_ids) != 1 for page_ids in page_case_ids.values())
            or not conflicting_case_ids.issubset(
                {
                    next(iter(page_ids))
                    for page_ids in page_case_ids.values()
                }
            )
        ):
            return False

        for candidate in candidates:
            page_ids = page_case_ids.get(candidate.page_index)
            if page_ids is None:
                return False
            page_case_id = next(iter(page_ids))
            if candidate.field_name == "case_id":
                if (
                    candidate.value != page_case_id
                    or candidate.case_id_hint != page_case_id
                ):
                    return False
            elif candidate.case_id_hint != page_case_id:
                return False
        return True

    @staticmethod
    def _exact_lower_corroboration(
        case_id: str,
        lower_clusters: Iterable[list[CandidateEvidence]],
    ) -> tuple[str, set[str]] | None:
        """Find one verbatim lower name repeated across pages and source types."""

        by_value: dict[str, list[CandidateEvidence]] = {}
        for cluster in lower_clusters:
            for candidate in cluster:
                if candidate.value is not None:
                    by_value.setdefault(candidate.value, []).append(candidate)
        qualifying = [
            candidates
            for candidates in by_value.values()
            if len(
                CaseLinker._independent_observation_groups(candidates)
            )
            >= 2
            and len({candidate.page_index for candidate in candidates}) >= 2
            and len({candidate.evidence_type for candidate in candidates}) >= 2
            and all(
                candidate.case_id_hint == case_id
                and CaseLinker._is_clean_visible_candidate(candidate)
                for candidate in candidates
            )
        ]
        if len(qualifying) != 1:
            return None
        value = qualifying[0][0].value
        return (value, {value}) if value is not None else None

    def link(
        self,
        expected_case_id: str | None,
        candidates: Iterable[CandidateEvidence],
    ) -> LinkedCase:
        candidates = tuple(candidates)
        expected = (
            expected_case_id.strip()
            if isinstance(expected_case_id, str)
            and CASE_ID_PATTERN.fullmatch(expected_case_id.strip())
            else None
        )
        visible_case_ids = {
            candidate.value
            for candidate in candidates
            if candidate.field_name == "case_id"
            and self._is_clean_visible_candidate(candidate)
            and CASE_ID_PATTERN.fullmatch(candidate.value)
        }
        reasons: list[str] = []
        conflicting_visible_case_ids: set[str] = set()
        if expected is not None:
            case_id = expected
            conflicting_visible_case_ids = visible_case_ids - {expected}
            if (
                conflicting_visible_case_ids
                and not self._case_pages_are_separable(
                    expected,
                    candidates,
                    conflicting_visible_case_ids,
                )
            ):
                reasons.append("visible case_id conflicts with source filename")
        elif len(visible_case_ids) == 1:
            case_id = next(iter(visible_case_ids))
        else:
            case_id = ""
            reasons.append("active case_id cannot be determined")

        if case_id and conflicting_visible_case_ids:
            # The filename keeps the active case association, but an unhinted
            # fact could belong to either visibly-present case.  Retain only
            # diagnostic case identities and facts explicitly anchored to the
            # expected case.
            case_scoped = tuple(
                candidate
                for candidate in candidates
                if candidate.field_name == "case_id"
                or candidate.case_id_hint == case_id
            )
        elif case_id:
            case_scoped = tuple(
                candidate
                for candidate in candidates
                if not candidate.case_id_hint
                or candidate.case_id_hint == case_id
            )
        else:
            case_scoped = tuple(
                candidate
                for candidate in candidates
                if candidate.field_name in {"case_id", "applicant_name"}
                or not candidate.case_id_hint
            )
        applicant_evidence = tuple(
            candidate
            for candidate in case_scoped
            if candidate.field_name == "applicant_name"
            and self._is_clean_visible_candidate(candidate)
        )
        clusters = self._cluster_applicants(applicant_evidence)

        def cluster_strength(cluster: list[CandidateEvidence]) -> tuple[int, int, int, float]:
            independent = self._independent_observation_groups(cluster)
            group_ranks = tuple(
                min(
                    EvidencePrecedenceHierarchy.rank(
                        candidate.evidence_type
                    )
                    for candidate in group
                )
                for group in independent
            )
            best_rank = min(group_ranks)
            return (
                -best_rank,
                group_ranks.count(best_rank),
                len(independent),
                max(candidate.ocr_confidence for candidate in cluster),
            )

        ranked_clusters = sorted(clusters, key=self._cluster_sort_key)
        ranked_clusters.sort(key=cluster_strength, reverse=True)
        active_aliases: set[str] = set()
        corroborated_sponsor = self._corroborated_sponsor_applicant(
            applicant_evidence
        )
        if corroborated_sponsor is not None:
            active_applicant, active_aliases = corroborated_sponsor
        elif ranked_clusters:
            winning_cluster = ranked_clusters[0]
            tied = (
                len(ranked_clusters) > 1
                and cluster_strength(ranked_clusters[1]) == cluster_strength(winning_cluster)
            )
            if tied:
                active_applicant = None
                reasons.append("multiple applicants cannot be reliably separated")
            else:
                representative = min(
                    winning_cluster,
                    key=lambda candidate: (
                        EvidencePrecedenceHierarchy.rank(
                            candidate.evidence_type
                        ),
                        -candidate.ocr_confidence,
                        self._candidate_sort_key(candidate),
                    ),
                )
                active_applicant = representative.value
                active_aliases = {
                    candidate.value
                    for candidate in winning_cluster
                    if candidate.value is not None
                }
        else:
            active_applicant = None

        def scope_evidence(
            selected_applicant: str | None,
            selected_aliases: set[str],
        ) -> tuple[CandidateEvidence, ...]:
            # A visible same-page manual applicant correction changes the
            # subject of the intake record without invalidating earlier fields.
            correction_pages = {
                candidate.page_index
                for candidate in case_scoped
                if candidate.field_name == "applicant_name"
                and candidate.value in selected_aliases
                and "correction" in candidate.visual_cues
                and self._is_clean_visible_candidate(candidate)
            }
            corrected_page_aliases = {
                (candidate.page_index, candidate.value)
                for candidate in case_scoped
                if candidate.page_index in correction_pages
                and candidate.field_name == "applicant_name"
                and candidate.value
                and candidate.value not in selected_aliases
                and (
                    candidate.superseded
                    or "strikethrough" in candidate.visual_cues
                )
            }
            if selected_applicant is None:
                return tuple(
                    candidate
                    for candidate in case_scoped
                    if candidate.field_name in {"case_id", "applicant_name"}
                    or not candidate.applicant_hint
                )
            return tuple(
                candidate
                for candidate in case_scoped
                if not candidate.applicant_hint
                or not selected_applicant
                or candidate.applicant_hint in selected_aliases
                or (candidate.page_index, candidate.applicant_hint)
                in corrected_page_aliases
            )

        applicant_scoped = scope_evidence(active_applicant, active_aliases)
        if (
            corroborated_sponsor is None
            and not reasons
            and active_applicant is not None
            and len(ranked_clusters) >= 2
            and len(ranked_clusters[0]) == 1
        ):
            current = ranked_clusters[0][0]
            lower_corroboration = self._exact_lower_corroboration(
                case_id,
                ranked_clusters[1:],
            )
            if (
                current.evidence_type is EvidenceType.INTAKE_FORM
                and current.case_id_hint == case_id
                and not current.superseded
                and "strikethrough" not in current.visual_cues
                and "sample_denial_watermark" not in current.visual_cues
                and active_applicant == current.value
                and lower_corroboration is not None
            ):
                alternative_applicant, alternative_aliases = lower_corroboration
                alternative_scoped = scope_evidence(
                    alternative_applicant,
                    alternative_aliases,
                )
                if self._non_identity_scope_is_stable(
                    applicant_scoped,
                    alternative_scoped,
                ):
                    active_applicant = alternative_applicant
                    active_aliases = alternative_aliases
                    applicant_scoped = alternative_scoped

        # A struck same-page spelling that is visibly replaced by the selected
        # applicant remains a local alias for the already-scoped fields on that
        # page.  The linker has excluded that spelling everywhere else, so
        # forwarding it cannot reopen foreign-applicant evidence.
        if active_applicant is not None:
            selected_correction_pages = {
                candidate.page_index
                for candidate in applicant_scoped
                if candidate.field_name == "applicant_name"
                and candidate.value in active_aliases
                and "correction" in candidate.visual_cues
                and self._is_clean_visible_candidate(candidate)
            }
            active_aliases.update(
                candidate.value
                for candidate in applicant_scoped
                if candidate.field_name == "applicant_name"
                and candidate.page_index in selected_correction_pages
                and candidate.value
                and (
                    candidate.superseded
                    or "strikethrough" in candidate.visual_cues
                )
            )

        return LinkedCase(
            case_id=case_id,
            active_applicant=active_applicant,
            evidence=applicant_scoped,
            unresolved=bool(reasons),
            unresolved_reasons=tuple(reasons),
            active_applicant_aliases=tuple(
                sorted(
                    active_aliases,
                    key=lambda value: (self._normalized_name(value), value),
                )
            ),
            cross_applicant_candidates_excluded=(
                len(case_scoped) - len(applicant_scoped)
            ),
        )


class RescindedDecisionHandler:
    """Neutralize decorative or visibly overturned denial decisions."""

    @staticmethod
    def _sequence(candidate: CandidateEvidence) -> tuple[int, float]:
        return candidate.page_index, candidate.box.top

    def filter(
        self,
        candidates: Iterable[CandidateEvidence],
        *,
        expected_case_id: str | None = None,
        active_applicant: str | None = None,
        active_applicant_aliases: Iterable[str] = (),
    ) -> tuple[tuple[CandidateEvidence, ...], bool]:
        candidates = tuple(candidates)
        active_applicant_aliases = tuple(active_applicant_aliases)
        eligible = [
            candidate
            for candidate in candidates
            if "sample_denial_watermark" not in candidate.visual_cues
        ]
        decisions = [
            candidate
            for candidate in eligible
            if candidate.field_name == "adjudication" and candidate.value
        ]
        valid_applicant_keys = {
            CaseLinker._normalized_name(value)
            for value in (
                active_applicant,
                *active_applicant_aliases,
            )
            if value
        }

        def coherent_scope(candidate: CandidateEvidence) -> bool:
            if (
                expected_case_id
                and candidate.case_id_hint
                and candidate.case_id_hint != expected_case_id
            ):
                return False
            if valid_applicant_keys:
                if (
                    candidate.applicant_hint is not None
                    and CaseLinker._normalized_name(
                        candidate.applicant_hint
                    )
                    not in valid_applicant_keys
                ):
                    return False
                if any(
                    item.observation.applicant_scope is not None
                    and CaseLinker._normalized_name(
                        item.observation.applicant_scope
                    )
                    not in valid_applicant_keys
                    for item in candidate.ocr_provenance
                ):
                    return False
            elif (
                candidate.applicant_hint is not None
                or any(
                    item.observation.applicant_scope is not None
                    for item in candidate.ocr_provenance
                )
            ):
                return False
            return (
                not candidate.ocr_provenance
                or candidate_has_complete_provenance(
                    candidate,
                    expected_case_id=expected_case_id,
                    active_applicant=active_applicant,
                    active_applicant_aliases=active_applicant_aliases,
                )
            )

        later_signed_approvals = [
            candidate
            for candidate in decisions
            if candidate.value == "APPROVED"
            and candidate.evidence_type is EvidenceType.SIGNED_MANUAL_NOTE
            and "correction" in candidate.visual_cues
            and CaseLinker._is_clean_visible_candidate(candidate)
            and coherent_scope(candidate)
        ]
        rescinded = False
        if later_signed_approvals:
            latest_approval = max(later_signed_approvals, key=self._sequence)
            filtered: list[CandidateEvidence] = []
            for candidate in eligible:
                is_overturned_denial = (
                    candidate.field_name == "adjudication"
                    and candidate.value == "DENIED"
                    and candidate.evidence_type is EvidenceType.ADJUDICATOR_STAMP
                    and self._sequence(candidate) < self._sequence(latest_approval)
                )
                if is_overturned_denial:
                    rescinded = True
                    continue
                filtered.append(candidate)
            eligible = filtered
        return tuple(eligible), rescinded


class EvidencePrecedenceResolver:
    """Resolve one coherent field set after case/applicant scoping."""

    _STRUCTURED_SPONSOR_REPAIR_FIELDS = frozenset(
        {"sponsor_id", "visa_class"}
    )
    _STRUCTURED_SPONSOR_MINIMUM_CONFIDENCE = 0.90
    _STRUCTURED_SPONSOR_NAME_SIMILARITY = 0.80

    def __init__(
        self,
        *,
        hierarchy: type[EvidencePrecedenceHierarchy] = EvidencePrecedenceHierarchy,
        rescinded_handler: RescindedDecisionHandler | None = None,
        fusion_enabled: bool = True,
    ) -> None:
        if not isinstance(fusion_enabled, bool):
            raise TypeError("fusion_enabled must be a boolean")
        self._hierarchy = hierarchy
        self._rescinded = rescinded_handler or RescindedDecisionHandler()
        self._fusion_enabled = fusion_enabled

    @property
    def fusion_enabled(self) -> bool:
        """Whether applicant-aware provenance fusion is the active resolver."""

        return self._fusion_enabled

    def _resolve_field(
        self,
        field_name: str,
        candidates: Iterable[CandidateEvidence],
    ) -> ResolvedField:
        considered = tuple(
            candidate
            for candidate in candidates
            if candidate.field_name == field_name
        )
        eligible = tuple(
            candidate
            for candidate in considered
            if candidate.legible
            and candidate.value is not None
            and not candidate.superseded
            and "strikethrough" not in candidate.visual_cues
        )
        if not eligible:
            return ResolvedField(
                field_name=field_name,
                state=FieldState.UNKNOWN,
                value=None,
                winning_evidence=None,
                considered=considered,
                reason="no eligible evidence",
            )

        winning_rank = min(
            self._hierarchy.rank(candidate.evidence_type) for candidate in eligible
        )
        top_rank = tuple(
            candidate
            for candidate in eligible
            if self._hierarchy.rank(candidate.evidence_type) == winning_rank
        )
        corrections = tuple(
            candidate
            for candidate in top_rank
            if "correction" in candidate.visual_cues
        )
        finalists = corrections or top_rank
        values = {candidate.value for candidate in finalists}
        if len(values) != 1:
            return ResolvedField(
                field_name=field_name,
                state=FieldState.CONTESTED,
                value=None,
                winning_evidence=None,
                considered=considered,
                reason=f"same-rank conflict at precedence rank {winning_rank}",
            )
        value = next(iter(values))
        winning_evidence = max(
            finalists,
            key=lambda candidate: (
                candidate.ocr_confidence,
                candidate.page_index,
                candidate.box.top,
            ),
        )
        return ResolvedField(
            field_name=field_name,
            state=FieldState.RESOLVED,
            value=value,
            winning_evidence=winning_evidence,
            considered=considered,
            reason=f"resolved at precedence rank {winning_rank}",
        )

    def _resolve_field_with_fusion(
        self,
        field_name: str,
        candidates: Iterable[CandidateEvidence],
        linked_case: LinkedCase,
    ) -> ResolvedField:
        decision = EvidenceFuser.resolve(
            field_name,
            candidates,
            ranker=self._hierarchy,
            expected_case_id=linked_case.case_id or None,
            active_applicant=linked_case.active_applicant,
            active_applicant_aliases=linked_case.active_applicant_aliases,
        )
        return ResolvedField(
            field_name=decision.field_name,
            state=FieldState(decision.state),
            value=decision.value,
            winning_evidence=decision.winning_evidence,
            considered=decision.considered,
            reason=decision.reason,
            fusion_trace=decision.trace,
        )

    @classmethod
    def _structured_sponsor_conflict_repair(
        cls,
        field_name: str,
        case_id: str,
        candidates: Iterable[CandidateEvidence],
    ) -> ResolvedField | None:
        """Repair two narrow OCR conflicts using a redundant sponsor sentence.

        Sponsor prose repeats the applicant, sponsor ID, and visa class in one
        structured sentence.  It is usable as an OCR repair only when every
        relevant source is clean, explicitly exact-case, applicant-aligned,
        and the high-confidence sponsor reading has one unique value.  This is
        not a general sponsor-over-intake precedence override.
        """

        if field_name not in cls._STRUCTURED_SPONSOR_REPAIR_FIELDS or not case_id:
            return None
        considered = tuple(
            candidate
            for candidate in candidates
            if candidate.field_name == field_name
        )

        def clean(candidate: CandidateEvidence) -> bool:
            return (
                candidate.legible
                and candidate.value is not None
                and not candidate.superseded
                and "strikethrough" not in candidate.visual_cues
                and "sample_denial_watermark" not in candidate.visual_cues
                and candidate.case_id_hint == case_id
                and bool(candidate.applicant_hint)
            )

        intake = tuple(
            candidate
            for candidate in considered
            if candidate.evidence_type is EvidenceType.INTAKE_FORM
            and clean(candidate)
        )
        sponsor = tuple(
            candidate
            for candidate in considered
            if candidate.evidence_type is EvidenceType.SPONSOR_ATTESTATION
            and "structured_sponsor_narrative" in candidate.visual_cues
            and candidate.ocr_confidence
            >= cls._STRUCTURED_SPONSOR_MINIMUM_CONFIDENCE
            and clean(candidate)
        )
        if not intake or not sponsor or any(
            "correction" in candidate.visual_cues for candidate in intake
        ):
            return None

        intake_values = {candidate.value for candidate in intake}
        sponsor_values = {candidate.value for candidate in sponsor}
        if (
            len(intake_values) != 1
            or len(sponsor_values) != 1
            or intake_values == sponsor_values
            or any(
                CaseLinker._name_similarity(
                    intake_candidate.applicant_hint or "",
                    sponsor_candidate.applicant_hint or "",
                )
                < cls._STRUCTURED_SPONSOR_NAME_SIMILARITY
                for intake_candidate in intake
                for sponsor_candidate in sponsor
            )
        ):
            return None

        value = next(iter(sponsor_values))
        winning_evidence = max(
            sponsor,
            key=lambda candidate: (
                candidate.ocr_confidence,
                candidate.page_index,
                candidate.box.top,
            ),
        )
        return ResolvedField(
            field_name=field_name,
            state=FieldState.RESOLVED,
            value=value,
            winning_evidence=winning_evidence,
            considered=considered,
            reason="structured exact-case sponsor narrative OCR repair",
        )

    def _with_case_and_applicant_associations(
        self,
        linked_case: LinkedCase,
        evidence: tuple[CandidateEvidence, ...],
        fields: dict[str, ResolvedField],
    ) -> None:
        """Make the linker identity authoritative without losing diagnostics."""

        if linked_case.case_id:
            case_candidates = tuple(
                candidate
                for candidate in evidence
                if candidate.field_name == "case_id"
                and CaseLinker._is_clean_visible_candidate(candidate)
                and candidate.value == linked_case.case_id
            )
            current_case_winner = fields.get("case_id")
            case_winner = (
                current_case_winner.winning_evidence
                if current_case_winner is not None
                and current_case_winner.value == linked_case.case_id
                and current_case_winner.winning_evidence in case_candidates
                else (
                    min(
                        case_candidates,
                        key=lambda candidate: (
                            self._hierarchy.rank(candidate.evidence_type),
                            -candidate.ocr_confidence,
                            CaseLinker._candidate_sort_key(candidate),
                        ),
                    )
                    if case_candidates
                    else None
                )
            )
            fields["case_id"] = ResolvedField(
                field_name="case_id",
                state=FieldState.RESOLVED,
                value=linked_case.case_id,
                winning_evidence=case_winner,
                considered=case_candidates,
                reason="active case association",
                fusion_trace=(
                    fields["case_id"].fusion_trace
                    if "case_id" in fields
                    else None
                ),
            )
        if linked_case.active_applicant:
            applicant_candidates = tuple(
                candidate
                for candidate in evidence
                if candidate.field_name == "applicant_name"
                and CaseLinker._is_clean_visible_candidate(candidate)
                and candidate.value in {
                    linked_case.active_applicant,
                    *linked_case.active_applicant_aliases,
                }
            )
            exact_candidates = tuple(
                candidate
                for candidate in applicant_candidates
                if candidate.value == linked_case.active_applicant
            )
            existing = fields.get("applicant_name")
            existing_winner = (
                existing.winning_evidence if existing is not None else None
            )
            if (
                existing is not None
                and existing.value == linked_case.active_applicant
                and existing_winner in exact_candidates
            ):
                winning_applicant = existing_winner
            elif exact_candidates:
                winning_applicant = min(
                    exact_candidates,
                    key=lambda candidate: (
                        self._hierarchy.rank(candidate.evidence_type),
                        not candidate_has_complete_provenance(
                            candidate,
                            expected_case_id=linked_case.case_id or None,
                            active_applicant=linked_case.active_applicant,
                            active_applicant_aliases=(
                                linked_case.active_applicant_aliases
                            ),
                        ),
                        -candidate.ocr_confidence,
                        CaseLinker._candidate_sort_key(candidate),
                    ),
                )
            else:
                winning_applicant = None
            fields["applicant_name"] = ResolvedField(
                field_name="applicant_name",
                state=(
                    FieldState.RESOLVED
                    if winning_applicant is not None
                    else FieldState.UNKNOWN
                ),
                value=(
                    linked_case.active_applicant
                    if winning_applicant is not None
                    else None
                ),
                winning_evidence=winning_applicant,
                considered=applicant_candidates,
                reason=(
                    "active applicant association"
                    if winning_applicant is not None
                    else "active applicant has no surviving exact evidence"
                ),
                fusion_trace=(
                    fields["applicant_name"].fusion_trace
                    if "applicant_name" in fields
                    else None
                ),
            )

    @staticmethod
    def _winner_has_complete_provenance(
        field: ResolvedField,
        linked_case: LinkedCase,
    ) -> bool:
        winner = field.winning_evidence
        return winner is not None and candidate_has_complete_provenance(
            winner,
            expected_case_id=linked_case.case_id or None,
            active_applicant=linked_case.active_applicant,
            active_applicant_aliases=linked_case.active_applicant_aliases,
        )

    @staticmethod
    def _is_clean_legacy_winner(field: ResolvedField) -> bool:
        winner = field.winning_evidence
        if winner is None:
            return False
        cues = {
            re.sub(r"[\s-]+", "_", cue.strip().casefold())
            for cue in winner.visual_cues
        }
        return (
            winner.legible
            and winner.value is not None
            and not winner.superseded
            and winner.source == "visible_ocr"
            and "strikethrough" not in cues
            and not any("watermark" in cue for cue in cues)
            and winner.evidence_type is not EvidenceType.TEXT_LAYER
        )

    def _fusion_audit_counts(
        self,
        linked_case: LinkedCase,
        fusion_fields: Mapping[str, ResolvedField],
        legacy_fields: Mapping[str, ResolvedField],
    ) -> Mapping[str, int]:
        counts = {
            "changed_field_count": 0,
            "changed_field_complete_provenance_count": 0,
            "clean_higher_authority_override_count": 0,
            "binding_authority_override_count": 0,
            "text_layer_winner_count": 0,
            "serialization_default_used_as_evidence_count": 0,
            "correlated_views_collapsed": 0,
            "independent_agreement_resolutions": 0,
            "same_rank_contested_count": 0,
            "cross_applicant_candidates_excluded": (
                linked_case.cross_applicant_candidates_excluded
            ),
        }
        for field_name in RESOLVABLE_FIELDS:
            fusion = fusion_fields[field_name]
            legacy = legacy_fields[field_name]
            trace = fusion.fusion_trace
            if trace is not None:
                counts["correlated_views_collapsed"] += (
                    trace.correlated_candidate_count
                )
                counts["serialization_default_used_as_evidence_count"] += (
                    trace.safety_count(
                        "serialization_default_used_as_evidence_count"
                    )
                )
                counts["independent_agreement_resolutions"] += int(
                    fusion.state is FieldState.RESOLVED
                    and trace.independent_agreement_count >= 2
                )
                counts["same_rank_contested_count"] += int(
                    fusion.state is FieldState.CONTESTED
                )

            changed = (
                fusion.state is not legacy.state
                or fusion.value != legacy.value
            )
            if not changed:
                continue
            counts["changed_field_count"] += 1
            counts["changed_field_complete_provenance_count"] += int(
                self._winner_has_complete_provenance(fusion, linked_case)
            )
            counts["text_layer_winner_count"] += int(
                fusion.winning_evidence is not None
                and fusion.winning_evidence.evidence_type
                is EvidenceType.TEXT_LAYER
            )

            if (
                legacy.winning_evidence is not None
                and self._is_clean_legacy_winner(legacy)
            ):
                legacy_rank = self._hierarchy.rank(
                    legacy.winning_evidence.evidence_type
                )
                if fusion.winning_evidence is None:
                    continue
                fusion_rank = self._hierarchy.rank(
                    fusion.winning_evidence.evidence_type
                )
                counts["binding_authority_override_count"] += int(
                    legacy_rank == 1
                    and (
                        fusion_rank != 1
                        or fusion.value != legacy.value
                    )
                )
                counts["clean_higher_authority_override_count"] += int(
                    fusion_rank > legacy_rank
                )
        return MappingProxyType(dict(sorted(counts.items())))

    def resolve(self, linked_case: LinkedCase) -> ResolvedCase:
        evidence, rescinded = self._rescinded.filter(
            linked_case.evidence,
            expected_case_id=linked_case.case_id or None,
            active_applicant=linked_case.active_applicant,
            active_applicant_aliases=linked_case.active_applicant_aliases,
        )
        legacy_fields: dict[str, ResolvedField] = {}
        if self._fusion_enabled:
            fields = {
                field_name: self._resolve_field_with_fusion(
                    field_name,
                    evidence,
                    linked_case,
                )
                for field_name in RESOLVABLE_FIELDS
            }
            legacy_fields = {
                field_name: (
                    self._structured_sponsor_conflict_repair(
                        field_name,
                        linked_case.case_id,
                        evidence,
                    )
                    or self._resolve_field(field_name, evidence)
                )
                for field_name in RESOLVABLE_FIELDS
            }
        else:
            fields = {
                field_name: (
                    self._structured_sponsor_conflict_repair(
                        field_name,
                        linked_case.case_id,
                        evidence,
                    )
                    or self._resolve_field(field_name, evidence)
                )
                for field_name in RESOLVABLE_FIELDS
            }

        self._with_case_and_applicant_associations(
            linked_case,
            evidence,
            fields,
        )
        if self._fusion_enabled:
            self._with_case_and_applicant_associations(
                linked_case,
                evidence,
                legacy_fields,
            )
            audit_counts = self._fusion_audit_counts(
                linked_case,
                fields,
                legacy_fields,
            )
        else:
            audit_counts = MappingProxyType({})

        return ResolvedCase(
            case_id=linked_case.case_id,
            active_applicant=linked_case.active_applicant,
            fields=fields,
            unresolved_linkage=linked_case.unresolved,
            unresolved_reasons=linked_case.unresolved_reasons,
            rescinded_decision=rescinded,
            fusion_audit_counts=audit_counts,
        )
