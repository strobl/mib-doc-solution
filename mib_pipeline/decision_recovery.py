"""Frozen, identity-free recovery of a narrow subset of review decisions.

The normal policy engine is always evaluated first.  Synthetic recovery rules
are a separately inspectable stage and, when trusted late evidence changes the
resolved case, both stages are run again from the recovered evidence.  This
prevents a synthetic reason computed from an earlier evidence gap from
surviving after that gap has been repaired.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from types import MappingProxyType
from typing import Mapping, Protocol

from .adjudication import AdjudicationOutcome, PolicyRuleSet
from .extraction import EvidenceType
from .models import PredictionRow
from .resolution import FieldState, ResolvedCase


REVIEW_DENIAL_CONFIDENCE = 0.551819438046983
REVIEW_APPROVAL_CONFIDENCE = 0.98
# Kept as a public compatibility alias for the first approval recovery head.
REVIEW_DIPLOMATIC_APPROVAL_CONFIDENCE = REVIEW_APPROVAL_CONFIDENCE
_MAXIMUM_BASELINE_CONFIDENCE = 0.35
_MAXIMUM_DIPLOMATIC_APPROVAL_BASELINE_CONFIDENCE = 0.25
_MAXIMUM_SPONSOR_XW1_APPROVAL_BASELINE_CONFIDENCE = 0.20
_MINIMUM_CURRENT_VISA_UNKNOWN_APPROVAL_BASELINE_CONFIDENCE = 0.20
_MAXIMUM_CURRENT_VISA_UNKNOWN_APPROVAL_BASELINE_CONFIDENCE = 0.25
_FEE_RECEIPT_MARKER = "page_type_present_fee_receipt"
_SPONSOR_ATTESTATION_MARKER = "page_type_present_sponsor_attestation"
_MARKER_CUES = {
    _FEE_RECEIPT_MARKER: "packet_page_type:fee_receipt",
    _SPONSOR_ATTESTATION_MARKER: "packet_page_type:sponsor_attestation",
}
SYNTHETIC_DENIAL_REASONS = frozenset(
    {
        "review_denial_other_missing_biohazard",
        "review_denial_sponsor_stale_gt180",
        "review_denial_three_required_outputs_unknown",
    }
)
POLICY_AUDIT_COUNT_NAMES = (
    "late_recovery_before_revalidation_count",
    "contradicted_synthetic_reason_removed_count",
    "independent_denial_reason_retained_count",
    "review_confidence_restored_count",
    "normal_policy_rerun_count",
    "signed_late_authority_recovery_count",
    "late_adjudication_evidence_preserved_count",
    "late_biohazard_evidence_preserved_count",
    "forced_approval_count",
    "serialization_default_used_as_policy_evidence_count",
    "sentinel_value_used_as_policy_evidence_count",
    "placeholder_value_used_as_evidence_count",
    "stale_threshold_mismatch_count",
    "contradicted_synthetic_reason_left_active_count",
)


def empty_policy_audit_counts() -> Mapping[str, int]:
    """Return the complete immutable zero-valued WO-17 audit contract."""

    return MappingProxyType({name: 0 for name in POLICY_AUDIT_COUNT_NAMES})


def _frozen_policy_audit_counts(
    counts: Mapping[str, int],
) -> Mapping[str, int]:
    """Validate and freeze one complete policy-order audit mapping."""

    copied = dict(counts)
    if set(copied) != set(POLICY_AUDIT_COUNT_NAMES):
        raise ValueError("policy audit counts do not match the frozen contract")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        for value in copied.values()
    ):
        raise ValueError("policy audit counts must be non-negative integers")
    return MappingProxyType(dict(sorted(copied.items())))


@dataclass(frozen=True)
class StagedAdjudication:
    """Normal-policy and post-synthetic outcomes for one resolved case."""

    policy_outcome: AdjudicationOutcome
    outcome: AdjudicationOutcome
    audit_counts: Mapping[str, int] = field(
        default_factory=empty_policy_audit_counts
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_counts",
            _frozen_policy_audit_counts(self.audit_counts),
        )


@dataclass(frozen=True)
class RevalidatedAdjudication:
    """Accepted post-recovery outcome and aggregate-only ordering audit."""

    outcome: AdjudicationOutcome
    audit_counts: Mapping[str, int] = field(
        default_factory=empty_policy_audit_counts
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "audit_counts",
            _frozen_policy_audit_counts(self.audit_counts),
        )


class OutcomeAdjudicator(Protocol):
    def adjudicate_case(self, resolved_case: ResolvedCase) -> AdjudicationOutcome:
        """Return the baseline policy outcome and trace."""


class ReviewDenialRecoveryAdjudicator:
    """Apply frozen identity-free recovery rules to baseline reviews.

    The wrapper consumes only generic policy trace categories, bounded output
    categories (arrival age and visa class), the baseline confidence, and
    packet-level visible page-type markers.  Case IDs, names, filenames, and
    truth labels are never inspected.  All row values other than
    adjudication/confidence remain byte-for-byte equivalent at the typed model
    boundary.
    """

    def __init__(
        self,
        baseline: OutcomeAdjudicator,
        *,
        rules: PolicyRuleSet | None = None,
    ) -> None:
        self._baseline = baseline
        self._rules = rules or PolicyRuleSet()

    @staticmethod
    def _visible_marker(resolved_case: ResolvedCase, field_name: str) -> bool:
        field = resolved_case.fields.get(field_name)
        evidence = field.winning_evidence if field is not None else None
        return bool(
            field is not None
            and field.state is FieldState.RESOLVED
            and field.value == "present"
            and evidence is not None
            and evidence.legible
            and not evidence.superseded
            and evidence.source == "visible_ocr"
            and evidence.evidence_type is not EvidenceType.TEXT_LAYER
            and "strikethrough" not in evidence.visual_cues
            and "sample_denial_watermark" not in evidence.visual_cues
            and _MARKER_CUES.get(field_name) in evidence.visual_cues
            and evidence.case_id_hint
            in {None, resolved_case.case_id}
            and evidence.applicant_hint
            in {None, resolved_case.active_applicant}
        )

    @classmethod
    def _visible_marker_page_count(
        cls,
        resolved_case: ResolvedCase,
        field_name: str,
    ) -> int:
        field = resolved_case.fields.get(field_name)
        if not cls._visible_marker(resolved_case, field_name) or field is None:
            return 0
        cue = _MARKER_CUES.get(field_name)
        pages = {
            evidence.page_index
            for evidence in field.considered
            if evidence.value == "present"
            and evidence.legible
            and not evidence.superseded
            and evidence.source == "visible_ocr"
            and evidence.evidence_type is not EvidenceType.TEXT_LAYER
            and "strikethrough" not in evidence.visual_cues
            and "sample_denial_watermark" not in evidence.visual_cues
            and cue in evidence.visual_cues
            and evidence.case_id_hint
            in {None, resolved_case.case_id}
            and evidence.applicant_hint
            in {None, resolved_case.active_applicant}
        }
        return len(pages)

    @staticmethod
    def _visible_date(
        resolved_case: ResolvedCase,
        field_name: str,
    ) -> date | None:
        """Parse one live, scoped date while rejecting schema sentinels."""

        field = resolved_case.fields.get(field_name)
        evidence = field.winning_evidence if field is not None else None
        if (
            field is None
            or field.state is not FieldState.RESOLVED
            or field.value in {None, "1900-01-01"}
            or evidence is None
            or not evidence.legible
            or evidence.superseded
            or evidence.source != "visible_ocr"
            or evidence.evidence_type is EvidenceType.TEXT_LAYER
            or "strikethrough" in evidence.visual_cues
            or "sample_denial_watermark" in evidence.visual_cues
            or evidence.case_id_hint not in {None, resolved_case.case_id}
            or evidence.applicant_hint
            not in {None, resolved_case.active_applicant}
        ):
            return None
        try:
            return date.fromisoformat(field.value)
        except ValueError:
            return None

    def _arrival_is_stale(
        self,
        resolved_case: ResolvedCase,
    ) -> bool:
        """Mirror normal policy's visible receipt and exact 180-day rule."""

        arrival = self._visible_date(resolved_case, "arrival_date")
        if arrival is None:
            return False
        effective_receipt = (
            self._visible_date(resolved_case, "packet_receipt_date")
            or self._rules.snapshot_receipt_date
        )
        return (
            effective_receipt - arrival
        ).days > self._rules.stale_after_days

    @staticmethod
    def _visible_diplomatic_visa(resolved_case: ResolvedCase) -> bool:
        field = resolved_case.fields.get("visa_class")
        evidence = field.winning_evidence if field is not None else None
        return bool(
            field is not None
            and field.state is FieldState.RESOLVED
            and field.value == "DIP-1"
            and evidence is not None
            and evidence.legible
            and not evidence.superseded
            and evidence.source == "visible_ocr"
            and evidence.evidence_type is not EvidenceType.TEXT_LAYER
            and "strikethrough" not in evidence.visual_cues
            and "sample_denial_watermark" not in evidence.visual_cues
            and evidence.case_id_hint in {None, resolved_case.case_id}
            and evidence.applicant_hint
            in {None, resolved_case.active_applicant}
        )

    @classmethod
    def _matches_diplomatic_fee_approval(
        cls,
        resolved_case: ResolvedCase,
        outcome: AdjudicationOutcome,
    ) -> bool:
        if outcome.row.confidence > _MAXIMUM_DIPLOMATIC_APPROVAL_BASELINE_CONFIDENCE:
            return False
        if outcome.trace.denial_reasons:
            return False
        diplomatic_fact = (
            "diplomatic_sponsor_exemption" in outcome.trace.approval_facts
        )
        return (
            cls._visible_marker_page_count(
                resolved_case,
                _FEE_RECEIPT_MARKER,
            )
            == 1
            and (diplomatic_fact or cls._visible_diplomatic_visa(resolved_case))
        )

    @classmethod
    def _matching_rules(
        cls,
        resolved_case: ResolvedCase,
        outcome: AdjudicationOutcome,
        *,
        arrival_is_stale: bool,
    ) -> tuple[str, ...]:
        reasons = frozenset(outcome.trace.review_reasons)
        low_confidence = outcome.row.confidence <= _MAXIMUM_BASELINE_CONFIDENCE
        matches: list[str] = []
        # ``page_type_present_other`` is produced by the page classifier's
        # fallback branch for an unrecognized or even empty heading.  It is
        # diagnostic topology, not affirmative policy evidence, and therefore
        # cannot support the historical missing-biohazard synthetic denial.
        if (
            low_confidence
            and arrival_is_stale
            and cls._visible_marker(resolved_case, _SPONSOR_ATTESTATION_MARKER)
        ):
            matches.append("review_denial_sponsor_stale_gt180")
        # Missing output topology is not affirmative evidence of a policy
        # violation.  The former three-gap rule converted an evidence-poor
        # review into a denial without any visible disqualifier.  Keep its
        # reason token in ``SYNTHETIC_DENIAL_REASONS`` solely so revalidation
        # can remove it from historical staged outcomes; never emit it again.
        return tuple(matches)

    @classmethod
    def _matching_approval_rules(
        cls,
        resolved_case: ResolvedCase,
        outcome: AdjudicationOutcome,
    ) -> tuple[str, ...]:
        """Return frozen approval heads in stable evaluation order."""

        # An approval recovery must never erase an explicit policy denial,
        # including one supplied by a malformed/custom baseline.
        if outcome.trace.denial_reasons:
            return ()
        # Direct review-to-approval heads may never consume row serialization
        # fallbacks.  Every scored output must already have a substantive,
        # visible, scoped winner; otherwise late evidence recovery and normal
        # policy adjudication are the only permitted path to approval.
        if any(
            cls._raw_output_kind(resolved_case, field_name)
            != "substantive"
            for field_name in (
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
        ):
            return ()

        confidence = outcome.row.confidence
        reasons = frozenset(outcome.trace.review_reasons)
        facts = frozenset(outcome.trace.approval_facts)
        current_application = "application_date_current_or_exempt" in facts
        matches: list[str] = []

        if cls._matches_diplomatic_fee_approval(resolved_case, outcome):
            matches.append("review_diplomatic_fee_receipt_recovery")
        if (
            confidence <= _MAXIMUM_SPONSOR_XW1_APPROVAL_BASELINE_CONFIDENCE
            and cls._visible_visa_class(resolved_case) == "XW-1"
            and cls._visible_marker(
                resolved_case,
                _SPONSOR_ATTESTATION_MARKER,
            )
        ):
            matches.append("review_approval_sponsor_attestation_xw1")
        if (
            _MINIMUM_CURRENT_VISA_UNKNOWN_APPROVAL_BASELINE_CONFIDENCE
            < confidence
            <= _MAXIMUM_CURRENT_VISA_UNKNOWN_APPROVAL_BASELINE_CONFIDENCE
            and current_application
            and "visa_class_unknown" in reasons
        ):
            matches.append("review_approval_current_application_visa_unknown")
        if (
            confidence <= _MAXIMUM_BASELINE_CONFIDENCE
            and current_application
            and "required_output_unknown:home_world" in reasons
            and "unsupported_fee_waiver" not in reasons
        ):
            matches.append("review_approval_current_application_home_world_unknown")
        return tuple(matches)

    @staticmethod
    def _visible_visa_class(resolved_case: ResolvedCase) -> str | None:
        """Return a trusted visa value, never the serialized schema fallback."""

        field = resolved_case.fields.get("visa_class")
        evidence = field.winning_evidence if field is not None else None
        if (
            field is None
            or field.state is not FieldState.RESOLVED
            or field.value is None
            or evidence is None
            or not evidence.legible
            or evidence.superseded
            or evidence.source != "visible_ocr"
            or evidence.evidence_type is EvidenceType.TEXT_LAYER
            or "strikethrough" in evidence.visual_cues
            or "sample_denial_watermark" in evidence.visual_cues
            or evidence.case_id_hint not in {None, resolved_case.case_id}
            or evidence.applicant_hint
            not in {None, resolved_case.active_applicant}
        ):
            return None
        return field.value

    @staticmethod
    def _raw_output_kind(
        resolved_case: ResolvedCase,
        field_name: str,
    ) -> str:
        """Classify one output as substantive, default, sentinel, or placeholder."""

        field = resolved_case.fields.get(field_name)
        if (
            field is None
            or field.state is not FieldState.RESOLVED
            or not isinstance(field.value, str)
        ):
            return "default"
        normalized = " ".join(field.value.strip().split()).casefold()
        if (
            (field_name == "sponsor_id" and normalized == "spn-0000")
            or (
                field_name == "arrival_date"
                and normalized == "1900-01-01"
            )
        ):
            return "sentinel"
        if (
            normalized in {"", "unknown", "null", "other"}
            or (
                normalized == "none"
                and field_name != "risk_flags"
            )
        ):
            return "placeholder"
        evidence = field.winning_evidence
        if (
            evidence is None
            or not evidence.legible
            or evidence.superseded
            or evidence.source != "visible_ocr"
            or evidence.evidence_type is EvidenceType.TEXT_LAYER
            or "strikethrough" in evidence.visual_cues
            or "sample_denial_watermark" in evidence.visual_cues
            or "synthetic_default" in evidence.visual_cues
            or evidence.case_id_hint
            not in {None, resolved_case.case_id}
            or evidence.applicant_hint
            not in {None, resolved_case.active_applicant}
        ):
            return "default"
        return "substantive"

    @staticmethod
    def _field_consumed_by_policy(
        outcome: AdjudicationOutcome,
        field_name: str,
    ) -> bool:
        """Conservatively detect a value-specific policy fact or strict approval."""

        trace = outcome.trace
        if trace.authoritative_source:
            return False
        denial = trace.denial_reasons
        review = trace.review_reasons
        approval = trace.approval_facts
        if "strict_approval_bar_cleared" in approval:
            if (
                field_name == "sponsor_id"
                and "diplomatic_sponsor_exemption" in approval
            ):
                return False
            return True
        signals = {
            "sponsor_id": (
                "barred_sponsor:",
                "sponsor_present_and_not_publicly_barred",
            ),
            "arrival_date": (
                "stale_application",
                "stale_diplomatic_note_",
                "application_date_current_or_exempt",
            ),
            "risk_flags": (
                "disqualifying_flag:",
                "review_flag:",
                "no_visible_biohazard_risk",
            ),
            "home_world": ("embargoed_home_world:",),
            "visa_class": (
                "transit_work_authorization",
                "diplomatic_sponsor_exemption",
                "stay_limit_exceeded:",
                "stay_within_visa_limit",
            ),
            "fee_status": (
                "unpaid_without_valid_waiver",
                "fee_paid",
                "valid_fee_waiver",
            ),
        }
        combined = (*denial, *review, *approval)
        return any(
            any(item.startswith(signal) for signal in signals.get(field_name, ()))
            for item in combined
        )

    def _policy_safety_counts(
        self,
        resolved_case: ResolvedCase,
        staged: StagedAdjudication,
        *,
        monitor_forced_approval: bool,
    ) -> Mapping[str, int]:
        """Measure unsafe final-policy behavior; no counter is a constant claim."""

        counts = dict(empty_policy_audit_counts())
        if monitor_forced_approval:
            counts["forced_approval_count"] = int(
                staged.outcome.row.adjudication == "APPROVED"
                and staged.outcome.trace.decision == "APPROVED"
                and (
                    staged.policy_outcome.row.adjudication != "APPROVED"
                    or staged.policy_outcome.trace.decision != "APPROVED"
                )
            )

        for field_name in (
            "applicant_name",
            "species_code",
            "home_world",
            "visa_class",
            "sponsor_id",
            "arrival_date",
            "declared_purpose",
            "risk_flags",
            "fee_status",
        ):
            if not self._field_consumed_by_policy(
                staged.policy_outcome,
                field_name,
            ):
                continue
            kind = self._raw_output_kind(
                resolved_case,
                field_name,
            )
            if kind == "default":
                counts[
                    "serialization_default_used_as_policy_evidence_count"
                ] += 1
            elif kind == "sentinel":
                counts[
                    "sentinel_value_used_as_policy_evidence_count"
                ] += 1
            elif kind == "placeholder":
                counts[
                    "placeholder_value_used_as_evidence_count"
                ] += 1

        expected_stale = bool(
            staged.policy_outcome.row.adjudication == "NEEDS_REVIEW"
            and staged.policy_outcome.trace.decision == "NEEDS_REVIEW"
            and staged.policy_outcome.row.confidence
            <= _MAXIMUM_BASELINE_CONFIDENCE
            and self._arrival_is_stale(resolved_case)
            and self._visible_marker(
                resolved_case,
                _SPONSOR_ATTESTATION_MARKER,
            )
        )
        active_stale = (
            "review_denial_sponsor_stale_gt180"
            in staged.outcome.trace.denial_reasons
        )
        counts["stale_threshold_mismatch_count"] = int(
            expected_stale != active_stale
        )
        return _frozen_policy_audit_counts(counts)

    def _apply_synthetic_rules(
        self,
        resolved_case: ResolvedCase,
        baseline: AdjudicationOutcome,
        *,
        allow_approval_recovery: bool = True,
    ) -> AdjudicationOutcome:
        """Apply the frozen synthetic stage to one fresh policy outcome."""

        # Requiring both typed outputs to agree makes a malformed/custom
        # baseline fail closed instead of broadening the override surface.
        if (
            baseline.row.adjudication != "NEEDS_REVIEW"
            or baseline.trace.decision != "NEEDS_REVIEW"
        ):
            return baseline
        matching_rules = self._matching_rules(
            resolved_case,
            baseline,
            arrival_is_stale=self._arrival_is_stale(resolved_case),
        )
        if matching_rules:
            trace = replace(
                baseline.trace,
                decision="DENIED",
                authoritative_source=False,
                denial_reasons=tuple(
                    sorted(set(baseline.trace.denial_reasons) | set(matching_rules))
                ),
            )
            row = replace(
                baseline.row,
                adjudication="DENIED",
                confidence=REVIEW_DENIAL_CONFIDENCE,
            )
            return AdjudicationOutcome(row=row, trace=trace)

        # WO-17 forbids direct REVIEW -> APPROVED rewrites.  Approval now
        # requires recovered visible evidence followed by ordinary policy
        # adjudication (or a binding signed authoritative decision).
        del allow_approval_recovery
        return baseline

    def adjudicate_staged(
        self,
        resolved_case: ResolvedCase,
    ) -> StagedAdjudication:
        """Expose normal-policy-before-synthetic order without hidden state."""

        policy_outcome = self._baseline.adjudicate_case(resolved_case)
        staged = StagedAdjudication(
            policy_outcome=policy_outcome,
            outcome=self._apply_synthetic_rules(
                resolved_case,
                policy_outcome,
            ),
        )
        return StagedAdjudication(
            policy_outcome=staged.policy_outcome,
            outcome=staged.outcome,
            audit_counts=self._policy_safety_counts(
                resolved_case,
                staged,
                monitor_forced_approval=True,
            ),
        )

    def adjudicate_case(self, resolved_case: ResolvedCase) -> AdjudicationOutcome:
        return self.adjudicate_staged(resolved_case).outcome

    def revalidate_after_recovery(
        self,
        resolved_case: ResolvedCase,
        *,
        original: StagedAdjudication,
    ) -> RevalidatedAdjudication:
        """Rerun normal policy and synthetic rules after accepted late evidence.

        The result is rebuilt from the recovered resolved case.  Consequently,
        an old synthetic reason survives only when its predicate still holds,
        while independent normal-policy reasons are retained by the ordinary
        adjudicator.  Returning to review restores the confidence that the
        normal policy assigned before the pre-recovery synthetic override.
        """

        revalidated_policy = self._baseline.adjudicate_case(resolved_case)
        revalidated = StagedAdjudication(
            policy_outcome=revalidated_policy,
            outcome=self._apply_synthetic_rules(
                resolved_case,
                revalidated_policy,
                allow_approval_recovery=False,
            ),
        )
        original_synthetic = (
            set(original.outcome.trace.denial_reasons)
            & SYNTHETIC_DENIAL_REASONS
        )
        final_synthetic = (
            set(revalidated.outcome.trace.denial_reasons)
            & SYNTHETIC_DENIAL_REASONS
        )
        contradicted = original_synthetic - final_synthetic
        # Independent-reason carryover belongs only to the narrow malformed
        # REVIEW -> synthetic-DENIED shape.  A genuine original policy denial
        # is rerun normally, and a recovered binding signed decision remains
        # absolute over that old non-authoritative result.
        synthetic_override_of_review = bool(
            original_synthetic
            and original.policy_outcome.row.adjudication == "NEEDS_REVIEW"
            and original.policy_outcome.trace.decision == "NEEDS_REVIEW"
            and original.outcome.row.adjudication == "DENIED"
            and original.outcome.trace.decision == "DENIED"
        )
        independent_original = (
            set(original.outcome.trace.denial_reasons)
            - SYNTHETIC_DENIAL_REASONS
            if synthetic_override_of_review
            else set()
        )
        final_outcome = revalidated.outcome
        missing_independent = independent_original.difference(
            final_outcome.trace.denial_reasons
        )
        if missing_independent:
            # A malformed/custom normal-policy implementation can emit a
            # review decision carrying an independent denial reason.  Late
            # recovery is never permitted to erase that independent reason:
            # preserve it and fail closed while still dropping any
            # contradicted synthetic reasons.
            final_outcome = AdjudicationOutcome(
                row=replace(
                    final_outcome.row,
                    adjudication="DENIED",
                    confidence=original.outcome.row.confidence,
                ),
                trace=replace(
                    final_outcome.trace,
                    decision="DENIED",
                    authoritative_source=False,
                    denial_reasons=tuple(
                        sorted(
                            set(final_outcome.trace.denial_reasons)
                            | independent_original
                        )
                    ),
                ),
            )
        independent_retained = independent_original.intersection(
            final_outcome.trace.denial_reasons
        )
        confidence_restored = bool(
            original.policy_outcome.row.adjudication == "NEEDS_REVIEW"
            and original.policy_outcome.trace.decision == "NEEDS_REVIEW"
            and original.outcome.row.adjudication == "DENIED"
            and original.outcome.trace.decision == "DENIED"
            and final_outcome.row.adjudication == "NEEDS_REVIEW"
            and final_outcome.trace.decision == "NEEDS_REVIEW"
            and not final_outcome.trace.authoritative_source
            and "authoritative_visible_decision"
            not in final_outcome.trace.review_reasons
        )
        if confidence_restored:
            final_outcome = AdjudicationOutcome(
                row=replace(
                    final_outcome.row,
                    confidence=original.policy_outcome.row.confidence,
                ),
                trace=final_outcome.trace,
            )

        currently_matching = set()
        if (
            revalidated.policy_outcome.row.adjudication == "NEEDS_REVIEW"
            and revalidated.policy_outcome.trace.decision == "NEEDS_REVIEW"
        ):
            currently_matching = set(
                self._matching_rules(
                    resolved_case,
                    revalidated.policy_outcome,
                    arrival_is_stale=self._arrival_is_stale(
                        resolved_case
                    ),
                )
            )
        invalid_active = final_synthetic - currently_matching
        counts = dict(
            self._policy_safety_counts(
                resolved_case,
                StagedAdjudication(
                    policy_outcome=revalidated.policy_outcome,
                    outcome=final_outcome,
                ),
                monitor_forced_approval=True,
            )
        )
        counts.update(
            {
                "late_recovery_before_revalidation_count": 1,
                "contradicted_synthetic_reason_removed_count": len(
                    contradicted
                ),
                "independent_denial_reason_retained_count": len(
                    independent_retained
                ),
                "review_confidence_restored_count": int(
                    confidence_restored
                ),
                "normal_policy_rerun_count": 1,
                "contradicted_synthetic_reason_left_active_count": len(
                    invalid_active
                ),
            }
        )
        return RevalidatedAdjudication(
            outcome=final_outcome,
            audit_counts=counts,
        )

    def adjudicate(self, resolved_case: ResolvedCase) -> PredictionRow:
        return self.adjudicate_case(resolved_case).row
