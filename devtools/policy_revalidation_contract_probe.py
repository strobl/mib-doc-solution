"""Identity-free executable contract probes for WO-17.

These probes complement, but never replace, the frozen production cohort.
They deliberately exercise rare branches whose natural cohort occurrence may
be zero.  A failed assertion aborts evidence generation; counts are returned
only after the real policy/recovery methods produce the required outcomes.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from devtools.policy_revalidation_audit_contract import CONTRACT_AUDIT_COUNTS
from mib_pipeline.adjudication import (
    AdjudicationEngine,
    AdjudicationOutcome,
    DecisionTrace,
    PolicyRuleSet,
)
from mib_pipeline.decision_recovery import (
    REVIEW_DENIAL_CONFIDENCE,
    ReviewDenialRecoveryAdjudicator,
    StagedAdjudication,
)
from mib_pipeline.extraction import CandidateEvidence, EvidenceType
from mib_pipeline.ingestion import Rect
from mib_pipeline.models import PredictionRow
from mib_pipeline.provenance import make_ocr_provenance
from mib_pipeline.rapid_recovery import RapidOutputRecoveryProcessor
from mib_pipeline.resolution import (
    CaseLinker,
    EvidencePrecedenceResolver,
    FieldState,
    ResolvedCase,
    ResolvedField,
)


_CASE = "MIB-000001"
_APPLICANT = "Contract Probe"
_SOURCE_SHA256 = "0" * 64


class PolicyContractProbeError(RuntimeError):
    """One required policy behavior was not observed."""


def contract_fixture_sha256() -> str:
    """Bind evidence to the exact executable probe source bytes."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _candidate(
    field_name: str,
    value: str,
    *,
    evidence_type: EvidenceType = EvidenceType.INTAKE_FORM,
    cues: tuple[str, ...] = (),
    page_index: int = 0,
    left: float = 1,
) -> CandidateEvidence:
    box = Rect(left, 2, left + 100, 32)
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=page_index,
        box=box,
        legible=True,
        superseded=False,
        ocr_confidence=0.99,
        visual_cues=cues,
        source="visible_ocr",
        case_id_hint=_CASE,
        applicant_hint=_APPLICANT,
        ocr_provenance=(
            make_ocr_provenance(
                source_sha256=_SOURCE_SHA256,
                page_index=page_index,
                view_box=box,
                applicant_scope=_APPLICANT,
                route_id="targeted_rapidocr",
                engine_id="contract:ocr",
                view_id="contract_probe",
            ),
        ),
    )


class _RenderedContractCase:
    case_id = _CASE
    source_sha256 = _SOURCE_SHA256


class _ContractRenderer:
    def render(self, _path: Path) -> _RenderedContractCase:
        return _RenderedContractCase()


class _ContractExtractor:
    def __init__(self, candidates: tuple[CandidateEvidence, ...]) -> None:
        self._candidates = candidates

    def extract(
        self, _rendered: _RenderedContractCase
    ) -> tuple[CandidateEvidence, ...]:
        return self._candidates


def _output_candidates(
    *,
    omit: frozenset[str] = frozenset(),
    overrides: Mapping[str, str] | None = None,
) -> tuple[CandidateEvidence, ...]:
    values = list(
        (
        ("applicant_name", _APPLICANT),
        ("species_code", "HUM"),
        ("home_world", "Earth"),
        ("visa_class", "XW-2"),
        ("sponsor_id", "SPN-1042"),
        ("arrival_date", "2026-04-17"),
        ("declared_purpose", "contract probe"),
        ("risk_flags", "none"),
        ("fee_status", "paid"),
        )
    )
    replacements = dict(overrides or {})
    return tuple(
        _candidate(
            field_name,
            replacements.get(field_name, value),
            page_index=index,
            left=float(10 + 130 * index),
        )
        for index, (field_name, value) in enumerate(values)
        if field_name not in omit
    )


def _run_full_fusion_processor(
    *,
    primary_candidates: tuple[CandidateEvidence, ...],
    rapid_candidates: tuple[CandidateEvidence, ...],
) -> object:
    """Exercise the real linker, resolver, wrapper, and accepted-final path."""

    processor = RapidOutputRecoveryProcessor(
        renderer=_ContractRenderer(),
        primary_extractor=_ContractExtractor(primary_candidates),
        linker=CaseLinker(),
        resolver=EvidencePrecedenceResolver(),
        adjudicator=ReviewDenialRecoveryAdjudicator(
            AdjudicationEngine(default_confidence=0.23)
        ),
        rapid_extractor_factory=lambda: _ContractExtractor(
            rapid_candidates
        ),
    )
    return processor.process_case_with_audit(Path(f"{_CASE}.pdf"))


def _staged_output_decision(
    candidates: tuple[CandidateEvidence, ...],
) -> str:
    _resolved_case, staged = _staged_output(candidates)
    return staged.outcome.row.adjudication


def _staged_output(
    candidates: tuple[CandidateEvidence, ...],
) -> tuple[ResolvedCase, object]:
    resolved = EvidencePrecedenceResolver().resolve(
        CaseLinker().link(_CASE, candidates)
    )
    staged = ReviewDenialRecoveryAdjudicator(
        AdjudicationEngine(default_confidence=0.23)
    ).adjudicate_staged(resolved)
    return resolved, staged


def _field(candidate: CandidateEvidence) -> ResolvedField:
    return ResolvedField(
        field_name=candidate.field_name,
        state=FieldState.RESOLVED,
        value=candidate.value,
        winning_evidence=candidate,
        considered=(candidate,),
        reason="contract probe",
    )


def _resolved(*candidates: CandidateEvidence) -> ResolvedCase:
    return ResolvedCase(
        case_id=_CASE,
        active_applicant=_APPLICANT,
        fields={
            candidate.field_name: _field(candidate)
            for candidate in candidates
        },
        unresolved_linkage=False,
        unresolved_reasons=(),
    )


def _row(
    *,
    adjudication: str = "NEEDS_REVIEW",
    confidence: float = 0.23,
) -> PredictionRow:
    return PredictionRow.from_mapping(
        {
            "case_id": _CASE,
            "applicant_name": _APPLICANT,
            "species_code": "HUM",
            "home_world": "Earth",
            "visa_class": "XW-2",
            "sponsor_id": "SPN-1042",
            "arrival_date": "2026-04-17",
            "declared_purpose": "contract probe",
            "risk_flags": "none",
            "fee_status": "paid",
            "adjudication": adjudication,
            "confidence": confidence,
        }
    )


def _outcome(
    *,
    review_reasons: tuple[str, ...] = (),
    denial_reasons: tuple[str, ...] = (),
    confidence: float = 0.23,
    adjudication: str = "NEEDS_REVIEW",
) -> AdjudicationOutcome:
    row = _row(adjudication=adjudication, confidence=confidence)
    return AdjudicationOutcome(
        row=row,
        trace=DecisionTrace(
            decision=adjudication,
            authoritative_source=False,
            denial_reasons=denial_reasons,
            review_reasons=review_reasons,
            approval_facts=("fee_paid",),
            exception_ids=(),
        ),
    )


class _RecoveredOutputAwareBaseline:
    """Generic normal-policy stub whose reason follows recovered evidence."""

    def __init__(self, *, independent_denial: bool = False) -> None:
        self.independent_denial = independent_denial

    def adjudicate_case(
        self, resolved_case: ResolvedCase
    ) -> AdjudicationOutcome:
        required_fields = (
            "home_world",
            "risk_flags",
            "sponsor_id",
        )
        complete = all(
            (field := resolved_case.fields.get(field_name)) is not None
            and field.state is FieldState.RESOLVED
            and field.value is not None
            for field_name in required_fields
        )
        return _outcome(
            review_reasons=()
            if complete
            else tuple(
                f"required_output_unknown:{field_name}"
                for field_name in required_fields
            ),
            denial_reasons=(
                ("barred_sponsor:visible",)
                if self.independent_denial
                else ()
            ),
            confidence=0.23,
            adjudication=(
                "DENIED"
                if self.independent_denial and complete
                else "NEEDS_REVIEW"
            ),
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PolicyContractProbeError(message)


def run_contract_probes() -> Mapping[str, int]:
    """Execute all rare WO-17 branches and return aggregate counters."""

    counts = {name: 0 for name in CONTRACT_AUDIT_COUNTS}
    clean_biohazard = _candidate(
        "biohazard_check",
        "clean",
        evidence_type=EvidenceType.BIOMETRIC_SLIP,
    )
    recovered_home = _candidate("home_world", "Earth")
    recovered_risk = _candidate("risk_flags", "none")
    recovered_sponsor = _candidate("sponsor_id", "SPN-1042")

    adjudicator = ReviewDenialRecoveryAdjudicator(
        _RecoveredOutputAwareBaseline()
    )
    primary = _resolved()
    current = adjudicator.adjudicate_staged(primary)
    synthetic_reason = "review_denial_three_required_outputs_unknown"
    _require(
        current.policy_outcome.row.adjudication == "NEEDS_REVIEW"
        and current.outcome.row.adjudication == "NEEDS_REVIEW"
        and synthetic_reason not in current.outcome.trace.denial_reasons,
        "retired missingness-only denial was reintroduced",
    )
    # Preserve WO-17's backwards-compatibility probe by replaying the exact
    # historical staged shape.  Current production must not create this
    # missingness-only denial, but revalidation must still remove one from a
    # persisted/pre-upgrade staged record after trusted evidence arrives.
    original = StagedAdjudication(
        policy_outcome=current.policy_outcome,
        outcome=_outcome(
            denial_reasons=(synthetic_reason,),
            confidence=REVIEW_DENIAL_CONFIDENCE,
            adjudication="DENIED",
        ),
    )
    counts["legacy_synthetic_before_late_recovery_count"] += 1

    recovered = _resolved(
        recovered_home,
        recovered_risk,
        recovered_sponsor,
        clean_biohazard,
    )
    revalidated = adjudicator.revalidate_after_recovery(
        recovered, original=original
    )
    _require(
        revalidated.outcome.row.adjudication == "NEEDS_REVIEW"
        and synthetic_reason not in revalidated.outcome.trace.denial_reasons,
        "late recovery did not remove the contradicted synthetic reason",
    )
    policy_counts = revalidated.audit_counts
    _require(
        policy_counts["late_recovery_before_revalidation_count"] == 1
        and policy_counts["normal_policy_rerun_count"] == 1,
        "candidate execution order was not observed",
    )
    counts["contradicted_synthetic_reason_before_count"] += 1
    counts["contradicted_synthetic_reason_remaining_count"] += int(
        synthetic_reason in revalidated.outcome.trace.denial_reasons
    )
    counts["review_confidence_restored_count"] += int(
        policy_counts["review_confidence_restored_count"] > 0
        and revalidated.outcome.row.confidence
        == original.policy_outcome.row.confidence
    )

    independent_adjudicator = ReviewDenialRecoveryAdjudicator(
        _RecoveredOutputAwareBaseline(independent_denial=True)
    )
    independent_current = independent_adjudicator.adjudicate_staged(primary)
    independent_original = StagedAdjudication(
        policy_outcome=independent_current.policy_outcome,
        outcome=_outcome(
            denial_reasons=(
                synthetic_reason,
                "barred_sponsor:visible",
            ),
            confidence=REVIEW_DENIAL_CONFIDENCE,
            adjudication="DENIED",
        ),
    )
    independent_final = independent_adjudicator.revalidate_after_recovery(
        recovered,
        original=independent_original,
    )
    _require(
        "barred_sponsor:visible"
        in independent_final.outcome.trace.denial_reasons
        and independent_final.outcome.row.adjudication == "DENIED",
        "independent denial reason was erased by late recovery",
    )
    counts["independent_denial_reason_retained_count"] += int(
        independent_final.audit_counts[
            "independent_denial_reason_retained_count"
        ]
        > 0
    )

    signed = _candidate(
        "adjudication",
        "DENIED",
        evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
    )
    admitted = RapidOutputRecoveryProcessor._fusion_rapid_candidates(
        (signed, clean_biohazard),
        recover_risk=False,
        recover_policy=True,
        recover_outputs=False,
    )
    admitted_fields = {candidate.field_name for candidate in admitted}
    _require(
        "adjudication" in admitted_fields,
        "late signed adjudication evidence was filtered out",
    )
    _require(
        "biohazard_check" in admitted_fields,
        "late biohazard evidence was filtered out",
    )
    fused_policy = _resolved(signed, clean_biohazard)
    rendered = SimpleNamespace(
        case_id=_CASE,
        source_sha256=_SOURCE_SHA256,
    )
    _require(
        RapidOutputRecoveryProcessor._trusted_fused_policy_change(
            rendered=rendered,
            fused_resolved=fused_policy,
            field_name="adjudication",
        ),
        "late signed adjudication did not pass the provenance-complete gate",
    )
    _require(
        RapidOutputRecoveryProcessor._trusted_fused_policy_change(
            rendered=rendered,
            fused_resolved=fused_policy,
            field_name="biohazard_check",
        ),
        "late biohazard fact did not pass the provenance-complete gate",
    )
    authority = AdjudicationEngine().adjudicate_case(_resolved(signed))
    _require(
        authority.row.adjudication == "DENIED"
        and authority.trace.authoritative_source,
        "signed late authority did not retain absolute precedence",
    )
    missing_for_synthetic = frozenset(
        {"home_world", "risk_flags", "sponsor_id"}
    )
    recovered_outputs = tuple(
        candidate
        for candidate in _output_candidates()
        if candidate.field_name in missing_for_synthetic
    )
    full_synthetic = _run_full_fusion_processor(
        primary_candidates=_output_candidates(
            omit=missing_for_synthetic
        ),
        rapid_candidates=(
            *recovered_outputs,
            _candidate(
                "biohazard_check",
                "clean",
                evidence_type=EvidenceType.BIOMETRIC_SLIP,
                page_index=21,
                left=21,
            ),
        ),
    )
    full_synthetic_counts = full_synthetic.policy_audit_counts
    _require(
        full_synthetic.row.adjudication == "APPROVED"
        and full_synthetic_counts[
            "late_recovery_before_revalidation_count"
        ]
        == 1
        and full_synthetic_counts["normal_policy_rerun_count"] == 1
        and full_synthetic_counts[
            "late_biohazard_evidence_preserved_count"
        ]
        == 1
        and full_synthetic_counts[
            "contradicted_synthetic_reason_left_active_count"
        ]
        == 0,
        "full production path did not revalidate recovered evidence",
    )
    counts["candidate_late_recovery_before_revalidation_count"] += 1
    counts["candidate_revalidation_after_late_recovery_count"] += 1
    counts["normal_policy_rerun_count"] += 1
    counts["contradicted_synthetic_reason_removed_count"] += int(
        policy_counts["contradicted_synthetic_reason_removed_count"] > 0
    )
    counts["late_biohazard_evidence_preserved_count"] += 1

    signed_scenarios = (
        (
            "APPROVED",
            _output_candidates(),
            "DENIED",
        ),
        (
            "DENIED",
            _output_candidates(
                overrides={"risk_flags": "active_warrant"}
            ),
            "APPROVED",
        ),
        (
            "NEEDS_REVIEW",
            _output_candidates(omit=frozenset({"fee_status"})),
            "DENIED",
        ),
    )
    for index, (
        expected_primary,
        primary_candidates,
        signed_decision,
    ) in enumerate(signed_scenarios, start=1):
        _require(
            _staged_output_decision(primary_candidates)
            == expected_primary,
            "signed-authority probe did not start from the expected decision",
        )
        late_signed = _candidate(
            "adjudication",
            signed_decision,
            evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
            page_index=30 + index,
            left=float(30 + index),
        )
        signed_final = _run_full_fusion_processor(
            primary_candidates=primary_candidates,
            rapid_candidates=(late_signed,),
        )
        signed_counts = signed_final.policy_audit_counts
        _require(
            signed_final.row.adjudication == signed_decision
            and signed_counts[
                "signed_late_authority_recovery_count"
            ]
            == 1
            and signed_counts[
                "late_adjudication_evidence_preserved_count"
            ]
            == 1
            and signed_counts[
                "late_recovery_before_revalidation_count"
            ]
            == 1
            and signed_counts["normal_policy_rerun_count"] == 1
            and signed_counts["forced_approval_count"] == 0,
            "signed late authority did not survive the full production path",
        )
        counts["signed_late_authority_recovery_count"] += 1
        counts["late_adjudication_evidence_preserved_count"] += 1

    rules = PolicyRuleSet()
    stale_value = (
        rules.snapshot_receipt_date
        - timedelta(days=rules.stale_after_days + 1)
    ).isoformat()
    sponsor_marker = _candidate(
        "page_type_present_sponsor_attestation",
        "present",
        evidence_type=EvidenceType.SPONSOR_ATTESTATION,
        cues=("packet_page_type:sponsor_attestation",),
    )
    stale_field = _candidate("arrival_date", stale_value)
    low_review = _outcome(confidence=0.23)

    class _FixedBaseline:
        def adjudicate_case(
            self, _resolved_case: ResolvedCase
        ) -> AdjudicationOutcome:
            return low_review

    stale_stage = ReviewDenialRecoveryAdjudicator(
        _FixedBaseline(), rules=rules
    ).adjudicate_staged(_resolved(sponsor_marker, stale_field))
    _require(
        "review_denial_sponsor_stale_gt180"
        in stale_stage.outcome.trace.denial_reasons,
        "synthetic stale rule is not aligned to the published 180-day policy",
    )

    sentinel_date = _candidate("arrival_date", "1900-01-01")
    sentinel_stage = ReviewDenialRecoveryAdjudicator(
        _FixedBaseline(), rules=rules
    ).adjudicate_staged(_resolved(sponsor_marker, sentinel_date))
    _require(
        "review_denial_sponsor_stale_gt180"
        not in sentinel_stage.outcome.trace.denial_reasons,
        "serialization date sentinel was treated as stale evidence",
    )

    complete_candidates = _output_candidates()
    output_fields = tuple(
        candidate.field_name for candidate in complete_candidates
    )
    generic_placeholders = ("unknown", "null", "other", "none")
    probe_index = 0
    for field_name in output_fields:
        for placeholder in generic_placeholders:
            if field_name == "risk_flags" and placeholder == "none":
                continue
            probe_index += 1
            retained = tuple(
                candidate
                for candidate in complete_candidates
                if candidate.field_name != field_name
            )
            placeholder_candidate = _candidate(
                field_name,
                placeholder,
                cues=("synthetic_default",),
                page_index=50 + probe_index,
                left=float(50 + probe_index),
            )
            decision = _staged_output_decision(
                (*retained, placeholder_candidate)
            )
            _require(
                decision != "APPROVED",
                f"{field_name}={placeholder} broadened a policy approval",
            )
            counts["placeholder_guard_probe_count"] += 1
            counts["serialization_default_guard_probe_count"] += 1
            counts["forced_approval_guard_probe_count"] += 1

    _require(
        _staged_output_decision(complete_candidates) == "APPROVED",
        "visible risk_flags=none positive control did not remain legitimate",
    )

    for index, (field_name, sentinel) in enumerate(
        (
            ("sponsor_id", "SPN-0000"),
            ("arrival_date", "1900-01-01"),
        ),
        start=1,
    ):
        retained = tuple(
            candidate
            for candidate in complete_candidates
            if candidate.field_name != field_name
        )
        placeholder_candidate = _candidate(
            field_name,
            sentinel,
            cues=("synthetic_default",),
            page_index=100 + index,
            left=float(100 + index),
        )
        decision = _staged_output_decision(
            (*retained, placeholder_candidate)
        )
        _require(
            decision != "APPROVED",
            f"{field_name} sentinel broadened a policy approval",
        )
        counts["sentinel_guard_probe_count"] += 1

    review_candidates = _output_candidates(
        omit=frozenset({"fee_status"})
    )
    review_resolved, review_staged = _staged_output(review_candidates)
    guarded_review = RapidOutputRecoveryProcessor._apply_review_approval_heads(
        final_row=review_staged.outcome.row,
        source_sha256=_SOURCE_SHA256,
        primary_candidates=review_candidates,
        primary_outcome=review_staged.outcome,
        primary_resolved=review_resolved,
        rapid_candidates=(),
        rapid_resolved=None,
    )
    _require(
        review_staged.outcome.row.adjudication == "NEEDS_REVIEW"
        and guarded_review == review_staged.outcome.row,
        "a direct Rapid review-to-approval head remained active",
    )
    counts["direct_approval_head_guard_probe_count"] += 1
    counts["stale_threshold_guard_probe_count"] += 2

    _require(
        all(
            counts[name] > 0
            for name in (
                "legacy_synthetic_before_late_recovery_count",
                "candidate_late_recovery_before_revalidation_count",
                "candidate_revalidation_after_late_recovery_count",
                "contradicted_synthetic_reason_before_count",
                "contradicted_synthetic_reason_removed_count",
                "independent_denial_reason_retained_count",
                "review_confidence_restored_count",
                "normal_policy_rerun_count",
                "signed_late_authority_recovery_count",
                "late_adjudication_evidence_preserved_count",
                "late_biohazard_evidence_preserved_count",
                "placeholder_guard_probe_count",
                "sentinel_guard_probe_count",
                "serialization_default_guard_probe_count",
                "stale_threshold_guard_probe_count",
                "forced_approval_guard_probe_count",
                "direct_approval_head_guard_probe_count",
            )
        ),
        "one or more required policy probes were vacuous",
    )
    return counts
