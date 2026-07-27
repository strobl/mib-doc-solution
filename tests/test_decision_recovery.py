import unittest
from dataclasses import replace

from mib_pipeline import (
    AdjudicationOutcome,
    CandidateEvidence,
    CaseLinker,
    DecisionTrace,
    EvidencePrecedenceResolver,
    EvidenceType,
    FieldState,
    OcrLine,
    PredictionRow,
    Rect,
    REVIEW_APPROVAL_CONFIDENCE,
    REVIEW_DENIAL_CONFIDENCE,
    REVIEW_DIPLOMATIC_APPROVAL_CONFIDENCE,
    RenderedPage,
    ResolvedCase,
    ResolvedField,
    ReviewDenialRecoveryAdjudicator,
    VisibleEvidenceExtractor,
)
from mib_pipeline.decision_recovery import StagedAdjudication


CASE_ID = "MIB-000001"


def row(**overrides):
    values = {
        "case_id": CASE_ID,
        "applicant_name": "Veenax Qortari",
        "species_code": "ANDROMEDAN",
        "home_world": "Mars Dome-7",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-1042",
        "arrival_date": "2026-04-17",
        "declared_purpose": "technical work",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.25,
    }
    values.update(overrides)
    return PredictionRow.from_mapping(values)


def outcome(
    *,
    review_reasons=(),
    denial_reasons=(),
    approval_facts=("fee_paid",),
    prediction=None,
    trace_decision=None,
    authoritative_source=False,
):
    prediction = prediction or row()
    decision = trace_decision or prediction.adjudication
    return AdjudicationOutcome(
        row=prediction,
        trace=DecisionTrace(
            decision=decision,
            authoritative_source=authoritative_source,
            denial_reasons=tuple(denial_reasons),
            review_reasons=tuple(review_reasons),
            approval_facts=tuple(approval_facts),
            exception_ids=(),
        ),
    )


def marker(field_name, page_type, *, source="visible_ocr", cues=None):
    evidence = CandidateEvidence(
        field_name=field_name,
        value="present",
        evidence_type=(
            EvidenceType.SPONSOR_ATTESTATION
            if page_type == "sponsor_attestation"
            else EvidenceType.INTAKE_FORM
        ),
        page_index=0,
        box=Rect(0, 0, 100, 30),
        legible=True,
        superseded=False,
        ocr_confidence=0.95,
        visual_cues=(f"packet_page_type:{page_type}",) if cues is None else cues,
        source=source,
        case_id_hint=CASE_ID,
        applicant_hint=None,
    )
    return ResolvedField(
        field_name=field_name,
        state=FieldState.RESOLVED,
        value="present",
        winning_evidence=evidence,
        considered=(evidence,),
        reason="test marker",
    )


def resolved_case(*markers):
    return ResolvedCase(
        case_id=CASE_ID,
        active_applicant="Veenax Qortari",
        fields={item.field_name: item for item in markers},
        unresolved_linkage=False,
        unresolved_reasons=(),
    )


def visible_field(
    field_name,
    value,
    *,
    source="visible_ocr",
    evidence_type=EvidenceType.INTAKE_FORM,
    superseded=False,
    cues=(),
):
    evidence = CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=1,
        box=Rect(0, 40, 100, 70),
        legible=True,
        superseded=superseded,
        ocr_confidence=0.95,
        visual_cues=tuple(cues),
        source=source,
        case_id_hint=CASE_ID,
        applicant_hint="Veenax Qortari",
    )
    return ResolvedField(
        field_name=field_name,
        state=FieldState.RESOLVED,
        value=value,
        winning_evidence=evidence,
        considered=(evidence,),
        reason="test visible field",
    )


class FakeAdjudicator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def adjudicate_case(self, resolved):
        self.calls += 1
        return self.result


def recover(result, resolved):
    baseline = FakeAdjudicator(result)
    recovered = ReviewDenialRecoveryAdjudicator(baseline).adjudicate_case(resolved)
    return recovered, baseline


def line(text, index=0):
    return OcrLine(
        page_index=0,
        text=text,
        confidence=0.91,
        box=Rect(0, index * 20, 200, index * 20 + 15),
        tokens=(),
    )


class PacketPageTypeMarkerTests(unittest.TestCase):
    def test_classification_exactly_uses_first_four_lines_and_mining_priority(self):
        self.assertEqual(
            VisibleEvidenceExtractor.packet_page_type(
                (line("Sponsor Attestation Letter"),)
            ),
            "sponsor_attestation",
        )
        self.assertEqual(
            VisibleEvidenceExtractor.packet_page_type(
                (line("Biometric Sponsor Attestation Letter"),)
            ),
            "biometric_slip",
        )
        self.assertEqual(
            VisibleEvidenceExtractor.packet_page_type(
                tuple(line("ordinary text", index) for index in range(4))
                + (line("Sponsor Attestation Letter", 4),)
            ),
            "other",
        )

    def test_only_frozen_types_create_visible_policy_markers(self):
        page = RenderedPage(
            index=0,
            image_png=b"",
            width_px=100,
            height_px=100,
            dpi=200,
            rotation_deg=0,
            skew_correction_deg=0.0,
            crop_box=Rect(0, 0, 100, 100),
            text_spans=(),
        )
        sponsor = VisibleEvidenceExtractor._packet_page_type_marker(
            page=page,
            lines=(line("Sponsor Letter"),),
            case_id=CASE_ID,
        )
        fee = VisibleEvidenceExtractor._packet_page_type_marker(
            page=page,
            lines=(line("MIB Fee Receipt"),),
            case_id=CASE_ID,
        )
        intake = VisibleEvidenceExtractor._packet_page_type_marker(
            page=page,
            lines=(line("FORM I-8090"),),
            case_id=CASE_ID,
        )
        empty = VisibleEvidenceExtractor._packet_page_type_marker(
            page=page,
            lines=(),
            case_id=CASE_ID,
        )
        unrecognized = VisibleEvidenceExtractor._packet_page_type_marker(
            page=page,
            lines=(line("ordinary unclassified page"),),
            case_id=CASE_ID,
        )

        self.assertIsNotNone(sponsor)
        self.assertEqual(
            sponsor.field_name,
            "page_type_present_sponsor_attestation",
        )
        self.assertEqual(sponsor.value, "present")
        self.assertEqual(sponsor.source, "visible_ocr")
        self.assertIsNotNone(fee)
        self.assertEqual(fee.field_name, "page_type_present_fee_receipt")
        self.assertEqual(fee.visual_cues, ("packet_page_type:fee_receipt",))
        self.assertIsNone(intake)
        self.assertIsNone(empty)
        self.assertIsNone(unrecognized)

        linked = CaseLinker().link(CASE_ID, (sponsor, fee))
        resolved = EvidencePrecedenceResolver().resolve(linked)
        self.assertEqual(
            resolved.value("page_type_present_sponsor_attestation"),
            "present",
        )
        self.assertEqual(
            resolved.value("page_type_present_fee_receipt"),
            "present",
        )


class ReviewDiplomaticApprovalRecoveryTests(unittest.TestCase):
    FEE = marker("page_type_present_fee_receipt", "fee_receipt")
    DIPLOMATIC_VISA = visible_field("visa_class", "DIP-1")

    def assert_abstained(self, original, recovered):
        self.assertIs(recovered, original)

    def test_policy_fact_and_visible_dip_each_recover_at_frozen_boundary(self):
        cases = (
            (
                outcome(
                    approval_facts=(
                        "application_date_current_or_exempt",
                        "diplomatic_sponsor_exemption",
                    ),
                    prediction=row(visa_class="DIP-1", confidence=0.25),
                    review_reasons=("fee_status_unknown",),
                ),
                resolved_case(self.FEE),
            ),
            (
                outcome(
                    approval_facts=("application_date_current_or_exempt",),
                    prediction=row(visa_class="DIP-1", confidence=0.25),
                    review_reasons=("required_output_unknown:risk_flags",),
                ),
                resolved_case(self.FEE, self.DIPLOMATIC_VISA),
            ),
        )
        for original, resolved in cases:
            with self.subTest(approval_facts=original.trace.approval_facts):
                recovered, baseline = recover(original, resolved)
                self.assertEqual(baseline.calls, 1)
                self.assert_abstained(original, recovered)

    def test_approval_vetoes_missing_or_untrusted_features(self):
        diplomatic = outcome(
            approval_facts=("diplomatic_sponsor_exemption",),
            prediction=row(visa_class="DIP-1"),
        )
        second_fee = replace(self.FEE.winning_evidence, page_index=2)
        duplicate_fee_pages = replace(
            self.FEE,
            considered=(self.FEE.winning_evidence, second_fee),
        )
        variants = (
            (diplomatic, resolved_case()),
            (diplomatic, resolved_case(duplicate_fee_pages)),
            (
                diplomatic,
                resolved_case(
                    marker(
                        "page_type_present_fee_receipt",
                        "fee_receipt",
                        source="text_layer",
                    )
                ),
            ),
            (
                outcome(prediction=row(visa_class="DIP-1")),
                resolved_case(self.FEE),
            ),
            (
                outcome(
                    prediction=row(visa_class="DIP-1"),
                    approval_facts=(),
                ),
                resolved_case(
                    self.FEE,
                    visible_field(
                        "visa_class",
                        "DIP-1",
                        evidence_type=EvidenceType.TEXT_LAYER,
                    ),
                ),
            ),
            (
                outcome(
                    approval_facts=("diplomatic_sponsor_exemption",),
                    prediction=row(visa_class="DIP-1", confidence=0.250001),
                ),
                resolved_case(self.FEE),
            ),
            (
                outcome(
                    approval_facts=("diplomatic_sponsor_exemption",),
                    denial_reasons=("policy_denial",),
                    prediction=row(visa_class="DIP-1"),
                ),
                resolved_case(self.FEE),
            ),
        )
        for original, resolved in variants:
            with self.subTest(
                confidence=original.row.confidence,
                fields=tuple(resolved.fields),
                denial_reasons=original.trace.denial_reasons,
            ):
                recovered, _baseline = recover(original, resolved)
                self.assertIs(recovered, original)

    def test_denial_recovery_has_priority_over_approval_recovery(self):
        sponsor = marker(
            "page_type_present_sponsor_attestation",
            "sponsor_attestation",
        )
        original = outcome(
            approval_facts=("diplomatic_sponsor_exemption",),
            prediction=row(
                arrival_date="2025-01-01",
                visa_class="DIP-1",
                confidence=0.25,
            ),
        )

        recovered, _baseline = recover(
            original,
            resolved_case(
                self.FEE,
                sponsor,
                visible_field("arrival_date", "2025-01-01"),
            ),
        )

        self.assertEqual(recovered.row.adjudication, "DENIED")
        self.assertIn(
            "review_denial_sponsor_stale_gt180",
            recovered.trace.denial_reasons,
        )


class ReviewApprovalRecoveryTests(unittest.TestCase):
    SPONSOR = marker(
        "page_type_present_sponsor_attestation",
        "sponsor_attestation",
    )
    OTHER = marker("page_type_present_other", "other")
    CURRENT_APPLICATION = "application_date_current_or_exempt"

    def assert_abstained(self, original, recovered, _expected_fact):
        self.assertIs(recovered, original)

    def test_all_three_frozen_approval_rules_recover_at_boundaries(self):
        cases = (
            (
                outcome(prediction=row(visa_class="XW-1", confidence=0.20)),
                resolved_case(
                    self.SPONSOR,
                    visible_field("visa_class", "XW-1"),
                ),
                "review_approval_sponsor_attestation_xw1",
            ),
            (
                outcome(
                    review_reasons=("visa_class_unknown",),
                    approval_facts=(self.CURRENT_APPLICATION,),
                    prediction=row(confidence=0.25),
                ),
                resolved_case(),
                "review_approval_current_application_visa_unknown",
            ),
            (
                outcome(
                    review_reasons=("required_output_unknown:home_world",),
                    approval_facts=(self.CURRENT_APPLICATION,),
                    prediction=row(confidence=0.35),
                ),
                resolved_case(),
                "review_approval_current_application_home_world_unknown",
            ),
        )
        for original, resolved, expected_fact in cases:
            with self.subTest(expected_fact=expected_fact):
                recovered, baseline = recover(original, resolved)
                self.assertEqual(baseline.calls, 1)
                self.assert_abstained(original, recovered, expected_fact)

    def test_current_home_world_unknown_rule_vetoes_unsupported_fee_waiver(self):
        blocked = outcome(
            review_reasons=(
                "required_output_unknown:home_world",
                "unsupported_fee_waiver",
            ),
            approval_facts=(self.CURRENT_APPLICATION,),
            prediction=row(confidence=0.35),
        )
        recovered, _baseline = recover(blocked, resolved_case())
        self.assertIs(recovered, blocked)

        boundary = outcome(
            review_reasons=("required_output_unknown:home_world",),
            approval_facts=(self.CURRENT_APPLICATION,),
            prediction=row(confidence=0.35),
        )
        recovered, _baseline = recover(boundary, resolved_case())
        self.assert_abstained(
            boundary,
            recovered,
            "review_approval_current_application_home_world_unknown",
        )

    def test_sponsor_xw1_rule_vetoes_each_missing_or_untrusted_feature(self):
        valid = outcome(prediction=row(visa_class="XW-1", confidence=0.20))
        variants = (
            (valid, resolved_case()),
            (
                outcome(prediction=row(visa_class="XW-2", confidence=0.20)),
                resolved_case(self.SPONSOR),
            ),
            (
                outcome(prediction=row(visa_class="XW-1", confidence=0.200001)),
                resolved_case(self.SPONSOR),
            ),
            (
                valid,
                resolved_case(
                    marker(
                        "page_type_present_sponsor_attestation",
                        "sponsor_attestation",
                        source="text_layer",
                    )
                ),
            ),
        )
        for original, resolved in variants:
            with self.subTest(
                visa_class=original.row.visa_class,
                confidence=original.row.confidence,
                fields=tuple(resolved.fields),
            ):
                recovered, _baseline = recover(original, resolved)
                self.assertIs(recovered, original)

    def test_current_visa_unknown_rule_enforces_fact_reason_and_open_lower_bound(self):
        variants = (
            outcome(
                review_reasons=("visa_class_unknown",),
                approval_facts=(),
                prediction=row(confidence=0.25),
            ),
            outcome(
                review_reasons=(),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.25),
            ),
            outcome(
                review_reasons=("visa_class_unknown",),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.20),
            ),
            outcome(
                review_reasons=("visa_class_unknown",),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.250001),
            ),
        )
        for original in variants:
            with self.subTest(
                confidence=original.row.confidence,
                review_reasons=original.trace.review_reasons,
                approval_facts=original.trace.approval_facts,
            ):
                recovered, _baseline = recover(original, resolved_case())
                self.assertIs(recovered, original)

    def test_current_home_world_rule_enforces_fact_reason_and_upper_bound(self):
        variants = (
            outcome(
                review_reasons=("required_output_unknown:home_world",),
                approval_facts=(),
                prediction=row(confidence=0.35),
            ),
            outcome(
                review_reasons=(),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.35),
            ),
            outcome(
                review_reasons=("required_output_unknown:home_world",),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.350001),
            ),
        )
        for original in variants:
            with self.subTest(
                confidence=original.row.confidence,
                review_reasons=original.trace.review_reasons,
                approval_facts=original.trace.approval_facts,
            ):
                recovered, _baseline = recover(original, resolved_case())
                self.assertIs(recovered, original)

    def test_three_missing_outputs_are_not_a_denial_or_approval_signal(self):
        cases = (
            outcome(
                review_reasons=(
                    "required_output_unknown:home_world",
                    "required_output_unknown:risk_flags",
                    "required_output_unknown:sponsor_id",
                ),
                prediction=row(visa_class="XW-1", confidence=0.20),
            ),
            outcome(
                review_reasons=(
                    "required_output_unknown:home_world",
                    "required_output_unknown:risk_flags",
                    "required_output_unknown:sponsor_id",
                    "visa_class_unknown",
                ),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.25),
            ),
            outcome(
                review_reasons=(
                    "required_output_unknown:home_world",
                    "required_output_unknown:risk_flags",
                    "required_output_unknown:sponsor_id",
                ),
                approval_facts=(self.CURRENT_APPLICATION,),
                prediction=row(confidence=0.35),
            ),
        )
        for original in cases:
            with self.subTest(
                confidence=original.row.confidence,
                review_reasons=original.trace.review_reasons,
            ):
                recovered, _baseline = recover(
                    original,
                    resolved_case(
                        self.SPONSOR,
                        visible_field("visa_class", "XW-1"),
                    ),
                )
                self.assertIs(recovered, original)
                self.assertEqual(recovered.row.adjudication, "NEEDS_REVIEW")
                self.assertNotIn(
                    "review_denial_three_required_outputs_unknown",
                    recovered.trace.denial_reasons,
                )


class ReviewDenialRecoveryTests(unittest.TestCase):
    OTHER = marker("page_type_present_other", "other")
    SPONSOR = marker(
        "page_type_present_sponsor_attestation",
        "sponsor_attestation",
    )

    def assert_recovered(self, original, recovered, expected_reason):
        self.assertEqual(recovered.row.adjudication, "DENIED")
        self.assertEqual(recovered.row.confidence, REVIEW_DENIAL_CONFIDENCE)
        self.assertEqual(recovered.trace.decision, "DENIED")
        self.assertFalse(recovered.trace.authoritative_source)
        self.assertIn(expected_reason, recovered.trace.denial_reasons)
        original_values = original.row.to_dict()
        recovered_values = recovered.row.to_dict()
        for field_name in original_values:
            if field_name not in {"adjudication", "confidence"}:
                self.assertEqual(
                    recovered_values[field_name],
                    original_values[field_name],
                    field_name,
                )

    def test_visible_stale_sponsor_rule_recovers_review(self):
        original = outcome(prediction=row(arrival_date="2025-01-01"))
        resolved = resolved_case(
            self.SPONSOR,
            visible_field("arrival_date", "2025-01-01"),
        )
        recovered, baseline = recover(original, resolved)
        self.assertEqual(baseline.calls, 1)
        self.assert_recovered(
            original,
            recovered,
            "review_denial_sponsor_stale_gt180",
        )

    def test_missingness_only_remains_review_with_original_confidence(self):
        original = outcome(
            review_reasons=(
                "required_output_unknown:home_world",
                "required_output_unknown:risk_flags",
                "required_output_unknown:sponsor_id",
            ),
            prediction=row(confidence=0.29),
        )
        recovered, baseline = recover(original, resolved_case())
        self.assertEqual(baseline.calls, 1)
        self.assertIs(recovered, original)
        self.assertEqual(recovered.row.adjudication, "NEEDS_REVIEW")
        self.assertEqual(recovered.row.confidence, 0.29)
        self.assertNotIn(
            "review_denial_three_required_outputs_unknown",
            recovered.trace.denial_reasons,
        )

    def test_other_page_fallback_is_never_policy_evidence(self):
        baseline = outcome(review_reasons=("clean_biohazard_check_missing",))
        variants = (
            (baseline, resolved_case(self.OTHER)),
            (
                outcome(
                    review_reasons=("clean_biohazard_check_missing",),
                    prediction=row(confidence=0.01),
                ),
                resolved_case(self.OTHER),
            ),
            (
                baseline,
                resolved_case(
                    marker(
                        "page_type_present_other",
                        "other",
                        source="text_layer",
                    )
                ),
            ),
        )
        for original, resolved in variants:
            with self.subTest(original=original, fields=tuple(resolved.fields)):
                recovered, _baseline = recover(original, resolved)
                self.assertIs(recovered, original)

    def test_rule_two_vetoes_missing_marker_current_arrival_and_high_confidence(self):
        stale = outcome(prediction=row(arrival_date="2025-01-01"))
        variants = (
            (stale, resolved_case()),
            (
                outcome(prediction=row(arrival_date="1900-01-01")),
                resolved_case(
                    self.SPONSOR,
                    visible_field("arrival_date", "1900-01-01"),
                ),
            ),
            (
                outcome(prediction=row(arrival_date="2026-01-08")),
                resolved_case(
                    self.SPONSOR,
                    visible_field("arrival_date", "2026-01-08"),
                ),
            ),
            (
                outcome(
                    prediction=row(
                        arrival_date="2025-01-01",
                        confidence=0.350001,
                    )
                ),
                resolved_case(
                    self.SPONSOR,
                    visible_field("arrival_date", "2025-01-01"),
                ),
            ),
        )
        for original, resolved in variants:
            with self.subTest(arrival=original.row.arrival_date):
                recovered, _baseline = recover(original, resolved)
                self.assertIs(recovered, original)

    def test_rule_three_requires_each_of_its_three_reasons(self):
        required = (
            "required_output_unknown:home_world",
            "required_output_unknown:risk_flags",
            "required_output_unknown:sponsor_id",
        )
        for omitted in required:
            with self.subTest(omitted=omitted):
                original = outcome(
                    review_reasons=tuple(
                        reason for reason in required if reason != omitted
                    )
                )
                recovered, _baseline = recover(original, resolved_case())
                self.assertIs(recovered, original)

    def test_non_review_or_inconsistent_baseline_is_unchanged(self):
        triggers = resolved_case(self.OTHER)
        for decision in ("APPROVED", "DENIED"):
            original = outcome(
                review_reasons=("clean_biohazard_check_missing",),
                prediction=row(adjudication=decision),
            )
            recovered, _baseline = recover(original, triggers)
            self.assertIs(recovered, original)

        inconsistent = outcome(
            review_reasons=("clean_biohazard_check_missing",),
            trace_decision="APPROVED",
        )
        recovered, _baseline = recover(inconsistent, triggers)
        self.assertIs(recovered, inconsistent)

    def test_adjudicate_returns_schema_typed_row(self):
        original = outcome(
            review_reasons=(
                "required_output_unknown:home_world",
                "required_output_unknown:risk_flags",
                "required_output_unknown:sponsor_id",
            )
        )
        wrapper = ReviewDenialRecoveryAdjudicator(FakeAdjudicator(original))

        recovered = wrapper.adjudicate(resolved_case())

        self.assertIsInstance(recovered, PredictionRow)
        self.assertEqual(tuple(recovered.to_dict()), tuple(original.row.to_dict()))
        self.assertEqual(recovered.confidence, original.row.confidence)


class PostRecoveryPolicyRevalidationTests(unittest.TestCase):
    REQUIRED_GAPS = (
        "required_output_unknown:home_world",
        "required_output_unknown:risk_flags",
        "required_output_unknown:sponsor_id",
    )

    def test_contradicted_synthetic_reason_is_removed_and_review_confidence_restored(
        self,
    ):
        baseline = FakeAdjudicator(
            outcome(
                review_reasons=self.REQUIRED_GAPS,
                prediction=row(confidence=0.23),
            )
        )
        wrapper = ReviewDenialRecoveryAdjudicator(baseline)
        policy_outcome = baseline.result
        original = StagedAdjudication(
            policy_outcome=policy_outcome,
            outcome=AdjudicationOutcome(
                row=replace(
                    policy_outcome.row,
                    adjudication="DENIED",
                    confidence=REVIEW_DENIAL_CONFIDENCE,
                ),
                trace=replace(
                    policy_outcome.trace,
                    decision="DENIED",
                    denial_reasons=(
                        "review_denial_three_required_outputs_unknown",
                    ),
                ),
            ),
        )
        self.assertEqual(original.outcome.row.adjudication, "DENIED")

        baseline.result = outcome(
            review_reasons=("fee_status_unknown",),
            prediction=row(confidence=0.31),
        )
        revalidated = wrapper.revalidate_after_recovery(
            resolved_case(),
            original=original,
        )

        self.assertEqual(revalidated.outcome.row.adjudication, "NEEDS_REVIEW")
        self.assertEqual(revalidated.outcome.row.confidence, 0.23)
        self.assertNotIn(
            "review_denial_three_required_outputs_unknown",
            revalidated.outcome.trace.denial_reasons,
        )
        self.assertEqual(
            revalidated.audit_counts[
                "contradicted_synthetic_reason_removed_count"
            ],
            1,
        )
        self.assertEqual(
            revalidated.audit_counts["review_confidence_restored_count"],
            1,
        )
        self.assertEqual(
            revalidated.audit_counts["normal_policy_rerun_count"],
            1,
        )
        self.assertEqual(
            revalidated.audit_counts[
                "contradicted_synthetic_reason_left_active_count"
            ],
            0,
        )

    def test_independent_denial_reason_is_retained_fail_closed(self):
        independent = "independent_policy_denial"
        baseline = FakeAdjudicator(
            outcome(
                review_reasons=self.REQUIRED_GAPS,
                denial_reasons=(independent,),
                prediction=row(confidence=0.24),
            )
        )
        wrapper = ReviewDenialRecoveryAdjudicator(baseline)
        policy_outcome = baseline.result
        original = StagedAdjudication(
            policy_outcome=policy_outcome,
            outcome=AdjudicationOutcome(
                row=replace(
                    policy_outcome.row,
                    adjudication="DENIED",
                    confidence=REVIEW_DENIAL_CONFIDENCE,
                ),
                trace=replace(
                    policy_outcome.trace,
                    decision="DENIED",
                    denial_reasons=(
                        independent,
                        "review_denial_three_required_outputs_unknown",
                    ),
                ),
            ),
        )

        baseline.result = outcome(
            review_reasons=("fee_status_unknown",),
            prediction=row(confidence=0.32),
        )
        revalidated = wrapper.revalidate_after_recovery(
            resolved_case(),
            original=original,
        )

        self.assertEqual(revalidated.outcome.row.adjudication, "DENIED")
        self.assertIn(
            independent,
            revalidated.outcome.trace.denial_reasons,
        )
        self.assertNotIn(
            "review_denial_three_required_outputs_unknown",
            revalidated.outcome.trace.denial_reasons,
        )
        self.assertEqual(
            revalidated.audit_counts[
                "independent_denial_reason_retained_count"
            ],
            1,
        )

    def test_signed_authoritative_review_keeps_authoritative_confidence(self):
        baseline = FakeAdjudicator(
            outcome(
                review_reasons=self.REQUIRED_GAPS,
                prediction=row(confidence=0.22),
            )
        )
        wrapper = ReviewDenialRecoveryAdjudicator(baseline)
        original = wrapper.adjudicate_staged(resolved_case())

        baseline.result = outcome(
            review_reasons=("authoritative_visible_decision",),
            prediction=row(confidence=0.94),
            authoritative_source=True,
        )
        revalidated = wrapper.revalidate_after_recovery(
            resolved_case(),
            original=original,
        )

        self.assertEqual(revalidated.outcome.row.adjudication, "NEEDS_REVIEW")
        self.assertEqual(revalidated.outcome.row.confidence, 0.94)
        self.assertEqual(
            revalidated.audit_counts["review_confidence_restored_count"],
            0,
        )

    def test_late_signed_authority_is_absolute_across_full_decision_matrix(self):
        for initial_decision in (
            "APPROVED",
            "DENIED",
            "NEEDS_REVIEW",
        ):
            for signed_decision in (
                "APPROVED",
                "DENIED",
                "NEEDS_REVIEW",
            ):
                with self.subTest(
                    initial=initial_decision,
                    signed=signed_decision,
                ):
                    initial = outcome(
                        prediction=row(
                            adjudication=initial_decision,
                            confidence=0.41,
                        ),
                        trace_decision=initial_decision,
                        denial_reasons=(
                            ("ordinary_policy_denial",)
                            if initial_decision == "DENIED"
                            else ()
                        ),
                        review_reasons=(
                            ("ordinary_policy_review",)
                            if initial_decision == "NEEDS_REVIEW"
                            else ()
                        ),
                    )
                    baseline = FakeAdjudicator(initial)
                    wrapper = ReviewDenialRecoveryAdjudicator(baseline)
                    original = wrapper.adjudicate_staged(
                        resolved_case()
                    )

                    baseline.result = outcome(
                        prediction=row(
                            adjudication=signed_decision,
                            confidence=0.93,
                        ),
                        trace_decision=signed_decision,
                        authoritative_source=True,
                        denial_reasons=(
                            ("authoritative_visible_decision",)
                            if signed_decision == "DENIED"
                            else ()
                        ),
                        review_reasons=(
                            ("authoritative_visible_decision",)
                            if signed_decision == "NEEDS_REVIEW"
                            else ()
                        ),
                        approval_facts=(
                            ("authoritative_visible_decision",)
                            if signed_decision == "APPROVED"
                            else ()
                        ),
                    )
                    revalidated = wrapper.revalidate_after_recovery(
                        resolved_case(),
                        original=original,
                    )

                    self.assertEqual(
                        revalidated.outcome.row.adjudication,
                        signed_decision,
                    )
                    self.assertEqual(
                        revalidated.outcome.trace.decision,
                        signed_decision,
                    )
                    self.assertTrue(
                        revalidated.outcome.trace.authoritative_source
                    )
                    self.assertEqual(
                        revalidated.outcome.row.confidence,
                        0.93,
                    )

    def test_stale_rule_uses_exact_published_180_day_boundary(self):
        sponsor = marker(
            "page_type_present_sponsor_attestation",
            "sponsor_attestation",
        )
        for arrival, denied in (
            ("2026-01-08", False),
            ("2026-01-07", True),
        ):
            with self.subTest(arrival=arrival):
                original = outcome(
                    prediction=row(arrival_date=arrival, confidence=0.25)
                )
                recovered, _baseline = recover(
                    original,
                    resolved_case(
                        sponsor,
                        visible_field("arrival_date", arrival),
                    ),
                )
                self.assertEqual(
                    recovered.row.adjudication == "DENIED",
                    denied,
                )

    def test_stale_rule_uses_visible_receipt_and_defaults_invalid_receipt_to_snapshot(
        self,
    ):
        sponsor = marker(
            "page_type_present_sponsor_attestation",
            "sponsor_attestation",
        )
        original = outcome(
            prediction=row(arrival_date="2026-01-01", confidence=0.25)
        )
        fresh_by_receipt, _baseline = recover(
            original,
            resolved_case(
                sponsor,
                visible_field("arrival_date", "2026-01-01"),
                visible_field("packet_receipt_date", "2026-01-15"),
            ),
        )
        self.assertIs(fresh_by_receipt, original)

        for receipt in (
            visible_field(
                "packet_receipt_date",
                "2026-01-15",
                source="text_layer",
                evidence_type=EvidenceType.TEXT_LAYER,
            ),
            visible_field("packet_receipt_date", "1900-01-01"),
        ):
            with self.subTest(receipt=receipt.value):
                recovered, _baseline = recover(
                    original,
                    resolved_case(
                        sponsor,
                        visible_field("arrival_date", "2026-01-01"),
                        receipt,
                    ),
                )
                self.assertEqual(
                    recovered.row.adjudication,
                    "DENIED",
                )
                self.assertIn(
                    "review_denial_sponsor_stale_gt180",
                    recovered.trace.denial_reasons,
                )

    def test_wrong_scope_marker_cannot_activate_synthetic_denial(self):
        wrong_case_marker = marker(
            "page_type_present_sponsor_attestation",
            "sponsor_attestation",
        )
        wrong_evidence = replace(
            wrong_case_marker.winning_evidence,
            case_id_hint="MIB-999999",
        )
        wrong_case_marker = replace(
            wrong_case_marker,
            winning_evidence=wrong_evidence,
            considered=(wrong_evidence,),
        )
        original = outcome(
            prediction=row(arrival_date="2025-01-01", confidence=0.25)
        )

        recovered, _baseline = recover(
            original,
            resolved_case(
                wrong_case_marker,
                visible_field("arrival_date", "2025-01-01"),
            ),
        )

        self.assertIs(recovered, original)

    def test_unsafe_policy_monitors_have_executable_nonzero_paths(self):
        wrapper = ReviewDenialRecoveryAdjudicator(
            FakeAdjudicator(outcome())
        )
        policy_review = outcome(
            review_reasons=("ordinary_review",),
            prediction=row(confidence=0.2),
        )
        forced_approval = outcome(
            prediction=row(
                adjudication="APPROVED",
                confidence=0.9,
            ),
            trace_decision="APPROVED",
            approval_facts=("strict_approval_bar_cleared",),
        )
        forced_counts = wrapper._policy_safety_counts(
            resolved_case(),
            StagedAdjudication(
                policy_outcome=policy_review,
                outcome=forced_approval,
            ),
            monitor_forced_approval=True,
        )
        self.assertEqual(forced_counts["forced_approval_count"], 1)
        self.assertGreater(
            forced_counts[
                "serialization_default_used_as_policy_evidence_count"
            ],
            0,
        )

        malformed_policy = forced_approval
        sentinel_counts = wrapper._policy_safety_counts(
            resolved_case(
                visible_field("sponsor_id", "SPN-0000"),
                visible_field("arrival_date", "1900-01-01"),
            ),
            StagedAdjudication(
                policy_outcome=malformed_policy,
                outcome=malformed_policy,
            ),
            monitor_forced_approval=True,
        )
        self.assertEqual(
            sentinel_counts[
                "sentinel_value_used_as_policy_evidence_count"
            ],
            2,
        )

        placeholder_counts = wrapper._policy_safety_counts(
            resolved_case(
                visible_field("visa_class", "unknown"),
            ),
            StagedAdjudication(
                policy_outcome=malformed_policy,
                outcome=malformed_policy,
            ),
            monitor_forced_approval=True,
        )
        self.assertEqual(
            placeholder_counts[
                "placeholder_value_used_as_evidence_count"
            ],
            1,
        )

        stale_expected = outcome(
            prediction=row(arrival_date="2025-01-01", confidence=0.2)
        )
        stale_counts = wrapper._policy_safety_counts(
            resolved_case(
                marker(
                    "page_type_present_sponsor_attestation",
                    "sponsor_attestation",
                ),
                visible_field("arrival_date", "2025-01-01"),
            ),
            StagedAdjudication(
                policy_outcome=stale_expected,
                outcome=stale_expected,
            ),
            monitor_forced_approval=True,
        )
        self.assertEqual(
            stale_counts["stale_threshold_mismatch_count"],
            1,
        )


if __name__ == "__main__":
    unittest.main()
