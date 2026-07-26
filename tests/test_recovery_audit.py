import json
import unittest

from mib_pipeline.extraction import CandidateEvidence, EvidenceType
from mib_pipeline.ingestion import Rect
from mib_pipeline.models import PredictionRow
from mib_pipeline.provenance import CoordinateTransform, make_ocr_provenance
from mib_pipeline.recovery_audit import (
    CandidateValidationFailure,
    CandidateValidationPolicy,
    RecoveryAuditOverlay,
    SerializationOrigin,
    VisibleRecoveryResult,
    recovered_field_audit,
    unchanged_field_audit,
    validate_recovered_candidate,
    visible_repair_field_audit,
)
from mib_pipeline.resolution import FieldState, ResolvedField


SOURCE_SHA256 = "a" * 64
OTHER_SOURCE_SHA256 = "b" * 64
APPLICANT = "Zed Zarnax"
BOX = Rect(10, 20, 80, 42)


def visible_candidate(
    field_name="risk_flags",
    value="none",
    *,
    source_sha256=SOURCE_SHA256,
    page_index=2,
    applicant_scope=APPLICANT,
    applicant_hint=APPLICANT,
    confidence=0.97,
    box=BOX,
    provenance_box=BOX,
    cues=(),
    legible=True,
    superseded=False,
    with_provenance=True,
    route_id="targeted_rapidocr",
    source="visible_ocr",
    evidence_type=EvidenceType.INTAKE_FORM,
):
    provenance = ()
    if with_provenance:
        provenance = (
            make_ocr_provenance(
                source_sha256=source_sha256,
                page_index=page_index,
                view_box=provenance_box,
                applicant_scope=applicant_scope,
                route_id=route_id,
                engine_id="rapidocr:pinned-test-models",
                view_id="rendered_page",
            ),
        )
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=page_index,
        box=box,
        legible=legible,
        superseded=superseded,
        ocr_confidence=confidence,
        visual_cues=tuple(cues),
        source=source,
        case_id_hint="MIB-000001",
        applicant_hint=applicant_hint,
        ocr_provenance=provenance,
    )


def resolved_field(
    field_name,
    value,
    *,
    state=FieldState.RESOLVED,
    winner=None,
):
    return ResolvedField(
        field_name=field_name,
        state=state,
        value=value,
        winning_evidence=winner,
        considered=(() if winner is None else (winner,)),
        reason="test field",
    )


def policy(
    *,
    field_name="risk_flags",
    source_sha256=SOURCE_SHA256,
    page_index=2,
    applicant_scope=APPLICANT,
    minimum_confidence=0.95,
):
    return CandidateValidationPolicy(
        expected_field_name=field_name,
        expected_source_sha256=source_sha256,
        expected_page_index=page_index,
        expected_applicant_scope=applicant_scope,
        minimum_confidence=minimum_confidence,
    )


class RecoveryAuditTests(unittest.TestCase):
    def test_output_default_none_is_not_visible_none(self):
        primary = resolved_field(
            "risk_flags",
            None,
            state=FieldState.UNKNOWN,
        )

        audit = unchanged_field_audit(
            primary=primary,
            serialized_value="none",
            linked_recovery_scope=APPLICANT,
        )

        self.assertEqual(audit.primary_state, FieldState.UNKNOWN)
        self.assertIsNone(audit.primary_evidence_value)
        self.assertEqual(audit.serialization_before, "none")
        self.assertEqual(
            audit.serialization_before_origin,
            SerializationOrigin.OUTPUT_DEFAULT,
        )
        self.assertTrue(audit.before_is_output_default)
        self.assertFalse(audit.before_is_explicit_visible_value)

    def test_contested_evidence_remains_distinct_from_output_default(self):
        contested = unchanged_field_audit(
            primary=resolved_field(
                "sponsor_id",
                None,
                state=FieldState.CONTESTED,
            ),
            serialized_value="SPN-0000",
            linked_recovery_scope=APPLICANT,
        )
        unknown = unchanged_field_audit(
            primary=resolved_field(
                "sponsor_id",
                None,
                state=FieldState.UNKNOWN,
            ),
            serialized_value="SPN-0000",
            linked_recovery_scope=APPLICANT,
        )

        self.assertEqual(contested.primary_state, FieldState.CONTESTED)
        self.assertEqual(contested.final_evidence_state, FieldState.CONTESTED)
        self.assertEqual(contested.serialization_before, "SPN-0000")
        self.assertTrue(contested.before_is_output_default)
        self.assertNotEqual(contested.primary_state, unknown.primary_state)

    def test_primary_explicit_unknown_fee_is_not_an_output_default(self):
        winner = visible_candidate(
            field_name="fee_status",
            value="unknown",
        )
        primary = resolved_field("fee_status", "unknown", winner=winner)

        audit = unchanged_field_audit(
            primary=primary,
            serialized_value="unknown",
            linked_recovery_scope=APPLICANT,
        )

        self.assertEqual(
            audit.serialization_before_origin,
            SerializationOrigin.PRIMARY_VISIBLE_EVIDENCE,
        )
        self.assertEqual(audit.primary_evidence_value, "unknown")
        self.assertTrue(audit.before_is_explicit_visible_value)
        self.assertFalse(audit.before_is_output_default)

    def test_recovered_visible_none_remains_distinct_from_same_serialized_default(self):
        primary = resolved_field(
            "risk_flags",
            None,
            state=FieldState.UNKNOWN,
        )
        winner = visible_candidate(value="none")

        audit, validation = recovered_field_audit(
            primary=primary,
            serialization_before="none",
            serialization_after="none",
            candidate=winner,
            recovery_source="targeted_rapidocr",
            linked_recovery_scope=APPLICANT,
            validation_policy=policy(),
        )

        self.assertTrue(validation.accepted)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(
            audit.serialization_before_origin,
            SerializationOrigin.OUTPUT_DEFAULT,
        )
        self.assertEqual(
            audit.serialization_after_origin,
            SerializationOrigin.RECOVERED_VISIBLE_EVIDENCE,
        )
        self.assertEqual(audit.final_evidence_state, FieldState.RESOLVED)
        self.assertEqual(audit.final_evidence_value, "none")
        self.assertIs(audit.winning_evidence, winner)
        self.assertEqual(audit.ocr_provenance, winner.ocr_provenance)
        self.assertEqual(audit.observed_applicant_scopes, (APPLICANT,))
        self.assertEqual(audit.linked_recovery_scope, APPLICANT)
        self.assertTrue(audit.after_is_explicit_visible_value)
        self.assertFalse(audit.after_is_output_default)

    def test_candidate_validation_accepts_only_complete_matching_provenance(self):
        winner = visible_candidate()

        result = validate_recovered_candidate(winner, policy())

        self.assertTrue(result.accepted)
        self.assertEqual(result.failures, ())
        self.assertEqual(result.observed_applicant_scopes, (APPLICANT,))
        self.assertEqual(
            result.provenance_fingerprints,
            tuple(item.fingerprint for item in winner.ocr_provenance),
        )

    def test_candidate_validation_accepts_crop_mapped_physical_provenance(self):
        view_box = Rect(1, 2, 20, 10)
        candidate = CandidateEvidence(
            field_name="risk_flags",
            value="none",
            evidence_type=EvidenceType.INTAKE_FORM,
            page_index=2,
            box=view_box,
            legible=True,
            superseded=False,
            ocr_confidence=0.97,
            applicant_hint=APPLICANT,
            ocr_provenance=(
                make_ocr_provenance(
                    source_sha256=SOURCE_SHA256,
                    page_index=2,
                    view_box=view_box,
                    applicant_scope=APPLICANT,
                    route_id="targeted_rapidocr_crop",
                    engine_id="rapidocr:pinned-test-models",
                    view_id="risk_footer_crop",
                    transform=CoordinateTransform.crop_translation(
                        left=100,
                        upper=300,
                    ),
                ),
            ),
        )

        result = validate_recovered_candidate(candidate, policy())

        self.assertTrue(result.accepted)
        self.assertNotEqual(
            candidate.box,
            candidate.ocr_provenance[0].observation.box,
        )

    def test_candidate_validation_accepts_physical_candidate_box(self):
        view_box = Rect(1, 2, 20, 10)
        transform = CoordinateTransform.crop_translation(
            left=100,
            upper=300,
        )
        provenance = make_ocr_provenance(
            source_sha256=SOURCE_SHA256,
            page_index=2,
            view_box=view_box,
            applicant_scope=APPLICANT,
            route_id="targeted_rapidocr_crop",
            engine_id="rapidocr:pinned-test-models",
            view_id="risk_footer_crop",
            transform=transform,
        )
        candidate = CandidateEvidence(
            field_name="risk_flags",
            value="none",
            evidence_type=EvidenceType.INTAKE_FORM,
            page_index=2,
            box=provenance.observation.box,
            legible=True,
            superseded=False,
            ocr_confidence=0.97,
            applicant_hint=APPLICANT,
            ocr_provenance=(provenance,),
        )

        result = validate_recovered_candidate(candidate, policy())

        self.assertTrue(result.accepted)

    def test_candidate_validation_fails_closed_for_missing_candidate(self):
        result = validate_recovered_candidate(None, policy())

        self.assertFalse(result.accepted)
        self.assertEqual(
            result.failures,
            (CandidateValidationFailure.MISSING_CANDIDATE,),
        )

    def test_candidate_validation_fails_closed_for_missing_provenance(self):
        result = validate_recovered_candidate(
            visible_candidate(with_provenance=False),
            policy(),
        )

        self.assertFalse(result.accepted)
        self.assertIn(
            CandidateValidationFailure.MISSING_PROVENANCE,
            result.failures,
        )

    def test_candidate_validation_fails_closed_for_low_confidence(self):
        result = validate_recovered_candidate(
            visible_candidate(confidence=0.949),
            policy(),
        )

        self.assertFalse(result.accepted)
        self.assertIn(CandidateValidationFailure.LOW_CONFIDENCE, result.failures)

    def test_candidate_validation_fails_closed_for_source_page_and_scope(self):
        wrong_source = validate_recovered_candidate(
            visible_candidate(source_sha256=OTHER_SOURCE_SHA256),
            policy(),
        )
        wrong_page = validate_recovered_candidate(
            visible_candidate(page_index=3),
            policy(),
        )
        wrong_scope = validate_recovered_candidate(
            visible_candidate(
                applicant_scope="Other Applicant",
                applicant_hint="Other Applicant",
            ),
            policy(),
        )

        self.assertIn(
            CandidateValidationFailure.SOURCE_MISMATCH,
            wrong_source.failures,
        )
        self.assertIn(
            CandidateValidationFailure.PAGE_MISMATCH,
            wrong_page.failures,
        )
        self.assertIn(
            CandidateValidationFailure.APPLICANT_SCOPE_MISMATCH,
            wrong_scope.failures,
        )

    def test_candidate_validation_rejects_unscoped_or_ambiguous_applicant(self):
        unscoped = validate_recovered_candidate(
            visible_candidate(
                applicant_scope=None,
                applicant_hint=None,
            ),
            policy(),
        )
        ambiguous_candidate = CandidateEvidence(
            field_name="risk_flags",
            value="none",
            evidence_type=EvidenceType.INTAKE_FORM,
            page_index=2,
            box=BOX,
            legible=True,
            superseded=False,
            ocr_confidence=0.97,
            applicant_hint=None,
            ocr_provenance=(
                make_ocr_provenance(
                    source_sha256=SOURCE_SHA256,
                    page_index=2,
                    view_box=BOX,
                    applicant_scope=APPLICANT,
                    route_id="targeted_rapidocr",
                    engine_id="rapidocr:pinned-test-models",
                    view_id="rendered_page",
                ),
                make_ocr_provenance(
                    source_sha256=SOURCE_SHA256,
                    page_index=2,
                    view_box=BOX,
                    applicant_scope="Other Applicant",
                    route_id="targeted_rapidocr_retry",
                    engine_id="rapidocr:pinned-test-models",
                    view_id="rendered_page",
                ),
            ),
        )
        ambiguous = validate_recovered_candidate(ambiguous_candidate, policy())

        self.assertIn(
            CandidateValidationFailure.APPLICANT_SCOPE_MISSING,
            unscoped.failures,
        )
        self.assertIn(
            CandidateValidationFailure.APPLICANT_SCOPE_MISMATCH,
            ambiguous.failures,
        )

    def test_candidate_validation_rejects_unprovenanced_candidate_box(self):
        result = validate_recovered_candidate(
            visible_candidate(provenance_box=Rect(1, 2, 4, 6)),
            policy(),
        )

        self.assertIn(
            CandidateValidationFailure.CANDIDATE_BOX_NOT_PROVENANCED,
            result.failures,
        )

    def test_candidate_validation_rejects_ineligible_visible_evidence(self):
        illegible = validate_recovered_candidate(
            visible_candidate(legible=False),
            policy(),
        )
        superseded = validate_recovered_candidate(
            visible_candidate(superseded=True),
            policy(),
        )
        forbidden = validate_recovered_candidate(
            visible_candidate(cues=("sample_denial_watermark",)),
            policy(),
        )

        self.assertIn(CandidateValidationFailure.ILLEGIBLE, illegible.failures)
        self.assertIn(
            CandidateValidationFailure.SUPERSEDED,
            superseded.failures,
        )
        self.assertIn(
            CandidateValidationFailure.FORBIDDEN_VISUAL_CUE,
            forbidden.failures,
        )

    def test_recovered_builder_returns_no_audit_when_validation_fails(self):
        primary = resolved_field(
            "risk_flags",
            None,
            state=FieldState.UNKNOWN,
        )

        audit, result = recovered_field_audit(
            primary=primary,
            serialization_before="none",
            serialization_after="hazard",
            candidate=visible_candidate(
                value="hazard",
                confidence=0.50,
            ),
            recovery_source="targeted_rapidocr",
            linked_recovery_scope=APPLICANT,
            validation_policy=policy(),
        )

        self.assertIsNone(audit)
        self.assertFalse(result.accepted)

    def test_recovered_builder_rejects_non_unknown_primary_state(self):
        winner = visible_candidate(value="none")
        for primary_state, primary_value in (
            (FieldState.RESOLVED, "none"),
            (FieldState.CONTESTED, None),
        ):
            with self.subTest(primary_state=primary_state):
                primary = resolved_field(
                    "risk_flags",
                    primary_value,
                    state=primary_state,
                )

                audit, result = recovered_field_audit(
                    primary=primary,
                    serialization_before="none",
                    serialization_after="none",
                    candidate=winner,
                    recovery_source="targeted_rapidocr",
                    linked_recovery_scope=APPLICANT,
                    validation_policy=policy(),
                )

                self.assertIsNone(audit)
                self.assertIn(
                    CandidateValidationFailure.PRIMARY_STATE_NOT_UNKNOWN,
                    result.failures,
                )

    def test_recovered_builder_binds_serialized_value_and_route(self):
        primary = resolved_field(
            "risk_flags",
            None,
            state=FieldState.UNKNOWN,
        )
        winner = visible_candidate(value="none")

        wrong_value, wrong_value_result = recovered_field_audit(
            primary=primary,
            serialization_before="none",
            serialization_after="hazard",
            candidate=winner,
            recovery_source="targeted_rapidocr",
            linked_recovery_scope=APPLICANT,
            validation_policy=policy(),
        )
        wrong_route, wrong_route_result = recovered_field_audit(
            primary=primary,
            serialization_before="none",
            serialization_after="none",
            candidate=winner,
            recovery_source="unrelated_route",
            linked_recovery_scope=APPLICANT,
            validation_policy=policy(),
        )

        self.assertIsNone(wrong_value)
        self.assertIn(
            CandidateValidationFailure.SERIALIZATION_VALUE_MISMATCH,
            wrong_value_result.failures,
        )
        self.assertIsNone(wrong_route)
        self.assertIn(
            CandidateValidationFailure.RECOVERY_ROUTE_MISMATCH,
            wrong_route_result.failures,
        )

    def test_recovery_route_and_candidate_box_need_one_provenance_witness(self):
        other_box = Rect(100, 120, 180, 142)
        candidate = CandidateEvidence(
            field_name="risk_flags",
            value="none",
            evidence_type=EvidenceType.INTAKE_FORM,
            page_index=2,
            box=BOX,
            legible=True,
            superseded=False,
            ocr_confidence=0.97,
            applicant_hint=APPLICANT,
            ocr_provenance=(
                make_ocr_provenance(
                    source_sha256=SOURCE_SHA256,
                    page_index=2,
                    view_box=BOX,
                    applicant_scope=APPLICANT,
                    route_id="unrelated_route",
                    engine_id="rapidocr:pinned-test-models",
                    view_id="rendered_page",
                ),
                make_ocr_provenance(
                    source_sha256=SOURCE_SHA256,
                    page_index=2,
                    view_box=other_box,
                    applicant_scope=APPLICANT,
                    route_id="targeted_rapidocr",
                    engine_id="rapidocr:pinned-test-models",
                    view_id="supporting_crop",
                ),
            ),
        )

        audit, result = recovered_field_audit(
            primary=resolved_field(
                "risk_flags",
                None,
                state=FieldState.UNKNOWN,
            ),
            serialization_before="none",
            serialization_after="none",
            candidate=candidate,
            recovery_source="targeted_rapidocr",
            linked_recovery_scope=APPLICANT,
            validation_policy=policy(),
        )

        self.assertIsNone(audit)
        self.assertIn(
            CandidateValidationFailure.RECOVERY_ROUTE_MISMATCH,
            result.failures,
        )

    def test_candidate_validation_rejects_non_ocr_sources(self):
        non_ocr_source = visible_candidate(source="text_layer")
        text_layer_type = visible_candidate(
            evidence_type=EvidenceType.TEXT_LAYER,
        )

        source_result = validate_recovered_candidate(
            non_ocr_source,
            policy(),
        )
        type_result = validate_recovered_candidate(
            text_layer_type,
            policy(),
        )

        self.assertIn(
            CandidateValidationFailure.NON_OCR_SOURCE,
            source_result.failures,
        )
        self.assertIn(
            CandidateValidationFailure.NON_OCR_EVIDENCE_TYPE,
            type_result.failures,
        )

    def test_visible_repair_audits_resolved_primary_and_retains_both_winners(self):
        primary_winner = visible_candidate(
            field_name="applicant_name",
            value="Zed Zornax",
            route_id="primary_visible_ocr",
        )
        repair_winner = visible_candidate(
            field_name="applicant_name",
            value=APPLICANT,
            route_id="primary_visible_ocr",
        )
        primary = resolved_field(
            "applicant_name",
            "Zed Zornax",
            winner=primary_winner,
        )

        audit, validation = visible_repair_field_audit(
            primary=primary,
            serialization_before="Zed Zornax",
            serialization_after=APPLICANT,
            candidate=repair_winner,
            recovery_source="primary_visible_ocr",
            linked_recovery_scope=APPLICANT,
            validation_policy=policy(field_name="applicant_name"),
        )

        self.assertTrue(validation.accepted)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit.primary_state, FieldState.RESOLVED)
        self.assertEqual(audit.primary_evidence_value, "Zed Zornax")
        self.assertIs(audit.primary_winning_evidence, primary_winner)
        self.assertEqual(audit.final_evidence_value, APPLICANT)
        self.assertIs(audit.winning_evidence, repair_winner)
        self.assertEqual(
            audit.serialization_after_origin,
            SerializationOrigin.RECOVERED_VISIBLE_EVIDENCE,
        )

    def test_recovered_builder_rejects_linked_scope_policy_mismatch(self):
        primary = resolved_field(
            "risk_flags",
            None,
            state=FieldState.UNKNOWN,
        )

        audit, result = recovered_field_audit(
            primary=primary,
            serialization_before="none",
            serialization_after="none",
            candidate=visible_candidate(),
            recovery_source="targeted_rapidocr",
            linked_recovery_scope="Other Applicant",
            validation_policy=policy(),
        )

        self.assertIsNone(audit)
        self.assertEqual(
            result.failures,
            (CandidateValidationFailure.APPLICANT_SCOPE_MISMATCH,),
        )

    def test_unscoped_observation_retains_independent_linked_recovery_scope(self):
        primary = resolved_field(
            "fee_status",
            None,
            state=FieldState.UNKNOWN,
        )
        candidate = visible_candidate(
            field_name="fee_status",
            value="unknown",
            applicant_scope=None,
            applicant_hint=None,
        )

        audit, result = recovered_field_audit(
            primary=primary,
            serialization_before="paid",
            serialization_after="unknown",
            candidate=candidate,
            recovery_source="targeted_rapidocr",
            linked_recovery_scope=APPLICANT,
            validation_policy=policy(
                field_name="fee_status",
                applicant_scope=None,
            ),
        )

        self.assertTrue(result.accepted)
        self.assertIsNotNone(audit)
        assert audit is not None
        self.assertEqual(audit.observed_applicant_scopes, ())
        self.assertEqual(audit.linked_recovery_scope, APPLICANT)

    def test_overlay_copies_and_freezes_field_mapping(self):
        primary = resolved_field(
            "risk_flags",
            None,
            state=FieldState.UNKNOWN,
        )
        audit = unchanged_field_audit(
            primary=primary,
            serialized_value="none",
            linked_recovery_scope=APPLICANT,
        )
        original = {"risk_flags": audit}

        overlay = RecoveryAuditOverlay(
            case_id="MIB-000001",
            fields=original,
        )
        original.clear()

        self.assertIs(overlay.field("risk_flags"), audit)
        with self.assertRaises(TypeError):
            overlay.fields["fee_status"] = audit

    def test_overlay_rejects_duplicate_fields(self):
        audit = unchanged_field_audit(
            primary=resolved_field(
                "risk_flags",
                None,
                state=FieldState.UNKNOWN,
            ),
            serialized_value="none",
            linked_recovery_scope=APPLICANT,
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            RecoveryAuditOverlay.from_fields(
                case_id="MIB-000001",
                fields=(audit, audit),
            )

    def test_visible_recovery_result_requires_matching_case_id(self):
        audit = unchanged_field_audit(
            primary=resolved_field(
                "risk_flags",
                None,
                state=FieldState.UNKNOWN,
            ),
            serialized_value="none",
            linked_recovery_scope=APPLICANT,
        )
        overlay = RecoveryAuditOverlay.from_fields(
            case_id="MIB-000001",
            fields=(audit,),
        )
        matching_row = PredictionRow.from_mapping(
            {
                "case_id": "MIB-000001",
                "adjudication": "NEEDS_REVIEW",
            }
        )
        other_row = PredictionRow.from_mapping(
            {
                "case_id": "MIB-000002",
                "adjudication": "NEEDS_REVIEW",
            }
        )

        result = VisibleRecoveryResult(row=matching_row, audit=overlay)

        self.assertIs(result.audit, overlay)
        with self.assertRaisesRegex(ValueError, "case IDs"):
            VisibleRecoveryResult(row=other_row, audit=overlay)

    def test_visible_recovery_result_rejects_row_audit_value_mismatch(self):
        primary = resolved_field(
            "risk_flags",
            None,
            state=FieldState.UNKNOWN,
        )
        audit, validation = recovered_field_audit(
            primary=primary,
            serialization_before="none",
            serialization_after="none",
            candidate=visible_candidate(value="none"),
            recovery_source="targeted_rapidocr",
            linked_recovery_scope=APPLICANT,
            validation_policy=policy(),
        )
        self.assertTrue(validation.accepted)
        assert audit is not None
        overlay = RecoveryAuditOverlay.from_fields(
            case_id="MIB-000001",
            fields=(audit,),
        )
        mismatched_row = PredictionRow.from_mapping(
            {
                "case_id": "MIB-000001",
                "risk_flags": "hazard",
                "adjudication": "NEEDS_REVIEW",
            }
        )

        with self.assertRaisesRegex(ValueError, "risk_flags"):
            VisibleRecoveryResult(row=mismatched_row, audit=overlay)

    def test_explicit_serialization_is_deterministic_and_json_safe(self):
        primary = resolved_field(
            "risk_flags",
            None,
            state=FieldState.UNKNOWN,
        )
        audit, validation = recovered_field_audit(
            primary=primary,
            serialization_before="none",
            serialization_after="none",
            candidate=visible_candidate(value="none"),
            recovery_source="targeted_rapidocr",
            linked_recovery_scope=APPLICANT,
            validation_policy=policy(),
        )
        self.assertTrue(validation.accepted)
        assert audit is not None
        overlay = RecoveryAuditOverlay.from_fields(
            case_id="MIB-000001",
            fields=(audit,),
        )
        row = PredictionRow.from_mapping(
            {
                "case_id": "MIB-000001",
                "risk_flags": "none",
                "adjudication": "NEEDS_REVIEW",
            }
        )
        result = VisibleRecoveryResult(row=row, audit=overlay)

        first = result.to_dict()
        second = result.to_dict()

        self.assertEqual(first, second)
        self.assertEqual(
            first["audit"]["fields"]["risk_flags"]["ocr_provenance"],
            [
                item.to_dict()
                for item in audit.winning_evidence.ocr_provenance
            ],
        )
        json.dumps(first, sort_keys=True)


if __name__ == "__main__":
    unittest.main()
