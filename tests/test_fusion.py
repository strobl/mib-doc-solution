import itertools
import unittest
from dataclasses import FrozenInstanceError, replace

from mib_pipeline.extraction import CandidateEvidence, EvidenceType
from mib_pipeline.fusion import (
    EvidenceFuser,
    candidate_has_complete_provenance,
    candidates_share_physical_observation,
)
from mib_pipeline.ingestion import Rect
from mib_pipeline.provenance import make_ocr_provenance
from mib_pipeline.resolution import (
    CaseLinker,
    EvidencePrecedenceHierarchy,
    EvidencePrecedenceResolver,
    LinkedCase,
)


CASE_ID = "MIB-000001"
APPLICANT = "Zed Zarnax"
SOURCE_SHA256 = "a" * 64


def candidate(
    field_name,
    value,
    evidence_type=EvidenceType.INTAKE_FORM,
    *,
    page=0,
    left=10,
    top=20,
    confidence=0.8,
    legible=True,
    superseded=False,
    cues=(),
    case_hint=CASE_ID,
    applicant_hint=APPLICANT,
    provenance=True,
    route="primary",
    view="page",
    source="visible_ocr",
    provenance_applicant=None,
):
    box = Rect(left, top, left + 100, top + 20)
    if provenance:
        scope = (
            applicant_hint
            if provenance_applicant is None
            else provenance_applicant
        )
        ocr_provenance = (
            make_ocr_provenance(
                source_sha256=SOURCE_SHA256,
                page_index=page,
                view_box=box,
                applicant_scope=scope,
                route_id=route,
                engine_id="test-engine",
                view_id=view,
            ),
        )
    else:
        ocr_provenance = ()
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=page,
        box=box,
        legible=legible,
        superseded=superseded,
        ocr_confidence=confidence,
        visual_cues=tuple(cues),
        source=source,
        case_id_hint=case_hint,
        applicant_hint=applicant_hint,
        ocr_provenance=ocr_provenance,
    )


def resolve(field_name, *candidates):
    return EvidenceFuser.resolve(
        field_name,
        candidates,
        ranker=EvidencePrecedenceHierarchy,
        expected_case_id=CASE_ID,
        active_applicant=APPLICANT,
    )


class EvidenceFuserAuthorityTests(unittest.TestCase):
    def test_higher_authority_point_seven_beats_lower_rank_point_nine_nine(self):
        higher = candidate(
            "fee_status",
            "paid",
            EvidenceType.INTAKE_FORM,
            confidence=0.70,
        )
        lower = candidate(
            "fee_status",
            "unpaid",
            EvidenceType.REGISTRY_EXTRACT,
            confidence=0.99,
            page=1,
        )

        decision = resolve("fee_status", lower, higher)

        self.assertEqual(decision.state, "resolved")
        self.assertEqual(decision.value, "paid")
        self.assertIs(decision.winner, higher)
        self.assertEqual(decision.trace.winning_rank, 2)
        self.assertEqual(
            decision.trace.safety_count("lower_authority_ignored_count"),
            1,
        )

    def test_signed_rank_one_is_binding(self):
        signed = candidate(
            "adjudication",
            "APPROVED",
            EvidenceType.SIGNED_MANUAL_NOTE,
            confidence=0.51,
        )
        intake = candidate(
            "adjudication",
            "DENIED",
            EvidenceType.INTAKE_FORM,
            confidence=0.99,
            page=1,
        )

        decision = resolve("adjudication", intake, signed)

        self.assertEqual(decision.value, "APPROVED")
        self.assertEqual(decision.trace.winning_rank, 1)
        self.assertEqual(
            decision.trace.safety_count("binding_rank_candidate_count"),
            1,
        )

    def test_rank_one_conflict_stays_contested_even_when_rank_two_agrees(self):
        first = candidate(
            "adjudication",
            "APPROVED",
            EvidenceType.ADJUDICATOR_STAMP,
        )
        second = candidate(
            "adjudication",
            "DENIED",
            EvidenceType.SIGNED_MANUAL_NOTE,
            page=1,
        )
        lower = candidate(
            "adjudication",
            "APPROVED",
            EvidenceType.INTAKE_FORM,
            page=2,
            confidence=0.99,
        )

        decision = resolve("adjudication", lower, second, first)

        self.assertEqual(decision.state, "contested")
        self.assertIsNone(decision.value)
        self.assertEqual(decision.trace.winning_rank, 1)
        self.assertGreater(decision.trace.entropy_bits, 0.0)

    def test_same_rank_irreconcilable_values_are_contested(self):
        decision = resolve(
            "home_world",
            candidate("home_world", "Mars", page=0),
            candidate("home_world", "Europa", page=1),
        )

        self.assertEqual(decision.state, "contested")
        self.assertEqual(decision.trace.disagreement_count, 1)
        self.assertEqual(
            decision.trace.safety_count("same_rank_conflict_count"),
            1,
        )


class EvidenceFuserIndependenceTests(unittest.TestCase):
    def test_complete_provenance_precedes_confidence_for_equal_values(self):
        complete = candidate(
            "fee_status",
            "paid",
            confidence=0.70,
            provenance=True,
        )
        incomplete = candidate(
            "fee_status",
            "paid",
            page=1,
            confidence=0.99,
            provenance=False,
        )

        decision = resolve("fee_status", incomplete, complete)

        self.assertIs(decision.winner, complete)
        self.assertEqual(decision.trace.provenance_completeness, 0.5)

    def test_ten_ocr_views_of_one_crop_count_as_one_vote(self):
        repeated = tuple(
            candidate(
                "fee_status",
                "paid",
                route=f"route-{index}",
                view=f"view-{index}",
                confidence=0.70 + index / 100,
            )
            for index in range(10)
        )
        conflicting_independent = candidate(
            "fee_status",
            "unpaid",
            page=1,
            route="independent",
        )

        repeated_only = resolve("fee_status", *repeated)
        with_conflict = resolve(
            "fee_status",
            *repeated,
            conflicting_independent,
        )

        self.assertEqual(repeated_only.state, "resolved")
        self.assertEqual(repeated_only.trace.observation_count, 1)
        self.assertEqual(repeated_only.trace.independent_evidence_count, 1)
        self.assertEqual(repeated_only.trace.correlated_candidate_count, 9)
        self.assertEqual(repeated_only.trace.independent_agreement_count, 1)
        # Ten correlated reads cannot outvote one different physical record.
        self.assertEqual(with_conflict.state, "contested")
        self.assertEqual(with_conflict.trace.independent_evidence_count, 2)

    def test_independent_agreement_is_reported_by_page_and_observation(self):
        decision = resolve(
            "visa_class",
            candidate("visa_class", "XW-1", page=0, route="page-zero"),
            candidate("visa_class", "xw-1", page=2, route="page-two"),
        )

        self.assertEqual(decision.state, "resolved")
        self.assertEqual(decision.value, "XW-1")
        self.assertEqual(decision.trace.observation_count, 2)
        self.assertEqual(decision.trace.independent_evidence_count, 2)
        self.assertEqual(decision.trace.independent_agreement_count, 2)
        self.assertEqual(decision.trace.independent_page_count, 2)
        self.assertEqual(decision.trace.independent_evidence_type_count, 1)
        self.assertEqual(decision.trace.provenance_completeness, 1.0)

    def test_identical_unprovenanced_boxes_are_conservatively_correlated(self):
        copies = (
            candidate(
                "species_code",
                "TRIANGULAN",
                provenance=False,
                source="visible_ocr",
            ),
            candidate(
                "species_code",
                "TRIANGULAN",
                provenance=False,
                source="visible_ocr",
            ),
        )

        decision = resolve("species_code", *copies)

        self.assertEqual(decision.trace.independent_evidence_count, 1)
        self.assertEqual(decision.trace.correlated_candidate_count, 1)
        self.assertEqual(decision.trace.provenance_completeness, 0.0)

    def test_one_pixel_route_jitter_is_one_physical_vote(self):
        primary = candidate(
            "fee_status",
            "paid",
            left=10,
            route="primary",
        )
        rapid = candidate(
            "fee_status",
            "paid",
            left=11,
            route="targeted_rapidocr",
        )
        independent = candidate(
            "fee_status",
            "unpaid",
            page=1,
            route="independent",
        )

        decision = resolve("fee_status", primary, rapid, independent)

        self.assertEqual(decision.state, "contested")
        self.assertEqual(decision.trace.independent_evidence_count, 2)
        self.assertEqual(decision.trace.correlated_candidate_count, 1)

    def test_neighboring_regions_are_not_geometrically_correlated(self):
        left = candidate("fee_status", "paid", left=10, route="left")
        neighbor = candidate(
            "fee_status",
            "unpaid",
            left=120,
            route="neighbor",
        )

        decision = resolve("fee_status", left, neighbor)

        self.assertEqual(decision.state, "contested")
        self.assertEqual(decision.trace.independent_evidence_count, 2)
        self.assertEqual(decision.trace.correlated_candidate_count, 0)

    def test_mismatched_candidate_box_is_not_complete_provenance(self):
        mismatched = replace(
            candidate("fee_status", "paid", confidence=0.99),
            box=Rect(500, 500, 600, 520),
        )
        complete = candidate(
            "fee_status",
            "paid",
            page=1,
            confidence=0.70,
        )

        decision = resolve("fee_status", mismatched, complete)

        self.assertIs(decision.winner, complete)
        self.assertEqual(decision.trace.provenance_complete_count, 1)
        self.assertEqual(decision.trace.provenance_completeness, 0.5)
        self.assertFalse(
            candidate_has_complete_provenance(
                mismatched,
                expected_case_id=CASE_ID,
                active_applicant=APPLICANT,
            )
        )

    def test_mixed_provenance_local_record_ignores_semantic_ocr_metadata(self):
        first = candidate(
            "fee_status",
            "paid",
            EvidenceType.INTAKE_FORM,
            left=10,
            applicant_hint="Boris Beta",
            provenance_applicant="Boris Beta",
            cues=("record_id:b",),
        )
        variant = candidate(
            "fee_status",
            "paid",
            EvidenceType.BIOMETRIC_SLIP,
            left=300,
            applicant_hint="Boris Beto",
            provenance_applicant="Boris Beto",
            cues=("record_id:b",),
            provenance=False,
        )

        self.assertTrue(
            candidates_share_physical_observation(first, variant)
        )
        decision = EvidenceFuser.resolve(
            "fee_status",
            (first, variant),
            ranker=lambda evidence_type: 1,
            expected_case_id=CASE_ID,
            active_applicant="Boris Beta",
            active_applicant_aliases=("Boris Beto",),
        )

        self.assertEqual(decision.state, "resolved")
        self.assertEqual(decision.trace.independent_evidence_count, 1)
        self.assertEqual(decision.trace.correlated_candidate_count, 1)

    def test_mixed_provenance_same_crop_is_one_physical_vote(self):
        complete = candidate(
            "fee_status",
            "paid",
            left=10,
            route="primary",
        )
        unprovenanced = candidate(
            "fee_status",
            "paid",
            left=11,
            route="secondary",
            provenance=False,
            case_hint=None,
        )

        decision = resolve("fee_status", complete, unprovenanced)

        self.assertEqual(decision.state, "resolved")
        self.assertEqual(decision.trace.independent_evidence_count, 1)
        self.assertEqual(decision.trace.correlated_candidate_count, 1)


class EvidenceFuserCorrectionTests(unittest.TestCase):
    def test_nested_page_box_cannot_bridge_a_local_correction(self):
        original = candidate(
            "sponsor_id",
            "SPN-0007",
            route="field",
        )
        page_sized = replace(
            candidate(
                "sponsor_id",
                "SPN-1234",
                route="page",
                cues=("correction",),
                provenance=False,
            ),
            box=Rect(0, 0, 1000, 1000),
        )

        self.assertFalse(
            candidates_share_physical_observation(original, page_sized)
        )
        decision = resolve("sponsor_id", original, page_sized)

        self.assertEqual(decision.state, "contested")
        self.assertEqual(decision.trace.independent_evidence_count, 2)
        self.assertEqual(
            decision.trace.safety_count(
                "local_correction_override_count"
            ),
            0,
        )

    def test_correction_only_overrides_same_physical_observation(self):
        original = candidate("sponsor_id", "SPN-0007", route="primary")
        correction = candidate(
            "sponsor_id",
            "SPN-1234",
            route="refinement",
            cues=("correction",),
            confidence=0.75,
        )

        decision = resolve("sponsor_id", original, correction)

        self.assertEqual(decision.state, "resolved")
        self.assertEqual(decision.value, "SPN-1234")
        self.assertIs(decision.winner, correction)
        self.assertEqual(
            decision.trace.safety_count("local_correction_override_count"),
            1,
        )

    def test_overlap_chain_cannot_bridge_a_distant_correction(self):
        chain = tuple(
            candidate(
                "sponsor_id",
                "SPN-1234" if index == 10 else "SPN-0007",
                left=10 + index * 10,
                provenance=False,
                route=f"drift-{index}",
                cues=("correction",) if index == 10 else (),
            )
            for index in range(11)
        )

        decision = resolve("sponsor_id", *chain)

        self.assertEqual(decision.state, "contested")
        self.assertGreater(decision.trace.independent_evidence_count, 1)
        self.assertEqual(
            resolve("sponsor_id", *reversed(chain)),
            decision,
        )

    def test_unrelated_correction_does_not_globally_erase_evidence(self):
        ordinary = candidate("sponsor_id", "SPN-0007", page=0)
        unrelated_correction = candidate(
            "sponsor_id",
            "SPN-1234",
            page=1,
            cues=("correction",),
        )

        decision = resolve(
            "sponsor_id",
            ordinary,
            unrelated_correction,
        )

        self.assertEqual(decision.state, "contested")
        self.assertEqual(
            decision.trace.safety_count("local_correction_override_count"),
            0,
        )

    def test_explicit_local_record_can_bind_different_boxes(self):
        original = candidate(
            "sponsor_id",
            "SPN-0007",
            left=10,
            cues=("record_id:application-7",),
        )
        correction = candidate(
            "sponsor_id",
            "SPN-1234",
            left=300,
            cues=("record_id:application-7", "correction"),
        )

        decision = resolve("sponsor_id", correction, original)

        self.assertEqual(decision.state, "resolved")
        self.assertEqual(decision.value, "SPN-1234")


class EvidenceFuserEligibilityTests(unittest.TestCase):
    def test_orphan_case_scope_is_fail_closed_without_active_case(self):
        scoped = candidate("fee_status", "paid")

        decision = EvidenceFuser.resolve(
            "fee_status",
            (scoped,),
            ranker=EvidencePrecedenceHierarchy,
            expected_case_id=None,
            active_applicant=APPLICANT,
        )

        self.assertEqual(decision.state, "unknown")
        self.assertEqual(
            decision.trace.veto_count("orphan_case_scope"),
            1,
        )

    def test_fuser_does_not_establish_case_identity_without_linker(self):
        identity = candidate(
            "case_id",
            CASE_ID,
            case_hint=None,
            applicant_hint=None,
            provenance_applicant=None,
        )

        decision = EvidenceFuser.resolve(
            "case_id",
            (identity,),
            ranker=EvidencePrecedenceHierarchy,
            expected_case_id=None,
            active_applicant=None,
        )

        self.assertEqual(decision.state, "unknown")
        self.assertEqual(
            decision.trace.veto_count("case_identity_not_linked"),
            1,
        )

    def test_orphan_applicant_scope_is_fail_closed_without_active_link(self):
        scoped = candidate("fee_status", "paid")

        decision = EvidenceFuser.resolve(
            "fee_status",
            (scoped,),
            ranker=EvidencePrecedenceHierarchy,
            expected_case_id=CASE_ID,
            active_applicant=None,
        )

        self.assertEqual(decision.state, "unknown")
        self.assertEqual(
            decision.trace.veto_count("orphan_applicant_scope"),
            1,
        )

    def test_fuser_does_not_establish_applicant_identity_without_linker(self):
        identity = candidate(
            "applicant_name",
            APPLICANT,
            applicant_hint=None,
            provenance_applicant=None,
        )

        decision = EvidenceFuser.resolve(
            "applicant_name",
            (identity,),
            ranker=EvidencePrecedenceHierarchy,
            expected_case_id=CASE_ID,
            active_applicant=None,
        )

        self.assertEqual(decision.state, "unknown")
        self.assertEqual(
            decision.trace.veto_count("applicant_identity_not_linked"),
            1,
        )

    def test_only_production_visible_ocr_source_can_influence_fusion(self):
        injected = candidate(
            "fee_status",
            "unpaid",
            EvidenceType.INTAKE_FORM,
            source="secondary_visible_ocr",
            confidence=0.99,
        )
        visible = candidate(
            "fee_status",
            "paid",
            EvidenceType.REGISTRY_EXTRACT,
            page=1,
            confidence=0.60,
        )

        decision = resolve("fee_status", injected, visible)

        self.assertEqual(decision.value, "paid")
        self.assertIs(decision.winner, visible)
        self.assertEqual(
            decision.trace.veto_count("non_production_ocr_source"),
            1,
        )
        self.assertEqual(
            decision.trace.safety_count(
                "veto_non_production_ocr_source_count"
            ),
            1,
        )

    def test_text_layer_is_diagnostic_and_never_resolves(self):
        diagnostic = candidate(
            "visa_class",
            "XW-1",
            EvidenceType.TEXT_LAYER,
            confidence=1.0,
        )

        decision = resolve("visa_class", diagnostic)

        self.assertEqual(decision.state, "unknown")
        self.assertIsNone(decision.value)
        self.assertEqual(
            decision.trace.veto_count("text_layer_diagnostic_only"),
            1,
        )
        self.assertEqual(decision.considered, (diagnostic,))

    def test_wrong_case_and_wrong_applicant_are_excluded_exactly(self):
        wrong_case = candidate(
            "fee_status",
            "unpaid",
            case_hint="MIB-000999",
            applicant_hint=APPLICANT,
        )
        wrong_applicant = candidate(
            "fee_status",
            "unpaid",
            applicant_hint="Other Person",
        )
        right = candidate("fee_status", "paid", page=2)

        decision = resolve(
            "fee_status",
            wrong_applicant,
            right,
            wrong_case,
        )

        self.assertEqual(decision.value, "paid")
        self.assertEqual(decision.trace.veto_count("wrong_case"), 1)
        self.assertEqual(decision.trace.veto_count("wrong_applicant"), 1)

    def test_wrong_provenance_applicant_scope_is_excluded_without_hint(self):
        wrong = candidate(
            "fee_status",
            "unpaid",
            applicant_hint=None,
            provenance_applicant="Other Person",
        )
        right = candidate("fee_status", "paid", page=1)

        decision = resolve("fee_status", wrong, right)

        self.assertEqual(decision.value, "paid")
        self.assertEqual(decision.trace.veto_count("wrong_applicant"), 1)

    def test_linker_selected_applicant_alias_is_eligible(self):
        alias = "Zed Zarnaks"
        selected_alias = candidate(
            "fee_status",
            "paid",
            applicant_hint=alias,
            provenance_applicant=alias,
        )
        foreign = candidate(
            "fee_status",
            "unpaid",
            page=1,
            applicant_hint="Other Person",
        )

        decision = EvidenceFuser.resolve(
            "fee_status",
            (foreign, selected_alias),
            ranker=EvidencePrecedenceHierarchy,
            expected_case_id=CASE_ID,
            active_applicant=APPLICANT,
            active_applicant_aliases=(alias,),
        )

        self.assertEqual(decision.value, "paid")
        self.assertIs(decision.winner, selected_alias)
        self.assertEqual(decision.trace.veto_count("wrong_applicant"), 1)

    def test_all_quality_and_safety_vetoes_are_traceable(self):
        rejected = (
            candidate("fee_status", "unpaid", legible=False),
            candidate(
                "fee_status",
                "unpaid",
                superseded=True,
                page=1,
            ),
            candidate(
                "fee_status",
                "unpaid",
                cues=("strikethrough",),
                page=2,
            ),
            candidate(
                "fee_status",
                "unpaid",
                cues=("sample_denial_watermark",),
                page=3,
            ),
            candidate(
                "fee_status",
                "unknown",
                source="output_default",
                page=4,
            ),
        )
        visible = candidate(
            "fee_status",
            "paid",
            EvidenceType.SPONSOR_ATTESTATION,
            page=5,
        )

        decision = resolve("fee_status", *rejected, visible)

        self.assertEqual(decision.value, "paid")
        for reason in (
            "illegible",
            "superseded",
            "struck",
            "watermarked",
            "serialization_default",
        ):
            self.assertEqual(decision.trace.veto_count(reason), 1)
        self.assertEqual(
            decision.trace.safety_count(
                "serialization_default_used_as_evidence_count"
            ),
            0,
        )


class EvidenceFuserNormalizationTests(unittest.TestCase):
    def test_risk_sets_are_canonical_and_order_independent(self):
        decision = resolve(
            "risk_flags",
            candidate(
                "risk_flags",
                "Planetary Embargo | active-warrant",
                page=0,
            ),
            candidate(
                "risk_flags",
                "active_warrant;planetary_embargo",
                page=1,
            ),
        )

        self.assertEqual(decision.state, "resolved")
        self.assertEqual(
            decision.value,
            "active_warrant|planetary_embargo",
        )
        self.assertEqual(decision.trace.independent_agreement_count, 2)

    def test_internally_contradictory_risk_observation_is_invalid(self):
        decision = resolve(
            "risk_flags",
            candidate("risk_flags", "none|active_warrant"),
        )

        self.assertEqual(decision.state, "unknown")
        self.assertEqual(decision.trace.veto_count("invalid_value"), 1)

    def test_names_use_no_fuzzy_or_punctuation_merge(self):
        whitespace_and_case = resolve(
            "applicant_name",
            candidate("applicant_name", "  Zed   Zarnax  ", page=0),
            candidate("applicant_name", "zed zarnax", page=1),
        )
        punctuation_change = resolve(
            "applicant_name",
            candidate("applicant_name", "Zed Zarnax", page=0),
            candidate("applicant_name", "Zed-Zarnax", page=1),
        )

        self.assertEqual(whitespace_and_case.state, "resolved")
        self.assertEqual(punctuation_change.state, "contested")

    def test_free_text_normalizes_only_whitespace_and_case(self):
        same = resolve(
            "declared_purpose",
            candidate("declared_purpose", " Reactor   Maintenance ", page=0),
            candidate("declared_purpose", "reactor maintenance", page=1),
        )
        semantically_similar = resolve(
            "declared_purpose",
            candidate("declared_purpose", "reactor repair", page=0),
            candidate("declared_purpose", "reactor maintenance", page=1),
        )

        self.assertEqual(same.value, "reactor maintenance")
        self.assertEqual(semantically_similar.state, "contested")

    def test_ids_and_dates_are_strict_not_fuzzy_repaired(self):
        invalid_id = resolve(
            "sponsor_id",
            candidate("sponsor_id", "SPM-12O4"),
        )
        invalid_date = resolve(
            "arrival_date",
            candidate("arrival_date", "07/26/2026"),
        )

        self.assertEqual(invalid_id.state, "unknown")
        self.assertEqual(invalid_date.state, "unknown")
        self.assertEqual(invalid_id.trace.veto_count("invalid_value"), 1)
        self.assertEqual(invalid_date.trace.veto_count("invalid_value"), 1)


class EvidenceFuserInvariantTests(unittest.TestCase):
    def test_empty_evidence_does_not_manufacture_schema_defaults(self):
        for field_name, prohibited_default in (
            ("risk_flags", "none"),
            ("fee_status", "unknown"),
            ("arrival_date", "1900-01-01"),
            ("sponsor_id", "SPN-0000"),
        ):
            with self.subTest(field_name=field_name):
                decision = resolve(field_name)
                self.assertEqual(decision.state, "unknown")
                self.assertIsNone(decision.value)
                self.assertNotEqual(decision.value, prohibited_default)
                self.assertEqual(
                    decision.trace.safety_count(
                        "serialization_default_used_as_evidence_count"
                    ),
                    0,
                )

    def test_candidate_order_does_not_change_decision_or_trace(self):
        evidence = (
            candidate("fee_status", "paid", page=0, confidence=0.75),
            candidate("fee_status", "paid", page=1, confidence=0.85),
            candidate(
                "fee_status",
                "unpaid",
                EvidenceType.REGISTRY_EXTRACT,
                page=2,
                confidence=0.99,
            ),
            candidate(
                "fee_status",
                "unknown",
                EvidenceType.TEXT_LAYER,
                page=3,
            ),
        )
        expected = resolve("fee_status", *evidence)

        for permutation in itertools.permutations(evidence):
            self.assertEqual(resolve("fee_status", *permutation), expected)

    def test_decision_and_trace_are_immutable(self):
        decision = resolve(
            "fee_status",
            candidate("fee_status", "paid"),
        )

        with self.assertRaises(FrozenInstanceError):
            decision.value = "unpaid"
        with self.assertRaises(FrozenInstanceError):
            decision.trace.winning_rank = 9

    def test_callable_ranker_is_supported(self):
        ranks = {
            EvidenceType.INTAKE_FORM: 2,
            EvidenceType.REGISTRY_EXTRACT: 5,
        }
        decision = EvidenceFuser.resolve(
            "fee_status",
            (
                candidate(
                    "fee_status",
                    "unpaid",
                    EvidenceType.REGISTRY_EXTRACT,
                    page=1,
                ),
                candidate("fee_status", "paid"),
            ),
            ranker=lambda evidence_type: ranks[evidence_type],
            expected_case_id=CASE_ID,
            active_applicant=APPLICANT,
        )

        self.assertEqual(decision.value, "paid")

    def test_empty_unresolved_case_scope_does_not_invent_a_case(self):
        decision = EvidenceFuser.resolve(
            "fee_status",
            (
                candidate(
                    "fee_status",
                    "paid",
                    case_hint=None,
                    applicant_hint=None,
                    provenance_applicant=None,
                ),
            ),
            ranker=EvidencePrecedenceHierarchy,
            expected_case_id="",
            active_applicant=None,
        )

        self.assertEqual(decision.value, "paid")

    def test_resolver_audit_does_not_credit_mismatched_box_provenance(self):
        applicant = candidate("applicant_name", APPLICANT)
        purpose = replace(
            candidate("declared_purpose", "Visit", page=1),
            box=Rect(500, 500, 600, 520),
        )
        linked = LinkedCase(
            case_id=CASE_ID,
            active_applicant=APPLICANT,
            evidence=(applicant, purpose),
            unresolved=False,
            active_applicant_aliases=(APPLICANT,),
        )

        resolved = EvidencePrecedenceResolver().resolve(linked)

        self.assertEqual(resolved.value("declared_purpose"), "visit")
        self.assertEqual(
            resolved.fusion_audit_counts["changed_field_count"],
            1,
        )
        self.assertEqual(
            resolved.fusion_audit_counts[
                "changed_field_complete_provenance_count"
            ],
            0,
        )

    def test_applicant_association_winner_is_order_independent(self):
        complete = candidate(
            "applicant_name",
            APPLICANT,
            confidence=0.70,
        )
        incomplete = candidate(
            "applicant_name",
            APPLICANT,
            page=1,
            confidence=0.99,
            provenance=False,
        )

        winners = []
        for evidence in ((complete, incomplete), (incomplete, complete)):
            linked = LinkedCase(
                case_id=CASE_ID,
                active_applicant=APPLICANT,
                evidence=evidence,
                unresolved=False,
                active_applicant_aliases=(APPLICANT,),
            )
            winners.append(
                EvidencePrecedenceResolver()
                .resolve(linked)
                .fields["applicant_name"]
                .winning_evidence
            )

        self.assertEqual(winners, [complete, complete])

    def test_provenance_jitter_with_different_name_scopes_is_one_vote(self):
        alice = "Alice Aster"
        evidence = (
            candidate(
                "applicant_name",
                alice,
                page=0,
                applicant_hint=alice,
                provenance_applicant=alice,
                confidence=0.80,
            ),
            candidate(
                "applicant_name",
                alice,
                page=1,
                applicant_hint=alice,
                provenance_applicant=alice,
                confidence=0.80,
            ),
            candidate(
                "applicant_name",
                "Boris Beta",
                page=2,
                left=10,
                applicant_hint="Boris Beta",
                provenance_applicant="Boris Beta",
                confidence=0.95,
            ),
            candidate(
                "applicant_name",
                "Boris Beto",
                page=2,
                left=11,
                applicant_hint="Boris Beto",
                provenance_applicant="Boris Beto",
                confidence=0.95,
            ),
        )

        linked = CaseLinker().link(CASE_ID, evidence)

        self.assertEqual(linked.active_applicant, alice)
        self.assertFalse(linked.unresolved)

    def test_mixed_provenance_name_jitter_cannot_multiply_linker_votes(self):
        alice = "Alice Aster"
        evidence = (
            candidate(
                "applicant_name",
                alice,
                page=0,
                applicant_hint=alice,
                provenance_applicant=alice,
                confidence=0.99,
            ),
            candidate(
                "applicant_name",
                "Boris Beta",
                page=1,
                left=10,
                applicant_hint="Boris Beta",
                provenance_applicant="Boris Beta",
                confidence=0.95,
            ),
            candidate(
                "applicant_name",
                "Boris Beto",
                page=1,
                left=11,
                applicant_hint="Boris Beto",
                confidence=0.95,
                provenance=False,
                case_hint=None,
            ),
        )

        linked = CaseLinker().link(CASE_ID, evidence)

        self.assertEqual(linked.active_applicant, alice)
        self.assertFalse(linked.unresolved)

    def test_mixed_provenance_local_record_cannot_flip_linker_winner(self):
        alice = "Alice Aster"
        evidence = (
            candidate(
                "applicant_name",
                alice,
                page=0,
                applicant_hint=alice,
                provenance_applicant=alice,
                confidence=0.99,
            ),
            candidate(
                "applicant_name",
                "Boris Beta",
                page=1,
                left=10,
                applicant_hint="Boris Beta",
                provenance_applicant="Boris Beta",
                confidence=0.95,
                cues=("record_id:b",),
            ),
            candidate(
                "applicant_name",
                "Boris Beto",
                page=1,
                left=300,
                applicant_hint="Boris Beto",
                confidence=0.95,
                provenance=False,
                cues=("record_id:b",),
            ),
        )

        linked = CaseLinker().link(CASE_ID, evidence)

        self.assertEqual(linked.active_applicant, alice)
        self.assertFalse(linked.unresolved)

    def test_linker_observation_groups_do_not_transitively_bridge_drift(self):
        chain = tuple(
            candidate(
                "applicant_name",
                "Boris Beta",
                page=2,
                left=10 + index * 10,
                applicant_hint="Boris Beta",
                provenance=False,
                route=f"drift-{index}",
            )
            for index in range(11)
        )

        groups = CaseLinker._independent_observation_groups(chain)

        self.assertGreater(len(groups), 1)
        self.assertEqual(
            CaseLinker._independent_observation_groups(reversed(chain)),
            groups,
        )

    def test_applicant_association_never_synthesizes_missing_exact_name(self):
        alias = candidate("applicant_name", "Zed Zarnaks")
        linked = LinkedCase(
            case_id=CASE_ID,
            active_applicant=APPLICANT,
            evidence=(alias,),
            unresolved=False,
            active_applicant_aliases=("Zed Zarnaks",),
        )

        field = EvidencePrecedenceResolver().resolve(linked).fields[
            "applicant_name"
        ]

        self.assertEqual(field.state.value, "unknown")
        self.assertIsNone(field.value)
        self.assertIsNone(field.winning_evidence)

    def test_case_association_never_attaches_foreign_visible_winner(self):
        foreign = candidate(
            "case_id",
            "MIB-000002",
            case_hint="MIB-000002",
            applicant_hint=None,
            provenance_applicant=None,
        )
        linked = LinkedCase(
            case_id=CASE_ID,
            active_applicant=None,
            evidence=(foreign,),
            unresolved=True,
            unresolved_reasons=(
                "visible case_id conflicts with source filename",
            ),
        )

        field = EvidencePrecedenceResolver().resolve(linked).fields["case_id"]

        self.assertEqual(field.value, CASE_ID)
        self.assertIsNone(field.winning_evidence)
        self.assertEqual(field.considered, ())


if __name__ == "__main__":
    unittest.main()
