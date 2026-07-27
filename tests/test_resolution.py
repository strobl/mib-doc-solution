import itertools
import unittest

from mib_pipeline import (
    CandidateEvidence,
    CaseLinker,
    EvidencePrecedenceHierarchy,
    EvidencePrecedenceResolver,
    EvidenceType,
    FieldState,
    LinkedCase,
    Rect,
)


def candidate(
    field_name,
    value,
    evidence_type=EvidenceType.INTAKE_FORM,
    *,
    page=0,
    top=20,
    legible=True,
    superseded=False,
    cues=(),
    case_hint="MIB-000001",
    applicant_hint="Zed Zarnax",
    confidence=0.9,
    source="visible_ocr",
):
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=page,
        box=Rect(10, top - 10, 200, top),
        legible=legible,
        superseded=superseded,
        ocr_confidence=confidence,
        visual_cues=tuple(cues),
        source=source,
        case_id_hint=case_hint,
        applicant_hint=applicant_hint,
    )


def linked(*candidates, expected="MIB-000001"):
    return CaseLinker().link(expected, candidates)


def resolver_linked(*candidates, expected="MIB-000001"):
    """Give resolver-only tests one explicit visible active applicant."""

    has_applicant = any(
        item.field_name == "applicant_name" for item in candidates
    )
    evidence = candidates if has_applicant else (
        candidate("applicant_name", "Zed Zarnax"),
        *candidates,
    )
    return CaseLinker().link(expected, evidence)


class CaseLinkerTests(unittest.TestCase):
    def test_evidence_for_other_case_and_applicant_is_excluded(self):
        evidence = [
            candidate("applicant_name", "Zed Zarnax"),
            candidate("home_world", "Kepler-186f"),
            candidate(
                "applicant_name",
                "Other Person",
                case_hint="MIB-000999",
                applicant_hint="Other Person",
                page=2,
            ),
            candidate(
                "home_world",
                "Mars",
                case_hint="MIB-000999",
                applicant_hint="Other Person",
                page=2,
            ),
        ]

        result = linked(*evidence)

        self.assertEqual(result.case_id, "MIB-000001")
        self.assertEqual(result.active_applicant, "Zed Zarnax")
        self.assertFalse(result.unresolved)
        self.assertNotIn("Mars", {item.value for item in result.evidence})

    def test_unseparable_multiple_applicants_are_marked_unresolved(self):
        evidence = [
            candidate("applicant_name", "Zed Zarnax", applicant_hint=None),
            candidate("applicant_name", "Other Person", applicant_hint=None),
            candidate("home_world", "Mars", applicant_hint=None),
        ]

        result = linked(*evidence)

        self.assertTrue(result.unresolved)
        self.assertIsNone(result.active_applicant)
        self.assertIn("multiple applicants", result.unresolved_reasons[0])
        self.assertTrue(any(item.field_name == "home_world" for item in result.evidence))

    def test_orphan_applicant_scoped_field_is_excluded_without_active_applicant(self):
        result = linked(
            candidate(
                "fee_status",
                "unpaid",
                applicant_hint="Unlinked Person",
            ),
            candidate(
                "page_type_present_fee_receipt",
                "true",
                applicant_hint=None,
            ),
        )
        resolved = EvidencePrecedenceResolver().resolve(result)

        self.assertIsNone(result.active_applicant)
        self.assertIsNone(resolved.value("fee_status"))
        self.assertEqual(
            resolved.value("page_type_present_fee_receipt"),
            "true",
        )

    def test_higher_precedence_applicant_scopes_lower_conflicting_evidence(self):
        evidence = [
            candidate("applicant_name", "Zed Zarnax", EvidenceType.INTAKE_FORM),
            candidate(
                "applicant_name",
                "Other Person",
                EvidenceType.SPONSOR_ATTESTATION,
                applicant_hint="Other Person",
                page=2,
            ),
            candidate("home_world", "Kepler-186f", EvidenceType.INTAKE_FORM),
        ]

        result = linked(*evidence)

        self.assertEqual(result.active_applicant, "Zed Zarnax")
        self.assertFalse(result.unresolved)
        self.assertNotIn("Other Person", {item.value for item in result.evidence})

    def test_foreign_applicant_unique_field_is_excluded(self):
        evidence = [
            candidate("applicant_name", "Zed Zarnax", EvidenceType.INTAKE_FORM),
            candidate(
                "applicant_name",
                "Other Person",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint="Other Person",
                page=1,
            ),
            candidate(
                "home_world",
                "Kepler-186f",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint="Other Person",
                page=1,
            ),
        ]

        linked_case = linked(*evidence)
        resolved = EvidencePrecedenceResolver().resolve(linked_case)

        self.assertEqual(linked_case.active_applicant, "Zed Zarnax")
        self.assertIsNone(resolved.value("home_world"))

    def test_packet_scoped_unique_field_is_allowed(self):
        evidence = [
            candidate("applicant_name", "Zed Zarnax", EvidenceType.INTAKE_FORM),
            candidate(
                "applicant_name",
                "Other Person",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint="Other Person",
                page=1,
            ),
            candidate(
                "home_world",
                "Kepler-186f",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint=None,
                page=1,
            ),
        ]

        linked_case = linked(*evidence)
        resolved = EvidencePrecedenceResolver().resolve(linked_case)

        self.assertEqual(linked_case.active_applicant, "Zed Zarnax")
        self.assertEqual(resolved.value("home_world"), "Kepler-186f")

    def test_foreign_applicant_conflict_and_foreign_case_are_excluded(self):
        evidence = [
            candidate("applicant_name", "Zed Zarnax", EvidenceType.INTAKE_FORM),
            candidate(
                "applicant_name",
                "Other Person",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint="Other Person",
                page=1,
            ),
            candidate(
                "home_world",
                "Kepler-186f",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint="Other Person",
                page=1,
            ),
            candidate(
                "home_world",
                "Mars Dome-7",
                EvidenceType.SPONSOR_ATTESTATION,
                applicant_hint="Other Person",
                page=2,
            ),
            candidate(
                "arrival_date",
                "2026-07-01",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint="Other Person",
                case_hint="MIB-000999",
                page=3,
            ),
        ]

        linked_case = linked(*evidence)
        resolved = EvidencePrecedenceResolver().resolve(linked_case)

        self.assertIsNone(resolved.value("home_world"))
        self.assertIsNone(resolved.value("arrival_date"))

    def test_near_identical_ocr_names_are_clustered_and_authority_wins(self):
        evidence = [
            candidate(
                "applicant_name",
                "Xannax Onitx",
                confidence=0.67,
                applicant_hint="Xannax Onitx",
            ),
            candidate(
                "applicant_name",
                "Xannax Oriix",
                EvidenceType.BIOMETRIC_SLIP,
                confidence=0.93,
                applicant_hint="Xannax Oriix",
                page=1,
            ),
            candidate(
                "home_world",
                "Eris Relay",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint="Xannax Oriix",
                page=2,
            ),
        ]

        result = linked(*evidence)

        self.assertEqual(result.active_applicant, "Xannax Onitx")
        self.assertFalse(result.unresolved)
        self.assertIn("Eris Relay", {item.value for item in result.evidence})

    def test_fuzzy_name_chain_does_not_merge_incompatible_endpoints(self):
        first = "Xannax Onitx"
        bridge = "Xannax Oriix"
        other = "Xannax Orxxx"
        self.assertGreaterEqual(CaseLinker._name_similarity(first, bridge), 0.80)
        self.assertGreaterEqual(CaseLinker._name_similarity(bridge, other), 0.80)
        self.assertLess(CaseLinker._name_similarity(first, other), 0.80)

        result = linked(
            candidate(
                "applicant_name",
                first,
                applicant_hint=first,
                confidence=0.91,
            ),
            candidate(
                "applicant_name",
                bridge,
                EvidenceType.BIOMETRIC_SLIP,
                applicant_hint=bridge,
                confidence=0.95,
                page=1,
            ),
            candidate(
                "applicant_name",
                other,
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint=other,
                confidence=0.92,
                page=2,
            ),
            candidate(
                "home_world",
                "Endpoint A",
                applicant_hint=first,
                page=0,
            ),
            candidate(
                "home_world",
                "Endpoint C",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint=other,
                page=2,
            ),
        )

        self.assertEqual(result.active_applicant, first)
        self.assertIn("Endpoint A", {item.value for item in result.evidence})
        self.assertNotIn("Endpoint C", {item.value for item in result.evidence})

    def test_applicant_linking_is_invariant_to_candidate_order(self):
        first = "Xannax Onitx"
        bridge = "Xannax Oriix"
        other = "Xannax Orxxx"
        evidence = (
            candidate(
                "applicant_name",
                first,
                applicant_hint=first,
                confidence=0.91,
            ),
            candidate(
                "applicant_name",
                bridge,
                EvidenceType.BIOMETRIC_SLIP,
                applicant_hint=bridge,
                confidence=0.95,
                page=1,
            ),
            candidate(
                "applicant_name",
                other,
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint=other,
                confidence=0.92,
                page=2,
            ),
        )

        outcomes = set()
        for permutation in itertools.permutations(evidence):
            result = linked(*permutation)
            outcomes.add(
                (
                    result.active_applicant,
                    result.unresolved,
                    result.unresolved_reasons,
                    frozenset(
                        (
                            item.field_name,
                            item.value,
                            item.page_index,
                            item.applicant_hint,
                        )
                        for item in result.evidence
                    ),
                )
            )

        self.assertEqual(len(outcomes), 1)

    def test_duplicate_or_jittered_identity_read_cannot_flip_winner(self):
        alice = "Alice Aster"
        boris = "Boris Boreal"
        base = (
            candidate(
                "applicant_name",
                alice,
                page=0,
                applicant_hint=alice,
                confidence=0.80,
            ),
            candidate(
                "applicant_name",
                alice,
                page=1,
                applicant_hint=alice,
                confidence=0.80,
            ),
            candidate(
                "applicant_name",
                boris,
                page=2,
                applicant_hint=boris,
                confidence=0.95,
            ),
        )
        duplicate = candidate(
            "applicant_name",
            boris,
            page=2,
            applicant_hint=boris,
            confidence=0.95,
        )
        jittered = candidate(
            "applicant_name",
            boris,
            page=2,
            top=21,
            applicant_hint=boris,
            confidence=0.95,
        )
        variant = candidate(
            "applicant_name",
            "Boris Beto",
            page=2,
            applicant_hint="Boris Beto",
            confidence=0.95,
        )
        jittered_variant = candidate(
            "applicant_name",
            "Boris Beto",
            page=2,
            top=21,
            applicant_hint="Boris Beto",
            confidence=0.95,
        )

        for evidence in (
            base,
            (*base, duplicate),
            (*base, jittered),
            (*base, variant),
            (*base, jittered_variant),
        ):
            with self.subTest(candidate_count=len(evidence)):
                result = linked(*evidence)
                self.assertEqual(result.active_applicant, alice)
                self.assertFalse(result.unresolved)

    def test_same_physical_record_cannot_self_corroborate_sponsor_name(self):
        intake_name = "Other Person"
        claimed_name = "Zed Zarnax"
        result = linked(
            candidate(
                "applicant_name",
                intake_name,
                applicant_hint=intake_name,
                confidence=0.60,
            ),
            candidate(
                "applicant_name",
                claimed_name,
                EvidenceType.SPONSOR_ATTESTATION,
                page=1,
                applicant_hint=claimed_name,
                confidence=0.96,
                cues=("structured_sponsor_narrative",),
            ),
            candidate(
                "applicant_name",
                claimed_name,
                EvidenceType.REGISTRY_EXTRACT,
                page=1,
                applicant_hint=claimed_name,
                confidence=0.95,
            ),
        )

        self.assertEqual(result.active_applicant, intake_name)

    def test_local_record_name_variants_cannot_multiply_identity_votes(self):
        alice = "Alice Aster"
        result = linked(
            candidate(
                "applicant_name",
                alice,
                applicant_hint=alice,
                confidence=0.99,
                cues=("record_id:a",),
            ),
            candidate(
                "applicant_name",
                "Boris Beta",
                applicant_hint="Boris Beta",
                confidence=0.95,
                page=1,
                cues=("record_id:b",),
            ),
            candidate(
                "applicant_name",
                "Boris Beto",
                applicant_hint="Boris Beto",
                confidence=0.95,
                page=1,
                top=100,
                cues=("record_id:b",),
            ),
        )

        self.assertEqual(result.active_applicant, alice)
        self.assertFalse(result.unresolved)

    def test_corroborated_sponsor_name_replaces_unrelated_low_confidence_intake(self):
        sponsor_name = "Aridane Tekrix"
        registry_name = "Xandane Teknax"
        damaged_intake_name = "Qorul Xarvara"
        evidence = [
            candidate(
                "applicant_name",
                sponsor_name,
                EvidenceType.SPONSOR_ATTESTATION,
                confidence=0.96,
                cues=("structured_sponsor_narrative",),
                applicant_hint=sponsor_name,
                page=2,
            ),
            candidate(
                "sponsor_id",
                "SPN-7922",
                EvidenceType.SPONSOR_ATTESTATION,
                confidence=0.96,
                applicant_hint=sponsor_name,
                page=2,
            ),
            candidate(
                "applicant_name",
                registry_name,
                EvidenceType.REGISTRY_EXTRACT,
                confidence=0.80,
                applicant_hint=registry_name,
                page=1,
            ),
            candidate(
                "arrival_date",
                "2026-06-28",
                EvidenceType.REGISTRY_EXTRACT,
                confidence=0.80,
                applicant_hint=registry_name,
                page=1,
            ),
            candidate(
                "applicant_name",
                damaged_intake_name,
                EvidenceType.INTAKE_FORM,
                confidence=0.63,
                applicant_hint=damaged_intake_name,
            ),
            candidate(
                "sponsor_id",
                "SPN-0000",
                EvidenceType.INTAKE_FORM,
                confidence=0.64,
                applicant_hint=damaged_intake_name,
            ),
        ]

        linked_case = linked(*evidence)
        resolved = EvidencePrecedenceResolver().resolve(linked_case)

        self.assertEqual(linked_case.active_applicant, sponsor_name)
        self.assertNotIn(damaged_intake_name, {item.value for item in linked_case.evidence})
        self.assertEqual(resolved.value("applicant_name"), sponsor_name)
        self.assertEqual(resolved.value("sponsor_id"), "SPN-7922")
        self.assertEqual(resolved.value("arrival_date"), "2026-06-28")

    def test_corroboration_filters_nearby_damaged_intake_alias_and_its_fields(self):
        sponsor_name = "Nexnax Oriul"
        registry_name = "Nexnax Onul"
        damaged_intake_name = "Nexnex Ortul"
        evidence = [
            candidate(
                "applicant_name",
                sponsor_name,
                EvidenceType.SPONSOR_ATTESTATION,
                confidence=0.96,
                cues=("structured_sponsor_narrative",),
                applicant_hint=sponsor_name,
                page=1,
            ),
            candidate(
                "sponsor_id",
                "SPN-3945",
                EvidenceType.SPONSOR_ATTESTATION,
                confidence=0.96,
                applicant_hint=sponsor_name,
                page=1,
            ),
            candidate(
                "visa_class",
                "XW-2",
                EvidenceType.SPONSOR_ATTESTATION,
                confidence=0.96,
                applicant_hint=sponsor_name,
                page=1,
            ),
            candidate(
                "applicant_name",
                registry_name,
                EvidenceType.REGISTRY_EXTRACT,
                confidence=0.92,
                applicant_hint=registry_name,
                page=2,
            ),
            candidate(
                "applicant_name",
                sponsor_name,
                EvidenceType.BIOMETRIC_SLIP,
                confidence=0.92,
                applicant_hint=sponsor_name,
                page=3,
            ),
            candidate(
                "applicant_name",
                damaged_intake_name,
                EvidenceType.INTAKE_FORM,
                confidence=0.59,
                applicant_hint=damaged_intake_name,
            ),
            candidate(
                "sponsor_id",
                "SPN-3845",
                EvidenceType.INTAKE_FORM,
                confidence=0.55,
                applicant_hint=damaged_intake_name,
            ),
            candidate(
                "visa_class",
                "DIP-1",
                EvidenceType.INTAKE_FORM,
                confidence=0.53,
                applicant_hint=damaged_intake_name,
            ),
        ]

        linked_case = linked(*evidence)
        resolved = EvidencePrecedenceResolver().resolve(linked_case)

        self.assertEqual(linked_case.active_applicant, sponsor_name)
        self.assertNotIn("SPN-3845", {item.value for item in linked_case.evidence})
        self.assertNotIn("DIP-1", {item.value for item in linked_case.evidence})
        self.assertEqual(resolved.value("sponsor_id"), "SPN-3945")
        self.assertEqual(resolved.value("visa_class"), "XW-2")

    def test_sponsor_override_requires_support_and_only_low_confidence_intake(self):
        sponsor = candidate(
            "applicant_name",
            "Aridane Tekrix",
            EvidenceType.SPONSOR_ATTESTATION,
            confidence=0.96,
            cues=("structured_sponsor_narrative",),
            applicant_hint="Aridane Tekrix",
            page=1,
        )
        low_intake = candidate(
            "applicant_name",
            "Qorul Xarvara",
            EvidenceType.INTAKE_FORM,
            confidence=0.63,
            applicant_hint="Qorul Xarvara",
        )

        uncorroborated = linked(sponsor, low_intake)
        self.assertEqual(uncorroborated.active_applicant, "Qorul Xarvara")

        high_intake = candidate(
            "applicant_name",
            "Qorul Xarvara",
            EvidenceType.INTAKE_FORM,
            confidence=0.80,
            applicant_hint="Qorul Xarvara",
        )
        registry_support = candidate(
            "applicant_name",
            "Xandane Teknax",
            EvidenceType.REGISTRY_EXTRACT,
            confidence=0.80,
            applicant_hint="Xandane Teknax",
            page=2,
        )
        guarded = linked(sponsor, high_intake, registry_support)
        self.assertEqual(guarded.active_applicant, "Qorul Xarvara")

    def test_superseded_applicant_does_not_scope_away_valid_packet(self):
        evidence = [
            candidate(
                "applicant_name",
                "Wrong Person",
                superseded=True,
                cues=("strikethrough",),
                applicant_hint="Wrong Person",
            ),
            candidate(
                "applicant_name",
                "Zed Zarnax",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint="Zed Zarnax",
                page=1,
            ),
            candidate(
                "home_world",
                "Kepler-186f",
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint="Zed Zarnax",
                page=1,
            ),
        ]

        result = linked(*evidence)

        self.assertEqual(result.active_applicant, "Zed Zarnax")
        self.assertIn("Kepler-186f", {item.value for item in result.evidence})

    def test_same_page_applicant_correction_keeps_other_intake_fields(self):
        evidence = [
            candidate(
                "applicant_name",
                "Wrong Person",
                superseded=True,
                cues=("strikethrough",),
                applicant_hint="Wrong Person",
            ),
            candidate(
                "home_world",
                "Mars Dome-7",
                applicant_hint="Wrong Person",
            ),
            candidate(
                "visa_class",
                "XW-1",
                applicant_hint="Wrong Person",
            ),
            candidate(
                "applicant_name",
                "Zed Zarnax",
                cues=("correction",),
                applicant_hint="Zed Zarnax",
                top=90,
            ),
            candidate(
                "applicant_name",
                "Zed Zarnax",
                EvidenceType.BIOMETRIC_SLIP,
                applicant_hint="Zed Zarnax",
                page=1,
            ),
        ]

        result = linked(*evidence)
        resolved = EvidencePrecedenceResolver().resolve(result)

        self.assertEqual(result.active_applicant, "Zed Zarnax")
        self.assertEqual(resolved.value("home_world"), "Mars Dome-7")
        self.assertEqual(resolved.value("visa_class"), "XW-1")

    def test_exact_case_cross_source_name_wins_when_other_scope_is_stable(self):
        decoy = "Other Person"
        corroborated = "Zed Zarnax"
        evidence = [
            candidate(
                "applicant_name",
                decoy,
                EvidenceType.INTAKE_FORM,
                applicant_hint=decoy,
                confidence=0.95,
            ),
            candidate(
                "applicant_name",
                corroborated,
                EvidenceType.BIOMETRIC_SLIP,
                applicant_hint=corroborated,
                page=1,
            ),
            candidate(
                "applicant_name",
                corroborated,
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint=corroborated,
                page=2,
            ),
            # The same physical risk observation is emitted once with its page
            # name hint and once as a case-level aggregate. It must not look
            # like a safety-source change merely because the hint differs.
            candidate(
                "risk_flags",
                "identity_conflict",
                EvidenceType.BIOMETRIC_SLIP,
                applicant_hint=corroborated,
                page=1,
                top=80,
            ),
            candidate(
                "risk_flags",
                "identity_conflict",
                EvidenceType.BIOMETRIC_SLIP,
                applicant_hint=None,
                page=1,
                top=80,
            ),
        ]

        linked_case = linked(*evidence)
        resolved = EvidencePrecedenceResolver().resolve(linked_case)

        self.assertEqual(linked_case.active_applicant, corroborated)
        self.assertEqual(resolved.value("applicant_name"), corroborated)
        self.assertEqual(resolved.value("risk_flags"), "identity_conflict")
        self.assertNotIn(decoy, {item.value for item in linked_case.evidence})

    def test_cross_source_name_vetoes_a_sponsor_value_change(self):
        decoy = "Other Person"
        corroborated = "Zed Zarnax"
        evidence = [
            candidate("applicant_name", decoy, applicant_hint=decoy),
            candidate(
                "sponsor_id",
                "SPN-1111",
                applicant_hint=decoy,
            ),
            candidate(
                "applicant_name",
                corroborated,
                EvidenceType.BIOMETRIC_SLIP,
                applicant_hint=corroborated,
                page=1,
            ),
            candidate(
                "applicant_name",
                corroborated,
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint=corroborated,
                page=2,
            ),
            candidate(
                "sponsor_id",
                "SPN-2222",
                EvidenceType.SPONSOR_ATTESTATION,
                applicant_hint=corroborated,
                page=3,
            ),
        ]

        linked_case = linked(*evidence)
        resolved = EvidencePrecedenceResolver().resolve(linked_case)

        self.assertEqual(linked_case.active_applicant, decoy)
        self.assertEqual(resolved.value("sponsor_id"), "SPN-1111")

    def test_cross_source_name_vetoes_risk_or_policy_source_changes(self):
        source_cases = (
            (
                "risk_flags",
                "none",
                EvidenceType.INTAKE_FORM,
                EvidenceType.BIOMETRIC_SLIP,
            ),
            (
                "packet_receipt_date",
                "2026-07-01",
                EvidenceType.INTAKE_FORM,
                EvidenceType.REGISTRY_EXTRACT,
            ),
            (
                "adjudication",
                "NEEDS_REVIEW",
                EvidenceType.ADJUDICATOR_STAMP,
                EvidenceType.SIGNED_MANUAL_NOTE,
            ),
        )
        for field_name, value, current_type, alternative_type in source_cases:
            with self.subTest(field_name=field_name):
                decoy = "Other Person"
                corroborated = "Zed Zarnax"
                linked_case = linked(
                    candidate("applicant_name", decoy, applicant_hint=decoy),
                    candidate(
                        "applicant_name",
                        corroborated,
                        EvidenceType.BIOMETRIC_SLIP,
                        applicant_hint=corroborated,
                        page=1,
                    ),
                    candidate(
                        "applicant_name",
                        corroborated,
                        EvidenceType.REGISTRY_EXTRACT,
                        applicant_hint=corroborated,
                        page=2,
                    ),
                    candidate(
                        field_name,
                        value,
                        current_type,
                        applicant_hint=decoy,
                        page=0,
                        top=80,
                    ),
                    candidate(
                        field_name,
                        value,
                        alternative_type,
                        applicant_hint=corroborated,
                        page=3,
                        top=80,
                    ),
                )

                self.assertEqual(linked_case.active_applicant, decoy)

    def test_cross_source_name_requires_only_exact_case_anchors(self):
        for case_hint in (None, "MIB-000999"):
            with self.subTest(case_hint=case_hint):
                decoy = "Other Person"
                corroborated = "Zed Zarnax"
                linked_case = linked(
                    candidate("applicant_name", decoy, applicant_hint=decoy),
                    candidate(
                        "applicant_name",
                        corroborated,
                        EvidenceType.BIOMETRIC_SLIP,
                        applicant_hint=corroborated,
                        case_hint=case_hint,
                        page=1,
                    ),
                    candidate(
                        "applicant_name",
                        corroborated,
                        EvidenceType.REGISTRY_EXTRACT,
                        applicant_hint=corroborated,
                        case_hint=case_hint,
                        page=2,
                    ),
                )

                self.assertEqual(linked_case.active_applicant, decoy)

    def test_cross_source_name_requires_distinct_pages_and_evidence_types(self):
        variants = (
            (
                (EvidenceType.BIOMETRIC_SLIP, 1),
                (EvidenceType.REGISTRY_EXTRACT, 1),
            ),
            (
                (EvidenceType.REGISTRY_EXTRACT, 1),
                (EvidenceType.REGISTRY_EXTRACT, 2),
            ),
        )
        for first, second in variants:
            with self.subTest(first=first, second=second):
                decoy = "Other Person"
                corroborated = "Zed Zarnax"
                linked_case = linked(
                    candidate("applicant_name", decoy, applicant_hint=decoy),
                    candidate(
                        "applicant_name",
                        corroborated,
                        first[0],
                        applicant_hint=corroborated,
                        page=first[1],
                    ),
                    candidate(
                        "applicant_name",
                        corroborated,
                        second[0],
                        applicant_hint=corroborated,
                        page=second[1],
                    ),
                )

                self.assertEqual(linked_case.active_applicant, decoy)

    def test_cross_source_name_rejects_vetoed_support(self):
        decoy = "Other Person"
        corroborated = "Zed Zarnax"
        linked_case = linked(
            candidate("applicant_name", decoy, applicant_hint=decoy),
            candidate(
                "applicant_name",
                corroborated,
                EvidenceType.BIOMETRIC_SLIP,
                applicant_hint=corroborated,
                page=1,
            ),
            candidate(
                "applicant_name",
                corroborated,
                EvidenceType.REGISTRY_EXTRACT,
                applicant_hint=corroborated,
                cues=("sample_denial_watermark",),
                page=2,
            ),
        )

        self.assertEqual(linked_case.active_applicant, decoy)

    def test_conflicting_visible_case_id_marks_linkage_unresolved(self):
        result = linked(
            candidate(
                "case_id",
                "MIB-000999",
                case_hint="MIB-000999",
                applicant_hint=None,
            )
        )

        self.assertEqual(result.case_id, "MIB-000001")
        self.assertTrue(result.unresolved)

    def test_fully_scoped_foreign_case_page_is_excluded_without_ambiguity(self):
        result = linked(
            candidate(
                "case_id",
                "MIB-000001",
                case_hint="MIB-000001",
                applicant_hint=None,
            ),
            candidate(
                "applicant_name",
                "Zed Zarnax",
                case_hint="MIB-000001",
                applicant_hint="Zed Zarnax",
            ),
            candidate(
                "fee_status",
                "paid",
                case_hint="MIB-000001",
                applicant_hint="Zed Zarnax",
            ),
            candidate(
                "case_id",
                "MIB-000002",
                case_hint="MIB-000002",
                applicant_hint=None,
                page=1,
            ),
            candidate(
                "applicant_name",
                "Other Person",
                case_hint="MIB-000002",
                applicant_hint="Other Person",
                page=1,
            ),
            candidate(
                "fee_status",
                "unpaid",
                case_hint="MIB-000002",
                applicant_hint="Other Person",
                page=1,
            ),
        )
        resolved = EvidencePrecedenceResolver().resolve(result)

        self.assertFalse(result.unresolved)
        self.assertEqual(result.active_applicant, "Zed Zarnax")
        self.assertEqual(resolved.value("fee_status"), "paid")
        self.assertNotIn(
            "Other Person",
            {item.value for item in result.evidence},
        )
        self.assertNotIn(
            "unpaid",
            {item.value for item in result.evidence},
        )

    def test_mixed_or_unhinted_foreign_case_page_remains_unresolved(self):
        unsafe_variants = (
            (
                candidate(
                    "case_id",
                    "MIB-000002",
                    case_hint="MIB-000002",
                    applicant_hint=None,
                    page=1,
                ),
                candidate(
                    "fee_status",
                    "unpaid",
                    case_hint=None,
                    applicant_hint=None,
                    page=1,
                ),
            ),
            (
                candidate(
                    "case_id",
                    "MIB-000002",
                    case_hint="MIB-000002",
                    applicant_hint=None,
                    page=0,
                ),
            ),
            (
                candidate(
                    "case_id",
                    "MIB-000002",
                    case_hint=None,
                    applicant_hint=None,
                    page=1,
                ),
                candidate(
                    "fee_status",
                    "unpaid",
                    case_hint="MIB-000002",
                    applicant_hint="Other Person",
                    page=1,
                ),
            ),
            (
                candidate(
                    "case_id",
                    "MIB-000002",
                    case_hint="MIB-000002",
                    applicant_hint=None,
                    page=1,
                ),
                candidate(
                    "fee_status",
                    "unpaid",
                    case_hint="MIB-000001",
                    applicant_hint="Other Person",
                    page=1,
                ),
            ),
        )
        expected_anchor = candidate(
            "case_id",
            "MIB-000001",
            case_hint="MIB-000001",
            applicant_hint=None,
        )

        for unsafe in unsafe_variants:
            with self.subTest(unsafe=unsafe):
                result = linked(expected_anchor, *unsafe)

                self.assertTrue(result.unresolved)
                self.assertIn(
                    "visible case_id conflicts with source filename",
                    result.unresolved_reasons,
                )

    def test_expected_and_foreign_visible_ids_quarantine_unhinted_facts(self):
        result = linked(
            candidate(
                "case_id",
                "MIB-000001",
                case_hint="MIB-000001",
                applicant_hint=None,
            ),
            candidate(
                "case_id",
                "MIB-000002",
                case_hint="MIB-000002",
                applicant_hint=None,
                page=1,
            ),
            candidate(
                "fee_status",
                "paid",
                case_hint=None,
                applicant_hint=None,
                page=2,
            ),
        )
        resolved = EvidencePrecedenceResolver().resolve(result)

        self.assertEqual(result.case_id, "MIB-000001")
        self.assertTrue(result.unresolved)
        self.assertTrue(resolved.unresolved_linkage)
        self.assertFalse(
            any(
                item.field_name == "fee_status"
                for item in result.evidence
            )
        )
        self.assertEqual(
            resolved.fields["fee_status"].state,
            FieldState.UNKNOWN,
        )

    def test_watermarked_or_non_ocr_sponsor_cannot_select_applicant(self):
        for cues, source in (
            (("structured_sponsor_narrative", "sample_watermark"), "visible_ocr"),
            (("structured_sponsor_narrative",), "output_default"),
        ):
            with self.subTest(cues=cues, source=source):
                clean_name = "Aridane Tekris"
                result = linked(
                    candidate(
                        "applicant_name",
                        clean_name,
                        confidence=0.60,
                        applicant_hint=clean_name,
                    ),
                    candidate(
                        "applicant_name",
                        clean_name,
                        EvidenceType.REGISTRY_EXTRACT,
                        page=1,
                        confidence=0.90,
                        applicant_hint=clean_name,
                    ),
                    candidate(
                        "applicant_name",
                        "Aridane Tekrix",
                        EvidenceType.SPONSOR_ATTESTATION,
                        page=2,
                        confidence=0.96,
                        applicant_hint="Aridane Tekrix",
                        cues=cues,
                        source=source,
                    ),
                )
                resolved = EvidencePrecedenceResolver().resolve(result)

                self.assertEqual(result.active_applicant, clean_name)
                self.assertEqual(resolved.value("applicant_name"), clean_name)
                self.assertEqual(
                    resolved.fields["applicant_name"].winning_evidence.value,
                    clean_name,
                )

    def test_ambiguous_case_ids_exclude_case_hinted_facts(self):
        evidence = (
            candidate(
                "case_id",
                "MIB-000001",
                applicant_hint=None,
                case_hint="MIB-000001",
            ),
            candidate(
                "fee_status",
                "paid",
                applicant_hint=None,
                case_hint="MIB-000001",
            ),
            candidate(
                "case_id",
                "MIB-000002",
                applicant_hint=None,
                case_hint="MIB-000002",
                page=1,
            ),
            candidate(
                "fee_status",
                "unpaid",
                applicant_hint=None,
                case_hint="MIB-000002",
                page=1,
            ),
        )

        linked_case = CaseLinker().link(None, evidence)
        resolved = EvidencePrecedenceResolver().resolve(linked_case)

        self.assertEqual(linked_case.case_id, "")
        self.assertTrue(linked_case.unresolved)
        self.assertFalse(
            any(
                item.field_name == "fee_status"
                for item in linked_case.evidence
            )
        )
        self.assertEqual(
            resolved.fields["fee_status"].state,
            FieldState.UNKNOWN,
        )

    def test_text_layer_cannot_establish_case_or_applicant_linkage(self):
        result = CaseLinker().link(
            None,
            (
                candidate(
                    "case_id",
                    "MIB-000001",
                    EvidenceType.TEXT_LAYER,
                    applicant_hint=None,
                ),
                candidate(
                    "applicant_name",
                    "Text Layer Person",
                    EvidenceType.TEXT_LAYER,
                    applicant_hint="Text Layer Person",
                ),
            ),
        )

        self.assertEqual(result.case_id, "")
        self.assertIsNone(result.active_applicant)
        self.assertTrue(result.unresolved)


class PrecedenceResolverTests(unittest.TestCase):
    def test_precedence_ranks_match_field_manual(self):
        self.assertEqual(
            [
                EvidencePrecedenceHierarchy.rank(evidence_type)
                for evidence_type in (
                    EvidenceType.ADJUDICATOR_STAMP,
                    EvidenceType.INTAKE_FORM,
                    EvidenceType.BIOMETRIC_SLIP,
                    EvidenceType.SPONSOR_ATTESTATION,
                    EvidenceType.REGISTRY_EXTRACT,
                    EvidenceType.TEXT_LAYER,
                )
            ],
            [1, 2, 3, 4, 5, 6],
        )

    def test_higher_rank_wins_and_lower_rank_cannot_override(self):
        case = resolver_linked(
            candidate("fee_status", "paid", EvidenceType.INTAKE_FORM),
            candidate("fee_status", "unpaid", EvidenceType.REGISTRY_EXTRACT),
        )

        resolved = EvidencePrecedenceResolver().resolve(case)

        field = resolved.fields["fee_status"]
        self.assertEqual(field.state, FieldState.RESOLVED)
        self.assertEqual(field.value, "paid")
        self.assertEqual(field.winning_evidence.evidence_type, EvidenceType.INTAKE_FORM)

    def test_structured_sponsor_narrative_repairs_two_redundant_fields(self):
        examples = (
            ("sponsor_id", "SPN-1111", "SPN-2222"),
            ("visa_class", "XW-2", "XW-1"),
        )
        for field_name, intake_value, sponsor_value in examples:
            with self.subTest(field_name=field_name):
                case = resolver_linked(
                    candidate("applicant_name", "Zed Zarnax"),
                    candidate(field_name, intake_value),
                    candidate(
                        field_name,
                        sponsor_value,
                        EvidenceType.SPONSOR_ATTESTATION,
                        page=1,
                        confidence=0.95,
                        cues=("structured_sponsor_narrative",),
                    ),
                )

                field = EvidencePrecedenceResolver(
                    fusion_enabled=False
                ).resolve(case).fields[field_name]

                self.assertEqual(field.value, sponsor_value)
                self.assertEqual(
                    field.winning_evidence.evidence_type,
                    EvidenceType.SPONSOR_ATTESTATION,
                )
                self.assertIn("sponsor narrative OCR repair", field.reason)

    def test_structured_sponsor_repair_abstains_without_every_safeguard(self):
        def resolved_value(
            *,
            field_name="visa_class",
            sponsor_value="XW-1",
            sponsor_confidence=0.95,
            sponsor_cues=("structured_sponsor_narrative",),
            sponsor_case="MIB-000001",
            sponsor_applicant="Zed Zarnax",
            sponsor_superseded=False,
            intake_cues=(),
            extra_sponsor=None,
        ):
            evidence = [
                candidate("applicant_name", "Zed Zarnax"),
                candidate(field_name, "XW-2", cues=intake_cues),
                candidate(
                    field_name,
                    sponsor_value,
                    EvidenceType.SPONSOR_ATTESTATION,
                    page=1,
                    confidence=sponsor_confidence,
                    cues=sponsor_cues,
                    case_hint=sponsor_case,
                    applicant_hint=sponsor_applicant,
                    superseded=sponsor_superseded,
                ),
            ]
            if extra_sponsor is not None:
                evidence.append(
                    candidate(
                        field_name,
                        extra_sponsor,
                        EvidenceType.SPONSOR_ATTESTATION,
                        page=2,
                        confidence=0.95,
                        cues=("structured_sponsor_narrative",),
                    )
                )
            return EvidencePrecedenceResolver(
                fusion_enabled=False
            ).resolve(resolver_linked(*evidence)).fields[field_name].value

        variants = (
            {"sponsor_cues": ()},
            {"sponsor_confidence": 0.89},
            {"sponsor_case": "MIB-000999"},
            {"sponsor_applicant": "Other Person"},
            {"sponsor_superseded": True},
            {"intake_cues": ("correction",)},
            {"extra_sponsor": "DIP-1"},
            {
                "field_name": "declared_purpose",
                "sponsor_value": "field repair",
            },
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertEqual(resolved_value(**variant), "XW-2")

    def test_same_rank_conflict_is_contested(self):
        case = resolver_linked(
            candidate("home_world", "Mars", EvidenceType.INTAKE_FORM),
            candidate("home_world", "Europa", EvidenceType.INTAKE_FORM, page=1),
        )

        field = EvidencePrecedenceResolver().resolve(case).fields["home_world"]

        self.assertEqual(field.state, FieldState.CONTESTED)
        self.assertIsNone(field.value)

    def test_struck_through_value_is_dropped(self):
        case = resolver_linked(
            candidate(
                "fee_status",
                "unpaid",
                EvidenceType.INTAKE_FORM,
                superseded=True,
                cues=("strikethrough",),
            ),
            candidate("fee_status", "paid", EvidenceType.SPONSOR_ATTESTATION),
        )

        field = EvidencePrecedenceResolver().resolve(case).fields["fee_status"]

        self.assertEqual(field.value, "paid")

    def test_visible_correction_wins_within_same_rank(self):
        case = resolver_linked(
            candidate("sponsor_id", "SPN-0007", EvidenceType.INTAKE_FORM),
            candidate(
                "sponsor_id",
                "SPN-1234",
                EvidenceType.INTAKE_FORM,
                page=1,
                cues=("correction",),
            ),
        )

        field = EvidencePrecedenceResolver(
            fusion_enabled=False
        ).resolve(case).fields["sponsor_id"]

        self.assertEqual(field.value, "SPN-1234")

    def test_text_layer_is_diagnostic_in_fusion_and_legacy_fallback_remains(self):
        text_only = resolver_linked(
            candidate("visa_class", "XW-1", EvidenceType.TEXT_LAYER)
        )
        self.assertEqual(
            EvidencePrecedenceResolver().resolve(text_only).fields[
                "visa_class"
            ].state,
            FieldState.UNKNOWN,
        )
        self.assertEqual(
            EvidencePrecedenceResolver(fusion_enabled=False)
            .resolve(text_only)
            .fields["visa_class"]
            .value,
            "XW-1",
        )

        with_visible = resolver_linked(
            candidate("visa_class", "XW-1", EvidenceType.TEXT_LAYER),
            candidate("visa_class", "XW-2", EvidenceType.INTAKE_FORM),
        )
        self.assertEqual(
            EvidencePrecedenceResolver().resolve(with_visible).fields["visa_class"].value,
            "XW-2",
        )

    def test_missing_and_illegible_fields_are_unknown(self):
        case = resolver_linked(
            candidate("species_code", None, legible=False),
        )

        resolved = EvidencePrecedenceResolver().resolve(case)

        self.assertEqual(resolved.fields["species_code"].state, FieldState.UNKNOWN)
        self.assertEqual(resolved.fields["home_world"].state, FieldState.UNKNOWN)

    def test_later_signed_approval_rescinds_denial_stamp(self):
        case = resolver_linked(
            candidate(
                "adjudication",
                "DENIED",
                EvidenceType.ADJUDICATOR_STAMP,
                page=0,
            ),
            candidate(
                "adjudication",
                "APPROVED",
                EvidenceType.SIGNED_MANUAL_NOTE,
                page=1,
                cues=("correction",),
            ),
        )

        resolved = EvidencePrecedenceResolver().resolve(case)

        self.assertTrue(resolved.rescinded_decision)
        self.assertEqual(resolved.fields["adjudication"].value, "APPROVED")

    def test_invalid_signed_approval_cannot_rescind_binding_denial(self):
        applicant = candidate("applicant_name", "Zed Zarnax")
        denial = candidate(
            "adjudication",
            "DENIED",
            EvidenceType.ADJUDICATOR_STAMP,
            page=0,
        )
        variants = (
            candidate(
                "adjudication",
                "APPROVED",
                EvidenceType.SIGNED_MANUAL_NOTE,
                page=1,
                cues=("correction",),
                source="output_default",
            ),
            candidate(
                "adjudication",
                "APPROVED",
                EvidenceType.SIGNED_MANUAL_NOTE,
                page=1,
                cues=("correction",),
                legible=False,
            ),
            candidate(
                "adjudication",
                "APPROVED",
                EvidenceType.SIGNED_MANUAL_NOTE,
                page=1,
                cues=("correction", "official_watermark"),
            ),
            candidate(
                "adjudication",
                "APPROVED",
                EvidenceType.SIGNED_MANUAL_NOTE,
                page=1,
                cues=("correction",),
                case_hint="MIB-000999",
            ),
            candidate(
                "adjudication",
                "APPROVED",
                EvidenceType.SIGNED_MANUAL_NOTE,
                page=1,
                cues=("correction",),
                applicant_hint="Other Person",
            ),
        )
        for approval in variants:
            with self.subTest(
                source=approval.source,
                cues=approval.visual_cues,
                case_hint=approval.case_id_hint,
                applicant_hint=approval.applicant_hint,
            ):
                linked_case = LinkedCase(
                    case_id="MIB-000001",
                    active_applicant="Zed Zarnax",
                    evidence=(applicant, denial, approval),
                    unresolved=False,
                    active_applicant_aliases=("Zed Zarnax",),
                )
                resolved = EvidencePrecedenceResolver().resolve(linked_case)

                self.assertFalse(resolved.rescinded_decision)
                self.assertEqual(
                    resolved.fields["adjudication"].value,
                    "DENIED",
                )

    def test_sample_denial_watermark_never_becomes_live_decision(self):
        case = resolver_linked(
            candidate(
                "adjudication",
                "DENIED",
                EvidenceType.ADJUDICATOR_STAMP,
                cues=("sample_denial_watermark",),
            )
        )

        field = EvidencePrecedenceResolver().resolve(case).fields["adjudication"]

        self.assertEqual(field.state, FieldState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
