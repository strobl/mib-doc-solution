import unittest
from dataclasses import replace

from mib_pipeline import (
    FINAL_CONFIDENCE_CONTEXT_SCHEMA_VERSION,
    FinalConfidenceContext,
    FinalPredictionWithConfidenceContext,
    PredictionRow,
)


def row(*, adjudication="NEEDS_REVIEW"):
    return PredictionRow.from_mapping(
        {
            "case_id": "MIB-000001",
            "applicant_name": "Test Applicant",
            "species_code": "TEST",
            "home_world": "Test World",
            "visa_class": "TEST-1",
            "sponsor_id": "SPN-0001",
            "arrival_date": "2026-01-01",
            "declared_purpose": "test",
            "risk_flags": "none",
            "fee_status": "paid",
            "adjudication": adjudication,
            "confidence": 0.5,
        }
    )


def context(**overrides):
    values = {
        "schema_version": FINAL_CONFIDENCE_CONTEXT_SCHEMA_VERSION,
        "final_class": "NEEDS_REVIEW",
        "policy_route": "revalidated_policy",
        "authoritative": False,
        "visible_completeness": 8 / 9,
        "has_conflict": True,
        "ocr_disagreement": 0.25,
        "recovery_route": "rapid_visible",
        "model_margin": None,
        "ensemble_agreement": None,
        "resolution_entropy": 0.125,
    }
    values.update(overrides)
    return FinalConfidenceContext(**values)


class FinalConfidenceContextTests(unittest.TestCase):
    def test_contract_has_only_closed_identity_free_fields(self):
        value = context().to_dict()

        self.assertEqual(
            tuple(value),
            (
                "schema_version",
                "final_class",
                "policy_route",
                "authoritative",
                "visible_completeness",
                "has_conflict",
                "ocr_disagreement",
                "recovery_route",
                "model_margin",
                "ensemble_agreement",
                "resolution_entropy",
            ),
        )
        self.assertFalse(
            {
                "case_id",
                "applicant_name",
                "sponsor_id",
                "source_path",
                "source_sha256",
                "filename",
                "truth",
            }.intersection(value)
        )

    def test_rejects_open_routes_invalid_ranges_and_partial_model_diagnostics(self):
        with self.assertRaisesRegex(ValueError, "policy_route"):
            context(policy_route="case-MIB-000001")
        with self.assertRaisesRegex(ValueError, "within"):
            context(ocr_disagreement=1.01)
        with self.assertRaisesRegex(ValueError, "both present"):
            context(model_margin=0.2)
        with self.assertRaisesRegex(ValueError, "binding_authority"):
            context(authoritative=True)

    def test_authoritative_context_is_explicitly_bound(self):
        bound = context(
            final_class="DENIED",
            policy_route="binding_authority",
            authoritative=True,
        )

        self.assertTrue(bound.authoritative)
        self.assertEqual(bound.policy_route, "binding_authority")

    def test_binding_authority_route_cannot_claim_non_authoritative_context(self):
        with self.assertRaisesRegex(ValueError, "must be equivalent"):
            context(
                policy_route="binding_authority",
                authoritative=False,
            )

    def test_final_row_and_context_class_must_match(self):
        prediction = row()
        FinalPredictionWithConfidenceContext(
            row=prediction,
            context=context(),
        )

        with self.assertRaisesRegex(ValueError, "does not match"):
            FinalPredictionWithConfidenceContext(
                row=prediction,
                context=replace(context(), final_class="APPROVED"),
            )


if __name__ == "__main__":
    unittest.main()
