import math
import unittest
from dataclasses import replace

from mib_pipeline import (
    AdjudicationOutcome,
    CandidateEvidence,
    CompactModelArtifact,
    CompactThreeClassModel,
    DecisionTrace,
    EvidenceType,
    FEATURE_NAMES,
    FieldState,
    GatedHybridDecisionRecoveryAdjudicator,
    IdentityFreeDecisionFeatures,
    IdentityFreeFeatureBuilder,
    ModelPrediction,
    PredictionRow,
    Rect,
    ResolvedCase,
    ResolvedField,
)
from mib_pipeline.model_recovery import GatedHybridDecisionRule, MODEL_CLASSES


CASE_ID = "MIB-000001"
SUBJECT = "Veenax Qortari"
OUTPUT_VALUES = {
    "applicant_name": SUBJECT,
    "species_code": "ANDROMEDAN",
    "home_world": "Mars Dome-7",
    "visa_class": "XW-2",
    "sponsor_id": "SPN-1042",
    "arrival_date": "2026-04-17",
    "declared_purpose": "technical work",
    "risk_flags": "none",
    "fee_status": "paid",
}


def candidate(
    field_name,
    value,
    *,
    evidence_type=EvidenceType.INTAKE_FORM,
    source="visible_ocr",
    cues=(),
    case_hint=CASE_ID,
    subject_hint=SUBJECT,
):
    return CandidateEvidence(
        field_name=field_name,
        value=value,
        evidence_type=evidence_type,
        page_index=0,
        box=Rect(0, 0, 100, 20),
        legible=True,
        superseded=False,
        ocr_confidence=0.95,
        visual_cues=tuple(cues),
        source=source,
        case_id_hint=case_hint,
        applicant_hint=subject_hint,
    )


def resolved_field(field_name, value, **kwargs):
    evidence = candidate(field_name, value, **kwargs)
    return ResolvedField(
        field_name=field_name,
        state=FieldState.RESOLVED,
        value=value,
        winning_evidence=evidence,
        considered=(evidence,),
        reason="test",
    )


def complete_case(*, overrides=None, extras=(), unresolved=False):
    values = dict(OUTPUT_VALUES)
    values.update(overrides or {})
    fields = {
        field_name: resolved_field(field_name, value)
        for field_name, value in values.items()
    }
    fields.update({field.field_name: field for field in extras})
    return ResolvedCase(
        case_id=CASE_ID,
        active_applicant=SUBJECT,
        fields=fields,
        unresolved_linkage=unresolved,
        unresolved_reasons=("ambiguous",) if unresolved else (),
    )


def prediction(decision="NEEDS_REVIEW", confidence=0.37):
    return PredictionRow.from_mapping(
        {
            "case_id": CASE_ID,
            **OUTPUT_VALUES,
            "adjudication": decision,
            "confidence": confidence,
        }
    )


def outcome(
    decision="NEEDS_REVIEW",
    *,
    authoritative=False,
    denial=(),
    review=("residual_model_candidate",),
    approval=(),
    exception_ids=(),
    confidence=0.37,
):
    return AdjudicationOutcome(
        row=prediction(decision, confidence),
        trace=DecisionTrace(
            decision=decision,
            authoritative_source=authoritative,
            denial_reasons=tuple(denial),
            review_reasons=tuple(review),
            approval_facts=tuple(approval),
            exception_ids=tuple(exception_ids),
        ),
    )


class FakeAdjudicator:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def adjudicate_case(self, resolved_case):
        self.calls += 1
        return self.result


def forced_model(decision, *, second_intercept=0.0):
    intercepts = {
        "APPROVED": 0.0,
        "DENIED": 0.0,
        "NEEDS_REVIEW": 0.0,
    }
    intercepts[decision] = 6.0
    if decision != "NEEDS_REVIEW":
        intercepts["NEEDS_REVIEW"] = second_intercept
    artifact = CompactModelArtifact(
        schema_version=1,
        class_names=MODEL_CLASSES,
        feature_names=FEATURE_NAMES,
        intercepts=tuple(intercepts[name] for name in MODEL_CLASSES),
        coefficients=tuple(
            tuple(0.0 for _name in FEATURE_NAMES) for _class in MODEL_CLASSES
        ),
    )
    return CompactThreeClassModel.from_artifact(artifact)


class FeatureAndArtifactContractTests(unittest.TestCase):
    def test_feature_contract_is_fixed_numeric_and_identity_free(self):
        features = IdentityFreeFeatureBuilder().build(
            complete_case(),
            outcome(),
        )
        self.assertEqual(tuple(features.values), FEATURE_NAMES)
        self.assertTrue(
            all(
                isinstance(value, float)
                and math.isfinite(value)
                and 0.0 <= value <= 1.0
                for value in features.values.values()
            )
        )
        forbidden = (
            "case_id",
            "applicant",
            "name",
            "sponsor",
            "filename",
            "path",
            "hash",
            "raw",
            "order",
        )
        rendered = "|".join(FEATURE_NAMES)
        self.assertFalse(any(token in rendered for token in forbidden))

    def test_placeholder_and_sentinel_values_never_count_as_clean(self):
        placeholder_case = complete_case(
            overrides={
                "applicant_name": "unknown",
                "species_code": "other",
                "home_world": "null",
                "visa_class": "unknown",
                "sponsor_id": "SPN-0000",
                "arrival_date": "1900-01-01",
                "declared_purpose": "none",
                "fee_status": "unknown",
            }
        )
        features = IdentityFreeFeatureBuilder().build(
            placeholder_case,
            outcome(),
        )
        self.assertLess(features.values["clean_fraction"], 1.0)
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(outcome()),
            forced_model("APPROVED"),
            enabled=True,
        ).evaluate_case(placeholder_case)
        self.assertEqual(result.outcome.row.adjudication, "NEEDS_REVIEW")
        self.assertIn("incomplete_clean_fraction", result.veto_reasons)

    def test_common_placeholder_spellings_cannot_pass_approval(self):
        for placeholder in (
            "N/A",
            "N / A",
            "TBD",
            "not provided",
            "not available",
            "-",
            "pending",
        ):
            with self.subTest(placeholder=placeholder):
                test_case = complete_case(
                    overrides={"sponsor_id": placeholder},
                )
                features = IdentityFreeFeatureBuilder().build(
                    test_case,
                    outcome(review=()),
                )
                self.assertLess(features.values["clean_fraction"], 1.0)
                result = GatedHybridDecisionRecoveryAdjudicator(
                    FakeAdjudicator(outcome(review=())),
                    forced_model("APPROVED"),
                    enabled=True,
                ).evaluate_case(test_case)
                self.assertEqual(
                    result.outcome.row.adjudication,
                    "NEEDS_REVIEW",
                )

    def test_untrusted_cue_normalization_catches_spaces_and_hyphens(self):
        variants = (
            "synthetic default",
            "crossed-out",
            "white text",
            "prompt-injection",
            "qr prompt",
            "foreign applicant",
        )
        for cue in variants:
            with self.subTest(cue=cue):
                field = resolved_field("fee_status", "paid", cues=(cue,))
                result = GatedHybridDecisionRecoveryAdjudicator(
                    FakeAdjudicator(outcome()),
                    forced_model("APPROVED"),
                    enabled=True,
                ).evaluate_case(complete_case(extras=(field,)))
                self.assertEqual(
                    result.outcome.row.adjudication,
                    "NEEDS_REVIEW",
                )
                self.assertIn(
                    "untrusted_visual_content",
                    result.veto_reasons,
                )

    def test_plain_text_layer_considered_candidate_vetoes_approval(self):
        visible = resolved_field("fee_status", "paid")
        hidden = candidate(
            "fee_status",
            "waived",
            evidence_type=EvidenceType.TEXT_LAYER,
            source="text_layer",
            cues=(),
        )
        field = replace(
            visible,
            considered=(visible.winning_evidence, hidden),
        )
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(outcome(review=())),
            forced_model("APPROVED"),
            enabled=True,
        ).evaluate_case(complete_case(extras=(field,)))
        self.assertEqual(result.outcome.row.adjudication, "NEEDS_REVIEW")
        self.assertIn("untrusted_visual_content", result.veto_reasons)

    def test_model_artifact_rejects_metadata_and_wrong_feature_contract(self):
        artifact = forced_model("APPROVED").members[0].to_dict()
        artifact["case_id"] = CASE_ID
        with self.assertRaises(ValueError):
            CompactModelArtifact.from_mapping(artifact)
        artifact = forced_model("APPROVED").members[0].to_dict()
        artifact["feature_names"][0] = "case_id"
        with self.assertRaises(ValueError):
            CompactModelArtifact.from_mapping(artifact)

    def test_fit_is_order_invariant_and_supports_external_feature_rows(self):
        base = IdentityFreeFeatureBuilder().build(complete_case(), outcome())
        rows = (
            base,
            IdentityFreeDecisionFeatures.from_mapping(
                {
                    name: (
                        1.0 - value
                        if name in {"baseline_review", "baseline_approved"}
                        else value
                    )
                    for name, value in base.values.items()
                }
            ),
            base,
        )
        labels = ("NEEDS_REVIEW", "APPROVED", "DENIED")
        first = CompactThreeClassModel.fit(rows, labels)
        second = CompactThreeClassModel.fit(
            tuple(reversed(rows)),
            tuple(reversed(labels)),
        )
        self.assertEqual(
            first.members[0].to_dict(),
            second.members[0].to_dict(),
        )
        self.assertEqual(first.predict(base), second.predict(base))

    def test_prediction_rejects_spoofed_argmax_margin_and_disagreement(self):
        probabilities = {
            "APPROVED": 0.80,
            "DENIED": 0.10,
            "NEEDS_REVIEW": 0.10,
        }
        for kwargs in (
            {"decision": "DENIED", "margin": 0.70, "disagreement": 0.0},
            {"decision": "APPROVED", "margin": 0.10, "disagreement": 0.0},
            {"decision": "APPROVED", "margin": 0.70, "disagreement": 0.4},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ModelPrediction(
                    probabilities=probabilities,
                    member_probabilities=(probabilities,),
                    **kwargs,
                )
        one_hot = {
            "APPROVED": 1.0,
            "DENIED": 0.0,
            "NEEDS_REVIEW": 0.0,
        }
        for diagnostics in (
            {"margin": True, "disagreement": 0.0},
            {"margin": 1.0, "disagreement": False},
        ):
            with self.subTest(diagnostics=diagnostics), self.assertRaises(
                ValueError
            ):
                ModelPrediction(
                    decision="APPROVED",
                    probabilities=one_hot,
                    member_probabilities=(one_hot,),
                    **diagnostics,
                )


class HardOrderAndGateTests(unittest.TestCase):
    def assert_nondecision_unchanged(self, before, after):
        for field_name, value in before.row.to_dict().items():
            if field_name != "adjudication":
                self.assertEqual(after.row.to_dict()[field_name], value)

    def test_validated_authoritative_policy_precedes_visible_violation(self):
        authority = resolved_field(
            "adjudication",
            "APPROVED",
            evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
            case_hint=None,
            subject_hint=None,
        )
        baseline = outcome(
            "APPROVED",
            authoritative=True,
            review=(),
            approval=("authoritative_visible_decision",),
            confidence=0.91,
        )
        risky = complete_case(
            overrides={"risk_flags": "active_warrant"},
            extras=(authority,),
        )
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(baseline),
            forced_model("DENIED"),
            enabled=True,
        ).evaluate_case(risky)
        self.assertIs(result.outcome, baseline)
        self.assertEqual(result.route, "validated_authoritative_policy")

    def test_unsubstantiated_authoritative_flag_cannot_bypass_violation(self):
        baseline = outcome(
            "APPROVED",
            authoritative=True,
            review=(),
            approval=("authoritative_visible_decision",),
        )
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(baseline),
            forced_model("APPROVED"),
            enabled=True,
        ).evaluate_case(
            complete_case(overrides={"risk_flags": "active_warrant"})
        )
        self.assertEqual(result.outcome.row.adjudication, "DENIED")
        self.assertEqual(result.route, "visible_policy_violation")

    def test_exact_signed_authority_is_binding_before_model(self):
        authority = resolved_field(
            "adjudication",
            "APPROVED",
            evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
        )
        baseline = outcome()
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(baseline),
            forced_model("DENIED"),
            enabled=True,
        ).evaluate_case(complete_case(extras=(authority,)))
        self.assertEqual(result.outcome.row.adjudication, "APPROVED")
        self.assertEqual(result.route, "binding_authority")
        self.assertEqual(result.outcome.row.confidence, baseline.row.confidence)

    def test_clean_signed_authority_precedes_stale_baseline_authority(self):
        authority = resolved_field(
            "adjudication",
            "DENIED",
            evidence_type=EvidenceType.ADJUDICATOR_STAMP,
        )
        baseline = outcome(
            "APPROVED",
            authoritative=True,
            review=(),
            approval=("authoritative_visible_decision",),
        )
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(baseline),
            forced_model("APPROVED"),
            enabled=True,
        ).evaluate_case(complete_case(extras=(authority,)))
        self.assertEqual(result.outcome.row.adjudication, "DENIED")
        self.assertEqual(result.route, "binding_authority")
        self.assertTrue(result.outcome.trace.authoritative_source)

    def test_explicit_clean_visible_violation_precedes_residual_model(self):
        baseline = outcome()
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(baseline),
            forced_model("APPROVED"),
            enabled=True,
        ).evaluate_case(
            complete_case(overrides={"risk_flags": "active_warrant"})
        )
        self.assertEqual(result.outcome.row.adjudication, "DENIED")
        self.assertEqual(result.route, "visible_policy_violation")
        self.assert_nondecision_unchanged(baseline, result.outcome)

    def test_deterministic_policy_approval_and_denial_are_immutable(self):
        for decision in ("APPROVED", "DENIED"):
            with self.subTest(decision=decision):
                baseline = outcome(
                    decision,
                    denial=("ordinary_policy_denial",)
                    if decision == "DENIED"
                    else (),
                    review=(),
                    approval=("strict_approval_bar_cleared",)
                    if decision == "APPROVED"
                    else (),
                )
                result = GatedHybridDecisionRecoveryAdjudicator(
                    FakeAdjudicator(baseline),
                    forced_model(
                        "DENIED" if decision == "APPROVED" else "APPROVED"
                    ),
                    enabled=True,
                ).evaluate_case(complete_case())
                self.assertIs(result.outcome, baseline)
                self.assertEqual(result.route, "deterministic_policy")

    def test_candidate_is_disabled_until_external_promotion(self):
        baseline = outcome(review=())
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(baseline),
            forced_model("APPROVED"),
        ).evaluate_case(complete_case())
        self.assertIs(result.outcome, baseline)
        self.assertEqual(result.route, "candidate_disabled")
        with self.assertRaises(TypeError):
            GatedHybridDecisionRecoveryAdjudicator(
                FakeAdjudicator(baseline),
                forced_model("APPROVED"),
                enabled="false",
            )

    def test_enabled_model_can_only_approve_fully_clean_empty_trace_residual(self):
        baseline = outcome(review=(), confidence=0.42)
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(baseline),
            forced_model("APPROVED"),
            enabled=True,
        ).evaluate_case(complete_case())
        self.assertEqual(result.outcome.row.adjudication, "APPROVED")
        self.assertEqual(result.outcome.row.confidence, 0.42)
        self.assert_nondecision_unchanged(baseline, result.outcome)

    def test_any_policy_review_category_vetoes_approval(self):
        reasons = (
            "required_output_unknown:home_world",
            "contested_field:fee_status",
            "required_output_not_visible:visa_class",
            "unsupported_fee_waiver",
            "unrecognized_policy_reason",
        )
        for reason in reasons:
            with self.subTest(reason=reason):
                result = GatedHybridDecisionRecoveryAdjudicator(
                    FakeAdjudicator(outcome(review=(reason,))),
                    forced_model("APPROVED"),
                    enabled=True,
                ).evaluate_case(complete_case())
                self.assertEqual(
                    result.outcome.row.adjudication,
                    "NEEDS_REVIEW",
                )

    def test_model_denial_without_explicit_violation_abstains(self):
        baseline = outcome(review=())
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(baseline),
            forced_model("DENIED"),
            enabled=True,
        ).evaluate_case(complete_case())
        self.assertIs(result.outcome, baseline)
        self.assertEqual(result.route, "denial_guard_review")
        self.assertIn(
            "denial_requires_visible_violation",
            result.veto_reasons,
        )

    def test_nonempty_unresolved_reasons_veto_even_without_flag(self):
        test_case = replace(
            complete_case(),
            unresolved_linkage=False,
            unresolved_reasons=("conflicting identity evidence",),
        )
        features = IdentityFreeFeatureBuilder().build(
            test_case,
            outcome(review=()),
        )
        self.assertEqual(features.values["unresolved_linkage"], 1.0)
        self.assertEqual(features.values["link_confidence"], 0.0)
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(outcome(review=())),
            forced_model("APPROVED"),
            enabled=True,
        ).evaluate_case(test_case)
        self.assertEqual(result.outcome.row.adjudication, "NEEDS_REVIEW")
        self.assertIn("unresolved_linkage", result.veto_reasons)

    def test_rescinded_decision_vetoes_model_approval(self):
        test_case = replace(complete_case(), rescinded_decision=True)
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(outcome(review=())),
            forced_model("APPROVED"),
            enabled=True,
        ).evaluate_case(test_case)
        self.assertEqual(result.outcome.row.adjudication, "NEEDS_REVIEW")
        self.assertIn("rescinded_decision", result.veto_reasons)

    def test_any_residual_denial_trace_vetoes_model_approval(self):
        for reason in (
            "disqualifying_flag:active_warrant",
            "barred_sponsor:visible",
            "ordinary_policy_denial",
        ):
            with self.subTest(reason=reason):
                result = GatedHybridDecisionRecoveryAdjudicator(
                    FakeAdjudicator(
                        outcome(review=(), denial=(reason,))
                    ),
                    forced_model("APPROVED"),
                    enabled=True,
                ).evaluate_case(complete_case())
                self.assertEqual(
                    result.outcome.row.adjudication,
                    "NEEDS_REVIEW",
                )
                self.assertIn(
                    "policy_review_other",
                    result.veto_reasons,
                )

    def test_exception_and_unsupported_authority_reviews_veto_approval(self):
        baselines = (
            outcome(review=(), exception_ids=("generic_exception",)),
            outcome(review=(), authoritative=True),
        )
        for baseline in baselines:
            with self.subTest(trace=baseline.trace):
                result = GatedHybridDecisionRecoveryAdjudicator(
                    FakeAdjudicator(baseline),
                    forced_model("APPROVED"),
                    enabled=True,
                ).evaluate_case(complete_case())
                self.assertEqual(
                    result.outcome.row.adjudication,
                    "NEEDS_REVIEW",
                )
                if baseline.trace.exception_ids:
                    self.assertIn(
                        "policy_review_other",
                        result.veto_reasons,
                    )
                else:
                    self.assertEqual(
                        result.route,
                        "deterministic_policy",
                    )

    def test_ensemble_disagreement_fails_closed(self):
        approved = forced_model("APPROVED").members[0]
        denied = forced_model("DENIED").members[0]
        ensemble = CompactThreeClassModel((approved, denied))
        result = GatedHybridDecisionRecoveryAdjudicator(
            FakeAdjudicator(outcome(review=())),
            ensemble,
            enabled=True,
            maximum_disagreement=0.01,
        ).evaluate_case(complete_case())
        self.assertEqual(result.outcome.row.adjudication, "NEEDS_REVIEW")
        self.assertIn("ensemble_disagreement", result.veto_reasons)

    def test_feature_level_rule_executes_same_hard_gate(self):
        baseline = outcome(review=())
        features = IdentityFreeFeatureBuilder().build(
            complete_case(),
            baseline,
        )
        approved = forced_model("APPROVED").predict(features)
        decision = GatedHybridDecisionRule().decide(
            "NEEDS_REVIEW",
            features,
            approved,
        )
        self.assertEqual(decision.decision, "APPROVED")
        denied = forced_model("DENIED").predict(features)
        decision = GatedHybridDecisionRule().decide(
            "NEEDS_REVIEW",
            features,
            denied,
        )
        self.assertEqual(decision.decision, "NEEDS_REVIEW")

    def test_unscoped_topology_does_not_become_predictive_presence(self):
        authority = resolved_field(
            "adjudication",
            "APPROVED",
            evidence_type=EvidenceType.SIGNED_MANUAL_NOTE,
            case_hint=None,
            subject_hint=None,
        )
        biometric = resolved_field(
            "biohazard_check",
            "clean",
            evidence_type=EvidenceType.BIOMETRIC_SLIP,
            case_hint=None,
            subject_hint=None,
        )
        features = IdentityFreeFeatureBuilder().build(
            complete_case(extras=(authority, biometric)),
            outcome(),
        )
        self.assertEqual(features.values["manual_authority_present"], 0.0)
        self.assertEqual(features.values["biometric_evidence_present"], 0.0)


if __name__ == "__main__":
    unittest.main()
