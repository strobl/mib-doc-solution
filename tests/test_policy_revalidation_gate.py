from __future__ import annotations

import unittest
from dataclasses import replace

from devtools.experiment_control import require_aggregate_only
from devtools.policy_revalidation_audit_contract import (
    CONTRACT_AUDIT_COUNTS,
    COHORT_AUDIT_COUNTS,
)
from devtools.policy_revalidation_gate import (
    PolicyExecutionAudit,
    PolicyRevalidationEvidence,
    PolicyRevalidationGate,
    PolicyRunAggregate,
)


def _run(**overrides: object) -> PolicyRunAggregate:
    values: dict[str, object] = {
        "total_score": 130.0,
        "extraction_score": 50.0,
        "classification_score": 65.0,
        "calibration_score": 15.0,
        "record_count": 32,
        "catastrophic_false_approvals": 0,
        "false_positive_denials": 0,
        "missing_records": 0,
        "invalid_records": 0,
        "duplicate_records": 0,
        "extra_records": 0,
        "approved_to_needs_review_count": 3,
        "denied_to_needs_review_count": 4,
        "deterministic": True,
    }
    values.update(overrides)
    return PolicyRunAggregate(**values)  # type: ignore[arg-type]


def _cohort(**overrides: int) -> dict[str, int]:
    values = {name: 0 for name in COHORT_AUDIT_COUNTS}
    values["accepted_final_policy_result_count"] = 32
    values.update(overrides)
    return values


def _contract(**overrides: int) -> dict[str, int]:
    values = {name: 0 for name in CONTRACT_AUDIT_COUNTS}
    values.update(
        {
            "legacy_synthetic_before_late_recovery_count": 2,
            "candidate_late_recovery_before_revalidation_count": 4,
            "candidate_revalidation_after_late_recovery_count": 4,
            "contradicted_synthetic_reason_before_count": 2,
            "contradicted_synthetic_reason_removed_count": 2,
            "independent_denial_reason_retained_count": 1,
            "review_confidence_restored_count": 1,
            "normal_policy_rerun_count": 4,
            "signed_late_authority_recovery_count": 3,
            "late_adjudication_evidence_preserved_count": 3,
            "late_biohazard_evidence_preserved_count": 1,
            "placeholder_guard_probe_count": 35,
            "sentinel_guard_probe_count": 2,
            "serialization_default_guard_probe_count": 35,
            "stale_threshold_guard_probe_count": 2,
            "forced_approval_guard_probe_count": 35,
            "direct_approval_head_guard_probe_count": 1,
        }
    )
    values.update(overrides)
    return values


def _evidence(**overrides: object) -> PolicyRevalidationEvidence:
    values: dict[str, object] = {
        "layout_manifest_sha256": "a" * 64,
        "expected_record_count": 32,
        "layout_group_count": 9,
        "legacy_control": _run(),
        "candidate": _run(
            total_score=134.0,
            classification_score=69.0,
            approved_to_needs_review_count=2,
            denied_to_needs_review_count=3,
        ),
        "audit": PolicyExecutionAudit(
            cohort_counts=_cohort(),
            contract_counts=_contract(),
            deterministic=True,
        ),
        "new_catastrophic_false_approval_count": 0,
        "new_false_positive_denial_count": 0,
        "new_missing_record_count": 0,
        "new_invalid_record_count": 0,
        "non_policy_field_change_count": 0,
        "manifest_frozen_before_scoring": True,
    }
    values.update(overrides)
    return PolicyRevalidationEvidence(**values)  # type: ignore[arg-type]


class PolicyRevalidationGateTests(unittest.TestCase):
    def test_passing_gate_emits_identity_free_before_after_evidence(self):
        decision = PolicyRevalidationGate().evaluate(_evidence())
        aggregate = decision.to_aggregate_evidence()

        self.assertTrue(decision.passed)
        self.assertEqual(decision.blocking_reasons, ())
        require_aggregate_only(aggregate)
        self.assertEqual(aggregate["status"], "passed")
        self.assertEqual(aggregate["score_delta"], 4.0)
        self.assertEqual(
            aggregate["confusion_counts"],
            {
                "approved_to_needs_review_control_count": 3,
                "approved_to_needs_review_candidate_count": 2,
                "approved_to_needs_review_delta": -1,
                "denied_to_needs_review_control_count": 4,
                "denied_to_needs_review_candidate_count": 3,
                "denied_to_needs_review_delta": -1,
            },
        )
        self.assertEqual(
            aggregate["counts"][
                "contract_signed_late_authority_recovery_count"
            ],
            3,
        )

    def test_zero_cohort_occurrence_is_allowed_when_contract_probe_is_nonvacuous(self):
        evidence = _evidence(
            candidate=_run(
                total_score=130.0,
                approved_to_needs_review_count=3,
                denied_to_needs_review_count=4,
            )
        )

        decision = PolicyRevalidationGate().evaluate(evidence)

        self.assertTrue(decision.passed)
        aggregate = decision.to_aggregate_evidence()
        self.assertFalse(
            aggregate["checks"]["targeted_confusion_changed"]
        )
        self.assertFalse(
            aggregate["checks"]["cohort_revalidation_observed"]
        )

    def test_nonzero_cohort_trace_must_be_fully_accounted(self):
        consistent = _cohort(
            late_recovery_before_revalidation_count=3,
            revalidation_after_late_recovery_count=3,
            normal_policy_rerun_count=3,
            contradicted_synthetic_reason_before_count=2,
            contradicted_synthetic_reason_removed_count=2,
            independent_denial_reason_retained_count=1,
        )
        evidence = _evidence(
            audit=PolicyExecutionAudit(
                cohort_counts=consistent,
                contract_counts=_contract(),
                deterministic=True,
            )
        )
        self.assertTrue(PolicyRevalidationGate().evaluate(evidence).passed)

        inconsistent = dict(consistent)
        inconsistent["revalidation_after_late_recovery_count"] = 2
        decision = PolicyRevalidationGate().evaluate(
            replace(
                evidence,
                audit=PolicyExecutionAudit(
                    cohort_counts=inconsistent,
                    contract_counts=_contract(),
                    deterministic=True,
                ),
            )
        )
        self.assertFalse(decision.passed)
        self.assertIn(
            "cohort_execution_order_accounted",
            decision.blocking_reasons,
        )

    def test_every_named_contract_behavior_is_a_hard_gate(self):
        scenarios = {
            "legacy order": (
                {"legacy_synthetic_before_late_recovery_count": 0},
                "contract_legacy_order_exercised",
            ),
            "candidate order": (
                {
                    "candidate_late_recovery_before_revalidation_count": 0,
                    "candidate_revalidation_after_late_recovery_count": 0,
                    "normal_policy_rerun_count": 0,
                },
                "contract_candidate_order_exercised",
            ),
            "contradiction": (
                {
                    "contradicted_synthetic_reason_before_count": 1,
                    "contradicted_synthetic_reason_removed_count": 0,
                    "contradicted_synthetic_reason_remaining_count": 1,
                },
                "contract_contradiction_removed",
            ),
            "independent denial": (
                {"independent_denial_reason_retained_count": 0},
                "contract_independent_denial_retained",
            ),
            "review confidence": (
                {"review_confidence_restored_count": 0},
                "contract_review_confidence_restored",
            ),
            "signed authority": (
                {"signed_late_authority_recovery_count": 0},
                "contract_signed_late_authority_recovered",
            ),
            "adjudication evidence": (
                {"late_adjudication_evidence_preserved_count": 0},
                "contract_late_decision_evidence_preserved",
            ),
            "biohazard evidence": (
                {"late_biohazard_evidence_preserved_count": 0},
                "contract_late_biohazard_evidence_preserved",
            ),
        }
        for label, (changes, expected_gate) in scenarios.items():
            with self.subTest(label=label):
                decision = PolicyRevalidationGate().evaluate(
                    _evidence(
                        audit=PolicyExecutionAudit(
                            cohort_counts=_cohort(),
                            contract_counts=_contract(**changes),
                            deterministic=True,
                        )
                    )
                )
                self.assertFalse(decision.passed)
                self.assertIn(expected_gate, decision.blocking_reasons)

    def test_placeholder_sentinel_threshold_and_forced_approval_are_hard_failures(self):
        for name in (
            "forced_approval_count",
            "sentinel_value_used_as_evidence_count",
            "placeholder_value_used_as_evidence_count",
            "serialization_default_used_as_evidence_count",
            "stale_threshold_mismatch_count",
            "contradicted_synthetic_reason_remaining_count",
        ):
            with self.subTest(name=name):
                values = _contract(**{name: 1})
                if name == "contradicted_synthetic_reason_remaining_count":
                    values["contradicted_synthetic_reason_before_count"] = 3
                decision = PolicyRevalidationGate().evaluate(
                    _evidence(
                        audit=PolicyExecutionAudit(
                            cohort_counts=_cohort(),
                            contract_counts=values,
                            deterministic=True,
                        )
                    )
                )
                self.assertFalse(decision.passed)
                self.assertIn(
                    "contract_unsafe_counters_zero",
                    decision.blocking_reasons,
                )

    def test_contract_guard_coverage_thresholds_fail_closed(self):
        scenarios = {
            "placeholder matrix": (
                {"placeholder_guard_probe_count": 34},
                "contract_placeholder_guards_exercised",
            ),
            "sentinel matrix": (
                {"sentinel_guard_probe_count": 1},
                "contract_placeholder_guards_exercised",
            ),
            "serialization-default matrix": (
                {"serialization_default_guard_probe_count": 34},
                "contract_placeholder_guards_exercised",
            ),
            "stale-threshold boundaries": (
                {"stale_threshold_guard_probe_count": 1},
                "contract_policy_safety_guards_exercised",
            ),
            "forced-approval matrix": (
                {"forced_approval_guard_probe_count": 34},
                "contract_policy_safety_guards_exercised",
            ),
            "direct approval head": (
                {"direct_approval_head_guard_probe_count": 0},
                "contract_policy_safety_guards_exercised",
            ),
        }
        for label, (changes, expected_gate) in scenarios.items():
            with self.subTest(label=label):
                decision = PolicyRevalidationGate().evaluate(
                    _evidence(
                        audit=PolicyExecutionAudit(
                            cohort_counts=_cohort(),
                            contract_counts=_contract(**changes),
                            deterministic=True,
                        )
                    )
                )
                self.assertFalse(decision.passed)
                self.assertIn(expected_gate, decision.blocking_reasons)

    def test_cohort_unsafe_counter_is_a_hard_failure(self):
        decision = PolicyRevalidationGate().evaluate(
            _evidence(
                audit=PolicyExecutionAudit(
                    cohort_counts=_cohort(
                        placeholder_value_used_as_evidence_count=1
                    ),
                    contract_counts=_contract(),
                    deterministic=True,
                )
            )
        )

        self.assertFalse(decision.passed)
        self.assertIn(
            "cohort_unsafe_counters_zero",
            decision.blocking_reasons,
        )

    def test_confusion_safety_completeness_and_scope_rules_fail_closed(self):
        scenarios = {
            "approved review regression": (
                {
                    "candidate": _run(
                        total_score=134.0,
                        classification_score=69.0,
                        approved_to_needs_review_count=4,
                        denied_to_needs_review_count=3,
                    )
                },
                "approved_to_review_non_regression",
            ),
            "denied review regression": (
                {
                    "candidate": _run(
                        total_score=134.0,
                        classification_score=69.0,
                        approved_to_needs_review_count=2,
                        denied_to_needs_review_count=5,
                    )
                },
                "denied_to_review_non_regression",
            ),
            "new catastrophic": (
                {"new_catastrophic_false_approval_count": 1},
                "no_new_catastrophic_false_approvals",
            ),
            "new false denial": (
                {"new_false_positive_denial_count": 1},
                "no_new_false_positive_denials",
            ),
            "missing": (
                {
                    "candidate": _run(
                        total_score=134.0,
                        record_count=31,
                        missing_records=1,
                    ),
                    "new_missing_record_count": 1,
                },
                "candidate_complete",
            ),
            "invalid": (
                {
                    "candidate": _run(
                        total_score=134.0, invalid_records=1
                    ),
                    "new_invalid_record_count": 1,
                },
                "candidate_complete",
            ),
            "field drift": (
                {"non_policy_field_change_count": 1},
                "non_policy_fields_unchanged",
            ),
            "nondeterministic audit": (
                {
                    "audit": PolicyExecutionAudit(
                        cohort_counts=_cohort(),
                        contract_counts=_contract(),
                        deterministic=False,
                    )
                },
                "execution_audits_deterministic",
            ),
        }
        for label, (changes, expected_gate) in scenarios.items():
            with self.subTest(label=label):
                decision = PolicyRevalidationGate().evaluate(
                    _evidence(**changes)
                )
                self.assertFalse(decision.passed)
                self.assertIn(expected_gate, decision.blocking_reasons)

    def test_audit_count_contract_rejects_missing_extra_and_invalid_values(self):
        with self.assertRaisesRegex(
            Exception, "exact required keys"
        ):
            PolicyExecutionAudit(
                cohort_counts={
                    key: value
                    for key, value in _cohort().items()
                    if key != "accepted_final_policy_result_count"
                },
                contract_counts=_contract(),
                deterministic=True,
            )
        with self.assertRaisesRegex(Exception, "non-negative"):
            PolicyExecutionAudit(
                cohort_counts=_cohort(),
                contract_counts=_contract(
                    signed_late_authority_recovery_count=-1
                ),
                deterministic=True,
            )


if __name__ == "__main__":
    unittest.main()
