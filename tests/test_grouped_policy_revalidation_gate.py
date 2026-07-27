from __future__ import annotations

import unittest
from dataclasses import replace

from devtools.experiment_control import require_aggregate_only
from devtools.grouped_policy_revalidation_gate import (
    GroupedPolicyEvidence,
    GroupedPolicyGateError,
    GroupedPolicyRevalidationGate,
    PolicyActivityAggregate,
    PolicyArmAggregate,
    PolicyFoldPair,
)


def _arm(*, candidate: bool = False, **overrides: object) -> PolicyArmAggregate:
    values: dict[str, object] = {
        "total_score": 131.0 if candidate else 130.0,
        "extraction_score": 50.0,
        "classification_score": 66.0 if candidate else 65.0,
        "calibration_score": 15.0,
        "missing_penalty": 0.0,
        "record_count": 1000,
        "catastrophic_false_approvals": 0,
        "false_approvals": 0,
        "missing_records": 0,
        "invalid_records": 0,
        "duplicate_records": 0,
        "extra_records": 0,
        "deterministic": True,
    }
    values.update(overrides)
    return PolicyArmAggregate(**values)  # type: ignore[arg-type]


def _folds() -> tuple[PolicyFoldPair, ...]:
    return tuple(
        PolicyFoldPair(
            repeat=repeat,
            fold=fold,
            record_count=200,
            layout_group_count=5,
            control_score=130.0,
            candidate_score=131.0,
        )
        for repeat in range(3)
        for fold in range(5)
    )


def _evidence(**overrides: object) -> GroupedPolicyEvidence:
    values: dict[str, object] = {
        "control_source_revision_sha": "a" * 40,
        "candidate_source_revision_sha": "b" * 40,
        "experiment_plan_sha256": "1" * 64,
        "runtime_contract_sha256": "2" * 64,
        "split_manifest_sha256": "3" * 64,
        "input_tree_sha256": "4" * 64,
        "truth_sha256": "5" * 64,
        "evaluator_sha256": "6" * 64,
        "candidate_diff_manifest_sha256": "7" * 64,
        "expected_record_count": 1000,
        "expected_layout_group_count": 25,
        "control": _arm(),
        "candidate": _arm(candidate=True),
        "activity": PolicyActivityAggregate(
            eligible_guarded_initial_count=1,
            guarded_initial_approval_count=1,
            unguarded_initial_approval_count=0,
            late_revalidation_approval_count=0,
            legacy_forced_approval_count=1,
        ),
        "folds": _folds(),
        "new_false_approval_count": 0,
        "new_catastrophic_false_approval_count": 0,
        "non_decision_field_change_count": 0,
        "decision_or_confidence_change_count": 1,
        "source_and_diff_bound": True,
        "population_bound": True,
        "group_exclusive": True,
        "paired_fold_members": True,
        "split_deterministic": True,
        "runtime_contract_bound": True,
        "capture_contract_checks": {
            "authoritative_review_veto": True,
            "wrong_scope_veto": True,
        },
        "regression_counts": {
            "focused_failure": 0,
            "full_failure": 0,
        },
    }
    values.update(overrides)
    return GroupedPolicyEvidence(**values)  # type: ignore[arg-type]


class GroupedPolicyRevalidationGateTests(unittest.TestCase):
    def test_complete_positive_3x5_evidence_passes_and_is_aggregate_only(self):
        decision = GroupedPolicyRevalidationGate().evaluate(_evidence())
        aggregate = decision.to_aggregate_evidence()

        self.assertTrue(decision.passed)
        self.assertEqual(len(aggregate["fold_deltas"]), 15)
        self.assertEqual(sum(aggregate["fold_weights"][:5]), 1000)
        self.assertEqual(
            aggregate["counts"]["legacy_forced_approval_count"], 1
        )
        self.assertEqual(
            aggregate["counts"]["guarded_initial_approval_count"], 1
        )
        require_aggregate_only(aggregate)
        self.assertNotIn("MIB-", str(aggregate))

    def test_negative_fold_and_leave_best_concentration_block(self):
        folds = list(_folds())
        folds[0] = replace(
            folds[0], control_score=130.0, candidate_score=129.0
        )
        decision = GroupedPolicyRevalidationGate().evaluate(
            _evidence(folds=tuple(folds))
        )
        self.assertFalse(decision.passed)
        self.assertIn("no_negative_folds", decision.blocking_reasons)

        concentrated = list(_folds())
        for index in range(5):
            concentrated[index] = replace(
                concentrated[index],
                candidate_score=140.0 if index == 0 else 130.0,
            )
        decision = GroupedPolicyRevalidationGate().evaluate(
            _evidence(folds=tuple(concentrated))
        )
        self.assertIn(
            "leave_best_fold_out_positive", decision.blocking_reasons
        )

    def test_safety_activity_contract_and_regression_fail_closed(self):
        scenarios = {
            "new_false": {
                "new_false_approval_count": 1,
            },
            "unguarded": {
                "activity": replace(
                    _evidence().activity,
                    guarded_initial_approval_count=0,
                    unguarded_initial_approval_count=1,
                )
            },
            "late": {
                "activity": replace(
                    _evidence().activity,
                    guarded_initial_approval_count=0,
                    late_revalidation_approval_count=1,
                )
            },
            "legacy_mismatch": {
                "activity": replace(
                    _evidence().activity,
                    legacy_forced_approval_count=2,
                )
            },
            "contract": {
                "capture_contract_checks": {
                    "authoritative_review_veto": False
                }
            },
            "regression": {
                "regression_counts": {"focused_failure": 1}
            },
        }
        for label, changes in scenarios.items():
            with self.subTest(label=label):
                decision = GroupedPolicyRevalidationGate().evaluate(
                    _evidence(**changes)
                )
                self.assertFalse(decision.passed)

    def test_exact_fold_coordinates_and_component_arithmetic_are_strict(self):
        with self.assertRaises(GroupedPolicyGateError):
            GroupedPolicyRevalidationGate().evaluate(
                _evidence(folds=_folds()[:-1])
            )
        with self.assertRaises(GroupedPolicyGateError):
            _arm(total_score=129.0)


if __name__ == "__main__":
    unittest.main()
