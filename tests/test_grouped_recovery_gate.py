from __future__ import annotations

import math
import unittest
from dataclasses import replace

from devtools.experiment_control import require_aggregate_only
from devtools.grouped_recovery_gate import (
    PUBLIC_EXPOSED_SCOPE_LABEL,
    FoldScorePair,
    FullRunAggregate,
    GroupedRecoveryEvidence,
    GroupedRecoveryEvidenceError,
    GroupedRecoveryGate,
    RecoveryAuditAggregate,
)


MANIFEST_SHA256 = "a" * 64
RECORD_COUNTS = (4, 3, 3, 3, 2)
GROUP_COUNTS = (2, 2, 2, 1, 1)
PASSING_DELTAS = (
    (0.40, 0.20, 0.10, 0.0, 0.0),
    (0.30, 0.20, 0.08, 0.0, 0.0),
    (0.25, 0.15, 0.05, 0.0, 0.0),
)


def _folds(
    deltas: tuple[tuple[float, ...], ...] = PASSING_DELTAS,
) -> tuple[FoldScorePair, ...]:
    return tuple(
        FoldScorePair(
            repeat=repeat,
            fold=fold,
            control_score=100.0,
            candidate_score=100.0 + deltas[repeat][fold],
            control_record_count=RECORD_COUNTS[fold],
            candidate_record_count=RECORD_COUNTS[fold],
            layout_group_count=GROUP_COUNTS[fold],
        )
        for repeat in range(3)
        for fold in range(5)
    )


def _full_run(**overrides: object) -> FullRunAggregate:
    values: dict[str, object] = {
        "total_score": 130.0,
        "record_count": 15,
        "catastrophic_false_approvals": 0,
        "missing_records": 0,
        "invalid_records": 0,
        "false_positive_denial_recoveries": 0,
        "deterministic": True,
        "duplicate_records": 0,
        "extra_records": 0,
    }
    values.update(overrides)
    return FullRunAggregate(**values)  # type: ignore[arg-type]


def _evidence(**overrides: object) -> GroupedRecoveryEvidence:
    values: dict[str, object] = {
        "layout_manifest_sha256": MANIFEST_SHA256,
        "expected_record_count": 15,
        "expected_layout_group_count": 8,
        "control_full": _full_run(),
        "candidate_full": _full_run(total_score=132.0),
        "candidate_recovery_audit": RecoveryAuditAggregate(
            recovered_field_count=12,
            recovered_field_complete_provenance_count=12,
            serialization_default_used_as_evidence_count=0,
        ),
        "folds": _folds(),
        "manifest_frozen_before_scoring": True,
        "group_exclusive": True,
        "paired_fold_members": True,
        "split_deterministic": True,
    }
    values.update(overrides)
    return GroupedRecoveryEvidence(**values)  # type: ignore[arg-type]


class GroupedRecoveryGateTests(unittest.TestCase):
    def test_passing_evidence_computes_paired_weighted_deltas(self):
        decision = GroupedRecoveryGate().evaluate(_evidence())

        self.assertTrue(decision.passed)
        self.assertEqual(decision.decision, "PASSED")
        self.assertEqual(decision.blocking_reasons, ())
        self.assertEqual(decision.full_score_delta, 2.0)
        self.assertEqual(decision.repeat_positive_fold_counts, (3, 3, 3))

        first_weighted = sum(
            delta * count
            for delta, count in zip(PASSING_DELTAS[0], RECORD_COUNTS)
        ) / sum(RECORD_COUNTS)
        self.assertAlmostEqual(decision.repeat_weighted_deltas[0], first_weighted)
        self.assertTrue(
            all(delta > 0 for delta in decision.repeat_weighted_deltas)
        )
        self.assertTrue(
            all(
                delta > 0
                for delta in decision.repeat_leave_best_fold_out_deltas
            )
        )
        self.assertEqual(decision.folds[0].score_delta, 0.4)
        self.assertEqual(decision.folds[-1].score_delta, 0.0)
        self.assertFalse(decision.concentration_warning)
        self.assertTrue(all(dict(decision.diagnostic_results).values()))

    def test_serialized_result_is_public_exposed_and_aggregate_only(self):
        aggregate = GroupedRecoveryGate().evaluate(_evidence()).to_aggregate_evidence()

        require_aggregate_only(aggregate)
        self.assertEqual(
            aggregate["evaluation_mode"], PUBLIC_EXPOSED_SCOPE_LABEL
        )
        self.assertEqual(
            PUBLIC_EXPOSED_SCOPE_LABEL,
            "public_grouped_robustness_not_unseen",
        )
        self.assertEqual(aggregate["status"], "passed")
        self.assertEqual(aggregate["layout_manifest_sha256"], MANIFEST_SHA256)
        self.assertEqual(aggregate["repeat_count"], 3)
        self.assertEqual(aggregate["fold_count"], 15)
        self.assertEqual(aggregate["provenance_coverage_fraction"], 1.0)
        self.assertEqual(
            aggregate["serialization_default_used_as_evidence_count"], 0
        )
        self.assertEqual(aggregate["warning_count"], 0)
        self.assertFalse(
            aggregate["checks"]["score_gain_concentration_warning"]
        )
        self.assertNotIn("blocking_reasons", aggregate)

    def test_input_order_does_not_change_decision_or_aggregate_bytes(self):
        evidence = _evidence()
        reversed_evidence = replace(evidence, folds=tuple(reversed(evidence.folds)))

        normal = GroupedRecoveryGate().evaluate(evidence)
        reversed_result = GroupedRecoveryGate().evaluate(reversed_evidence)

        self.assertEqual(normal, reversed_result)
        self.assertEqual(
            normal.to_aggregate_evidence(),
            reversed_result.to_aggregate_evidence(),
        )

    def test_full_score_must_be_strictly_positive(self):
        evidence = _evidence(candidate_full=_full_run(total_score=130.0))

        decision = GroupedRecoveryGate().evaluate(evidence)

        self.assertFalse(decision.passed)
        self.assertIn("full_score_positive", decision.blocking_reasons)

    def test_each_repeat_weighted_delta_must_be_positive(self):
        deltas = (
            (0.01, 0.01, 0.01, -1.0, -1.0),
            PASSING_DELTAS[1],
            PASSING_DELTAS[2],
        )

        decision = GroupedRecoveryGate().evaluate(
            _evidence(folds=_folds(deltas))
        )

        self.assertFalse(decision.passed)
        self.assertIn(
            "repeat_weighted_deltas_positive", decision.blocking_reasons
        )

    def test_each_repeat_must_have_at_least_one_positive_fold(self):
        deltas = (
            (0.0, 0.0, 0.0, 0.0, 0.0),
            PASSING_DELTAS[1],
            PASSING_DELTAS[2],
        )

        decision = GroupedRecoveryGate().evaluate(
            _evidence(folds=_folds(deltas))
        )

        self.assertFalse(decision.passed)
        self.assertIn(
            "repeat_weighted_deltas_positive", decision.blocking_reasons
        )
        self.assertIn(
            "at_least_one_positive_fold_per_repeat",
            decision.blocking_reasons,
        )

    def test_negative_fold_blocks_even_when_repeat_and_leave_best_are_positive(self):
        deltas = (
            (1.0, 1.0, 0.0, 0.0, -0.01),
            PASSING_DELTAS[1],
            PASSING_DELTAS[2],
        )

        decision = GroupedRecoveryGate().evaluate(
            _evidence(folds=_folds(deltas))
        )

        self.assertFalse(decision.passed)
        self.assertNotIn(
            "repeat_weighted_deltas_positive", decision.blocking_reasons
        )
        self.assertNotIn(
            "at_least_one_positive_fold_per_repeat",
            decision.blocking_reasons,
        )
        self.assertNotIn(
            "leave_best_fold_out_nonnegative", decision.blocking_reasons
        )
        self.assertIn("no_negative_folds", decision.blocking_reasons)

    def test_single_positive_fold_passes_with_non_hard_concentration_warning(self):
        sparse = (
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0),
        )

        decision = GroupedRecoveryGate().evaluate(
            _evidence(folds=_folds(sparse))
        )
        aggregate = decision.to_aggregate_evidence()

        self.assertTrue(decision.passed)
        self.assertEqual(decision.blocking_reasons, ())
        self.assertEqual(
            decision.repeat_positive_fold_counts,
            (1, 1, 1),
        )
        self.assertTrue(
            all(delta > 0 for delta in decision.repeat_weighted_deltas)
        )
        self.assertTrue(
            all(
                delta == 0
                for delta in decision.repeat_leave_best_fold_out_deltas
            )
        )
        self.assertEqual(
            dict(decision.diagnostic_results),
            {
                "fold_majority_positive": False,
                "leave_best_fold_out_positive": False,
            },
        )
        self.assertTrue(decision.concentration_warning)
        self.assertEqual(aggregate["status"], "passed")
        self.assertEqual(aggregate["hard_gate_failure_count"], 0)
        self.assertEqual(aggregate["warning_count"], 1)
        self.assertTrue(
            aggregate["checks"]["score_gain_concentration_warning"]
        )

    def test_safety_completeness_and_recovery_audits_are_hard_gates(self):
        scenarios = {
            "catastrophic": (
                {
                    "candidate_full": _full_run(
                        total_score=132.0,
                        catastrophic_false_approvals=1,
                    )
                },
                "no_catastrophic_false_approvals",
            ),
            "missing": (
                {
                    "candidate_full": _full_run(
                        total_score=132.0, missing_records=1
                    )
                },
                "candidate_complete",
            ),
            "invalid": (
                {
                    "candidate_full": _full_run(
                        total_score=132.0, invalid_records=1
                    )
                },
                "candidate_complete",
            ),
            "duplicate": (
                {
                    "candidate_full": _full_run(
                        total_score=132.0, duplicate_records=1
                    )
                },
                "candidate_complete",
            ),
            "extra": (
                {
                    "candidate_full": _full_run(
                        total_score=132.0, extra_records=1
                    )
                },
                "candidate_complete",
            ),
            "false_positive_denial_recovery": (
                {
                    "candidate_full": _full_run(
                        total_score=132.0,
                        false_positive_denial_recoveries=1,
                    )
                },
                "no_increased_false_positive_denial_recoveries",
            ),
            "candidate_nondeterministic": (
                {
                    "candidate_full": _full_run(
                        total_score=132.0, deterministic=False
                    )
                },
                "run_deterministic",
            ),
            "control_nondeterministic": (
                {"control_full": _full_run(deterministic=False)},
                "run_deterministic",
            ),
            "incomplete_provenance": (
                {
                    "candidate_recovery_audit": RecoveryAuditAggregate(
                        12, 11, 0
                    )
                },
                "provenance_complete",
            ),
            "vacuous_provenance": (
                {
                    "candidate_recovery_audit": RecoveryAuditAggregate(
                        0, 0, 0
                    )
                },
                "provenance_complete",
            ),
            "serialization_default": (
                {
                    "candidate_recovery_audit": RecoveryAuditAggregate(
                        12, 12, 1
                    )
                },
                "no_serialization_default_as_evidence",
            ),
            "manifest_not_frozen": (
                {"manifest_frozen_before_scoring": False},
                "manifest_frozen_before_scoring",
            ),
            "not_group_exclusive": (
                {"group_exclusive": False},
                "group_exclusive",
            ),
            "unpaired": (
                {"paired_fold_members": False},
                "paired_fold_members",
            ),
            "split_nondeterministic": (
                {"split_deterministic": False},
                "split_deterministic",
            ),
        }

        for name, (overrides, reason) in scenarios.items():
            with self.subTest(name=name):
                decision = GroupedRecoveryGate().evaluate(
                    _evidence(**overrides)
                )
                self.assertFalse(decision.passed)
                self.assertIn(reason, decision.blocking_reasons)
                require_aggregate_only(decision.to_aggregate_evidence())

    def test_unchanged_false_positive_denial_recoveries_are_allowed(self):
        evidence = _evidence(
            control_full=_full_run(false_positive_denial_recoveries=2),
            candidate_full=_full_run(
                total_score=132.0, false_positive_denial_recoveries=2
            ),
        )

        decision = GroupedRecoveryGate().evaluate(evidence)

        self.assertTrue(decision.passed)

    def test_missing_duplicate_or_incomplete_fold_grid_is_rejected(self):
        complete = _folds()
        malformed = {
            "missing": complete[:-1],
            "duplicate": complete[:-1] + (complete[0],),
            "outside_grid": complete[:-1]
            + (replace(complete[-1], repeat=3),),
        }

        for name, folds in malformed.items():
            with self.subTest(name=name):
                with self.assertRaises(GroupedRecoveryEvidenceError):
                    GroupedRecoveryGate().evaluate(_evidence(folds=folds))

    def test_every_repeat_must_cover_the_frozen_records_and_groups(self):
        complete = list(_folds())
        complete[0] = replace(
            complete[0],
            control_record_count=5,
            candidate_record_count=5,
        )
        with self.assertRaisesRegex(
            GroupedRecoveryEvidenceError, "expected record count"
        ):
            GroupedRecoveryGate().evaluate(_evidence(folds=tuple(complete)))

        complete = list(_folds())
        complete[0] = replace(complete[0], layout_group_count=3)
        with self.assertRaisesRegex(
            GroupedRecoveryEvidenceError, "expected layout group count"
        ):
            GroupedRecoveryGate().evaluate(_evidence(folds=tuple(complete)))

    def test_candidate_and_control_fold_counts_must_be_identical(self):
        with self.assertRaisesRegex(
            GroupedRecoveryEvidenceError, "record counts must match"
        ):
            FoldScorePair(
                repeat=0,
                fold=0,
                control_score=100,
                candidate_score=101,
                control_record_count=3,
                candidate_record_count=2,
                layout_group_count=1,
            )

    def test_malformed_hash_counts_flags_scores_and_provenance_are_rejected(self):
        with self.assertRaises(GroupedRecoveryEvidenceError):
            _evidence(layout_manifest_sha256="not-a-hash")
        with self.assertRaises(GroupedRecoveryEvidenceError):
            _evidence(expected_record_count=0)
        with self.assertRaises(GroupedRecoveryEvidenceError):
            _evidence(group_exclusive=1)
        with self.assertRaises(GroupedRecoveryEvidenceError):
            _full_run(total_score=math.nan)
        with self.assertRaises(GroupedRecoveryEvidenceError):
            RecoveryAuditAggregate(1, 2, 0)


if __name__ == "__main__":
    unittest.main()
