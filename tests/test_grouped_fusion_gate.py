from __future__ import annotations

import hashlib
import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

from devtools.experiment_control import require_aggregate_only
from devtools.grouped_fusion_gate import (
    PUBLIC_EXPOSED_SCOPE_LABEL,
    FoldScorePair,
    FusionAuditAggregate,
    FusionRunAggregate,
    GroupedFusionEvidence,
    GroupedFusionEvidenceError,
    GroupedFusionGate,
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


def _run(**overrides: object) -> FusionRunAggregate:
    values: dict[str, object] = {
        "total_score": 130.0,
        "record_count": 15,
        "catastrophic_false_approvals": 0,
        "false_positive_denials": 0,
        "missing_records": 0,
        "invalid_records": 0,
        "deterministic": True,
        "duplicate_records": 0,
        "extra_records": 0,
    }
    values.update(overrides)
    return FusionRunAggregate(**values)  # type: ignore[arg-type]


def _audit(**overrides: object) -> FusionAuditAggregate:
    values: dict[str, object] = {
        "changed_field_count": 12,
        "changed_field_complete_provenance_count": 12,
        "clean_higher_authority_override_count": 0,
        "binding_authority_override_count": 0,
        "text_layer_winner_count": 0,
        "serialization_default_used_as_evidence_count": 0,
        "correlated_views_collapsed": 7,
        "independent_agreement_resolutions": 4,
        "same_rank_contested_count": 3,
        "cross_applicant_candidates_excluded": 2,
    }
    values.update(overrides)
    return FusionAuditAggregate(**values)  # type: ignore[arg-type]


def _evidence(**overrides: object) -> GroupedFusionEvidence:
    values: dict[str, object] = {
        "layout_manifest_sha256": MANIFEST_SHA256,
        "expected_record_count": 15,
        "expected_layout_group_count": 8,
        "legacy_control_full": _run(),
        "candidate_full": _run(total_score=132.0),
        "candidate_fusion_audit": _audit(),
        "new_catastrophic_false_approval_count": 0,
        "new_false_positive_denial_count": 0,
        "folds": _folds(),
        "manifest_frozen_before_scoring": True,
        "group_exclusive": True,
        "paired_fold_members": True,
        "split_deterministic": True,
    }
    values.update(overrides)
    return GroupedFusionEvidence(**values)  # type: ignore[arg-type]


class GroupedFusionGateTests(unittest.TestCase):
    def test_passing_evidence_uses_weighted_paired_deltas(self):
        decision = GroupedFusionGate().evaluate(_evidence())

        self.assertTrue(decision.passed)
        self.assertEqual(decision.blocking_reasons, ())
        self.assertEqual(decision.full_score_delta, 2.0)
        self.assertEqual(decision.repeat_positive_fold_counts, (3, 3, 3))
        expected = sum(
            delta * count
            for delta, count in zip(PASSING_DELTAS[0], RECORD_COUNTS)
        ) / sum(RECORD_COUNTS)
        self.assertAlmostEqual(decision.repeat_weighted_deltas[0], expected)
        self.assertTrue(
            all(value > 0 for value in decision.repeat_leave_best_fold_out_deltas)
        )
        self.assertFalse(decision.concentration_warning)

    def test_aggregate_is_public_exposed_identity_free_and_has_all_counters(self):
        aggregate = GroupedFusionGate().evaluate(
            _evidence()
        ).to_aggregate_evidence()

        require_aggregate_only(aggregate)
        self.assertEqual(
            aggregate["evaluation_mode"], PUBLIC_EXPOSED_SCOPE_LABEL
        )
        self.assertEqual(
            PUBLIC_EXPOSED_SCOPE_LABEL,
            "public_grouped_robustness_not_unseen",
        )
        self.assertEqual(aggregate["status"], "passed")
        self.assertEqual(aggregate["repeat_count"], 3)
        self.assertEqual(aggregate["fold_count"], 5)
        self.assertEqual(aggregate["evaluated_fold_count"], 15)
        self.assertEqual(len(aggregate["fold_weights"]), 15)
        self.assertEqual(aggregate["provenance_coverage_fraction"], 1.0)
        self.assertEqual(
            aggregate["new_catastrophic_false_approval_count"], 0
        )
        self.assertEqual(aggregate["new_false_positive_denial_count"], 0)
        self.assertEqual(
            aggregate["counts"],
            {
                "changed_field_count": 12,
                "changed_field_complete_provenance_count": 12,
                "clean_higher_authority_override_count": 0,
                "binding_authority_override_count": 0,
                "text_layer_winner_count": 0,
                "serialization_default_used_as_evidence_count": 0,
                "correlated_views_collapsed": 7,
                "independent_agreement_resolutions": 4,
                "same_rank_contested_count": 3,
                "cross_applicant_candidates_excluded": 2,
            },
        )
        self.assertNotIn("blocking_reasons", aggregate)

    def test_fold_input_order_does_not_change_output(self):
        evidence = _evidence()
        reversed_evidence = replace(evidence, folds=tuple(reversed(evidence.folds)))

        normal = GroupedFusionGate().evaluate(evidence)
        reversed_result = GroupedFusionGate().evaluate(reversed_evidence)

        self.assertEqual(normal, reversed_result)
        self.assertEqual(
            normal.to_aggregate_evidence(),
            reversed_result.to_aggregate_evidence(),
        )

    def test_single_positive_fold_per_repeat_fails_leave_best_gate(self):
        sparse = (
            (1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0, 0.0),
        )

        decision = GroupedFusionGate().evaluate(
            _evidence(folds=_folds(sparse))
        )

        self.assertFalse(decision.passed)
        self.assertEqual(
            decision.blocking_reasons, ("leave_best_fold_out_positive",)
        )
        self.assertFalse(decision.to_aggregate_evidence()["fold_consistent"])
        self.assertEqual(decision.repeat_positive_fold_counts, (1, 1, 1))
        self.assertEqual(
            decision.repeat_leave_best_fold_out_deltas, (0.0, 0.0, 0.0)
        )
        self.assertEqual(
            dict(decision.diagnostic_results),
            {
                "no_negative_folds": True,
                "leave_best_fold_out_nonnegative": True,
                "fold_majority_positive": False,
                "leave_best_fold_out_positive": False,
            },
        )
        self.assertTrue(decision.concentration_warning)

    def test_score_and_fold_acceptance_rules_are_hard(self):
        scenarios = {
            "full": (
                {"candidate_full": _run(total_score=130.0)},
                "full_score_positive",
            ),
            "repeat": (
                {
                    "folds": _folds(
                        (
                            (0.0, 0.0, 0.0, 0.0, 0.0),
                            PASSING_DELTAS[1],
                            PASSING_DELTAS[2],
                        )
                    )
                },
                "repeat_weighted_deltas_positive",
            ),
            "positive_fold": (
                {
                    "folds": _folds(
                        (
                            (0.0, 0.0, 0.0, 0.0, 0.0),
                            PASSING_DELTAS[1],
                            PASSING_DELTAS[2],
                        )
                    )
                },
                "at_least_one_positive_fold_per_repeat",
            ),
        }
        for name, (overrides, expected_gate) in scenarios.items():
            with self.subTest(name=name):
                decision = GroupedFusionGate().evaluate(
                    _evidence(**overrides)
                )
                self.assertFalse(decision.passed)
                self.assertIn(expected_gate, decision.blocking_reasons)

    def test_negative_folds_block_even_when_weighted_repeats_are_positive(self):
        mixed = (
            (1.0, 0.5, 0.2, -0.10, 0.1),
            (0.5, 1.0, -0.10, 0.2, 0.1),
            (0.2, -0.10, 1.0, 0.5, 0.1),
        )

        decision = GroupedFusionGate().evaluate(
            _evidence(folds=_folds(mixed))
        )

        self.assertFalse(decision.passed)
        self.assertEqual(decision.blocking_reasons, ("no_negative_folds",))
        self.assertFalse(decision.to_aggregate_evidence()["fold_consistent"])
        checks = dict(decision.diagnostic_results)
        self.assertFalse(checks["no_negative_folds"])
        self.assertTrue(checks["fold_majority_positive"])
        self.assertTrue(decision.concentration_warning)

    def test_completeness_safety_and_fusion_audit_rules_are_hard(self):
        scenarios = {
            "candidate_missing": (
                {"candidate_full": _run(total_score=132.0, missing_records=1)},
                "candidate_complete",
            ),
            "control_extra": (
                {"legacy_control_full": _run(extra_records=1)},
                "legacy_control_complete",
            ),
            "candidate_invalid": (
                {"candidate_full": _run(total_score=132.0, invalid_records=1)},
                "candidate_complete",
            ),
            "candidate_duplicate": (
                {"candidate_full": _run(total_score=132.0, duplicate_records=1)},
                "candidate_complete",
            ),
            "nondeterministic": (
                {"candidate_full": _run(total_score=132.0, deterministic=False)},
                "run_deterministic",
            ),
            "catastrophic_increase": (
                {
                    "legacy_control_full": _run(
                        catastrophic_false_approvals=1
                    ),
                    "candidate_full": _run(
                        total_score=132.0,
                        catastrophic_false_approvals=2,
                    ),
                },
                "no_increased_catastrophic_false_approvals",
            ),
            "denial_increase": (
                {"candidate_full": _run(total_score=132.0, false_positive_denials=1)},
                "no_increased_false_positive_denials",
            ),
            "new_catastrophic_swap": (
                {"new_catastrophic_false_approval_count": 1},
                "no_new_catastrophic_false_approvals",
            ),
            "new_denial_swap": (
                {"new_false_positive_denial_count": 1},
                "no_new_false_positive_denials",
            ),
            "vacuous_changes": (
                {
                    "candidate_fusion_audit": _audit(
                        changed_field_count=0,
                        changed_field_complete_provenance_count=0,
                    )
                },
                "changed_fields_nonvacuous",
            ),
            "incomplete_provenance": (
                {
                    "candidate_fusion_audit": _audit(
                        changed_field_complete_provenance_count=11
                    )
                },
                "changed_field_provenance_complete",
            ),
            "clean_override": (
                {
                    "candidate_fusion_audit": _audit(
                        clean_higher_authority_override_count=1
                    )
                },
                "no_clean_higher_authority_overrides",
            ),
            "binding_override": (
                {
                    "candidate_fusion_audit": _audit(
                        binding_authority_override_count=1
                    )
                },
                "no_binding_authority_overrides",
            ),
            "text_layer": (
                {"candidate_fusion_audit": _audit(text_layer_winner_count=1)},
                "no_text_layer_winners",
            ),
            "serialization_default": (
                {
                    "candidate_fusion_audit": _audit(
                        serialization_default_used_as_evidence_count=1
                    )
                },
                "no_serialization_default_as_evidence",
            ),
            "manifest": (
                {"manifest_frozen_before_scoring": False},
                "manifest_frozen_before_scoring",
            ),
            "exclusive": ({"group_exclusive": False}, "group_exclusive"),
            "paired": ({"paired_fold_members": False}, "paired_fold_members"),
            "split": ({"split_deterministic": False}, "split_deterministic"),
        }
        for name, (overrides, expected_gate) in scenarios.items():
            with self.subTest(name=name):
                decision = GroupedFusionGate().evaluate(
                    _evidence(**overrides)
                )
                self.assertFalse(decision.passed)
                self.assertIn(expected_gate, decision.blocking_reasons)

    def test_nonzero_control_safety_counts_may_be_preserved_not_increased(self):
        decision = GroupedFusionGate().evaluate(
            _evidence(
                legacy_control_full=_run(
                    catastrophic_false_approvals=2,
                    false_positive_denials=3,
                ),
                candidate_full=_run(
                    total_score=132.0,
                    catastrophic_false_approvals=2,
                    false_positive_denials=3,
                ),
            )
        )

        self.assertTrue(decision.passed)

    def test_malformed_evidence_fails_closed(self):
        with self.assertRaises(GroupedFusionEvidenceError):
            _audit(changed_field_count=1, changed_field_complete_provenance_count=2)
        with self.assertRaises(GroupedFusionEvidenceError):
            _evidence(layout_manifest_sha256="short")
        with self.assertRaises(GroupedFusionEvidenceError):
            _run(total_score=math.nan)
        with self.assertRaises(GroupedFusionEvidenceError):
            FoldScorePair(
                repeat=0,
                fold=0,
                control_score=1,
                candidate_score=2,
                control_record_count=2,
                candidate_record_count=1,
                layout_group_count=1,
            )

        missing = _evidence(folds=_folds()[:-1])
        with self.assertRaisesRegex(
            GroupedFusionEvidenceError, "exactly three repeats"
        ):
            GroupedFusionGate().evaluate(missing)

    def test_committed_wo16_strict_reclassification_is_exact(self):
        root = Path(__file__).resolve().parents[1]
        reclassification = json.loads(
            (
                root
                / "evaluation/WO16_STRICT_GATE_RECLASSIFICATION.json"
            ).read_text(encoding="utf-8")
        )
        source_binding = reclassification["source_evidence"]
        source_path = root / source_binding["path"]
        source_bytes = source_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(source_bytes).hexdigest(),
            source_binding["sha256"],
        )
        gate_binding = reclassification["strict_gate_source"]
        gate_bytes = (root / gate_binding["path"]).read_bytes()
        self.assertEqual(
            hashlib.sha256(gate_bytes).hexdigest(),
            gate_binding["sha256"],
        )

        source = json.loads(source_bytes)
        fold_deltas = source["fold_deltas"]
        observed = reclassification["observed"]
        self.assertEqual(observed["evaluated_fold_count"], len(fold_deltas))
        self.assertEqual(
            observed["negative_fold_count"],
            sum(delta < 0 for delta in fold_deltas),
        )
        self.assertEqual(
            observed["positive_fold_count"],
            sum(delta > 0 for delta in fold_deltas),
        )
        self.assertEqual(
            observed["zero_fold_count"],
            sum(delta == 0 for delta in fold_deltas),
        )
        self.assertEqual(
            reclassification["classification"],
            "historical_candidate_rejected_negative_folds",
        )
        self.assertEqual(
            reclassification["outcome"]["blocking_reasons"],
            ["no_negative_folds"],
        )
        self.assertEqual(reclassification["outcome"]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
