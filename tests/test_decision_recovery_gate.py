from __future__ import annotations

import unittest
from dataclasses import replace

from devtools.decision_recovery_gate import (
    APPROACHES,
    PROTECTED_ROLES,
    ApproachRunAggregate,
    DecisionRecoveryContractAudit,
    DecisionRecoveryEvidence,
    DecisionRecoveryEvidenceError,
    DecisionRecoveryGate,
    ProtectedRoleScore,
    ScoreSlice,
)
from devtools.experiment_control import canonical_json, require_aggregate_only


def _scores(delta: float = 1.0) -> dict[str, float]:
    return {
        "deterministic_engine": 100.0,
        "evidence_completion_only": 100.2,
        "compact_identity_free_model": 100.4,
        "gated_hybrid": 100.0 + delta,
    }


def _run(score: float, *, deterministic: bool = True) -> ApproachRunAggregate:
    return ApproachRunAggregate(
        total_score=score,
        classification_score=70.0,
        record_count=32,
        catastrophic_false_approvals=0,
        false_positive_denials=0,
        missing_records=0,
        invalid_records=0,
        duplicate_records=0,
        extra_records=0,
        deterministic=deterministic,
    )


def _counts(**changes: int) -> dict[str, int]:
    values = {
        name: 1
        for name in DecisionRecoveryContractAudit.REQUIRED_POSITIVE_COUNTS
    }
    values.update(
        {
            name: 0
            for name in DecisionRecoveryContractAudit.REQUIRED_ZERO_COUNTS
        }
    )
    values.update(changes)
    return values


def _evidence() -> DecisionRecoveryEvidence:
    return DecisionRecoveryEvidence(
        source_revision_sha="a" * 40,
        layout_manifest_sha256="b" * 64,
        protected_role_manifest_sha256="c" * 64,
        input_tree_sha256="d" * 64,
        cohort_set_sha256="3" * 64,
        truth_sha256="e" * 64,
        official_evaluator_sha256="f" * 64,
        feature_schema_sha256="1" * 64,
        artifact_set_sha256="2" * 64,
        expected_record_count=32,
        expected_layout_group_count=9,
        full_runs={
            approach: _run(100.0 + index * 0.25)
            for index, approach in enumerate(APPROACHES)
        },
        repeat_scores=tuple(
            ScoreSlice(
                repeat=repeat,
                fold=None,
                record_count=32,
                layout_group_count=9,
                scores=_scores(),
            )
            for repeat in range(3)
        ),
        fold_scores=tuple(
            ScoreSlice(
                repeat=repeat,
                fold=fold,
                record_count=(7 if fold < 2 else 6),
                layout_group_count=(1 if fold == 4 else 2),
                scores=_scores(),
            )
            for repeat in range(3)
            for fold in range(5)
        ),
        protected_role_scores=tuple(
            ProtectedRoleScore(
                role=role,
                repeat=repeat,
                record_count=6,
                layout_group_count=3,
                scores=_scores(),
            )
            for role in PROTECTED_ROLES
            for repeat in range(3)
        ),
        contract_audit=DecisionRecoveryContractAudit(
            counts=_counts(),
            leakage_clean=True,
            feature_schema_exact=True,
            deterministic=True,
        ),
        new_catastrophic_false_approval_count=0,
        new_false_positive_denial_count=0,
        manifest_frozen_before_scoring=True,
        roles_frozen_before_scoring=True,
        group_exclusive=True,
        paired_fold_members=True,
    )


class DecisionRecoveryGateTests(unittest.TestCase):
    def test_passing_evidence_is_aggregate_only_and_compares_four_arms(self):
        decision = DecisionRecoveryGate().evaluate(_evidence())

        self.assertTrue(decision.passed)
        aggregate = decision.to_aggregate_evidence()
        self.assertEqual(aggregate["approach_count"], 4)
        self.assertEqual(aggregate["fold_count"], 15)
        self.assertEqual(aggregate["folds_per_repeat_count"], 5)
        self.assertTrue(aggregate["checks"]["positive_every_fold"])
        self.assertEqual(set(aggregate["class_metrics"]), set(APPROACHES))
        require_aggregate_only(aggregate)
        rendered = canonical_json(aggregate)
        for forbidden in ("MIB-", ".pdf", "Applicant", "case_id"):
            self.assertNotIn(forbidden, rendered)

    def test_one_nonpositive_fold_blocks_promotion(self):
        evidence = _evidence()
        folds = list(evidence.fold_scores)
        folds[7] = replace(folds[7], scores=_scores(delta=0.0))

        decision = DecisionRecoveryGate().evaluate(
            replace(evidence, fold_scores=tuple(folds))
        )

        self.assertFalse(decision.passed)
        self.assertIn(
            "positive_gain_every_fold", decision.blocking_reasons
        )

    def test_one_nonpositive_role_repeat_blocks_promotion(self):
        evidence = _evidence()
        roles = list(evidence.protected_role_scores)
        roles[-1] = replace(roles[-1], scores=_scores(delta=-0.1))

        decision = DecisionRecoveryGate().evaluate(
            replace(evidence, protected_role_scores=tuple(roles))
        )

        self.assertFalse(decision.passed)
        self.assertIn(
            "positive_gain_every_protected_role",
            decision.blocking_reasons,
        )

    def test_missingness_only_denial_is_an_explicit_hard_failure(self):
        evidence = replace(
            _evidence(),
            contract_audit=DecisionRecoveryContractAudit(
                counts=_counts(denial_without_visible_violation_count=1),
                leakage_clean=True,
                feature_schema_exact=True,
                deterministic=True,
            ),
        )

        decision = DecisionRecoveryGate().evaluate(evidence)

        self.assertFalse(decision.passed)
        self.assertIn(
            "contract_unsafe_counts_zero", decision.blocking_reasons
        )

    def test_every_arm_requires_byte_determinism(self):
        evidence = _evidence()
        runs = dict(evidence.full_runs)
        runs["compact_identity_free_model"] = replace(
            runs["compact_identity_free_model"], deterministic=False
        )

        decision = DecisionRecoveryGate().evaluate(
            replace(evidence, full_runs=runs)
        )

        self.assertFalse(decision.passed)
        self.assertIn(
            "all_four_approaches_complete", decision.blocking_reasons
        )

    def test_fold_totals_must_cover_cohort_and_groups_once_per_repeat(self):
        evidence = _evidence()
        folds = list(evidence.fold_scores)
        folds[0] = replace(folds[0], record_count=6)

        decision = DecisionRecoveryGate().evaluate(
            replace(evidence, fold_scores=tuple(folds))
        )

        self.assertFalse(decision.passed)
        self.assertIn(
            "grouped_split_integrity", decision.blocking_reasons
        )

    def test_probability_margin_and_disagreement_contract_is_nonvacuous(self):
        for counter in (
            "probability_simplex_probe_count",
            "true_margin_probe_count",
            "ensemble_disagreement_probe_count",
        ):
            with self.subTest(counter=counter):
                evidence = replace(
                    _evidence(),
                    contract_audit=DecisionRecoveryContractAudit(
                        counts=_counts(**{counter: 0}),
                        leakage_clean=True,
                        feature_schema_exact=True,
                        deterministic=True,
                    ),
                )
                decision = DecisionRecoveryGate().evaluate(evidence)
                self.assertFalse(decision.passed)
                self.assertIn(
                    "contract_probes_exercised",
                    decision.blocking_reasons,
                )

    def test_scores_must_be_nonnegative(self):
        with self.assertRaises(DecisionRecoveryEvidenceError):
            _run(-0.01)


if __name__ == "__main__":
    unittest.main()
