from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from devtools.confidence_refit_cv import (
    FOLD_COUNT,
    MINIMUM_GROUP_COUNT,
    MINIMUM_SLICE_SUPPORT,
    REPEAT_SEEDS,
    SELECTION_TIE_BREAK,
    TRACKED_BRIER_TARGET,
    ConfidenceCVError,
    GroupedCalibrationExample,
    compare_confidence_families,
    grouped_folds,
    load_grouped_examples,
)
from mib_pipeline.confidence_refit import CALIBRATION_FAMILIES, CalibrationExample
from mib_pipeline.final_confidence import FinalConfidenceContext


def make_context(index: int) -> FinalConfidenceContext:
    classes = ("APPROVED", "DENIED", "NEEDS_REVIEW")
    policies = (
        "deterministic_policy",
        "revalidated_policy",
        "visible_policy_violation",
    )
    recoveries = ("primary", "late_visible", "rapid_visible")
    return FinalConfidenceContext(
        schema_version=1,
        final_class=classes[index % len(classes)],
        policy_route=policies[index % len(policies)],
        authoritative=False,
        visible_completeness=0.5 + 0.1 * (index % 5),
        has_conflict=index % 7 == 0,
        ocr_disagreement=0.05 * (index % 5),
        recovery_route=recoveries[index % len(recoveries)],
        model_margin=None,
        ensemble_agreement=None,
        resolution_entropy=0.1 * (index % 5),
    )


def examples() -> tuple[GroupedCalibrationExample, ...]:
    return tuple(
        GroupedCalibrationExample(
            sample_key=f"sample-{index:02d}",
            group_key=f"group-{index % 9}",
            calibration=CalibrationExample(
                input_confidence=0.1 + 0.8 * (index % 10) / 9.0,
                correct=index % 6 != 0,
                context=make_context(index),
            ),
        )
        for index in range(32)
    )


class ConfidenceRefitCVTests(unittest.TestCase):
    def test_grouped_folds_are_exclusive_complete_and_deterministic(self) -> None:
        values = examples()
        first = grouped_folds(values, seed=REPEAT_SEEDS[0])
        second = grouped_folds(values, seed=REPEAT_SEEDS[0])
        self.assertEqual(first, second)
        self.assertEqual(len(first), FOLD_COUNT)
        self.assertEqual(
            sorted(index for fold in first for index in fold),
            list(range(len(values))),
        )
        for fold in first:
            self.assertTrue(fold)
            self.assertLess(len(fold), len(values))
            test_groups = {values[index].group_key for index in fold}
            train_groups = {
                value.group_key
                for index, value in enumerate(values)
                if index not in set(fold)
            }
            self.assertFalse(test_groups & train_groups)

    def test_comparison_is_exactly_reproducible_and_aggregate_only(self) -> None:
        first_report, first_artifacts = compare_confidence_families(examples())
        second_report, second_artifacts = compare_confidence_families(examples())
        self.assertEqual(first_report, second_report)
        self.assertEqual(
            {
                family: artifact.to_runtime_mapping()
                for family, artifact in first_artifacts.items()
            },
            {
                family: artifact.to_runtime_mapping()
                for family, artifact in second_artifacts.items()
            },
        )
        self.assertEqual(set(first_report["candidates"]), set(CALIBRATION_FAMILIES))
        for candidate in first_report["candidates"].values():
            self.assertEqual(len(candidate["repeats"]), 3)
            self.assertEqual(len(candidate["folds"]), 15)
            self.assertTrue(
                all(fold["group_overlap_count"] == 0 for fold in candidate["folds"])
            )
        encoded_report = json.dumps(first_report, sort_keys=True)
        self.assertNotIn("sample-", encoded_report)
        self.assertNotIn("group-", encoded_report)
        self.assertNotIn("\"labels\"", encoded_report)
        self.assertNotIn("\"predictions\"", encoded_report)

        selection = first_report["selection"]
        self.assertEqual(
            selection["tie_break_order"],
            list(SELECTION_TIE_BREAK),
        )
        self.assertEqual(
            selection["promotion_recommended"],
            all(selection["promotion_checks"].values()),
        )
        self.assertEqual(len(selection["repeat_brier_deltas"]), 3)
        tracked_target = selection["tracked_brier_target"]
        self.assertEqual(tracked_target["threshold"], TRACKED_BRIER_TARGET)
        self.assertFalse(tracked_target["met"])
        self.assertFalse(tracked_target["promotion_gate"])
        self.assertNotIn("tracked_brier_target", selection["promotion_checks"])
        for candidate in first_report["candidates"].values():
            logo = candidate["leave_one_group_out"]
            self.assertEqual(logo["holdout_count"], 9)
            self.assertEqual(
                logo["non_regression"],
                logo["regression_count"] == 0,
            )
            metrics = candidate["aggregate"]["overall"]
            self.assertIn("ece_10_bin", metrics)
            self.assertEqual(len(metrics["reliability_bins_10"]), 10)
            self.assertEqual(metrics["unique_support"], 32)
            for distribution in candidate["aggregate"][
                "confidence_distribution"
            ].values():
                self.assertIn("standard_deviation", distribution)
                self.assertIn("p10", distribution)
                self.assertIn("p90", distribution)

    def test_tracked_brier_target_can_be_met_without_forcing_promotion(self) -> None:
        perfect = tuple(
            GroupedCalibrationExample(
                sample_key=f"perfect-{index}",
                group_key=f"perfect-group-{index}",
                calibration=CalibrationExample(
                    input_confidence=1.0,
                    correct=True,
                    context=make_context(index),
                ),
            )
            for index in range(10)
        )
        report, _ = compare_confidence_families(perfect)
        selection = report["selection"]
        tracked_target = selection["tracked_brier_target"]
        self.assertTrue(tracked_target["met"])
        self.assertLessEqual(
            tracked_target["selected_mean_brier"],
            TRACKED_BRIER_TARGET,
        )
        self.assertFalse(tracked_target["promotion_gate"])
        self.assertFalse(selection["promotion_recommended"])
        self.assertFalse(
            selection["promotion_checks"]["mean_brier_strictly_improved"]
        )

    def test_sparse_slice_support_counts_unique_samples_not_repeats(self) -> None:
        values = list(examples())
        for index in range(2):
            original = values[index]
            original_context = original.calibration.context
            binding_context = FinalConfidenceContext(
                **{
                    **original_context.to_dict(),
                    "policy_route": "binding_authority",
                    "authoritative": True,
                }
            )
            values[index] = replace(
                original,
                calibration=CalibrationExample(
                    original.calibration.input_confidence,
                    original.calibration.correct,
                    binding_context,
                ),
            )
        report, _ = compare_confidence_families(values)
        self.assertEqual(MINIMUM_SLICE_SUPPORT, 5)
        for candidate in [
            report["baseline"],
            *report["candidates"].values(),
        ]:
            policy_slices = candidate["aggregate"]["slices"]["policy_route"]
            self.assertNotIn("binding_authority", policy_slices)
            sparse = policy_slices["OTHER_SPARSE"]
            self.assertEqual(sparse["unique_support"], 2)
            self.assertEqual(sparse["support"], 6)

    def test_fewer_than_five_groups_is_rejected(self) -> None:
        insufficient = tuple(
            example
            for example in examples()
            if example.group_key in {"group-0", "group-1", "group-2", "group-3"}
        )
        self.assertEqual(
            len({example.group_key for example in insufficient}),
            MINIMUM_GROUP_COUNT - 1,
        )
        with self.assertRaises(ConfidenceCVError):
            grouped_folds(insufficient, seed=REPEAT_SEEDS[0])
        with self.assertRaises(ConfidenceCVError):
            compare_confidence_families(insufficient)

    def test_loader_requires_exact_schema(self) -> None:
        sample = examples()[0]
        payload = {
            "schema_version": "mib-confidence-refit-samples/v1",
            "samples": [
                {
                    "sample_key": sample.sample_key,
                    "group_key": sample.group_key,
                    "input_confidence": sample.calibration.input_confidence,
                    "correct": sample.calibration.correct,
                    "context": sample.calibration.context.to_dict(),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_grouped_examples(path)
        self.assertEqual(loaded, (sample,))


if __name__ == "__main__":
    unittest.main()
