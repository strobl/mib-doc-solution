from __future__ import annotations

import json
import math
import unittest
from copy import deepcopy

from mib_pipeline.confidence_refit import (
    CALIBRATION_FAMILIES,
    CONFIDENCE_FEATURE_ORDER,
    CalibrationExample,
    ConfidenceRefitError,
    FittedConfidenceCalibrator,
    fit_confidence_calibrator,
    identity_free_feature_vector,
)
from mib_pipeline.final_confidence import FinalConfidenceContext


def context(
    *,
    final_class: str = "NEEDS_REVIEW",
    policy_route: str = "deterministic_policy",
    recovery_route: str = "primary",
    conflict: bool = False,
    model: bool = False,
) -> FinalConfidenceContext:
    return FinalConfidenceContext(
        schema_version=1,
        final_class=final_class,
        policy_route=policy_route,
        authoritative=policy_route == "binding_authority",
        visible_completeness=0.8,
        has_conflict=conflict,
        ocr_disagreement=0.1,
        recovery_route=recovery_route,
        model_margin=0.4 if model else None,
        ensemble_agreement=0.8 if model else None,
        resolution_entropy=0.2,
    )


class ConfidenceRefitTests(unittest.TestCase):
    def examples(self) -> list[CalibrationExample]:
        return [
            CalibrationExample(0.10, False, context(final_class="APPROVED")),
            CalibrationExample(0.20, False, context(final_class="DENIED")),
            CalibrationExample(0.35, True, context(conflict=True)),
            CalibrationExample(0.55, True, context(recovery_route="late_visible")),
            CalibrationExample(0.75, True, context(model=True)),
            CalibrationExample(0.95, True, context(policy_route="revalidated_policy")),
        ]

    def test_feature_vector_is_fixed_finite_and_identity_free(self) -> None:
        vector = identity_free_feature_vector(0.7, context(model=True))
        self.assertEqual(len(vector), len(CONFIDENCE_FEATURE_ORDER))
        self.assertTrue(all(math.isfinite(value) for value in vector))
        self.assertNotIn("case", " ".join(CONFIDENCE_FEATURE_ORDER).casefold())

    def test_all_families_fit_predict_and_round_trip(self) -> None:
        for family in CALIBRATION_FAMILIES:
            with self.subTest(family=family):
                fitted = fit_confidence_calibrator(family, self.examples())
                mapping = fitted.to_runtime_mapping()
                encoded = json.dumps(mapping, sort_keys=True)
                self.assertNotIn("MIB-", encoded)
                self.assertNotIn(".pdf", encoded.casefold())
                loaded = FittedConfidenceCalibrator.from_runtime_mapping(mapping)
                first = loaded.predict(0.42, context(model=True))
                second = loaded.predict(0.42, context(model=True))
                self.assertEqual(first, second)
                self.assertTrue(0.0 < first < 1.0)

    def test_empty_and_single_class_folds_have_finite_fallbacks(self) -> None:
        single_class = [
            CalibrationExample(0.1, True, context()),
            CalibrationExample(0.9, True, context()),
        ]
        for family in CALIBRATION_FAMILIES:
            with self.subTest(family=family, shape="empty"):
                fitted = fit_confidence_calibrator(family, [])
                self.assertTrue(math.isfinite(fitted.predict(0.0, context())))
                self.assertTrue(math.isfinite(fitted.predict(1.0, context())))
            with self.subTest(family=family, shape="single"):
                fitted = fit_confidence_calibrator(family, single_class)
                self.assertTrue(math.isfinite(fitted.predict(0.5, context())))

    def test_fit_is_invariant_to_training_record_order(self) -> None:
        examples = self.examples()
        for family in CALIBRATION_FAMILIES:
            with self.subTest(family=family):
                forward = fit_confidence_calibrator(
                    family,
                    examples,
                ).to_runtime_mapping()
                reverse = fit_confidence_calibrator(
                    family,
                    reversed(examples),
                ).to_runtime_mapping()
                self.assertEqual(forward, reverse)

    def test_artifact_rejects_identity_bearing_parameter(self) -> None:
        with self.assertRaises(ConfidenceRefitError):
            FittedConfidenceCalibrator(
                family="temperature",
                parameters={"temperature": 1.0, "note": "MIB-000001.pdf"},
            )

    def test_beta_backtracking_handles_adversarial_identical_inputs(self) -> None:
        adversarial = [
            CalibrationExample(0.01, False, context()),
            CalibrationExample(0.01, False, context()),
            CalibrationExample(0.01, True, context()),
        ]
        fitted = fit_confidence_calibrator("beta", adversarial)
        probability = fitted.predict(0.01, context())
        self.assertAlmostEqual(probability, 1.0 / 3.0, delta=0.02)
        self.assertTrue(
            all(
                math.isfinite(value)
                for value in fitted.parameters["coefficients"]
            )
        )

    def test_numeric_hyperparameters_are_exact_and_not_coerced(self) -> None:
        beta = fit_confidence_calibrator("beta", self.examples()).to_runtime_mapping()
        beta["parameters"]["l2_strength"] = "1.0"
        with self.assertRaises(ConfidenceRefitError):
            FittedConfidenceCalibrator.from_runtime_mapping(beta)

        hierarchical = fit_confidence_calibrator(
            "hierarchical_shrunk",
            self.examples(),
        ).to_runtime_mapping()
        hierarchical["parameters"]["shrinkage"] = "4.0"
        with self.assertRaises(ConfidenceRefitError):
            FittedConfidenceCalibrator.from_runtime_mapping(hierarchical)

    def test_isotonic_thresholds_must_be_probabilities(self) -> None:
        fitted = fit_confidence_calibrator(
            "isotonic",
            self.examples(),
        ).to_runtime_mapping()
        fitted["parameters"]["thresholds"][0] = -0.01
        with self.assertRaises(ConfidenceRefitError):
            FittedConfidenceCalibrator.from_runtime_mapping(fitted)

    def test_hierarchical_tokens_are_closed_and_have_exact_ancestry(self) -> None:
        fitted = fit_confidence_calibrator(
            "hierarchical_shrunk",
            [CalibrationExample(0.5, True, context())],
        ).to_runtime_mapping()

        free_form = deepcopy(fitted)
        free_form["parameters"]["class_nodes"][0]["tokens"] = [
            "applicant_alpha"
        ]
        with self.assertRaises(ConfidenceRefitError):
            FittedConfidenceCalibrator.from_runtime_mapping(free_form)

        wrong_depth = deepcopy(fitted)
        wrong_depth["parameters"]["recovery_nodes"][0]["tokens"] = [
            "NEEDS_REVIEW",
            "deterministic_policy",
        ]
        with self.assertRaises(ConfidenceRefitError):
            FittedConfidenceCalibrator.from_runtime_mapping(wrong_depth)

        orphan = deepcopy(fitted)
        orphan["parameters"]["policy_nodes"][0]["tokens"] = [
            "DENIED",
            "deterministic_policy",
        ]
        with self.assertRaises(ConfidenceRefitError):
            FittedConfidenceCalibrator.from_runtime_mapping(orphan)

    def test_isotonic_output_is_monotone(self) -> None:
        fitted = fit_confidence_calibrator("isotonic", self.examples())
        values = [
            fitted.predict(index / 20.0, context()) for index in range(21)
        ]
        self.assertEqual(values, sorted(values))


if __name__ == "__main__":
    unittest.main()
