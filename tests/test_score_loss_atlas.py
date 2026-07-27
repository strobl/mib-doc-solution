import copy
import json
import re
import unittest

from scripts import evaluate
from scripts.score_loss_atlas import (
    AtlasInputError,
    FIELD_NAMES,
    build_atlas,
    render_markdown,
)


def truth_row(
    case_id,
    applicant_name,
    *,
    adjudication,
    risk_flags="none",
):
    return {
        "case_id": case_id,
        "applicant_name": applicant_name,
        "species_code": "ORION_GRAYS",
        "home_world": "Kepler-186f",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-1234",
        "arrival_date": "2026-04-17",
        "declared_purpose": "research",
        "risk_flags": risk_flags,
        "fee_status": "paid",
        "adjudication": adjudication,
    }


def prediction_row(
    case_id,
    applicant_name,
    *,
    adjudication,
    confidence,
    risk_flags="none",
):
    row = truth_row(
        case_id,
        applicant_name,
        adjudication=adjudication,
        risk_flags=risk_flags,
    )
    row["confidence"] = confidence
    return row


def evaluated_inputs(truth_rows, prediction_rows):
    evaluation, case_scores = evaluate.build_results(
        {row["case_id"]: row for row in truth_rows},
        prediction_rows,
    )
    return truth_rows, prediction_rows, evaluation, case_scores


def atlas_inputs():
    truth_rows = [
        truth_row(
            "MIB-123456",
            "Private Alpha",
            adjudication="APPROVED",
        ),
        truth_row(
            "MIB-654321",
            "Private Beta",
            adjudication="DENIED",
        ),
    ]
    prediction_rows = [
        prediction_row(
            "MIB-123456",
            "unknown",
            adjudication="NEEDS_REVIEW",
            confidence=0.25,
        ),
        prediction_row(
            "MIB-654321",
            "Private Beta",
            adjudication="DENIED",
            confidence=0.75,
        ),
    ]
    return evaluated_inputs(truth_rows, prediction_rows)


def extended_atlas_inputs():
    truth_rows = [
        truth_row(
            "MIB-100001",
            "Private One",
            adjudication="APPROVED",
            risk_flags="sponsor_mismatch",
        ),
        truth_row(
            "MIB-100002",
            "Private Two",
            adjudication="DENIED",
        ),
        truth_row(
            "MIB-100003",
            "Private Three",
            adjudication="APPROVED",
        ),
        truth_row(
            "MIB-100004",
            "Private Four",
            adjudication="APPROVED",
        ),
    ]
    prediction_rows = [
        prediction_row(
            "MIB-100001",
            "unknown",
            adjudication="NEEDS_REVIEW",
            confidence=0.25,
            risk_flags="none",
        ),
        prediction_row(
            "MIB-100002",
            "Private Two",
            adjudication="DENIED",
            confidence=0.75,
        ),
        prediction_row(
            "MIB-100003",
            "unknown",
            adjudication="APPROVED",
            confidence=0.80,
        ),
        prediction_row(
            "MIB-100004",
            "Private Four",
            adjudication="DENIED",
            confidence=0.20,
        ),
    ]
    return evaluated_inputs(truth_rows, prediction_rows)


class ScoreLossAtlasTests(unittest.TestCase):
    def build(self, *, target_score=148.0):
        truth, predictions, evaluation, case_scores = atlas_inputs()
        return build_atlas(
            truth_rows=truth,
            prediction_rows=predictions,
            evaluation=evaluation,
            case_scores=case_scores,
            target_score=target_score,
        )

    def test_aggregate_field_classification_and_calibration_losses_balance(self):
        atlas = self.build()
        applicant = atlas["field_losses"]["applicant_name"]
        approved_review = atlas["confusion_losses"]["APPROVED->NEEDS_REVIEW"]
        denied_denied = atlas["confusion_losses"]["DENIED->DENIED"]

        self.assertEqual(applicant["missed"], 1)
        self.assertEqual(applicant["raw_weighted_loss"], 5.0)
        self.assertAlmostEqual(
            applicant["normalized_score_loss"],
            50.0 * 5.0 / 90.0,
        )
        self.assertEqual(applicant["default_on_missed_cases"], 1)
        self.assertEqual(
            applicant["modal_wrong_output"]["value"],
            "<configured_default>",
        )
        self.assertEqual(atlas["field_priority"][0]["field"], "applicant_name")

        self.assertEqual(approved_review["cases"], 1)
        self.assertEqual(approved_review["classification_raw_loss"], 6.0)
        self.assertAlmostEqual(
            approved_review["classification_score_loss"],
            80.0 * 6.0 / 16.0,
        )
        self.assertEqual(approved_review["all_fields_correct_cases"], 0)
        self.assertAlmostEqual(
            approved_review["calibration_score_loss_contribution"],
            2.0 * 20.0 * 0.0625 / 2.0,
        )
        self.assertEqual(denied_denied["all_fields_correct_cases"], 1)

        self.assertAlmostEqual(
            sum(
                field["normalized_score_loss"]
                for field in atlas["field_losses"].values()
            ),
            atlas["score_gaps"]["extraction"],
        )
        self.assertAlmostEqual(
            sum(
                group["classification_score_loss"]
                for group in atlas["confusion_losses"].values()
            ),
            atlas["score_gaps"]["classification"],
        )
        self.assertAlmostEqual(
            sum(
                group["calibration_score_loss_contribution"]
                for group in atlas["confusion_losses"].values()
            ),
            atlas["score_gaps"]["calibration"],
        )

    def test_oracle_ceilings_are_additive_section_bounds(self):
        atlas = self.build()
        scores = atlas["current_scores"]
        gaps = atlas["score_gaps"]
        ceilings = atlas["oracle_ceilings"]

        self.assertAlmostEqual(
            ceilings["perfect_extraction_only"],
            scores["total"] + gaps["extraction"],
        )
        self.assertAlmostEqual(
            ceilings["perfect_classification_only"],
            scores["total"] + gaps["classification"],
        )
        self.assertAlmostEqual(
            ceilings["perfect_calibration_only"],
            scores["total"] + gaps["calibration"],
        )
        self.assertAlmostEqual(
            ceilings["perfect_extraction_and_classification"],
            scores["total"] + gaps["extraction"] + gaps["classification"],
        )
        self.assertEqual(ceilings["perfect_all_components"], 150.0)

    def test_target_recovery_fraction_uses_only_remaining_error(self):
        atlas = self.build(target_score=148.0)
        total = atlas["current_scores"]["total"]
        expected = (148.0 - total) / (150.0 - total)

        self.assertAlmostEqual(
            atlas["score_gaps"]["gain_required_for_target"],
            148.0 - total,
        )
        self.assertAlmostEqual(
            atlas["score_gaps"]["remaining_error_recovery_required"],
            expected,
        )

    def test_confusion_calibration_contributions_respect_score_floor(self):
        truth, predictions, evaluation, case_scores = atlas_inputs()
        predictions[0]["confidence"] = 0.9
        predictions[1]["confidence"] = 0.1
        evaluation, case_scores = evaluate.build_results(
            {row["case_id"]: row for row in truth},
            predictions,
        )

        atlas = build_atlas(
            truth_rows=truth,
            prediction_rows=predictions,
            evaluation=evaluation,
            case_scores=case_scores,
        )

        self.assertEqual(evaluation["scores"]["calibration_score"], 0.0)
        self.assertAlmostEqual(
            sum(
                group["calibration_score_loss_contribution"]
                for group in atlas["confusion_losses"].values()
            ),
            20.0,
        )

    def test_field_miss_pairs_and_decoupling_are_computed(self):
        truth, predictions, evaluation, case_scores = extended_atlas_inputs()
        atlas = build_atlas(
            truth_rows=truth,
            prediction_rows=predictions,
            evaluation=evaluation,
            case_scores=case_scores,
        )

        summary = atlas["field_miss_case_summary"]
        self.assertEqual(summary["total_field_misses"], 3)
        self.assertEqual(summary["cases_with_zero_field_misses"], 2)
        self.assertEqual(summary["cases_with_one_field_miss"], 1)
        self.assertEqual(summary["cases_with_multiple_field_misses"], 1)

        pair = next(
            item
            for item in atlas["pairwise_field_miss_diagnostics"]
            if {item["field_a"], item["field_b"]}
            == {"applicant_name", "risk_flags"}
        )
        self.assertEqual(pair["eligible_cases"], 4)
        self.assertEqual(pair["co_missed_cases"], 1)
        self.assertEqual(pair["expected_co_misses_under_independence"], 0.5)
        self.assertEqual(pair["enrichment_lift"], 2.0)

        decoupling = atlas["decision_field_decoupling"]
        self.assertEqual(decoupling["correct_decision_all_fields_correct"], 1)
        self.assertEqual(
            decoupling["correct_decision_one_or_more_field_misses"],
            1,
        )
        self.assertEqual(decoupling["wrong_decision_all_fields_correct"], 1)
        self.assertEqual(
            decoupling["wrong_decision_one_or_more_field_misses"],
            1,
        )

    def test_empirical_output_confidence_ceiling_is_aggregate_only(self):
        truth, predictions, evaluation, case_scores = extended_atlas_inputs()
        atlas = build_atlas(
            truth_rows=truth,
            prediction_rows=predictions,
            evaluation=evaluation,
            case_scores=case_scores,
        )
        ceiling = atlas["output_confidence_empirical_calibration_ceiling"]

        self.assertEqual(ceiling["group_count"], 4)
        self.assertEqual(ceiling["support_histogram"], {"1": 4})
        self.assertEqual(ceiling["empirical_oracle_mean_brier"], 0.0)
        self.assertEqual(ceiling["empirical_oracle_calibration_score"], 20.0)
        self.assertFalse(ceiling["exact_confidence_values_emitted"])
        self.assertNotIn("groups", ceiling)

    def test_source_hashes_are_emitted_and_cli_hashes_are_accepted(self):
        atlas = self.build()
        self.assertEqual(atlas["source_hash_basis"], "canonical_json")
        self.assertEqual(
            set(atlas["source_sha256"]),
            {"truth", "submission", "evaluation", "case_scores"},
        )
        for digest in atlas["source_sha256"].values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")

        truth, predictions, evaluation, case_scores = atlas_inputs()
        explicit = {name: character * 64 for name, character in {
            "truth": "a",
            "submission": "b",
            "evaluation": "c",
            "case_scores": "d",
        }.items()}
        explicit_atlas = build_atlas(
            truth_rows=truth,
            prediction_rows=predictions,
            evaluation=evaluation,
            case_scores=case_scores,
            source_sha256=explicit,
        )
        self.assertEqual(explicit_atlas["source_hash_basis"], "file_bytes")
        self.assertEqual(explicit_atlas["source_sha256"], explicit)

    def test_rejects_score_version_scale_and_count_tampering(self):
        for label, mutate in (
            (
                "version",
                lambda evaluation: evaluation.__setitem__(
                    "score_version", "mib_weighted_v2"
                ),
            ),
            (
                "scale",
                lambda evaluation: evaluation["score_scale"].__setitem__(
                    "classification_points", 81.0
                ),
            ),
            (
                "count",
                lambda evaluation: evaluation["counts"].__setitem__(
                    "truth_cases", 3
                ),
            ),
        ):
            with self.subTest(label=label):
                truth, predictions, evaluation, case_scores = atlas_inputs()
                mutate(evaluation)
                with self.assertRaises(AtlasInputError):
                    build_atlas(
                        truth_rows=truth,
                        prediction_rows=predictions,
                        evaluation=evaluation,
                        case_scores=case_scores,
                    )

    def test_rejects_adjudication_confidence_and_field_tampering(self):
        for label, mutate in (
            (
                "truth adjudication",
                lambda scores: scores[0].__setitem__(
                    "truth_adjudication", "DENIED"
                ),
            ),
            (
                "pred adjudication",
                lambda scores: scores[0].__setitem__(
                    "pred_adjudication", "APPROVED"
                ),
            ),
            (
                "confidence",
                lambda scores: scores[0].__setitem__("confidence", 0.26),
            ),
            (
                "Brier",
                lambda scores: scores[0].__setitem__(
                    "confidence_brier", 0.07
                ),
            ),
            (
                "field result",
                lambda scores: scores[0]["field_results"][
                    "applicant_name"
                ].__setitem__("status", "matched"),
            ),
        ):
            with self.subTest(label=label):
                truth, predictions, evaluation, case_scores = atlas_inputs()
                mutate(case_scores)
                with self.assertRaises(AtlasInputError):
                    build_atlas(
                        truth_rows=truth,
                        prediction_rows=predictions,
                        evaluation=evaluation,
                        case_scores=case_scores,
                    )

    def test_rejects_aggregate_raw_and_component_tampering(self):
        for label, mutate in (
            (
                "raw extraction",
                lambda evaluation: evaluation["raw"].__setitem__(
                    "extraction_raw",
                    evaluation["raw"]["extraction_raw"] + 1,
                ),
            ),
            (
                "raw classification",
                lambda evaluation: evaluation["raw"].__setitem__(
                    "classification_raw",
                    evaluation["raw"]["classification_raw"] + 1,
                ),
            ),
            (
                "mean Brier",
                lambda evaluation: evaluation["raw"].__setitem__(
                    "mean_confidence_brier", 0.1
                ),
            ),
            (
                "component score",
                lambda evaluation: evaluation["scores"].__setitem__(
                    "extraction_score",
                    evaluation["scores"]["extraction_score"] + 0.01,
                ),
            ),
            (
                "total score",
                lambda evaluation: evaluation["scores"].__setitem__(
                    "total_score",
                    evaluation["scores"]["total_score"] + 0.01,
                ),
            ),
        ):
            with self.subTest(label=label):
                truth, predictions, evaluation, case_scores = atlas_inputs()
                mutate(evaluation)
                with self.assertRaises(AtlasInputError):
                    build_atlas(
                        truth_rows=truth,
                        prediction_rows=predictions,
                        evaluation=evaluation,
                        case_scores=case_scores,
                    )

    def test_rejects_submission_changed_after_case_scores_were_created(self):
        truth, predictions, evaluation, case_scores = atlas_inputs()
        changed_predictions = copy.deepcopy(predictions)
        changed_predictions[0]["applicant_name"] = "Private Alpha"

        with self.assertRaises(AtlasInputError):
            build_atlas(
                truth_rows=truth,
                prediction_rows=changed_predictions,
                evaluation=evaluation,
                case_scores=case_scores,
            )

    def test_rejects_invalid_source_hash(self):
        truth, predictions, evaluation, case_scores = atlas_inputs()
        with self.assertRaises(AtlasInputError):
            build_atlas(
                truth_rows=truth,
                prediction_rows=predictions,
                evaluation=evaluation,
                case_scores=case_scores,
                source_sha256={
                    "truth": "not-a-hash",
                    "submission": "b" * 64,
                    "evaluation": "c" * 64,
                    "case_scores": "d" * 64,
                },
            )

    def test_aggregate_json_and_markdown_do_not_retain_case_identifiers(self):
        truth, predictions, evaluation, case_scores = extended_atlas_inputs()
        atlas = build_atlas(
            truth_rows=truth,
            prediction_rows=predictions,
            evaluation=evaluation,
            case_scores=case_scores,
        )
        json_output = json.dumps(atlas, sort_keys=True)
        markdown_output = render_markdown(atlas)
        combined = json_output + markdown_output

        self.assertIn("## Evidence integrity", markdown_output)
        self.assertIn("## Pairwise miss co-occurrence", markdown_output)
        self.assertIn("## Wrong-decision / extraction decoupling", markdown_output)
        self.assertIn(
            "## Output + confidence empirical calibration ceiling",
            markdown_output,
        )
        self.assertIsNone(re.search(r"\bMIB-[0-9]{6}\b", combined))
        for sensitive_value in (
            "MIB-100001",
            "MIB-100002",
            "Private One",
            "Private Two",
            "SPN-1234",
        ):
            self.assertNotIn(sensitive_value, combined)

    def test_every_expected_field_is_still_present(self):
        atlas = self.build()
        self.assertEqual(set(atlas["field_losses"]), set(FIELD_NAMES))


if __name__ == "__main__":
    unittest.main()
