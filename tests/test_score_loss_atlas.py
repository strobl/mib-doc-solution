import copy
import hashlib
import json
import re
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import evaluate
from devtools.wo13_trace_capture import TraceCaptureError
from devtools.wo13_trace_contract import canonical_json_bytes
from scripts.score_loss_atlas import (
    AtlasInputError,
    FIELD_NAMES,
    REQUIRED_ATLAS_DIMENSIONS,
    TRACE_DIMENSIONS,
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


def layout_rows_for(truth_rows):
    return [
        {
            "case_id": row["case_id"],
            "layout_group": (
                "page-count-03__ink-bucket-00"
                if index < 3
                else "page-count-01__ink-bucket-09"
            ),
        }
        for index, row in enumerate(truth_rows)
    ]


def trace_rows_for(truth_rows):
    rows = []
    for index, truth in enumerate(truth_rows):
        common = index < 10
        rows.append(
            {
                "case_id": truth["case_id"],
                "provenance_route": (
                    "visible_ocr" if common else "authoritative_source"
                ),
                "applicant_linking_state": (
                    "linked_unique" if common else "linked_ambiguous"
                ),
                "evidence_conflict": (
                    "none" if common else "field_conflict"
                ),
                "ocr_recovery_path": (
                    "primary" if common else "orientation_retry"
                ),
                "policy_trace": (
                    "deterministic_policy" if common else "revalidated_policy"
                ),
                "runtime_seconds": (0.5, 1.5, 3.0, 9.0)[index % 4],
            }
        )
    return rows


def dimension_atlas_inputs():
    truth_rows = []
    prediction_rows = []
    for index in range(12):
        case_id = f"MIB-{200000 + index:06d}"
        adjudication = "DENIED" if index % 3 == 0 else "APPROVED"
        truth = truth_row(
            case_id,
            f"Private Person {index}",
            adjudication=adjudication,
        )
        prediction = dict(truth)
        prediction["confidence"] = 0.55 if index < 10 else 0.75
        if index < 4:
            prediction["applicant_name"] = "unknown"
        if index < 3:
            prediction["adjudication"] = "NEEDS_REVIEW"
        truth_rows.append(truth)
        prediction_rows.append(prediction)
    return evaluated_inputs(truth_rows, prediction_rows)


def write_dimension_evidence(directory, truth_rows, *, truth_hash, submission_hash):
    layout_path = Path(directory) / "layout.json"
    layout_payload = {
        "schema": "mib-wo12-layout-groups/v2",
        "label_blind_construction": True,
        "layout_signature": {
            "version": "page-count-plus-first-page-ink-v1",
        },
        "cases": [
            {
                "case_id": row["case_id"],
                "layout_group": (
                    "page-count-03__ink-bucket-00"
                    if index < 10
                    else "page-count-01__ink-bucket-09"
                ),
            }
            for index, row in enumerate(truth_rows)
        ],
    }
    layout_path.write_text(json.dumps(layout_payload), encoding="utf-8")

    trace_path = Path(directory) / "trace.json"
    trace_rows = trace_rows_for(truth_rows)
    trace_payload = {
        "schema_version": "mib-score-loss-dimension-trace/v1",
        "source_revision_sha": "e" * 40,
        "input_tree_sha256": "f" * 64,
        "truth_sha256": truth_hash,
        "submission_sha256": submission_hash,
        "case_count": len(trace_rows),
        "rows": trace_rows,
    }
    trace_path.write_text(json.dumps(trace_payload), encoding="utf-8")
    return layout_path, trace_path


def write_current_source_dimension_evidence(
    directory,
    truth_rows,
    *,
    source_hashes,
    source_revision="a" * 40,
    input_tree_sha256="f" * 64,
):
    layout_path, _legacy_trace = write_dimension_evidence(
        directory,
        truth_rows,
        truth_hash=source_hashes["truth"],
        submission_hash=source_hashes["submission"],
    )
    layout_sha256 = hashlib.sha256(layout_path.read_bytes()).hexdigest()
    rows = trace_rows_for(truth_rows)
    runtime_path = Path(directory) / "runtime-contract.json"
    runtime_path.write_text(
        json.dumps(
            {
                "schema_version": "mib-wo17-runtime-contract/v1",
                "evaluation": {
                    "input_tree_sha256": input_tree_sha256,
                    "layout_manifest_sha256": layout_sha256,
                    "expected_record_count": len(rows),
                },
            }
        ),
        encoding="utf-8",
    )
    runtime_sha256 = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    dataset_archive_path = Path(directory) / "dataset.zip"
    with zipfile.ZipFile(
        dataset_archive_path,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for row in truth_rows:
            archive.writestr(
                f"{row['case_id']}.pdf",
                f"%PDF-fixture-{row['case_id']}".encode("ascii"),
            )
    dataset_archive_sha256 = hashlib.sha256(
        dataset_archive_path.read_bytes()
    ).hexdigest()
    authority_manifest_path = Path(directory) / "authority.json"
    authority_manifest_path.write_text(
        '{"fixture":"authority"}\n',
        encoding="utf-8",
    )
    authority_manifest_sha256 = hashlib.sha256(
        authority_manifest_path.read_bytes()
    ).hexdigest()
    baseline_predictions_path = Path(directory) / "baseline.jsonl"
    baseline_predictions_path.write_bytes(b"synthetic baseline fixture\n")
    trace_path = Path(directory) / "wo13-trace.json"
    artifact_paths = {
        "truth": "data/train_labels.csv",
        "submission": "external/full1000_predictions.jsonl",
        "evaluation": "external/full1000_evaluation.json",
        "case_scores": "external/full1000_case_scores.jsonl",
    }
    frozen_path = Path(directory) / "frozen-baseline.json"
    frozen_path.write_text(
        json.dumps(
            {
                "schema": "mib-frozen-baseline/v1",
                "metadata": {
                    "baseline_commit_sha": source_revision,
                    "count": len(rows),
                },
                "artifacts": [
                    {
                        "path": artifact_paths[name],
                        "sha256": source_hashes[name],
                        "size_bytes": 1,
                    }
                    for name in (
                        "truth",
                        "submission",
                        "evaluation",
                        "case_scores",
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    frozen_sha256 = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    trace_path.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "mib-wo13-truth-blind-trace/v1",
                "capture_mode": "authoritative_production",
                "source_revision_sha": source_revision,
                "checkout_revision_sha": "2" * 40,
                "capture_source_revision_sha": "2" * 40,
                "input_tree_sha256": input_tree_sha256,
                "processing_snapshot_input_tree_sha256": (
                    input_tree_sha256
                ),
                "layout_manifest_sha256": layout_sha256,
                "dataset_archive_sha256": dataset_archive_sha256,
                "runtime_contract_sha256": runtime_sha256,
                "frozen_baseline_manifest_sha256": frozen_sha256,
                "baseline_predictions_sha256": source_hashes[
                    "submission"
                ],
                "runtime_graph_sha256": "7" * 64,
                "trace_tool_sha256": "8" * 64,
                "container_graph_sha256": "9" * 64,
                "source_snapshot_sha256": "0" * 64,
                "authority_manifest_sha256": (
                    authority_manifest_sha256
                ),
                "runtime_identity_sha256": "1" * 64,
                "dependency_identity_sha256": "2" * 64,
                "python_executable_sha256": "3" * 64,
                "predictions_sha256": source_hashes["submission"],
                "production_tree_verified": True,
                "runtime_contract_verified": True,
                "runtime_environment_verified": True,
                "runtime_interface_verified": True,
                "container_limits_verified": True,
                "processing_snapshot_verified": True,
                "stability_checks": {
                    "input_tree_unchanged": True,
                    "layout_manifest_unchanged": True,
                    "dataset_archive_unchanged": True,
                    "runtime_contract_unchanged": True,
                    "frozen_baseline_manifest_unchanged": True,
                    "baseline_predictions_unchanged": True,
                    "runtime_graph_unchanged": True,
                    "trace_tool_unchanged": True,
                    "container_graph_unchanged": True,
                    "source_snapshot_unchanged": True,
                    "authority_manifest_unchanged": True,
                    "runtime_identity_unchanged": True,
                    "checkout_revision_unchanged": True,
                    "capture_source_revision_unchanged": True,
                    "production_tree_unchanged": True,
                },
                "case_count": len(rows),
                "attempted": len(rows),
                "answered": len(rows),
                "omitted": 0,
                "max_workers": 4,
                "retry_missing_attempts": 1,
                "retry_passes_used": 0,
                "batch_wall_seconds": 23.5,
                "rows": rows,
            }
        )
    )
    return (
        layout_path,
        trace_path,
        frozen_path,
        runtime_path,
        dataset_archive_path,
        baseline_predictions_path,
        {
            "source_revision_sha": source_revision,
            "checkout_revision_sha": "2" * 40,
            "input_tree_sha256": input_tree_sha256,
            "layout_manifest_sha256": layout_sha256,
            "dataset_archive_sha256": dataset_archive_sha256,
            "runtime_contract_sha256": runtime_sha256,
            "frozen_baseline_manifest_sha256": frozen_sha256,
            "baseline_predictions_sha256": source_hashes["submission"],
            "runtime_graph_sha256": "7" * 64,
            "trace_tool_sha256": "8" * 64,
            "container_graph_sha256": "9" * 64,
            "source_snapshot_sha256": "0" * 64,
            "authority_manifest_sha256": authority_manifest_sha256,
            "runtime_identity_sha256": "1" * 64,
            "dependency_identity_sha256": "2" * 64,
            "python_executable_sha256": "3" * 64,
            "case_count": len(rows),
            "max_workers": 4,
            "retry_missing_attempts": 1,
            "authority_manifest_path": authority_manifest_path,
            "approved_paths": {
                "input_dir": Path(directory),
                "layout_manifest": layout_path.resolve(),
                "dataset_archive": dataset_archive_path.resolve(),
                "runtime_contract": runtime_path.resolve(),
                "frozen_baseline_manifest": frozen_path.resolve(),
                "baseline_predictions": baseline_predictions_path.resolve(),
            },
        },
    )


def enter_current_source_authority_patches(stack, authority):
    stack.enter_context(
        patch(
            "scripts.score_loss_atlas.load_capture_authority",
            return_value=SimpleNamespace(
                approved_paths=authority["approved_paths"],
                expected_hashes={
                    "input_tree_sha256": authority[
                        "input_tree_sha256"
                    ],
                    "layout_manifest_sha256": authority[
                        "layout_manifest_sha256"
                    ],
                    "dataset_archive_sha256": authority[
                        "dataset_archive_sha256"
                    ],
                    "runtime_contract_sha256": authority[
                        "runtime_contract_sha256"
                    ],
                    "frozen_baseline_manifest_sha256": authority[
                        "frozen_baseline_manifest_sha256"
                    ],
                    "baseline_predictions_sha256": authority[
                        "baseline_predictions_sha256"
                    ],
                    "runtime_graph_sha256": authority[
                        "runtime_graph_sha256"
                    ],
                    "trace_tool_sha256": authority[
                        "trace_tool_sha256"
                    ],
                    "container_graph_sha256": authority[
                        "container_graph_sha256"
                    ],
                    "source_snapshot_sha256": authority[
                        "source_snapshot_sha256"
                    ],
                },
                runtime_identity={
                    "runtime_identity_sha256": authority[
                        "runtime_identity_sha256"
                    ],
                    "dependency_identity_sha256": authority[
                        "dependency_identity_sha256"
                    ],
                    "python_executable_sha256": authority[
                        "python_executable_sha256"
                    ],
                },
                capture_source_revision_sha=authority[
                    "checkout_revision_sha"
                ],
                manifest_sha256=authority[
                    "authority_manifest_sha256"
                ],
                expected_record_count=authority["case_count"],
                max_workers=authority["max_workers"],
                retry_missing_attempts=authority[
                    "retry_missing_attempts"
                ],
            ),
        )
    )
    stack.enter_context(
        patch(
            "scripts.score_loss_atlas.load_frozen_baseline_authority",
            return_value={
                "manifest_sha256": authority[
                    "frozen_baseline_manifest_sha256"
                ],
                "predictions_sha256": authority[
                    "baseline_predictions_sha256"
                ],
                "source_revision_sha": authority[
                    "source_revision_sha"
                ],
                "expected_record_count": authority["case_count"],
            },
        )
    )
    stack.enter_context(
        patch(
            "scripts.score_loss_atlas._verify_layout_and_input",
            return_value=authority["input_tree_sha256"],
        )
    )
    stack.enter_context(
        patch(
            "scripts.score_loss_atlas."
            "verify_dataset_archive_authority",
            return_value={
                "archive_sha256": authority[
                    "dataset_archive_sha256"
                ],
                "input_tree_sha256": authority[
                    "input_tree_sha256"
                ],
                "record_count": authority["case_count"],
                "uncompressed_bytes": 1,
            },
        )
    )


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

    def test_required_dimensions_are_explicitly_measured_or_blocked(self):
        atlas = self.build()

        self.assertEqual(
            set(atlas["required_dimension_coverage"]),
            set(REQUIRED_ATLAS_DIMENSIONS),
        )
        self.assertEqual(
            atlas["required_dimension_coverage"]["field"],
            "frozen_baseline",
        )
        self.assertEqual(
            atlas["required_dimension_coverage"]["adjudication_confusion"],
            "frozen_baseline",
        )
        self.assertEqual(
            atlas["required_dimension_coverage"]["confidence_bucket"],
            "frozen_baseline",
        )
        for dimension in ("page_template_family", *TRACE_DIMENSIONS, "runtime_cost"):
            self.assertEqual(
                atlas["required_dimension_coverage"][dimension],
                "blocked",
            )
        self.assertEqual(
            {item["dimension"] for item in atlas["dimension_measurement_blockers"]},
            {"page_template_family", *TRACE_DIMENSIONS, "runtime_cost"},
        )

    def test_supplied_layout_and_trace_remain_auxiliary_without_source_binding(self):
        truth, predictions, evaluation, case_scores = dimension_atlas_inputs()
        explicit_hashes = {
            "truth": "a" * 64,
            "submission": "b" * 64,
            "evaluation": "c" * 64,
            "case_scores": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            layout_path, trace_path = write_dimension_evidence(
                directory,
                truth,
                truth_hash=explicit_hashes["truth"],
                submission_hash=explicit_hashes["submission"],
            )
            atlas = build_atlas(
                truth_rows=truth,
                prediction_rows=predictions,
                evaluation=evaluation,
                case_scores=case_scores,
                source_sha256=explicit_hashes,
                layout_manifest_path=layout_path,
                trace_dimensions_path=trace_path,
            )

        for dimension in (
            "page_template_family",
            *TRACE_DIMENSIONS,
        ):
            details = atlas["dimension_atlas"][dimension]
            self.assertEqual(details["status"], "auxiliary_historical")
            self.assertEqual(details["emitted_case_count"], 12)
            self.assertIsNone(details["low_support_case_count"])
            self.assertTrue(details["low_support_pool_emitted"])
            self.assertTrue(details["complementary_suppression_applied"])
            self.assertFalse(details["under_k_residual_omitted"])
            self.assertTrue(
                all(group["case_count"] >= 10 for group in details["groups"])
            )
        runtime = atlas["dimension_atlas"]["runtime_cost"]
        self.assertEqual(runtime["status"], "auxiliary_historical")
        self.assertEqual(runtime["emitted_case_count"], 12)
        self.assertTrue(runtime["low_support_pool_emitted"])
        self.assertTrue(
            all(group["case_count"] >= 10 for group in runtime["groups"])
        )
        self.assertEqual(
            atlas["dimension_atlas"]["confidence_bucket"]["status"],
            "frozen_baseline",
        )
        self.assertEqual(
            set(atlas["dimension_source_sha256"]),
            {"layout_manifest", "trace_dimensions"},
        )
        self.assertTrue(
            all(
                re.fullmatch(r"[0-9a-f]{64}", digest)
                for digest in atlas["dimension_source_sha256"].values()
            )
        )
        self.assertEqual(
            {item["dimension"] for item in atlas["dimension_measurement_blockers"]},
            {"page_template_family", *TRACE_DIMENSIONS, "runtime_cost"},
        )

    def test_truth_blind_capture_becomes_current_only_with_complete_binding(self):
        truth, predictions, evaluation, case_scores = dimension_atlas_inputs()
        explicit_hashes = {
            "truth": "a" * 64,
            "submission": "b" * 64,
            "evaluation": "c" * 64,
            "case_scores": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            (
                layout_path,
                trace_path,
                frozen_path,
                runtime_path,
                dataset_archive_path,
                baseline_predictions_path,
                authority,
            ) = write_current_source_dimension_evidence(
                directory,
                truth,
                source_hashes=explicit_hashes,
            )
            with ExitStack() as stack:
                enter_current_source_authority_patches(stack, authority)
                atlas = build_atlas(
                    truth_rows=truth,
                    prediction_rows=predictions,
                    evaluation=evaluation,
                    case_scores=case_scores,
                    source_sha256=explicit_hashes,
                    layout_manifest_path=layout_path,
                    trace_dimensions_path=trace_path,
                    frozen_baseline_manifest_path=frozen_path,
                    runtime_contract_path=runtime_path,
                    dataset_archive_path=dataset_archive_path,
                    baseline_predictions_path=baseline_predictions_path,
                    authority_manifest_path=authority[
                        "authority_manifest_path"
                    ],
                )

        for dimension in (
            "page_template_family",
            *TRACE_DIMENSIONS,
            "runtime_cost",
        ):
            self.assertEqual(
                atlas["dimension_atlas"][dimension]["status"],
                "current_source",
            )
        runtime = atlas["dimension_atlas"]["runtime_cost"]
        self.assertEqual(runtime["batch_wall_seconds"], 23.5)
        self.assertIn("sum_case_latency_seconds", runtime)
        self.assertNotIn("total_runtime_seconds", runtime)
        self.assertEqual(atlas["dimension_measurement_blockers"], [])
        self.assertEqual(
            set(atlas["dimension_source_sha256"]),
            {
                "frozen_baseline_manifest",
                "layout_manifest",
                "trace_dimensions",
            },
        )
        aggregate_output = json.dumps(atlas, sort_keys=True) + render_markdown(
            atlas
        )
        self.assertIsNone(re.search(r"\bMIB-[0-9]{6}\b", aggregate_output))
        self.assertNotIn('"rows"', aggregate_output)

    def test_injected_test_capture_never_becomes_current_source(self):
        truth, predictions, evaluation, case_scores = dimension_atlas_inputs()
        explicit_hashes = {
            "truth": "a" * 64,
            "submission": "b" * 64,
            "evaluation": "c" * 64,
            "case_scores": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            (
                layout_path,
                trace_path,
                frozen_path,
                runtime_path,
                dataset_archive_path,
                baseline_predictions_path,
                authority,
            ) = write_current_source_dimension_evidence(
                directory,
                truth,
                source_hashes=explicit_hashes,
            )
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            payload["capture_mode"] = "test"
            for key in (
                "production_tree_verified",
                "runtime_contract_verified",
                "runtime_environment_verified",
                "runtime_interface_verified",
                "container_limits_verified",
                "processing_snapshot_verified",
            ):
                payload[key] = False
            payload["stability_checks"] = {
                key: False for key in payload["stability_checks"]
            }
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            with ExitStack() as stack:
                enter_current_source_authority_patches(stack, authority)
                atlas = build_atlas(
                    truth_rows=truth,
                    prediction_rows=predictions,
                    evaluation=evaluation,
                    case_scores=case_scores,
                    source_sha256=explicit_hashes,
                    layout_manifest_path=layout_path,
                    trace_dimensions_path=trace_path,
                    frozen_baseline_manifest_path=frozen_path,
                    runtime_contract_path=runtime_path,
                    dataset_archive_path=dataset_archive_path,
                    baseline_predictions_path=baseline_predictions_path,
                    authority_manifest_path=authority[
                        "authority_manifest_path"
                    ],
                )
        for dimension in (
            "page_template_family",
            *TRACE_DIMENSIONS,
            "runtime_cost",
        ):
            self.assertEqual(
                atlas["dimension_atlas"][dimension]["status"],
                "auxiliary_historical",
            )

    def test_truth_blind_capture_fails_closed_on_source_or_input_mismatch(self):
        truth, predictions, evaluation, case_scores = dimension_atlas_inputs()
        explicit_hashes = {
            "truth": "a" * 64,
            "submission": "b" * 64,
            "evaluation": "c" * 64,
            "case_scores": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            (
                layout_path,
                trace_path,
                frozen_path,
                runtime_path,
                dataset_archive_path,
                baseline_predictions_path,
                authority,
            ) = write_current_source_dimension_evidence(
                directory,
                truth,
                source_hashes=explicit_hashes,
            )
            payload = json.loads(trace_path.read_text(encoding="utf-8"))
            payload["source_revision_sha"] = "1" * 40
            mutated_trace = Path(directory) / "bad-source-trace.json"
            mutated_trace.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with self.assertRaises(AtlasInputError):
                build_atlas(
                    truth_rows=truth,
                    prediction_rows=predictions,
                    evaluation=evaluation,
                    case_scores=case_scores,
                    source_sha256=explicit_hashes,
                    layout_manifest_path=layout_path,
                    trace_dimensions_path=mutated_trace,
                    frozen_baseline_manifest_path=frozen_path,
                    runtime_contract_path=runtime_path,
                    dataset_archive_path=dataset_archive_path,
                    baseline_predictions_path=baseline_predictions_path,
                )

            mismatched_authority = dict(authority)
            mismatched_authority["input_tree_sha256"] = "0" * 64
            with ExitStack() as stack:
                enter_current_source_authority_patches(
                    stack,
                    mismatched_authority,
                )
                with self.assertRaises(AtlasInputError):
                    build_atlas(
                        truth_rows=truth,
                        prediction_rows=predictions,
                        evaluation=evaluation,
                        case_scores=case_scores,
                        source_sha256=explicit_hashes,
                        layout_manifest_path=layout_path,
                        trace_dimensions_path=trace_path,
                        frozen_baseline_manifest_path=frozen_path,
                        runtime_contract_path=runtime_path,
                        dataset_archive_path=dataset_archive_path,
                        baseline_predictions_path=(
                            baseline_predictions_path
                        ),
                        authority_manifest_path=authority[
                            "authority_manifest_path"
                        ],
                    )

    def test_low_support_trace_labels_and_case_ids_are_not_emitted(self):
        truth, predictions, evaluation, case_scores = dimension_atlas_inputs()
        explicit_hashes = {
            "truth": "a" * 64,
            "submission": "b" * 64,
            "evaluation": "c" * 64,
            "case_scores": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            layout_path, trace_path = write_dimension_evidence(
                directory,
                truth,
                truth_hash=explicit_hashes["truth"],
                submission_hash=explicit_hashes["submission"],
            )
            atlas = build_atlas(
                truth_rows=truth,
                prediction_rows=predictions,
                evaluation=evaluation,
                case_scores=case_scores,
                source_sha256=explicit_hashes,
                layout_manifest_path=layout_path,
                trace_dimensions_path=trace_path,
            )
        combined = json.dumps(atlas, sort_keys=True) + render_markdown(atlas)

        self.assertNotIn("authoritative_source", combined)
        self.assertNotIn("linked_ambiguous", combined)
        self.assertNotIn("field_conflict", combined)
        self.assertNotIn("orientation_retry", combined)
        self.assertNotIn("revalidated_policy", combined)
        self.assertNotIn("page-count-01__ink-bucket-09", combined)
        self.assertIsNone(re.search(r"\bMIB-[0-9]{6}\b", combined))
        for details in atlas["dimension_atlas"].values():
            for group in details.get("groups", []):
                if group["category"] == "<suppressed_low_support>":
                    self.assertGreaterEqual(group["case_count"], 10)

    def test_runtime_loss_stays_blocked_without_per_case_trace(self):
        truth, predictions, evaluation, case_scores = atlas_inputs()
        atlas = build_atlas(
            truth_rows=truth,
            prediction_rows=predictions,
            evaluation=evaluation,
            case_scores=case_scores,
        )
        runtime = atlas["dimension_atlas"]["runtime_cost"]

        self.assertEqual(runtime["status"], "blocked")
        self.assertNotIn("total_runtime_seconds", runtime)
        self.assertIn(
            "## Required dimension coverage",
            render_markdown(atlas),
        )

    def test_trace_dimensions_fail_closed_on_binding_schema_and_allowlist(self):
        truth, predictions, evaluation, case_scores = dimension_atlas_inputs()
        explicit_hashes = {
            "truth": "a" * 64,
            "submission": "b" * 64,
            "evaluation": "c" * 64,
            "case_scores": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            _layout_path, valid_path = write_dimension_evidence(
                directory,
                truth,
                truth_hash=explicit_hashes["truth"],
                submission_hash=explicit_hashes["submission"],
            )
            valid = json.loads(valid_path.read_text(encoding="utf-8"))
            mutations = []
            missing = copy.deepcopy(valid)
            missing["rows"] = missing["rows"][:-1]
            missing["case_count"] -= 1
            mutations.append(missing)
            extra_key = copy.deepcopy(valid)
            extra_key["rows"][0]["unexpected"] = "value"
            mutations.append(extra_key)
            bad_allowlist = copy.deepcopy(valid)
            bad_allowlist["rows"][0]["policy_trace"] = "per_case_secret"
            mutations.append(bad_allowlist)
            wrong_binding = copy.deepcopy(valid)
            wrong_binding["submission_sha256"] = "9" * 64
            mutations.append(wrong_binding)

            for index, payload in enumerate(mutations):
                trace_path = Path(directory) / f"mutated-{index}.json"
                trace_path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(AtlasInputError):
                    build_atlas(
                        truth_rows=truth,
                        prediction_rows=predictions,
                        evaluation=evaluation,
                        case_scores=case_scores,
                        source_sha256=explicit_hashes,
                        trace_dimensions_path=trace_path,
                    )

    def test_layout_manifest_rejects_identity_encodable_category(self):
        truth, predictions, evaluation, case_scores = dimension_atlas_inputs()
        with tempfile.TemporaryDirectory() as directory:
            layout_path, _trace_path = write_dimension_evidence(
                directory,
                truth,
                truth_hash="a" * 64,
                submission_hash="b" * 64,
            )
            payload = json.loads(layout_path.read_text(encoding="utf-8"))
            for row in payload["cases"]:
                row["layout_group"] = "john_smith"
            layout_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(AtlasInputError):
                build_atlas(
                    truth_rows=truth,
                    prediction_rows=predictions,
                    evaluation=evaluation,
                    case_scores=case_scores,
                    layout_manifest_path=layout_path,
                )

    def test_current_source_recomputes_and_rejects_engineered_layout_groups(self):
        truth, predictions, evaluation, case_scores = dimension_atlas_inputs()
        explicit_hashes = {
            "truth": "a" * 64,
            "submission": "b" * 64,
            "evaluation": "c" * 64,
            "case_scores": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            (
                layout_path,
                trace_path,
                frozen_path,
                runtime_path,
                dataset_archive_path,
                baseline_predictions_path,
                authority,
            ) = write_current_source_dimension_evidence(
                directory,
                truth,
                source_hashes=explicit_hashes,
            )
            payload = json.loads(layout_path.read_text(encoding="utf-8"))
            for index, row in enumerate(payload["cases"]):
                row["layout_group"] = (
                    "page-count-99__ink-bucket-99"
                    if index < 10
                    else "page-count-98__ink-bucket-98"
                )
            layout_path.write_text(json.dumps(payload), encoding="utf-8")
            engineered_hash = hashlib.sha256(
                layout_path.read_bytes()
            ).hexdigest()
            authority["layout_manifest_sha256"] = engineered_hash
            trace_payload = json.loads(
                trace_path.read_text(encoding="utf-8")
            )
            trace_payload["layout_manifest_sha256"] = engineered_hash
            trace_path.write_bytes(
                canonical_json_bytes(trace_payload)
            )
            with ExitStack() as stack:
                enter_current_source_authority_patches(stack, authority)
                stack.enter_context(
                    patch(
                        "scripts.score_loss_atlas."
                        "_verify_layout_and_input",
                        side_effect=TraceCaptureError(
                            "layout manifest does not match recomputed PDFs"
                        ),
                    )
                )
                with self.assertRaisesRegex(
                    AtlasInputError,
                    "recomputed PDFs",
                ):
                    build_atlas(
                        truth_rows=truth,
                        prediction_rows=predictions,
                        evaluation=evaluation,
                        case_scores=case_scores,
                        source_sha256=explicit_hashes,
                        layout_manifest_path=layout_path,
                        trace_dimensions_path=trace_path,
                        frozen_baseline_manifest_path=frozen_path,
                        runtime_contract_path=runtime_path,
                        dataset_archive_path=dataset_archive_path,
                        baseline_predictions_path=(
                            baseline_predictions_path
                        ),
                        authority_manifest_path=authority[
                            "authority_manifest_path"
                        ],
                    )

    def test_current_source_atlas_rejects_engineered_dataset_archive(self):
        truth, predictions, evaluation, case_scores = dimension_atlas_inputs()
        explicit_hashes = {
            "truth": "a" * 64,
            "submission": "b" * 64,
            "evaluation": "c" * 64,
            "case_scores": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            (
                layout_path,
                trace_path,
                frozen_path,
                runtime_path,
                dataset_archive_path,
                baseline_predictions_path,
                authority,
            ) = write_current_source_dimension_evidence(
                directory,
                truth,
                source_hashes=explicit_hashes,
            )
            with ExitStack() as stack:
                enter_current_source_authority_patches(stack, authority)
                stack.enter_context(
                    patch(
                        "scripts.score_loss_atlas."
                        "verify_dataset_archive_authority",
                        side_effect=TraceCaptureError(
                            "dataset archive PDF tree differs from input authority"
                        ),
                    )
                )
                with self.assertRaisesRegex(
                    AtlasInputError,
                    "archive PDF tree differs",
                ):
                    build_atlas(
                        truth_rows=truth,
                        prediction_rows=predictions,
                        evaluation=evaluation,
                        case_scores=case_scores,
                        source_sha256=explicit_hashes,
                        layout_manifest_path=layout_path,
                        trace_dimensions_path=trace_path,
                        frozen_baseline_manifest_path=frozen_path,
                        runtime_contract_path=runtime_path,
                        dataset_archive_path=dataset_archive_path,
                        baseline_predictions_path=(
                            baseline_predictions_path
                        ),
                        authority_manifest_path=authority[
                            "authority_manifest_path"
                        ],
                    )

    def test_every_expected_field_is_still_present(self):
        atlas = self.build()
        self.assertEqual(set(atlas["field_losses"]), set(FIELD_NAMES))


if __name__ == "__main__":
    unittest.main()
