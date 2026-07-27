from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from devtools.experiment_control import canonical_json
from devtools.confidence_refit_cv import compare_confidence_families
from devtools.final_confidence_capture import (
    OBSERVATION_SCHEMA,
    CaptureBindings,
    CapturedFinalPrediction,
    build_capture_payload,
    non_confidence_jsonl,
    prediction_jsonl,
)
from devtools.grouped_recovery_evidence import (
    LAYOUT_MANIFEST_SCHEMA,
    FrozenLayoutManifest,
)
from devtools.ocr_ablation import _input_tree_sha256
from devtools.wo19_confidence_evidence import (
    EVIDENCE_SCHEMA,
    TRUTH_FIELDS,
    WO19EvidenceError,
    assert_non_confidence_unchanged,
    build_wo19_evidence,
)
from mib_pipeline.final_confidence import (
    FinalConfidenceContext,
    FinalPredictionWithConfidenceContext,
)
from mib_pipeline.confidence_refit import FittedConfidenceCalibrator
from mib_pipeline.models import PredictionRow


REVISION = "a" * 40


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _row(case_id: str, confidence: float = 1.0) -> PredictionRow:
    return PredictionRow.from_mapping(
        {
            "case_id": case_id,
            "applicant_name": "Applicant",
            "species_code": "HUMAN",
            "home_world": "Earth",
            "visa_class": "XW-1",
            "sponsor_id": "SPN-0001",
            "arrival_date": "2026-07-27",
            "declared_purpose": "research",
            "risk_flags": "none",
            "fee_status": "paid",
            "adjudication": "APPROVED",
            "confidence": confidence,
        }
    )


def _context() -> FinalConfidenceContext:
    return FinalConfidenceContext(
        schema_version=1,
        final_class="APPROVED",
        policy_route="deterministic_policy",
        authoritative=False,
        visible_completeness=1.0,
        has_conflict=False,
        ocr_disagreement=0.0,
        recovery_route="primary",
        model_margin=None,
        ensemble_agreement=None,
        resolution_entropy=0.0,
    )


class WO19EvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "input"
        self.input_dir.mkdir()
        self.case_ids = tuple(
            f"MIB-{index:06d}" for index in range(1, 33)
        )
        for case_id in self.case_ids:
            (self.input_dir / f"{case_id}.pdf").write_bytes(
                case_id.encode("ascii")
            )
        manifest_value = {
            "schema": LAYOUT_MANIFEST_SCHEMA,
            "frozen_before_scoring": True,
            "split_seed": "wo19-test-v1",
            "repeats": 3,
            "folds": 5,
            "cases": [
                {
                    "case_id": case_id,
                    "layout_group": f"private-layout-{index % 9}",
                }
                for index, case_id in enumerate(self.case_ids)
            ],
        }
        self.layout_path = self.root / "layout.json"
        self.layout_path.write_bytes(_canonical(manifest_value))
        layout_sha = _sha(self.layout_path.read_bytes())
        layout = FrozenLayoutManifest(
            groups={
                f"private-layout-{group}": tuple(
                    case_id
                    for index, case_id in enumerate(self.case_ids)
                    if index % 9 == group
                )
                for group in range(9)
            },
            split_seed="wo19-test-v1",
            sha256=layout_sha,
            frozen_before_scoring=True,
        )
        input_sha, count = _input_tree_sha256(self.input_dir)
        self.assertEqual(count, 32)
        self.bindings = CaptureBindings(
            source_revision_sha=REVISION,
            layout_manifest_sha256=layout_sha,
            input_tree_sha256=input_sha,
            producer_graph_sha256="b" * 64,
            producer_source_sha256="c" * 64,
            runtime_confidence_artifact_sha256="d" * 64,
            runtime_confidence_artifact_file_sha256="e" * 64,
            runtime_confidence_artifact_id="current-test-artifact",
            context_contract_sha256="f" * 64,
            context_contract_source_sha256="1" * 64,
        )

        def capture_case(path: Path) -> CapturedFinalPrediction:
            row = _row(path.stem)
            accepted = FinalPredictionWithConfidenceContext(
                row=row,
                context=_context(),
            )
            return CapturedFinalPrediction(
                accepted=accepted,
                final_row=row,
            )

        self.capture = build_capture_payload(
            input_dir=self.input_dir,
            layout_manifest=layout,
            bindings=self.bindings,
            capture_case=capture_case,
        )
        self.capture_path = self.root / "capture.json"
        self.rerun_path = self.root / "capture-rerun.json"
        capture_bytes = _canonical(self.capture)
        self.capture_path.write_bytes(capture_bytes)
        self.rerun_path.write_bytes(capture_bytes)
        self.predictions = prediction_jsonl(self.capture)
        self.non_confidence = non_confidence_jsonl(self.capture)
        self.prediction_path = self.root / "predictions.jsonl"
        self.rerun_prediction_path = self.root / "predictions-rerun.jsonl"
        self.prediction_path.write_bytes(self.predictions)
        self.rerun_prediction_path.write_bytes(self.predictions)
        observation = {
            "schema_version": OBSERVATION_SCHEMA,
            "source_revision_sha": REVISION,
            "layout_manifest_sha256": layout_sha,
            "input_tree_sha256": input_sha,
            "producer_graph_sha256": "b" * 64,
            "producer_source_sha256": "c" * 64,
            "runtime_confidence_artifact_sha256": "d" * 64,
            "runtime_confidence_artifact_file_sha256": "e" * 64,
            "context_contract_sha256": "f" * 64,
            "context_contract_source_sha256": "1" * 64,
            "confidence_topology": self.capture["confidence_topology"],
            "capture_sha256": _sha(capture_bytes),
            "rerun_capture_sha256": _sha(capture_bytes),
            "full_predictions_jsonl_sha256": _sha(self.predictions),
            "rerun_full_predictions_jsonl_sha256": _sha(self.predictions),
            "non_confidence_jsonl_sha256": _sha(self.non_confidence),
            "rerun_non_confidence_jsonl_sha256": _sha(
                self.non_confidence
            ),
            "record_count": 32,
            "capture_run_count": 2,
            "missing_record_count": 0,
            "duplicate_record_count": 0,
            "invalid_record_count": 0,
            "out_of_range_confidence_count": 0,
            "byte_deterministic": True,
            "truth_or_label_input_count": 0,
            "identity_bearing_capture_storage": "external",
            "aggregate_only": True,
        }
        self.observation_path = self.root / "observation.json"
        self.observation_path.write_bytes(_canonical(observation))
        self.truth_path = self.root / "truth.csv"
        with self.truth_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRUTH_FIELDS)
            writer.writeheader()
            for case_id in self.case_ids:
                writer.writerow(
                    {
                        **{
                            key: value
                            for key, value in _row(case_id).to_dict().items()
                            if key != "confidence"
                        }
                    }
                )
        self.truth_sha = _sha(self.truth_path.read_bytes())
        self.solution_calls = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _runner(self, _input: Path, output: Path) -> int:
        self.solution_calls += 1
        output.write_bytes(self.predictions)
        return 0

    def _build(self, **overrides):
        arguments = {
            "capture_path": self.capture_path,
            "rerun_capture_path": self.rerun_path,
            "capture_predictions_path": self.prediction_path,
            "rerun_capture_predictions_path": self.rerun_prediction_path,
            "observation_path": self.observation_path,
            "layout_manifest_path": self.layout_path,
            "input_dir": self.input_dir,
            "truth_path": self.truth_path,
            "expected_truth_sha256": self.truth_sha,
            "source_revision_sha": REVISION,
            "verify_repository": False,
            "solution_runner": self._runner,
        }
        arguments.update(overrides)
        return build_wo19_evidence(**arguments)

    def test_happy_no_promotion_is_aggregate_only_and_runs_batch(self) -> None:
        result = self._build()

        self.assertEqual(self.solution_calls, 1)
        self.assertEqual(result.evidence["schema_version"], EVIDENCE_SCHEMA)
        self.assertEqual(result.evidence["status"], "evaluated_no_promotion")
        self.assertFalse(result.needs_s1)
        self.assertFalse(
            result.evidence["checks"]["candidate_runtime_composed"]
        )
        self.assertEqual(
            result.evidence["decision"]["runtime_artifact_action"],
            "retained_current_s0_artifact",
        )
        rendered = canonical_json(result.evidence) + result.markdown
        for private_value in (
            *self.case_ids,
            *(f"private-layout-{index}" for index in range(9)),
        ):
            self.assertNotIn(private_value, rendered)
        private_samples = json.loads(result.bound_samples_bytes)
        self.assertEqual(private_samples["sample_count"], 32)
        self.assertIn(self.case_ids[0], result.bound_samples_bytes.decode())

    def test_observation_binding_mutation_fails_closed(self) -> None:
        observation = json.loads(self.observation_path.read_text())
        observation["input_tree_sha256"] = "9" * 64
        self.observation_path.write_bytes(_canonical(observation))

        with self.assertRaisesRegex(WO19EvidenceError, "observation"):
            self._build()
        self.assertEqual(self.solution_calls, 0)

    def test_tool_generated_batch_mismatch_fails_closed(self) -> None:
        def mismatching_runner(_input: Path, output: Path) -> int:
            changed = self.predictions.replace(
                b'"confidence":1.0',
                b'"confidence":0.9',
                1,
            )
            output.write_bytes(changed)
            return 0

        with self.assertRaisesRegex(WO19EvidenceError, "BatchRunner bytes"):
            self._build(solution_runner=mismatching_runner)

    def test_non_confidence_mutation_fails_closed(self) -> None:
        baseline = _row(self.case_ids[0])
        changed = dataclasses.replace(baseline, fee_status="unpaid")

        with self.assertRaisesRegex(
            WO19EvidenceError,
            "non-confidence field",
        ):
            assert_non_confidence_unchanged(baseline, changed)

    def test_cv_pass_cannot_override_degrading_realized_shadow(self) -> None:
        def cv_pass_with_degrading_full_fit(examples):
            report, artifacts = compare_confidence_families(examples)
            report["selection"]["family"] = "isotonic"
            report["selection"]["promotion_checks"] = {
                key: True
                for key in report["selection"]["promotion_checks"]
            }
            report["selection"]["promotion_recommended"] = True
            artifacts["isotonic"] = FittedConfidenceCalibrator(
                family="isotonic",
                parameters={
                    "thresholds": [0.0, 1.0],
                    "values": [0.0, 0.0],
                },
            )
            return report, artifacts

        result = self._build(comparison_runner=cv_pass_with_degrading_full_fit)

        selection = result.evidence["selection"]
        self.assertTrue(selection["cv_promotion_recommended"])
        self.assertFalse(
            selection["realized_shadow_checks"][
                "mean_brier_strictly_improved"
            ]
        )
        self.assertFalse(
            selection["realized_shadow_checks"][
                "calibration_score_strictly_improved"
            ]
        )
        self.assertFalse(
            selection["realized_shadow_checks"][
                "total_score_strictly_improved"
            ]
        )
        self.assertFalse(selection["promotion_recommended"])
        self.assertFalse(result.needs_s1)
        self.assertEqual(result.evidence["status"], "evaluated_no_promotion")
        self.assertEqual(
            result.evidence["decision"]["runtime_artifact_action"],
            "retained_current_s0_artifact",
        )


if __name__ == "__main__":
    unittest.main()
