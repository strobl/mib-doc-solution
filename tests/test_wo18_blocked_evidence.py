from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from devtools.decision_recovery_cv import FEATURE_ROWS_SCHEMA
from devtools.decision_recovery_evidence import (
    PROTECTED_ROLE_TAXONOMY_SHA256,
    _feature_schema_hash,
    _runtime_feature_names,
)
from devtools.experiment_control import canonical_json
from devtools.wo18_blocked_evidence import (
    COVERAGE_GAP_SCHEMA,
    EXPECTED_SOURCE_REVISION,
    WO18BlockedEvidenceError,
    _revision,
    _sha256_file,
    _evaluated_graph,
    _validate_capture,
    _validate_coverage_gap,
    _validate_score,
    render_markdown,
)
from devtools.wo18_production_capture import CAPTURE_OBSERVATION_SCHEMA
from scripts import evaluate as official_evaluate


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return path


class WO18BlockedEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.layout_sha = "1" * 64
        self.input_sha = "2" * 64
        self.truth_sha = "3" * 64
        self.feature_schema_sha = _feature_schema_hash(
            _runtime_feature_names()
        )
        (
            self.producer_graph_sha,
            self.producer_source_sha,
            _,
            _,
        ) = _evaluated_graph(EXPECTED_SOURCE_REVISION)
        feature_path = _write_json(
            self.root / "feature-rows.json",
            {
                "schema_version": FEATURE_ROWS_SCHEMA,
                "frozen_before_fit": True,
                "source_revision_sha": EXPECTED_SOURCE_REVISION,
                "layout_manifest_sha256": self.layout_sha,
                "input_tree_sha256": self.input_sha,
                "feature_schema_sha256": self.feature_schema_sha,
                "rows": [{"fixture": index} for index in range(32)],
            },
        )
        (self.root / "feature-rows-rerun.json").write_bytes(
            feature_path.read_bytes()
        )
        feature_sha = _sha256_file(feature_path)
        _write_json(
            self.root / "capture-observation.json",
            {
                "schema_version": CAPTURE_OBSERVATION_SCHEMA,
                "source_revision_sha": EXPECTED_SOURCE_REVISION,
                "producer_source_sha256": self.producer_source_sha,
                "producer_graph_sha256": self.producer_graph_sha,
                "layout_manifest_sha256": self.layout_sha,
                "input_tree_sha256": self.input_sha,
                "feature_schema_sha256": self.feature_schema_sha,
                "feature_rows_sha256": feature_sha,
                "rerun_feature_rows_sha256": feature_sha,
                "record_count": 32,
                "capture_run_count": 2,
                "truth_or_role_input_count": 0,
                "byte_deterministic": True,
            },
        )
        self.capture = _validate_capture(
            root=self.root,
            source_revision=EXPECTED_SOURCE_REVISION,
            layout_sha=self.layout_sha,
        )
        self.coverage_payload = {
            "schema_version": COVERAGE_GAP_SCHEMA,
            "frozen_before_scoring": True,
            "source_revision_sha": EXPECTED_SOURCE_REVISION,
            "layout_manifest_sha256": self.layout_sha,
            "role_taxonomy_sha256": PROTECTED_ROLE_TAXONOMY_SHA256,
            "feature_rows_sha256": feature_sha,
            "truth_sha256": self.truth_sha,
            "cohort_record_count": 32,
            "eligible_case_count": 28,
            "unassigned_case_count": 4,
            "role_coverage": {
                "binding_authority": {
                    "eligible_case_count": 0,
                    "eligible_layout_group_count": 0,
                },
                "visible_disqualifier": {
                    "eligible_case_count": 4,
                    "eligible_layout_group_count": 3,
                },
                "approval_guard": {
                    "eligible_case_count": 0,
                    "eligible_layout_group_count": 0,
                },
                "denial_guard": {
                    "eligible_case_count": 4,
                    "eligible_layout_group_count": 3,
                },
                "uncertainty": {
                    "eligible_case_count": 26,
                    "eligible_layout_group_count": 7,
                },
            },
            "missing_required_roles": [
                "binding_authority",
                "approval_guard",
            ],
            "exact_one_role_full_cohort_manifest_possible": False,
            "promotion_blocked": True,
            "candidate_runtime_enabled": False,
        }
        _write_json(
            self.root / "protected-role-coverage-gap.json",
            self.coverage_payload,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_capture_requires_byte_identical_bound_rerun(self) -> None:
        rerun = self.root / "feature-rows-rerun.json"
        rerun.write_bytes(rerun.read_bytes() + b" ")
        with self.assertRaisesRegex(
            WO18BlockedEvidenceError, "byte-identical"
        ):
            _validate_capture(
                root=self.root,
                source_revision=EXPECTED_SOURCE_REVISION,
                layout_sha=self.layout_sha,
            )

    def test_coverage_gap_is_not_accepted_as_complete_roles(self) -> None:
        result = _validate_coverage_gap(
            root=self.root,
            source_revision=EXPECTED_SOURCE_REVISION,
            layout_sha=self.layout_sha,
            capture=self.capture,
            truth_sha=self.truth_sha,
        )
        self.assertEqual(
            result["coverage"]["binding_authority"]["record_count"], 0
        )
        self.assertEqual(result["coverage"]["approval_guard"]["record_count"], 0)

        tampered = dict(self.coverage_payload)
        tampered["missing_required_roles"] = ["binding_authority"]
        _write_json(
            self.root / "protected-role-coverage-gap.json", tampered
        )
        with self.assertRaisesRegex(
            WO18BlockedEvidenceError, "missing_required_roles"
        ):
            _validate_coverage_gap(
                root=self.root,
                source_revision=EXPECTED_SOURCE_REVISION,
                layout_sha=self.layout_sha,
                capture=self.capture,
                truth_sha=self.truth_sha,
            )

    def test_official_score_is_recomputed_not_trusted(self) -> None:
        fields = (
            "case_id",
            "applicant_name",
            "species_code",
            "home_world",
            "visa_class",
            "sponsor_id",
            "arrival_date",
            "declared_purpose",
            "risk_flags",
            "fee_status",
            "adjudication",
            "confidence",
        )
        rows = [
            {
                "case_id": f"MIB-{index:06d}",
                "applicant_name": f"Fixture {index}",
                "species_code": "HUM",
                "home_world": "Earth",
                "visa_class": "DIP-1",
                "sponsor_id": f"SPN-{index:04d}",
                "arrival_date": "2026-07-27",
                "declared_purpose": "Official visit",
                "risk_flags": "none",
                "fee_status": "paid",
                "adjudication": "APPROVED",
                "confidence": 1.0,
            }
            for index in range(1, 33)
        ]
        truth_path = self.root / "truth.csv"
        with truth_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        prediction_path = self.root / "predictions.jsonl"
        prediction_path.write_text(
            "".join(canonical_json(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        truth = official_evaluate.read_truth(truth_path)
        score, _ = official_evaluate.build_results(truth, rows)
        score_path = _write_json(self.root / "score.json", score)
        validated = _validate_score(
            score_path=score_path,
            truth=truth,
            prediction_path=prediction_path,
        )
        self.assertEqual(validated["scores"]["total_score"], 150.0)

        tampered = json.loads(score_path.read_text(encoding="utf-8"))
        tampered["scores"]["total_score"] -= 1
        _write_json(score_path, tampered)
        with self.assertRaisesRegex(
            WO18BlockedEvidenceError, "official evaluator"
        ):
            _validate_score(
                score_path=score_path,
                truth=truth,
                prediction_path=prediction_path,
            )

    def test_revision_is_pinned_and_markdown_states_precondition(self) -> None:
        self.assertEqual(_revision(EXPECTED_SOURCE_REVISION), EXPECTED_SOURCE_REVISION)
        with self.assertRaisesRegex(WO18BlockedEvidenceError, "e433"):
            _revision("a" * 40)
        graph_sha, producer_sha, runner_sha, candidate_not_composed = (
            _evaluated_graph(EXPECTED_SOURCE_REVISION)
        )
        self.assertEqual(
            graph_sha,
            "c2b07e71a354cda602d6eb9f8625897c15163518815ca0f528604b10c2bda076",
        )
        self.assertEqual(len(producer_sha), 64)
        self.assertEqual(len(runner_sha), 64)
        self.assertTrue(candidate_not_composed)
        evidence = {
            "schema_version": "mib-wo18-blocked-evidence/v1",
            "status": "blocked_precondition",
            "comparison_scope": "public_grouped_robustness_not_unseen",
            "source_revision_sha": EXPECTED_SOURCE_REVISION,
            "score_delta": 0.0,
            "checks": {
                "standard_promotion_gate_evaluated": False,
            },
            "gate_results": {
                "candidate_disabled": True,
                "model_promoted": False,
            },
            "class_metrics": {
                approach: {
                    "total_score": 100.0,
                    "score_min": 100.0,
                    "score_max": 100.0,
                }
                for approach in (
                    "deterministic_engine",
                    "evidence_completion_only",
                    "compact_identity_free_model",
                    "gated_hybrid",
                )
            },
        }
        rendered = render_markdown(evidence)
        self.assertIn("BLOCKED AT PRECONDITION", rendered)
        self.assertIn("standard `DecisionRecoveryGate` was not evaluated", rendered)
        self.assertIn("delta of `0.000000`", rendered)


if __name__ == "__main__":
    unittest.main()
