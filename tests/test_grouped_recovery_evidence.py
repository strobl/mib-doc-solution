from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from devtools.experiment_control import LeakageError, canonical_json, require_aggregate_only
from devtools.grouped_recovery_evidence import (
    GroupedRecoveryEvidenceBuildError,
    LAYOUT_MANIFEST_SCHEMA,
    build_aggregate_evidence,
    main,
    render_aggregate_markdown,
)
from scripts import evaluate as official_evaluate


SOURCE_SHA = "a" * 40
FIELDS = (
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
    "unrecoverable_fields",
)


class GroupedRecoveryEvidenceBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.case_ids = tuple(f"MIB-{index:06d}" for index in range(1, 11))
        manifest = {
            "schema": LAYOUT_MANIFEST_SCHEMA,
            "frozen_before_scoring": True,
            "split_seed": "wo15-test-v1",
            "repeats": 3,
            "folds": 5,
            "cases": [
                {
                    "case_id": case_id,
                    "layout_group": f"layout-{index // 2}",
                }
                for index, case_id in enumerate(self.case_ids)
            ],
        }
        self.manifest = self.root / "layout-manifest.json"
        self.manifest.write_text(
            canonical_json(manifest) + "\n", encoding="utf-8"
        )
        self.truth_rows = [
            {
                "case_id": case_id,
                "applicant_name": f"Applicant {index}",
                "species_code": "HUM",
                "home_world": "Earth",
                "visa_class": "DIPLOMATIC",
                "sponsor_id": f"SPN-{index:04d}",
                "arrival_date": "2026-07-27",
                "declared_purpose": "Official visit",
                "risk_flags": "none",
                "fee_status": "paid",
                "adjudication": "APPROVED",
                "confidence": "1.0",
                "unrecoverable_fields": "",
            }
            for index, case_id in enumerate(self.case_ids, start=1)
        ]
        self.truth = self.root / "truth.csv"
        with self.truth.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.truth_rows)
        self.control_rows = [
            {
                "case_id": row["case_id"],
                "applicant_name": "Unknown",
                "species_code": "UNK",
                "home_world": "Unknown",
                "visa_class": "UNKNOWN",
                "sponsor_id": "SPN-0000",
                "arrival_date": "1900-01-01",
                "declared_purpose": "Unknown",
                "risk_flags": "none",
                "fee_status": "unknown",
                "adjudication": "NEEDS_REVIEW",
                "confidence": 0.5,
            }
            for row in self.truth_rows
        ]
        self.candidate_rows = [
            {
                key: value
                for key, value in row.items()
                if key not in {"unrecoverable_fields"}
            }
            for row in self.truth_rows
        ]
        self.control_paths = self._write_repeats("control", self.control_rows)
        self.candidate_paths = self._write_repeats(
            "candidate", self.candidate_rows
        )
        self.audit = self.root / "recovery-audit.json"
        self.audit.write_text(
            canonical_json(
                {
                    "recovered_field_count": 20,
                    "recovered_field_complete_provenance_count": 20,
                    "serialization_default_used_as_evidence_count": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_repeats(
        self, stem: str, rows: list[dict[str, object]]
    ) -> tuple[Path, Path]:
        result = []
        for repeat in range(2):
            path = self.root / f"{stem}-{repeat}.jsonl"
            path.write_text(
                "\n".join(canonical_json(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            result.append(path)
        return result[0], result[1]

    def _build(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "layout_manifest_path": self.manifest,
            "expected_layout_manifest_sha256": hashlib.sha256(
                self.manifest.read_bytes()
            ).hexdigest(),
            "truth_path": self.truth,
            "control_prediction_paths": self.control_paths,
            "candidate_prediction_paths": self.candidate_paths,
            "recovery_audit_path": self.audit,
            "source_revision_sha": SOURCE_SHA,
        }
        arguments.update(overrides)
        return build_aggregate_evidence(**arguments)  # type: ignore[arg-type]

    def test_builder_uses_official_scores_and_emits_only_aggregate_evidence(self):
        aggregate = self._build()

        expected, _ = official_evaluate.build_results(
            {row["case_id"]: row for row in self.truth_rows},
            self.candidate_rows,
        )
        self.assertEqual(
            aggregate["full_candidate_score"],
            expected["scores"]["total_score"],
        )
        self.assertEqual(aggregate["status"], "passed")
        self.assertEqual(
            aggregate["evaluation_mode"],
            "public_grouped_robustness_not_unseen",
        )
        self.assertEqual(aggregate["source_revision_sha"], SOURCE_SHA)
        self.assertEqual(aggregate["repeat_count"], 3)
        self.assertEqual(aggregate["fold_count"], 15)
        self.assertTrue(all(aggregate["gate_results"].values()))
        require_aggregate_only(aggregate)

        serialized = canonical_json(aggregate)
        self.assertNotIn("MIB-", serialized)
        self.assertNotIn(".pdf", serialized)
        self.assertNotIn(str(self.root), serialized)

    def test_markdown_is_aggregate_only_and_explicitly_not_unseen(self):
        markdown = render_aggregate_markdown(self._build())

        self.assertIn("public_grouped_robustness_not_unseen", markdown)
        self.assertIn("this is not an unseen holdout result", markdown)
        self.assertNotIn("MIB-", markdown)
        self.assertNotIn(".pdf", markdown)
        self.assertNotIn(str(self.root), markdown)

    def test_new_false_positive_denial_is_counted_and_blocks(self):
        changed = [dict(row) for row in self.candidate_rows]
        changed[0]["adjudication"] = "DENIED"
        candidate_paths = self._write_repeats("candidate-denial", changed)

        aggregate = self._build(candidate_prediction_paths=candidate_paths)

        self.assertEqual(
            aggregate["false_positive_denial_recoveries_delta"], 1
        )
        self.assertFalse(
            aggregate["gate_results"][
                "no_increased_false_positive_denial_recoveries"
            ]
        )
        self.assertEqual(aggregate["status"], "blocked")

    def test_missing_candidate_record_is_reported_and_blocks_completeness(self):
        candidate_paths = self._write_repeats(
            "candidate-missing", self.candidate_rows[:-1]
        )

        aggregate = self._build(candidate_prediction_paths=candidate_paths)

        self.assertEqual(aggregate["missing_records"], 1)
        self.assertFalse(aggregate["gate_results"]["candidate_complete"])
        self.assertEqual(aggregate["status"], "blocked")

    def test_semantically_different_repeat_blocks_determinism(self):
        second_rows = [dict(row) for row in self.candidate_rows]
        second_rows[0]["confidence"] = 0.9
        second = self._write_repeats("candidate-nondeterministic", second_rows)[0]

        aggregate = self._build(
            candidate_prediction_paths=(self.candidate_paths[0], second)
        )

        self.assertFalse(aggregate["deterministic"])
        self.assertFalse(aggregate["gate_results"]["run_deterministic"])
        self.assertEqual(aggregate["status"], "blocked")

    def test_requires_two_runs_and_a_genuinely_frozen_manifest(self):
        with self.assertRaisesRegex(
            GroupedRecoveryEvidenceBuildError, "at least two"
        ):
            self._build(candidate_prediction_paths=(self.candidate_paths[0],))
        with self.assertRaisesRegex(
            GroupedRecoveryEvidenceBuildError, "distinct prediction artifacts"
        ):
            self._build(
                candidate_prediction_paths=(
                    self.candidate_paths[0],
                    self.candidate_paths[0],
                )
            )

        manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        manifest["frozen_before_scoring"] = False
        self.manifest.write_text(
            canonical_json(manifest) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            GroupedRecoveryEvidenceBuildError, "frozen_before_scoring"
        ):
            self._build()

    def test_frozen_manifest_must_match_its_pre_recorded_digest(self):
        with self.assertRaisesRegex(
            GroupedRecoveryEvidenceBuildError, "pre-recorded frozen digest"
        ):
            self._build(expected_layout_manifest_sha256="b" * 64)

    def test_recovery_audit_rejects_case_identity(self):
        self.audit.write_text(
            canonical_json(
                {
                    "recovered_field_count": 1,
                    "recovered_field_complete_provenance_count": 1,
                    "serialization_default_used_as_evidence_count": 0,
                    "case_id": self.case_ids[0],
                }
            ),
            encoding="utf-8",
        )
        with self.assertRaises(LeakageError):
            self._build()

    def test_source_revision_must_be_a_full_digest(self):
        with self.assertRaisesRegex(
            GroupedRecoveryEvidenceBuildError, "source_revision_sha"
        ):
            self._build(source_revision_sha="deadbeef")

    def test_cli_writes_canonical_json_and_identity_free_markdown(self):
        output_json = self.root / "evidence" / "wo15.json"
        output_markdown = self.root / "evidence" / "wo15.md"
        result = main(
            [
                "--layout-manifest",
                str(self.manifest),
                "--expected-layout-manifest-sha256",
                hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                "--truth",
                str(self.truth),
                "--control-prediction",
                str(self.control_paths[0]),
                "--control-prediction",
                str(self.control_paths[1]),
                "--candidate-prediction",
                str(self.candidate_paths[0]),
                "--candidate-prediction",
                str(self.candidate_paths[1]),
                "--recovery-audit",
                str(self.audit),
                "--source-revision-sha",
                SOURCE_SHA,
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_markdown),
            ]
        )

        self.assertEqual(result, 0)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertEqual(
            output_json.read_text(encoding="utf-8"),
            canonical_json(payload) + "\n",
        )
        self.assertNotIn("MIB-", output_json.read_text(encoding="utf-8"))
        self.assertNotIn("MIB-", output_markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
