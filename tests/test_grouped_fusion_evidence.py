from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

from devtools.experiment_control import (
    LeakageError,
    canonical_json,
    require_aggregate_only,
)
from devtools.grouped_fusion_evidence import (
    GroupedFusionEvidenceBuildError,
    LAYOUT_MANIFEST_SCHEMA,
    REPO_ROOT,
    build_aggregate_evidence,
    main,
    render_aggregate_markdown,
    verify_clean_candidate_checkout,
    verify_legacy_commit,
)
from devtools.fusion_audit_contract import (
    FUSION_AUDIT_COMPARISON_SCOPE,
    FUSION_AUDIT_INVOCATION_SCOPE,
)
from devtools.ocr_ablation import BASELINE_CONFIG, config_sha256
from scripts import evaluate as official_evaluate


SOURCE_SHA = "a" * 40
LEGACY_SOURCE_SHA = "b" * 40
INPUT_TREE_SHA256 = "c" * 64
BENCHMARK_ID = "wo16-fixture-v1"
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
AUDIT_COUNTS = {
    "changed_field_count": 20,
    "changed_field_complete_provenance_count": 20,
    "clean_higher_authority_override_count": 0,
    "binding_authority_override_count": 0,
    "text_layer_winner_count": 0,
    "serialization_default_used_as_evidence_count": 0,
    "correlated_views_collapsed": 6,
    "independent_agreement_resolutions": 4,
    "same_rank_contested_count": 2,
    "cross_applicant_candidates_excluded": 3,
}


class GroupedFusionEvidenceBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.case_ids = tuple(f"MIB-{index:06d}" for index in range(1, 11))
        manifest = {
            "schema": LAYOUT_MANIFEST_SCHEMA,
            "frozen_before_scoring": True,
            "split_seed": "wo16-test-v1",
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
                if key != "unrecoverable_fields"
            }
            for row in self.truth_rows
        ]
        self.control_paths = self._write_repeats("legacy", self.control_rows)
        self.candidate_paths = self._write_repeats(
            "candidate", self.candidate_rows
        )
        self.control_observations = self._write_observations(
            "legacy-observation",
            self.control_paths,
            source_sha=LEGACY_SOURCE_SHA,
        )
        self.candidate_observations = self._write_observations(
            "candidate-observation",
            self.candidate_paths,
            source_sha=SOURCE_SHA,
        )
        self.audit = self.root / "fusion-audit.json"
        self._write_audit()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_repeats(
        self, stem: str, rows: list[dict[str, object]]
    ) -> tuple[Path, Path]:
        paths: list[Path] = []
        for repeat in range(2):
            path = self.root / f"{stem}-{repeat}.jsonl"
            path.write_text(
                "\n".join(canonical_json(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            paths.append(path)
        return paths[0], paths[1]

    def _write_truth_rows(self) -> None:
        with self.truth.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.truth_rows)

    def _write_observations(
        self,
        stem: str,
        prediction_paths: Sequence[Path],
        *,
        source_sha: str,
        benchmark_id: str = BENCHMARK_ID,
        input_tree_sha256: str = INPUT_TREE_SHA256,
        input_pdf_count: int | None = None,
    ) -> tuple[Path, ...]:
        observations: list[Path] = []
        for repeat, prediction_path in enumerate(
            prediction_paths, start=1
        ):
            payload = {
                "schema_version": "mib_ocr_ablation_v1",
                "benchmark_id": benchmark_id,
                "variant_id": "baseline",
                "repeat_index": repeat,
                "source_revision": source_sha,
                "config_sha256": config_sha256(BASELINE_CONFIG),
                "input_tree_sha256": input_tree_sha256,
                "input_pdf_count": (
                    len(self.case_ids)
                    if input_pdf_count is None
                    else input_pdf_count
                ),
                "max_workers": 4,
                "predictions_path": str(prediction_path.resolve()),
                "predictions_sha256": hashlib.sha256(
                    prediction_path.read_bytes()
                ).hexdigest(),
                "attempted": (
                    len(self.case_ids)
                    if input_pdf_count is None
                    else input_pdf_count
                ),
                "answered": (
                    len(self.case_ids)
                    if input_pdf_count is None
                    else input_pdf_count
                ),
                "omitted": 0,
                "cpu_seconds": 1.0 + repeat,
                "wall_seconds": 0.5 + repeat,
                "metrics_source": (
                    "fresh_process_rusage_self_plus_waited_children_and_"
                    "monotonic_wall"
                ),
                "activity_counts": {},
            }
            observation_path = self.root / f"{stem}-{repeat}.json"
            observation_path.write_text(
                canonical_json(payload) + "\n", encoding="utf-8"
            )
            observations.append(observation_path)
        return tuple(observations)

    def _write_audit(
        self,
        *,
        counts: dict[str, int] | None = None,
        source_sha: str = SOURCE_SHA,
        extra: dict[str, object] | None = None,
    ) -> None:
        payload: dict[str, object] = {
            "comparison_scope": FUSION_AUDIT_COMPARISON_SCOPE,
            "source_revision_sha": source_sha,
            "invocation_scope": FUSION_AUDIT_INVOCATION_SCOPE,
            "input_tree_sha256": INPUT_TREE_SHA256,
            "input_pdf_count": len(self.case_ids),
            "case_id_set_sha256": hashlib.sha256(
                canonical_json(sorted(self.case_ids)).encode("utf-8")
            ).hexdigest(),
            "counts": dict(AUDIT_COUNTS if counts is None else counts),
        }
        payload.update(extra or {})
        self.audit.write_text(
            canonical_json(payload) + "\n", encoding="utf-8"
        )

    def _build(self, **overrides: object) -> dict[str, object]:
        arguments: dict[str, object] = {
            "layout_manifest_path": self.manifest,
            "expected_layout_manifest_sha256": hashlib.sha256(
                self.manifest.read_bytes()
            ).hexdigest(),
            "expected_input_tree_sha256": INPUT_TREE_SHA256,
            "truth_path": self.truth,
            "legacy_control_prediction_paths": self.control_paths,
            "candidate_prediction_paths": self.candidate_paths,
            "legacy_control_observation_paths": self.control_observations,
            "candidate_observation_paths": self.candidate_observations,
            "fusion_audit_path": self.audit,
            "legacy_control_source_revision_sha": LEGACY_SOURCE_SHA,
            "source_revision_sha": SOURCE_SHA,
            "checkout_verifier": lambda _root, _revision: None,
            "legacy_revision_verifier": lambda _root, _revision: None,
        }
        for arm in ("legacy_control", "candidate"):
            prediction_key = f"{arm}_prediction_paths"
            observation_key = f"{arm}_observation_paths"
            if (
                prediction_key in overrides
                and observation_key not in overrides
            ):
                source_sha = (
                    LEGACY_SOURCE_SHA if arm == "legacy_control" else SOURCE_SHA
                )
                arguments[observation_key] = self._write_observations(
                    f"{arm}-automatic-observation",
                    overrides[prediction_key],  # type: ignore[arg-type]
                    source_sha=source_sha,
                )
        arguments.update(overrides)
        return build_aggregate_evidence(**arguments)  # type: ignore[arg-type]

    def _cli_arguments(self, output_json: Path, output_markdown: Path) -> list[str]:
        return [
            "--layout-manifest",
            str(self.manifest),
            "--expected-layout-manifest-sha256",
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "--expected-input-tree-sha256",
            INPUT_TREE_SHA256,
            "--truth",
            str(self.truth),
            "--legacy-control-prediction",
            str(self.control_paths[0]),
            "--legacy-control-prediction",
            str(self.control_paths[1]),
            "--candidate-prediction",
            str(self.candidate_paths[0]),
            "--candidate-prediction",
            str(self.candidate_paths[1]),
            "--legacy-control-observation",
            str(self.control_observations[0]),
            "--legacy-control-observation",
            str(self.control_observations[1]),
            "--candidate-observation",
            str(self.candidate_observations[0]),
            "--candidate-observation",
            str(self.candidate_observations[1]),
            "--fusion-audit",
            str(self.audit),
            "--legacy-control-source-revision-sha",
            LEGACY_SOURCE_SHA,
            "--source-revision-sha",
            SOURCE_SHA,
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]

    @staticmethod
    def _run_cli(arguments: list[str]) -> int:
        with patch(
            "devtools.grouped_fusion_evidence."
            "verify_clean_candidate_checkout"
        ), patch(
            "devtools.grouped_fusion_evidence.verify_legacy_commit"
        ):
            return main(arguments)

    def test_builder_uses_official_evaluator_and_emits_aggregate_only(self):
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
        self.assertEqual(
            aggregate["legacy_control_source_revision_sha"],
            LEGACY_SOURCE_SHA,
        )
        self.assertEqual(aggregate["repeat_count"], 3)
        self.assertEqual(aggregate["fold_count"], 5)
        self.assertEqual(aggregate["evaluated_fold_count"], 15)
        self.assertEqual(len(aggregate["fold_weights"]), 15)
        self.assertEqual(aggregate["input_pdf_count"], len(self.case_ids))
        self.assertEqual(aggregate["input_tree_sha256"], INPUT_TREE_SHA256)
        self.assertRegex(
            str(aggregate["candidate_observation_set_sha256"]),
            r"^[0-9a-f]{64}$",
        )
        self.assertRegex(
            str(aggregate["legacy_control_observation_set_sha256"]),
            r"^[0-9a-f]{64}$",
        )
        self.assertEqual(aggregate["counts"], AUDIT_COUNTS)
        self.assertTrue(all(aggregate["gate_results"].values()))
        require_aggregate_only(aggregate)

        serialized = canonical_json(aggregate)
        self.assertNotIn("MIB-", serialized)
        self.assertNotIn(".pdf", serialized)
        self.assertNotIn(str(self.root), serialized)

    def test_markdown_is_identity_free_and_explicitly_not_unseen(self):
        markdown = render_aggregate_markdown(self._build())

        self.assertIn("Legacy-control comparison", markdown)
        self.assertIn(LEGACY_SOURCE_SHA, markdown)
        self.assertIn("public_grouped_robustness_not_unseen", markdown)
        self.assertIn("Work Order acceptance: **PASS**", markdown)
        self.assertIn("this is not an unseen holdout result", markdown)
        self.assertIn("Correlated views collapsed: 6", markdown)
        self.assertIn("resolver-level fusion vs legacy", markdown)
        self.assertNotIn("MIB-", markdown)
        self.assertNotIn(".pdf", markdown)
        self.assertNotIn(str(self.root), markdown)

    def test_semantically_different_repeat_blocks_exact_determinism(self):
        changed = [dict(row) for row in self.candidate_rows]
        changed[0]["confidence"] = 0.9
        second = self._write_repeats("different", changed)[0]

        aggregate = self._build(
            candidate_prediction_paths=(self.candidate_paths[0], second)
        )

        self.assertFalse(aggregate["deterministic"])
        self.assertFalse(aggregate["gate_results"]["run_deterministic"])
        self.assertEqual(aggregate["status"], "blocked")

    def test_missing_candidate_and_new_false_positive_denial_block(self):
        missing_paths = self._write_repeats(
            "candidate-missing", self.candidate_rows[:-1]
        )
        missing = self._build(candidate_prediction_paths=missing_paths)
        self.assertEqual(missing["missing_records"], 1)
        self.assertFalse(missing["gate_results"]["candidate_complete"])

        denied = [dict(row) for row in self.candidate_rows]
        denied[0]["adjudication"] = "DENIED"
        denied_paths = self._write_repeats("candidate-denied", denied)
        denial = self._build(candidate_prediction_paths=denied_paths)
        self.assertEqual(denial["false_positive_denials_delta"], 1)
        self.assertFalse(
            denial["gate_results"]["no_increased_false_positive_denials"]
        )

    def test_equal_total_catastrophic_swap_is_a_new_safety_regression(self):
        self.truth_rows[0]["adjudication"] = "DENIED"
        self.truth_rows[1]["adjudication"] = "DENIED"
        self._write_truth_rows()

        control = [dict(row) for row in self.control_rows]
        control[0]["adjudication"] = "APPROVED"
        control[1]["adjudication"] = "DENIED"
        candidate = [
            {
                key: value
                for key, value in row.items()
                if key != "unrecoverable_fields"
            }
            for row in self.truth_rows
        ]
        candidate[0]["adjudication"] = "DENIED"
        candidate[1]["adjudication"] = "APPROVED"

        aggregate = self._build(
            legacy_control_prediction_paths=self._write_repeats(
                "catastrophic-swap-control", control
            ),
            candidate_prediction_paths=self._write_repeats(
                "catastrophic-swap-candidate", candidate
            ),
        )

        self.assertEqual(aggregate["catastrophic_false_approvals_delta"], 0)
        self.assertEqual(
            aggregate["new_catastrophic_false_approval_count"], 1
        )
        self.assertFalse(
            aggregate["gate_results"][
                "no_new_catastrophic_false_approvals"
            ]
        )
        self.assertEqual(aggregate["status"], "blocked")

    def test_equal_total_false_denial_swap_is_a_new_safety_regression(self):
        control = [dict(row) for row in self.control_rows]
        control[0]["adjudication"] = "DENIED"
        control[1]["adjudication"] = "APPROVED"
        candidate = [
            {
                key: value
                for key, value in row.items()
                if key != "unrecoverable_fields"
            }
            for row in self.truth_rows
        ]
        candidate[0]["adjudication"] = "APPROVED"
        candidate[1]["adjudication"] = "DENIED"

        aggregate = self._build(
            legacy_control_prediction_paths=self._write_repeats(
                "denial-swap-control", control
            ),
            candidate_prediction_paths=self._write_repeats(
                "denial-swap-candidate", candidate
            ),
        )

        self.assertEqual(aggregate["false_positive_denials_delta"], 0)
        self.assertEqual(aggregate["new_false_positive_denial_count"], 1)
        self.assertFalse(
            aggregate["gate_results"]["no_new_false_positive_denials"]
        )
        self.assertEqual(aggregate["status"], "blocked")

    def test_requires_two_distinct_prediction_artifacts_per_arm(self):
        with self.assertRaisesRegex(Exception, "exactly two"):
            self._build(
                legacy_control_prediction_paths=(self.control_paths[0],)
            )
        with self.assertRaisesRegex(
            Exception, "distinct prediction artifacts"
        ):
            self._build(
                candidate_prediction_paths=(
                    self.candidate_paths[0],
                    self.candidate_paths[0],
                )
            )

    def test_observations_bind_exact_prediction_bytes_and_paths(self):
        self.candidate_paths[0].write_text(
            self.candidate_paths[0].read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "predictions hash mismatch"
        ):
            self._build()

    def test_observations_bind_revisions_repeats_and_shared_input(self):
        wrong_revision = self._write_observations(
            "wrong-revision",
            self.candidate_paths,
            source_sha=LEGACY_SOURCE_SHA,
        )
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "source revision"
        ):
            self._build(candidate_observation_paths=wrong_revision)

        repeated = json.loads(
            self.candidate_observations[1].read_text(encoding="utf-8")
        )
        repeated["repeat_index"] = 1
        self.candidate_observations[1].write_text(
            canonical_json(repeated) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "repeats 1 and 2"
        ):
            self._build()

    def test_control_candidate_and_audit_must_share_frozen_cohort(self):
        different_input = self._write_observations(
            "different-input",
            self.candidate_paths,
            source_sha=SOURCE_SHA,
            input_tree_sha256="d" * 64,
        )
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "input_tree_sha256"
        ):
            self._build(candidate_observation_paths=different_input)

        self._write_audit(extra={"input_tree_sha256": "d" * 64})
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "input tree"
        ):
            self._build()

    def test_pre_recorded_input_tree_digest_is_required_and_enforced(self):
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "pre-recorded input tree"
        ):
            self._build(expected_input_tree_sha256="d" * 64)
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError,
            "expected_input_tree_sha256",
        ):
            self._build(expected_input_tree_sha256="not-a-digest")

    def test_audit_case_set_digest_binds_frozen_manifest_members(self):
        self._write_audit(extra={"case_id_set_sha256": "d" * 64})
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "case cohort"
        ):
            self._build()

    def test_library_checkout_verifier_is_explicitly_invoked(self):
        calls: list[tuple[Path, str]] = []

        self._build(
            checkout_verifier=lambda root, revision: calls.append(
                (root, revision)
            )
        )

        self.assertEqual(calls, [(REPO_ROOT, SOURCE_SHA)])

    def test_checkout_verifier_requires_matching_fully_clean_head(self):
        clean = [
            subprocess.CompletedProcess(
                args=(), returncode=0, stdout=SOURCE_SHA + "\n", stderr=""
            ),
            subprocess.CompletedProcess(
                args=(), returncode=0, stdout="", stderr=""
            ),
        ]
        with patch(
            "devtools.grouped_fusion_evidence.subprocess.run",
            side_effect=clean,
        ) as run:
            verify_clean_candidate_checkout(REPO_ROOT, SOURCE_SHA)
        self.assertIn(
            "--untracked-files=all",
            run.call_args_list[1].args[0],
        )

        wrong_head = [
            subprocess.CompletedProcess(
                args=(),
                returncode=0,
                stdout=LEGACY_SOURCE_SHA + "\n",
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=(), returncode=0, stdout="", stderr=""
            ),
        ]
        with patch(
            "devtools.grouped_fusion_evidence.subprocess.run",
            side_effect=wrong_head,
        ), self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "does not match checkout HEAD"
        ):
            verify_clean_candidate_checkout(REPO_ROOT, SOURCE_SHA)

        for status_line in (" M tracked.py\n", "?? untracked.py\n"):
            with self.subTest(status_line=status_line):
                dirty = [
                    subprocess.CompletedProcess(
                        args=(),
                        returncode=0,
                        stdout=SOURCE_SHA + "\n",
                        stderr="",
                    ),
                    subprocess.CompletedProcess(
                        args=(),
                        returncode=0,
                        stdout=status_line,
                        stderr="",
                    ),
                ]
                with patch(
                    "devtools.grouped_fusion_evidence.subprocess.run",
                    side_effect=dirty,
                ), self.assertRaisesRegex(
                    GroupedFusionEvidenceBuildError,
                    "working tree modifications",
                ):
                    verify_clean_candidate_checkout(REPO_ROOT, SOURCE_SHA)

    def test_legacy_revision_must_name_an_existing_local_commit(self):
        missing = subprocess.CompletedProcess(
            args=(), returncode=1, stdout="", stderr="missing"
        )
        with patch(
            "devtools.grouped_fusion_evidence.subprocess.run",
            return_value=missing,
        ), self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "not a local Git commit"
        ):
            verify_legacy_commit(REPO_ROOT, LEGACY_SOURCE_SHA)

    def test_manifest_digest_and_source_revision_are_strictly_bound(self):
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "pre-recorded frozen digest"
        ):
            self._build(expected_layout_manifest_sha256="b" * 64)
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "source_revision_sha"
        ):
            self._build(source_revision_sha="deadbeef")
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError,
            "legacy_control_source_revision_sha",
        ):
            self._build(legacy_control_source_revision_sha="deadbeef")

        self._write_audit(source_sha="b" * 40)
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "does not match"
        ):
            self._build()

    def test_audit_requires_every_counter_and_matching_complete_provenance(self):
        counts = dict(AUDIT_COUNTS)
        del counts["same_rank_contested_count"]
        self._write_audit(counts=counts)
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError, "same_rank_contested_count"
        ):
            self._build()

        counts = dict(AUDIT_COUNTS)
        counts["changed_field_complete_provenance_count"] = 21
        self._write_audit(counts=counts)
        with self.assertRaisesRegex(
            Exception, "cannot exceed changed field count"
        ):
            self._build()

    def test_audit_rejects_case_identity(self):
        self._write_audit(extra={"case_id": self.case_ids[0]})
        with self.assertRaises(LeakageError):
            self._build()

    def test_audit_scope_is_explicit_and_strict(self):
        self._write_audit(extra={"comparison_scope": "ambiguous"})
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError,
            "comparison_scope",
        ):
            self._build()

        self._write_audit(extra={"invocation_scope": "accepted_only"})
        with self.assertRaisesRegex(
            GroupedFusionEvidenceBuildError,
            "invocation_scope",
        ):
            self._build()

    def test_cli_exit_zero_writes_canonical_identity_free_artifacts(self):
        output_json = self.root / "evidence" / "wo16.json"
        output_markdown = self.root / "evidence" / "wo16.md"

        result = self._run_cli(
            self._cli_arguments(output_json, output_markdown)
        )

        self.assertEqual(result, 0)
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertEqual(
            output_json.read_text(encoding="utf-8"),
            canonical_json(payload) + "\n",
        )
        self.assertNotIn("MIB-", output_json.read_text(encoding="utf-8"))
        self.assertNotIn("MIB-", output_markdown.read_text(encoding="utf-8"))

    def test_cli_exit_two_for_blocked_and_one_for_malformed(self):
        output_json = self.root / "blocked.json"
        output_markdown = self.root / "blocked.md"
        counts = dict(AUDIT_COUNTS)
        counts["clean_higher_authority_override_count"] = 1
        self._write_audit(counts=counts)

        self.assertEqual(
            self._run_cli(
                self._cli_arguments(output_json, output_markdown)
            ),
            2,
        )
        self.assertEqual(
            json.loads(output_json.read_text(encoding="utf-8"))["status"],
            "blocked",
        )

        arguments = self._cli_arguments(
            self.root / "malformed.json", self.root / "malformed.md"
        )
        digest_index = arguments.index("--expected-layout-manifest-sha256") + 1
        arguments[digest_index] = "bad"
        self.assertEqual(self._run_cli(arguments), 1)
        self.assertFalse((self.root / "malformed.json").exists())


if __name__ == "__main__":
    unittest.main()
