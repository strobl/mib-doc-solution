from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Mapping, Sequence
from unittest.mock import patch

from devtools.experiment_control import canonical_json, require_aggregate_only
from devtools.grouped_recovery_evidence import LAYOUT_MANIFEST_SCHEMA
from devtools.ocr_ablation import BASELINE_CONFIG, config_sha256
from devtools.policy_revalidation_audit_contract import (
    CONTRACT_AUDIT_COUNTS,
    COHORT_AUDIT_COUNTS,
    POLICY_REVALIDATION_AUDIT_SCHEMA,
)
from devtools.policy_revalidation_evidence import (
    PolicyRevalidationEvidenceBuildError,
    build_aggregate_evidence,
    main,
    render_aggregate_markdown,
)


SOURCE_SHA = "a" * 40
LEGACY_SHA = "b" * 40
INPUT_TREE_SHA = "c" * 64
FIXTURE_SHA = "d" * 64
BENCHMARK_ID = "wo17-fixture-v1"
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


def _cohort_counts() -> dict[str, int]:
    values = {name: 0 for name in COHORT_AUDIT_COUNTS}
    values["accepted_final_policy_result_count"] = 32
    return values


def _contract_counts() -> dict[str, int]:
    values = {name: 0 for name in CONTRACT_AUDIT_COUNTS}
    values.update(
        {
            "legacy_synthetic_before_late_recovery_count": 2,
            "candidate_late_recovery_before_revalidation_count": 4,
            "candidate_revalidation_after_late_recovery_count": 4,
            "contradicted_synthetic_reason_before_count": 2,
            "contradicted_synthetic_reason_removed_count": 2,
            "independent_denial_reason_retained_count": 1,
            "review_confidence_restored_count": 1,
            "normal_policy_rerun_count": 4,
            "signed_late_authority_recovery_count": 3,
            "late_adjudication_evidence_preserved_count": 3,
            "late_biohazard_evidence_preserved_count": 1,
            "placeholder_guard_probe_count": 35,
            "sentinel_guard_probe_count": 2,
            "serialization_default_guard_probe_count": 35,
            "stale_threshold_guard_probe_count": 2,
            "forced_approval_guard_probe_count": 35,
            "direct_approval_head_guard_probe_count": 1,
        }
    )
    return values


class PolicyRevalidationEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.case_ids = tuple(
            f"MIB-{index:06d}" for index in range(1, 33)
        )
        manifest = {
            "schema": LAYOUT_MANIFEST_SCHEMA,
            "frozen_before_scoring": True,
            "split_seed": "wo17-test-v1",
            "repeats": 3,
            "folds": 5,
            "cases": [
                {
                    "case_id": case_id,
                    "layout_group": f"layout-{(index - 1) % 8}",
                }
                for index, case_id in enumerate(self.case_ids, start=1)
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
                "visa_class": "DIP-1",
                "sponsor_id": f"SPN-{index:04d}",
                "arrival_date": "2026-07-27",
                "declared_purpose": "Official visit",
                "risk_flags": "none",
                "fee_status": "paid",
                "adjudication": (
                    "APPROVED" if index % 2 else "DENIED"
                ),
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
                key: value
                for key, value in row.items()
                if key != "unrecoverable_fields"
            }
            for row in self.truth_rows
        ]
        self.control_rows[0]["adjudication"] = "NEEDS_REVIEW"
        self.control_rows[0]["confidence"] = 0.5
        self.control_rows[1]["adjudication"] = "NEEDS_REVIEW"
        self.control_rows[1]["confidence"] = 0.5
        self.candidate_rows = [
            {
                key: value
                for key, value in row.items()
                if key != "unrecoverable_fields"
            }
            for row in self.truth_rows
        ]
        self.control_predictions = self._write_predictions(
            "legacy", self.control_rows
        )
        self.candidate_predictions = self._write_predictions(
            "candidate", self.candidate_rows
        )
        self.control_observations = self._write_observations(
            "legacy-observation",
            self.control_predictions,
            source_sha=LEGACY_SHA,
        )
        self.candidate_observations = self._write_observations(
            "candidate-observation",
            self.candidate_predictions,
            source_sha=SOURCE_SHA,
        )
        self.audit_paths = self._write_audits()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_predictions(
        self, stem: str, rows: Sequence[Mapping[str, object]]
    ) -> tuple[Path, Path]:
        paths = []
        for repeat in (1, 2):
            path = self.root / f"{stem}-{repeat}.jsonl"
            path.write_text(
                "\n".join(canonical_json(dict(row)) for row in rows) + "\n",
                encoding="utf-8",
            )
            paths.append(path)
        return paths[0], paths[1]

    def _write_observations(
        self,
        stem: str,
        prediction_paths: Sequence[Path],
        *,
        source_sha: str,
    ) -> tuple[Path, Path]:
        paths = []
        for repeat, prediction_path in enumerate(
            prediction_paths, start=1
        ):
            payload = {
                "schema_version": "mib_ocr_ablation_v1",
                "benchmark_id": BENCHMARK_ID,
                "variant_id": "baseline",
                "repeat_index": repeat,
                "source_revision": source_sha,
                "config_sha256": config_sha256(BASELINE_CONFIG),
                "input_tree_sha256": INPUT_TREE_SHA,
                "input_pdf_count": 32,
                "max_workers": 4,
                "predictions_path": str(prediction_path.resolve()),
                "predictions_sha256": hashlib.sha256(
                    prediction_path.read_bytes()
                ).hexdigest(),
                "attempted": 32,
                "answered": 32,
                "omitted": 0,
                "cpu_seconds": 1.0 + repeat,
                "wall_seconds": 0.5 + repeat,
                "metrics_source": (
                    "fresh_process_rusage_self_plus_waited_children_and_"
                    "monotonic_wall"
                ),
                "activity_counts": {},
            }
            path = self.root / f"{stem}-{repeat}.json"
            path.write_text(
                canonical_json(payload) + "\n", encoding="utf-8"
            )
            paths.append(path)
        return paths[0], paths[1]

    def _write_audits(
        self,
        *,
        source_sha: str = SOURCE_SHA,
        input_tree_sha: str = INPUT_TREE_SHA,
        case_set_sha: str | None = None,
        fixture_shas: tuple[str, str] = (FIXTURE_SHA, FIXTURE_SHA),
        cohort_by_repeat: tuple[dict[str, int], dict[str, int]]
        | None = None,
        contract_by_repeat: tuple[dict[str, int], dict[str, int]]
        | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> tuple[Path, Path]:
        if case_set_sha is None:
            case_set_sha = hashlib.sha256(
                canonical_json(sorted(self.case_ids)).encode("utf-8")
            ).hexdigest()
        cohort_by_repeat = cohort_by_repeat or (
            _cohort_counts(),
            _cohort_counts(),
        )
        contract_by_repeat = contract_by_repeat or (
            _contract_counts(),
            _contract_counts(),
        )
        paths = []
        for repeat in (1, 2):
            payload: dict[str, object] = {
                "schema_version": POLICY_REVALIDATION_AUDIT_SCHEMA,
                "source_revision_sha": source_sha,
                "input_tree_sha256": input_tree_sha,
                "input_pdf_count": 32,
                "case_id_set_sha256": case_set_sha,
                "repeat_index": repeat,
                "predictions_sha256": hashlib.sha256(
                    self.candidate_predictions[repeat - 1].read_bytes()
                ).hexdigest(),
                "contract_fixture_sha256": fixture_shas[repeat - 1],
                "cohort_counts": cohort_by_repeat[repeat - 1],
                "contract_counts": contract_by_repeat[repeat - 1],
            }
            payload.update(extra or {})
            path = self.root / f"policy-audit-{repeat}.json"
            path.write_text(
                canonical_json(payload) + "\n", encoding="utf-8"
            )
            paths.append(path)
        return paths[0], paths[1]

    def _build(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "layout_manifest_path": self.manifest,
            "expected_layout_manifest_sha256": hashlib.sha256(
                self.manifest.read_bytes()
            ).hexdigest(),
            "expected_input_tree_sha256": INPUT_TREE_SHA,
            "truth_path": self.truth,
            "legacy_control_prediction_paths": self.control_predictions,
            "candidate_prediction_paths": self.candidate_predictions,
            "legacy_control_observation_paths": self.control_observations,
            "candidate_observation_paths": self.candidate_observations,
            "candidate_policy_audit_paths": self.audit_paths,
            "legacy_control_source_revision_sha": LEGACY_SHA,
            "source_revision_sha": SOURCE_SHA,
            "checkout_verifier": lambda _root, _revision: None,
            "legacy_revision_verifier": lambda _root, _revision: None,
        }
        values.update(overrides)
        return build_aggregate_evidence(**values)  # type: ignore[arg-type]

    def test_builds_passing_bound_identity_free_evidence(self):
        aggregate = self._build()

        self.assertEqual(aggregate["status"], "passed")
        self.assertGreater(aggregate["score_delta"], 0)
        self.assertEqual(
            aggregate["confusion_counts"][
                "approved_to_needs_review_delta"
            ],
            -1,
        )
        self.assertEqual(
            aggregate["confusion_counts"][
                "denied_to_needs_review_delta"
            ],
            -1,
        )
        self.assertEqual(aggregate["non_policy_field_change_count"], 0)
        self.assertEqual(
            aggregate["contract_fixture_sha256"], FIXTURE_SHA
        )
        require_aggregate_only(aggregate)
        serialized = canonical_json(aggregate)
        for forbidden in ("MIB-", ".pdf", "Applicant", "Official visit"):
            self.assertNotIn(forbidden, serialized)
        markdown = render_aggregate_markdown(aggregate)
        self.assertIn("APPROVED → NEEDS_REVIEW", markdown)
        self.assertIn("Contract probes", markdown)
        self.assertNotIn("Applicant 1", markdown)

    def test_audit_must_bind_source_tree_cohort_predictions_and_fixture(self):
        scenarios = {
            "source": {
                "candidate_policy_audit_paths": self._write_audits(
                    source_sha="e" * 40
                )
            },
            "tree": {
                "candidate_policy_audit_paths": self._write_audits(
                    input_tree_sha="e" * 64
                )
            },
            "cohort": {
                "candidate_policy_audit_paths": self._write_audits(
                    case_set_sha="e" * 64
                )
            },
            "fixture": {
                "candidate_policy_audit_paths": self._write_audits(
                    fixture_shas=(FIXTURE_SHA, "e" * 64)
                )
            },
        }
        for label, values in scenarios.items():
            with self.subTest(label=label):
                with self.assertRaises(PolicyRevalidationEvidenceBuildError):
                    self._build(**values)

        first = json.loads(self.audit_paths[0].read_text(encoding="utf-8"))
        first["predictions_sha256"] = "e" * 64
        self.audit_paths[0].write_text(
            canonical_json(first) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            PolicyRevalidationEvidenceBuildError,
            "prediction bytes",
        ):
            self._build()

    def test_extra_or_missing_audit_keys_and_counts_fail_closed(self):
        extra = self._write_audits(extra={"applicant_name": "forbidden"})
        with self.assertRaisesRegex(
            PolicyRevalidationEvidenceBuildError, "exact contract keys"
        ):
            self._build(candidate_policy_audit_paths=extra)

        cohort = _cohort_counts()
        del cohort["accepted_final_policy_result_count"]
        broken = self._write_audits(
            cohort_by_repeat=(cohort, cohort)
        )
        with self.assertRaisesRegex(
            PolicyRevalidationEvidenceBuildError,
            "exact required counters",
        ):
            self._build(candidate_policy_audit_paths=broken)

    def test_nondeterministic_audits_build_a_blocked_artifact(self):
        second_contract = _contract_counts()
        second_contract["signed_late_authority_recovery_count"] = 4
        audits = self._write_audits(
            contract_by_repeat=(
                _contract_counts(),
                second_contract,
            )
        )

        aggregate = self._build(candidate_policy_audit_paths=audits)

        self.assertEqual(aggregate["status"], "blocked")
        self.assertFalse(aggregate["audit_deterministic"])
        self.assertFalse(
            aggregate["gate_results"]["execution_audits_deterministic"]
        )

    def test_non_policy_field_drift_is_counted_and_blocks(self):
        changed_rows = [dict(row) for row in self.candidate_rows]
        changed_rows[0]["home_world"] = "Mars"
        paths = self._write_predictions("changed-candidate", changed_rows)
        observations = self._write_observations(
            "changed-observation", paths, source_sha=SOURCE_SHA
        )
        original = self.candidate_predictions
        self.candidate_predictions = paths
        try:
            audits = self._write_audits()
        finally:
            self.candidate_predictions = original

        aggregate = self._build(
            candidate_prediction_paths=paths,
            candidate_observation_paths=observations,
            candidate_policy_audit_paths=audits,
        )

        self.assertEqual(aggregate["status"], "blocked")
        self.assertEqual(aggregate["non_policy_field_change_count"], 1)
        self.assertFalse(
            aggregate["gate_results"]["non_policy_fields_unchanged"]
        )

    def test_cli_is_canonical_atomic_and_returns_gate_status(self):
        output_json = self.root / "evidence.json"
        output_markdown = self.root / "evidence.md"
        arguments = [
            "--layout-manifest",
            str(self.manifest),
            "--expected-layout-manifest-sha256",
            hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
            "--expected-input-tree-sha256",
            INPUT_TREE_SHA,
            "--truth",
            str(self.truth),
        ]
        for option, paths in (
            ("--legacy-control-prediction", self.control_predictions),
            ("--candidate-prediction", self.candidate_predictions),
            ("--legacy-control-observation", self.control_observations),
            ("--candidate-observation", self.candidate_observations),
            ("--candidate-policy-audit", self.audit_paths),
        ):
            for path in paths:
                arguments.extend([option, str(path)])
        arguments.extend(
            [
                "--legacy-control-source-revision-sha",
                LEGACY_SHA,
                "--source-revision-sha",
                SOURCE_SHA,
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_markdown),
            ]
        )
        with (
            patch(
                "devtools.policy_revalidation_evidence."
                "verify_clean_candidate_checkout"
            ),
            patch(
                "devtools.policy_revalidation_evidence.verify_legacy_commit"
            ),
        ):
            exit_code = main(arguments)

        self.assertEqual(exit_code, 0)
        parsed = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertEqual(
            output_json.read_text(encoding="utf-8"),
            canonical_json(parsed) + "\n",
        )
        self.assertEqual(
            output_markdown.read_text(encoding="utf-8"),
            render_aggregate_markdown(parsed),
        )


if __name__ == "__main__":
    unittest.main()
