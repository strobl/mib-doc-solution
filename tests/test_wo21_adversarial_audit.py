import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from devtools.wo21_adversarial_audit import (
    IdentityScan,
    ORACLE_DECOY_INVARIANT,
    REQUIRED_CATEGORIES,
    STATUS_BLOCKED_ENVIRONMENT,
    STATUS_BLOCKED_REGRESSION,
    WO21AuditError,
    build_external_corpus,
    identity_scan,
    render_markdown,
    require_identity_free_evidence,
    run_audit,
)


def valid_row(case_id, **changes):
    row = {
        "case_id": case_id,
        "applicant_name": "Astra Vale",
        "species_code": "ORION_GRAYS",
        "home_world": "Kepler-186f",
        "visa_class": "XW-2",
        "sponsor_id": "SPN-2468",
        "arrival_date": "2026-08-15",
        "declared_purpose": "research",
        "risk_flags": "none",
        "fee_status": "paid",
        "adjudication": "NEEDS_REVIEW",
        "confidence": 0.75,
    }
    row.update(changes)
    return row


class StableProcessor:
    def process_case(self, path):
        return valid_row(path.stem)


class DecoyProcessor:
    def process_case(self, path):
        if path.parent.name == "qr_barcode_prompt_injection":
            return valid_row(path.stem, sponsor_id="SPN-9999")
        return valid_row(path.stem)


class MissingProcessor:
    def process_case(self, path):
        if path.parent.name == "blur_destructive":
            return None
        return valid_row(path.stem)


class WO21CorpusTests(unittest.TestCase):
    def test_corpus_is_external_complete_and_byte_deterministic(self):
        with tempfile.TemporaryDirectory() as first_dir:
            first_golden, first_scenarios, first_sha = build_external_corpus(
                Path(first_dir)
            )
            with tempfile.TemporaryDirectory() as second_dir:
                second_golden, second_scenarios, second_sha = (
                    build_external_corpus(Path(second_dir))
                )

                self.assertEqual(first_sha, second_sha)
                self.assertEqual(
                    {item.category for item in first_scenarios},
                    set(REQUIRED_CATEGORIES),
                )
                self.assertEqual(len(first_scenarios), 15)
                self.assertEqual(
                    first_golden.read_bytes(),
                    second_golden.read_bytes(),
                )
                self.assertEqual(
                    [item.source_path.read_bytes() for item in first_scenarios],
                    [item.source_path.read_bytes() for item in second_scenarios],
                )
                by_name = {
                    item.scenario_name: item.source_path.read_bytes()
                    for item in first_scenarios
                }
                self.assertNotEqual(
                    by_name["foreign_applicant"],
                    by_name["identity_conflict"],
                )
                by_scenario = {
                    item.scenario_name: item
                    for item in first_scenarios
                }
                self.assertEqual(
                    by_scenario["foreign_applicant"].oracle,
                    ORACLE_DECOY_INVARIANT,
                )

    def test_repo_internal_corpus_path_fails_closed(self):
        repository_internal = (
            Path(__file__).resolve().parents[1] / "tmp" / "unsafe-wo21"
        )
        with self.assertRaisesRegex(WO21AuditError, "outside"):
            build_external_corpus(repository_internal)


class WO21EvidenceTests(unittest.TestCase):
    def build(self, processor, *, regression_filed_count=0):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        with patch(
            "devtools.wo21_adversarial_audit.identity_scan",
            return_value=IdentityScan(
                runtime_source_scan_count=27,
                model_artifact_scan_count=3,
                finding_count=0,
            ),
        ):
            return run_audit(
                external_root=Path(temporary.name),
                processor_factory=lambda: processor,
                docker_is_available=False,
                regression_filed_count=regression_filed_count,
            )

    def test_two_runs_cover_every_category_and_block_only_on_environment(self):
        build = self.build(StableProcessor())
        evidence = build.evidence

        self.assertEqual(evidence["status"], STATUS_BLOCKED_ENVIRONMENT)
        self.assertEqual(evidence["counts"]["category_count"], 13)
        self.assertEqual(evidence["counts"]["scenario_count"], 15)
        self.assertEqual(evidence["counts"]["host_run_count"], 2)
        self.assertEqual(evidence["counts"]["failed_scenario_count"], 0)
        self.assertEqual(evidence["counts"]["regression_waiver_count"], 0)
        self.assertTrue(evidence["checks"]["all_categories_exercised"])
        self.assertTrue(evidence["checks"]["host_runs_byte_identical"])
        self.assertTrue(evidence["checks"]["host_oracles_passed"])
        self.assertFalse(evidence["checks"]["docker_available"])
        self.assertEqual(
            evidence["host_run_one_sha256"],
            evidence["host_run_two_sha256"],
        )
        require_identity_free_evidence(evidence)

    def test_missing_required_category_is_a_hard_host_failure(self):
        def incomplete_corpus(external_root):
            golden, scenarios, corpus_sha = build_external_corpus(
                external_root
            )
            return (
                golden,
                tuple(
                    scenario
                    for scenario in scenarios
                    if scenario.category != "contrast"
                ),
                corpus_sha,
            )

        with patch(
            "devtools.wo21_adversarial_audit.build_external_corpus",
            side_effect=incomplete_corpus,
        ):
            build = self.build(StableProcessor())

        self.assertEqual(build.evidence["status"], STATUS_BLOCKED_REGRESSION)
        self.assertFalse(
            build.evidence["checks"]["all_categories_exercised"]
        )
        self.assertTrue(build.evidence["checks"]["host_oracles_passed"])
        overclaimed = json.loads(json.dumps(build.evidence))
        overclaimed["status"] = "pass"
        with self.assertRaisesRegex(
            WO21AuditError,
            "every required category",
        ):
            require_identity_free_evidence(overclaimed)

    def test_trusted_clean_docker_attestation_can_close_environment_gates(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        with patch(
            "devtools.wo21_adversarial_audit.identity_scan",
            return_value=IdentityScan(
                runtime_source_scan_count=27,
                model_artifact_scan_count=3,
                finding_count=0,
            ),
        ):
            build = run_audit(
                external_root=Path(temporary.name),
                processor_factory=lambda: StableProcessor(),
                docker_is_available=True,
                docker_reproducibility_verified=True,
                docker_runtime_verified=True,
                verified_source_revision="b" * 40,
                clean_checkout_verified=True,
                github_repository="example/mib-doc-solution",
                workflow_run_id=123456,
                workflow_run_attempt=2,
                wo20_aggregate_sha256="c" * 64,
            )

        self.assertEqual(build.evidence["status"], "pass")
        self.assertTrue(
            build.evidence["checks"][
                "source_revision_external_attestation_verified"
            ]
        )
        self.assertTrue(build.evidence["checks"]["source_revision_clean"])
        self.assertTrue(build.evidence["checks"]["docker_available"])
        self.assertTrue(
            build.evidence["checks"]["docker_reproducibility_verified"]
        )
        self.assertTrue(build.evidence["checks"]["docker_runtime_verified"])
        self.assertTrue(
            build.evidence["checks"][
                "workflow_attestation_provenance_verified"
            ]
        )
        self.assertEqual(
            build.evidence["attestation_provenance"][
                "wo20_aggregate_sha256"
            ],
            "c" * 64,
        )

    def test_docker_attestation_without_wo20_provenance_fails_closed(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        with patch(
            "devtools.wo21_adversarial_audit.identity_scan",
            return_value=IdentityScan(
                runtime_source_scan_count=27,
                model_artifact_scan_count=3,
                finding_count=0,
            ),
        ), self.assertRaisesRegex(
            WO21AuditError,
            "WO20 aggregate provenance",
        ):
            run_audit(
                external_root=Path(temporary.name),
                processor_factory=lambda: StableProcessor(),
                docker_is_available=True,
                docker_reproducibility_verified=True,
                docker_runtime_verified=True,
                verified_source_revision="b" * 40,
                clean_checkout_verified=True,
            )

    def test_external_source_revision_fails_closed_without_clean_attestation(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(
            WO21AuditError,
            "clean-checkout",
        ):
            run_audit(
                external_root=Path(temporary.name),
                processor_factory=lambda: StableProcessor(),
                verified_source_revision="b" * 40,
            )

    def test_decoy_adoption_is_counted_without_identity_bearing_details(self):
        build = self.build(DecoyProcessor(), regression_filed_count=1)
        evidence = build.evidence

        self.assertEqual(evidence["status"], STATUS_BLOCKED_REGRESSION)
        self.assertGreater(evidence["counts"]["failed_scenario_count"], 0)
        self.assertEqual(evidence["counts"]["decoy_adoption_count"], 1)
        self.assertEqual(evidence["counts"]["regression_filed_count"], 1)
        self.assertEqual(
            evidence["field_metrics"]["qr_barcode_prompt_injection"][
                "failed_count"
            ],
            1,
        )
        self.assertNotIn("SPN-9999", json.dumps(evidence))

    def test_missing_destructive_record_fails_closed(self):
        build = self.build(MissingProcessor())
        evidence = build.evidence

        self.assertEqual(evidence["status"], STATUS_BLOCKED_REGRESSION)
        self.assertEqual(evidence["counts"]["missing_record_count"], 1)
        self.assertGreater(evidence["regression_counts"]["blur"], 0)
        self.assertFalse(evidence["checks"]["host_oracles_passed"])

    def test_evidence_rejects_case_pdf_sequences_and_waivers(self):
        build = self.build(StableProcessor())
        safe = json.loads(json.dumps(build.evidence))
        unsafe_values = []

        case_identity = json.loads(json.dumps(safe))
        case_identity["comparison_scope"] = "MIB-" + "1" * 6
        unsafe_values.append(case_identity)

        pdf_identity = json.loads(json.dumps(safe))
        pdf_identity["comparison_scope"] = "packet" + ".pdf"
        unsafe_values.append(pdf_identity)

        per_case_list = json.loads(json.dumps(safe))
        per_case_list["field_metrics"]["hidden_text"]["details"] = [1]
        unsafe_values.append(per_case_list)

        waiver = json.loads(json.dumps(safe))
        waiver["counts"]["regression_waiver_count"] = 1
        unsafe_values.append(waiver)

        overclaimed_filing = json.loads(json.dumps(safe))
        overclaimed_filing["counts"]["regression_filed_count"] = 1
        unsafe_values.append(overclaimed_filing)

        for value in unsafe_values:
            with self.subTest(value=value), self.assertRaises(WO21AuditError):
                require_identity_free_evidence(value)

    def test_markdown_is_aggregate_only(self):
        build = self.build(StableProcessor())
        markdown = render_markdown(build.evidence)

        self.assertIn("blocked_environment", markdown)
        self.assertIn("hidden_text", markdown)
        self.assertNotRegex(markdown, r"MIB-[0-9]{6}")
        self.assertNotRegex(markdown, r"(?i)\b[^\s/\\]+\.pdf\b")
        self.assertNotIn("Astra Vale", markdown)


class WO21IdentityScanTests(unittest.TestCase):
    def test_installed_scope_scans_onnx_traineddata_and_runtime_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roots = (
                root / "app",
                root / "opt",
                root / "site-packages",
                root / "tesseract",
            )
            for scan_root in roots:
                scan_root.mkdir(parents=True)
            runtime_artifacts = roots[0] / "mib_pipeline" / "artifacts"
            runtime_artifacts.mkdir(parents=True)
            (runtime_artifacts / "policy.json").write_text(
                '{"policy":"safe"}',
                encoding="utf-8",
            )
            (roots[2] / "detector.onnx").write_bytes(b"safe-model")
            (roots[3] / "eng.traineddata").write_bytes(
                b"unsafe-token:MIB-123456"
            )
            (roots[2] / "unrelated.json").write_text(
                '{"ignored":"MIB-654321"}',
                encoding="utf-8",
            )

            scan = identity_scan(installed_roots=roots)

        self.assertEqual(scan.model_artifact_scan_count, 3)
        self.assertGreaterEqual(scan.finding_count, 1)

    def test_workflow_hard_gates_coverage_and_wo20_provenance(self):
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "wo20-runtime.yml"
        ).read_text(encoding="utf-8")

        self.assertIn(
            ".checks.all_categories_exercised == true",
            workflow,
        )
        self.assertIn(
            ".checks.workflow_attestation_provenance_verified == true",
            workflow,
        )
        self.assertIn(".checks.model_identity_scan_clean == true", workflow)
        self.assertIn("--wo20-aggregate-sha256", workflow)
        self.assertIn("sha256sum \"$wo20_aggregate\"", workflow)


if __name__ == "__main__":
    unittest.main()
