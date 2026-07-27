import concurrent.futures
import copy
import hashlib
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

from devtools.experiment_control import (
    GENESIS_HASH,
    BudgetExhaustedError,
    CandidatePromotionGate,
    CandidateStateStore,
    CanonicalHashChainStore,
    CompareAndSwapError,
    ExperimentControlError,
    ExperimentLedger,
    FrozenBaselineManifest,
    IntegrityError,
    LeakageError,
    ProtectedAccessBudget,
    RepeatedGroupedSplitManager,
    RuntimeLeakageScanner,
    TaintRegistry,
    build_retrospective_reconciliation,
    canonical_json,
    require_retrospective_reconciliation,
    verify_retrospective_reconciliation,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


class TemporaryDirectoryTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()


class CanonicalHashChainStoreTests(TemporaryDirectoryTestCase):
    def test_append_is_canonical_chained_and_verifiable(self):
        store = CanonicalHashChainStore(self.root / "ledger.jsonl")

        first = store.append({"z": 1, "a": {"value": True}})
        second = store.append({"event": "next"}, expected_head=first["record_hash"])

        records = store.verify(
            expected_head=second["record_hash"], expected_length=2
        )
        self.assertEqual(records[0]["previous_hash"], GENESIS_HASH)
        self.assertEqual(records[1]["previous_hash"], first["record_hash"])
        self.assertEqual(store.head, second["record_hash"])
        raw_lines = store.path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(raw_lines, [canonical_json(record) for record in records])

    def test_compare_and_swap_rejects_stale_writer(self):
        store = CanonicalHashChainStore(self.root / "ledger.jsonl")
        first = store.append({"event": "first"})

        with self.assertRaises(CompareAndSwapError):
            store.append({"event": "stale"}, expected_head=GENESIS_HASH)

        self.assertEqual(store.verify(expected_head=first["record_hash"])[0], first)

    def test_tamper_is_detected(self):
        store = CanonicalHashChainStore(self.root / "ledger.jsonl")
        store.append({"score": 130.37})
        raw = store.path.read_text(encoding="utf-8")
        store.path.write_text(raw.replace("130.37", "148.0"), encoding="utf-8")

        with self.assertRaises(IntegrityError):
            store.verify()

    def test_partial_and_valid_prefix_truncation_are_detected(self):
        store = CanonicalHashChainStore(self.root / "ledger.jsonl")
        first = store.append({"event": "first"})
        second = store.append({"event": "second"})
        lines = store.path.read_text(encoding="utf-8").splitlines(keepends=True)

        store.path.write_text(lines[0], encoding="utf-8")
        with self.assertRaises(IntegrityError):
            store.verify(
                expected_head=second["record_hash"],
                expected_length=2,
            )
        self.assertEqual(
            store.verify(expected_head=first["record_hash"], expected_length=1)[0],
            first,
        )

        store.path.write_text(lines[0][:-1], encoding="utf-8")
        with self.assertRaises(IntegrityError):
            store.verify()


class ExperimentLedgerTests(TemporaryDirectoryTestCase):
    @staticmethod
    def plan(**overrides):
        plan = {
            "changed_files": [
                "mib_pipeline/resolution.py",
                "tests/test_resolution.py",
            ],
            "evidence_label": "public_grouped_robustness_not_unseen",
            "hypothesis_sha256": SHA_A,
            "parent_commit_sha": "c" * 40,
            "primary_variable_sha256": SHA_B,
            "split_manifest_sha256": SHA_A,
        }
        plan.update(overrides)
        return plan

    @staticmethod
    def result_evidence(**overrides):
        evidence = {
            "baseline_artifact_sha256": SHA_A,
            "candidate_artifact_sha256": SHA_B,
            "checks": {
                "decision_freeze_verified": True,
                "deterministic": True,
                "fold_consistent": True,
                "runtime_leakage_clean": True,
            },
            "confusion_counts": {"approved_to_review_count": 0},
            "field_metrics": {
                "all_fields": {"count": 1, "score_delta": 0.0}
            },
            "metrics": {
                "calibration_score": 18.0,
                "candidate_image_bytes": 1024,
                "candidate_model_bytes": 512,
                "catastrophic_false_approvals": 0,
                "classification_score": 72.0,
                "extraction_score": 46.0,
                "invalid_records": 0,
                "missing_records": 0,
                "output_bytes": 256,
                "peak_rss_bytes": 2048,
                "process_cpu_seconds": 2.0,
                "runtime_seconds": 3.0,
                "tmp_bytes": 128,
                "total_score": 136.0,
            },
            "fold_count": 5,
            "fold_deltas": [
                0.3,
                0.15,
                0.0,
                0.0,
                0.0,
                0.15,
                0.3,
                0.0,
                0.0,
                0.0,
                0.2,
                0.1,
                0.0,
                0.0,
                0.0,
            ],
            "regression_counts": {"adversarial": 0, "golden": 0},
            "repeat_count": 3,
            "repeat_scores": [0.09, 0.09, 0.06],
        }
        evidence.update(overrides)
        return evidence

    def test_records_aggregate_evidence_and_makes_identical_retry_idempotent(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        evidence = {
            "total_score": 130.37,
            "confusion_counts": {"APPROVED_TO_REVIEW": 115},
        }
        stored = ledger.store.append(
            {
                "event": "experiment",
                "experiment_id": "exp-001",
                "evidence": evidence,
            }
        )
        first = ledger.record(
            "exp-001",
            evidence,
        )
        retry = ledger.record(
            "exp-001",
            evidence,
        )

        self.assertEqual(first, stored)
        self.assertEqual(first, retry)
        self.assertEqual(ledger.experiments()[0]["experiment_id"], "exp-001")
        with self.assertRaises(ExperimentControlError):
            ledger.record("exp-001", {"total_score": 131.0})
        with self.assertRaises(ExperimentControlError):
            ledger.record("new-legacy", {"total_score": 131.0})
        self.assertEqual(ledger.store.length, 1)

    def test_rejects_nested_case_ids_filenames_and_case_level_keys(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        unsafe_values = (
            {"notes": ["MIB-000042 was wrong"]},
            {"artifact": "train/MIB-000042.pdf"},
            {"case_scores": {"anything": 1}},
            {"validation_case_ids": ["opaque-public-row"]},
        )
        for index, value in enumerate(unsafe_values):
            with self.subTest(index=index), self.assertRaises(LeakageError):
                ledger.record(f"exp-{index}", value)

    def test_strict_schema_rejects_record_shapes_and_retains_aggregate_hashes(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        unsafe_values = (
            {"rows": [{"score": 1.0}]},
            {"samples": [{"score": 1.0}]},
            {"outcomes": {"approved": 1}},
            {"truth_pred_pairs": [["APPROVED", "DENIED"]]},
            {"metrics": {"bucket": {"filename": "document.pdf"}}},
            {"field_metrics": {"case_0001": {"score": 1.0}}},
            {"metrics": {"outcomes": 1}},
            {"file_sha256": SHA_A},
            {"document_id": "opaque"},
        )
        for index, value in enumerate(unsafe_values):
            with self.subTest(index=index), self.assertRaises(LeakageError):
                ledger.record(f"unsafe-{index}", value)

        safe_evidence = {
            "artifact_sha256": SHA_A,
            "fold_scores": [129.0, 130.0, 131.0],
            "field_metrics": {
                "risk": {"accuracy": 0.9, "error_count": 10}
            },
        }
        ledger.store.append(
            {
                "event": "experiment",
                "experiment_id": "safe",
                "evidence": safe_evidence,
            }
        )
        ledger.record("safe", safe_evidence)

    def test_preregister_then_record_result_are_distinct_and_plan_bound(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")

        plan_record = ledger.preregister(
            "exp-two-stage",
            self.plan(changed_files=list(reversed(self.plan()["changed_files"]))),
            expected_head=GENESIS_HASH,
        )
        result_record = ledger.record_result(
            "exp-two-stage",
            self.result_evidence(),
            decision="reject",
            rationale="grouped_gate_failed",
            expected_head=plan_record["record_hash"],
        )

        self.assertEqual(plan_record["payload"]["event"], "experiment_plan")
        self.assertEqual(result_record["payload"]["event"], "experiment_result")
        self.assertEqual(
            result_record["payload"]["plan_record_hash"],
            plan_record["record_hash"],
        )
        self.assertEqual(
            plan_record["payload"]["plan"]["changed_files"],
            sorted(self.plan()["changed_files"]),
        )
        self.assertEqual(result_record["previous_hash"], plan_record["record_hash"])
        self.assertEqual(len(ledger.plans()), 1)
        self.assertEqual(len(ledger.results()), 1)
        self.assertEqual(ledger.experiments(), ())

    def test_two_stage_order_conflicts_and_duplicate_result_fail_closed(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        evidence = self.result_evidence()

        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "exp-two-stage",
                evidence,
                decision="reject",
                rationale="missing_plan",
            )

        first_plan = ledger.preregister("exp-two-stage", self.plan())
        identical_retry = ledger.preregister("exp-two-stage", self.plan())
        self.assertEqual(first_plan, identical_retry)
        with self.assertRaises(ExperimentControlError):
            ledger.preregister(
                "exp-two-stage",
                self.plan(hypothesis_sha256="d" * 64),
            )
        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "exp-two-stage",
                evidence,
                decision="reject",
                rationale="A free-form result narrative could leak diagnostics.",
            )

        ledger.record_result(
            "exp-two-stage",
            evidence,
            decision="reject",
            rationale="grouped_gate_failed",
        )
        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "exp-two-stage",
                evidence,
                decision="reject",
                rationale="grouped_gate_failed",
            )
        self.assertEqual(ledger.store.length, 2)

    def test_result_requires_full_contract_and_adoption_requires_clean_gates(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("strict-result", self.plan())

        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "strict-result",
                {"status": "failed"},
                decision="adopt",
                rationale="insufficient_contract",
            )
        unsafe = self.result_evidence()
        unsafe["metrics"]["catastrophic_false_approvals"] = 1
        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "strict-result",
                unsafe,
                decision="adopt",
                rationale="unsafe_candidate",
            )
        false_extra_check = self.result_evidence()
        false_extra_check["checks"]["baseline_verified"] = False
        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "strict-result",
                false_extra_check,
                decision="adopt",
                rationale="false_extra_check",
            )
        concentrated = self.result_evidence(
            fold_deltas=[
                0.3,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.3,
                0.0,
                0.0,
            ],
            repeat_scores=[0.06, 0.06, 0.06],
        )
        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "strict-result",
                concentrated,
                decision="adopt",
                rationale="single_fold_concentration",
            )
        inconsistent_total = self.result_evidence()
        inconsistent_total["metrics"]["total_score"] = 135.0
        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "strict-result",
                inconsistent_total,
                decision="reject",
                rationale="component_mismatch",
            )

        accepted = ledger.record_result(
            "strict-result",
            self.result_evidence(),
            decision="adopt",
            rationale="all_hard_gates_passed",
        )
        self.assertEqual(accepted["payload"]["decision"], "adopt")
        self.assertEqual(ledger.store.length, 2)

    def test_protected_result_requires_matching_access_record_and_candidate(self):
        protected_path = self.root / "protected.jsonl"
        budget = ProtectedAccessBudget(protected_path, maximum_accesses=2)
        recorded_evidence = self.result_evidence()
        budget.record_access(
            "milestone-access",
            candidate_sha256=SHA_B,
            aggregate_result=recorded_evidence,
            purpose="milestone-candidate",
        )
        access_hash = budget.store.verify()[-1]["record_hash"]
        protected_plan = self.plan(evidence_label="protected")

        missing_store = ExperimentLedger(self.root / "missing-store.jsonl")
        missing_store.preregister("protected", protected_plan)
        with self.assertRaises(ExperimentControlError):
            missing_store.record_result(
                "protected",
                self.result_evidence(
                    protected_access_record_hash=access_hash
                ),
                decision="reject",
                rationale="protected_gate_rejected",
            )

        ledger = ExperimentLedger(
            self.root / "experiments.jsonl",
            protected_access_path=protected_path,
        )
        ledger.preregister("protected", protected_plan)
        wrong_candidate = self.result_evidence(
            candidate_artifact_sha256=SHA_A,
            protected_access_record_hash=access_hash,
        )
        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "protected",
                wrong_candidate,
                decision="reject",
                rationale="candidate_binding_mismatch",
            )
        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "protected",
                self.result_evidence(
                    protected_access_record_hash="d" * 64
                ),
                decision="reject",
                rationale="access_binding_missing",
            )
        invented_aggregate = self.result_evidence(
            protected_access_record_hash=access_hash
        )
        invented_aggregate["field_metrics"]["all_fields"]["count"] = 2
        with self.assertRaises(ExperimentControlError):
            ledger.record_result(
                "protected",
                invented_aggregate,
                decision="reject",
                rationale="unrecorded_aggregate",
            )

        result = ledger.record_result(
            "protected",
            self.result_evidence(protected_access_record_hash=access_hash),
            decision="reject",
            rationale="protected_gate_rejected",
        )
        self.assertEqual(
            result["payload"]["evidence"]["protected_access_record_hash"],
            access_hash,
        )
        self.assertEqual(ledger.store.length, 2)

    def test_protected_result_rejects_ledger_without_budget_configuration(self):
        protected_path = self.root / "forged-protected.jsonl"
        aggregate_result = self.result_evidence()
        forged_record = CanonicalHashChainStore(protected_path).append(
            {
                "access_id": "forged",
                "aggregate_result": aggregate_result,
                "candidate_sha256": SHA_B,
                "event": "protected_access",
                "purpose": "forged-access",
            }
        )
        ledger = ExperimentLedger(
            self.root / "experiments.jsonl",
            protected_access_path=protected_path,
        )
        ledger.preregister(
            "protected-forgery",
            self.plan(evidence_label="protected"),
        )

        with self.assertRaises(IntegrityError):
            ledger.record_result(
                "protected-forgery",
                self.result_evidence(
                    protected_access_record_hash=forged_record["record_hash"]
                ),
                decision="reject",
                rationale="forged_access",
            )

    def test_plan_schema_rejects_identity_and_non_normalized_changed_files(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        unsafe_plans = (
            {
                **self.plan(),
                "hypothesis": "Protected applicant diagnostic prose.",
            },
            self.plan(hypothesis_sha256="not-a-hash"),
            self.plan(changed_files=["../mib_pipeline/resolution.py"]),
            self.plan(changed_files=["mib_pipeline//resolution.py"]),
            self.plan(changed_files=["fixtures/special.pdf"]),
            self.plan(changed_files=[]),
            self.plan(parent_commit_sha=SHA_A),
            self.plan(evidence_label="passed"),
        )

        for index, plan in enumerate(unsafe_plans):
            with self.subTest(index=index), self.assertRaises(
                ExperimentControlError
            ):
                ledger.preregister(f"unsafe-plan-{index}", plan)
        self.assertEqual(ledger.store.length, 0)

    def test_legacy_and_two_stage_ids_cannot_collide(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.store.append(
            {
                "event": "experiment",
                "experiment_id": "legacy",
                "evidence": {"total_score": 130.37},
            }
        )
        with self.assertRaises(ExperimentControlError):
            ledger.preregister("legacy", self.plan())

        ledger.preregister("planned", self.plan())
        with self.assertRaises(ExperimentControlError):
            ledger.record("planned", {"total_score": 131.0})
        self.assertEqual(len(ledger.experiments()), 1)
        self.assertEqual(len(ledger.plans()), 1)

    def test_concurrent_result_writers_append_exactly_one_result(self):
        path = self.root / "experiments.jsonl"
        ExperimentLedger(path).preregister("concurrent", self.plan())
        barrier = threading.Barrier(2)

        def attempt(index):
            ledger = ExperimentLedger(path)
            barrier.wait(timeout=5)
            try:
                ledger.record_result(
                    "concurrent",
                    self.result_evidence(
                        metrics={
                            **self.result_evidence()["metrics"],
                            "calibration_score": 18.0 + index,
                            "total_score": 136.0 + index,
                        }
                    ),
                    decision="reject",
                    rationale=f"writer_{index}_gate_failed",
                )
                return "accepted"
            except ExperimentControlError:
                return "rejected"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, range(2)))

        self.assertEqual(outcomes.count("accepted"), 1)
        self.assertEqual(outcomes.count("rejected"), 1)
        reopened = ExperimentLedger(path)
        self.assertEqual(len(reopened.plans()), 1)
        self.assertEqual(len(reopened.results()), 1)
        self.assertEqual(reopened.store.length, 2)


class RetrospectiveReconciliationTests(unittest.TestCase):
    REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
    ARTIFACT_PATH = (
        REPOSITORY_ROOT
        / "evaluation/program/RETROSPECTIVE_RECONCILIATION_WO15_WO21.json"
    )

    @staticmethod
    def build():
        root = RetrospectiveReconciliationTests.REPOSITORY_ROOT
        return build_retrospective_reconciliation(
            integrity_heads_path=(
                root / "evaluation/program/integrity_heads.json"
            ),
            evidence_paths={
                "wo15": root / "evaluation/WO15_GROUPED_RECOVERY_EVIDENCE.json",
                "wo16": root / "evaluation/WO16_GROUPED_FUSION_EVIDENCE.json",
                "wo17": root / "evaluation/WO17_POLICY_REVALIDATION_EVIDENCE.json",
                "wo18": root / "evaluation/WO18_DECISION_RECOVERY_EVIDENCE.json",
                "wo19": root / "evaluation/WO19_CONFIDENCE_REFIT_EVIDENCE.json",
            },
        )

    def test_builder_is_non_authoritative_and_canonical(self):
        artifact = self.build()

        require_retrospective_reconciliation(artifact)
        verify_retrospective_reconciliation(
            artifact,
            repository_root=self.REPOSITORY_ROOT,
        )
        self.assertEqual(
            artifact["classification"], "retrospective_not_preregistered"
        )
        self.assertTrue(all(not value for value in artifact["authority"].values()))
        self.assertEqual(
            artifact["work_order_dispositions"]["wo15"],
            "rejected_single_fold_concentration",
        )
        self.assertEqual(
            artifact["work_order_dispositions"]["wo19"],
            "evaluated_no_promotion",
        )
        self.assertEqual(
            artifact["evidence_files"]["wo15"],
            {
                "path": "evaluation/WO15_GROUPED_RECOVERY_EVIDENCE.json",
                "sha256": (
                    "25d01bbae8b0ce4d96de5906fd53b77a7"
                    "d6066dc0af2ad5e8ec76d8f48185dae"
                ),
            },
        )
        self.assertEqual(
            artifact["latest_governed_passing_candidate"]["total_score"],
            130.37185423344323,
        )
        self.assertEqual(json.loads(canonical_json(artifact)), artifact)

    def test_committed_reconciliation_exactly_matches_verified_repository(self):
        committed_bytes = self.ARTIFACT_PATH.read_text(encoding="utf-8")
        committed = json.loads(committed_bytes)
        generated = self.build()

        self.assertEqual(committed_bytes, canonical_json(committed) + "\n")
        self.assertEqual(committed, generated)
        verify_retrospective_reconciliation(
            committed,
            repository_root=self.REPOSITORY_ROOT,
        )
        self.assertFalse(committed["authority"]["preregistered"])
        self.assertFalse(committed["authority"]["promotion_authority"])
        self.assertFalse(committed["authority"]["protected_access_consumed"])
        self.assertFalse(committed["authority"]["baseline_state_changed"])
        self.assertFalse(committed["authority"]["candidate_state_changed"])

    def test_validator_rejects_authority_and_evidence_fabrication(self):
        artifact = self.build()
        invalid_values = []
        promoted = copy.deepcopy(artifact)
        promoted["authority"]["candidate_promotion_recorded"] = True
        invalid_values.append(promoted)
        protected = copy.deepcopy(artifact)
        protected["authority"]["new_protected_access_recorded"] = True
        invalid_values.append(protected)
        preregistered = copy.deepcopy(artifact)
        preregistered["authority"]["preregistered"] = True
        invalid_values.append(preregistered)
        missing_evidence = copy.deepcopy(artifact)
        del missing_evidence["evidence_files"]["wo19"]
        invalid_values.append(missing_evidence)
        changed_disposition = copy.deepcopy(artifact)
        changed_disposition["work_order_dispositions"]["wo15"] = "passed"
        invalid_values.append(changed_disposition)

        for index, invalid in enumerate(invalid_values):
            with self.subTest(index=index), self.assertRaises(
                ExperimentControlError
            ):
                require_retrospective_reconciliation(invalid)

    def test_verifier_rejects_syntactically_valid_false_file_binding(self):
        artifact = self.build()
        artifact["evidence_files"]["wo15"]["sha256"] = SHA_A

        require_retrospective_reconciliation(artifact)
        with self.assertRaises(IntegrityError):
            verify_retrospective_reconciliation(
                artifact,
                repository_root=self.REPOSITORY_ROOT,
            )

    def test_verifier_rejects_anchor_that_contradicts_bound_integrity_heads(self):
        artifact = self.build()
        with tempfile.TemporaryDirectory() as directory:
            repository_root = Path(directory)
            shutil.copytree(
                self.REPOSITORY_ROOT / "evaluation",
                repository_root / "evaluation",
            )
            ledger_path = (
                repository_root / "evaluation/program/experiment_ledger.jsonl"
            )
            store = CanonicalHashChainStore(ledger_path)
            store.append(
                {
                    "event": "experiment",
                    "experiment_id": "contradiction-probe",
                    "evidence": {"total_score": 130.0},
                }
            )
            anchor = artifact["ledger_anchors"]["experiment_ledger"]
            anchor["expected_head"] = store.head
            anchor["expected_length"] = store.length
            anchor["sha256"] = hashlib.sha256(ledger_path.read_bytes()).hexdigest()

            require_retrospective_reconciliation(artifact)
            with self.assertRaises(IntegrityError):
                verify_retrospective_reconciliation(
                    artifact,
                    repository_root=repository_root,
                )


class TaintRegistryTests(TemporaryDirectoryTestCase):
    def test_taint_is_append_only_persistent_and_cannot_be_reversed(self):
        registry = TaintRegistry(self.root / "taint.jsonl")
        registry.taint("layout-family-07", reason="inspected", source="debug-session")
        reopened = TaintRegistry(registry.store.path)

        self.assertEqual(reopened.tainted_groups(), {"layout-family-07"})
        with self.assertRaises(ExperimentControlError):
            reopened.untaint("layout-family-07")
        self.assertEqual(reopened.tainted_groups(), {"layout-family-07"})


class FrozenBaselineManifestTests(TemporaryDirectoryTestCase):
    def test_create_is_idempotent_and_verify_checks_path_size_and_sha(self):
        artifact = self.root / "baseline.json"
        artifact.write_text('{"score":130.37}\n', encoding="utf-8")
        frozen = FrozenBaselineManifest(self.root / "baseline.manifest.json")
        artifacts = {"evaluation/baseline.json": artifact}

        first = frozen.create(artifacts, metadata={"total_score": 130.37})
        second = frozen.create(artifacts, metadata={"total_score": 130.37})

        self.assertEqual(first, second)
        self.assertEqual(
            first["artifacts"][0],
            {
                "path": "evaluation/baseline.json",
                "size_bytes": len(artifact.read_bytes()),
                "sha256": __import__("hashlib").sha256(artifact.read_bytes()).hexdigest(),
            },
        )
        self.assertEqual(frozen.verify(artifacts), first)

    def test_changed_artifact_and_changed_manifest_request_are_rejected(self):
        artifact = self.root / "baseline.json"
        artifact.write_text("v1", encoding="utf-8")
        frozen = FrozenBaselineManifest(self.root / "baseline.manifest.json")
        artifacts = {"baseline.json": artifact}
        frozen.create(artifacts)

        artifact.write_text("v2", encoding="utf-8")
        with self.assertRaises(IntegrityError):
            frozen.verify(artifacts)
        with self.assertRaises(IntegrityError):
            frozen.create(artifacts)

    def test_noncanonical_manifest_edit_is_detected(self):
        artifact = self.root / "baseline.json"
        artifact.write_text("v1", encoding="utf-8")
        frozen = FrozenBaselineManifest(self.root / "baseline.manifest.json")
        manifest = frozen.create({"baseline.json": artifact})
        frozen.path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        with self.assertRaises(IntegrityError):
            frozen.load()


class RepeatedGroupedSplitManagerTests(unittest.TestCase):
    GROUPS = {
        "layout-a": ("case-a1", "case-a2"),
        "layout-b": ("case-b1",),
        "layout-c": ("case-c1", "case-c2"),
        "layout-d": ("case-d1",),
        "layout-e": ("case-e1",),
        "layout-f": ("case-f1",),
    }

    def test_splits_are_deterministic_group_exclusive_repeated_and_taint_free(self):
        manager = RepeatedGroupedSplitManager(seed="wo12-v1", repeats=2, folds=2)

        first = manager.split_groups(
            self.GROUPS, tainted_groups={"layout-f", "unknown-group"}
        )
        second = manager.split_groups(
            dict(reversed(tuple(self.GROUPS.items()))),
            tainted_groups={"layout-f", "unknown-group"},
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        for split in first:
            self.assertFalse(
                set(split.tuning_groups) & set(split.validation_groups)
            )
            self.assertFalse(
                set(split.tuning_case_ids) & set(split.validation_case_ids)
            )
            self.assertNotIn("layout-f", split.tuning_groups)
            self.assertNotIn("layout-f", split.validation_groups)
        for repeat in range(2):
            validation_groups = [
                set(split.validation_groups)
                for split in first
                if split.repeat == repeat
            ]
            self.assertEqual(set.union(*validation_groups), set(self.GROUPS) - {"layout-f"})
            self.assertFalse(set.intersection(*validation_groups))

    def test_rejects_cross_group_case_and_insufficient_eligible_groups(self):
        manager = RepeatedGroupedSplitManager(seed="wo12-v1", folds=2)
        with self.assertRaises(ExperimentControlError):
            manager.split_groups({"a": ["shared"], "b": ["shared"]})
        with self.assertRaises(ExperimentControlError):
            manager.split_groups(
                {"a": ["a1"], "b": ["b1"]}, tainted_groups={"b"}
            )

    def test_split_rows_supports_explicit_layout_family_keys(self):
        rows = [
            {"document": "a1", "layout_family": "a"},
            {"document": "a2", "layout_family": "a"},
            {"document": "b1", "layout_family": "b"},
        ]
        splits = RepeatedGroupedSplitManager(
            seed="row-api", repeats=1, folds=2
        ).split_rows(
            rows,
            group_key="layout_family",
            case_key="document",
        )

        self.assertEqual(len(splits), 2)
        self.assertEqual(
            set(splits[0].validation_case_ids) | set(splits[1].validation_case_ids),
            {"a1", "a2", "b1"},
        )


class ProtectedAccessBudgetTests(TemporaryDirectoryTestCase):
    def test_identical_retry_is_idempotent_and_does_not_consume_budget(self):
        budget = ProtectedAccessBudget(
            self.root / "protected.jsonl", maximum_accesses=2
        )
        request = {
            "candidate_sha256": SHA_A,
            "aggregate_result": {"total_score": 136.2, "count": 1000},
            "purpose": "milestone-136",
        }

        first = budget.record_access("access-1", **request)
        retry = budget.record_access("access-1", **request)

        self.assertEqual(first, retry)
        self.assertEqual(budget.used, 1)
        self.assertEqual(budget.remaining, 1)
        self.assertEqual(budget.store.length, 2)
        self.assertEqual(
            budget.store.verify()[0]["payload"],
            {
                "event": "protected_budget_configuration",
                "maximum_accesses": 2,
            },
        )

    def test_mismatched_retry_is_rejected(self):
        budget = ProtectedAccessBudget(
            self.root / "protected.jsonl", maximum_accesses=2
        )
        budget.record_access(
            "access-1",
            candidate_sha256=SHA_A,
            aggregate_result={"total_score": 136.2},
            purpose="milestone-136",
        )

        with self.assertRaises(ExperimentControlError):
            budget.record_access(
                "access-1",
                candidate_sha256=SHA_B,
                aggregate_result={"total_score": 136.2},
                purpose="milestone-136",
            )

    def test_exhaustion_and_case_level_results_are_rejected(self):
        budget = ProtectedAccessBudget(
            self.root / "protected.jsonl", maximum_accesses=1
        )
        with self.assertRaises(LeakageError):
            budget.record_access(
                "unsafe",
                candidate_sha256=SHA_A,
                aggregate_result={"case_id": "MIB-000001"},
                purpose="debug",
            )
        budget.record_access(
            "access-1",
            candidate_sha256=SHA_A,
            aggregate_result={"total_score": 142.0},
            purpose="milestone-142",
        )
        with self.assertRaises(BudgetExhaustedError):
            budget.record_access(
                "access-2",
                candidate_sha256=SHA_B,
                aggregate_result={"total_score": 146.0},
                purpose="milestone-146",
            )

    def test_configuration_is_persisted_and_immutable(self):
        path = self.root / "protected.jsonl"
        ProtectedAccessBudget(path, maximum_accesses=2)

        with self.assertRaises(IntegrityError):
            ProtectedAccessBudget(path, maximum_accesses=3)

        reopened = ProtectedAccessBudget(path, maximum_accesses=2)
        self.assertEqual(reopened.maximum_accesses, 2)
        self.assertEqual(reopened.remaining, 2)

    def test_concurrent_budget_one_accepts_exactly_one_access(self):
        path = self.root / "protected.jsonl"
        workers = 8
        barrier = threading.Barrier(workers)

        def attempt(index):
            budget = ProtectedAccessBudget(path, maximum_accesses=1)
            barrier.wait(timeout=5)
            try:
                budget.record_access(
                    f"access-{index}",
                    candidate_sha256=SHA_A if index % 2 == 0 else SHA_B,
                    aggregate_result={"total_score": 130.0 + index / 100},
                    purpose=f"milestone-{index}",
                )
                return "accepted"
            except BudgetExhaustedError:
                return "exhausted"

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(attempt, range(workers)))

        self.assertEqual(results.count("accepted"), 1)
        self.assertEqual(results.count("exhausted"), workers - 1)
        reopened = ProtectedAccessBudget(path, maximum_accesses=1)
        self.assertEqual(reopened.used, 1)
        self.assertEqual(reopened.store.length, 2)


class RuntimeLeakageScannerTests(TemporaryDirectoryTestCase):
    def test_scans_python_and_json_for_all_forbidden_runtime_patterns(self):
        python_path = self.root / "runtime.py"
        python_path.write_text(
            'SPECIAL = "MIB-000042"\n'
            'PDF = "inputs/packet-000042.pdf"\n'
            'labels_by_case = {"x": "DENIED"}\n',
            encoding="utf-8",
        )
        json_path = self.root / "artifact.json"
        json_path.write_text(
            json.dumps(
                {
                    "case_label_lookup": {"opaque": "APPROVED"},
                    "source": "special-packet.pdf",
                    "MIB-000043": "DENIED",
                }
            ),
            encoding="utf-8",
        )

        findings = RuntimeLeakageScanner().scan([self.root])
        codes = [finding.code for finding in findings]

        self.assertEqual(codes.count("MIB_CASE_ID"), 2)
        self.assertEqual(codes.count("PDF_FILENAME"), 2)
        self.assertEqual(codes.count("CASE_LABEL_LOOKUP"), 2)
        with self.assertRaises(LeakageError):
            RuntimeLeakageScanner().require_clean([self.root])

    def test_exact_file_and_code_allowlist_is_narrow(self):
        allowed = self.root / "allowed.py"
        blocked = self.root / "blocked.py"
        allowed.write_text('EXAMPLE = "MIB-000000"\n', encoding="utf-8")
        blocked.write_text('EXAMPLE = "MIB-000001"\n', encoding="utf-8")
        scanner = RuntimeLeakageScanner(
            allowlist={allowed: {"MIB_CASE_ID"}}
        )

        findings = scanner.scan([self.root])

        self.assertEqual(len(findings), 1)
        self.assertEqual(Path(findings[0].path), blocked)

    def test_clean_generic_runtime_passes_and_syntax_error_fails_closed(self):
        clean = self.root / "clean.py"
        clean.write_text(
            'RULES = {"risk": "DENIED"}\n'
            'def decide(value):\n'
            '    return RULES.get(value, "NEEDS_REVIEW")\n',
            encoding="utf-8",
        )
        self.assertEqual(RuntimeLeakageScanner().scan([clean]), ())

        broken = self.root / "broken.py"
        broken.write_text("def nope(:\n", encoding="utf-8")
        findings = RuntimeLeakageScanner().scan([broken])
        self.assertEqual(findings[0].code, "UNSCANNABLE")

    def test_detects_file_hash_maps_and_per_file_digest_keys(self):
        python_path = self.root / "runtime_hashes.py"
        python_path.write_text(
            f'PDF_HASH_LOOKUP = {{"packet.pdf": "{SHA_A}"}}\n'
            f'file_sha256 = "{SHA_B}"\n',
            encoding="utf-8",
        )
        json_path = self.root / "artifact_hashes.json"
        json_path.write_text(
            json.dumps(
                {
                    "sha256_by_filename": {"opaque": SHA_A},
                    "document_digest": SHA_B,
                    "packet-two.pdf": SHA_A,
                    "files": {"opaque": {"sha256": SHA_B}},
                }
            ),
            encoding="utf-8",
        )

        codes = {
            finding.code
            for finding in RuntimeLeakageScanner().scan([python_path, json_path])
        }

        self.assertIn("FILE_HASH_LOOKUP", codes)
        self.assertIn("PER_FILE_DIGEST_KEY", codes)


class CandidateStateStoreTests(TemporaryDirectoryTestCase):
    def test_blocked_assessment_does_not_replace_latest_passing_candidate(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        gate = CandidatePromotionGate(state)
        passing = gate.evaluate_and_record(
            "assessment-1",
            candidate_id="candidate-130",
            candidate_sha256=SHA_A,
            baseline_verified=True,
            leakage_finding_count=0,
            deterministic=True,
            false_approvals=0,
            missing_records=0,
            invalid_records=0,
            regression_counts={"golden": 0},
            fold_consistent=True,
            access_authorized=True,
            aggregate_evidence={"score": 130.37},
        )
        state.assess(
            "assessment-2",
            candidate_id="candidate-131-unsafe",
            candidate_sha256=SHA_B,
            decision="BLOCKED",
            aggregate_evidence={"score": 131.0, "false_approvals": 1},
        )

        self.assertEqual(state.latest_passing(), passing)
        self.assertEqual(len(state.assessments()), 2)

    def test_assessment_retry_is_idempotent_but_mismatch_is_rejected(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        request = {
            "candidate_id": "candidate-130",
            "candidate_sha256": SHA_A,
            "decision": "BLOCKED",
            "aggregate_evidence": {"score": 130.37},
        }
        first = state.assess("assessment-1", **request)
        retry = state.assess("assessment-1", **request)
        self.assertEqual(first, retry)
        self.assertEqual(state.store.length, 1)

        with self.assertRaises(ExperimentControlError):
            state.assess(
                "assessment-1",
                candidate_id="candidate-other",
                candidate_sha256=SHA_B,
                decision="BLOCKED",
                aggregate_evidence={"score": 100.0},
            )

    def test_case_level_candidate_evidence_is_rejected(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        with self.assertRaises(LeakageError):
            state.assess(
                "assessment-unsafe",
                candidate_id="candidate-unsafe",
                candidate_sha256=SHA_A,
                decision="BLOCKED",
                aggregate_evidence={"prediction_rows": ["MIB-000001"]},
            )

    def test_direct_pass_is_rejected_and_gate_evaluates_all_hard_requirements(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        with self.assertRaises(ExperimentControlError):
            state.assess(
                "forged-pass",
                candidate_id="candidate-forged",
                candidate_sha256=SHA_A,
                decision="PASSED",
                aggregate_evidence={"score": 150.0},
            )

        gate = CandidatePromotionGate(state)
        blocked = gate.evaluate_and_record(
            "gate-block",
            candidate_id="candidate-unsafe",
            candidate_sha256=SHA_B,
            baseline_verified=True,
            leakage_finding_count=0,
            deterministic=True,
            false_approvals=1,
            missing_records=0,
            invalid_records=0,
            regression_counts={"golden": 0},
            fold_consistent=True,
            access_authorized=True,
            aggregate_evidence={"score": 149.0},
        )

        self.assertEqual(blocked["decision"], "BLOCKED")
        self.assertEqual(blocked["aggregate_evidence"]["hard_gate_failure_count"], 1)
        self.assertIsNone(state.latest_passing())

    def test_gate_derives_access_authorization_and_supports_explicit_waiver(self):
        budget = ProtectedAccessBudget(
            self.root / "protected.jsonl", maximum_accesses=1
        )
        budget.record_access(
            "access-1",
            candidate_sha256=SHA_A,
            aggregate_result={"total_score": 148.0},
            purpose="promotion",
        )
        state = CandidateStateStore(self.root / "candidates.jsonl")
        gate = CandidatePromotionGate(state, protected_budget=budget)

        passing = gate.evaluate_and_record(
            "gate-pass",
            candidate_id="candidate-148",
            candidate_sha256=SHA_A,
            baseline_verified=True,
            leakage_finding_count=0,
            deterministic=True,
            false_approvals=0,
            missing_records=0,
            invalid_records=0,
            regression_counts={"golden": 1, "schema": 0},
            regression_waivers={"golden": "owner_approved"},
            fold_consistent=True,
            access_id="access-1",
            aggregate_evidence={"score": 148.0},
        )
        blocked = gate.evaluate_and_record(
            "gate-wrong-access",
            candidate_id="candidate-other",
            candidate_sha256=SHA_B,
            baseline_verified=True,
            leakage_finding_count=0,
            deterministic=True,
            false_approvals=0,
            missing_records=0,
            invalid_records=0,
            regression_counts={"golden": 0},
            fold_consistent=True,
            access_id="access-1",
            aggregate_evidence={"score": 149.0},
        )

        self.assertEqual(passing["decision"], "PASSED")
        self.assertTrue(passing["aggregate_evidence"]["access_authorized"])
        self.assertEqual(blocked["decision"], "BLOCKED")
        self.assertEqual(state.latest_passing(), passing)


if __name__ == "__main__":
    unittest.main()
