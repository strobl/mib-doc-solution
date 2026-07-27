import concurrent.futures
import copy
import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

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
    canonical_json,
    protected_access_binding,
    protected_access_summary,
    require_aggregate_only,
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
    def setUp(self):
        super().setUp()
        (
            self.split_manifest_path,
            self.split_manifest_sha256,
            self.split_fold_layout,
        ) = self._write_split_manifest(8, name="layout-manifest.json")
        self.input_dir = self.root / "input-pdfs"
        self.input_dir.mkdir()
        manifest = json.loads(
            self.split_manifest_path.read_text(encoding="utf-8")
        )
        for row in manifest["cases"]:
            (self.input_dir / f"{row['case_id']}.pdf").write_bytes(
                b"%PDF-fixture\n" + row["case_id"].encode("ascii")
            )
        tree_digest = hashlib.sha256()
        for path in sorted(
            self.input_dir.glob("*.pdf"),
            key=lambda item: (item.name.casefold(), item.name),
        ):
            name_bytes = path.name.encode("utf-8")
            tree_digest.update(len(name_bytes).to_bytes(4, "big"))
            tree_digest.update(name_bytes)
            tree_digest.update(hashlib.sha256(path.read_bytes()).digest())
        self.input_tree_sha256 = tree_digest.hexdigest()
        self._candidate_ledgers = {}
        self._layout_signature_patcher = mock.patch(
            "devtools.layout_manifest_freezer.layout_signature",
            side_effect=self._fixture_layout_signature,
        )
        self._rendering_metadata_patcher = mock.patch(
            "devtools.layout_manifest_freezer._rendering_version_metadata",
            side_effect=self._fixture_version_metadata,
        )
        self._layout_signature_patcher.start()
        self._rendering_metadata_patcher.start()
        self.addCleanup(self._layout_signature_patcher.stop)
        self.addCleanup(self._rendering_metadata_patcher.stop)

    def _write_split_manifest(self, record_count, *, name):
        group_count = min(10, record_count)
        cases = [
            {
                "case_id": f"MIB-{index + 1:06d}",
                "layout_group": (
                    f"page-count-{index % group_count + 1:02d}__"
                    f"ink-bucket-{index % group_count + 1:02d}"
                ),
            }
            for index in range(record_count)
        ]
        payload = {
            "cases": cases,
            "folds": 5,
            "label_blind_construction": True,
            "layout_signature": {
                "first_page_grayscale_ink_bucket_width": 0.03,
                "first_page_grayscale_ink_pixel_threshold_exclusive": 210,
                "first_page_render_height": 166,
                "first_page_render_width": 128,
                "inputs": [
                    "pdf_page_count",
                    "first_page_rendered_pixels",
                ],
                "pdfium_version": "fixture-pdfium",
                "pillow_version": "fixture-pillow",
                "pypdfium2_version": "fixture-pypdfium2",
                "version": "page-count-plus-first-page-ink-v1",
            },
            "repeats": 3,
            "schema": "mib-wo12-layout-groups/v2",
            "split_seed": "wo12-fixture-v2",
        }
        path = self.root / name
        content = (canonical_json(payload) + "\n").encode("utf-8")
        path.write_bytes(content)
        groups = {}
        for row in cases:
            groups.setdefault(row["layout_group"], []).append(row["case_id"])
        splits = RepeatedGroupedSplitManager(
            seed=payload["split_seed"],
            repeats=3,
            folds=5,
        ).split_groups(groups)
        fold_layout = {
            f"repeat_{split.repeat + 1}_fold_{split.fold + 1}": {
                "record_count": len(split.validation_case_ids),
                "validation_group_count": len(split.validation_groups),
            }
            for split in splits
        }
        digest = __import__("hashlib").sha256(content).hexdigest()
        return path, digest, fold_layout

    def _fixture_layout_signature(self, pdf_path):
        case_number = int(Path(pdf_path).stem.split("-")[1])
        group_number = (case_number - 1) % 8 + 1
        return group_number, group_number

    @staticmethod
    def _fixture_version_metadata():
        return {
            "pillow_version": "fixture-pillow",
            "pdfium_version": "fixture-pdfium",
            "pypdfium2_version": "fixture-pypdfium2",
        }

    def _passing_candidate_state(self, candidate_sha256):
        existing = self._candidate_ledgers.get(candidate_sha256)
        if existing is not None:
            return existing
        path = self.root / f"candidate-{candidate_sha256[:8]}.jsonl"
        state = CandidateStateStore(path)
        gate = CandidatePromotionGate(state)
        gate.evaluate_and_record(
            f"assessment-{candidate_sha256[:8]}",
            candidate_id=f"candidate-{candidate_sha256[:8]}",
            candidate_sha256=candidate_sha256,
            baseline_verified=True,
            leakage_finding_count=0,
            deterministic=True,
            false_approvals=0,
            missing_records=0,
            invalid_records=0,
            regression_counts={"adversarial": 0, "golden": 0},
            fold_consistent=True,
            access_authorized=True,
        )
        result = (
            path,
            state.store.head,
            gate.authorization_for(candidate_sha256),
        )
        self._candidate_ledgers[candidate_sha256] = result
        return result

    def _invoke_record_result(
        self,
        ledger,
        experiment_id,
        evidence,
        *,
        input_dir=None,
        split_manifest_path=None,
        **kwargs,
    ):
        return ledger.record_result(
            experiment_id,
            evidence,
            input_dir=input_dir or self.input_dir,
            split_manifest_path=(
                split_manifest_path or self.split_manifest_path
            ),
            **kwargs,
        )

    def _record_result(
        self,
        ledger,
        experiment_id,
        evidence,
        *,
        split_manifest_path=None,
        **kwargs,
    ):
        decision = kwargs["decision"]
        if decision == "adopt":
            (
                candidate_path,
                candidate_record_hash,
                candidate_authorization,
            ) = (
                self._passing_candidate_state(
                    evidence["candidate_artifact_sha256"]
                )
            )
            if evidence.get("candidate_state_record_hash") is None:
                evidence["candidate_state_record_hash"] = (
                    candidate_record_hash
                )
            kwargs.setdefault(
                "candidate_state_ledger_path",
                candidate_path,
            )
            kwargs.setdefault(
                "candidate_state_authorization",
                candidate_authorization,
            )
        return self._invoke_record_result(
            ledger,
            experiment_id,
            evidence,
            split_manifest_path=(
                split_manifest_path or self.split_manifest_path
            ),
            **kwargs,
        )

    def plan(self, **overrides):
        value = {
            "changed_files": [
                "devtools/experiment_control.py",
                "tests/test_experiment_control.py",
            ],
            "evidence_label": "public_grouped_robustness_not_unseen",
            "evaluator_sha256": SHA_A,
            "expected_record_count": 8,
            "hypothesis_sha256": SHA_A,
            "input_tree_sha256": self.input_tree_sha256,
            "parent_commit_sha": "c" * 40,
            "primary_variable_sha256": SHA_B,
            "protected_access_binding_sha256": None,
            "runtime_contract_sha256": SHA_A,
            "split_manifest_sha256": self.split_manifest_sha256,
            "truth_sha256": SHA_A,
        }
        value.update(overrides)
        return value

    def fold_metrics(self, *, fold_layout=None, score_delta=1.0):
        fold_layout = fold_layout or self.split_fold_layout
        return {
            fold_key: {
                "baseline_score": 130.0,
                "candidate_score": 130.0 + score_delta,
                "catastrophic_false_approvals": 0,
                "invalid_records": 0,
                "missing_records": 0,
                "record_count": layout["record_count"],
                "score_delta": score_delta,
                "validation_group_count": layout[
                    "validation_group_count"
                ],
            }
            for fold_key, layout in fold_layout.items()
        }

    def result(self, **overrides):
        value = {
            "baseline_artifact_sha256": SHA_A,
            "candidate_artifact_sha256": SHA_B,
            "candidate_state_record_hash": None,
            "checks": {
                "baseline_verified": True,
                "candidate_verified": True,
                "decision_freeze_verified": True,
                "deterministic": True,
                "fold_consistent": True,
                "runtime_leakage_clean": True,
                "runtime_limits_verified": True,
            },
            "evaluator_sha256": SHA_A,
            "evidence_label": "public_grouped_robustness_not_unseen",
            "expected_record_count": 8,
            "fold_metrics": self.fold_metrics(),
            "input_tree_sha256": self.input_tree_sha256,
            "metrics": {
                "baseline_total_score": 130.0,
                "calibration_score": 17.0,
                "candidate_image_bytes": 1_000_000,
                "candidate_max_model_artifact_bytes": 0,
                "candidate_model_bytes": 0,
                "catastrophic_false_approvals": 0,
                "classification_score": 69.0,
                "duplicate_records": 0,
                "extra_records": 0,
                "extraction_score": 45.0,
                "invalid_records": 0,
                "missing_records": 0,
                "output_bytes": 2_500,
                "peak_container_memory_bytes": 1_000_000,
                "peak_rss_bytes": 1_000_000,
                "process_cpu_seconds": 20.0,
                "record_count": 8,
                "runtime_seconds": 40.0,
                "score_delta": 1.0,
                "tmp_bytes": 0,
                "total_score": 131.0,
            },
            "protected_access_record_hash": None,
            "runtime_contract_sha256": SHA_A,
            "runtime_evidence_sha256": SHA_B,
            "split_manifest_sha256": self.split_manifest_sha256,
            "truth_sha256": SHA_A,
        }
        value.update(overrides)
        return value

    def test_legacy_records_remain_readable_but_new_one_stage_writes_are_blocked(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        legacy = ledger.store.append(
            {
                "event": "experiment",
                "experiment_id": "historical-exp",
                "evidence": {"total_score": 130.37},
            }
        )
        with self.assertRaises(ExperimentControlError):
            ledger.record("new-exp", {"total_score": 131.0})
        self.assertEqual(ledger.experiments()[0], legacy["payload"])
        self.assertEqual(ledger.store.length, 1)

        with self.assertRaises(ExperimentControlError):
            ledger.preregister("historical-exp", self.plan())
        self.assertEqual(ledger.store.length, 1)

    def test_preregister_then_record_result_are_distinct_and_bound(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        plan = ledger.preregister("wo12-demo", self.plan())
        retry = ledger.preregister(
            "wo12-demo",
            self.plan(changed_files=list(reversed(self.plan()["changed_files"]))),
        )
        result = self._record_result(
            ledger,
            "wo12-demo",
            self.result(),
            decision="reject",
            rationale="no_runtime_candidate",
            expected_head=plan["record_hash"],
        )
        retry_result = self._record_result(
            ledger,
            "wo12-demo",
            self.result(),
            decision="reject",
            rationale="no_runtime_candidate",
            expected_head=plan["record_hash"],
        )

        self.assertEqual(plan, retry)
        self.assertEqual(result, retry_result)
        self.assertEqual(plan["payload"]["event"], "experiment_plan")
        self.assertEqual(result["payload"]["event"], "experiment_result")
        self.assertEqual(
            result["payload"]["plan_record_hash"],
            plan["record_hash"],
        )
        self.assertEqual(len(ledger.plans()), 1)
        self.assertEqual(len(ledger.results()), 1)
        self.assertEqual(ledger.store.length, 2)

    def test_two_stage_contract_fails_closed(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "missing-plan",
                self.result(),
                decision="reject",
                rationale="missing_plan",
            )

        ledger.preregister("bound-plan", self.plan())
        mismatched = self.result(input_tree_sha256=SHA_B)
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "bound-plan",
                mismatched,
                decision="reject",
                rationale="population_mismatch",
            )
        leaked = self.result()
        leaked["metrics"]["notes"] = "MIB-000042.pdf"
        with self.assertRaises(LeakageError):
            self._record_result(
                ledger,
                "bound-plan",
                leaked,
                decision="reject",
                rationale="identity_leakage",
            )

        self._record_result(
            ledger,
            "bound-plan",
            self.result(),
            decision="reject",
            rationale="no_runtime_candidate",
        )
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "bound-plan",
                self.result(),
                decision="reject",
                rationale="duplicate_result",
            )

    def test_experiment_id_is_unique_across_legacy_plan_and_result_events(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("globally-unique", self.plan())
        ledger.store.append(
            {
                "event": "experiment",
                "experiment_id": "globally-unique",
                "evidence": {"total_score": 1.0},
            }
        )

        with self.assertRaises(IntegrityError):
            ledger.preregister("globally-unique", self.plan())
        with self.assertRaises(IntegrityError):
            self._record_result(
                ledger,
                "globally-unique",
                self.result(),
                decision="reject",
                rationale="legacy_collision",
            )

    def test_result_schema_rejects_opaque_vectors_and_incomplete_contracts(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("strict-result", self.plan())

        attacks = []
        missing_binding = self.result()
        del missing_binding["candidate_artifact_sha256"]
        attacks.append(missing_binding)

        extra_vector = self.result()
        extra_vector["fold_scores"] = [0.0] * 15
        attacks.append(extra_vector)

        opaque_metrics = self.result()
        opaque_metrics["metrics"] = {
            f"opaque{i:04d}": i % 2
            for i in range(8)
        }
        attacks.append(opaque_metrics)

        incomplete_checks = self.result()
        incomplete_checks["checks"] = {}
        attacks.append(incomplete_checks)

        opaque_fold = self.result()
        opaque_fold["fold_metrics"]["repeat_1_fold_1"]["opaque"] = 1
        attacks.append(opaque_fold)

        for index, evidence in enumerate(attacks):
            with self.subTest(index=index), self.assertRaises(
                ExperimentControlError
            ):
                self._record_result(
                    ledger,
                    "strict-result",
                    evidence,
                    decision="reject",
                    rationale=f"schema_attack_{index}",
                )
        self.assertEqual(ledger.store.length, 1)

        with self.assertRaises(LeakageError):
            require_aggregate_only({"fold_scores": [0, 1] * 500})
        with self.assertRaises(LeakageError):
            require_aggregate_only(
                {
                    "metrics": {
                        f"opaque{i:04d}": i % 2
                        for i in range(1_000)
                    }
                }
            )

    def test_covert_aggregate_vectors_and_huge_resource_integers_are_rejected(self):
        with self.assertRaises(LeakageError):
            require_aggregate_only(
                {
                    f"opaque_{index:02d}_count": index % 2
                    for index in range(33)
                }
            )
        with self.assertRaises(LeakageError):
            require_aggregate_only({"fold_scores": [0, 1] * 8})
        with self.assertRaises(LeakageError):
            require_aggregate_only(
                {
                    f"opaque_{index}_hash": SHA_A
                    for index in range(4)
                }
            )
        with self.assertRaises(LeakageError):
            require_aggregate_only(
                {
                    "field_metrics": {
                        f"dimension_{dimension}": {
                            f"opaque_{metric}_count": metric % 2
                            for metric in range(32)
                        }
                        for dimension in range(32)
                    }
                }
            )

        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("huge-resource", self.plan())
        evidence = self.result()
        evidence["metrics"]["output_bytes"] = (1 << 999) + 12_345
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "huge-resource",
                evidence,
                decision="reject",
                rationale="covert_integer",
            )
        self.assertEqual(ledger.store.length, 1)

    def test_grouped_result_requires_exact_three_by_five_complete_coverage(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("grouped-shape", self.plan())

        missing_fold = self.result()
        del missing_fold["fold_metrics"]["repeat_3_fold_5"]
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "grouped-shape",
                missing_fold,
                decision="reject",
                rationale="missing_fold",
            )

        extra_fold = self.result()
        extra_fold["fold_metrics"]["repeat_4_fold_1"] = dict(
            extra_fold["fold_metrics"]["repeat_1_fold_1"]
        )
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "grouped-shape",
                extra_fold,
                decision="reject",
                rationale="extra_fold",
            )

        incomplete_population = self.result()
        incomplete_population["fold_metrics"]["repeat_2_fold_1"][
            "record_count"
        ] -= 1
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "grouped-shape",
                incomplete_population,
                decision="reject",
                rationale="incomplete_population",
            )

        wrong_group_count = self.result()
        wrong_group_count["fold_metrics"]["repeat_1_fold_1"][
            "validation_group_count"
        ] += 1
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "grouped-shape",
                wrong_group_count,
                decision="reject",
                rationale="fabricated_group_count",
            )

        fabricated_scores = self.result()
        for row in fabricated_scores["fold_metrics"].values():
            row["baseline_score"] = 0.0
            row["candidate_score"] = 1.0
            row["score_delta"] = 1.0
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "grouped-shape",
                fabricated_scores,
                decision="adopt",
                rationale="fabricated_fold_scores",
            )

        changed_manifest = self.root / "changed-layout-manifest.json"
        changed_manifest.write_bytes(
            self.split_manifest_path.read_bytes() + b" "
        )
        with self.assertRaises(IntegrityError):
            self._record_result(
                ledger,
                "grouped-shape",
                self.result(),
                decision="reject",
                rationale="manifest_hash_mismatch",
                split_manifest_path=changed_manifest,
            )

    def test_adopt_requires_a_bound_passing_candidate_state_record(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("state-bound", self.plan())

        same_artifact = self.result(
            baseline_artifact_sha256=SHA_A,
            candidate_artifact_sha256=SHA_A,
        )
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "state-bound",
                same_artifact,
                decision="adopt",
                rationale="same_as_baseline",
            )

        missing_state = self.result(candidate_state_record_hash=SHA_A)
        with self.assertRaises(ExperimentControlError):
            self._invoke_record_result(
                ledger,
                "state-bound",
                missing_state,
                decision="adopt",
                rationale="missing_state_ledger",
            )

        blocked_path = self.root / "blocked-candidate.jsonl"
        blocked_state = CandidateStateStore(blocked_path)
        blocked_state.assess(
            "blocked-assessment",
            candidate_id="blocked-candidate",
            candidate_sha256=SHA_B,
            decision="BLOCKED",
            aggregate_evidence={"status": "blocked"},
        )
        blocked = self.result(
            candidate_state_record_hash=blocked_state.store.head
        )
        with self.assertRaises(IntegrityError):
            self._invoke_record_result(
                ledger,
                "state-bound",
                blocked,
                decision="adopt",
                rationale="blocked_state",
                candidate_state_ledger_path=blocked_path,
            )

        forged_path = self.root / "forged-passed-candidate.jsonl"
        forged_store = CanonicalHashChainStore(forged_path)
        forged_store.append(
            {
                "aggregate_evidence": {
                    "access_authorized": True,
                    "baseline_verified": True,
                    "deterministic": True,
                    "false_approvals": 0,
                    "fold_consistent": True,
                    "gate_results": {
                        "access_authorized": True,
                        "baseline_verified": True,
                        "deterministic": True,
                        "fold_consistent": True,
                        "no_false_approvals": True,
                        "no_invalid_records": True,
                        "no_leakage": True,
                        "no_missing_records": True,
                        "regressions_cleared": True,
                    },
                    "hard_gate_failure_count": 0,
                    "invalid_records": 0,
                    "leakage_finding_count": 0,
                    "missing_records": 0,
                    "promotion_gate_verified": True,
                    "regression_counts": {
                        "adversarial": 0,
                        "golden": 0,
                    },
                    "regression_waiver_count": 0,
                },
                "assessment_id": "forged-assessment",
                "candidate_id": "forged-candidate",
                "candidate_sha256": SHA_B,
                "decision": "PASSED",
                "event": "candidate_assessment",
            }
        )
        forged = self.result(
            candidate_state_record_hash=forged_store.head
        )
        with self.assertRaises(IntegrityError):
            self._invoke_record_result(
                ledger,
                "state-bound",
                forged,
                decision="adopt",
                rationale="forged_passed_state",
                candidate_state_ledger_path=forged_path,
            )

        _, _, genuine_authorization = self._passing_candidate_state(SHA_A)
        cloned_authorization = copy.copy(genuine_authorization)
        self.assertIsNot(cloned_authorization, genuine_authorization)
        with self.assertRaises(IntegrityError):
            self._invoke_record_result(
                ledger,
                "state-bound",
                forged,
                decision="adopt",
                rationale="cloned_capability",
                candidate_state_ledger_path=forged_path,
                candidate_state_authorization=cloned_authorization,
            )

    def test_protected_result_requires_the_exact_budgeted_access_record(self):
        access_path = self.root / "protected-access.jsonl"
        budget = ProtectedAccessBudget(access_path, maximum_accesses=2)
        binding = protected_access_binding(
            access_path,
            maximum_accesses=2,
        )
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister(
            "protected-binding",
            self.plan(
                evidence_label="protected",
                protected_access_binding_sha256=binding,
            ),
        )
        evidence = self.result(
            evidence_label="protected",
        )
        budget.record_access(
            "protected-binding-access",
            candidate_sha256=SHA_B,
            aggregate_result=protected_access_summary(evidence),
            purpose="protected_binding",
        )
        evidence["protected_access_record_hash"] = budget.store.head
        recorded = self._invoke_record_result(
            ledger,
            "protected-binding",
            evidence,
            decision="reject",
            rationale="diagnostic_only",
            protected_access_ledger_path=access_path,
            protected_access_authorization=budget.authorization_for(
                "protected-binding-access"
            ),
        )
        self.assertEqual(recorded["payload"]["decision"], "reject")

        second_access_path = self.root / "second-protected-access.jsonl"
        second_budget = ProtectedAccessBudget(
            second_access_path,
            maximum_accesses=2,
        )
        second = ExperimentLedger(self.root / "second-experiments.jsonl")
        second.preregister(
            "syntax-only-access",
            self.plan(
                evidence_label="protected",
                protected_access_binding_sha256=protected_access_binding(
                    second_access_path,
                    maximum_accesses=2,
                ),
            ),
        )
        syntax_only = self.result(
            evidence_label="protected",
            protected_access_record_hash=SHA_A,
        )
        with self.assertRaises(ExperimentControlError):
            self._invoke_record_result(
                second,
                "syntax-only-access",
                syntax_only,
                decision="reject",
                rationale="missing_access_ledger",
            )

        wrong_candidate = self.result(
            evidence_label="protected",
        )
        second_budget.record_access(
            "wrong-candidate-access",
            candidate_sha256=SHA_A,
            aggregate_result=protected_access_summary(wrong_candidate),
            purpose="wrong_candidate",
        )
        wrong_candidate["protected_access_record_hash"] = (
            second_budget.store.head
        )
        with self.assertRaises(IntegrityError):
            self._invoke_record_result(
                second,
                "syntax-only-access",
                wrong_candidate,
                decision="reject",
                rationale="wrong_candidate",
                protected_access_ledger_path=second_access_path,
                protected_access_authorization=(
                    second_budget.authorization_for(
                        "wrong-candidate-access"
                    )
                ),
            )

        contradictory_path = self.root / "contradictory-access.jsonl"
        contradictory_budget = ProtectedAccessBudget(
            contradictory_path,
            maximum_accesses=1,
        )
        contradictory = ExperimentLedger(
            self.root / "contradictory-experiments.jsonl"
        )
        contradictory.preregister(
            "contradictory-access",
            self.plan(
                evidence_label="protected",
                protected_access_binding_sha256=protected_access_binding(
                    contradictory_path,
                    maximum_accesses=1,
                ),
            ),
        )
        contradictory_evidence = self.result(
            evidence_label="protected"
        )
        contradictory_summary = protected_access_summary(
            contradictory_evidence
        )
        contradictory_summary["metrics"]["total_score"] = 1.0
        contradictory_budget.record_access(
            "contradictory-access",
            candidate_sha256=SHA_B,
            aggregate_result=contradictory_summary,
            purpose="contradictory_access",
        )
        contradictory_evidence["protected_access_record_hash"] = (
            contradictory_budget.store.head
        )
        with self.assertRaises(IntegrityError):
            self._invoke_record_result(
                contradictory,
                "contradictory-access",
                contradictory_evidence,
                decision="reject",
                rationale="contradictory_result",
                protected_access_ledger_path=contradictory_path,
                protected_access_authorization=(
                    contradictory_budget.authorization_for(
                        "contradictory-access"
                    )
                ),
            )

    def test_older_passed_retry_reissues_capability_for_its_exact_record(self):
        candidate_path = self.root / "retry-candidates.jsonl"
        state = CandidateStateStore(candidate_path)
        gate = CandidatePromotionGate(state)

        def evaluate(assessment_id, candidate_sha256):
            return gate.evaluate_and_record(
                assessment_id,
                candidate_id=f"candidate-{assessment_id}",
                candidate_sha256=candidate_sha256,
                baseline_verified=True,
                leakage_finding_count=0,
                deterministic=True,
                false_approvals=0,
                missing_records=0,
                invalid_records=0,
                regression_counts={"adversarial": 0, "golden": 0},
                fold_consistent=True,
                access_authorized=True,
            )

        first = evaluate("older-pass", SHA_B)
        first_record_hash = state.store.head
        evaluate("newer-pass", SHA_A)
        self.assertEqual(evaluate("older-pass", SHA_B), first)

        ledger = ExperimentLedger(self.root / "retry-experiments.jsonl")
        ledger.preregister("older-pass-retry", self.plan())
        evidence = self.result(
            candidate_state_record_hash=first_record_hash,
        )
        recorded = self._invoke_record_result(
            ledger,
            "older-pass-retry",
            evidence,
            decision="adopt",
            rationale="exact_retry_record",
            candidate_state_ledger_path=candidate_path,
            candidate_state_authorization=gate.authorization_for(SHA_B),
        )
        self.assertEqual(recorded["payload"]["decision"], "adopt")

    def test_protected_budget_cannot_be_replaced_after_preregistration(self):
        planned_path = self.root / "planned-protected-access.jsonl"
        ProtectedAccessBudget(planned_path, maximum_accesses=2)
        ledger = ExperimentLedger(self.root / "bound-experiments.jsonl")
        ledger.preregister(
            "budget-replacement",
            self.plan(
                evidence_label="protected",
                protected_access_binding_sha256=protected_access_binding(
                    planned_path,
                    maximum_accesses=2,
                ),
            ),
        )

        replacement_path = self.root / "replacement-access.jsonl"
        replacement = ProtectedAccessBudget(
            replacement_path,
            maximum_accesses=10_000_000,
        )
        evidence = self.result(evidence_label="protected")
        replacement.record_access(
            "replacement-access",
            candidate_sha256=SHA_B,
            aggregate_result=protected_access_summary(evidence),
            purpose="replacement_access",
        )
        evidence["protected_access_record_hash"] = replacement.store.head

        with self.assertRaises(IntegrityError):
            self._invoke_record_result(
                ledger,
                "budget-replacement",
                evidence,
                decision="reject",
                rationale="replacement_budget",
                protected_access_ledger_path=replacement_path,
                protected_access_authorization=(
                    replacement.authorization_for("replacement-access")
                ),
            )

    def test_split_manifest_groups_are_recomputed_from_the_bound_pdfs(self):
        engineered = json.loads(
            self.split_manifest_path.read_text(encoding="utf-8")
        )
        engineered["cases"][0]["layout_group"] = (
            engineered["cases"][1]["layout_group"]
        )
        engineered_path = self.root / "label-engineered-manifest.json"
        engineered_bytes = (
            canonical_json(engineered) + "\n"
        ).encode("utf-8")
        engineered_path.write_bytes(engineered_bytes)
        engineered_sha256 = hashlib.sha256(engineered_bytes).hexdigest()

        ledger = ExperimentLedger(
            self.root / "engineered-experiments.jsonl"
        )
        ledger.preregister(
            "engineered-layout",
            self.plan(split_manifest_sha256=engineered_sha256),
        )
        evidence = self.result(
            split_manifest_sha256=engineered_sha256
        )
        with self.assertRaises(IntegrityError):
            self._record_result(
                ledger,
                "engineered-layout",
                evidence,
                decision="reject",
                rationale="label_engineered_groups",
                split_manifest_path=engineered_path,
            )

    def test_preregister_detects_multiple_results_for_one_plan(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("duplicate-result-records", self.plan())
        first = self._record_result(
            ledger,
            "duplicate-result-records",
            self.result(),
            decision="reject",
            rationale="first_result",
        )
        ledger.store.append(first["payload"])

        with self.assertRaises(IntegrityError):
            ledger.preregister(
                "duplicate-result-records",
                self.plan(),
            )

    def test_plan_population_is_bounded_and_result_cas_detects_intervening_write(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        for count in (4, 5_001):
            with self.subTest(count=count), self.assertRaises(
                ExperimentControlError
            ):
                ledger.preregister(
                    f"bad-population-{count}",
                    self.plan(expected_record_count=count),
                )

        first = ledger.preregister("first-plan", self.plan())
        ledger.preregister("intervening-plan", self.plan())
        with self.assertRaises(CompareAndSwapError):
            self._record_result(
                ledger,
                "first-plan",
                self.result(),
                decision="reject",
                rationale="stale_head",
                expected_head=first["record_hash"],
            )
        self.assertEqual(len(ledger.results()), 0)

    def test_adopt_requires_every_gate_gain_count_and_runtime_cap(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("adoption-gates", self.plan())

        failed_gate = self.result()
        failed_gate["checks"]["runtime_limits_verified"] = False
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "adoption-gates",
                failed_gate,
                decision="adopt",
                rationale="failed_gate",
            )

        wrong_count = self.result()
        wrong_count["metrics"]["record_count"] = 7
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "adoption-gates",
                wrong_count,
                decision="adopt",
                rationale="wrong_count",
            )

        unsafe = self.result()
        unsafe["metrics"]["catastrophic_false_approvals"] = 1
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "adoption-gates",
                unsafe,
                decision="adopt",
                rationale="unsafe",
            )

        slow = self.result()
        slow["metrics"]["runtime_seconds"] = 48.01
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "adoption-gates",
                slow,
                decision="adopt",
                rationale="runtime_cap",
            )

        concentrated = self.result()
        for repeat in range(1, 4):
            for fold in range(1, 6):
                row = concentrated["fold_metrics"][
                    f"repeat_{repeat}_fold_{fold}"
                ]
                row["candidate_score"] = (
                    135.0 if fold == 1 else 129.0
                )
                row["score_delta"] = (
                    5.0 if fold == 1 else -1.0
                )
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "adoption-gates",
                concentrated,
                decision="adopt",
                rationale="single_fold_gain",
            )

        negative_fold = self.result()
        for repeat in range(1, 4):
            for fold in range(1, 6):
                row = negative_fold["fold_metrics"][
                    f"repeat_{repeat}_fold_{fold}"
                ]
                delta = -0.5 if fold == 1 else 1.5
                row["candidate_score"] = row["baseline_score"] + delta
                row["score_delta"] = delta
        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "adoption-gates",
                negative_fold,
                decision="adopt",
                rationale="negative_fold",
            )

        adopted = self._record_result(
            ledger,
            "adoption-gates",
            self.result(),
            decision="adopt",
            rationale="all_gates_passed",
        )
        self.assertEqual(adopted["payload"]["decision"], "adopt")

    def test_adopt_enforces_the_official_four_hour_runtime_cap(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("official-runtime-cap", self.plan())
        evidence = self.result()
        evidence["metrics"]["runtime_seconds"] = 14_400.01

        with self.assertRaises(ExperimentControlError):
            self._record_result(
                ledger,
                "official-runtime-cap",
                evidence,
                decision="adopt",
                rationale="over_four_hours",
            )

    def test_exact_result_retry_is_idempotent_under_concurrency(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        plan = ledger.preregister("concurrent-result", self.plan())

        def write_result():
            return self._record_result(
                ledger,
                "concurrent-result",
                self.result(),
                decision="reject",
                rationale="concurrent_retry",
                expected_head=plan["record_hash"],
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            records = list(executor.map(lambda _: write_result(), range(16)))

        self.assertEqual(
            {record["record_hash"] for record in records},
            {records[0]["record_hash"]},
        )
        self.assertEqual(ledger.store.length, 2)

    def test_concurrent_mismatched_results_append_exactly_one(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        ledger.preregister("mismatched-writers", self.plan())
        barrier = threading.Barrier(2)

        def write_result(rationale):
            barrier.wait()
            return self._record_result(
                ledger,
                "mismatched-writers",
                self.result(),
                decision="reject",
                rationale=rationale,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(write_result, rationale)
                for rationale in ("first_writer", "second_writer")
            ]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(("ok", future.result()))
                except ExperimentControlError as exc:
                    outcomes.append(("error", exc))

        self.assertEqual([kind for kind, _ in outcomes].count("ok"), 1)
        self.assertEqual([kind for kind, _ in outcomes].count("error"), 1)
        self.assertEqual(ledger.store.length, 2)

    def test_rejects_nested_case_ids_filenames_and_case_level_keys(self):
        unsafe_values = (
            {"notes": ["MIB-000042 was wrong"]},
            {"artifact": "train/MIB-000042.pdf"},
            {"case_scores": {"anything": 1}},
            {"validation_case_ids": ["opaque-public-row"]},
        )
        for index, value in enumerate(unsafe_values):
            with self.subTest(index=index), self.assertRaises(LeakageError):
                require_aggregate_only(value)

    def test_strict_schema_rejects_record_shapes_and_retains_aggregate_hashes(self):
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
            {"counts": {"MIB-000001": 1, "MIB-000002": 0}},
            {"counts": {"arbitrary_token": 1}},
        )
        for index, value in enumerate(unsafe_values):
            with self.subTest(index=index), self.assertRaises(LeakageError):
                require_aggregate_only(value)

        require_aggregate_only(
            {
                "artifact_sha256": SHA_A,
                "fold_scores": [129.0, 130.0, 131.0],
                "field_metrics": {
                    "risk": {"accuracy": 0.9, "error_count": 10}
                },
            },
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
