import concurrent.futures
import copy
import hashlib
import json
import os
import shutil
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
    CheckpointAuthorityResolver,
    CompareAndSwapError,
    ExperimentControlError,
    ExperimentLedger,
    FrozenBaselineManifest,
    GovernedMutationReceipt,
    IntegrityError,
    LeakageError,
    ProgramIntegrityCheckpoint,
    ProgramIntegritySuccessor,
    PublishedCheckpointReference,
    ProtectedAccessBudget,
    RepeatedGroupedSplitManager,
    RuntimeLeakageScanner,
    TaintRegistry,
    build_program_integrity_checkpoint,
    build_retrospective_reconciliation,
    canonical_json,
    require_retrospective_reconciliation,
    verify_retrospective_reconciliation,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


class TemporaryDirectoryTestCase(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self._checkpoint_counter = 0
        self._default_program_stores = {
            name: CanonicalHashChainStore(
                self.root / "_program" / f"{name}.jsonl"
            )
            for name in (
                "candidate_state_ledger",
                "experiment_ledger",
                "protected_access_ledger",
                "taint_registry",
            )
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def _governed_store(value):
        if value is None:
            return None
        if isinstance(value, CanonicalHashChainStore):
            return value
        return value.store

    def program_stores(
        self,
        *,
        candidate_state_ledger=None,
        experiment_ledger=None,
        protected_access_ledger=None,
        taint_registry=None,
    ):
        supplied = {
            "candidate_state_ledger": candidate_state_ledger,
            "experiment_ledger": experiment_ledger,
            "protected_access_ledger": protected_access_ledger,
            "taint_registry": taint_registry,
        }
        return {
            name: self._governed_store(value)
            or self._default_program_stores[name]
            for name, value in supplied.items()
        }

    @staticmethod
    def default_promotion_population(**overrides):
        population = {
            "evaluator_sha256": SHA_A,
            "expected_record_count": 5,
            "input_tree_sha256": SHA_A,
            "runtime_contract_sha256": SHA_A,
            "split_manifest_sha256": SHA_A,
            "truth_sha256": SHA_A,
        }
        population.update(overrides)
        return population

    def checkpoint_for(
        self,
        *,
        promotion_population=None,
        runtime_leakage_finding_count=0,
        **stores,
    ):
        normalized = self.program_stores(**stores)
        raw, digest = build_program_integrity_checkpoint(
            stores=normalized,
            baseline_manifest_sha256=SHA_A,
            checkpoint_directory=self.root / "_checkpoints",
            runtime_leakage_finding_count=(
                runtime_leakage_finding_count
            ),
            promotion_population=(
                promotion_population
                if promotion_population is not None
                else self.default_promotion_population()
            ),
        )
        self._checkpoint_counter += 1
        path = (
            self.root
            / "_checkpoints"
            / f"{self._checkpoint_counter:04d}-{digest}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        reference = PublishedCheckpointReference(path=path, sha256=digest)
        return CheckpointAuthorityResolver(
            lambda: reference,
            trusted_checkpoint_root=self.root / "_checkpoints",
        ).resolve(stores=normalized)

    def publish_successor(self, checkpoint, successor):
        CheckpointAuthorityResolver.validate_successor(
            checkpoint,
            successor,
        )
        self._checkpoint_counter += 1
        path = (
            self.root
            / "_checkpoints"
            / (
                f"{self._checkpoint_counter:04d}-"
                f"{successor.checkpoint_sha256}.json"
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(successor.checkpoint_bytes)
        reference = PublishedCheckpointReference(
            path=path,
            sha256=successor.checkpoint_sha256,
        )
        return CheckpointAuthorityResolver(
            lambda: reference,
            trusted_checkpoint_root=self.root / "_checkpoints",
        ).resolve(stores=checkpoint.stores)

    def create_budget(
        self,
        path,
        *,
        maximum_accesses,
        candidate_state_ledger=None,
        experiment_ledger=None,
        taint_registry=None,
    ):
        protected_store = CanonicalHashChainStore(path)
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=candidate_state_ledger,
            experiment_ledger=experiment_ledger,
            protected_access_ledger=protected_store,
            taint_registry=taint_registry,
        )
        budget = ProtectedAccessBudget(
            path,
            maximum_accesses=maximum_accesses,
            integrity_checkpoint=checkpoint,
        )
        self.publish_successor(
            checkpoint,
            budget.initialization_receipt.integrity,
        )
        return budget

    def governed_preregister(self, ledger, experiment_id, plan, **kwargs):
        checkpoint = self.checkpoint_for(
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
        )
        return ledger.preregister(
            experiment_id,
            plan,
            integrity_checkpoint=checkpoint,
            **kwargs,
        )

    def governed_record_result(
        self,
        ledger,
        experiment_id,
        evidence,
        **kwargs,
    ):
        checkpoint = self.checkpoint_for(
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
        )
        return ledger.record_result(
            experiment_id,
            evidence,
            integrity_checkpoint=checkpoint,
            **kwargs,
        )

    def governed_record_access(
        self,
        budget,
        access_id,
        *,
        experiment_ledger,
        candidate_state_ledger=None,
        taint_registry=None,
        **kwargs,
    ):
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=candidate_state_ledger,
            experiment_ledger=experiment_ledger,
            protected_access_ledger=budget,
            taint_registry=taint_registry,
        )
        return budget.record_access(
            access_id,
            experiment_ledger=experiment_ledger,
            integrity_checkpoint=checkpoint,
            **kwargs,
        )

    def governed_taint(self, registry, group_id, **kwargs):
        checkpoint = self.checkpoint_for(taint_registry=registry)
        return registry.taint(
            group_id,
            integrity_checkpoint=checkpoint,
            **kwargs,
        )

    def governed_assess(self, state, assessment_id, **kwargs):
        checkpoint = self.checkpoint_for(candidate_state_ledger=state)
        return state.assess(
            assessment_id,
            integrity_checkpoint=checkpoint,
            **kwargs,
        )


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


class ProgramIntegrityCheckpointTests(TemporaryDirectoryTestCase):
    def test_builder_is_canonical_exact_and_historical_checkpoint_is_readable(self):
        stores = self.program_stores()
        raw, digest = build_program_integrity_checkpoint(
            stores=stores,
            baseline_manifest_sha256=SHA_A,
            checkpoint_directory=self.root,
            runtime_leakage_finding_count=0,
        )
        value = json.loads(raw)
        self.assertEqual(raw, (canonical_json(value) + "\n").encode("utf-8"))
        path = self.root / "checkpoint.json"
        path.write_bytes(raw)
        checkpoint = ProgramIntegrityCheckpoint(
            path,
            expected_sha256=digest,
            stores=stores,
        )
        self.assertEqual(checkpoint.verify(), value)

        repository_root = Path(__file__).resolve().parents[1]
        program_root = repository_root / "evaluation/program"
        historical_path = program_root / "integrity_heads.json"
        historical_digest = hashlib.sha256(
            historical_path.read_bytes()
        ).hexdigest()
        historical = ProgramIntegrityCheckpoint(
            historical_path,
            expected_sha256=historical_digest,
            stores={
                "candidate_state_ledger": CanonicalHashChainStore(
                    program_root / "candidate_state_ledger.jsonl"
                ),
                "experiment_ledger": CanonicalHashChainStore(
                    program_root / "experiment_ledger.jsonl"
                ),
                "protected_access_ledger": CanonicalHashChainStore(
                    program_root / "protected_access_ledger.jsonl"
                ),
                "taint_registry": CanonicalHashChainStore(
                    program_root / "taint_registry.jsonl"
                ),
            },
        )
        self.assertEqual(
            historical.verify()["stores"]["candidate_state_ledger"][
                "expected_length"
            ],
            1,
        )

    def test_wrong_digest_noncanonical_and_invalid_schema_fail_closed(self):
        stores = self.program_stores()
        raw, digest = build_program_integrity_checkpoint(
            stores=stores,
            baseline_manifest_sha256=SHA_A,
            checkpoint_directory=self.root,
            runtime_leakage_finding_count=0,
        )
        path = self.root / "checkpoint.json"
        path.write_bytes(raw)
        with self.assertRaises(IntegrityError):
            ProgramIntegrityCheckpoint(
                path,
                expected_sha256=SHA_B,
                stores=stores,
            ).verify()

        noncanonical = json.dumps(json.loads(raw), indent=2).encode("utf-8")
        path.write_bytes(noncanonical)
        with self.assertRaises(IntegrityError):
            ProgramIntegrityCheckpoint(
                path,
                expected_sha256=hashlib.sha256(noncanonical).hexdigest(),
                stores=stores,
            ).verify()

        scenarios = {}
        missing_store = json.loads(raw)
        del missing_store["stores"]["taint_registry"]
        scenarios["missing-store"] = missing_store
        extra_store = json.loads(raw)
        extra_store["stores"]["other"] = copy.deepcopy(
            extra_store["stores"]["taint_registry"]
        )
        scenarios["extra-store"] = extra_store
        boolean_length = json.loads(raw)
        boolean_length["stores"]["experiment_ledger"][
            "expected_length"
        ] = False
        scenarios["boolean-length"] = boolean_length
        wrong_file_hash = json.loads(raw)
        wrong_file_hash["stores"]["experiment_ledger"]["sha256"] = SHA_B
        scenarios["wrong-file-hash"] = wrong_file_hash

        for name, value in scenarios.items():
            with self.subTest(name=name):
                candidate_raw = (
                    canonical_json(value) + "\n"
                ).encode("utf-8")
                path.write_bytes(candidate_raw)
                checkpoint = ProgramIntegrityCheckpoint(
                    path,
                    expected_sha256=hashlib.sha256(
                        candidate_raw
                    ).hexdigest(),
                    stores=stores,
                )
                with self.assertRaises(IntegrityError):
                    checkpoint.verify()

    def test_valid_prefix_truncation_is_rejected_for_every_governed_store(self):
        stores = self.program_stores()
        original_bytes = {}
        for name, store in stores.items():
            store.append({"event": f"{name}-one"})
            store.append({"event": f"{name}-two"})
            original_bytes[name] = store.path.read_bytes()
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=stores["candidate_state_ledger"],
            experiment_ledger=stores["experiment_ledger"],
            protected_access_ledger=stores["protected_access_ledger"],
            taint_registry=stores["taint_registry"],
        )

        for name, store in stores.items():
            with self.subTest(name=name):
                first_line = original_bytes[name].splitlines(
                    keepends=True
                )[0]
                store.path.write_bytes(first_line)
                with self.assertRaises(IntegrityError):
                    checkpoint.verify()
                store.path.write_bytes(original_bytes[name])
        checkpoint.verify()

    def test_self_consistent_rewritten_checkpoint_cannot_replace_external_digest(self):
        stores = self.program_stores()
        experiment = stores["experiment_ledger"]
        experiment.append({"event": "one"})
        experiment.append({"event": "two"})
        checkpoint = self.checkpoint_for(
            experiment_ledger=experiment
        )
        old_digest = checkpoint.expected_sha256

        first_line = experiment.path.read_bytes().splitlines(
            keepends=True
        )[0]
        experiment.path.write_bytes(first_line)
        rewritten_raw, rewritten_digest = build_program_integrity_checkpoint(
            stores=checkpoint.stores,
            baseline_manifest_sha256=SHA_A,
            checkpoint_directory=checkpoint.path.parent,
            runtime_leakage_finding_count=0,
        )
        checkpoint.path.write_bytes(rewritten_raw)
        self.assertNotEqual(old_digest, rewritten_digest)
        with self.assertRaises(IntegrityError):
            checkpoint.verify()

    def test_stale_checkpoint_and_crash_window_block_until_successor_is_published(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        checkpoint = self.checkpoint_for(experiment_ledger=ledger)
        with mock.patch.object(
            checkpoint,
            "successor",
            side_effect=RuntimeError("checkpoint publication interrupted"),
        ), self.assertRaises(RuntimeError):
            ledger.preregister(
                "first-plan",
                ExperimentLedgerTests.plan(),
                integrity_checkpoint=checkpoint,
            )
        plan_record = ledger.store.verify()[-1]

        with self.assertRaises(IntegrityError):
            checkpoint.verify()
        with self.assertRaises(IntegrityError):
            ledger.preregister(
                "blocked-during-crash-window",
                ExperimentLedgerTests.plan(),
                integrity_checkpoint=checkpoint,
            )
        self.assertEqual(ledger.store.length, 1)

        recovered_successor = checkpoint.successor(
            mutated_store="experiment_ledger",
            record_hash=plan_record["record_hash"],
        )
        published = self.publish_successor(
            checkpoint,
            recovered_successor,
        )
        published.verify()
        ledger.preregister(
            "after-recovery",
            ExperimentLedgerTests.plan(hypothesis_sha256=SHA_C),
            integrity_checkpoint=published,
        )
        self.assertEqual(ledger.store.length, 2)

    def test_successor_rejects_concurrent_non_target_mutation(self):
        stores = self.program_stores()
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=stores["candidate_state_ledger"],
            experiment_ledger=stores["experiment_ledger"],
            protected_access_ledger=stores["protected_access_ledger"],
            taint_registry=stores["taint_registry"],
        )
        target = stores["experiment_ledger"].append({"event": "plan"})
        stores["taint_registry"].append({"event": "concurrent-taint"})

        with self.assertRaises(IntegrityError):
            checkpoint.successor(
                mutated_store="experiment_ledger",
                record_hash=target["record_hash"],
            )

    def test_checkpoint_cannot_be_reused_for_a_different_store_binding(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        checkpoint = self.checkpoint_for(experiment_ledger=ledger)
        shadow = ExperimentLedger(self.root / "shadow-experiments.jsonl")

        with self.assertRaises(IntegrityError):
            shadow.preregister(
                "shadow-plan",
                ExperimentLedgerTests.plan(),
                integrity_checkpoint=checkpoint,
            )
        self.assertEqual(shadow.store.length, 0)
        rebound_stores = dict(checkpoint.stores)
        rebound_stores["experiment_ledger"] = shadow.store
        with self.assertRaises(IntegrityError):
            ProgramIntegrityCheckpoint(
                checkpoint.path,
                expected_sha256=checkpoint.expected_sha256,
                stores=rebound_stores,
            ).verify()

    def test_locally_reminted_checkpoint_cannot_authorize_mutation(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        stores = self.program_stores(experiment_ledger=ledger)
        raw, digest = build_program_integrity_checkpoint(
            stores=stores,
            baseline_manifest_sha256=SHA_A,
            checkpoint_directory=self.root / "_checkpoints",
            runtime_leakage_finding_count=0,
            promotion_population=self.default_promotion_population(),
        )
        path = self.root / "_checkpoints" / "locally-reminted.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        untrusted = ProgramIntegrityCheckpoint(
            path,
            expected_sha256=digest,
            stores=stores,
        )
        untrusted.verify()

        with self.assertRaises(IntegrityError):
            ledger.preregister(
                "locally-reminted",
                ExperimentLedgerTests.plan(),
                integrity_checkpoint=untrusted,
            )
        self.assertEqual(ledger.store.length, 0)

    def test_authority_resolver_rejects_reference_outside_trusted_root(self):
        stores = self.program_stores()
        raw, digest = build_program_integrity_checkpoint(
            stores=stores,
            baseline_manifest_sha256=SHA_A,
            checkpoint_directory=self.root,
            runtime_leakage_finding_count=0,
            promotion_population=self.default_promotion_population(),
        )
        external = self.root / "outside.json"
        external.write_bytes(raw)
        resolver = CheckpointAuthorityResolver(
            lambda: PublishedCheckpointReference(
                path=external,
                sha256=digest,
            ),
            trusted_checkpoint_root=self.root / "_checkpoints",
        )

        with self.assertRaises(IntegrityError):
            resolver.resolve(stores=stores)

    def test_authority_accepts_only_the_exact_recomputed_successor(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        checkpoint = self.checkpoint_for(experiment_ledger=ledger)
        receipt = ledger.preregister(
            "exact-successor",
            ExperimentLedgerTests.plan(),
            integrity_checkpoint=checkpoint,
        )
        exact = receipt.integrity

        CheckpointAuthorityResolver.validate_successor(checkpoint, exact)

        def forge_checkpoint(mutator):
            value = json.loads(exact.checkpoint_bytes)
            mutator(value)
            raw = (canonical_json(value) + "\n").encode("utf-8")
            return ProgramIntegritySuccessor(
                previous_checkpoint_sha256=(
                    exact.previous_checkpoint_sha256
                ),
                checkpoint_bytes=raw,
                checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                mutated_store=exact.mutated_store,
                record_hash=exact.record_hash,
                mutated=exact.mutated,
            )

        checkpoint_forgeries = {
            "promotion-population": lambda value: value[
                "promotion_population"
            ].__setitem__("truth_sha256", SHA_B),
            "runtime-leakage-count": lambda value: value.__setitem__(
                "runtime_leakage_finding_count", 7
            ),
            "store-anchor": lambda value: value["stores"][
                "experiment_ledger"
            ].__setitem__("sha256", SHA_B),
            "baseline": lambda value: value.__setitem__(
                "baseline_manifest_sha256", SHA_B
            ),
        }
        for name, mutator in checkpoint_forgeries.items():
            with self.subTest(name=name), self.assertRaises(IntegrityError):
                CheckpointAuthorityResolver.validate_successor(
                    checkpoint,
                    forge_checkpoint(mutator),
                )

        envelope_forgeries = {
            "parent": ProgramIntegritySuccessor(
                previous_checkpoint_sha256=SHA_B,
                checkpoint_bytes=exact.checkpoint_bytes,
                checkpoint_sha256=exact.checkpoint_sha256,
                mutated_store=exact.mutated_store,
                record_hash=exact.record_hash,
                mutated=exact.mutated,
            ),
            "digest": ProgramIntegritySuccessor(
                previous_checkpoint_sha256=(
                    exact.previous_checkpoint_sha256
                ),
                checkpoint_bytes=exact.checkpoint_bytes,
                checkpoint_sha256=SHA_B,
                mutated_store=exact.mutated_store,
                record_hash=exact.record_hash,
                mutated=exact.mutated,
            ),
            "store": ProgramIntegritySuccessor(
                previous_checkpoint_sha256=(
                    exact.previous_checkpoint_sha256
                ),
                checkpoint_bytes=exact.checkpoint_bytes,
                checkpoint_sha256=exact.checkpoint_sha256,
                mutated_store="taint_registry",
                record_hash=exact.record_hash,
                mutated=exact.mutated,
            ),
            "record-hash": ProgramIntegritySuccessor(
                previous_checkpoint_sha256=(
                    exact.previous_checkpoint_sha256
                ),
                checkpoint_bytes=exact.checkpoint_bytes,
                checkpoint_sha256=exact.checkpoint_sha256,
                mutated_store=exact.mutated_store,
                record_hash=SHA_B,
                mutated=exact.mutated,
            ),
            "mutated-flag": ProgramIntegritySuccessor(
                previous_checkpoint_sha256=(
                    exact.previous_checkpoint_sha256
                ),
                checkpoint_bytes=exact.checkpoint_bytes,
                checkpoint_sha256=exact.checkpoint_sha256,
                mutated_store=exact.mutated_store,
                record_hash=exact.record_hash,
                mutated=not exact.mutated,
            ),
        }
        for name, forged in envelope_forgeries.items():
            with self.subTest(name=name), self.assertRaises(IntegrityError):
                CheckpointAuthorityResolver.validate_successor(
                    checkpoint,
                    forged,
                )

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "requires O_NOFOLLOW")
    def test_program_lock_rejects_symlink_without_touching_target(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        checkpoint = self.checkpoint_for(experiment_ledger=ledger)
        target = self.root / "lock-target"
        target.write_text("sentinel", encoding="utf-8")
        lock_path = checkpoint.path.parent / ".program-integrity.lock"
        lock_path.symlink_to(target)

        with self.assertRaises(IntegrityError):
            ledger.preregister(
                "symlink-lock",
                ExperimentLedgerTests.plan(),
                integrity_checkpoint=checkpoint,
            )
        self.assertEqual(target.read_text(encoding="utf-8"), "sentinel")
        self.assertEqual(ledger.store.length, 0)

    def test_all_new_governed_mutations_require_a_current_checkpoint(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        state = CandidateStateStore(self.root / "candidates.jsonl")
        taint = TaintRegistry(self.root / "taint.jsonl")
        protected_path = self.root / "protected.jsonl"

        with self.assertRaises(ExperimentControlError):
            ProtectedAccessBudget(
                protected_path,
                maximum_accesses=1,
            )
        budget = self.create_budget(
            protected_path,
            maximum_accesses=1,
            experiment_ledger=ledger,
            candidate_state_ledger=state,
            taint_registry=taint,
        )
        with self.assertRaises(TypeError):
            ledger.preregister("missing-checkpoint", ExperimentLedgerTests.plan())
        with self.assertRaises(TypeError):
            ledger.record_result(
                "missing-checkpoint",
                ExperimentLedgerTests.result_evidence(),
                decision="reject",
                rationale="missing_checkpoint",
            )
        with self.assertRaises(TypeError):
            budget.record_access(
                "missing-checkpoint",
                candidate_sha256=SHA_A,
                aggregate_result={"total_score": 136.0},
                experiment_ledger=ledger,
                experiment_plan_record_hash=SHA_A,
                purpose="milestone-136",
            )
        with self.assertRaises(TypeError):
            taint.taint(
                "layout-family",
                reason="inspected",
                source="debug",
            )
        with self.assertRaises(TypeError):
            state.assess(
                "missing-checkpoint",
                candidate_id="candidate",
                candidate_sha256=SHA_A,
                decision="BLOCKED",
                aggregate_evidence={"score": 1.0},
            )
        gate = CandidatePromotionGate(
            state,
            experiment_ledger=ExperimentLedger(
                self.root / "gate-experiments.jsonl",
                protected_access_path=protected_path,
            ),
            taint_registry=taint,
        )
        with self.assertRaises(TypeError):
            gate.evaluate_and_record(
                "missing-checkpoint",
                candidate_id="candidate",
                candidate_sha256=SHA_A,
                experiment_result_record_hash=SHA_A,
            )

        none_checkpoint_calls = (
            lambda: ledger.preregister(
                "none-plan",
                ExperimentLedgerTests.plan(),
                integrity_checkpoint=None,
            ),
            lambda: ledger.record_result(
                "none-result",
                ExperimentLedgerTests.result_evidence(),
                decision="reject",
                rationale="none_checkpoint",
                integrity_checkpoint=None,
            ),
            lambda: budget.record_access(
                "none-access",
                candidate_sha256=SHA_A,
                aggregate_result={"total_score": 136.0},
                experiment_ledger=ledger,
                experiment_plan_record_hash=SHA_A,
                integrity_checkpoint=None,
                purpose="milestone-136",
            ),
            lambda: taint.taint(
                "none-taint",
                reason="inspection",
                source="test",
                integrity_checkpoint=None,
            ),
            lambda: state.assess(
                "none-assessment",
                candidate_id="candidate",
                candidate_sha256=SHA_A,
                decision="BLOCKED",
                aggregate_evidence={"score": 1.0},
                integrity_checkpoint=None,
            ),
            lambda: gate.evaluate_and_record(
                "none-gate",
                candidate_id="candidate",
                candidate_sha256=SHA_A,
                experiment_result_record_hash=SHA_A,
                integrity_checkpoint=None,
            ),
        )
        for index, call in enumerate(none_checkpoint_calls):
            with self.subTest(index=index), self.assertRaises(
                ExperimentControlError
            ):
                call()


class ExperimentLedgerTests(TemporaryDirectoryTestCase):
    @staticmethod
    def plan(**overrides):
        plan = {
            "changed_files": [
                "mib_pipeline/resolution.py",
                "tests/test_resolution.py",
            ],
            "evidence_label": "public_grouped_robustness_not_unseen",
            "evaluator_sha256": SHA_A,
            "expected_record_count": 5,
            "hypothesis_sha256": SHA_A,
            "input_tree_sha256": SHA_A,
            "parent_commit_sha": "c" * 40,
            "primary_variable_sha256": SHA_B,
            "runtime_contract_sha256": SHA_A,
            "split_manifest_sha256": SHA_A,
            "truth_sha256": SHA_A,
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
                "runtime_limits_verified": True,
                "runtime_leakage_clean": True,
            },
            "confusion_counts": {"approved_to_review_count": 0},
            "evaluator_sha256": SHA_A,
            "expected_record_count": 5,
            "field_metrics": {
                "all_fields": {"count": 1, "score_delta": 0.0}
            },
            "input_tree_sha256": SHA_A,
            "metrics": {
                "calibration_score": 18.0,
                "candidate_image_bytes": 1024,
                "candidate_max_model_artifact_bytes": 256,
                "candidate_model_bytes": 512,
                "catastrophic_false_approvals": 0,
                "classification_score": 72.0,
                "extraction_score": 46.0,
                "invalid_records": 0,
                "missing_records": 0,
                "output_bytes": 256,
                "peak_container_memory_bytes": 2048,
                "peak_rss_bytes": 2048,
                "process_cpu_seconds": 2.0,
                "record_count": 5,
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
            "fold_weights": [1] * 15,
            "regression_counts": {"adversarial": 0, "golden": 0},
            "repeat_count": 3,
            "repeat_scores": [0.09, 0.09, 0.06],
            "runtime_contract_sha256": SHA_A,
            "runtime_evidence_sha256": SHA_A,
            "split_manifest_sha256": SHA_A,
            "truth_sha256": SHA_A,
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

        plan_record = self.governed_preregister(
            ledger,
            "exp-two-stage",
            self.plan(changed_files=list(reversed(self.plan()["changed_files"]))),
            expected_head=GENESIS_HASH,
        )
        result_record = self.governed_record_result(
            ledger,
            "exp-two-stage",
            self.result_evidence(),
            decision="reject",
            rationale="grouped_gate_failed",
            expected_head=plan_record["record_hash"],
        )

        self.assertIsInstance(plan_record, GovernedMutationReceipt)
        self.assertIsInstance(result_record, GovernedMutationReceipt)
        self.assertTrue(plan_record.mutated)
        self.assertTrue(result_record.mutated)
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

    def test_result_binds_population_contract_but_runtime_evidence_is_post_run(self):
        bound_hashes = (
            "evaluator_sha256",
            "input_tree_sha256",
            "runtime_contract_sha256",
            "split_manifest_sha256",
            "truth_sha256",
        )
        for index, binding in enumerate(bound_hashes):
            with self.subTest(binding=binding):
                ledger = ExperimentLedger(
                    self.root / f"binding-{index}.jsonl"
                )
                self.governed_preregister(
                    ledger, f"binding-{index}", self.plan()
                )
                evidence = self.result_evidence()
                evidence[binding] = SHA_C
                with self.assertRaises(ExperimentControlError):
                    self.governed_record_result(
                        ledger,
                        f"binding-{index}",
                        evidence,
                        decision="reject",
                        rationale="population_binding_mismatch",
                    )

        ledger = ExperimentLedger(self.root / "record-count-binding.jsonl")
        self.governed_preregister(
            ledger, "record-count-binding", self.plan()
        )
        evidence = self.result_evidence(expected_record_count=6)
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "record-count-binding",
                evidence,
                decision="reject",
                rationale="record_count_binding_mismatch",
            )

        ledger = ExperimentLedger(self.root / "runtime-evidence.jsonl")
        plan_record = self.governed_preregister(
            ledger, "runtime-evidence", self.plan()
        )
        self.assertNotIn(
            "runtime_evidence_sha256",
            plan_record["payload"]["plan"],
        )
        result = self.governed_record_result(
            ledger,
            "runtime-evidence",
            self.result_evidence(runtime_evidence_sha256=SHA_C),
            decision="reject",
            rationale="runtime_evidence_recorded_after_run",
        )
        self.assertEqual(
            result["payload"]["evidence"]["runtime_evidence_sha256"],
            SHA_C,
        )

    def test_two_stage_order_conflicts_and_duplicate_result_fail_closed(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        evidence = self.result_evidence()

        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "exp-two-stage",
                evidence,
                decision="reject",
                rationale="missing_plan",
            )

        first_plan = self.governed_preregister(
            ledger, "exp-two-stage", self.plan()
        )
        identical_retry = self.governed_preregister(
            ledger, "exp-two-stage", self.plan()
        )
        self.assertEqual(first_plan, identical_retry)
        self.assertTrue(first_plan.mutated)
        self.assertFalse(identical_retry.mutated)
        self.assertEqual(
            identical_retry.previous_checkpoint_sha256,
            identical_retry.next_checkpoint_sha256,
        )
        with self.assertRaises(ExperimentControlError):
            self.governed_preregister(
                ledger,
                "exp-two-stage",
                self.plan(hypothesis_sha256="d" * 64),
            )
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "exp-two-stage",
                evidence,
                decision="reject",
                rationale="A free-form result narrative could leak diagnostics.",
            )

        self.governed_record_result(
            ledger,
            "exp-two-stage",
            evidence,
            decision="reject",
            rationale="grouped_gate_failed",
        )
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "exp-two-stage",
                evidence,
                decision="reject",
                rationale="grouped_gate_failed",
            )
        self.assertEqual(ledger.store.length, 2)

    def test_result_requires_full_contract_and_adoption_requires_clean_gates(self):
        ledger = ExperimentLedger(self.root / "experiments.jsonl")
        self.governed_preregister(ledger, "strict-result", self.plan())

        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "strict-result",
                {"status": "failed"},
                decision="adopt",
                rationale="insufficient_contract",
            )
        unsafe = self.result_evidence()
        unsafe["metrics"]["catastrophic_false_approvals"] = 1
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "strict-result",
                unsafe,
                decision="adopt",
                rationale="unsafe_candidate",
            )
        false_extra_check = self.result_evidence()
        false_extra_check["checks"]["baseline_verified"] = False
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
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
            self.governed_record_result(
                ledger,
                "strict-result",
                concentrated,
                decision="adopt",
                rationale="single_fold_concentration",
            )
        negative_fold = self.result_evidence(
            fold_deltas=[
                0.4,
                0.1,
                -0.01,
                0.05,
                0.01,
                0.1,
                0.4,
                0.05,
                0.01,
                0.0,
                0.2,
                0.1,
                0.05,
                0.01,
                0.0,
            ],
            repeat_scores=[0.11, 0.112, 0.072],
        )
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "strict-result",
                negative_fold,
                decision="adopt",
                rationale="negative_fold",
            )
        inconsistent_total = self.result_evidence()
        inconsistent_total["metrics"]["total_score"] = 135.0
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "strict-result",
                inconsistent_total,
                decision="reject",
                rationale="component_mismatch",
            )

        accepted = self.governed_record_result(
            ledger,
            "strict-result",
            self.result_evidence(),
            decision="adopt",
            rationale="all_hard_gates_passed",
        )
        self.assertEqual(accepted["payload"]["decision"], "adopt")
        self.assertEqual(ledger.store.length, 2)

    def test_adoption_rejects_missing_folds_population_mismatch_and_runtime_limits(self):
        scenarios = {}
        missing_folds = self.result_evidence()
        for key in (
            "fold_count",
            "fold_deltas",
            "fold_weights",
            "repeat_count",
            "repeat_scores",
        ):
            del missing_folds[key]
        scenarios["missing-folds"] = missing_folds

        population_mismatch = self.result_evidence()
        population_mismatch["metrics"]["record_count"] = 6
        population_mismatch["fold_weights"] = [2, 1, 1, 1, 1] * 3
        scenarios["population-mismatch"] = population_mismatch

        runtime_overage = self.result_evidence()
        runtime_overage["metrics"]["candidate_image_bytes"] = 4 * 1024**3 + 1
        scenarios["runtime-overage"] = runtime_overage

        runtime_vacuous = self.result_evidence()
        for metric in (
            "candidate_image_bytes",
            "output_bytes",
            "peak_container_memory_bytes",
            "peak_rss_bytes",
            "process_cpu_seconds",
            "runtime_seconds",
        ):
            runtime_vacuous["metrics"][metric] = 0
        scenarios["runtime-vacuous"] = runtime_vacuous

        for suffix, evidence in scenarios.items():
            with self.subTest(suffix=suffix):
                ledger = ExperimentLedger(
                    self.root / f"{suffix}.jsonl"
                )
                self.governed_preregister(
                    ledger,
                    suffix,
                    self.plan(evidence_label="protected"),
                )
                with self.assertRaises(ExperimentControlError):
                    self.governed_record_result(
                        ledger,
                        suffix,
                        evidence,
                        decision="adopt",
                        rationale="hard_gate_failed",
                    )

    def test_adoption_requires_exact_five_by_three_grouped_shape(self):
        scenarios = (
            (
                "six-by-three",
                self.plan(expected_record_count=6),
                self.result_evidence(
                    expected_record_count=6,
                    fold_count=6,
                    fold_deltas=[0.1] * 18,
                    fold_weights=[1] * 18,
                    repeat_count=3,
                    repeat_scores=[0.1] * 3,
                ),
            ),
            (
                "five-by-four",
                self.plan(),
                self.result_evidence(
                    fold_count=5,
                    fold_deltas=[0.1] * 20,
                    fold_weights=[1] * 20,
                    repeat_count=4,
                    repeat_scores=[0.1] * 4,
                ),
            ),
        )
        scenarios[0][2]["metrics"]["record_count"] = 6

        for suffix, plan, evidence in scenarios:
            with self.subTest(suffix=suffix):
                ledger = ExperimentLedger(
                    self.root / f"shape-{suffix}.jsonl"
                )
                self.governed_preregister(ledger, suffix, plan)
                with self.assertRaises(ExperimentControlError):
                    self.governed_record_result(
                        ledger,
                        suffix,
                        evidence,
                        decision="adopt",
                        rationale="noncanonical_group_shape",
                    )

    def test_protected_result_requires_matching_access_record_and_candidate(self):
        protected_path = self.root / "protected.jsonl"
        budget = self.create_budget(protected_path, maximum_accesses=2)
        recorded_evidence = self.result_evidence()
        protected_plan = self.plan(evidence_label="protected")
        ledger = ExperimentLedger(
            self.root / "experiments.jsonl",
            protected_access_path=protected_path,
        )
        plan_record = self.governed_preregister(
            ledger, "protected", protected_plan
        )
        self.governed_record_access(
            budget,
            "milestone-access",
            candidate_sha256=SHA_B,
            aggregate_result=recorded_evidence,
            experiment_ledger=ledger,
            experiment_plan_record_hash=plan_record["record_hash"],
            purpose="milestone-candidate",
        )
        access_hash = budget.store.verify()[-1]["record_hash"]

        missing_store = ExperimentLedger(self.root / "missing-store.jsonl")
        self.governed_preregister(
            missing_store, "protected", protected_plan
        )
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                missing_store,
                "protected",
                self.result_evidence(
                    protected_access_record_hash=access_hash
                ),
                decision="reject",
                rationale="protected_gate_rejected",
            )

        wrong_candidate = self.result_evidence(
            candidate_artifact_sha256=SHA_A,
            protected_access_record_hash=access_hash,
        )
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "protected",
                wrong_candidate,
                decision="reject",
                rationale="candidate_binding_mismatch",
            )
        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
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
            self.governed_record_result(
                ledger,
                "protected",
                invented_aggregate,
                decision="reject",
                rationale="unrecorded_aggregate",
            )

        result = self.governed_record_result(
            ledger,
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

    def test_protected_result_requires_exact_numeric_aggregate_shape(self):
        protected_path = self.root / "protected-numeric.jsonl"
        budget = self.create_budget(protected_path, maximum_accesses=1)
        ledger = ExperimentLedger(
            self.root / "experiments-numeric.jsonl",
            protected_access_path=protected_path,
        )
        plan_record = self.governed_preregister(
            ledger,
            "protected-numeric",
            self.plan(evidence_label="protected"),
        )
        accessed = self.result_evidence()
        accessed["metrics"]["runtime_seconds"] = 3
        self.governed_record_access(
            budget,
            "numeric-shape-access",
            candidate_sha256=SHA_B,
            aggregate_result=accessed,
            experiment_ledger=ledger,
            experiment_plan_record_hash=plan_record["record_hash"],
            purpose="milestone-136",
        )
        access_record = budget.store.verify()[-1]
        result_evidence = copy.deepcopy(accessed)
        result_evidence["metrics"]["runtime_seconds"] = 3.0
        result_evidence["protected_access_record_hash"] = access_record[
            "record_hash"
        ]

        with self.assertRaises(ExperimentControlError):
            self.governed_record_result(
                ledger,
                "protected-numeric",
                result_evidence,
                decision="reject",
                rationale="numeric_shape_mismatch",
            )

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
        self.governed_preregister(
            ledger,
            "protected-forgery",
            self.plan(evidence_label="protected"),
        )

        with self.assertRaises(IntegrityError):
            self.governed_record_result(
                ledger,
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
                self.governed_preregister(
                    ledger, f"unsafe-plan-{index}", plan
                )
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
            self.governed_preregister(ledger, "legacy", self.plan())

        self.governed_preregister(ledger, "planned", self.plan())
        with self.assertRaises(ExperimentControlError):
            ledger.record("planned", {"total_score": 131.0})
        self.assertEqual(len(ledger.experiments()), 1)
        self.assertEqual(len(ledger.plans()), 1)

    def test_concurrent_result_writers_append_exactly_one_result(self):
        path = self.root / "experiments.jsonl"
        self.governed_preregister(
            ExperimentLedger(path), "concurrent", self.plan()
        )
        barrier = threading.Barrier(2)

        def attempt(index):
            ledger = ExperimentLedger(path)
            barrier.wait(timeout=5)
            try:
                self.governed_record_result(
                    ledger,
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
        receipt = self.governed_taint(
            registry,
            "layout-family-07",
            reason="inspected",
            source="debug-session",
        )
        reopened = TaintRegistry(registry.store.path)

        self.assertIsInstance(receipt, GovernedMutationReceipt)
        self.assertTrue(receipt.mutated)
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
    def _experiment_binding(self, suffix: str):
        ledger = ExperimentLedger(self.root / f"experiments-{suffix}.jsonl")
        plan = self.governed_preregister(
            ledger,
            f"protected-{suffix}",
            ExperimentLedgerTests.plan(evidence_label="protected"),
        )
        return {
            "experiment_ledger": ledger,
            "experiment_plan_record_hash": plan["record_hash"],
        }

    def test_identical_retry_is_idempotent_and_does_not_consume_budget(self):
        budget = self.create_budget(
            self.root / "protected.jsonl", maximum_accesses=2
        )
        request = {
            "candidate_sha256": SHA_A,
            "aggregate_result": {"total_score": 136.2, "count": 1000},
            **self._experiment_binding("retry"),
            "purpose": "milestone-136",
        }

        first = self.governed_record_access(
            budget, "access-1", **request
        )
        retry = self.governed_record_access(
            budget, "access-1", **request
        )

        self.assertEqual(first, retry)
        self.assertIsInstance(first, GovernedMutationReceipt)
        self.assertTrue(first.mutated)
        self.assertFalse(retry.mutated)
        self.assertEqual(
            retry.previous_checkpoint_sha256,
            retry.next_checkpoint_sha256,
        )
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
        budget = self.create_budget(
            self.root / "protected.jsonl", maximum_accesses=2
        )
        binding = self._experiment_binding("mismatch")
        self.governed_record_access(
            budget,
            "access-1",
            candidate_sha256=SHA_A,
            aggregate_result={"total_score": 136},
            **binding,
            purpose="milestone-136",
        )

        with self.assertRaises(ExperimentControlError):
            self.governed_record_access(
                budget,
                "access-1",
                candidate_sha256=SHA_A,
                aggregate_result={"total_score": 136.0},
                **binding,
                purpose="milestone-136",
            )

    def test_access_requires_plan_to_exist_before_measurement(self):
        budget = self.create_budget(
            self.root / "protected.jsonl", maximum_accesses=1
        )
        ledger = ExperimentLedger(self.root / "empty-experiments.jsonl")

        with self.assertRaises(ExperimentControlError):
            self.governed_record_access(
                budget,
                "access-before-plan",
                candidate_sha256=SHA_A,
                aggregate_result={"total_score": 136.0},
                experiment_ledger=ledger,
                experiment_plan_record_hash=SHA_A,
                purpose="milestone-136",
            )

    def test_exhaustion_and_case_level_results_are_rejected(self):
        budget = self.create_budget(
            self.root / "protected.jsonl", maximum_accesses=1
        )
        with self.assertRaises(LeakageError):
            self.governed_record_access(
                budget,
                "unsafe",
                candidate_sha256=SHA_A,
                aggregate_result={"case_id": "MIB-000001"},
                **self._experiment_binding("unsafe"),
                purpose="debug",
            )
        first_binding = self._experiment_binding("first")
        self.governed_record_access(
            budget,
            "access-1",
            candidate_sha256=SHA_A,
            aggregate_result={"total_score": 142.0},
            **first_binding,
            purpose="milestone-142",
        )
        with self.assertRaises(BudgetExhaustedError):
            self.governed_record_access(
                budget,
                "access-2",
                candidate_sha256=SHA_B,
                aggregate_result={"total_score": 146.0},
                **self._experiment_binding("exhausted"),
                purpose="milestone-146",
            )

    def test_configuration_is_persisted_and_immutable(self):
        path = self.root / "protected.jsonl"
        created = self.create_budget(path, maximum_accesses=2)
        self.assertIsInstance(
            created.initialization_receipt,
            GovernedMutationReceipt,
        )
        self.assertTrue(created.initialization_receipt.mutated)

        with self.assertRaises(IntegrityError):
            ProtectedAccessBudget(path, maximum_accesses=3)

        reopened = ProtectedAccessBudget(path, maximum_accesses=2)
        self.assertIsNone(reopened.initialization_receipt)
        self.assertEqual(reopened.maximum_accesses, 2)
        self.assertEqual(reopened.remaining, 2)

    def test_concurrent_budget_one_accepts_exactly_one_access(self):
        path = self.root / "protected.jsonl"
        workers = 8
        barrier = threading.Barrier(workers)
        binding = self._experiment_binding("concurrent")
        initialized = self.create_budget(path, maximum_accesses=1)
        checkpoint = self.checkpoint_for(
            experiment_ledger=binding["experiment_ledger"],
            protected_access_ledger=initialized,
        )

        def attempt(index):
            budget = ProtectedAccessBudget(path, maximum_accesses=1)
            barrier.wait(timeout=5)
            try:
                budget.record_access(
                    f"access-{index}",
                    candidate_sha256=SHA_A if index % 2 == 0 else SHA_B,
                    aggregate_result={"total_score": 130.0 + index / 100},
                    **binding,
                    integrity_checkpoint=checkpoint,
                    purpose=f"milestone-{index}",
                )
                return "accepted"
            except (BudgetExhaustedError, CompareAndSwapError, IntegrityError):
                return "rejected"

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(attempt, range(workers)))

        self.assertEqual(results.count("accepted"), 1)
        self.assertEqual(results.count("rejected"), workers - 1)
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
    @staticmethod
    def _seed_passing(
        state: CandidateStateStore,
        *,
        candidate_sha256: str = SHA_A,
    ):
        return state.store.append(
            {
                "event": "candidate_assessment",
                "assessment_id": "governed-baseline",
                "candidate_id": "candidate-baseline",
                "candidate_sha256": candidate_sha256,
                "decision": "PASSED",
                "aggregate_evidence": {
                    "metrics": {"total_score": 130.37},
                    "promotion_gate_verified": True,
                },
            }
        )

    def _protected_result(
        self,
        *,
        suffix: str,
        total_score: float = 136.0,
        decision: str = "adopt",
        candidate_sha256: str = SHA_B,
        baseline_sha256: str = SHA_A,
        purpose: str = "milestone-136",
        forged_aggregate_mismatch: bool = False,
        population_overrides=None,
    ):
        protected_path = self.root / f"protected-{suffix}.jsonl"
        ledger = ExperimentLedger(
            self.root / f"experiments-{suffix}.jsonl",
            protected_access_path=protected_path,
        )
        taint_registry = TaintRegistry(
            self.root / f"taint-{suffix}.jsonl"
        )
        budget = self.create_budget(
            protected_path,
            maximum_accesses=5,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        )
        plan_checkpoint = self.checkpoint_for(
            experiment_ledger=ledger,
            protected_access_ledger=budget,
            taint_registry=taint_registry,
        )
        population_overrides = dict(population_overrides or {})
        plan_record = ledger.preregister(
            f"promotion-{suffix}",
            ExperimentLedgerTests.plan(
                evidence_label="protected",
                **population_overrides,
            ),
            integrity_checkpoint=plan_checkpoint,
        )
        evidence = ExperimentLedgerTests.result_evidence(
            baseline_artifact_sha256=baseline_sha256,
            candidate_artifact_sha256=candidate_sha256,
            **population_overrides,
        )
        evidence["metrics"].update(
            {
                "calibration_score": total_score - 130.0,
                "classification_score": 80.0,
                "extraction_score": 50.0,
                "total_score": total_score,
            }
        )
        expected_record_count = population_overrides.get(
            "expected_record_count",
            5,
        )
        if expected_record_count != 5:
            if expected_record_count < 5:
                raise AssertionError(
                    "protected grouped fixture needs at least five records"
                )
            quotient, remainder = divmod(expected_record_count, 5)
            repeat_weights = [
                quotient + (index < remainder)
                for index in range(5)
            ]
            evidence["metrics"]["record_count"] = expected_record_count
            evidence["fold_weights"] = repeat_weights * 3
            evidence["repeat_scores"] = [
                sum(
                    delta * weight
                    for delta, weight in zip(
                        evidence["fold_deltas"][offset : offset + 5],
                        repeat_weights,
                    )
                )
                / expected_record_count
                for offset in (0, 5, 10)
            ]
        access_checkpoint = self.checkpoint_for(
            experiment_ledger=ledger,
            protected_access_ledger=budget,
            taint_registry=taint_registry,
        )
        budget.record_access(
            f"access-{suffix}",
            candidate_sha256=candidate_sha256,
            aggregate_result=evidence,
            experiment_ledger=ledger,
            experiment_plan_record_hash=plan_record["record_hash"],
            integrity_checkpoint=access_checkpoint,
            purpose=purpose,
        )
        access_hash = budget.store.verify()[-1]["record_hash"]
        evidence["protected_access_record_hash"] = access_hash
        if forged_aggregate_mismatch:
            evidence["metrics"]["runtime_seconds"] += 1.0
            result_record = ledger.store.append(
                {
                    "decision": decision,
                    "event": "experiment_result",
                    "evidence": evidence,
                    "experiment_id": f"promotion-{suffix}",
                    "plan_record_hash": plan_record["record_hash"],
                    "rationale": "forged_aggregate",
                }
            )
        else:
            result_checkpoint = self.checkpoint_for(
                experiment_ledger=ledger,
                protected_access_ledger=budget,
                taint_registry=taint_registry,
            )
            result_record = ledger.record_result(
                f"promotion-{suffix}",
                evidence,
                decision=decision,
                integrity_checkpoint=result_checkpoint,
                rationale="promotion_evaluated",
            )
        return ledger, result_record, taint_registry

    def test_blocked_assessment_does_not_replace_latest_passing_candidate(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        passing = self._seed_passing(state)["payload"]
        self.governed_assess(
            state,
            "assessment-2",
            candidate_id="candidate-131-unsafe",
            candidate_sha256=SHA_B,
            decision="BLOCKED",
            aggregate_evidence={"score": 131.0, "false_approvals": 1},
        )

        self.assertEqual(
            state.latest_passing(
                integrity_checkpoint=self.checkpoint_for(
                    candidate_state_ledger=state
                )
            ),
            passing,
        )
        self.assertEqual(len(state.assessments()), 2)

    def test_assessment_retry_is_idempotent_but_mismatch_is_rejected(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        request = {
            "candidate_id": "candidate-130",
            "candidate_sha256": SHA_A,
            "decision": "BLOCKED",
            "aggregate_evidence": {"score": 130.37},
        }
        first = self.governed_assess(state, "assessment-1", **request)
        retry = self.governed_assess(state, "assessment-1", **request)
        self.assertEqual(first, retry)
        self.assertIsInstance(first, GovernedMutationReceipt)
        self.assertTrue(first.mutated)
        self.assertFalse(retry.mutated)
        self.assertEqual(state.store.length, 1)

        with self.assertRaises(ExperimentControlError):
            self.governed_assess(
                state,
                "assessment-1",
                candidate_id="candidate-other",
                candidate_sha256=SHA_B,
                decision="BLOCKED",
                aggregate_evidence={"score": 100.0},
            )

    def test_case_level_candidate_evidence_is_rejected(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        with self.assertRaises(LeakageError):
            self.governed_assess(
                state,
                "assessment-unsafe",
                candidate_id="candidate-unsafe",
                candidate_sha256=SHA_A,
                decision="BLOCKED",
                aggregate_evidence={"prediction_rows": ["MIB-000001"]},
            )

    def test_direct_pass_is_rejected_and_milestone_score_is_derived(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        with self.assertRaises(ExperimentControlError):
            self.governed_assess(
                state,
                "forged-pass",
                candidate_id="candidate-forged",
                candidate_sha256=SHA_A,
                decision="PASSED",
                aggregate_evidence={"score": 150.0},
            )

        self._seed_passing(state)
        ledger, result, taint_registry = self._protected_result(
            suffix="below",
            total_score=135.999,
        )
        gate = CandidatePromotionGate(
            state,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        )
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=state,
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
            taint_registry=taint_registry,
        )
        blocked = gate.evaluate_and_record(
            "gate-block",
            candidate_id="candidate-below",
            candidate_sha256=SHA_B,
            experiment_result_record_hash=result["record_hash"],
            integrity_checkpoint=checkpoint,
        )

        self.assertEqual(blocked["decision"], "BLOCKED")
        self.assertEqual(blocked["aggregate_evidence"]["hard_gate_failure_count"], 1)
        self.assertFalse(
            blocked["aggregate_evidence"]["gate_results"][
                "milestone_score_reached"
            ]
        )
        published = self.publish_successor(
            checkpoint,
            blocked.integrity,
        )
        self.assertEqual(
            state.latest_passing(
                integrity_checkpoint=published
            )["candidate_sha256"],
            SHA_A,
        )

    def test_gate_persists_exact_plan_result_access_and_milestone_bindings(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        self._seed_passing(state)
        ledger, result, taint_registry = self._protected_result(
            suffix="pass",
            total_score=136.0,
        )
        taint_checkpoint = self.checkpoint_for(
            candidate_state_ledger=state,
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
            taint_registry=taint_registry,
        )
        taint_receipt = taint_registry.taint(
            "layout-family-inspected",
            reason="inspected",
            source="debug-session",
            integrity_checkpoint=taint_checkpoint,
        )
        gate = CandidatePromotionGate(
            state,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        )
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=state,
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
            taint_registry=taint_registry,
        )

        passing = gate.evaluate_and_record(
            "gate-pass",
            candidate_id="candidate-136",
            candidate_sha256=SHA_B,
            experiment_result_record_hash=result["record_hash"],
            integrity_checkpoint=checkpoint,
        )
        self.assertEqual(passing["decision"], "PASSED")
        evidence = passing["aggregate_evidence"]
        self.assertEqual(
            evidence["experiment_result_record_hash"], result["record_hash"]
        )
        self.assertEqual(
            evidence["experiment_plan_record_hash"],
            result["payload"]["plan_record_hash"],
        )
        self.assertEqual(evidence["milestone_total_score"], 136.0)
        self.assertEqual(
            evidence["protected_access_record_hash"],
            result["payload"]["evidence"]["protected_access_record_hash"],
        )
        self.assertEqual(
            evidence["taint_registry_head_sha256"],
            taint_receipt.record_hash,
        )
        self.assertIsInstance(passing, GovernedMutationReceipt)
        self.assertTrue(passing.mutated)
        self.assertNotEqual(
            passing.next_checkpoint_sha256,
            passing.previous_checkpoint_sha256,
        )
        self.assertTrue(all(evidence["gate_results"].values()))

    def test_milestone_sequence_uses_persisted_milestone_after_score_jump(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        self._seed_passing(state)

        first_ledger, first_result, first_taint = self._protected_result(
            suffix="jump-first",
            total_score=142.0,
            candidate_sha256=SHA_B,
            baseline_sha256=SHA_A,
            purpose="milestone-136",
        )
        first = CandidatePromotionGate(
            state,
            experiment_ledger=first_ledger,
            taint_registry=first_taint,
        ).evaluate_and_record(
            "jump-first",
            candidate_id="candidate-jump-first",
            candidate_sha256=SHA_B,
            experiment_result_record_hash=first_result["record_hash"],
            integrity_checkpoint=self.checkpoint_for(
                candidate_state_ledger=state,
                experiment_ledger=first_ledger,
                protected_access_ledger=(
                    first_ledger.protected_access_store
                ),
                taint_registry=first_taint,
            ),
        )
        self.assertEqual(first["decision"], "PASSED")
        self.assertEqual(
            first["aggregate_evidence"]["milestone_total_score"],
            136.0,
        )

        second_ledger, second_result, second_taint = self._protected_result(
            suffix="jump-second",
            total_score=143.0,
            candidate_sha256=SHA_C,
            baseline_sha256=SHA_B,
            purpose="milestone-142",
        )
        second = CandidatePromotionGate(
            state,
            experiment_ledger=second_ledger,
            taint_registry=second_taint,
        ).evaluate_and_record(
            "jump-second",
            candidate_id="candidate-jump-second",
            candidate_sha256=SHA_C,
            experiment_result_record_hash=second_result["record_hash"],
            integrity_checkpoint=self.checkpoint_for(
                candidate_state_ledger=state,
                experiment_ledger=second_ledger,
                protected_access_ledger=(
                    second_ledger.protected_access_store
                ),
                taint_registry=second_taint,
            ),
        )
        self.assertEqual(second["decision"], "PASSED")
        self.assertEqual(
            second["aggregate_evidence"]["milestone_total_score"],
            142.0,
        )
        self.assertTrue(
            second["aggregate_evidence"]["gate_results"][
                "milestone_sequence_verified"
            ]
        )

    def test_cross_milestone_population_drift_is_blocked(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        self._seed_passing(state)
        first_ledger, first_result, first_taint = self._protected_result(
            suffix="population-first",
            total_score=142.0,
            candidate_sha256=SHA_B,
            baseline_sha256=SHA_A,
            purpose="milestone-136",
        )
        first = CandidatePromotionGate(
            state,
            experiment_ledger=first_ledger,
            taint_registry=first_taint,
        ).evaluate_and_record(
            "population-first",
            candidate_id="candidate-population-first",
            candidate_sha256=SHA_B,
            experiment_result_record_hash=first_result["record_hash"],
            integrity_checkpoint=self.checkpoint_for(
                candidate_state_ledger=state,
                experiment_ledger=first_ledger,
                protected_access_ledger=(
                    first_ledger.protected_access_store
                ),
                taint_registry=first_taint,
            ),
        )
        self.assertEqual(first["decision"], "PASSED")

        changed_population = {
            "evaluator_sha256": SHA_C,
            "expected_record_count": 6,
            "input_tree_sha256": SHA_C,
            "runtime_contract_sha256": SHA_C,
            "split_manifest_sha256": SHA_C,
            "truth_sha256": SHA_C,
        }
        second_ledger, second_result, second_taint = self._protected_result(
            suffix="population-second",
            total_score=143.0,
            candidate_sha256=SHA_C,
            baseline_sha256=SHA_B,
            purpose="milestone-142",
            population_overrides=changed_population,
        )
        second = CandidatePromotionGate(
            state,
            experiment_ledger=second_ledger,
            taint_registry=second_taint,
        ).evaluate_and_record(
            "population-second",
            candidate_id="candidate-population-second",
            candidate_sha256=SHA_C,
            experiment_result_record_hash=second_result["record_hash"],
            integrity_checkpoint=self.checkpoint_for(
                candidate_state_ledger=state,
                experiment_ledger=second_ledger,
                protected_access_ledger=(
                    second_ledger.protected_access_store
                ),
                taint_registry=second_taint,
            ),
        )

        self.assertEqual(second["decision"], "BLOCKED")
        self.assertFalse(
            second["aggregate_evidence"]["gate_results"][
                "population_binding_verified"
            ]
        )

    def test_checkpoint_runtime_leakage_findings_block_promotion(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        self._seed_passing(state)
        ledger, result, taint_registry = self._protected_result(
            suffix="checkpoint-leakage"
        )
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=state,
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
            taint_registry=taint_registry,
            runtime_leakage_finding_count=7,
        )
        assessment = CandidatePromotionGate(
            state,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        ).evaluate_and_record(
            "checkpoint-leakage",
            candidate_id="candidate-checkpoint-leakage",
            candidate_sha256=SHA_B,
            experiment_result_record_hash=result["record_hash"],
            integrity_checkpoint=checkpoint,
        )

        self.assertEqual(assessment["decision"], "BLOCKED")
        self.assertEqual(
            assessment["aggregate_evidence"][
                "checkpoint_runtime_leakage_finding_count"
            ],
            7,
        )
        self.assertFalse(
            assessment["aggregate_evidence"]["gate_results"][
                "checkpoint_runtime_leakage_clean"
            ]
        )

    def test_same_candidate_digest_cannot_advance_another_milestone(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        self._seed_passing(state)
        first_ledger, first_result, first_taint = self._protected_result(
            suffix="same-digest-first",
            total_score=142.0,
            candidate_sha256=SHA_B,
            baseline_sha256=SHA_A,
            purpose="milestone-136",
        )
        first = CandidatePromotionGate(
            state,
            experiment_ledger=first_ledger,
            taint_registry=first_taint,
        ).evaluate_and_record(
            "same-digest-first",
            candidate_id="candidate-same-first",
            candidate_sha256=SHA_B,
            experiment_result_record_hash=first_result["record_hash"],
            integrity_checkpoint=self.checkpoint_for(
                candidate_state_ledger=state,
                experiment_ledger=first_ledger,
                protected_access_ledger=(
                    first_ledger.protected_access_store
                ),
                taint_registry=first_taint,
            ),
        )
        self.assertEqual(first["decision"], "PASSED")

        second_ledger, second_result, second_taint = self._protected_result(
            suffix="same-digest-second",
            total_score=143.0,
            candidate_sha256=SHA_B,
            baseline_sha256=SHA_B,
            purpose="milestone-142",
        )
        second = CandidatePromotionGate(
            state,
            experiment_ledger=second_ledger,
            taint_registry=second_taint,
        ).evaluate_and_record(
            "same-digest-second",
            candidate_id="candidate-same-second",
            candidate_sha256=SHA_B,
            experiment_result_record_hash=second_result["record_hash"],
            integrity_checkpoint=self.checkpoint_for(
                candidate_state_ledger=state,
                experiment_ledger=second_ledger,
                protected_access_ledger=(
                    second_ledger.protected_access_store
                ),
                taint_registry=second_taint,
            ),
        )

        self.assertEqual(second["decision"], "BLOCKED")
        gates = second["aggregate_evidence"]["gate_results"]
        self.assertFalse(gates["candidate_changed_verified"])
        self.assertFalse(gates["candidate_digest_unique_verified"])

    def test_protected_access_snapshot_must_precede_selected_result(self):
        protected_path = self.root / "protected-ordering.jsonl"
        budget = self.create_budget(protected_path, maximum_accesses=1)
        aggregate = ExperimentLedgerTests.result_evidence()
        access_record = budget.store.append(
            {
                "event": "protected_access",
                "access_id": "ordering-access",
                "candidate_sha256": SHA_B,
                "aggregate_result": aggregate,
                "experiment_ledger_head_sha256": SHA_C,
                "experiment_plan_record_hash": SHA_A,
                "purpose": "milestone-136",
            }
        )
        ledger = ExperimentLedger(
            self.root / "experiments-ordering.jsonl",
            protected_access_path=protected_path,
        )
        gate = CandidatePromotionGate(
            CandidateStateStore(self.root / "candidates-ordering.jsonl"),
            experiment_ledger=ledger,
            taint_registry=TaintRegistry(
                self.root / "taint-ordering.jsonl"
            ),
        )
        evidence = copy.deepcopy(aggregate)
        evidence["protected_access_record_hash"] = access_record["record_hash"]
        experiment_records = (
            {"record_hash": SHA_A, "sequence": 1},
            {"record_hash": SHA_C, "sequence": 2},
        )

        with mock.patch.object(
            ledger.store,
            "verify",
            return_value=experiment_records,
        ), self.assertRaises(ExperimentControlError):
            gate._verified_protected_access(
                evidence,
                candidate_sha256=SHA_B,
                plan_record={"record_hash": SHA_A, "sequence": 1},
                result_record={"record_hash": SHA_B, "sequence": 2},
            )

    def test_gate_rejects_unknown_result_candidate_mismatch_and_forged_aggregate(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        self._seed_passing(state)
        ledger, result, taint_registry = self._protected_result(
            suffix="strict"
        )
        gate = CandidatePromotionGate(
            state,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        )
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=state,
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
            taint_registry=taint_registry,
        )

        with self.assertRaises(ExperimentControlError):
            gate.evaluate_and_record(
                "unknown-result",
                candidate_id="candidate-unknown",
                candidate_sha256=SHA_B,
                experiment_result_record_hash="f" * 64,
                integrity_checkpoint=checkpoint,
            )
        with self.assertRaises(ExperimentControlError):
            gate.evaluate_and_record(
                "wrong-candidate",
                candidate_id="candidate-wrong",
                candidate_sha256=SHA_A,
                experiment_result_record_hash=result["record_hash"],
                integrity_checkpoint=checkpoint,
            )

        forged_ledger, forged, forged_taint = self._protected_result(
            suffix="forged",
            forged_aggregate_mismatch=True,
        )
        forged_gate = CandidatePromotionGate(
            state,
            experiment_ledger=forged_ledger,
            taint_registry=forged_taint,
        )
        with self.assertRaises(ExperimentControlError):
            forged_gate.evaluate_and_record(
                "forged-aggregate",
                candidate_id="candidate-forged",
                candidate_sha256=SHA_B,
                experiment_result_record_hash=forged["record_hash"],
                integrity_checkpoint=self.checkpoint_for(
                    candidate_state_ledger=state,
                    experiment_ledger=forged_ledger,
                    protected_access_ledger=(
                        forged_ledger.protected_access_store
                    ),
                    taint_registry=forged_taint,
                ),
            )

    def test_gate_rejects_valid_prefix_truncation_of_any_governed_ledger(self):
        state = CandidateStateStore(self.root / "candidates-truncation.jsonl")
        self._seed_passing(state)
        state.store.append(
            {
                "event": "candidate_assessment",
                "assessment_id": "historical-block",
                "candidate_id": "historical-block",
                "candidate_sha256": SHA_C,
                "decision": "BLOCKED",
                "aggregate_evidence": {"score": 1.0},
            }
        )
        ledger, result, taint_registry = self._protected_result(
            suffix="truncation"
        )
        taint_registry.store.append(
            {
                "event": "taint",
                "group_id": "layout-family",
                "reason": "inspected",
                "source": "debug",
            }
        )
        gate = CandidatePromotionGate(
            state,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        )
        stores = {
            "candidate_state_ledger": state.store,
            "experiment_ledger": ledger.store,
            "protected_access_ledger": ledger.protected_access_store,
            "taint_registry": taint_registry.store,
        }
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=state,
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
            taint_registry=taint_registry,
        )
        originals = {
            name: store.path.read_bytes()
            for name, store in stores.items()
        }

        for name, store in stores.items():
            with self.subTest(name=name):
                for restore_name, restore_store in stores.items():
                    restore_store.path.write_bytes(originals[restore_name])
                lines = originals[name].splitlines(keepends=True)
                store.path.write_bytes(lines[0] if len(lines) > 1 else b"")
                state_before = state.store.path.read_bytes()
                with self.assertRaises(IntegrityError):
                    gate.evaluate_and_record(
                        f"truncated-{name}",
                        candidate_id="candidate-truncated",
                        candidate_sha256=SHA_B,
                        experiment_result_record_hash=result["record_hash"],
                        integrity_checkpoint=checkpoint,
                    )
                self.assertEqual(state.store.path.read_bytes(), state_before)

        for name, store in stores.items():
            store.path.write_bytes(originals[name])
        checkpoint.verify()

    def test_program_lock_serializes_gate_and_taint_mutations(self):
        state = CandidateStateStore(self.root / "candidates-race.jsonl")
        self._seed_passing(state)
        ledger, result, taint_registry = self._protected_result(
            suffix="post-append-race"
        )
        gate = CandidatePromotionGate(
            state,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        )
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=state,
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
            taint_registry=taint_registry,
        )
        barrier = threading.Barrier(2)

        def promote():
            barrier.wait()
            return (
                "gate",
                gate.evaluate_and_record(
                    "post-append-race",
                    candidate_id="candidate-race",
                    candidate_sha256=SHA_B,
                    experiment_result_record_hash=result["record_hash"],
                    integrity_checkpoint=checkpoint,
                ),
            )

        def taint():
            barrier.wait()
            return (
                "taint",
                taint_registry.taint(
                    "concurrent-layout",
                    reason="concurrent-inspection",
                    source="debug",
                    integrity_checkpoint=checkpoint,
                ),
            )

        outcomes = []
        failures = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2
        ) as pool:
            futures = (pool.submit(promote), pool.submit(taint))
            for future in futures:
                try:
                    outcomes.append(future.result())
                except IntegrityError as exc:
                    failures.append(exc)

        self.assertEqual(len(outcomes), 1)
        self.assertEqual(len(failures), 1)
        winner, receipt = outcomes[0]
        published = self.publish_successor(
            checkpoint,
            receipt.integrity,
        )
        published.verify()
        latest = state.latest_passing(
            integrity_checkpoint=published
        )
        if winner == "gate":
            self.assertEqual(latest["candidate_sha256"], SHA_B)
            self.assertEqual(len(taint_registry.events()), 0)
        else:
            self.assertEqual(latest["candidate_sha256"], SHA_A)
            self.assertEqual(len(taint_registry.events()), 1)

    def test_latest_passing_requires_published_successor(self):
        state = CandidateStateStore(self.root / "candidates-finalized.jsonl")
        self._seed_passing(state)
        ledger, result, taint_registry = self._protected_result(
            suffix="finalized"
        )
        gate = CandidatePromotionGate(
            state,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        )
        checkpoint = self.checkpoint_for(
            candidate_state_ledger=state,
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
            taint_registry=taint_registry,
        )
        receipt = gate.evaluate_and_record(
            "finalized",
            candidate_id="candidate-finalized",
            candidate_sha256=SHA_B,
            experiment_result_record_hash=result["record_hash"],
            integrity_checkpoint=checkpoint,
        )

        with self.assertRaises(IntegrityError):
            state.latest_passing(integrity_checkpoint=checkpoint)
        published = self.publish_successor(
            checkpoint,
            receipt.integrity,
        )
        self.assertEqual(
            state.latest_passing(
                integrity_checkpoint=published
            )["candidate_sha256"],
            SHA_B,
        )

    def test_gate_retry_is_idempotent_and_binding_mismatch_is_rejected(self):
        state = CandidateStateStore(self.root / "candidates.jsonl")
        self._seed_passing(state)
        ledger, result, taint_registry = self._protected_result(
            suffix="retry"
        )
        gate = CandidatePromotionGate(
            state,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        )
        request = {
            "candidate_id": "candidate-retry",
            "candidate_sha256": SHA_B,
            "experiment_result_record_hash": result["record_hash"],
        }

        checkpoint = self.checkpoint_for(
            candidate_state_ledger=state,
            experiment_ledger=ledger,
            protected_access_ledger=ledger.protected_access_store,
            taint_registry=taint_registry,
        )
        first = gate.evaluate_and_record(
            "gate-retry",
            integrity_checkpoint=checkpoint,
            **request,
        )
        published = self.publish_successor(
            checkpoint,
            first.integrity,
        )
        retry = gate.evaluate_and_record(
            "gate-retry",
            integrity_checkpoint=published,
            **request,
        )
        self.assertEqual(first.assessment, retry.assessment)
        self.assertFalse(retry.mutated)
        with self.assertRaises(ExperimentControlError):
            gate.evaluate_and_record(
                "gate-retry",
                candidate_id="candidate-other",
                candidate_sha256=SHA_B,
                experiment_result_record_hash=result["record_hash"],
                integrity_checkpoint=published,
            )

    def test_idempotent_retry_rejects_injected_passed_record(self):
        state = CandidateStateStore(self.root / "candidates-injected.jsonl")
        self._seed_passing(state)
        ledger, result, taint_registry = self._protected_result(
            suffix="injected-pass"
        )
        state.store.append(
            {
                "event": "candidate_assessment",
                "assessment_id": "injected-pass",
                "candidate_id": "candidate-injected",
                "candidate_sha256": SHA_B,
                "decision": "PASSED",
                "aggregate_evidence": {
                    "experiment_result_record_hash": result["record_hash"],
                    "metrics": {"total_score": 136.0},
                    "promotion_gate_verified": True,
                },
            }
        )
        gate = CandidatePromotionGate(
            state,
            experiment_ledger=ledger,
            taint_registry=taint_registry,
        )

        with self.assertRaises(IntegrityError):
            gate.evaluate_and_record(
                "injected-pass",
                candidate_id="candidate-injected",
                candidate_sha256=SHA_B,
                experiment_result_record_hash=result["record_hash"],
                integrity_checkpoint=self.checkpoint_for(
                    candidate_state_ledger=state,
                    experiment_ledger=ledger,
                    protected_access_ledger=(
                        ledger.protected_access_store
                    ),
                    taint_registry=taint_registry,
                ),
            )


if __name__ == "__main__":
    unittest.main()
