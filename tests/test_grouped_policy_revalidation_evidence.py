from __future__ import annotations

import csv
import copy
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from devtools.experiment_control import (
    CanonicalHashChainStore,
    build_program_integrity_checkpoint,
    canonical_json,
    require_aggregate_only,
)
from devtools.grouped_policy_revalidation_evidence import (
    GovernedExperimentTopology,
    GroupedPolicyEvidenceBuildError,
    _EVIDENCE_BOOTSTRAP_TIMEOUT_SECONDS,
    _FreshCapture,
    _canonical_object,
    _complete_confusion,
    _early_evidence_archive_tree_sha256,
    _folds,
    _load_bound_evaluator,
    _parser,
    _require_evidence_archive_binding,
    _row_change_counts,
    _validate_runtime_contract,
    _truth_rows_from_bytes,
    build_aggregate_evidence,
    validate_aggregate_artifact,
    verify_governed_experiment_topology,
)
from devtools.grouped_split_evidence import FrozenLayoutManifest
from devtools.policy_grouped_capture import (
    _CONTRACT_CHECK_NAMES,
    _MATCHER_COUNT_NAMES,
)
from devtools.policy_revalidation_audit_contract import (
    CONTRACT_AUDIT_COUNTS,
)
from mib_pipeline.decision_recovery import POLICY_AUDIT_COUNT_NAMES
from mib_pipeline.models import FIELD_NAMES, PredictionRow
from scripts import evaluate as official_evaluate


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
TOOL_SHA = "f" * 64


def _write_canonical(path: Path, value: dict[str, object]) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _runtime(
    *,
    evaluator_sha: str,
    input_tree_sha: str,
    manifest_sha: str,
    truth_sha: str,
) -> dict[str, object]:
    return {
        "capture": {
            "arm_repeat_count": 2,
            "execution": "sequential",
            "max_workers": 4,
            "metrics_source": (
                "fresh_process_rusage_self_plus_waited_children_and_"
                "monotonic_wall"
            ),
            "required_byte_determinism": True,
        },
        "container_limits": {
            "image_bytes": 4294967296,
            "max_model_artifact_bytes": 262144000,
            "model_bytes": 1073741824,
            "network": "none",
            "output_bytes": 26214400,
            "peak_memory_bytes": 8589934592,
            "per_record_runtime_seconds": 6,
            "runtime_seconds": 30000,
            "tmp_bytes": 2147483648,
        },
        "environment": {
            "MIB_MAX_WORKERS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        },
        "evaluation": {
            "evaluator_sha256": evaluator_sha,
            "evidence_label": "public_grouped_robustness_not_unseen",
            "expected_record_count": 1000,
            "input_tree_sha256": input_tree_sha,
            "layout_manifest_sha256": manifest_sha,
            "truth_sha256": truth_sha,
        },
        "interface": {
            "entrypoint": "solution.py",
            "input": "directory containing canonical PDF cases",
            "output": "canonical twelve-field JSONL",
            "runner": "run.sh",
        },
        "schema_version": "mib-wo17-runtime-contract/v1",
    }


class TopologyFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        program = root / "evaluation" / "program"
        program.mkdir(parents=True)
        self.ledger_names = {
            "candidate_state_ledger": "candidate_state_ledger.jsonl",
            "experiment_ledger": "experiment_ledger.jsonl",
            "protected_access_ledger": "protected_access_ledger.jsonl",
            "taint_registry": "taint_registry.jsonl",
        }
        for filename in self.ledger_names.values():
            (program / filename).write_bytes(b"")
        baseline = program / "frozen_baseline_manifest.json"
        _write_canonical(baseline, {"schema": "test"})
        source = root / "mib_pipeline"
        source.mkdir()
        (source / "decision_recovery.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        self.promotion = {
            "evaluator_sha256": SHA_A,
            "expected_record_count": 1000,
            "input_tree_sha256": SHA_B,
            "runtime_contract_sha256": SHA_C,
            "split_manifest_sha256": SHA_A,
            "truth_sha256": SHA_B,
        }
        self.stores = {
            name: CanonicalHashChainStore(program / filename)
            for name, filename in self.ledger_names.items()
        }
        baseline_sha = hashlib.sha256(baseline.read_bytes()).hexdigest()
        raw, digest = build_program_integrity_checkpoint(
            stores=self.stores,
            baseline_manifest_sha256=baseline_sha,
            checkpoint_directory=program,
            runtime_leakage_finding_count=0,
            promotion_population=self.promotion,
        )
        (program / f"{digest}.json").write_bytes(raw)
        _write_canonical(
            program / "current_checkpoint.json",
            {
                "checkpoint_path": f"evaluation/program/{digest}.json",
                "checkpoint_sha256": digest,
                "previous_checkpoint_sha256": None,
                "schema": "mib-program-current-checkpoint/v1",
            },
        )
        self.git("add", ".")
        self.git("commit", "-m", "A")
        self.base = self.git("rev-parse", "HEAD")
        self.plan = {
            "changed_files": ["mib_pipeline/decision_recovery.py"],
            "evidence_label": "public_grouped_robustness_not_unseen",
            "evaluator_sha256": SHA_A,
            "expected_record_count": 1000,
            "hypothesis_sha256": SHA_A,
            "input_tree_sha256": SHA_B,
            "parent_commit_sha": self.base,
            "primary_variable_sha256": SHA_C,
            "runtime_contract_sha256": SHA_C,
            "split_manifest_sha256": SHA_A,
            "truth_sha256": SHA_B,
        }
        record = self.stores["experiment_ledger"].append(
            {
                "event": "experiment_plan",
                "experiment_id": "wo17policy",
                "plan": self.plan,
            }
        )
        successor_raw, successor = build_program_integrity_checkpoint(
            stores=self.stores,
            baseline_manifest_sha256=baseline_sha,
            checkpoint_directory=program,
            runtime_leakage_finding_count=0,
            promotion_population=self.promotion,
        )
        (program / f"{successor}.json").write_bytes(successor_raw)
        _write_canonical(
            program / "current_checkpoint.json",
            {
                "checkpoint_path": (
                    f"evaluation/program/{successor}.json"
                ),
                "checkpoint_sha256": successor,
                "previous_checkpoint_sha256": digest,
                "schema": "mib-program-current-checkpoint/v1",
            },
        )
        self.git("add", ".")
        self.git("commit", "-m", "P")
        self.control = self.git("rev-parse", "HEAD")
        (source / "decision_recovery.py").write_text(
            "VALUE = 2\n", encoding="utf-8"
        )
        self.git("add", "mib_pipeline/decision_recovery.py")
        self.git("commit", "-m", "C")
        self.candidate = self.git("rev-parse", "HEAD")
        self.plan_sha = hashlib.sha256(
            (canonical_json(self.plan) + "\n").encode("utf-8")
        ).hexdigest()
        self.record_hash = record["record_hash"]
        self.checkpoint_sha = successor

    def git(self, *arguments: str) -> str:
        return subprocess.run(
            ("/usr/bin/git", *arguments),
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()


class GroupedPolicyEvidenceTests(unittest.TestCase):
    def test_archive_builder_timeout_covers_full_runtime_contract(self):
        self.assertEqual(_EVIDENCE_BOOTSTRAP_TIMEOUT_SECONDS, 31000)
        self.assertGreater(
            _EVIDENCE_BOOTSTRAP_TIMEOUT_SECONDS,
            _runtime(
                evaluator_sha=SHA_A,
                input_tree_sha=SHA_B,
                manifest_sha=SHA_C,
                truth_sha=SHA_A,
            )["container_limits"]["runtime_seconds"],
        )

    def test_hash_bound_json_and_truth_are_parsed_without_reopening(self):
        raw = (canonical_json({"status": "passed"}) + "\n").encode("utf-8")
        with mock.patch(
            "devtools.grouped_policy_revalidation_evidence."
            "_read_regular_bytes",
            return_value=raw,
        ) as reader:
            value, bound_raw = _canonical_object(
                Path("/private/tmp/bound.json"), label="bound"
            )
        self.assertEqual(value, {"status": "passed"})
        self.assertEqual(bound_raw, raw)
        reader.assert_called_once()

        truth_raw = b"case_id,adjudication\nMIB-000001,APPROVED\n"
        with mock.patch(
            "devtools.grouped_policy_revalidation_evidence."
            "_read_regular_bytes",
            side_effect=AssertionError("truth bytes were reopened"),
        ):
            truth = _truth_rows_from_bytes(
                truth_raw, case_ids=("MIB-000001",)
            )
        self.assertEqual(truth["MIB-000001"]["adjudication"], "APPROVED")

    def test_direct_builder_without_verified_archive_fails_closed(self):
        with mock.patch(
            "devtools.grouped_policy_revalidation_evidence."
            "_EVIDENCE_ARCHIVE_CONTRACT",
            None,
        ):
            with self.assertRaisesRegex(
                GroupedPolicyEvidenceBuildError,
                "verified candidate archive",
            ):
                _require_evidence_archive_binding("a" * 40)

    def test_evidence_archive_binding_rejects_transitive_source_mutation(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
            root = Path(name).resolve()
            origin = root / "origin"
            source = root / "source"
            origin.mkdir()
            source.mkdir()

            def git(*arguments: str, text: bool = True):
                return subprocess.run(
                    ("/usr/bin/git", *arguments),
                    cwd=origin,
                    check=True,
                    capture_output=True,
                    text=text,
                ).stdout

            git("init")
            git("config", "user.email", "test@example.invalid")
            git("config", "user.name", "Test")
            dependency = origin / "dependency.py"
            dependency.write_text("VALUE = 1\n", encoding="utf-8")
            git("add", "dependency.py")
            git("commit", "-m", "candidate")
            revision = git("rev-parse", "HEAD").strip()
            archive = git(
                "archive", "--format=tar", revision, text=False
            )
            with tarfile.open(
                fileobj=io.BytesIO(archive), mode="r:"
            ) as bundle:
                bundle.extractall(source, filter="data")
            tree_sha = _early_evidence_archive_tree_sha256(archive)
            contract = {
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "bootstrap_pid": 1,
                "origin_root": str(origin),
                "revision": revision,
                "schema": "mib-wo17-evidence-archive/v1",
                "source_root": str(source),
                "tree_sha256": tree_sha,
            }
            with mock.patch(
                "devtools.grouped_policy_revalidation_evidence."
                "_EVIDENCE_ARCHIVE_CONTRACT",
                contract,
            ), mock.patch(
                "devtools.grouped_policy_revalidation_evidence.REPO_ROOT",
                source,
            ), mock.patch(
                "devtools.grouped_policy_revalidation_evidence."
                "GIT_AUTHORITY_ROOT",
                origin,
            ):
                _require_evidence_archive_binding(revision)
                (source / "dependency.py").write_text(
                    "VALUE = 2\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(
                    GroupedPolicyEvidenceBuildError,
                    "archive binding failed",
                ):
                    _require_evidence_archive_binding(revision)

    def test_bound_evaluator_ignores_a_preloaded_module_mismatch(self):
        path = Path(official_evaluate.__file__).resolve()
        raw = path.read_bytes()
        with mock.patch.object(
            official_evaluate,
            "index_submission",
            side_effect=AssertionError("preloaded evaluator was used"),
        ):
            bound = _load_bound_evaluator(raw, source_path=path)
            indexed, duplicates, blank = bound.index_submission(
                ({"case_id": "MIB-000001"},)
            )
        self.assertEqual(set(indexed), {"MIB-000001"})
        self.assertEqual(duplicates, [])
        self.assertEqual(blank, 0)

    def test_exported_topology_verifier_derives_exact_a_p_c_bindings(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
            fixture = TopologyFixture(Path(name).resolve())
            result = verify_governed_experiment_topology(
                fixture.root,
                plan=fixture.plan,
                experiment_plan_sha256=fixture.plan_sha,
                control_revision_sha=fixture.control,
                candidate_revision_sha=fixture.candidate,
            )
            self.assertEqual(result.base_revision_sha, fixture.base)
            self.assertEqual(
                result.experiment_plan_record_sha256,
                fixture.record_hash,
            )
            self.assertEqual(
                result.prereg_checkpoint_sha256,
                fixture.checkpoint_sha,
            )

    def test_topology_rejects_plan_hash_governance_scope_and_extra_diff(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
            fixture = TopologyFixture(Path(name).resolve())
            with self.assertRaisesRegex(
                GroupedPolicyEvidenceBuildError, "canonical plan bytes"
            ):
                verify_governed_experiment_topology(
                    fixture.root,
                    plan=fixture.plan,
                    experiment_plan_sha256="0" * 64,
                    control_revision_sha=fixture.control,
                    candidate_revision_sha=fixture.candidate,
                )
            malicious = dict(fixture.plan)
            malicious["changed_files"] = [
                "evaluation/program/experiment_ledger.jsonl"
            ]
            malicious_sha = hashlib.sha256(
                (canonical_json(malicious) + "\n").encode("utf-8")
            ).hexdigest()
            with self.assertRaisesRegex(
                GroupedPolicyEvidenceBuildError,
                "governance or control",
            ):
                verify_governed_experiment_topology(
                    fixture.root,
                    plan=malicious,
                    experiment_plan_sha256=malicious_sha,
                    control_revision_sha=fixture.control,
                    candidate_revision_sha=fixture.candidate,
                )

            fixture.git("checkout", fixture.control)
            source = fixture.root / "mib_pipeline"
            (source / "decision_recovery.py").write_text(
                "VALUE = 3\n", encoding="utf-8"
            )
            (source / "extra.py").write_text(
                "EXTRA = True\n", encoding="utf-8"
            )
            fixture.git("add", "mib_pipeline")
            fixture.git("commit", "-m", "bad C")
            bad_candidate = fixture.git("rev-parse", "HEAD")
            with self.assertRaisesRegex(
                GroupedPolicyEvidenceBuildError,
                "exactly match",
            ):
                verify_governed_experiment_topology(
                    fixture.root,
                    plan=fixture.plan,
                    experiment_plan_sha256=fixture.plan_sha,
                    control_revision_sha=fixture.control,
                    candidate_revision_sha=bad_candidate,
                )

    def test_exact_nested_runtime_contract_and_3x5_slicing(self):
        runtime = _runtime(
            evaluator_sha=SHA_A,
            input_tree_sha=SHA_B,
            manifest_sha=SHA_C,
            truth_sha=TOOL_SHA,
        )
        _validate_runtime_contract(
            runtime,
            evaluator_sha256=SHA_A,
            input_tree_sha256=SHA_B,
            manifest_sha256=SHA_C,
            truth_sha256=TOOL_SHA,
        )
        unsafe = json.loads(canonical_json(runtime))
        unsafe["capture"]["max_workers"] = 5
        with self.assertRaisesRegex(
            GroupedPolicyEvidenceBuildError, "exact WO-17"
        ):
            _validate_runtime_contract(
                unsafe,
                evaluator_sha256=SHA_A,
                input_tree_sha256=SHA_B,
                manifest_sha256=SHA_C,
                truth_sha256=TOOL_SHA,
            )

        case_ids = tuple(
            f"MIB-{index:06d}" for index in range(1, 1001)
        )
        groups = {
            f"layout-{group:02d}": tuple(
                case_id
                for index, case_id in enumerate(case_ids)
                if index % 25 == group
            )
            for group in range(25)
        }
        manifest = FrozenLayoutManifest(
            groups=groups,
            split_seed="wo17-test",
            sha256=SHA_A,
            frozen_before_scoring=False,
        )
        truth = {
            case_id: {
                "case_id": case_id,
                "adjudication": "APPROVED",
                "unrecoverable_fields": "",
            }
            for case_id in case_ids
        }
        control = tuple(
            {
                "case_id": case_id,
                "adjudication": "NEEDS_REVIEW",
                "confidence": 0.5,
            }
            for case_id in case_ids
        )
        candidate = tuple(
            {
                "case_id": case_id,
                "adjudication": "APPROVED",
                "confidence": 1.0,
            }
            for case_id in case_ids
        )
        folds, exclusive, paired, deterministic = _folds(
            official_evaluate, manifest, truth, control, candidate
        )
        self.assertEqual(len(folds), 15)
        self.assertTrue(exclusive and paired and deterministic)
        for repeat in range(3):
            self.assertEqual(
                sum(
                    fold.record_count
                    for fold in folds
                    if fold.repeat == repeat
                ),
                1000,
            )

    def test_full_builder_emits_passing_identity_free_aggregate(self):
        with tempfile.TemporaryDirectory(dir="/private/tmp") as name:
            root = Path(name).resolve()
            input_dir = root / "inputs"
            input_dir.mkdir()
            case_ids = tuple(
                f"MIB-{index:06d}" for index in range(1, 1001)
            )
            groups = {
                f"layout-{group:02d}": tuple(
                    case_id
                    for index, case_id in enumerate(case_ids)
                    if index % 25 == group
                )
                for group in range(25)
            }
            manifest_path = root / "manifest.json"
            _write_canonical(manifest_path, {"fixture": "manifest"})
            manifest_sha = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            frozen = FrozenLayoutManifest(
                groups=groups,
                split_seed="wo17-builder-test",
                sha256=manifest_sha,
                frozen_before_scoring=False,
            )
            truth_path = root / "truth.csv"
            truth_fields = (
                *FIELD_NAMES[:-1],
                "confidence",
                "unrecoverable_fields",
            )
            truth_rows = []
            with truth_path.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=truth_fields
                )
                writer.writeheader()
                for index, case_id in enumerate(case_ids, start=1):
                    row = {
                        "case_id": case_id,
                        "applicant_name": f"Applicant {index}",
                        "species_code": "HUM",
                        "home_world": "Earth",
                        "visa_class": "XW-1",
                        "sponsor_id": f"SPN-{index % 10000:04d}",
                        "arrival_date": "2026-07-27",
                        "declared_purpose": "Visit",
                        "risk_flags": "none",
                        "fee_status": "paid",
                        "adjudication": "APPROVED",
                        "confidence": "1.0",
                        "unrecoverable_fields": "",
                    }
                    writer.writerow(row)
                    truth_rows.append(row)
            truth_sha = hashlib.sha256(
                truth_path.read_bytes()
            ).hexdigest()
            control_rows = []
            candidate_rows = []
            for row in truth_rows:
                common = {
                    name: row[name]
                    for name in FIELD_NAMES
                    if name not in {"adjudication", "confidence"}
                }
                control_rows.append(
                    PredictionRow.from_mapping(
                        {
                            **common,
                            "adjudication": "NEEDS_REVIEW",
                            "confidence": 0.5,
                        }
                    ).to_dict()
                )
                candidate_rows.append(
                    PredictionRow.from_mapping(
                        {
                            **common,
                            "adjudication": "APPROVED",
                            "confidence": 1.0,
                        }
                    ).to_dict()
                )

            def predictions(stem: str, rows: list[dict[str, object]]):
                raw = "".join(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                    for row in rows
                ).encode("utf-8")
                paths = (root / f"{stem}-1.jsonl", root / f"{stem}-2.jsonl")
                for path in paths:
                    path.write_bytes(raw)
                return paths, hashlib.sha256(raw).hexdigest()

            control_predictions, control_prediction_sha = predictions(
                "control", control_rows
            )
            candidate_predictions, candidate_prediction_sha = predictions(
                "candidate", candidate_rows
            )
            input_tree_sha = SHA_B
            evaluator_path = Path(official_evaluate.__file__).resolve()
            evaluator_sha = hashlib.sha256(
                evaluator_path.read_bytes()
            ).hexdigest()
            hypothesis = root / "hypothesis.txt"
            hypothesis.write_text("guarded matcher only\n", encoding="utf-8")
            primary = root / "primary.txt"
            primary.write_text("one production variable\n", encoding="utf-8")
            hypothesis_sha = hashlib.sha256(
                hypothesis.read_bytes()
            ).hexdigest()
            primary_sha = hashlib.sha256(
                primary.read_bytes()
            ).hexdigest()
            runtime_path = root / "runtime.json"
            runtime_value = _runtime(
                evaluator_sha=evaluator_sha,
                input_tree_sha=input_tree_sha,
                manifest_sha=manifest_sha,
                truth_sha=truth_sha,
            )
            _write_canonical(runtime_path, runtime_value)
            runtime_sha = hashlib.sha256(
                runtime_path.read_bytes()
            ).hexdigest()
            base_revision = "1" * 40
            control_revision = "2" * 40
            candidate_revision = "3" * 40
            plan = {
                "changed_files": ["mib_pipeline/decision_recovery.py"],
                "evidence_label": "public_grouped_robustness_not_unseen",
                "evaluator_sha256": evaluator_sha,
                "expected_record_count": 1000,
                "hypothesis_sha256": hypothesis_sha,
                "input_tree_sha256": input_tree_sha,
                "parent_commit_sha": base_revision,
                "primary_variable_sha256": primary_sha,
                "runtime_contract_sha256": runtime_sha,
                "split_manifest_sha256": manifest_sha,
                "truth_sha256": truth_sha,
            }
            plan_path = root / "plan.json"
            _write_canonical(plan_path, plan)
            plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()

            nonvacuous_contract = {
                "legacy_synthetic_before_late_recovery_count",
                "candidate_late_recovery_before_revalidation_count",
                "candidate_revalidation_after_late_recovery_count",
                "contradicted_synthetic_reason_before_count",
                "contradicted_synthetic_reason_removed_count",
                "independent_denial_reason_retained_count",
                "review_confidence_restored_count",
                "normal_policy_rerun_count",
                "signed_late_authority_recovery_count",
                "late_adjudication_evidence_preserved_count",
                "late_biohazard_evidence_preserved_count",
                "placeholder_guard_probe_count",
                "sentinel_guard_probe_count",
                "serialization_default_guard_probe_count",
                "stale_threshold_guard_probe_count",
                "forced_approval_guard_probe_count",
                "direct_approval_head_guard_probe_count",
            }

            def audit(
                arm: str,
                prediction_sha: str,
                graph_sha: str,
                *,
                candidate_arm: bool,
            ):
                counts = {
                    f"policy_{name}": 0
                    for name in POLICY_AUDIT_COUNT_NAMES
                }
                counts[
                    "policy_accepted_final_policy_result_count"
                ] = 1000
                counts.update(
                    {
                        f"matcher_{name}": 0
                        for name in _MATCHER_COUNT_NAMES
                    }
                )
                if candidate_arm:
                    counts.update(
                        {
                            "policy_forced_approval_count": 1,
                            "matcher_eligible_guarded_initial_count": 1,
                            "matcher_guarded_initial_approval_count": 1,
                        }
                    )
                contract = {
                    name: (1 if name in nonvacuous_contract else 0)
                    for name in CONTRACT_AUDIT_COUNTS
                }
                counts.update(
                    {
                        f"contract_{name}": value
                        for name, value in contract.items()
                    }
                )
                payload = {
                    "evaluation_mode": (
                        "public_grouped_robustness_not_unseen"
                    ),
                    "evidence_label": "aggregate_only",
                    "status": arm,
                    "source_revision_sha": (
                        candidate_revision
                        if candidate_arm
                        else control_revision
                    ),
                    "layout_manifest_sha256": manifest_sha,
                    "input_tree_sha256": input_tree_sha,
                    "producer_graph_sha256": graph_sha,
                    "predictions_sha256": prediction_sha,
                    "counts": counts,
                    "checks": {
                        name: True for name in _CONTRACT_CHECK_NAMES
                    },
                }
                raw = (canonical_json(payload) + "\n").encode("utf-8")
                paths = (
                    root / f"{arm}-audit-1.json",
                    root / f"{arm}-audit-2.json",
                )
                for path in paths:
                    path.write_bytes(raw)
                return paths, hashlib.sha256(raw).hexdigest()

            control_audits, control_audit_sha = audit(
                "baseline",
                control_prediction_sha,
                "4" * 64,
                candidate_arm=False,
            )
            candidate_audits, candidate_audit_sha = audit(
                "candidate",
                candidate_prediction_sha,
                "5" * 64,
                candidate_arm=True,
            )

            def observation(
                arm: str,
                revision: str,
                prediction_sha: str,
                audit_sha: str,
                graph_sha: str,
            ) -> Path:
                payload = {
                    "evaluation_mode": (
                        "public_grouped_robustness_not_unseen"
                    ),
                    "evidence_label": "aggregate_only",
                    "status": arm,
                    "source_revision_sha": revision,
                    "layout_manifest_sha256": manifest_sha,
                    "input_tree_sha256": input_tree_sha,
                    "max_worker_count": 4,
                    "producer_graph_sha256": graph_sha,
                    "sandbox_backend_sha256": hashlib.sha256(
                        b"macos_sandbox_exec_v1"
                    ).hexdigest(),
                    "sandbox_policy_sha256": "a" * 64,
                    "source_archive_sha256": "b" * 64,
                    "source_tree_sha256": "c" * 64,
                    "capture_tool_sha256": TOOL_SHA,
                    "first_predictions_sha256": prediction_sha,
                    "second_predictions_sha256": prediction_sha,
                    "first_audit_sha256": audit_sha,
                    "second_audit_sha256": audit_sha,
                    "record_count": 1000,
                    "repeat_count": 2,
                    "deterministic": True,
                    "checks": {
                        "audit_deterministic": True,
                        "byte_deterministic": True,
                        "input_recomputed": True,
                        "producer_graph_stable": True,
                        "source_clean": True,
                        "label_access_absent": True,
                        "archive_source_bound": True,
                        "network_access_denied": True,
                        "runtime_environment_exact": True,
                        "sandbox_enforced": True,
                        "sensitive_access_denied": True,
                    },
                    "counts": {
                        "first_answered_count": 1000,
                        "first_attempted_count": 1000,
                        "first_omitted_count": 0,
                        "second_answered_count": 1000,
                        "second_attempted_count": 1000,
                        "second_omitted_count": 0,
                        "label_access_count": 0,
                    },
                    "metrics": {
                        "first_output_bytes": 1000,
                        "first_peak_rss_bytes": 1000,
                        "first_process_cpu_seconds": 1.0,
                        "first_runtime_seconds": 1.0,
                        "second_output_bytes": 1000,
                        "second_peak_rss_bytes": 1000,
                        "second_process_cpu_seconds": 1.0,
                        "second_runtime_seconds": 1.0,
                    },
                }
                path = root / f"{arm}-observation.json"
                _write_canonical(path, payload)
                return path

            control_observation = observation(
                "baseline",
                control_revision,
                control_prediction_sha,
                control_audit_sha,
                "4" * 64,
            )
            candidate_observation = observation(
                "candidate",
                candidate_revision,
                candidate_prediction_sha,
                candidate_audit_sha,
                "5" * 64,
            )
            topology = GovernedExperimentTopology(
                base_revision_sha=base_revision,
                control_revision_sha=control_revision,
                candidate_revision_sha=candidate_revision,
                experiment_plan_sha256=plan_sha,
                experiment_plan_record_sha256="6" * 64,
                prereg_checkpoint_sha256="7" * 64,
                prereg_governance_diff_sha256="8" * 64,
                candidate_source_diff_sha256="9" * 64,
                planned_scope_manifest_sha256="0" * 64,
            )
            observation_by_arm = {
                "baseline": control_observation,
                "candidate": candidate_observation,
            }
            prediction_by_arm = {
                "baseline": control_predictions,
                "candidate": candidate_predictions,
            }
            audit_by_arm = {
                "baseline": control_audits,
                "candidate": candidate_audits,
            }

            def fresh_capture(*, arm: str, **_kwargs: object) -> _FreshCapture:
                observation_path = observation_by_arm[arm]
                observation_raw = observation_path.read_bytes()
                return _FreshCapture(
                    prediction_raw=tuple(
                        path.read_bytes() for path in prediction_by_arm[arm]
                    ),
                    audit_raw=tuple(
                        path.read_bytes() for path in audit_by_arm[arm]
                    ),
                    observation=json.loads(observation_raw.decode("utf-8")),
                    observation_raw=observation_raw,
                )

            with mock.patch(
                "devtools.grouped_policy_revalidation_evidence."
                "_require_evidence_archive_binding"
            ), mock.patch(
                "devtools.grouped_policy_revalidation_evidence."
                "_require_exact_candidate_checkout"
            ), mock.patch(
                "devtools.grouped_policy_revalidation_evidence."
                "_run_fresh_truth_blind_capture",
                side_effect=fresh_capture,
            ), mock.patch(
                "devtools.grouped_policy_revalidation_evidence."
                "_verify_capture_archive_binding"
            ), mock.patch(
                "devtools.grouped_policy_revalidation_evidence."
                "_run_regression_suites",
                return_value={
                    "adversarial_failure": 0,
                    "focused_failure": 0,
                    "full_failure": 0,
                },
            ), mock.patch(
                "devtools.grouped_policy_revalidation_evidence."
                "_git_bound_tool_sha256",
                return_value=TOOL_SHA,
            ):
                aggregate = build_aggregate_evidence(
                    input_dir=input_dir,
                    layout_manifest_path=manifest_path,
                    truth_path=truth_path,
                    evaluator_path=evaluator_path,
                    experiment_plan_path=plan_path,
                    expected_experiment_plan_sha256=plan_sha,
                    runtime_contract_path=runtime_path,
                    expected_runtime_contract_sha256=runtime_sha,
                    hypothesis_path=hypothesis,
                    primary_variable_path=primary,
                    control_revision_sha=control_revision,
                    candidate_revision_sha=candidate_revision,
                    control_prediction_paths=control_predictions,
                    candidate_prediction_paths=candidate_predictions,
                    control_audit_paths=control_audits,
                    candidate_audit_paths=candidate_audits,
                    control_observation_path=control_observation,
                    candidate_observation_path=candidate_observation,
                    topology_verifier=lambda *_args, **_kwargs: topology,
                    population_verifier=(
                        lambda *_args, **_kwargs: frozen
                    ),
                )
            self.assertEqual(aggregate["status"], "passed")
            self.assertEqual(len(aggregate["fold_deltas"]), 15)
            self.assertEqual(
                aggregate["counts"]["legacy_forced_approval_count"], 1
            )
            self.assertTrue(
                aggregate["gate_results"][
                    "guarded_initial_approval_nonvacuous"
                ]
            )
            require_aggregate_only(aggregate)
            serialized = canonical_json(aggregate)
            self.assertNotIn("MIB-", serialized)
            self.assertNotIn("Applicant", serialized)

            for label, mutation in (
                (
                    "gate",
                    lambda value: value["gate_results"].__setitem__(
                        "full_score_positive", False
                    ),
                ),
                (
                    "repeat",
                    lambda value: value["metrics"].__setitem__(
                        "repeat_1_weighted_score_delta", 999.0
                    ),
                ),
                (
                    "safety",
                    lambda value: value["counts"].__setitem__(
                        "new_false_approval_count", 1
                    ),
                ),
            ):
                with self.subTest(mutation=label):
                    forged = copy.deepcopy(aggregate)
                    mutation(forged)
                    with self.assertRaises(
                        GroupedPolicyEvidenceBuildError
                    ):
                        validate_aggregate_artifact(forged)

    def test_regression_counts_cannot_be_supplied_by_cli(self):
        destinations = {action.dest for action in _parser()._actions}
        self.assertNotIn("adversarial_failure_count", destinations)
        self.assertNotIn("focused_failure_count", destinations)
        self.assertNotIn("full_failure_count", destinations)


if __name__ == "__main__":
    unittest.main()
