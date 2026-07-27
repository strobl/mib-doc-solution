from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from devtools.experiment_control import (
    CanonicalHashChainStore,
    CheckpointAuthorityResolver,
    ExperimentLedger,
    PublishedCheckpointReference,
    build_program_integrity_checkpoint,
    canonical_json,
)
from devtools.governed_result_cli import (
    GovernedResultCLIError,
    _CONFUSION_KEYS,
    _COUNT_KEYS,
    _GROUPED_GATE_KEYS,
    _PRIMARY_ROOT_KEYS,
    _SCORE_COMPONENT_KEYS,
    _authenticated_artifact_entries,
    _authenticated_grouped_provenance,
    _authenticated_runtime,
    _authenticated_workflow_run,
    _atomic_replace_exact,
    _canonical_bytes,
    _derive_gate_results,
    _ledger_evidence,
    _planned_scope_manifest_sha256,
    _runtime_defaults,
    _safe_zip_json_entry,
    _sha256_bytes,
    _trusted_runtime,
    _validate_decision,
    _validate_primary_committed_bindings,
    _validate_primary_evidence,
    _validated_published_result_payload,
    _verify_adopted_result_provenance,
    record_result,
    verify_candidate_authority,
    verify_result_authority,
)
from devtools.grouped_policy_revalidation_evidence import (
    GovernedExperimentTopology,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
COMMIT_A = "1" * 40
COMMIT_P = "2" * 40
COMMIT_C = "3" * 40
EXPERIMENT_ID = "wo17-guarded-review-approval-v1"


def write_canonical(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def plan() -> dict:
    return {
        "changed_files": [
            "mib_pipeline/decision_recovery.py",
            "tests/test_decision_recovery.py",
        ],
        "evidence_label": "public_grouped_robustness_not_unseen",
        "evaluator_sha256": SHA_A,
        "expected_record_count": 1000,
        "hypothesis_sha256": SHA_B,
        "input_tree_sha256": SHA_C,
        "parent_commit_sha": COMMIT_A,
        "primary_variable_sha256": SHA_D,
        "runtime_contract_sha256": SHA_E,
        "split_manifest_sha256": SHA_A,
        "truth_sha256": SHA_B,
    }


def topology() -> GovernedExperimentTopology:
    return GovernedExperimentTopology(
        base_revision_sha=COMMIT_A,
        control_revision_sha=COMMIT_P,
        candidate_revision_sha=COMMIT_C,
        experiment_plan_sha256=SHA_C,
        experiment_plan_record_sha256=SHA_D,
        prereg_checkpoint_sha256=SHA_E,
        prereg_governance_diff_sha256=SHA_A,
        candidate_source_diff_sha256=SHA_B,
        planned_scope_manifest_sha256=_planned_scope_manifest_sha256(
            plan()
        ),
    )


def primary_evidence(*, passed: bool = True) -> dict:
    current_plan = plan()
    current_topology = topology()
    fold_delta = 1.0 if passed else -1.0
    control_fold_score = 100.0
    candidate_fold_score = control_fold_score + fold_delta
    components = {
        "control_extraction_score": 40.0,
        "control_classification_score": 60.0,
        "control_calibration_score": 15.0,
        "control_missing_penalty": 0.0,
        "control_total_score": 115.0,
        "candidate_extraction_score": 40.0,
        "candidate_classification_score": 61.0 if passed else 59.0,
        "candidate_calibration_score": 15.0,
        "candidate_missing_penalty": 0.0,
        "candidate_total_score": 116.0 if passed else 114.0,
    }
    counts = {name: 0 for name in _COUNT_KEYS}
    counts.update(
        {
            "record_count": 1000,
            "layout_group_count": 24,
            "decision_or_confidence_change_count": 4,
            "eligible_guarded_initial_count": 4,
            "guarded_initial_approval_count": 4,
            "legacy_forced_approval_count": 4,
        }
    )
    checks = {
        "authoritative_review_veto": True,
        "legacy_contract_nonvacuous": True,
        "legacy_contract_order_accounted": True,
        "legacy_contract_unsafe_counters_zero": True,
        "runtime_limits_satisfied": True,
    }
    evidence = {
        "base_revision_sha": COMMIT_A,
        "candidate_capture_set_sha256": SHA_A,
        "candidate_diff_manifest_sha256": SHA_B,
        "candidate_producer_graph_sha256": SHA_C,
        "candidate_revision_sha": COMMIT_C,
        "candidate_score": components["candidate_total_score"],
        "candidate_source_diff_sha256": SHA_B,
        "candidate_source_revision_sha": COMMIT_C,
        "capture_tool_sha256": SHA_D,
        "catastrophic_false_approvals": 0,
        "checks": checks,
        "confusion_counts": {
            name: 0 for name in _CONFUSION_KEYS
        },
        "control_capture_set_sha256": SHA_B,
        "control_producer_graph_sha256": SHA_D,
        "control_revision_sha": COMMIT_P,
        "control_score": components["control_total_score"],
        "control_source_revision_sha": COMMIT_P,
        "counts": counts,
        "deterministic": True,
        "duplicate_records": 0,
        "evaluated_fold_count": 15,
        "evaluation_mode": "public_grouped_robustness_not_unseen",
        "evaluator_sha256": current_plan["evaluator_sha256"],
        "evidence_label": "aggregate_only",
        "evidence_tool_sha256": SHA_E,
        "experiment_plan_record_sha256": SHA_D,
        "experiment_plan_sha256": SHA_C,
        "extra_records": 0,
        "false_approvals": 0,
        "fold_consistent": passed,
        "fold_count": 5,
        "fold_deltas": [fold_delta] * 15,
        "fold_metrics": {
            f"repeat_{repeat}_fold_{fold}": {
                "candidate_score": candidate_fold_score,
                "control_score": control_fold_score,
                "layout_group_count": 1,
                "record_count": 200,
                "score_delta": fold_delta,
            }
            for repeat in range(1, 4)
            for fold in range(1, 6)
        },
        "fold_weights": [200] * 15,
        "gate_results": {name: True for name in _GROUPED_GATE_KEYS},
        "gate_tool_sha256": SHA_A,
        "hard_gate_failure_count": 0,
        "hypothesis_sha256": current_plan["hypothesis_sha256"],
        "input_tree_sha256": current_plan["input_tree_sha256"],
        "invalid_records": 0,
        "layout_group_count": 24,
        "metrics": {
            "control_total_score": components["control_total_score"],
            "candidate_total_score": components["candidate_total_score"],
            "extraction_score": components[
                "candidate_extraction_score"
            ],
            "classification_score": components[
                "candidate_classification_score"
            ],
            "calibration_score": components[
                "candidate_calibration_score"
            ],
            "control_missing_penalty": 0.0,
            "candidate_missing_penalty": 0.0,
            "extraction_score_delta": 0.0,
            "classification_score_delta": 1.0 if passed else -1.0,
            "calibration_score_delta": 0.0,
            **{
                f"repeat_{repeat}_weighted_score_delta": fold_delta
                for repeat in range(1, 4)
            },
            **{
                f"repeat_{repeat}_positive_fold_count": (
                    5 if passed else 0
                )
                for repeat in range(1, 4)
            },
            **{
                f"repeat_{repeat}_leave_best_fold_out_delta": (
                    fold_delta
                )
                for repeat in range(1, 4)
            },
        },
        "missing_records": 0,
        "planned_scope_manifest_sha256": (
            current_topology.planned_scope_manifest_sha256
        ),
        "prereg_checkpoint_sha256": SHA_E,
        "prereg_governance_diff_sha256": SHA_A,
        "primary_variable_sha256": current_plan[
            "primary_variable_sha256"
        ],
        "record_count": 1000,
        "repeat_count": 3,
        "repeat_scores": [fold_delta] * 3,
        "runtime_contract_sha256": current_plan[
            "runtime_contract_sha256"
        ],
        "score_components": components,
        "score_delta": 1.0 if passed else -1.0,
        "split_manifest_sha256": current_plan[
            "split_manifest_sha256"
        ],
        "status": "passed" if passed else "blocked",
        "truth_sha256": current_plan["truth_sha256"],
    }
    gates = _derive_gate_results(
        evidence,
        deltas=evidence["fold_deltas"],
        weights=evidence["fold_weights"],
        repeat_scores=evidence["repeat_scores"],
    )
    evidence["gate_results"] = gates
    evidence["hard_gate_failure_count"] = sum(
        not value for value in gates.values()
    )
    evidence["status"] = (
        "passed"
        if evidence["hard_gate_failure_count"] == 0
        else "blocked"
    )
    evidence["fold_consistent"] = (
        gates["no_negative_folds"]
        and gates["leave_best_fold_out_positive"]
    )
    assert set(evidence) == _PRIMARY_ROOT_KEYS
    assert set(components) == _SCORE_COMPONENT_KEYS
    return evidence


def runtime_result() -> dict:
    return {
        "actions_authenticated": False,
        "candidate_image_bytes": 1024,
        "candidate_max_model_artifact_bytes": 0,
        "candidate_model_bytes": 0,
        "deterministic": True,
        "output_bytes": 2048,
        "peak_container_memory_bytes": 4096,
        "peak_rss_bytes": 4096,
        "prediction_output_sha256": SHA_B,
        "process_cpu_seconds": 40.0,
        "runtime_evidence_sha256": SHA_A,
        "runtime_limits_verified": True,
        "runtime_seconds": 10.0,
        "tmp_bytes": 1024,
    }


def authenticated_runtime_result() -> dict:
    value = runtime_result()
    value.update(
        {
            "actions_authenticated": True,
            "capture_a_api_archive_sha256": SHA_A,
            "capture_a_archive_sha256": SHA_A,
            "capture_a_entry_sha256": SHA_B,
            "capture_a_identifier": 101,
            "capture_b_api_archive_sha256": SHA_C,
            "capture_b_archive_sha256": SHA_C,
            "capture_b_entry_sha256": SHA_D,
            "capture_b_identifier": 102,
            "head_revision_sha": COMMIT_C,
            "provenance_sha256": SHA_E,
            "repository_identifier": 201,
            "repository_sha256": SHA_A,
            "run_attempt": 1,
            "run_identifier": 301,
            "workflow_identifier": 401,
            "workflow_sha256": SHA_B,
        }
    )
    return value


def grouped_provenance() -> dict:
    return {
        "aggregate_api_archive_sha256": SHA_A,
        "aggregate_archive_sha256": SHA_A,
        "aggregate_entry_sha256": SHA_B,
        "artifact_identifier": 501,
        "head_revision_sha": COMMIT_C,
        "provenance_sha256": SHA_C,
        "repository_identifier": 201,
        "repository_sha256": SHA_D,
        "run_attempt": 1,
        "run_identifier": 601,
        "workflow_identifier": 701,
        "workflow_sha256": SHA_E,
    }


def zip_json_entry(name: str, raw: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(name, raw)
    return output.getvalue()


class _FakeAuthority:
    def __init__(self, current, pointer, revision, *, drift=False):
        self.current = current
        self.pointer = pointer
        self.revision = revision
        self.drift = drift
        self.calls = 0

    def resolve(self, *, stores):
        del stores
        self.calls += 1
        revision = (
            "f" * 40 if self.drift and self.calls > 1 else self.revision
        )
        return self.current, dict(self.pointer), revision


class GovernedResultCLITests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repository = self.root / "repo"
        self.program = self.repository / "evaluation/program"
        self.program.mkdir(parents=True)
        self.external = self.root / "external"
        self.external.mkdir()
        self.evidence_path = self.external / "evidence.json"
        self.stores = {
            name: CanonicalHashChainStore(
                self.program / f"{name}.jsonl"
            )
            for name in (
                "candidate_state_ledger",
                "experiment_ledger",
                "protected_access_ledger",
                "taint_registry",
            )
        }
        for store in self.stores.values():
            store.path.write_bytes(b"")
        plan_record = self.stores["experiment_ledger"].append(
            {
                "event": "experiment_plan",
                "experiment_id": EXPERIMENT_ID,
                "plan": plan(),
            }
        )
        self.plan_record_hash = plan_record["record_hash"]
        checkpoint_raw, checkpoint_sha = build_program_integrity_checkpoint(
            stores=self.stores,
            baseline_manifest_sha256=SHA_A,
            checkpoint_directory=self.program,
            runtime_leakage_finding_count=0,
            promotion_population={
                name: plan()[name]
                for name in (
                    "evaluator_sha256",
                    "expected_record_count",
                    "input_tree_sha256",
                    "runtime_contract_sha256",
                    "split_manifest_sha256",
                    "truth_sha256",
                )
            },
        )
        self.checkpoint_path = self.program / f"{checkpoint_sha}.json"
        self.checkpoint_path.write_bytes(checkpoint_raw)
        reference = PublishedCheckpointReference(
            path=self.checkpoint_path,
            sha256=checkpoint_sha,
        )
        self.current = CheckpointAuthorityResolver(
            lambda: reference,
            trusted_checkpoint_root=self.program,
        ).resolve(stores=self.stores)
        self.pointer_path = self.program / "current_checkpoint.json"
        self.pointer = {
            "checkpoint_path": (
                f"evaluation/program/{checkpoint_sha}.json"
            ),
            "checkpoint_sha256": checkpoint_sha,
            "previous_checkpoint_sha256": SHA_B,
            "schema": "mib-program-current-checkpoint/v1",
        }
        write_canonical(self.pointer_path, self.pointer)
        self.primary = primary_evidence(passed=False)
        self.primary["experiment_plan_record_sha256"] = (
            self.plan_record_hash
        )
        write_canonical(self.evidence_path, self.primary)

    def tearDown(self):
        self.temporary.cleanup()

    def _patches(
        self,
        *,
        authority=None,
        remote_head=COMMIT_C,
        runtime=None,
    ):
        authority = authority or _FakeAuthority(
            self.current, self.pointer, COMMIT_C
        )
        patched_topology = topology()
        patched_topology = GovernedExperimentTopology(
            **{
                **patched_topology.__dict__,
                "experiment_plan_record_sha256": self.plan_record_hash,
                "prereg_checkpoint_sha256": (
                    self.current.expected_sha256
                ),
            }
        )
        return (
            mock.patch(
                "devtools.governed_result_cli._program_paths",
                return_value=(
                    self.program,
                    self.pointer_path,
                    self.program,
                    self.stores,
                ),
            ),
            mock.patch(
                "devtools.governed_result_cli.GitHubCheckpointAuthority",
                return_value=authority,
            ),
            mock.patch(
                "devtools.governed_result_cli._validate_all_bindings",
                return_value=(
                    plan(),
                    self.plan_record_hash,
                    patched_topology,
                    self.primary,
                ),
            ),
            mock.patch(
                "devtools.governed_result_cli._require_external_canonical_object",
                return_value=self.primary,
            ),
            mock.patch(
                "devtools.governed_result_cli._runtime_leakage_finding_count",
                return_value=0,
            ),
            mock.patch(
                "devtools.governed_result_cli._require_exact_worktree_changes"
            ),
            mock.patch(
                "devtools.governed_result_cli._remote_head",
                return_value=remote_head,
            ),
            mock.patch(
                "devtools.governed_result_cli._trusted_runtime",
                return_value=runtime or runtime_result(),
            ),
        )

    def _record(
        self,
        *,
        decision="reject",
        runtime_paths=(),
        actions_runtime=None,
        actions_grouped=None,
    ):
        patches = self._patches()
        with patches[0], patches[1], patches[2], patches[3], patches[
            4
        ], patches[5], patches[6], patches[7], mock.patch(
            "devtools.governed_result_cli._authenticated_runtime",
            return_value=(
                actions_runtime or authenticated_runtime_result()
            ),
        ), mock.patch(
            "devtools.governed_result_cli._authenticated_grouped_provenance",
            return_value=actions_grouped or grouped_provenance(),
        ):
            return record_result(
                repository_root=self.repository,
                github_repository="example/repository",
                branch="candidate",
                experiment_id=EXPERIMENT_ID,
                evidence_path=self.evidence_path,
                decision=decision,
                rationale="strict_gate_failure",
                runtime_capture_paths=runtime_paths,
            )

    def test_primary_contract_accepts_exact_pass_and_rejects_forged_binding(self):
        evidence = primary_evidence()
        validated = _validate_primary_evidence(
            evidence,
            plan=plan(),
            plan_record_hash=SHA_D,
            topology=topology(),
            current_checkpoint_sha256=SHA_E,
            candidate_revision=COMMIT_C,
        )
        self.assertEqual(validated["status"], "passed")
        forged = copy.deepcopy(evidence)
        forged["candidate_source_diff_sha256"] = SHA_E
        with self.assertRaisesRegex(
            GovernedResultCLIError, "candidate_source_diff"
        ):
            _validate_primary_evidence(
                forged,
                plan=plan(),
                plan_record_hash=SHA_D,
                topology=topology(),
                current_checkpoint_sha256=SHA_E,
                candidate_revision=COMMIT_C,
            )

    def test_primary_contract_rejects_forged_gate_and_topology(self):
        evidence = primary_evidence()
        evidence["gate_results"]["no_negative_folds"] = False
        with self.assertRaisesRegex(
            GovernedResultCLIError, "gate_results.no_negative_folds"
        ):
            _validate_primary_evidence(
                evidence,
                plan=plan(),
                plan_record_hash=SHA_D,
                topology=topology(),
                current_checkpoint_sha256=SHA_E,
                candidate_revision=COMMIT_C,
            )
        bad_topology = GovernedExperimentTopology(
            **{
                **topology().__dict__,
                "candidate_revision_sha": "9" * 40,
            }
        )
        with self.assertRaisesRegex(
            GovernedResultCLIError, "topology"
        ):
            _validate_primary_evidence(
                primary_evidence(),
                plan=plan(),
                plan_record_hash=SHA_D,
                topology=bad_topology,
                current_checkpoint_sha256=SHA_E,
                candidate_revision=COMMIT_C,
            )

    def test_authenticated_workflow_run_binds_exact_candidate_and_repository(
        self,
    ):
        workflow_path = ".github/workflows/wo20-runtime.yml"
        run = {
            "conclusion": "success",
            "event": "workflow_dispatch",
            "head_branch": "candidate",
            "head_repository": {
                "full_name": "example/repository",
                "id": 91,
            },
            "head_sha": COMMIT_C,
            "id": 101,
            "path": workflow_path,
            "repository": {
                "full_name": "example/repository",
                "id": 91,
            },
            "run_attempt": 1,
            "status": "completed",
            "workflow_id": 81,
        }
        with mock.patch(
            "devtools.governed_result_cli._gh_json",
            side_effect=(
                {"total_count": 1, "workflow_runs": [run]},
                {
                    "id": 81,
                    "path": workflow_path,
                    "state": "active",
                },
                run,
            ),
        ):
            authenticated = _authenticated_workflow_run(
                self.repository,
                github_repository="example/repository",
                branch="candidate",
                candidate_revision=COMMIT_C,
                workflow_path=workflow_path,
            )
        self.assertEqual(authenticated["run_identifier"], 101)
        self.assertEqual(
            authenticated["head_revision_sha"],
            COMMIT_C,
        )
        self.assertEqual(authenticated["repository_identifier"], 91)

    def test_authenticated_artifact_verifies_api_digest_and_safe_zip(self):
        raw = _canonical_bytes({"status": "passed"})
        archive = zip_json_entry("evidence-a.json", raw)
        archive_sha = _sha256_bytes(archive)
        run = {
            "head_revision_sha": COMMIT_C,
            "repository_identifier": 91,
            "run_identifier": 101,
        }
        artifact = {
            "digest": f"sha256:{archive_sha}",
            "expired": False,
            "id": 201,
            "name": f"wo20-capture-a-{COMMIT_C}-1",
            "size_in_bytes": len(archive),
            "workflow_run": {
                "head_repository_id": 91,
                "head_sha": COMMIT_C,
                "id": 101,
                "repository_id": 91,
            },
        }
        with mock.patch(
            "devtools.governed_result_cli._gh_json",
            side_effect=(
                {"artifacts": [artifact], "total_count": 1},
                artifact,
            ),
        ), mock.patch(
            "devtools.governed_result_cli._gh_archive_bytes",
            return_value=archive,
        ):
            entries = _authenticated_artifact_entries(
                self.repository,
                github_repository="example/repository",
                run=run,
                expected_entries={
                    artifact["name"]: "evidence-a.json",
                },
            )
        observed_raw, provenance = entries[artifact["name"]]
        self.assertEqual(observed_raw, raw)
        self.assertEqual(provenance["archive_sha256"], archive_sha)
        self.assertEqual(provenance["artifact_identifier"], 201)

    def test_actions_zip_rejects_more_than_the_exact_json_entry(self):
        output = io.BytesIO()
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("evidence-a.json", b"{}")
            archive.writestr("unexpected.json", b"{}")
        with self.assertRaisesRegex(
            GovernedResultCLIError,
            "exactly one JSON entry",
        ):
            _safe_zip_json_entry(
                output.getvalue(),
                expected_entry="evidence-a.json",
                label="capture",
            )

    def test_published_result_rejects_noncanonical_adopt_decision(self):
        payload = {
            "decision": "adopt ",
            "event": "experiment_result",
            "evidence": {},
            "experiment_id": EXPERIMENT_ID,
            "plan_record_hash": SHA_A,
            "rationale": "strict_gate_failure",
        }
        with self.assertRaisesRegex(
            GovernedResultCLIError,
            "metadata is not canonical",
        ):
            _validated_published_result_payload(
                payload,
                experiment_id=EXPERIMENT_ID,
            )

    def test_published_result_rejects_extra_payload_keys(self):
        payload = {
            "decision": "reject",
            "event": "experiment_result",
            "evidence": {},
            "experiment_id": EXPERIMENT_ID,
            "plan_record_hash": SHA_A,
            "rationale": "strict_gate_failure",
            "unexpected": True,
        }
        with self.assertRaisesRegex(
            GovernedResultCLIError,
            "not an exact experiment result",
        ):
            _validated_published_result_payload(
                payload,
                experiment_id=EXPERIMENT_ID,
            )

    def test_published_result_rejects_noncanonical_rationale(self):
        payload = {
            "decision": "reject",
            "event": "experiment_result",
            "evidence": {},
            "experiment_id": EXPERIMENT_ID,
            "plan_record_hash": SHA_A,
            "rationale": " strict_gate_failure",
        }
        with self.assertRaisesRegex(
            GovernedResultCLIError,
            "metadata is not canonical",
        ):
            _validated_published_result_payload(
                payload,
                experiment_id=EXPERIMENT_ID,
            )

    def test_grouped_actions_artifact_must_equal_supplied_primary_bytes(self):
        supplied = primary_evidence()
        supplied_raw = _canonical_bytes(supplied)
        different = copy.deepcopy(supplied)
        different["candidate_score"] = 115.5
        artifact = {
            "api_archive_sha256": SHA_A,
            "archive_sha256": SHA_A,
            "entry_sha256": SHA_B,
            "artifact_identifier": 501,
        }
        run = {
            "head_revision_sha": COMMIT_C,
            "repository_identifier": 91,
            "run_attempt": 2,
            "run_identifier": 101,
            "workflow_identifier": 81,
        }
        with mock.patch(
            "devtools.governed_result_cli._authenticated_workflow_run",
            return_value=run,
        ), mock.patch(
            "devtools.governed_result_cli._authenticated_artifact_entries",
            return_value={
                f"wo17-grouped-aggregate-{COMMIT_C}-2": (
                    _canonical_bytes(different),
                    artifact,
                )
            },
        ), mock.patch(
            "devtools.governed_result_cli._validate_primary_committed_bindings"
        ), mock.patch(
            "devtools.governed_result_cli._revision_blob_sha256",
            return_value=SHA_E,
        ), self.assertRaisesRegex(
            GovernedResultCLIError,
            "differs from the authenticated",
        ):
            _authenticated_grouped_provenance(
                self.repository,
                github_repository="example/repository",
                branch="candidate",
                candidate_revision=COMMIT_C,
                primary=supplied,
                primary_raw=supplied_raw,
                topology=topology(),
            )

    def test_grouped_provenance_binds_exact_candidate_workflow_blob(self):
        supplied = primary_evidence()
        supplied_raw = _canonical_bytes(supplied)
        artifact = {
            "api_archive_sha256": SHA_A,
            "archive_sha256": SHA_A,
            "entry_sha256": _sha256_bytes(supplied_raw),
            "artifact_identifier": 501,
        }
        run = {
            "head_revision_sha": COMMIT_C,
            "repository_identifier": 91,
            "run_attempt": 3,
            "run_identifier": 101,
            "workflow_identifier": 81,
        }
        with mock.patch(
            "devtools.governed_result_cli._authenticated_workflow_run",
            return_value=run,
        ), mock.patch(
            "devtools.governed_result_cli._authenticated_artifact_entries",
            return_value={
                f"wo17-grouped-aggregate-{COMMIT_C}-3": (
                    supplied_raw,
                    artifact,
                )
            },
        ), mock.patch(
            "devtools.governed_result_cli._validate_primary_committed_bindings"
        ), mock.patch(
            "devtools.governed_result_cli._revision_blob_sha256",
            return_value=SHA_E,
        ) as blob_sha:
            provenance = _authenticated_grouped_provenance(
                self.repository,
                github_repository="example/repository",
                branch="candidate",
                candidate_revision=COMMIT_C,
                primary=supplied,
                primary_raw=supplied_raw,
                topology=topology(),
            )
        self.assertEqual(provenance["workflow_sha256"], SHA_E)
        self.assertEqual(provenance["run_attempt"], 3)
        blob_sha.assert_called_once_with(
            self.repository,
            COMMIT_C,
            ".github/workflows/wo17-grouped-revalidation.yml",
        )

    def test_runtime_provenance_binds_exact_candidate_workflow_blob(self):
        run = {
            "head_revision_sha": COMMIT_C,
            "repository_identifier": 91,
            "run_attempt": 4,
            "run_identifier": 101,
            "workflow_identifier": 81,
        }
        artifact_a = {
            "api_archive_sha256": SHA_A,
            "archive_sha256": SHA_A,
            "entry_sha256": SHA_B,
            "artifact_identifier": 501,
        }
        artifact_b = {
            "api_archive_sha256": SHA_C,
            "archive_sha256": SHA_C,
            "entry_sha256": SHA_D,
            "artifact_identifier": 502,
        }
        names = {
            f"wo20-capture-a-{COMMIT_C}-4": (b"{}", artifact_a),
            f"wo20-capture-b-{COMMIT_C}-4": (b"{}", artifact_b),
        }
        validated_runtime = runtime_result()
        with mock.patch(
            "devtools.governed_result_cli._authenticated_workflow_run",
            return_value=run,
        ), mock.patch(
            "devtools.governed_result_cli._authenticated_artifact_entries",
            return_value=names,
        ), mock.patch(
            "devtools.governed_result_cli.compare_capture_evidence",
            return_value={"status": "PASS"},
        ), mock.patch(
            "devtools.governed_result_cli._validate_runtime_comparison",
            return_value=validated_runtime,
        ), mock.patch(
            "devtools.governed_result_cli._revision_blob_sha256",
            return_value=SHA_E,
        ) as blob_sha:
            runtime = _authenticated_runtime(
                self.repository,
                github_repository="example/repository",
                branch="candidate",
                candidate_revision=COMMIT_C,
            )
        self.assertEqual(runtime["workflow_sha256"], SHA_E)
        self.assertEqual(runtime["run_attempt"], 4)
        blob_sha.assert_called_once_with(
            self.repository,
            COMMIT_C,
            ".github/workflows/wo20-runtime.yml",
        )

    def test_grouped_tool_hash_must_match_exact_candidate_blob(self):
        with mock.patch(
            "devtools.governed_result_cli._revision_blob_sha256",
            return_value=SHA_E,
        ), self.assertRaisesRegex(
            GovernedResultCLIError,
            "capture_tool_sha256",
        ):
            _validate_primary_committed_bindings(
                self.repository,
                evidence=primary_evidence(),
                topology=topology(),
            )

    def test_reject_records_one_result_and_successor(self):
        result = self._record()
        ledger = ExperimentLedger(self.stores["experiment_ledger"].path)
        self.assertEqual(len(ledger.plans()), 1)
        self.assertEqual(len(ledger.results()), 1)
        self.assertEqual(result["decision"], "reject")
        self.assertEqual(
            result["status"],
            "experiment_result_recorded_locally_not_published",
        )
        self.assertIn("verify-result-authority", result["next_required_action"])

    def test_reject_ledger_preserves_official_net_score_and_penalty(self):
        evidence = primary_evidence(passed=False)
        evidence["score_components"]["candidate_missing_penalty"] = 2.0
        evidence["score_components"]["candidate_total_score"] -= 2.0
        evidence["candidate_score"] -= 2.0
        evidence["score_delta"] -= 2.0
        recorded = _ledger_evidence(
            evidence,
            plan=plan(),
            runtime=_runtime_defaults(SHA_A),
            runtime_leakage_clean=True,
        )
        self.assertEqual(recorded["metrics"]["missing_penalty"], 2.0)
        self.assertEqual(
            recorded["metrics"]["official_total_score"],
            evidence["candidate_score"],
        )

    def test_adopt_rejects_local_runtime_and_binds_authenticated_actions(self):
        self.primary = primary_evidence(passed=True)
        self.primary["experiment_plan_record_sha256"] = (
            self.plan_record_hash
        )
        with self.assertRaisesRegex(
            GovernedResultCLIError, "authenticated grouped"
        ):
            _validate_decision(
                self.primary,
                decision="adopt",
                grouped_actions_authenticated=False,
                runtime_actions_authenticated=False,
                runtime_leakage_clean=True,
            )
        with self.assertRaisesRegex(
            GovernedResultCLIError, "forbids caller-authored"
        ):
            self._record(
                decision="adopt",
                runtime_paths=(
                    self.external / "runtime-1.json",
                    self.external / "runtime-2.json",
                ),
            )
        result = self._record(decision="adopt")
        self.assertEqual(result["decision"], "adopt")
        recorded = ExperimentLedger(
            self.stores["experiment_ledger"].path
        ).results()[0]["evidence"]
        self.assertEqual(recorded["runtime_evidence_sha256"], SHA_A)
        self.assertEqual(
            recorded["metrics"]["wo20_actions_run"],
            301,
        )
        self.assertEqual(
            recorded["metrics"]["wo17_actions_run"],
            601,
        )
        self.assertEqual(
            recorded["metrics"]["official_total_score"],
            self.primary["candidate_score"],
        )
        self.assertEqual(recorded["metrics"]["missing_penalty"], 0.0)
        self.assertTrue(
            recorded["checks"]["runtime_actions_authenticated"]
        )
        self.assertTrue(
            recorded["checks"]["grouped_actions_authenticated"]
        )

    def test_adopt_rejects_nonzero_missing_penalty(self):
        evidence = primary_evidence(passed=True)
        evidence["score_components"]["candidate_missing_penalty"] = 1.0
        evidence["score_components"]["candidate_total_score"] -= 1.0
        evidence["candidate_score"] -= 1.0
        evidence["score_delta"] -= 1.0
        with self.assertRaisesRegex(
            GovernedResultCLIError, "zero candidate missing"
        ):
            _validate_decision(
                evidence,
                decision="adopt",
                grouped_actions_authenticated=True,
                runtime_actions_authenticated=True,
                runtime_leakage_clean=True,
            )

    def test_published_adoption_requires_exact_reauthenticated_provenance(self):
        primary = primary_evidence(passed=True)
        unauthenticated = _ledger_evidence(
            primary,
            plan=plan(),
            runtime=_runtime_defaults(SHA_A),
            runtime_leakage_clean=True,
        )
        with mock.patch(
            "devtools.governed_result_cli._download_authenticated_grouped",
            return_value=(
                primary,
                _canonical_bytes(primary),
                grouped_provenance(),
            ),
        ), mock.patch(
            "devtools.governed_result_cli._authenticated_runtime",
            return_value=authenticated_runtime_result(),
        ), mock.patch(
            "devtools.governed_result_cli._runtime_leakage_finding_count",
            return_value=0,
        ), self.assertRaisesRegex(
            GovernedResultCLIError,
            "differs from authenticated GitHub Actions provenance",
        ):
            _verify_adopted_result_provenance(
                self.repository,
                github_repository="example/repository",
                branch="candidate",
                candidate_revision=COMMIT_C,
                plan=plan(),
                plan_record_hash=SHA_D,
                topology=topology(),
                recorded_evidence=unauthenticated,
            )

    def test_trusted_runtime_recomputes_exact_source_comparison(self):
        compared = {
            "schema_version": "mib-wo20-parallel-determinism/v1",
            "status": "PASS",
            "blocking_reasons": [],
            "warnings": [],
            "aggregate_only": True,
            "source_binding": {
                "git_revision": COMMIT_C,
                "dockerfile_sha256": SHA_A,
                "requirements_lock_sha256": SHA_A,
                "run_sh_sha256": SHA_A,
                "solution_sha256": SHA_A,
                "harness_sha256": SHA_A,
                "producer_graph_sha256": SHA_B,
            },
            "runtime": {
                "all_captures_within_official_runtime_limit": True,
                "all_captures_within_official_memory_limit": True,
                "max_elapsed_seconds": 10.0,
                "peak_container_memory_bytes": 4096,
                "peak_process_tree_rss_bytes": 2048,
            },
            "determinism": {
                "byte_identical": True,
                "coverage_identical": True,
                "output_bytes": 1024,
                "output_sha256": SHA_D,
            },
            "images": {
                "source_binding_verified": True,
                "size_bytes": [2048, 2048],
            },
            "installed_model_artifacts": {
                "maximum_artifact_bytes": 0,
                "total_bytes": 0,
            },
        }
        with mock.patch(
            "devtools.governed_result_cli.compare_capture_evidence",
            return_value=compared,
        ) as comparator, mock.patch(
            "devtools.governed_result_cli._revision_blob_sha256",
            return_value=SHA_A,
        ), mock.patch(
            "devtools.governed_result_cli._runtime_producer_graph_sha256",
            return_value=SHA_B,
        ):
            runtime = _trusted_runtime(
                self.repository,
                candidate_revision=COMMIT_C,
                capture_paths=("one", "two"),
            )
        comparator.assert_called_once()
        self.assertTrue(runtime["runtime_limits_verified"])
        self.assertEqual(runtime["runtime_seconds"], 10.0)
        self.assertEqual(runtime["prediction_output_sha256"], SHA_D)

    def test_concurrent_authority_change_blocks_before_append(self):
        before = self.stores["experiment_ledger"].path.read_bytes()
        authority = _FakeAuthority(
            self.current, self.pointer, COMMIT_C, drift=True
        )
        patches = self._patches(authority=authority)
        with patches[0], patches[1], patches[2], patches[3], patches[
            4
        ], patches[5], patches[6], patches[7], self.assertRaisesRegex(
            GovernedResultCLIError, "authority changed"
        ):
            record_result(
                repository_root=self.repository,
                github_repository="example/repository",
                branch="candidate",
                experiment_id=EXPERIMENT_ID,
                evidence_path=self.evidence_path,
                decision="reject",
                rationale="strict_gate_failure",
            )
        self.assertEqual(
            self.stores["experiment_ledger"].path.read_bytes(), before
        )

    def test_partial_pointer_write_rolls_back_every_local_artifact(self):
        before_ledger = self.stores["experiment_ledger"].path.read_bytes()
        before_pointer = self.pointer_path.read_bytes()
        before_checkpoints = set(self.program.glob("[0-9a-f]" * 64 + ".json"))
        original = _atomic_replace_exact
        calls = 0

        def replace_then_fail(*args, **kwargs):
            nonlocal calls
            calls += 1
            original(*args, **kwargs)
            if calls == 1:
                raise OSError("simulated partial pointer write")

        patches = self._patches()
        with patches[0], patches[1], patches[2], patches[3], patches[
            4
        ], patches[5], patches[6], patches[7], mock.patch(
            "devtools.governed_result_cli._atomic_replace_exact",
            side_effect=replace_then_fail,
        ), self.assertRaisesRegex(OSError, "partial pointer"):
            record_result(
                repository_root=self.repository,
                github_repository="example/repository",
                branch="candidate",
                experiment_id=EXPERIMENT_ID,
                evidence_path=self.evidence_path,
                decision="reject",
                rationale="strict_gate_failure",
            )
        self.assertEqual(self.pointer_path.read_bytes(), before_pointer)
        self.assertEqual(
            self.stores["experiment_ledger"].path.read_bytes(),
            before_ledger,
        )
        self.assertEqual(
            set(self.program.glob("[0-9a-f]" * 64 + ".json")),
            before_checkpoints,
        )

    def test_remote_head_drift_after_append_rolls_back(self):
        before_ledger = self.stores["experiment_ledger"].path.read_bytes()
        before_pointer = self.pointer_path.read_bytes()
        patches = self._patches(remote_head="f" * 40)
        with patches[0], patches[1], patches[2], patches[3], patches[
            4
        ], patches[5], patches[6], patches[7], self.assertRaisesRegex(
            GovernedResultCLIError, "branch moved"
        ):
            record_result(
                repository_root=self.repository,
                github_repository="example/repository",
                branch="candidate",
                experiment_id=EXPERIMENT_ID,
                evidence_path=self.evidence_path,
                decision="reject",
                rationale="strict_gate_failure",
            )
        self.assertEqual(self.pointer_path.read_bytes(), before_pointer)
        self.assertEqual(
            self.stores["experiment_ledger"].path.read_bytes(),
            before_ledger,
        )

    def test_candidate_authority_rejects_unpublished_head_drift(self):
        fake = _FakeAuthority(
            self.current, self.pointer, COMMIT_C
        )
        current_plan = plan()
        patched_topology = GovernedExperimentTopology(
            **{
                **topology().__dict__,
                "prereg_checkpoint_sha256": (
                    self.current.expected_sha256
                ),
            }
        )
        with mock.patch(
            "devtools.governed_result_cli._program_paths",
            return_value=(
                self.program,
                self.pointer_path,
                self.program,
                self.stores,
            ),
        ), mock.patch(
            "devtools.governed_result_cli.GitHubCheckpointAuthority",
            return_value=fake,
        ), mock.patch(
            "devtools.governed_result_cli._load_exact_plan",
            return_value=(current_plan, SHA_D),
        ), mock.patch(
            "devtools.governed_result_cli._direct_parent",
            return_value=COMMIT_P,
        ), mock.patch(
            "devtools.governed_result_cli.verify_governed_experiment_topology",
            return_value=patched_topology,
        ), mock.patch(
            "devtools.governed_result_cli._remote_head",
            return_value="f" * 40,
        ), self.assertRaisesRegex(
            GovernedResultCLIError, "moved during candidate verification"
        ):
            verify_candidate_authority(
                repository_root=self.repository,
                github_repository="example/repository",
                branch="candidate",
                experiment_id=EXPERIMENT_ID,
            )

    def test_result_authority_rejects_different_topology_plan_record(self):
        records = (
            {
                "payload": {
                    "event": "experiment_plan",
                    "experiment_id": EXPERIMENT_ID,
                    "plan": plan(),
                },
                "record_hash": SHA_D,
            },
            {
                "payload": {
                    "event": "experiment_result",
                    "experiment_id": EXPERIMENT_ID,
                },
                "record_hash": SHA_E,
            },
        )
        experiment_store = mock.Mock()
        experiment_store.verify.return_value = records
        stores = {
            **self.stores,
            "experiment_ledger": experiment_store,
        }
        mismatched = GovernedExperimentTopology(
            **{
                **topology().__dict__,
                "experiment_plan_record_sha256": SHA_A,
            }
        )
        authority = _FakeAuthority(
            self.current,
            self.pointer,
            "4" * 40,
        )
        with mock.patch(
            "devtools.governed_result_cli._program_paths",
            return_value=(
                self.program,
                self.pointer_path,
                self.program,
                stores,
            ),
        ), mock.patch(
            "devtools.governed_result_cli.GitHubCheckpointAuthority",
            return_value=authority,
        ), mock.patch(
            "devtools.governed_result_cli._direct_parent",
            side_effect=(COMMIT_C, COMMIT_P),
        ), mock.patch(
            "devtools.governed_result_cli.verify_governed_experiment_topology",
            return_value=mismatched,
        ), self.assertRaisesRegex(
            GovernedResultCLIError,
            "different plan record",
        ):
            verify_result_authority(
                repository_root=self.repository,
                github_repository="example/repository",
                branch="candidate",
                experiment_id=EXPERIMENT_ID,
            )

    def test_result_authority_rejects_adopt_with_trailing_space(self):
        records = (
            {
                "payload": {
                    "event": "experiment_plan",
                    "experiment_id": EXPERIMENT_ID,
                    "plan": plan(),
                },
                "record_hash": SHA_D,
            },
            {
                "payload": {
                    "decision": "adopt ",
                    "event": "experiment_result",
                    "evidence": {},
                    "experiment_id": EXPERIMENT_ID,
                    "plan_record_hash": SHA_D,
                    "rationale": "strict_gate_failure",
                },
                "record_hash": SHA_E,
            },
        )
        experiment_store = mock.Mock()
        experiment_store.verify.return_value = records
        stores = {
            **self.stores,
            "experiment_ledger": experiment_store,
        }
        bound_topology = GovernedExperimentTopology(
            **{
                **topology().__dict__,
                "experiment_plan_record_sha256": SHA_D,
            }
        )
        authority = _FakeAuthority(
            self.current,
            self.pointer,
            "4" * 40,
        )
        with mock.patch(
            "devtools.governed_result_cli._program_paths",
            return_value=(
                self.program,
                self.pointer_path,
                self.program,
                stores,
            ),
        ), mock.patch(
            "devtools.governed_result_cli.GitHubCheckpointAuthority",
            return_value=authority,
        ), mock.patch(
            "devtools.governed_result_cli._direct_parent",
            side_effect=(COMMIT_C, COMMIT_P),
        ), mock.patch(
            "devtools.governed_result_cli.verify_governed_experiment_topology",
            return_value=bound_topology,
        ), mock.patch(
            "devtools.governed_result_cli._verify_adopted_result_provenance"
        ) as reauthenticate, self.assertRaisesRegex(
            GovernedResultCLIError,
            "metadata is not canonical",
        ):
            verify_result_authority(
                repository_root=self.repository,
                github_repository="example/repository",
                branch="candidate",
                experiment_id=EXPERIMENT_ID,
            )
        reauthenticate.assert_not_called()

    def test_result_authority_accepts_event_aware_result_only_transition(self):
        current_plan = plan()
        primary = primary_evidence(passed=False)
        ledger_evidence = _ledger_evidence(
            primary,
            plan=current_plan,
            runtime=_runtime_defaults(SHA_A),
            runtime_leakage_clean=True,
        )
        plan_record = {
            "payload": {
                "event": "experiment_plan",
                "experiment_id": EXPERIMENT_ID,
                "plan": current_plan,
            },
            "record_hash": SHA_D,
            "sequence": 1,
        }
        result_record = {
            "payload": {
                "decision": "reject",
                "event": "experiment_result",
                "evidence": ledger_evidence,
                "experiment_id": EXPERIMENT_ID,
                "plan_record_hash": SHA_D,
                "rationale": "strict_gate_failure",
            },
            "record_hash": SHA_E,
            "sequence": 2,
        }
        experiment_store = mock.Mock()
        experiment_store.verify.return_value = (
            plan_record,
            result_record,
        )
        stores = {
            **self.stores,
            "experiment_ledger": experiment_store,
        }
        result_revision = "4" * 40
        result_checkpoint_sha = "9" * 64
        pointer = {
            "checkpoint_path": (
                f"evaluation/program/{result_checkpoint_sha}.json"
            ),
            "checkpoint_sha256": result_checkpoint_sha,
            "previous_checkpoint_sha256": SHA_E,
            "schema": "mib-program-current-checkpoint/v1",
        }
        current = mock.Mock(expected_sha256=result_checkpoint_sha)
        authority = _FakeAuthority(
            current, pointer, result_revision
        )
        parent_anchor = {
            "path": "experiment_ledger.jsonl",
            "sha256": SHA_A,
            "expected_length": 1,
            "expected_head": SHA_D,
        }
        other_anchor = {
            "path": "other.jsonl",
            "sha256": SHA_B,
            "expected_length": 0,
            "expected_head": "0" * 64,
        }
        parent_checkpoint = {
            "baseline_manifest_sha256": SHA_A,
            "promotion_population": {
                name: current_plan[name]
                for name in (
                    "evaluator_sha256",
                    "expected_record_count",
                    "input_tree_sha256",
                    "runtime_contract_sha256",
                    "split_manifest_sha256",
                    "truth_sha256",
                )
            },
            "runtime_leakage_finding_count": 0,
            "stores": {
                "candidate_state_ledger": other_anchor,
                "experiment_ledger": parent_anchor,
                "protected_access_ledger": other_anchor,
                "taint_registry": other_anchor,
            },
        }
        parent_ledger = b"before\n"
        result_ledger = b"before\nafter\n"
        result_checkpoint = copy.deepcopy(parent_checkpoint)
        result_checkpoint["stores"]["experiment_ledger"] = {
            **parent_anchor,
            "expected_length": 2,
            "expected_head": SHA_E,
            "sha256": _sha256_bytes(result_ledger),
        }
        transition = tuple(
            SimpleNamespace(
                path=path,
                status=status,
                old_mode=old_mode,
                new_mode="100644",
            )
            for path, status, old_mode in (
                (
                    "evaluation/program/experiment_ledger.jsonl",
                    "M",
                    "100644",
                ),
                (
                    "evaluation/program/current_checkpoint.json",
                    "M",
                    "100644",
                ),
                (pointer["checkpoint_path"], "A", "000000"),
            )
        )
        bound_topology = GovernedExperimentTopology(
            **{
                **topology().__dict__,
                "experiment_plan_record_sha256": SHA_D,
                "prereg_checkpoint_sha256": SHA_E,
            }
        )

        def blob(_root, revision, path):
            if path.endswith("current_checkpoint.json"):
                return b"pointer\n"
            if path.endswith("experiment_ledger.jsonl"):
                return (
                    parent_ledger
                    if revision == COMMIT_C
                    else result_ledger
                )
            raise AssertionError((revision, path))

        with mock.patch(
            "devtools.governed_result_cli._program_paths",
            return_value=(
                self.program,
                self.pointer_path,
                self.program,
                stores,
            ),
        ), mock.patch(
            "devtools.governed_result_cli.GitHubCheckpointAuthority",
            return_value=authority,
        ), mock.patch(
            "devtools.governed_result_cli._direct_parent",
            side_effect=(COMMIT_C, COMMIT_P),
        ), mock.patch(
            "devtools.governed_result_cli.verify_governed_experiment_topology",
            return_value=bound_topology,
        ), mock.patch(
            "devtools.governed_result_cli._raw_diff",
            return_value=(transition, SHA_C),
        ), mock.patch(
            "devtools.governed_result_cli._git_blob",
            side_effect=blob,
        ), mock.patch(
            "devtools.governed_result_cli._checkpoint_pointer",
            return_value={
                "checkpoint_path": (
                    f"evaluation/program/{SHA_E}.json"
                ),
                "checkpoint_sha256": SHA_E,
                "previous_checkpoint_sha256": SHA_B,
                "schema": "mib-program-current-checkpoint/v1",
            },
        ), mock.patch(
            "devtools.governed_result_cli._verified_checkpoint",
            side_effect=(
                (parent_checkpoint, b"parent"),
                (result_checkpoint, b"result"),
            ),
        ), mock.patch(
            "devtools.governed_result_cli._remote_head",
            return_value=result_revision,
        ):
            receipt = verify_result_authority(
                repository_root=self.repository,
                github_repository="example/repository",
                branch="candidate",
                experiment_id=EXPERIMENT_ID,
            )
        self.assertEqual(
            receipt["status"],
            "authenticated_experiment_result_verified",
        )
        self.assertEqual(receipt["decision"], "reject")


if __name__ == "__main__":
    unittest.main()
