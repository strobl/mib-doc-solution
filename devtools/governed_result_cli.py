#!/usr/bin/env python3
"""Authenticated, fail-closed publication of one governed experiment result.

The command deliberately stops before committing or pushing.  It resolves the
current checkpoint and candidate revision from authenticated GitHub, proves the
exact A→P→C experiment topology, validates one external aggregate evidence
artifact, and appends exactly one plan-bound ``experiment_result``.  The ledger,
successor checkpoint, and pointer are rolled back together on every detected
race or partial write.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import pwd
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.experiment_control import (  # noqa: E402
    CheckpointAuthorityResolver,
    ExperimentControlError,
    ExperimentLedger,
    IntegrityError,
    ProgramIntegrityCheckpoint,
    _EXPERIMENT_RESULT_DECISIONS,
    _SAFE_DIMENSION_RE,
    _normalize_experiment_result_evidence,
    _require_nonidentifying_control_text,
    canonical_json,
    require_aggregate_only,
)
from devtools.governed_experiment_cli import (  # noqa: E402
    CHECKPOINT_DIRECTORY_RELATIVE_PATH,
    LEDGER_FILENAMES,
    POINTER_RELATIVE_PATH,
    PROGRAM_RELATIVE_PATH,
    GitHubCheckpointAuthority,
    GovernedExperimentCLIError,
    _atomic_replace_exact,
    _canonical_bytes,
    _create_once,
    _decode_canonical_object,
    _gh_json,
    _git,
    _ledger_bytes,
    _pointer_for,
    _program_paths,
    _read_regular_file,
    _remote_head,
    _require_exact_worktree_changes,
    _require_external_canonical_object,
    _runtime_leakage_finding_count,
    _sha256_bytes,
    _trusted_gh_executable,
    _unlink_exact,
)
from devtools.grouped_policy_revalidation_evidence import (  # noqa: E402
    GovernedExperimentTopology,
    _checkpoint_pointer,
    _direct_parent,
    _git_blob,
    _raw_diff,
    _verified_checkpoint,
    verify_governed_experiment_topology,
)
from devtools.wo20_parallel_compare import (  # noqa: E402
    GIB,
    OUTPUT_SCHEMA as WO20_OUTPUT_SCHEMA,
    compare_capture_evidence,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_EXPERIMENT_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,79}")
_GITHUB_REPOSITORY_RE = re.compile(
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
)
_WO20_WORKFLOW_PATH = ".github/workflows/wo20-runtime.yml"
_WO17_WORKFLOW_PATH = (
    ".github/workflows/wo17-grouped-revalidation.yml"
)
_WO17_ARTIFACT_PREFIX = "wo17-grouped-aggregate"
_WO17_ARTIFACT_ENTRY = "wo17-grouped-aggregate.json"
_GROUPED_TRUTH_ADJACENT_CONTROL_PATHS = (
    "devtools/__init__.py",
    "devtools/evaluation.py",
    "devtools/experiment_control.py",
    "devtools/grouped_policy_revalidation_evidence.py",
    "devtools/grouped_policy_revalidation_gate.py",
    "devtools/grouped_split_evidence.py",
    "devtools/layout_manifest_freezer.py",
    "devtools/policy_grouped_capture.py",
    "devtools/policy_grouped_capture_contract.py",
    "devtools/policy_revalidation_audit_contract.py",
    "devtools/policy_revalidation_contract_probe.py",
    "devtools/wo18_production_capture.py",
    "scripts/evaluate.py",
)
_MAX_ACTIONS_ARCHIVE_BYTES = 16 * 1024 * 1024
_MAX_ACTIONS_JSON_BYTES = 8 * 1024 * 1024
_MAX_ACTIONS_RESULTS = 100
_RESULT_PAYLOAD_KEYS = frozenset(
    {
        "decision",
        "event",
        "evidence",
        "experiment_id",
        "plan_record_hash",
        "rationale",
    }
)

_PRIMARY_ROOT_KEYS = frozenset(
    {
        "base_revision_sha",
        "candidate_capture_set_sha256",
        "candidate_diff_manifest_sha256",
        "candidate_producer_graph_sha256",
        "candidate_revision_sha",
        "candidate_score",
        "candidate_source_diff_sha256",
        "candidate_source_revision_sha",
        "capture_tool_sha256",
        "catastrophic_false_approvals",
        "checks",
        "confusion_counts",
        "control_capture_set_sha256",
        "control_producer_graph_sha256",
        "control_revision_sha",
        "control_score",
        "control_source_revision_sha",
        "counts",
        "deterministic",
        "duplicate_records",
        "evaluated_fold_count",
        "evaluation_mode",
        "evaluator_sha256",
        "evidence_label",
        "evidence_tool_sha256",
        "experiment_plan_record_sha256",
        "experiment_plan_sha256",
        "extra_records",
        "false_approvals",
        "fold_consistent",
        "fold_count",
        "fold_deltas",
        "fold_metrics",
        "fold_weights",
        "gate_results",
        "gate_tool_sha256",
        "hard_gate_failure_count",
        "hypothesis_sha256",
        "input_tree_sha256",
        "invalid_records",
        "layout_group_count",
        "metrics",
        "missing_records",
        "planned_scope_manifest_sha256",
        "prereg_checkpoint_sha256",
        "prereg_governance_diff_sha256",
        "primary_variable_sha256",
        "record_count",
        "repeat_count",
        "repeat_scores",
        "runtime_contract_sha256",
        "score_components",
        "score_delta",
        "split_manifest_sha256",
        "status",
        "truth_sha256",
    }
)
_COUNT_KEYS = frozenset(
    {
        "candidate_catastrophic_false_approval_count",
        "candidate_duplicate_record_count",
        "candidate_extra_record_count",
        "candidate_false_approval_count",
        "candidate_invalid_record_count",
        "candidate_missing_record_count",
        "decision_or_confidence_change_count",
        "eligible_guarded_initial_count",
        "guarded_initial_approval_count",
        "late_revalidation_approval_count",
        "layout_group_count",
        "legacy_forced_approval_count",
        "new_catastrophic_false_approval_count",
        "new_false_approval_count",
        "non_decision_field_change_count",
        "record_count",
        "regression_adversarial_failure_count",
        "regression_focused_failure_count",
        "regression_full_failure_count",
        "unguarded_initial_approval_count",
    }
)
_SCORE_COMPONENT_KEYS = frozenset(
    {
        "candidate_calibration_score",
        "candidate_classification_score",
        "candidate_extraction_score",
        "candidate_missing_penalty",
        "candidate_total_score",
        "control_calibration_score",
        "control_classification_score",
        "control_extraction_score",
        "control_missing_penalty",
        "control_total_score",
    }
)
_GROUPED_GATE_KEYS = frozenset(
    {
        "candidate_complete",
        "capture_deterministic",
        "contract_probes_pass",
        "control_complete",
        "decision_variable_nonvacuous",
        "extraction_score_unchanged",
        "full_score_positive",
        "group_exclusive",
        "guarded_initial_approval_nonvacuous",
        "legacy_forced_counter_fully_accounted",
        "leave_best_fold_out_positive",
        "no_catastrophic_false_approvals",
        "no_late_revalidation_approval",
        "no_negative_folds",
        "no_new_false_approvals",
        "no_unguarded_initial_approval",
        "non_decision_fields_unchanged",
        "paired_fold_members",
        "population_bound",
        "public_exposed_evidence",
        "regression_suites_clean",
        "repeat_weighted_deltas_positive",
        "runtime_contract_bound",
        "source_and_diff_bound",
        "split_deterministic",
    }
)
_CONFUSION_KEYS = frozenset(
    f"{arm}_{truth}_to_{prediction}_count"
    for arm in ("control", "candidate")
    for truth in ("approved", "denied", "needs_review")
    for prediction in ("approved", "denied", "needs_review", "missing")
)


class GovernedResultCLIError(ExperimentControlError):
    """The authenticated experiment-result boundary failed closed."""


def _digest(value: Any, *, label: str) -> str:
    normalized = str(value).strip().casefold()
    if not _SHA256_RE.fullmatch(normalized):
        raise GovernedResultCLIError(
            f"{label} must be a full SHA-256 digest"
        )
    return normalized


def _commit(value: Any, *, label: str) -> str:
    normalized = str(value).strip().casefold()
    if not _COMMIT_RE.fullmatch(normalized):
        raise GovernedResultCLIError(
            f"{label} must be a full Git commit SHA"
        )
    return normalized


def _count(value: Any, *, label: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GovernedResultCLIError(
            f"{label} must be a "
            f"{'positive' if positive else 'non-negative'} integer"
        )
    return value


def _number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise GovernedResultCLIError(f"{label} must be a finite number")
    return float(value)


def _same(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise GovernedResultCLIError(f"{label} does not match its authority")


def _positive_identifier(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GovernedResultCLIError(
            f"{label} must be a positive GitHub numeric identifier"
        )
    return value


def _actions_environment() -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(
            ("GIT_", "GH_", "GITHUB_", "DYLD_", "LD_")
        )
        and name != "XDG_CONFIG_HOME"
    }
    if hasattr(os, "geteuid"):
        environment["HOME"] = pwd.getpwuid(os.geteuid()).pw_dir
    environment["NO_COLOR"] = "1"
    return environment


def _gh_archive_bytes(
    repository_root: Path,
    endpoint: str,
) -> bytes:
    """Download one bounded Actions archive through authenticated ``gh``."""

    arguments = (
        _trusted_gh_executable(),
        "api",
        "--hostname",
        "github.com",
        "--method",
        "GET",
        endpoint,
    )
    try:
        completed = subprocess.run(
            arguments,
            cwd=repository_root,
            check=False,
            capture_output=True,
            env=_actions_environment(),
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GovernedResultCLIError(
            "authenticated GitHub artifact download failed"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise GovernedResultCLIError(
            "authenticated GitHub artifact download failed"
            + (f": {detail}" if detail else "")
        )
    raw = bytes(completed.stdout)
    if not raw or len(raw) > _MAX_ACTIONS_ARCHIVE_BYTES:
        raise GovernedResultCLIError(
            "authenticated GitHub artifact archive has an invalid size"
        )
    return raw


def _safe_zip_json_entry(
    archive: bytes,
    *,
    expected_entry: str,
    label: str,
) -> bytes:
    if (
        not expected_entry
        or "/" in expected_entry
        or "\\" in expected_entry
        or "\0" in expected_entry
        or not expected_entry.endswith(".json")
    ):
        raise GovernedResultCLIError(
            f"{label} expected ZIP entry is invalid"
        )
    try:
        with zipfile.ZipFile(io.BytesIO(archive), mode="r") as bundle:
            members = bundle.infolist()
            if len(members) != 1:
                raise GovernedResultCLIError(
                    f"{label} archive must contain exactly one JSON entry"
                )
            member = members[0]
            unix_mode = (member.external_attr >> 16) & 0xFFFF
            file_type = stat.S_IFMT(unix_mode)
            if (
                member.filename != expected_entry
                or member.is_dir()
                or member.flag_bits & 0x1
                or member.compress_type
                not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or file_type not in {0, stat.S_IFREG}
                or member.file_size < 1
                or member.file_size > _MAX_ACTIONS_JSON_BYTES
                or member.compress_size > _MAX_ACTIONS_ARCHIVE_BYTES
            ):
                raise GovernedResultCLIError(
                    f"{label} archive entry violates the exact JSON contract"
                )
            raw = bundle.read(member)
    except GovernedResultCLIError:
        raise
    except (
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as exc:
        raise GovernedResultCLIError(
            f"{label} archive is not a safe readable ZIP"
        ) from exc
    if len(raw) != member.file_size:
        raise GovernedResultCLIError(
            f"{label} archive entry size changed during extraction"
        )
    return raw


def _repository_numeric_identifier(
    value: Any,
    *,
    github_repository: str,
    label: str,
) -> int:
    if not isinstance(value, Mapping):
        raise GovernedResultCLIError(
            f"{label} GitHub repository identity is malformed"
        )
    full_name = value.get("full_name")
    if (
        not isinstance(full_name, str)
        or full_name.casefold() != github_repository.casefold()
    ):
        raise GovernedResultCLIError(
            f"{label} GitHub repository identity does not match authority"
        )
    return _positive_identifier(
        value.get("id"),
        label=f"{label} repository identifier",
    )


def _normalize_workflow_run(
    value: Any,
    *,
    github_repository: str,
    branch: str,
    candidate_revision: str,
    workflow_path: str,
    workflow_identifier: int,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernedResultCLIError(
            "GitHub Actions workflow run is malformed"
        )
    revision = str(value.get("head_sha", "")).casefold()
    if (
        revision != candidate_revision
        or not _COMMIT_RE.fullmatch(revision)
        or value.get("head_branch") != branch
        or value.get("path") != workflow_path
        or value.get("status") != "completed"
        or value.get("conclusion") != "success"
        or value.get("event") not in {"push", "workflow_dispatch"}
    ):
        raise GovernedResultCLIError(
            "GitHub Actions run is not a successful exact-candidate "
            "approved-workflow run"
        )
    repository_identifier = _repository_numeric_identifier(
        value.get("repository"),
        github_repository=github_repository,
        label="workflow run",
    )
    head_repository_identifier = _repository_numeric_identifier(
        value.get("head_repository"),
        github_repository=github_repository,
        label="workflow run head",
    )
    if repository_identifier != head_repository_identifier:
        raise GovernedResultCLIError(
            "GitHub Actions run came from a different head repository"
        )
    observed_workflow = _positive_identifier(
        value.get("workflow_id"),
        label="workflow run workflow identifier",
    )
    if observed_workflow != workflow_identifier:
        raise GovernedResultCLIError(
            "GitHub Actions run does not bind the approved workflow"
        )
    return {
        "event": value["event"],
        "head_branch": branch,
        "head_revision_sha": revision,
        "repository_identifier": repository_identifier,
        "run_attempt": _positive_identifier(
            value.get("run_attempt"),
            label="workflow run attempt",
        ),
        "run_identifier": _positive_identifier(
            value.get("id"),
            label="workflow run identifier",
        ),
        "workflow_identifier": observed_workflow,
        "workflow_path": workflow_path,
    }


def _authenticated_workflow_run(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    candidate_revision: str,
    workflow_path: str,
) -> dict[str, Any]:
    if not _GITHUB_REPOSITORY_RE.fullmatch(github_repository):
        raise GovernedResultCLIError(
            "GitHub repository must be OWNER/REPOSITORY"
        )
    response = _gh_json(
        repository_root,
        f"repos/{github_repository}/actions/runs",
        fields={
            "head_sha": candidate_revision,
            "per_page": str(_MAX_ACTIONS_RESULTS),
            "status": "success",
        },
    )
    raw_runs = response.get("workflow_runs")
    total = response.get("total_count")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or not isinstance(raw_runs, list)
        or total != len(raw_runs)
        or total > _MAX_ACTIONS_RESULTS
    ):
        raise GovernedResultCLIError(
            "successful candidate workflow-run listing is truncated "
            "or malformed"
        )
    matching_runs = [
        value
        for value in raw_runs
        if isinstance(value, Mapping)
        and value.get("path") == workflow_path
    ]
    if len(matching_runs) != 1:
        raise GovernedResultCLIError(
            "exactly one successful approved workflow run must exist "
            "for the candidate"
        )
    raw_run = matching_runs[0]
    workflow_identifier = _positive_identifier(
        raw_run.get("workflow_id"),
        label="approved workflow identifier",
    )
    workflow = _gh_json(
        repository_root,
        (
            f"repos/{github_repository}/actions/workflows/"
            f"{workflow_identifier}"
        ),
    )
    if (
        workflow.get("id") != workflow_identifier
        or workflow.get("path") != workflow_path
        or workflow.get("state") != "active"
    ):
        raise GovernedResultCLIError(
            "approved GitHub Actions workflow is not active at its exact path"
        )
    normalized = _normalize_workflow_run(
        raw_run,
        github_repository=github_repository,
        branch=branch,
        candidate_revision=candidate_revision,
        workflow_path=workflow_path,
        workflow_identifier=workflow_identifier,
    )
    exact = _normalize_workflow_run(
        _gh_json(
            repository_root,
            (
                f"repos/{github_repository}/actions/runs/"
                f"{normalized['run_identifier']}"
            ),
        ),
        github_repository=github_repository,
        branch=branch,
        candidate_revision=candidate_revision,
        workflow_path=workflow_path,
        workflow_identifier=workflow_identifier,
    )
    if exact != normalized:
        raise GovernedResultCLIError(
            "GitHub Actions run identity changed during authentication"
        )
    return exact


def _normalize_actions_artifact(
    value: Any,
    *,
    expected_name: str,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GovernedResultCLIError(
            "GitHub Actions artifact metadata is malformed"
        )
    digest = value.get("digest")
    if (
        value.get("name") != expected_name
        or value.get("expired") is not False
        or not isinstance(digest, str)
        or not digest.startswith("sha256:")
        or not _SHA256_RE.fullmatch(digest[7:].casefold())
    ):
        raise GovernedResultCLIError(
            "GitHub Actions artifact is expired or lacks an exact API digest"
        )
    size = value.get("size_in_bytes")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
        or size > _MAX_ACTIONS_ARCHIVE_BYTES
    ):
        raise GovernedResultCLIError(
            "GitHub Actions artifact metadata has an invalid archive size"
        )
    workflow_run = value.get("workflow_run")
    if not isinstance(workflow_run, Mapping):
        raise GovernedResultCLIError(
            "GitHub Actions artifact lacks workflow-run provenance"
        )
    if (
        workflow_run.get("id") != run["run_identifier"]
        or str(workflow_run.get("head_sha", "")).casefold()
        != run["head_revision_sha"]
        or workflow_run.get("repository_id")
        != run["repository_identifier"]
        or workflow_run.get("head_repository_id")
        != run["repository_identifier"]
    ):
        raise GovernedResultCLIError(
            "GitHub Actions artifact does not belong to the authenticated run"
        )
    return {
        "api_archive_sha256": digest[7:].casefold(),
        "artifact_identifier": _positive_identifier(
            value.get("id"),
            label="Actions artifact identifier",
        ),
        "artifact_name": expected_name,
        "archive_size_bytes": size,
        "run_identifier": run["run_identifier"],
    }


def _authenticated_artifact_entries(
    repository_root: Path,
    *,
    github_repository: str,
    run: Mapping[str, Any],
    expected_entries: Mapping[str, str],
) -> dict[str, tuple[bytes, dict[str, Any]]]:
    response = _gh_json(
        repository_root,
        (
            f"repos/{github_repository}/actions/runs/"
            f"{run['run_identifier']}/artifacts"
        ),
        fields={"per_page": str(_MAX_ACTIONS_RESULTS)},
    )
    raw_artifacts = response.get("artifacts")
    total = response.get("total_count")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or not isinstance(raw_artifacts, list)
        or total != len(raw_artifacts)
        or total > _MAX_ACTIONS_RESULTS
    ):
        raise GovernedResultCLIError(
            "GitHub Actions artifact listing is truncated or malformed"
        )
    artifacts_by_name: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in expected_entries
    }
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise GovernedResultCLIError(
                "GitHub Actions artifact listing is malformed"
            )
        name = raw.get("name")
        if name in artifacts_by_name:
            artifacts_by_name[name].append(raw)
    if any(len(matches) != 1 for matches in artifacts_by_name.values()):
        raise GovernedResultCLIError(
            "GitHub Actions run lacks the exact unique required artifacts"
        )

    result: dict[str, tuple[bytes, dict[str, Any]]] = {}
    seen_identifiers: set[int] = set()
    for name, entry in sorted(expected_entries.items()):
        listed = _normalize_actions_artifact(
            artifacts_by_name[name][0],
            expected_name=name,
            run=run,
        )
        identifier = listed["artifact_identifier"]
        if identifier in seen_identifiers:
            raise GovernedResultCLIError(
                "required GitHub Actions artifacts reuse one identifier"
            )
        seen_identifiers.add(identifier)
        exact = _normalize_actions_artifact(
            _gh_json(
                repository_root,
                (
                    f"repos/{github_repository}/actions/artifacts/"
                    f"{identifier}"
                ),
            ),
            expected_name=name,
            run=run,
        )
        if exact != listed:
            raise GovernedResultCLIError(
                "GitHub Actions artifact identity changed during authentication"
            )
        archive = _gh_archive_bytes(
            repository_root,
            (
                f"repos/{github_repository}/actions/artifacts/"
                f"{identifier}/zip"
            ),
        )
        archive_sha = _sha256_bytes(archive)
        if (
            len(archive) != exact["archive_size_bytes"]
            or archive_sha != exact["api_archive_sha256"]
        ):
            raise GovernedResultCLIError(
                "GitHub Actions artifact archive differs from its API digest"
            )
        raw_entry = _safe_zip_json_entry(
            archive,
            expected_entry=entry,
            label=name,
        )
        provenance = {
            **exact,
            "archive_sha256": archive_sha,
            "entry_sha256": _sha256_bytes(raw_entry),
        }
        result[name] = raw_entry, provenance
    return result


def _actions_provenance_sha256(
    *,
    run: Mapping[str, Any],
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    value = {
        "artifacts": [dict(item) for item in artifacts],
        "run": dict(run),
    }
    return _sha256_bytes(_canonical_bytes(value))


def _planned_scope_manifest_sha256(plan: Mapping[str, Any]) -> str:
    raw = (canonical_json(sorted(plan["changed_files"])) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _validated_published_result_payload(
    value: Any,
    *,
    experiment_id: str,
) -> dict[str, Any]:
    """Require the exact canonical metadata emitted by ``record_result``."""

    if not isinstance(value, Mapping) or set(value) != _RESULT_PAYLOAD_KEYS:
        raise GovernedResultCLIError(
            "published result payload is not an exact experiment result"
        )
    decision = value.get("decision")
    rationale = value.get("rationale")
    plan_record_hash = value.get("plan_record_hash")
    if (
        value.get("event") != "experiment_result"
        or value.get("experiment_id") != experiment_id
        or not isinstance(decision, str)
        or decision not in _EXPERIMENT_RESULT_DECISIONS
        or not isinstance(rationale, str)
        or not _SAFE_DIMENSION_RE.fullmatch(rationale)
        or not isinstance(plan_record_hash, str)
        or not _SHA256_RE.fullmatch(plan_record_hash)
    ):
        raise GovernedResultCLIError(
            "published result metadata is not canonical"
        )
    _require_nonidentifying_control_text("rationale", rationale)
    return dict(value)


def _revision_paths(
    repository_root: Path,
    revision: str,
) -> tuple[str, ...]:
    raw = _git(
        repository_root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        revision,
    )
    paths = tuple(sorted(path for path in raw.split("\0") if path))
    if not paths or any(
        path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in Path(path).parts)
        for path in paths
    ):
        raise GovernedResultCLIError(
            "candidate Git tree contains an invalid path"
        )
    return paths


def _revision_blob_sha256(
    repository_root: Path,
    revision: str,
    path: str,
) -> str:
    return _sha256_bytes(_git_blob(repository_root, revision, path))


def _runtime_producer_graph_sha256(
    repository_root: Path,
    revision: str,
) -> str:
    available = set(_revision_paths(repository_root, revision))
    required = {
        "solution.py",
        "run.sh",
        "requirements.lock",
        "scripts/run_docker_submission.py",
    }
    if not required.issubset(available):
        raise GovernedResultCLIError(
            "candidate runtime producer graph is incomplete"
        )
    paths = required | {
        path
        for path in available
        if path.startswith("mib_pipeline/")
        and "__pycache__" not in Path(path).parts
    }
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            _revision_blob_sha256(
                repository_root,
                revision,
                path,
            ).encode("ascii")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _grouped_graph_sha256_exact(
    repository_root: Path,
    revision: str,
) -> str:
    """Mirror the truth-safe grouped capture producer graph at a commit."""

    available = set(_revision_paths(repository_root, revision))
    required = {
        "requirements.lock",
        "Dockerfile",
        "run.sh",
        "devtools/policy_grouped_capture_contract.py",
    }
    if not required.issubset(available):
        raise GovernedResultCLIError(
            "grouped producer graph lacks a required committed file"
        )
    paths = {
        path
        for path in available
        if "__pycache__" not in Path(path).parts
        and (
            (
                path.startswith("mib_pipeline/")
                and path.endswith(".py")
            )
            or path.startswith("mib_pipeline/artifacts/")
        )
    }
    paths.update(required)
    if not paths:
        raise GovernedResultCLIError(
            "grouped producer graph contains no committed files"
        )
    entries = []
    for path in sorted(paths):
        raw = _git_blob(repository_root, revision, path)
        entries.append(
            {
                "path": path,
                "sha256": _sha256_bytes(raw),
                "size_bytes": len(raw),
            }
        )
    return _sha256_bytes(
        (canonical_json(entries) + "\n").encode("utf-8")
    )


def _validate_grouped_control_modules_unchanged(
    repository_root: Path,
    *,
    control_revision: str,
    candidate_revision: str,
) -> None:
    """Independently reject candidate substitutions beside protected truth."""

    for path in _GROUPED_TRUTH_ADJACENT_CONTROL_PATHS:
        try:
            control = _git_blob(
                repository_root,
                control_revision,
                path,
            )
            candidate = _git_blob(
                repository_root,
                candidate_revision,
                path,
            )
        except (OSError, RuntimeError) as exc:
            raise GovernedResultCLIError(
                f"cannot bind truth-adjacent control module: {path}"
            ) from exc
        if candidate != control:
            raise GovernedResultCLIError(
                f"candidate replaced truth-adjacent control module: {path}"
            )


def _validate_primary_committed_bindings(
    repository_root: Path,
    *,
    evidence: Mapping[str, Any],
    topology: GovernedExperimentTopology,
) -> None:
    expected_tools = {
        "capture_tool_sha256": (
            "devtools/policy_grouped_capture.py"
        ),
        "evidence_tool_sha256": (
            "devtools/grouped_policy_revalidation_evidence.py"
        ),
        "gate_tool_sha256": (
            "devtools/grouped_policy_revalidation_gate.py"
        ),
    }
    for field, path in expected_tools.items():
        _same(
            f"evidence.{field}",
            evidence[field],
            _revision_blob_sha256(
                repository_root,
                topology.candidate_revision_sha,
                path,
            ),
        )
    _validate_grouped_control_modules_unchanged(
        repository_root,
        control_revision=topology.control_revision_sha,
        candidate_revision=topology.candidate_revision_sha,
    )
    _same(
        "evidence.control_producer_graph_sha256",
        evidence["control_producer_graph_sha256"],
        _grouped_graph_sha256_exact(
            repository_root,
            topology.control_revision_sha,
        ),
    )
    _same(
        "evidence.candidate_producer_graph_sha256",
        evidence["candidate_producer_graph_sha256"],
        _grouped_graph_sha256_exact(
            repository_root,
            topology.candidate_revision_sha,
        ),
    )


def _load_exact_plan(
    experiment_store: Any,
    *,
    experiment_id: str,
) -> tuple[dict[str, Any], str]:
    records = experiment_store.verify()
    plans = [
        record
        for record in records
        if record["payload"].get("event") == "experiment_plan"
        and record["payload"].get("experiment_id") == experiment_id
    ]
    results = [
        record
        for record in records
        if record["payload"].get("event") == "experiment_result"
        and record["payload"].get("experiment_id") == experiment_id
    ]
    if len(plans) != 1:
        raise GovernedResultCLIError(
            "result publication requires exactly one prior experiment plan"
        )
    if results:
        raise GovernedResultCLIError(
            "this experiment already has an immutable result"
        )
    plan = plans[0]["payload"].get("plan")
    if not isinstance(plan, Mapping):
        raise GovernedResultCLIError(
            "experiment plan record is malformed"
        )
    return dict(plan), _digest(
        plans[0]["record_hash"],
        label="experiment plan record hash",
    )


def _promotion_population(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: plan[name]
        for name in (
            "evaluator_sha256",
            "expected_record_count",
            "input_tree_sha256",
            "runtime_contract_sha256",
            "split_manifest_sha256",
            "truth_sha256",
        )
    }


def _fold_coordinates() -> tuple[str, ...]:
    return tuple(
        f"repeat_{repeat}_fold_{fold}"
        for repeat in range(1, 4)
        for fold in range(1, 6)
    )


def _validate_fold_contract(
    evidence: Mapping[str, Any],
    *,
    expected_record_count: int,
) -> tuple[list[float], list[int], list[float]]:
    if (
        evidence.get("repeat_count") != 3
        or evidence.get("fold_count") != 5
        or evidence.get("evaluated_fold_count") != 15
    ):
        raise GovernedResultCLIError(
            "grouped evidence must contain exactly three repeats of five folds"
        )
    raw_deltas = evidence.get("fold_deltas")
    raw_weights = evidence.get("fold_weights")
    raw_repeats = evidence.get("repeat_scores")
    if (
        not isinstance(raw_deltas, list)
        or len(raw_deltas) != 15
        or not isinstance(raw_weights, list)
        or len(raw_weights) != 15
        or not isinstance(raw_repeats, list)
        or len(raw_repeats) != 3
    ):
        raise GovernedResultCLIError(
            "grouped evidence fold vectors have an invalid shape"
        )
    deltas = [
        _number(value, label="fold delta") for value in raw_deltas
    ]
    weights = [
        _count(value, label="fold weight", positive=True)
        for value in raw_weights
    ]
    repeats = [
        _number(value, label="repeat score") for value in raw_repeats
    ]
    fold_metrics = evidence.get("fold_metrics")
    coordinates = _fold_coordinates()
    if (
        not isinstance(fold_metrics, Mapping)
        or set(fold_metrics) != set(coordinates)
    ):
        raise GovernedResultCLIError(
            "fold_metrics must contain the exact 3x5 coordinates"
        )
    for index, coordinate in enumerate(coordinates):
        row = fold_metrics[coordinate]
        if not isinstance(row, Mapping) or set(row) != {
            "candidate_score",
            "control_score",
            "layout_group_count",
            "record_count",
            "score_delta",
        }:
            raise GovernedResultCLIError(
                "one fold metric has an invalid schema"
            )
        candidate = _number(
            row["candidate_score"], label=f"{coordinate}.candidate_score"
        )
        control = _number(
            row["control_score"], label=f"{coordinate}.control_score"
        )
        _count(
            row["layout_group_count"],
            label=f"{coordinate}.layout_group_count",
            positive=True,
        )
        if row["record_count"] != weights[index]:
            raise GovernedResultCLIError(
                "fold metric record_count disagrees with fold_weights"
            )
        if (
            abs((candidate - control) - deltas[index]) > 1e-9
            or abs(_number(row["score_delta"], label="fold score delta")
                   - deltas[index]) > 1e-9
        ):
            raise GovernedResultCLIError(
                "fold metric delta disagrees with its scores"
            )
    for repeat in range(3):
        offset = repeat * 5
        current_weights = weights[offset : offset + 5]
        current_deltas = deltas[offset : offset + 5]
        if sum(current_weights) != expected_record_count:
            raise GovernedResultCLIError(
                "each repeat must cover the exact planned population"
            )
        weighted = sum(
            delta * weight
            for delta, weight in zip(current_deltas, current_weights)
        ) / sum(current_weights)
        if abs(weighted - repeats[repeat]) > 1e-9:
            raise GovernedResultCLIError(
                "repeat_scores disagree with weighted fold deltas"
            )
    return deltas, weights, repeats


def _derive_gate_results(
    evidence: Mapping[str, Any],
    *,
    deltas: Sequence[float],
    weights: Sequence[int],
    repeat_scores: Sequence[float],
) -> dict[str, bool]:
    counts = evidence["counts"]
    components = evidence["score_components"]
    checks = evidence["checks"]
    regressions_clean = all(
        counts[name] == 0
        for name in (
            "regression_adversarial_failure_count",
            "regression_focused_failure_count",
            "regression_full_failure_count",
        )
    )
    leave_best: list[float] = []
    for repeat in range(3):
        offset = repeat * 5
        current_deltas = deltas[offset : offset + 5]
        current_weights = weights[offset : offset + 5]
        numerator = sum(
            delta * weight
            for delta, weight in zip(current_deltas, current_weights)
        )
        total = sum(current_weights)
        leave_best.append(
            min(
                (numerator - delta * weight) / (total - weight)
                for delta, weight in zip(
                    current_deltas, current_weights
                )
            )
        )
    complete_control = (
        evidence["record_count"] > 0
        and evidence["record_count"] == counts["record_count"]
    )
    complete_candidate = (
        counts["candidate_missing_record_count"] == 0
        and counts["candidate_invalid_record_count"] == 0
        and counts["candidate_duplicate_record_count"] == 0
        and counts["candidate_extra_record_count"] == 0
        and evidence["missing_records"] == 0
        and evidence["invalid_records"] == 0
        and evidence["duplicate_records"] == 0
        and evidence["extra_records"] == 0
    )
    return {
        "candidate_complete": complete_candidate,
        "capture_deterministic": bool(evidence["deterministic"]),
        "contract_probes_pass": all(checks.values()),
        "control_complete": complete_control,
        "decision_variable_nonvacuous": (
            counts["decision_or_confidence_change_count"] > 0
        ),
        "extraction_score_unchanged": (
            components["candidate_extraction_score"]
            == components["control_extraction_score"]
        ),
        "full_score_positive": evidence["candidate_score"]
        > evidence["control_score"],
        # Membership is proven when the artifact is built from its external
        # frozen manifest.  The result boundary additionally enforces the
        # exact fold coordinates, weights, and A→P→C tool/source bindings.
        "group_exclusive": bool(evidence["gate_results"]["group_exclusive"]),
        "guarded_initial_approval_nonvacuous": (
            counts["guarded_initial_approval_count"] > 0
        ),
        "legacy_forced_counter_fully_accounted": (
            counts["legacy_forced_approval_count"]
            == counts["guarded_initial_approval_count"]
            + counts["unguarded_initial_approval_count"]
            + counts["late_revalidation_approval_count"]
        ),
        "leave_best_fold_out_positive": all(
            value > 0 for value in leave_best
        ),
        "no_catastrophic_false_approvals": (
            evidence["catastrophic_false_approvals"] == 0
            and counts["new_catastrophic_false_approval_count"] == 0
        ),
        "no_late_revalidation_approval": (
            counts["late_revalidation_approval_count"] == 0
        ),
        "no_negative_folds": all(value >= 0 for value in deltas),
        "no_new_false_approvals": (
            counts["new_false_approval_count"] == 0
            and counts["candidate_false_approval_count"]
            <= evidence["false_approvals"]
        ),
        "no_unguarded_initial_approval": (
            counts["unguarded_initial_approval_count"] == 0
        ),
        "non_decision_fields_unchanged": (
            counts["non_decision_field_change_count"] == 0
        ),
        "paired_fold_members": bool(
            evidence["gate_results"]["paired_fold_members"]
        ),
        "population_bound": True,
        "public_exposed_evidence": True,
        "regression_suites_clean": regressions_clean,
        "repeat_weighted_deltas_positive": all(
            value > 0 for value in repeat_scores
        ),
        "runtime_contract_bound": True,
        "source_and_diff_bound": True,
        "split_deterministic": bool(
            evidence["gate_results"]["split_deterministic"]
        ),
    }


def _validate_primary_evidence(
    evidence: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_record_hash: str,
    topology: GovernedExperimentTopology,
    current_checkpoint_sha256: str,
    candidate_revision: str,
) -> dict[str, Any]:
    require_aggregate_only(evidence)
    if set(evidence) != _PRIMARY_ROOT_KEYS:
        raise GovernedResultCLIError(
            "aggregate evidence has an inexact root schema"
        )
    if (
        evidence["evaluation_mode"]
        != "public_grouped_robustness_not_unseen"
        or evidence["evidence_label"] != "aggregate_only"
    ):
        raise GovernedResultCLIError(
            "result evidence must be public grouped robustness evidence"
        )
    if (
        topology.base_revision_sha != plan["parent_commit_sha"]
        or topology.control_revision_sha
        != evidence["control_revision_sha"]
        or topology.candidate_revision_sha != candidate_revision
        or topology.experiment_plan_record_sha256 != plan_record_hash
        or topology.prereg_checkpoint_sha256
        != current_checkpoint_sha256
    ):
        raise GovernedResultCLIError(
            "derived A→P→C topology disagrees with the plan or authority"
        )
    bindings = {
        "base_revision_sha": topology.base_revision_sha,
        "control_revision_sha": topology.control_revision_sha,
        "control_source_revision_sha": topology.control_revision_sha,
        "candidate_revision_sha": candidate_revision,
        "candidate_source_revision_sha": candidate_revision,
        "experiment_plan_sha256": topology.experiment_plan_sha256,
        "experiment_plan_record_sha256": plan_record_hash,
        "prereg_checkpoint_sha256": current_checkpoint_sha256,
        "prereg_governance_diff_sha256": (
            topology.prereg_governance_diff_sha256
        ),
        "candidate_diff_manifest_sha256": (
            topology.candidate_source_diff_sha256
        ),
        "candidate_source_diff_sha256": (
            topology.candidate_source_diff_sha256
        ),
        "planned_scope_manifest_sha256": (
            topology.planned_scope_manifest_sha256
        ),
        "hypothesis_sha256": plan["hypothesis_sha256"],
        "primary_variable_sha256": plan["primary_variable_sha256"],
        "runtime_contract_sha256": plan["runtime_contract_sha256"],
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "input_tree_sha256": plan["input_tree_sha256"],
        "truth_sha256": plan["truth_sha256"],
        "evaluator_sha256": plan["evaluator_sha256"],
        "record_count": plan["expected_record_count"],
    }
    for name, expected in bindings.items():
        _same(f"evidence.{name}", evidence[name], expected)
    _same(
        "planned scope digest",
        evidence["planned_scope_manifest_sha256"],
        _planned_scope_manifest_sha256(plan),
    )
    for name in (
        "candidate_capture_set_sha256",
        "candidate_producer_graph_sha256",
        "capture_tool_sha256",
        "control_capture_set_sha256",
        "control_producer_graph_sha256",
        "evidence_tool_sha256",
        "gate_tool_sha256",
    ):
        _digest(evidence[name], label=f"evidence.{name}")

    counts = evidence["counts"]
    if not isinstance(counts, Mapping) or set(counts) != _COUNT_KEYS:
        raise GovernedResultCLIError(
            "aggregate evidence counts have an inexact schema"
        )
    for name, value in counts.items():
        _count(value, label=f"counts.{name}")
    _same("counts.record_count", counts["record_count"], plan["expected_record_count"])
    _same(
        "counts.layout_group_count",
        counts["layout_group_count"],
        evidence["layout_group_count"],
    )
    for name in (
        "catastrophic_false_approvals",
        "duplicate_records",
        "extra_records",
        "false_approvals",
        "hard_gate_failure_count",
        "invalid_records",
        "layout_group_count",
        "missing_records",
    ):
        _count(evidence[name], label=f"evidence.{name}")
    _same(
        "candidate catastrophic count",
        evidence["catastrophic_false_approvals"],
        counts["candidate_catastrophic_false_approval_count"],
    )
    for root_name, count_name in (
        ("duplicate_records", "candidate_duplicate_record_count"),
        ("extra_records", "candidate_extra_record_count"),
        ("false_approvals", "candidate_false_approval_count"),
        ("invalid_records", "candidate_invalid_record_count"),
        ("missing_records", "candidate_missing_record_count"),
    ):
        _same(root_name, evidence[root_name], counts[count_name])

    components = evidence["score_components"]
    if (
        not isinstance(components, Mapping)
        or set(components) != _SCORE_COMPONENT_KEYS
    ):
        raise GovernedResultCLIError(
            "score_components have an inexact schema"
        )
    for name, value in components.items():
        _number(value, label=f"score_components.{name}")
    control_total = (
        float(components["control_extraction_score"])
        + float(components["control_classification_score"])
        + float(components["control_calibration_score"])
        - float(components["control_missing_penalty"])
    )
    candidate_total = (
        float(components["candidate_extraction_score"])
        + float(components["candidate_classification_score"])
        + float(components["candidate_calibration_score"])
        - float(components["candidate_missing_penalty"])
    )
    if (
        abs(control_total - float(components["control_total_score"])) > 1e-9
        or abs(candidate_total - float(components["candidate_total_score"]))
        > 1e-9
        or abs(float(evidence["control_score"]) - control_total) > 1e-9
        or abs(float(evidence["candidate_score"]) - candidate_total) > 1e-9
        or abs(
            float(evidence["score_delta"])
            - (candidate_total - control_total)
        )
        > 1e-9
    ):
        raise GovernedResultCLIError(
            "official score aggregates are internally inconsistent"
        )

    confusion = evidence["confusion_counts"]
    if (
        not isinstance(confusion, Mapping)
        or set(confusion) != _CONFUSION_KEYS
    ):
        raise GovernedResultCLIError(
            "confusion_counts have an inexact schema"
        )
    for name, value in confusion.items():
        _count(value, label=f"confusion_counts.{name}")

    checks = evidence["checks"]
    if (
        not isinstance(checks, Mapping)
        or not checks
        or any(
            not isinstance(name, str) or not isinstance(value, bool)
            for name, value in checks.items()
        )
        or "runtime_limits_satisfied" not in checks
    ):
        raise GovernedResultCLIError(
            "capture contract checks are invalid"
        )
    deltas, weights, repeat_scores = _validate_fold_contract(
        evidence,
        expected_record_count=plan["expected_record_count"],
    )
    gates = evidence["gate_results"]
    if (
        not isinstance(gates, Mapping)
        or set(gates) != _GROUPED_GATE_KEYS
        or any(not isinstance(value, bool) for value in gates.values())
    ):
        raise GovernedResultCLIError(
            "gate_results have an inexact schema"
        )
    derived = _derive_gate_results(
        evidence,
        deltas=deltas,
        weights=weights,
        repeat_scores=repeat_scores,
    )
    for name, expected in derived.items():
        _same(f"gate_results.{name}", gates[name], expected)
    failure_count = sum(not value for value in gates.values())
    _same(
        "hard_gate_failure_count",
        evidence["hard_gate_failure_count"],
        failure_count,
    )
    expected_status = "passed" if failure_count == 0 else "blocked"
    _same("evidence status", evidence["status"], expected_status)
    _same(
        "fold_consistent",
        evidence["fold_consistent"],
        gates["no_negative_folds"]
        and gates["leave_best_fold_out_positive"],
    )
    _same(
        "deterministic",
        evidence["deterministic"],
        gates["capture_deterministic"],
    )
    return dict(evidence)


def _validate_runtime_comparison(
    repository_root: Path,
    *,
    candidate_revision: str,
    compared: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        compared.get("schema_version") != WO20_OUTPUT_SCHEMA
        or compared.get("status") not in {"PASS", "PASS_WITH_WARNINGS"}
        or compared.get("blocking_reasons") != []
        or compared.get("aggregate_only") is not True
    ):
        raise GovernedResultCLIError(
            "trusted exact-source runtime evidence did not pass"
        )
    source = compared["source_binding"]
    runtime = compared["runtime"]
    determinism = compared["determinism"]
    images = compared["images"]
    models = compared["installed_model_artifacts"]
    if (
        source.get("git_revision") != candidate_revision
        or determinism.get("byte_identical") is not True
        or determinism.get("coverage_identical") is not True
        or runtime.get("all_captures_within_official_runtime_limit")
        is not True
        or runtime.get("all_captures_within_official_memory_limit")
        is not True
        or images.get("source_binding_verified") is not True
    ):
        raise GovernedResultCLIError(
            "trusted runtime evidence is not exact-source and deterministic"
        )
    source_paths = {
        "dockerfile_sha256": "Dockerfile",
        "requirements_lock_sha256": "requirements.lock",
        "run_sh_sha256": "run.sh",
        "solution_sha256": "solution.py",
        "harness_sha256": "scripts/run_docker_submission.py",
    }
    for field, path in source_paths.items():
        _same(
            f"runtime source_binding.{field}",
            source.get(field),
            _revision_blob_sha256(
                repository_root,
                candidate_revision,
                path,
            ),
        )
    _same(
        "runtime source_binding.producer_graph_sha256",
        source.get("producer_graph_sha256"),
        _runtime_producer_graph_sha256(
            repository_root,
            candidate_revision,
        ),
    )
    canonical = _canonical_bytes(compared)
    return {
        "candidate_image_bytes": max(images["size_bytes"]),
        "candidate_max_model_artifact_bytes": models[
            "maximum_artifact_bytes"
        ],
        "candidate_model_bytes": models["total_bytes"],
        "deterministic": True,
        "output_bytes": determinism["output_bytes"],
        "prediction_output_sha256": _digest(
            determinism["output_sha256"],
            label="runtime prediction output digest",
        ),
        "peak_container_memory_bytes": runtime[
            "peak_container_memory_bytes"
        ],
        "peak_rss_bytes": runtime["peak_process_tree_rss_bytes"],
        # Four CPUs provide a conservative, source-bound upper bound even
        # when the Docker envelope reports elapsed rather than CPU time.
        "process_cpu_seconds": 4.0 * runtime["max_elapsed_seconds"],
        "runtime_evidence_sha256": _sha256_bytes(canonical),
        "runtime_limits_verified": True,
        "runtime_seconds": runtime["max_elapsed_seconds"],
        # The enforced 2 GiB tmpfs cap proves this hard upper bound.
        "tmp_bytes": 2 * GIB,
    }


def _trusted_runtime(
    repository_root: Path,
    *,
    candidate_revision: str,
    capture_paths: Sequence[Path | str],
) -> dict[str, Any]:
    """Validate optional local captures for non-promotional decisions only."""

    if len(capture_paths) != 2:
        raise GovernedResultCLIError(
            "local runtime validation requires exactly two captures"
        )
    compared = dict(
        compare_capture_evidence(
            capture_paths,
            expected_source_revision=candidate_revision,
            expected_pdf_count=5000,
        )
    )
    return _validate_runtime_comparison(
        repository_root,
        candidate_revision=candidate_revision,
        compared=compared,
    )


def _download_authenticated_grouped(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    candidate_revision: str,
    topology: GovernedExperimentTopology,
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    run = _authenticated_workflow_run(
        repository_root,
        github_repository=github_repository,
        branch=branch,
        candidate_revision=candidate_revision,
        workflow_path=_WO17_WORKFLOW_PATH,
    )
    artifact_name = (
        f"{_WO17_ARTIFACT_PREFIX}-{candidate_revision}-"
        f"{run['run_attempt']}"
    )
    entries = _authenticated_artifact_entries(
        repository_root,
        github_repository=github_repository,
        run=run,
        expected_entries={
            artifact_name: _WO17_ARTIFACT_ENTRY,
        },
    )
    if _authenticated_workflow_run(
        repository_root,
        github_repository=github_repository,
        branch=branch,
        candidate_revision=candidate_revision,
        workflow_path=_WO17_WORKFLOW_PATH,
    ) != run:
        raise GovernedResultCLIError(
            "grouped workflow run changed during artifact authentication"
        )
    artifact_raw, artifact = entries[artifact_name]
    decoded = _decode_canonical_object(
        artifact_raw,
        label="authenticated grouped aggregate",
    )
    require_aggregate_only(decoded)
    if set(decoded) != _PRIMARY_ROOT_KEYS:
        raise GovernedResultCLIError(
            "authenticated grouped aggregate has an inexact root schema"
        )
    _validate_primary_committed_bindings(
        repository_root,
        evidence=decoded,
        topology=topology,
    )
    provenance = {
        "aggregate_api_archive_sha256": artifact[
            "api_archive_sha256"
        ],
        "aggregate_archive_sha256": artifact["archive_sha256"],
        "aggregate_entry_sha256": artifact["entry_sha256"],
        "artifact_identifier": artifact["artifact_identifier"],
        "head_revision_sha": run["head_revision_sha"],
        "provenance_sha256": _actions_provenance_sha256(
            run=run,
            artifacts=(artifact,),
        ),
        "repository_identifier": run["repository_identifier"],
        "repository_sha256": _sha256_bytes(
            github_repository.casefold().encode("utf-8")
        ),
        "run_attempt": run["run_attempt"],
        "run_identifier": run["run_identifier"],
        "workflow_identifier": run["workflow_identifier"],
        "workflow_sha256": _revision_blob_sha256(
            repository_root,
            candidate_revision,
            _WO17_WORKFLOW_PATH,
        ),
    }
    return decoded, artifact_raw, provenance


def _authenticated_grouped_provenance(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    candidate_revision: str,
    primary: Mapping[str, Any],
    primary_raw: bytes,
    topology: GovernedExperimentTopology,
) -> dict[str, Any]:
    decoded, artifact_raw, provenance = (
        _download_authenticated_grouped(
            repository_root,
            github_repository=github_repository,
            branch=branch,
            candidate_revision=candidate_revision,
            topology=topology,
        )
    )
    if artifact_raw != primary_raw or decoded != dict(primary):
        raise GovernedResultCLIError(
            "supplied grouped evidence differs from the authenticated "
            "GitHub Actions artifact"
        )
    return provenance


def _authenticated_runtime(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    candidate_revision: str,
) -> dict[str, Any]:
    run = _authenticated_workflow_run(
        repository_root,
        github_repository=github_repository,
        branch=branch,
        candidate_revision=candidate_revision,
        workflow_path=_WO20_WORKFLOW_PATH,
    )
    names = {
        (
            f"wo20-capture-a-{candidate_revision}-"
            f"{run['run_attempt']}"
        ): "evidence-a.json",
        (
            f"wo20-capture-b-{candidate_revision}-"
            f"{run['run_attempt']}"
        ): "evidence-b.json",
    }
    entries = _authenticated_artifact_entries(
        repository_root,
        github_repository=github_repository,
        run=run,
        expected_entries=names,
    )
    if _authenticated_workflow_run(
        repository_root,
        github_repository=github_repository,
        branch=branch,
        candidate_revision=candidate_revision,
        workflow_path=_WO20_WORKFLOW_PATH,
    ) != run:
        raise GovernedResultCLIError(
            "runtime workflow run changed during artifact authentication"
        )
    ordered = [entries[name] for name in sorted(names)]
    with tempfile.TemporaryDirectory(
        prefix="mib-governed-wo20-"
    ) as temporary_name:
        paths: list[Path] = []
        for index, (raw, _artifact) in enumerate(ordered, start=1):
            path = Path(temporary_name) / f"capture-{index}.json"
            path.write_bytes(raw)
            paths.append(path)
        compared = dict(
            compare_capture_evidence(
                paths,
                expected_source_revision=candidate_revision,
                expected_pdf_count=5000,
            )
        )
    runtime = _validate_runtime_comparison(
        repository_root,
        candidate_revision=candidate_revision,
        compared=compared,
    )
    artifacts = [artifact for _raw, artifact in ordered]
    runtime.update(
        {
            "actions_authenticated": True,
            "capture_a_api_archive_sha256": artifacts[0][
                "api_archive_sha256"
            ],
            "capture_a_archive_sha256": artifacts[0][
                "archive_sha256"
            ],
            "capture_a_entry_sha256": artifacts[0]["entry_sha256"],
            "capture_a_identifier": artifacts[0][
                "artifact_identifier"
            ],
            "capture_b_api_archive_sha256": artifacts[1][
                "api_archive_sha256"
            ],
            "capture_b_archive_sha256": artifacts[1][
                "archive_sha256"
            ],
            "capture_b_entry_sha256": artifacts[1]["entry_sha256"],
            "capture_b_identifier": artifacts[1][
                "artifact_identifier"
            ],
            "head_revision_sha": run["head_revision_sha"],
            "provenance_sha256": _actions_provenance_sha256(
                run=run,
                artifacts=artifacts,
            ),
            "repository_identifier": run["repository_identifier"],
            "repository_sha256": _sha256_bytes(
                github_repository.casefold().encode("utf-8")
            ),
            "run_attempt": run["run_attempt"],
            "run_identifier": run["run_identifier"],
            "workflow_identifier": run["workflow_identifier"],
            "workflow_sha256": _revision_blob_sha256(
                repository_root,
                candidate_revision,
                _WO20_WORKFLOW_PATH,
            ),
        }
    )
    return runtime


def _runtime_defaults(primary_evidence_sha256: str) -> dict[str, Any]:
    return {
        "actions_authenticated": False,
        "candidate_image_bytes": 0,
        "candidate_max_model_artifact_bytes": 0,
        "candidate_model_bytes": 0,
        "deterministic": False,
        "output_bytes": 0,
        "peak_container_memory_bytes": 0,
        "peak_rss_bytes": 0,
        "prediction_output_sha256": primary_evidence_sha256,
        "process_cpu_seconds": 0.0,
        "runtime_evidence_sha256": primary_evidence_sha256,
        "runtime_limits_verified": False,
        "runtime_seconds": 0.0,
        "tmp_bytes": 0,
    }


def _validate_decision(
    primary: Mapping[str, Any],
    *,
    decision: str,
    grouped_actions_authenticated: bool,
    runtime_actions_authenticated: bool,
    runtime_leakage_clean: bool,
) -> None:
    if decision != "adopt":
        return
    if primary["status"] != "passed" or not all(
        primary["gate_results"].values()
    ):
        raise GovernedResultCLIError(
            "a candidate with failed grouped hard gates cannot be adopted"
        )
    if (
        primary["score_components"]["candidate_missing_penalty"] != 0
    ):
        raise GovernedResultCLIError(
            "adoption requires a zero candidate missing-record penalty"
        )
    if not grouped_actions_authenticated:
        raise GovernedResultCLIError(
            "adoption requires authenticated grouped GitHub Actions evidence"
        )
    if not runtime_actions_authenticated:
        raise GovernedResultCLIError(
            "adoption requires authenticated exact-source GitHub Actions "
            "runtime evidence"
        )
    if not runtime_leakage_clean:
        raise GovernedResultCLIError(
            "adoption requires a clean candidate runtime leakage scan"
        )


def _ledger_evidence(
    primary: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    grouped_provenance: Mapping[str, Any] | None = None,
    runtime_leakage_clean: bool,
) -> dict[str, Any]:
    components = primary["score_components"]
    counts = primary["counts"]
    gross_total = (
        components["candidate_extraction_score"]
        + components["candidate_classification_score"]
        + components["candidate_calibration_score"]
    )
    official_total = components["candidate_total_score"]
    grouped_authenticated = grouped_provenance is not None
    runtime_authenticated = runtime.get("actions_authenticated") is True
    evidence: dict[str, Any] = {
        "baseline_artifact_sha256": primary[
            "control_capture_set_sha256"
        ],
        "candidate_artifact_sha256": primary[
            "candidate_capture_set_sha256"
        ],
        "checks": {
            "decision_freeze_verified": True,
            "deterministic": (
                primary["deterministic"] and runtime["deterministic"]
            ),
            "fold_consistent": primary["fold_consistent"],
            "grouped_actions_authenticated": grouped_authenticated,
            "grouped_tool_blobs_verified": grouped_authenticated,
            "runtime_actions_authenticated": runtime_authenticated,
            "runtime_limits_verified": runtime[
                "runtime_limits_verified"
            ],
            "runtime_leakage_clean": runtime_leakage_clean,
        },
        "confusion_counts": dict(primary["confusion_counts"]),
        "evaluator_sha256": plan["evaluator_sha256"],
        "expected_record_count": plan["expected_record_count"],
        "field_metrics": {
            "policy_activity": {
                "count": counts[
                    "decision_or_confidence_change_count"
                ],
                "score_delta": primary["score_delta"],
            }
        },
        "fold_count": primary["fold_count"],
        "fold_deltas": list(primary["fold_deltas"]),
        "fold_weights": list(primary["fold_weights"]),
        "input_tree_sha256": plan["input_tree_sha256"],
        "metrics": {
            "calibration_score": components[
                "candidate_calibration_score"
            ],
            "candidate_image_bytes": runtime[
                "candidate_image_bytes"
            ],
            "candidate_max_model_artifact_bytes": runtime[
                "candidate_max_model_artifact_bytes"
            ],
            "candidate_model_bytes": runtime[
                "candidate_model_bytes"
            ],
            "catastrophic_false_approvals": primary[
                "catastrophic_false_approvals"
            ],
            "classification_score": components[
                "candidate_classification_score"
            ],
            "extraction_score": components[
                "candidate_extraction_score"
            ],
            "invalid_records": primary["invalid_records"],
            "missing_penalty": components[
                "candidate_missing_penalty"
            ],
            "missing_records": primary["missing_records"],
            "official_total_score": official_total,
            "output_bytes": runtime["output_bytes"],
            "peak_container_memory_bytes": runtime[
                "peak_container_memory_bytes"
            ],
            "peak_rss_bytes": runtime["peak_rss_bytes"],
            "process_cpu_seconds": runtime["process_cpu_seconds"],
            "record_count": primary["record_count"],
            "runtime_seconds": runtime["runtime_seconds"],
            "tmp_bytes": runtime["tmp_bytes"],
            "total_score": gross_total,
        },
        "primary_evidence_sha256": _sha256_bytes(
            _canonical_bytes(primary)
        ),
        "regression_counts": {
            "adversarial": counts[
                "regression_adversarial_failure_count"
            ],
            "golden": (
                counts["regression_focused_failure_count"]
                + counts["regression_full_failure_count"]
            ),
        },
        "repeat_count": primary["repeat_count"],
        "repeat_scores": list(primary["repeat_scores"]),
        "runtime_contract_sha256": plan["runtime_contract_sha256"],
        "runtime_evidence_sha256": runtime[
            "runtime_evidence_sha256"
        ],
        "runtime_prediction_output_sha256": runtime[
            "prediction_output_sha256"
        ]
        if "prediction_output_sha256" in runtime
        else runtime["runtime_evidence_sha256"],
        "split_manifest_sha256": plan["split_manifest_sha256"],
        "truth_sha256": plan["truth_sha256"],
    }
    if grouped_provenance is not None:
        evidence.update(
            {
                "wo17_aggregate_api_archive_sha256": (
                    grouped_provenance[
                        "aggregate_api_archive_sha256"
                    ]
                ),
                "wo17_aggregate_archive_sha256": grouped_provenance[
                    "aggregate_archive_sha256"
                ],
                "wo17_aggregate_entry_sha256": grouped_provenance[
                    "aggregate_entry_sha256"
                ],
                "wo17_actions_provenance_sha256": grouped_provenance[
                    "provenance_sha256"
                ],
                "wo17_head_sha": grouped_provenance[
                    "head_revision_sha"
                ],
                "wo17_repository_sha256": grouped_provenance[
                    "repository_sha256"
                ],
                "wo17_workflow_sha256": grouped_provenance[
                    "workflow_sha256"
                ],
            }
        )
        evidence["metrics"].update(
            {
                "wo17_actions_artifact": grouped_provenance[
                    "artifact_identifier"
                ],
                "wo17_actions_repository": grouped_provenance[
                    "repository_identifier"
                ],
                "wo17_actions_run": grouped_provenance[
                    "run_identifier"
                ],
                "wo17_actions_run_attempt": grouped_provenance[
                    "run_attempt"
                ],
                "wo17_actions_workflow": grouped_provenance[
                    "workflow_identifier"
                ],
            }
        )
    if runtime_authenticated:
        evidence.update(
            {
                "wo20_actions_provenance_sha256": runtime[
                    "provenance_sha256"
                ],
                "wo20_capture_a_api_archive_sha256": runtime[
                    "capture_a_api_archive_sha256"
                ],
                "wo20_capture_a_archive_sha256": runtime[
                    "capture_a_archive_sha256"
                ],
                "wo20_capture_a_entry_sha256": runtime[
                    "capture_a_entry_sha256"
                ],
                "wo20_capture_b_api_archive_sha256": runtime[
                    "capture_b_api_archive_sha256"
                ],
                "wo20_capture_b_archive_sha256": runtime[
                    "capture_b_archive_sha256"
                ],
                "wo20_capture_b_entry_sha256": runtime[
                    "capture_b_entry_sha256"
                ],
                "wo20_head_sha": runtime["head_revision_sha"],
                "wo20_repository_sha256": runtime[
                    "repository_sha256"
                ],
                "wo20_workflow_sha256": runtime["workflow_sha256"],
            }
        )
        evidence["metrics"].update(
            {
                "wo20_actions_capture_a_artifact": runtime[
                    "capture_a_identifier"
                ],
                "wo20_actions_capture_b_artifact": runtime[
                    "capture_b_identifier"
                ],
                "wo20_actions_repository": runtime[
                    "repository_identifier"
                ],
                "wo20_actions_run": runtime["run_identifier"],
                "wo20_actions_run_attempt": runtime["run_attempt"],
                "wo20_actions_workflow": runtime[
                    "workflow_identifier"
                ],
            }
        )
    require_aggregate_only(evidence)
    return evidence


def _verify_adopted_result_provenance(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    candidate_revision: str,
    plan: Mapping[str, Any],
    plan_record_hash: str,
    topology: GovernedExperimentTopology,
    recorded_evidence: Mapping[str, Any],
) -> None:
    primary, _primary_raw, grouped_provenance = (
        _download_authenticated_grouped(
            repository_root,
            github_repository=github_repository,
            branch=branch,
            candidate_revision=candidate_revision,
            topology=topology,
        )
    )
    primary = _validate_primary_evidence(
        primary,
        plan=plan,
        plan_record_hash=plan_record_hash,
        topology=topology,
        current_checkpoint_sha256=(
            topology.prereg_checkpoint_sha256
        ),
        candidate_revision=candidate_revision,
    )
    runtime = _authenticated_runtime(
        repository_root,
        github_repository=github_repository,
        branch=branch,
        candidate_revision=candidate_revision,
    )
    leakage_clean = _runtime_leakage_finding_count(repository_root) == 0
    _validate_decision(
        primary,
        decision="adopt",
        grouped_actions_authenticated=True,
        runtime_actions_authenticated=True,
        runtime_leakage_clean=leakage_clean,
    )
    expected = _ledger_evidence(
        primary,
        plan=plan,
        runtime=runtime,
        grouped_provenance=grouped_provenance,
        runtime_leakage_clean=leakage_clean,
    )
    if dict(recorded_evidence) != expected:
        raise GovernedResultCLIError(
            "published adoption evidence differs from authenticated "
            "GitHub Actions provenance"
        )


def _validate_all_bindings(
    repository_root: Path,
    *,
    stores: Mapping[str, Any],
    current: ProgramIntegrityCheckpoint,
    current_pointer: Mapping[str, Any],
    revision: str,
    experiment_id: str,
    evidence: Mapping[str, Any],
) -> tuple[dict[str, Any], str, GovernedExperimentTopology, dict[str, Any]]:
    plan, plan_record_hash = _load_exact_plan(
        stores["experiment_ledger"],
        experiment_id=experiment_id,
    )
    plan_sha = _sha256_bytes(_canonical_bytes(plan))
    control = _commit(
        evidence.get("control_revision_sha"),
        label="evidence.control_revision_sha",
    )
    topology = verify_governed_experiment_topology(
        repository_root,
        plan=plan,
        experiment_plan_sha256=plan_sha,
        control_revision_sha=control,
        candidate_revision_sha=revision,
    )
    if topology.experiment_plan_record_sha256 != plan_record_hash:
        raise GovernedResultCLIError(
            "topology plan record does not match the current ledger"
        )
    if topology.prereg_checkpoint_sha256 != current.expected_sha256:
        raise GovernedResultCLIError(
            "current authority is not the preregistration checkpoint"
        )
    if current_pointer["checkpoint_sha256"] != current.expected_sha256:
        raise GovernedResultCLIError(
            "current pointer and checkpoint authority disagree"
        )
    verified_checkpoint = current.verify()
    if verified_checkpoint.get("promotion_population") != _promotion_population(
        plan
    ):
        raise GovernedResultCLIError(
            "plan population differs from the immutable promotion population"
        )
    normalized = _validate_primary_evidence(
        evidence,
        plan=plan,
        plan_record_hash=plan_record_hash,
        topology=topology,
        current_checkpoint_sha256=current.expected_sha256,
        candidate_revision=revision,
    )
    return plan, plan_record_hash, topology, normalized


def verify_candidate_authority(
    *,
    repository_root: Path | str,
    github_repository: str,
    branch: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Prove that capture would run on the published candidate C.

    The current checkpoint is intentionally still the preregistration
    checkpoint published in P.  The authenticated branch HEAD must be the clean
    source-only child C, and A→P→C must exactly match the immutable plan.
    """

    root = Path(repository_root).resolve()
    _program, _pointer, _directory, stores = _program_paths(root)
    current, pointer, candidate = GitHubCheckpointAuthority(
        repository_root=root,
        github_repository=github_repository,
        branch=branch,
    ).resolve(stores=stores)
    plan, plan_record_hash = _load_exact_plan(
        stores["experiment_ledger"],
        experiment_id=experiment_id,
    )
    control = _direct_parent(root, candidate)
    topology = verify_governed_experiment_topology(
        root,
        plan=plan,
        experiment_plan_sha256=_sha256_bytes(_canonical_bytes(plan)),
        control_revision_sha=control,
        candidate_revision_sha=candidate,
    )
    if (
        topology.experiment_plan_record_sha256 != plan_record_hash
        or topology.prereg_checkpoint_sha256
        != current.expected_sha256
        or pointer["checkpoint_sha256"] != current.expected_sha256
        or current.verify().get("promotion_population")
        != _promotion_population(plan)
    ):
        raise GovernedResultCLIError(
            "candidate authority is not bound to its preregistration"
        )
    if _remote_head(
        root,
        github_repository=github_repository,
        branch=branch,
    ) != candidate:
        raise GovernedResultCLIError(
            "authenticated GitHub branch moved during candidate verification"
        )
    return {
        "base_revision_sha": topology.base_revision_sha,
        "branch": branch,
        "candidate_revision_sha": candidate,
        "checkpoint_sha256": current.expected_sha256,
        "control_revision_sha": control,
        "experiment_id": experiment_id,
        "github_repository": github_repository,
        "plan_record_hash": plan_record_hash,
        "status": "authenticated_candidate_authority_verified",
    }


def verify_result_authority(
    *,
    repository_root: Path | str,
    github_repository: str,
    branch: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Verify one published C→R result-only governance transition."""

    root = Path(repository_root).resolve()
    _program, _pointer_path, _directory, stores = _program_paths(root)
    current, pointer, result_revision = GitHubCheckpointAuthority(
        repository_root=root,
        github_repository=github_repository,
        branch=branch,
    ).resolve(stores=stores)
    candidate = _direct_parent(root, result_revision)
    control = _direct_parent(root, candidate)
    records = stores["experiment_ledger"].verify()
    plans = [
        record
        for record in records
        if record["payload"].get("event") == "experiment_plan"
        and record["payload"].get("experiment_id") == experiment_id
    ]
    results = [
        record
        for record in records
        if record["payload"].get("event") == "experiment_result"
        and record["payload"].get("experiment_id") == experiment_id
    ]
    if len(plans) != 1 or len(results) != 1 or records[-1] != results[0]:
        raise GovernedResultCLIError(
            "published result must be the sole final event for its plan"
        )
    plan = plans[0]["payload"]["plan"]
    topology = verify_governed_experiment_topology(
        root,
        plan=plan,
        experiment_plan_sha256=_sha256_bytes(_canonical_bytes(plan)),
        control_revision_sha=control,
        candidate_revision_sha=candidate,
    )
    if (
        topology.experiment_plan_record_sha256
        != plans[0]["record_hash"]
    ):
        raise GovernedResultCLIError(
            "published result topology binds a different plan record"
        )
    result_record = results[0]
    payload = _validated_published_result_payload(
        result_record["payload"],
        experiment_id=experiment_id,
    )
    if payload["plan_record_hash"] != plans[0]["record_hash"]:
        raise GovernedResultCLIError(
            "published result does not bind the exact prior plan record"
        )
    evidence = _normalize_experiment_result_evidence(
        payload["evidence"],
        decision=payload["decision"],
        evidence_label=plan["evidence_label"],
    )
    for name in (
        "evaluator_sha256",
        "input_tree_sha256",
        "runtime_contract_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    ):
        if evidence[name] != plan[name]:
            raise GovernedResultCLIError(
                f"published result {name} differs from its plan"
            )
    if evidence["expected_record_count"] != plan["expected_record_count"]:
        raise GovernedResultCLIError(
            "published result population count differs from its plan"
        )

    transition, transition_sha = _raw_diff(
        root, candidate, result_revision
    )
    checkpoint_path = str(pointer["checkpoint_path"])
    expected_paths = {
        "evaluation/program/experiment_ledger.jsonl",
        "evaluation/program/current_checkpoint.json",
        checkpoint_path,
    }
    if {record.path for record in transition} != expected_paths:
        raise GovernedResultCLIError(
            "result publication is not an exact governance-only transition"
        )
    by_path = {record.path: record for record in transition}
    for path in (
        "evaluation/program/experiment_ledger.jsonl",
        "evaluation/program/current_checkpoint.json",
    ):
        record = by_path[path]
        if (
            record.status != "M"
            or record.old_mode != "100644"
            or record.new_mode != "100644"
        ):
            raise GovernedResultCLIError(
                "result ledger and pointer must be regular-file modifications"
            )
    added = by_path[checkpoint_path]
    if (
        added.status != "A"
        or added.old_mode != "000000"
        or added.new_mode != "100644"
    ):
        raise GovernedResultCLIError(
            "result checkpoint must be one newly added regular file"
        )
    parent_pointer = _checkpoint_pointer(
        _git_blob(
            root,
            candidate,
            "evaluation/program/current_checkpoint.json",
        ),
        label="candidate checkpoint pointer",
    )
    if (
        pointer["previous_checkpoint_sha256"]
        != parent_pointer["checkpoint_sha256"]
        or parent_pointer["checkpoint_sha256"]
        != topology.prereg_checkpoint_sha256
    ):
        raise GovernedResultCLIError(
            "result pointer does not extend the preregistration checkpoint"
        )
    parent_checkpoint, _ = _verified_checkpoint(
        root, candidate, parent_pointer
    )
    result_checkpoint, _ = _verified_checkpoint(
        root, result_revision, pointer
    )
    for name in (
        "baseline_manifest_sha256",
        "promotion_population",
        "runtime_leakage_finding_count",
    ):
        if result_checkpoint.get(name) != parent_checkpoint.get(name):
            raise GovernedResultCLIError(
                "result publication rewrote immutable checkpoint state"
            )
    for name, before in parent_checkpoint["stores"].items():
        after = result_checkpoint["stores"][name]
        if name == "experiment_ledger":
            if (
                after["path"] != before["path"]
                or after["expected_length"]
                != before["expected_length"] + 1
                or after["expected_head"] != result_record["record_hash"]
            ):
                raise GovernedResultCLIError(
                    "result checkpoint is not exactly one result append"
                )
        elif after != before:
            raise GovernedResultCLIError(
                "result publication changed an unrelated ledger anchor"
            )
    parent_ledger = _git_blob(
        root, candidate, "evaluation/program/experiment_ledger.jsonl"
    )
    result_ledger = _git_blob(
        root,
        result_revision,
        "evaluation/program/experiment_ledger.jsonl",
    )
    if (
        not result_ledger.startswith(parent_ledger)
        or result_checkpoint["stores"]["experiment_ledger"]["sha256"]
        != _sha256_bytes(result_ledger)
    ):
        raise GovernedResultCLIError(
            "result publication rewrote its ledger prefix or byte binding"
        )
    if payload["decision"] == "adopt":
        _verify_adopted_result_provenance(
            root,
            github_repository=github_repository,
            branch=branch,
            candidate_revision=candidate,
            plan=plan,
            plan_record_hash=plans[0]["record_hash"],
            topology=topology,
            recorded_evidence=evidence,
        )
    if _remote_head(
        root,
        github_repository=github_repository,
        branch=branch,
    ) != result_revision:
        raise GovernedResultCLIError(
            "authenticated GitHub branch moved during result verification"
        )
    return {
        "branch": branch,
        "candidate_revision_sha": candidate,
        "checkpoint_sha256": current.expected_sha256,
        "decision": payload["decision"],
        "experiment_id": experiment_id,
        "github_repository": github_repository,
        "result_record_hash": result_record["record_hash"],
        "result_revision_sha": result_revision,
        "status": "authenticated_experiment_result_verified",
        "transition_sha256": transition_sha,
    }


def record_result(
    *,
    repository_root: Path | str,
    github_repository: str,
    branch: str,
    experiment_id: str,
    evidence_path: Path | str,
    decision: str,
    rationale: str,
    runtime_capture_paths: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Record one authenticated, plan-bound result without publishing it."""

    if not _EXPERIMENT_ID_RE.fullmatch(str(experiment_id)):
        raise GovernedResultCLIError("experiment_id is invalid")
    normalized_decision = str(decision).strip().casefold()
    if normalized_decision not in {"adopt", "reject", "rollback"}:
        raise GovernedResultCLIError(
            "decision must be adopt, reject, or rollback"
        )
    root = Path(repository_root).resolve()
    (
        _program_root,
        pointer_path,
        checkpoint_directory,
        stores,
    ) = _program_paths(root)
    authority = GitHubCheckpointAuthority(
        repository_root=root,
        github_repository=github_repository,
        branch=branch,
    )
    current, current_pointer, revision = authority.resolve(stores=stores)
    primary = _require_external_canonical_object(
        evidence_path,
        repository_root=root,
        label="aggregate experiment evidence",
    )
    plan, plan_record_hash, topology, primary = _validate_all_bindings(
        root,
        stores=stores,
        current=current,
        current_pointer=current_pointer,
        revision=revision,
        experiment_id=experiment_id,
        evidence=primary,
    )
    primary_raw = _canonical_bytes(primary)
    primary_sha = _sha256_bytes(primary_raw)
    grouped_provenance: dict[str, Any] | None = None
    if normalized_decision == "adopt":
        if runtime_capture_paths:
            raise GovernedResultCLIError(
                "adoption forbids caller-authored local WO20 captures"
            )
        grouped_provenance = _authenticated_grouped_provenance(
            root,
            github_repository=github_repository,
            branch=branch,
            candidate_revision=revision,
            primary=primary,
            primary_raw=primary_raw,
            topology=topology,
        )
        runtime = _authenticated_runtime(
            root,
            github_repository=github_repository,
            branch=branch,
            candidate_revision=revision,
        )
    elif runtime_capture_paths:
        runtime = _trusted_runtime(
            root,
            candidate_revision=revision,
            capture_paths=runtime_capture_paths,
        )
    else:
        runtime = _runtime_defaults(primary_sha)
    leakage_clean = _runtime_leakage_finding_count(root) == 0
    _validate_decision(
        primary,
        decision=normalized_decision,
        grouped_actions_authenticated=(
            grouped_provenance is not None
        ),
        runtime_actions_authenticated=(
            runtime.get("actions_authenticated") is True
        ),
        runtime_leakage_clean=leakage_clean,
    )
    ledger_evidence = _ledger_evidence(
        primary,
        plan=plan,
        runtime=runtime,
        grouped_provenance=grouped_provenance,
        runtime_leakage_clean=leakage_clean,
    )

    ledger = ExperimentLedger(
        stores["experiment_ledger"].path,
        protected_access_path=stores["protected_access_ledger"].path,
    )
    checkpoint_path: Path | None = None
    checkpoint_created = False
    pointer_updated = False
    receipt = None
    with current.locked():
        locked_current, locked_pointer, locked_revision = authority.resolve(
            stores=stores
        )
        if (
            locked_revision != revision
            or locked_current.expected_sha256
            != current.expected_sha256
            or locked_pointer != current_pointer
        ):
            raise GovernedResultCLIError(
                "authenticated authority changed before result publication"
            )
        locked_primary = _require_external_canonical_object(
            evidence_path,
            repository_root=root,
            label="aggregate experiment evidence",
        )
        if _canonical_bytes(locked_primary) != primary_raw:
            raise GovernedResultCLIError(
                "aggregate evidence changed before result publication"
            )
        locked_plan, locked_plan_hash, locked_topology, locked_primary = (
            _validate_all_bindings(
                root,
                stores=stores,
                current=locked_current,
                current_pointer=locked_pointer,
                revision=locked_revision,
                experiment_id=experiment_id,
                evidence=locked_primary,
            )
        )
        if (
            locked_plan != plan
            or locked_plan_hash != plan_record_hash
            or locked_primary != primary
        ):
            raise GovernedResultCLIError(
                "plan or evidence binding changed under the mutation lock"
            )
        if normalized_decision == "adopt":
            locked_grouped_provenance = (
                _authenticated_grouped_provenance(
                    root,
                    github_repository=github_repository,
                    branch=branch,
                    candidate_revision=locked_revision,
                    primary=locked_primary,
                    primary_raw=primary_raw,
                    topology=locked_topology,
                )
            )
            locked_runtime = _authenticated_runtime(
                root,
                github_repository=github_repository,
                branch=branch,
                candidate_revision=locked_revision,
            )
            if (
                locked_grouped_provenance != grouped_provenance
                or locked_runtime != runtime
            ):
                raise GovernedResultCLIError(
                    "authenticated Actions evidence changed before publication"
                )
        elif runtime_capture_paths:
            locked_runtime = _trusted_runtime(
                root,
                candidate_revision=locked_revision,
                capture_paths=runtime_capture_paths,
            )
            if locked_runtime != runtime:
                raise GovernedResultCLIError(
                    "trusted runtime evidence changed before publication"
                )
        locked_leakage_clean = (
            _runtime_leakage_finding_count(root) == 0
        )
        if locked_leakage_clean != leakage_clean:
            raise GovernedResultCLIError(
                "runtime leakage state changed before publication"
            )
        _validate_decision(
            locked_primary,
            decision=normalized_decision,
            grouped_actions_authenticated=(
                grouped_provenance is not None
            ),
            runtime_actions_authenticated=(
                runtime.get("actions_authenticated") is True
            ),
            runtime_leakage_clean=locked_leakage_clean,
        )
        if _ledger_evidence(
            locked_primary,
            plan=locked_plan,
            runtime=runtime,
            grouped_provenance=grouped_provenance,
            runtime_leakage_clean=locked_leakage_clean,
        ) != ledger_evidence:
            raise GovernedResultCLIError(
                "result ledger evidence changed under the mutation lock"
            )
        current = locked_current
        before = _ledger_bytes(stores)
        pointer_before = _read_regular_file(
            pointer_path,
            label="current checkpoint pointer",
        )
        next_pointer_raw: bytes | None = None
        after: dict[str, bytes] | None = None
        try:
            receipt = ledger.record_result(
                experiment_id,
                ledger_evidence,
                decision=normalized_decision,
                rationale=rationale,
                integrity_checkpoint=current,
                expected_head=current.verify()["stores"][
                    "experiment_ledger"
                ]["expected_head"],
            )
            after = _ledger_bytes(stores)
            CheckpointAuthorityResolver.validate_successor(
                current, receipt.integrity
            )
            for name in sorted(stores):
                if name == "experiment_ledger":
                    continue
                if after[name] != before[name]:
                    raise IntegrityError(
                        f"{name} changed during result publication"
                    )
            if not receipt.mutated:
                raise GovernedResultCLIError(
                    "result publication did not append a new record"
                )
            if after["experiment_ledger"] == before["experiment_ledger"]:
                raise IntegrityError(
                    "result publication did not change the experiment ledger"
                )
            next_digest = receipt.next_checkpoint_sha256
            checkpoint_path = checkpoint_directory / f"{next_digest}.json"
            checkpoint_created = _create_once(
                checkpoint_path,
                receipt.next_checkpoint_bytes,
                label="result successor checkpoint",
            )
            next_pointer = _pointer_for(
                next_digest,
                previous_checkpoint_sha256=current.expected_sha256,
            )
            next_pointer_raw = _canonical_bytes(next_pointer)
            _atomic_replace_exact(
                pointer_path,
                expected=pointer_before,
                replacement=next_pointer_raw,
                label="current checkpoint pointer",
            )
            pointer_updated = True
            ProgramIntegrityCheckpoint(
                checkpoint_path,
                expected_sha256=next_digest,
                stores=stores,
            ).verify()
            if _read_regular_file(
                pointer_path,
                label="updated current checkpoint pointer",
            ) != next_pointer_raw:
                raise IntegrityError(
                    "result checkpoint pointer update is incomplete"
                )
            _require_exact_worktree_changes(
                root,
                expected_paths={
                    (
                        PROGRAM_RELATIVE_PATH
                        / LEDGER_FILENAMES["experiment_ledger"]
                    ).as_posix(),
                    next_pointer["checkpoint_path"],
                    POINTER_RELATIVE_PATH.as_posix(),
                },
            )
            if _remote_head(
                root,
                github_repository=github_repository,
                branch=branch,
            ) != revision:
                raise GovernedResultCLIError(
                    "authenticated GitHub branch moved during result publication"
                )
        except BaseException as operation_error:
            cleanup_errors: list[str] = []
            try:
                current_pointer_raw = _read_regular_file(
                    pointer_path,
                    label="partial current checkpoint pointer",
                )
                pointer_was_replaced = (
                    next_pointer_raw is not None
                    and current_pointer_raw == next_pointer_raw
                )
                if pointer_updated or pointer_was_replaced:
                    _atomic_replace_exact(
                        pointer_path,
                        expected=next_pointer_raw,
                        replacement=pointer_before,
                        label="current checkpoint pointer rollback",
                    )
                elif current_pointer_raw != pointer_before:
                    cleanup_errors.append(
                        "current checkpoint pointer changed to unknown bytes"
                    )
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    "current checkpoint pointer rollback failed: "
                    f"{cleanup_error}"
                )
            try:
                if (
                    checkpoint_created
                    and checkpoint_path is not None
                    and checkpoint_path.exists()
                ):
                    if receipt is None:
                        raise IntegrityError(
                            "successor checkpoint has no owning receipt"
                        )
                    _unlink_exact(
                        checkpoint_path,
                        expected=receipt.next_checkpoint_bytes,
                        label="result successor checkpoint rollback target",
                    )
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    "successor checkpoint rollback failed: "
                    f"{cleanup_error}"
                )
            for name, store in stores.items():
                try:
                    current_ledger = _read_regular_file(
                        store.path,
                        label=f"{name} rollback target",
                    )
                    if current_ledger != before[name]:
                        if after is None or current_ledger != after[name]:
                            raise IntegrityError(
                                f"{name} changed to bytes not owned by "
                                "this result publication"
                            )
                        _atomic_replace_exact(
                            store.path,
                            expected=after[name],
                            replacement=before[name],
                            label=f"{name} rollback",
                        )
                except BaseException as cleanup_error:
                    cleanup_errors.append(
                        f"{name} rollback failed: {cleanup_error}"
                    )
            if cleanup_errors:
                raise IntegrityError(
                    "result publication cleanup was incomplete: "
                    + "; ".join(cleanup_errors)
                ) from operation_error
            raise
    if receipt is None:  # pragma: no cover - defensive
        raise IntegrityError("result publication produced no receipt")
    return {
        "branch": branch,
        "checkpoint_path": (
            CHECKPOINT_DIRECTORY_RELATIVE_PATH
            / f"{receipt.next_checkpoint_sha256}.json"
        ).as_posix(),
        "checkpoint_sha256": receipt.next_checkpoint_sha256,
        "decision": normalized_decision,
        "experiment_id": experiment_id,
        "github_repository": github_repository,
        "next_required_action": (
            "commit and push the one experiment-result append, successor "
            "checkpoint, and current pointer as one governance-only commit; "
            "then run verify-result-authority"
        ),
        "plan_record_hash": plan_record_hash,
        "previous_checkpoint_sha256": receipt.previous_checkpoint_sha256,
        "remote_candidate_head": revision,
        "result_record_hash": receipt.record_hash,
        "status": "experiment_result_recorded_locally_not_published",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record one authenticated governed experiment result."
    )
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--github-repository", required=True)
    parser.add_argument("--branch", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("verify-candidate-authority")
    candidate.add_argument("--experiment-id", required=True)
    result = commands.add_parser("verify-result-authority")
    result.add_argument("--experiment-id", required=True)
    record = commands.add_parser("record-result")
    record.add_argument("--experiment-id", required=True)
    record.add_argument("--evidence", type=Path, required=True)
    record.add_argument(
        "--decision",
        choices=("adopt", "reject", "rollback"),
        required=True,
    )
    record.add_argument("--rationale", required=True)
    record.add_argument(
        "--runtime-capture",
        type=Path,
        action="append",
        default=[],
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    common = {
        "repository_root": arguments.repository_root,
        "github_repository": arguments.github_repository,
        "branch": arguments.branch,
    }
    try:
        if arguments.command == "verify-candidate-authority":
            result = verify_candidate_authority(
                **common,
                experiment_id=arguments.experiment_id,
            )
        elif arguments.command == "verify-result-authority":
            result = verify_result_authority(
                **common,
                experiment_id=arguments.experiment_id,
            )
        else:
            result = record_result(
                **common,
                experiment_id=arguments.experiment_id,
                evidence_path=arguments.evidence,
                decision=arguments.decision,
                rationale=arguments.rationale,
                runtime_capture_paths=arguments.runtime_capture,
            )
    except (ExperimentControlError, OSError) as exc:
        print(
            canonical_json({"error": str(exc), "status": "blocked"}),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
