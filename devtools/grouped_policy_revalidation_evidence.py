#!/usr/bin/env python3
"""Build strict aggregate-only evidence for the full WO-17 experiment.

The builder is deliberately separated from production capture.  It receives
identity-bearing truth, manifest, and predictions only after both truth-blind
arms are complete.  Those inputs remain external; JSON and Markdown outputs
contain hashes, counters, official score aggregates, and the exact 3x5 paired
fold metrics only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


_EVIDENCE_ARCHIVE_CONTRACT_FD_ENV = "MIB_WO17_EVIDENCE_CONTRACT_FD"
_EVIDENCE_ARCHIVE_CHILD_FLAG = "--_mib-wo17-evidence-archive-child"
_EVIDENCE_BOOTSTRAP_TIMEOUT_SECONDS = 31000
_EVIDENCE_ARCHIVE_CONTRACT_KEYS = frozenset(
    {
        "archive_sha256",
        "bootstrap_pid",
        "origin_root",
        "revision",
        "schema",
        "source_root",
        "tree_sha256",
    }
)
_EVIDENCE_PREIMPORT_CONTROL_PATHS = (
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


def _early_evidence_canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _early_evidence_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for path in root.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                "evidence archive contains a non-regular source entry"
            )
        paths.append(path)
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _early_evidence_archive_tree_sha256(raw: bytes) -> str:
    files: list[tuple[bytes, bytes]] = []
    seen: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as bundle:
        for member in bundle.getmembers():
            pure = PurePosixPath(member.name)
            if (
                pure.is_absolute()
                or "\\" in member.name
                or any(part in {"", ".", ".."} for part in pure.parts)
                or member.name in seen
                or not (member.isdir() or member.isfile())
            ):
                raise RuntimeError("Git evidence archive has an unsafe entry")
            seen.add(member.name)
            if member.isdir() or "__pycache__" in pure.parts:
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(
                    "Git evidence archive regular file has no bytes"
                )
            files.append(
                (pure.as_posix().encode("utf-8"), source.read())
            )
    digest = hashlib.sha256()
    for relative, content in sorted(files):
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(content)
    return digest.hexdigest()


def _consume_evidence_archive_contract() -> Mapping[str, Any] | None:
    descriptor_text = os.environ.pop(
        _EVIDENCE_ARCHIVE_CONTRACT_FD_ENV, None
    )
    if descriptor_text is None:
        return None
    if (
        len(sys.argv) < 2
        or sys.argv[1] != _EVIDENCE_ARCHIVE_CHILD_FLAG
    ):
        raise RuntimeError(
            "caller-supplied evidence archive state is forbidden"
        )
    del sys.argv[1]
    try:
        descriptor = int(descriptor_text)
    except ValueError as exc:
        raise RuntimeError(
            "evidence archive contract descriptor is invalid"
        ) from exc
    if descriptor <= 2:
        raise RuntimeError(
            "evidence archive contract descriptor is invalid"
        )
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISFIFO(opened.st_mode):
            raise RuntimeError(
                "evidence archive contract must arrive over a pipe"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            total += len(chunk)
            if total > 8192:
                raise RuntimeError(
                    "evidence archive contract is oversized"
                )
            chunks.append(chunk)
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    raw = b"".join(chunks)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "evidence archive contract is not canonical JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or set(value) != _EVIDENCE_ARCHIVE_CONTRACT_KEYS
        or raw != _early_evidence_canonical_bytes(value)
        or value["schema"] != "mib-wo17-evidence-archive/v1"
        or not isinstance(value["bootstrap_pid"], int)
        or isinstance(value["bootstrap_pid"], bool)
        or value["bootstrap_pid"] != os.getppid()
        or re.fullmatch(r"[0-9a-f]{40}", str(value["revision"])) is None
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(value[name])) is None
            for name in ("archive_sha256", "tree_sha256")
        )
    ):
        raise RuntimeError(
            "evidence archive contract has an inexact schema"
        )
    for name in ("origin_root", "source_root"):
        requested = Path(str(value[name]))
        resolved = requested.resolve(strict=True)
        if not requested.is_absolute() or requested != resolved:
            raise RuntimeError(
                f"evidence archive contract {name} must be canonical"
            )
    if value["origin_root"] == value["source_root"]:
        raise RuntimeError(
            "evidence archive origin and source must be distinct"
        )
    return dict(value)


try:
    _EVIDENCE_ARCHIVE_CONTRACT = _consume_evidence_archive_contract()
except (OSError, RuntimeError) as _evidence_contract_error:
    if __name__ == "__main__":
        print(
            f"grouped policy evidence error: {_evidence_contract_error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise


def _early_evidence_git(
    repository_root: Path,
    *arguments: str,
) -> bytes:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("GIT_", "PYTHON", "DYLD_", "LD_"))
    }
    environment.update({"GIT_NO_REPLACE_OBJECTS": "1", "LC_ALL": "C"})
    completed = subprocess.run(
        ["/usr/bin/git", "--no-replace-objects", *arguments],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Git could not verify the evidence archive source"
        )
    return completed.stdout


def _early_require_control_blobs_unchanged(
    repository_root: Path,
    *,
    candidate_revision: str,
    control_revision: str,
    paths: Sequence[str] = _EVIDENCE_PREIMPORT_CONTROL_PATHS,
) -> None:
    """Prove the candidate did not replace code that can observe truth."""

    parents = (
        _early_evidence_git(
            repository_root,
            "show",
            "-s",
            "--format=%P",
            candidate_revision,
        )
        .decode("ascii")
        .strip()
        .casefold()
        .split()
    )
    if parents != [control_revision]:
        raise RuntimeError(
            "candidate revision must have the supplied control as its "
            "single direct parent"
        )
    for path in paths:
        control = _early_evidence_git(
            repository_root,
            "cat-file",
            "blob",
            f"{control_revision}:{path}",
        )
        candidate = _early_evidence_git(
            repository_root,
            "cat-file",
            "blob",
            f"{candidate_revision}:{path}",
        )
        if candidate != control:
            raise RuntimeError(
                f"candidate replaced truth-adjacent control module: {path}"
            )


def _early_publish_evidence_outputs(
    values: Sequence[tuple[Path, bytes]],
) -> None:
    created: list[Path] = []
    try:
        for path, raw in values:
            if path.exists() or path.is_symlink():
                raise RuntimeError(
                    "evidence outputs are create-once"
                )
            parent = path.parent.resolve(strict=True)
            if parent != path.parent or not parent.is_dir():
                raise RuntimeError(
                    "evidence output parent is not canonical"
                )
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags, 0o600)
            created.append(path)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                os.close(descriptor)
            if path.is_symlink() or path.read_bytes() != raw:
                raise RuntimeError(
                    "published evidence output changed after write"
                )
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _bootstrap_evidence_archive_child(
    arguments: Sequence[str],
) -> int:
    """Run all evidence logic from the exact candidate Git archive."""

    if _EVIDENCE_ARCHIVE_CONTRACT_FD_ENV in os.environ:
        print(
            "grouped policy evidence error: caller-supplied child mode is forbidden",
            file=sys.stderr,
        )
        return 1
    values = list(arguments)

    def option(name: str) -> tuple[int, str]:
        try:
            index = values.index(name)
            return index, values[index + 1]
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f"{name} is required") from exc

    original_root = Path(__file__).resolve().parents[1]
    try:
        _, revision_raw = option("--candidate-revision-sha")
        revision = revision_raw.strip().casefold()
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise RuntimeError(
                "candidate revision must be a full Git SHA"
            )
        _, control_raw = option("--control-revision-sha")
        control_revision = control_raw.strip().casefold()
        if re.fullmatch(r"[0-9a-f]{40}", control_revision) is None:
            raise RuntimeError(
                "control revision must be a full Git SHA"
            )
        _early_require_control_blobs_unchanged(
            original_root,
            candidate_revision=revision,
            control_revision=control_revision,
        )
        evaluator_index, evaluator_raw = option("--evaluator")
        evaluator_path = Path(evaluator_raw).resolve(strict=True)
        expected_evaluator = (
            original_root / "scripts" / "evaluate.py"
        ).resolve(strict=True)
        if evaluator_path != expected_evaluator:
            raise RuntimeError(
                "evaluator must be the official checkout file"
            )
        output_indexes: list[int] = []
        output_paths: list[Path] = []
        for name in ("--output-json", "--output-markdown"):
            index, raw = option(name)
            path = Path(raw)
            if not path.is_absolute():
                raise RuntimeError(
                    "evidence output paths must be absolute"
                )
            parent = path.parent.resolve(strict=True)
            if parent != path.parent or path.exists() or path.is_symlink():
                raise RuntimeError(
                    "evidence outputs must be new canonical files"
                )
            path.relative_to(original_root)
            output_indexes.append(index + 1)
            output_paths.append(path)
        head = (
            _early_evidence_git(original_root, "rev-parse", "HEAD")
            .decode("ascii")
            .strip()
            .casefold()
        )
        if (
            head != revision
            or _early_evidence_git(
                original_root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
            or _early_evidence_git(
                original_root, "rev-parse", "--is-shallow-repository"
            ).strip()
            != b"false"
            or _early_evidence_git(
                original_root,
                "for-each-ref",
                "--format=%(refname)",
                "refs/replace/",
            )
        ):
            raise RuntimeError(
                "evidence build requires the exact clean candidate checkout"
            )
        if (
            _early_evidence_git(
                original_root,
                "cat-file",
                "blob",
                f"{revision}:devtools/grouped_policy_revalidation_evidence.py",
            )
            != Path(__file__).read_bytes()
        ):
            raise RuntimeError(
                "evidence bootstrap differs from the candidate commit"
            )
        index_flags = _early_evidence_git(
            original_root, "ls-files", "-v", "-z"
        ).decode("utf-8")
        if any(
            len(entry) < 3 or entry[0] != "H" or entry[1] != " "
            for entry in index_flags.split("\0")
            if entry
        ):
            raise RuntimeError(
                "evidence build forbids exceptional Git index flags"
            )
        staged = _early_evidence_git(
            original_root, "ls-files", "--stage", "-z"
        ).decode("utf-8")
        for entry in staged.split("\0"):
            if not entry:
                continue
            metadata = entry.split("\t", 1)[0].split()
            if (
                len(metadata) != 3
                or metadata[0] not in {"100644", "100755"}
                or metadata[2] != "0"
            ):
                raise RuntimeError(
                    "evidence build requires ordinary stage-zero files"
                )
        graft_raw = (
            _early_evidence_git(
                original_root, "rev-parse", "--git-path", "info/grafts"
            )
            .decode("utf-8")
            .strip()
        )
        graft_path = Path(graft_raw)
        if not graft_path.is_absolute():
            graft_path = original_root / graft_path
        if graft_path.exists() and graft_path.stat().st_size:
            raise RuntimeError("evidence build forbids grafted history")
        archive = _early_evidence_git(
            original_root, "archive", "--format=tar", revision
        )
        archive_sha256 = hashlib.sha256(archive).hexdigest()
        tree_sha256 = _early_evidence_archive_tree_sha256(archive)
        with tempfile.TemporaryDirectory(
            prefix="mib-wo17-evidence-archive-"
        ) as temporary_name:
            archive_root = Path(temporary_name).resolve() / "source"
            archive_root.mkdir(mode=0o700)
            seen: set[str] = set()
            with tarfile.open(
                fileobj=io.BytesIO(archive), mode="r:"
            ) as bundle:
                for member in bundle.getmembers():
                    pure = PurePosixPath(member.name)
                    if (
                        pure.is_absolute()
                        or "\\" in member.name
                        or any(
                            part in {"", ".", ".."}
                            for part in pure.parts
                        )
                        or member.name in seen
                        or not (member.isdir() or member.isfile())
                    ):
                        raise RuntimeError(
                            "Git evidence archive has an unsafe entry"
                        )
                    seen.add(member.name)
                    target = archive_root.joinpath(*pure.parts)
                    if member.isdir():
                        target.mkdir(
                            mode=0o700, parents=True, exist_ok=True
                        )
                        continue
                    target.parent.mkdir(
                        mode=0o700, parents=True, exist_ok=True
                    )
                    source = bundle.extractfile(member)
                    if source is None:
                        raise RuntimeError(
                            "Git evidence archive file has no bytes"
                        )
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                    descriptor = os.open(target, flags, 0o400)
                    try:
                        with os.fdopen(
                            descriptor, "wb", closefd=False
                        ) as handle:
                            for chunk in iter(
                                lambda: source.read(1024 * 1024), b""
                            ):
                                handle.write(chunk)
                    finally:
                        os.close(descriptor)
            if (
                _early_evidence_tree_sha256(archive_root)
                != tree_sha256
            ):
                raise RuntimeError(
                    "extracted evidence archive digest disagrees"
                )
            archive_evaluator = (
                archive_root / "scripts" / "evaluate.py"
            ).resolve(strict=True)
            values[evaluator_index + 1] = str(archive_evaluator)
            archive_outputs: list[Path] = []
            writable_parents: set[Path] = set()
            for index, requested in zip(
                output_indexes, output_paths, strict=True
            ):
                relative = requested.relative_to(original_root)
                target = archive_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                values[index] = str(target)
                archive_outputs.append(target)
                writable_parents.add(target.parent)
            for path in sorted(
                archive_root.rglob("*"),
                key=lambda item: len(item.parts),
                reverse=True,
            ):
                os.chmod(path, 0o444 if path.is_file() else 0o555)
            os.chmod(archive_root, 0o555)
            for parent in writable_parents:
                os.chmod(parent, 0o700)
            contract = {
                "archive_sha256": archive_sha256,
                "bootstrap_pid": os.getpid(),
                "origin_root": str(original_root),
                "revision": revision,
                "schema": "mib-wo17-evidence-archive/v1",
                "source_root": str(archive_root),
                "tree_sha256": tree_sha256,
            }
            read_fd, write_fd = os.pipe()
            try:
                os.set_inheritable(read_fd, True)
                with os.fdopen(
                    write_fd, "wb", closefd=True
                ) as handle:
                    handle.write(
                        _early_evidence_canonical_bytes(contract)
                    )
                write_fd = -1
                child_environment = {
                    "LC_ALL": "C",
                    "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    _EVIDENCE_ARCHIVE_CONTRACT_FD_ENV: str(read_fd),
                }
                child = subprocess.run(
                    (
                        sys.executable,
                        "-I",
                        "-B",
                        str(
                            archive_root
                            / "devtools"
                            / "grouped_policy_revalidation_evidence.py"
                        ),
                        _EVIDENCE_ARCHIVE_CHILD_FLAG,
                        *values,
                    ),
                    cwd=archive_root,
                    env=child_environment,
                    pass_fds=(read_fd,),
                    check=False,
                    capture_output=True,
                    timeout=_EVIDENCE_BOOTSTRAP_TIMEOUT_SECONDS,
                )
                if (
                    _early_evidence_git(
                        original_root, "rev-parse", "HEAD"
                    )
                    .decode("ascii")
                    .strip()
                    .casefold()
                    != revision
                    or hashlib.sha256(
                        _early_evidence_git(
                            original_root,
                            "archive",
                            "--format=tar",
                            revision,
                        )
                    ).hexdigest()
                    != archive_sha256
                ):
                    raise RuntimeError(
                        "evidence archive origin changed during build"
                    )
                if child.returncode == 0:
                    generated = tuple(
                        path.read_bytes() for path in archive_outputs
                    )
                    _early_publish_evidence_outputs(
                        tuple(zip(output_paths, generated, strict=True))
                    )
                sys.stdout.buffer.write(child.stdout)
                sys.stderr.buffer.write(child.stderr)
                return child.returncode
            finally:
                for descriptor in (read_fd, write_fd):
                    if descriptor >= 0:
                        try:
                            os.close(descriptor)
                        except OSError:
                            pass
                for path in (
                    archive_root,
                    *(
                        item
                        for item in archive_root.rglob("*")
                        if item.is_dir()
                    ),
                ):
                    try:
                        os.chmod(path, 0o700)
                    except OSError:
                        pass
    except (
        OSError,
        RuntimeError,
        subprocess.TimeoutExpired,
        tarfile.TarError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        print(f"grouped policy evidence error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__" and _EVIDENCE_ARCHIVE_CONTRACT is None:
    raise SystemExit(_bootstrap_evidence_archive_child(sys.argv[1:]))


REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_AUTHORITY_ROOT = (
    Path(str(_EVIDENCE_ARCHIVE_CONTRACT["origin_root"]))
    if _EVIDENCE_ARCHIVE_CONTRACT is not None
    else REPO_ROOT
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.experiment_control import (  # noqa: E402
    CanonicalHashChainStore,
    ExperimentControlError,
    IntegrityError,
    RepeatedGroupedSplitManager,
    _normalize_experiment_plan,
    _validate_program_integrity_checkpoint,
    canonical_json,
    require_aggregate_only,
)
from devtools.grouped_policy_revalidation_gate import (  # noqa: E402
    PUBLIC_EVIDENCE_LABEL,
    REQUIRED_FOLDS,
    REQUIRED_REPEATS,
    GroupedPolicyEvidence,
    GroupedPolicyRevalidationGate,
    PolicyActivityAggregate,
    PolicyArmAggregate,
    PolicyFoldPair,
)
from devtools.grouped_split_evidence import (  # noqa: E402
    FrozenLayoutManifest,
    _strict_freezer_manifest,
    _verify_input_tree_and_recomputed_manifest,
    _write_output_pair,
)
from devtools.policy_grouped_capture_contract import (  # noqa: E402
    AUDIT_ROOT_KEYS as _AUDIT_ROOT_KEYS,
    CAPTURE_REPEAT_COUNT,
    CONTRACT_CHECK_NAMES as _CONTRACT_CHECK_NAMES,
    FIELD_NAMES,
    MATCHER_COUNT_NAMES as _MATCHER_COUNT_NAMES,
    OBSERVATION_CHECKS as _OBSERVATION_CHECKS,
    OBSERVATION_COUNTS as _OBSERVATION_COUNTS,
    OBSERVATION_METRICS as _OBSERVATION_METRICS,
    OBSERVATION_ROOT_KEYS as _OBSERVATION_ROOT_KEYS,
    POLICY_AUDIT_COUNT_NAMES,
    PredictionContractError,
    SANDBOX_BACKEND as _SANDBOX_BACKEND,
    archive_tree_sha256 as _early_archive_tree_sha256,
    validate_prediction_bytes,
)
from devtools.policy_revalidation_audit_contract import (  # noqa: E402
    CONTRACT_AUDIT_COUNTS,
    UNSAFE_CONTRACT_AUDIT_COUNTS,
)


REQUIRED_PUBLIC_RECORDS = 1000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_RUNTIME_CONTRACT_KEYS = frozenset(
    {
        "capture",
        "container_limits",
        "environment",
        "evaluation",
        "interface",
        "schema_version",
    }
)
_RUNTIME_CAPTURE = {
    "arm_repeat_count": 2,
    "execution": "sequential",
    "max_workers": 4,
    "metrics_source": (
        "fresh_process_rusage_self_plus_waited_children_and_monotonic_wall"
    ),
    "required_byte_determinism": True,
}
_RUNTIME_CONTAINER_LIMITS = {
    "image_bytes": 4294967296,
    "max_model_artifact_bytes": 262144000,
    "model_bytes": 1073741824,
    "network": "none",
    "output_bytes": 26214400,
    "peak_memory_bytes": 8589934592,
    "per_record_runtime_seconds": 6,
    "runtime_seconds": 30000,
    "tmp_bytes": 2147483648,
}
_RUNTIME_ENVIRONMENT = {
    "MIB_MAX_WORKERS": "4",
    "MKL_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
}
_RUNTIME_INTERFACE = {
    "entrypoint": "solution.py",
    "input": "directory containing canonical PDF cases",
    "output": "canonical twelve-field JSONL",
    "runner": "run.sh",
}
_NON_DECISION_FIELDS = tuple(
    name for name in FIELD_NAMES if name not in {"adjudication", "confidence"}
)
_CONFUSION_TRUTH = ("APPROVED", "DENIED", "NEEDS_REVIEW")
_CONFUSION_PREDICTION = (
    "APPROVED",
    "DENIED",
    "NEEDS_REVIEW",
    "MISSING",
)
_REGRESSION_SUITE_MODULES = {
    "adversarial_failure": (
        "tests.test_policy_grouped_capture",
        "tests.test_grouped_policy_revalidation_evidence",
        "tests.test_grouped_policy_revalidation_gate",
    ),
    "focused_failure": (
        "tests.test_policy_revalidation_audit_run",
        "tests.test_policy_revalidation_evidence",
        "tests.test_policy_revalidation_gate",
        "tests.test_grouped_policy_revalidation_evidence",
        "tests.test_grouped_policy_revalidation_gate",
        "tests.test_policy_grouped_capture",
    ),
}
_GATE_RESULT_NAMES = frozenset(
    {
        "public_exposed_evidence",
        "source_and_diff_bound",
        "population_bound",
        "runtime_contract_bound",
        "control_complete",
        "candidate_complete",
        "capture_deterministic",
        "group_exclusive",
        "paired_fold_members",
        "split_deterministic",
        "full_score_positive",
        "extraction_score_unchanged",
        "non_decision_fields_unchanged",
        "decision_variable_nonvacuous",
        "repeat_weighted_deltas_positive",
        "no_negative_folds",
        "leave_best_fold_out_positive",
        "no_new_false_approvals",
        "no_catastrophic_false_approvals",
        "guarded_initial_approval_nonvacuous",
        "eligible_guarded_activity_nonvacuous",
        "no_unguarded_initial_approval",
        "no_late_revalidation_approval",
        "legacy_forced_counter_fully_accounted",
        "contract_probes_pass",
        "regression_suites_clean",
    }
)
_AGGREGATE_COUNT_NAMES = frozenset(
    {
        "record_count",
        "layout_group_count",
        "control_record_count",
        "control_false_approval_count",
        "control_catastrophic_false_approval_count",
        "control_missing_record_count",
        "control_invalid_record_count",
        "control_duplicate_record_count",
        "control_extra_record_count",
        "new_false_approval_count",
        "new_catastrophic_false_approval_count",
        "candidate_false_approval_count",
        "candidate_catastrophic_false_approval_count",
        "candidate_missing_record_count",
        "candidate_invalid_record_count",
        "candidate_duplicate_record_count",
        "candidate_extra_record_count",
        "non_decision_field_change_count",
        "decision_or_confidence_change_count",
        "guarded_initial_approval_count",
        "eligible_guarded_initial_count",
        "unguarded_initial_approval_count",
        "late_revalidation_approval_count",
        "legacy_forced_approval_count",
        "regression_adversarial_failure_count",
        "regression_focused_failure_count",
        "regression_full_failure_count",
    }
)
_AGGREGATE_CHECK_NAMES = frozenset(
    {
        *_CONTRACT_CHECK_NAMES,
        "legacy_contract_nonvacuous",
        "legacy_contract_unsafe_counters_zero",
        "legacy_contract_order_accounted",
        "legacy_contract_contradiction_accounted",
        "runtime_limits_satisfied",
        "source_and_diff_bound",
        "population_bound",
        "group_exclusive",
        "paired_fold_members",
        "split_deterministic",
        "runtime_contract_bound",
        "control_capture_deterministic",
        "candidate_capture_deterministic",
    }
)
_AGGREGATE_BINDING_CHECK_NAMES = frozenset(
    {
        "source_and_diff_bound",
        "population_bound",
        "group_exclusive",
        "paired_fold_members",
        "split_deterministic",
        "runtime_contract_bound",
        "control_capture_deterministic",
        "candidate_capture_deterministic",
    }
)
_AGGREGATE_METRIC_NAMES = frozenset(
    {
        "control_total_score",
        "candidate_total_score",
        "extraction_score",
        "classification_score",
        "calibration_score",
        "control_missing_penalty",
        "candidate_missing_penalty",
        "extraction_score_delta",
        "classification_score_delta",
        "calibration_score_delta",
        *(
            f"repeat_{repeat}_{suffix}"
            for repeat in range(1, REQUIRED_REPEATS + 1)
            for suffix in (
                "weighted_score_delta",
                "positive_fold_count",
                "leave_best_fold_out_delta",
            )
        ),
    }
)
_SCORE_COMPONENT_NAMES = frozenset(
    {
        f"{arm}_{name}"
        for arm in ("control", "candidate")
        for name in (
            "extraction_score",
            "classification_score",
            "calibration_score",
            "missing_penalty",
            "total_score",
        )
    }
)
_CONFUSION_NAMES = frozenset(
    {
        (
            f"{arm}_{actual.casefold()}_to_"
            f"{predicted.casefold()}_count"
        )
        for arm in ("control", "candidate")
        for actual in _CONFUSION_TRUTH
        for predicted in _CONFUSION_PREDICTION
    }
)
_AGGREGATE_ROOT_KEYS = frozenset(
    {
        "evaluation_mode",
        "evidence_label",
        "status",
        "control_source_revision_sha",
        "candidate_source_revision_sha",
        "experiment_plan_sha256",
        "runtime_contract_sha256",
        "split_manifest_sha256",
        "input_tree_sha256",
        "truth_sha256",
        "evaluator_sha256",
        "candidate_diff_manifest_sha256",
        "record_count",
        "layout_group_count",
        "repeat_count",
        "fold_count",
        "evaluated_fold_count",
        "control_score",
        "candidate_score",
        "score_delta",
        "fold_deltas",
        "fold_weights",
        "repeat_scores",
        "deterministic",
        "fold_consistent",
        "catastrophic_false_approvals",
        "false_approvals",
        "missing_records",
        "invalid_records",
        "duplicate_records",
        "extra_records",
        "hard_gate_failure_count",
        "counts",
        "checks",
        "metrics",
        "fold_metrics",
        "gate_results",
        "base_revision_sha",
        "control_revision_sha",
        "candidate_revision_sha",
        "experiment_plan_record_sha256",
        "prereg_checkpoint_sha256",
        "prereg_governance_diff_sha256",
        "candidate_source_diff_sha256",
        "planned_scope_manifest_sha256",
        "hypothesis_sha256",
        "primary_variable_sha256",
        "capture_tool_sha256",
        "evidence_tool_sha256",
        "gate_tool_sha256",
        "control_capture_set_sha256",
        "candidate_capture_set_sha256",
        "control_producer_graph_sha256",
        "candidate_producer_graph_sha256",
        "confusion_counts",
        "score_components",
    }
)


class GroupedPolicyEvidenceBuildError(ExperimentControlError):
    """One or more external experiment bindings failed closed."""


@dataclass(frozen=True)
class _Arm:
    rows: tuple[Mapping[str, Any], ...]
    official: Mapping[str, Any]
    aggregate: PolicyArmAggregate
    observation: Mapping[str, Any]
    audit: Mapping[str, Any]
    prediction_set_sha256: str
    audit_set_sha256: str
    observation_sha256: str


@dataclass(frozen=True)
class _FreshCapture:
    prediction_raw: tuple[bytes, bytes]
    audit_raw: tuple[bytes, bytes]
    observation: Mapping[str, Any]
    observation_raw: bytes


@dataclass(frozen=True)
class GovernedExperimentTopology:
    """Cryptographically derived A→P→C source and governance bindings."""

    base_revision_sha: str
    control_revision_sha: str
    candidate_revision_sha: str
    experiment_plan_sha256: str
    experiment_plan_record_sha256: str
    prereg_checkpoint_sha256: str
    prereg_governance_diff_sha256: str
    candidate_source_diff_sha256: str
    planned_scope_manifest_sha256: str

    @property
    def planned_files_sha256(self) -> str:
        """Compatibility accessor; committed evidence uses the safe alias."""

        return self.planned_scope_manifest_sha256


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path | str) -> str:
    return _sha256_bytes(_read_regular_bytes(path, label="bound file"))


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(dict(value)) + "\n").encode("utf-8")


def _read_regular_bytes(path: Path | str, *, label: str) -> bytes:
    """Read one stable regular file without following the leaf symlink."""

    requested = Path(path)
    if not requested.is_absolute():
        raise GroupedPolicyEvidenceBuildError(
            f"{label} path must be absolute"
        )
    try:
        resolved = requested.resolve(strict=True)
        supplied = requested.lstat()
    except (OSError, RuntimeError) as exc:
        raise GroupedPolicyEvidenceBuildError(
            f"{label} is not a readable file"
        ) from exc
    if resolved != requested or stat.S_ISLNK(supplied.st_mode):
        raise GroupedPolicyEvidenceBuildError(
            f"{label} path must be canonical and symlink-free"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise GroupedPolicyEvidenceBuildError(
            f"{label} cannot be opened safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        current = resolved.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino)
            != (current.st_dev, current.st_ino)
        ):
            raise GroupedPolicyEvidenceBuildError(
                f"{label} must be one regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final = os.fstat(descriptor)
        final_path = resolved.lstat()
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        ) != (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        ) or (final.st_dev, final.st_ino) != (
            final_path.st_dev,
            final_path.st_ino,
        ):
            raise GroupedPolicyEvidenceBuildError(
                f"{label} changed while being read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _canonical_object(path: Path | str, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(path, label=label)
    return _canonical_object_bytes(raw, label=label), raw


def _canonical_object_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    """Decode bytes that were already read and bound by the caller.

    Security-sensitive evidence must never be reopened between hashing and
    parsing.  Keeping this decoder byte-oriented makes that invariant explicit
    at every call site.
    """

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroupedPolicyEvidenceBuildError(
            f"{label} must be UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise GroupedPolicyEvidenceBuildError(
            f"{label} must be a canonical JSON object"
        )
    return value


def _load_bound_evaluator(
    raw: bytes,
    *,
    source_path: Path,
) -> types.ModuleType:
    """Compile only the stably read, hash-bound evaluator bytes."""

    module = types.ModuleType("_mib_wo17_bound_official_evaluator")
    module.__file__ = str(source_path)
    module.__package__ = None
    try:
        code = compile(raw, str(source_path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except Exception as exc:
        raise GroupedPolicyEvidenceBuildError(
            "bound evaluator bytes cannot be loaded"
        ) from exc
    required = {
        "build_results",
        "index_submission",
        "score_case",
    }
    if any(not callable(getattr(module, name, None)) for name in required):
        raise GroupedPolicyEvidenceBuildError(
            "bound evaluator does not expose the official scoring interface"
        )
    return module


def _digest(value: Any, *, label: str) -> str:
    normalized = str(value).strip().casefold()
    if not _SHA256_RE.fullmatch(normalized):
        raise GroupedPolicyEvidenceBuildError(
            f"{label} must be a full SHA-256 digest"
        )
    return normalized


def _commit(value: Any, *, label: str) -> str:
    normalized = str(value).strip().casefold()
    if not _COMMIT_RE.fullmatch(normalized):
        raise GroupedPolicyEvidenceBuildError(
            f"{label} must be a full Git commit SHA"
        )
    return normalized


def _count(value: Any, *, label: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GroupedPolicyEvidenceBuildError(
            f"{label} must be a {'positive' if positive else 'non-negative'} integer"
        )
    return value


def validate_aggregate_artifact(
    aggregate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact committed aggregate schema and its redundancies."""

    if (
        not isinstance(aggregate, Mapping)
        or set(aggregate) != _AGGREGATE_ROOT_KEYS
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate evidence has an inexact root schema"
        )
    require_aggregate_only(aggregate)

    def score(value: Any, *, label: str) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise GroupedPolicyEvidenceBuildError(
                f"{label} must be a finite score"
            )
        return float(value)

    if (
        aggregate["evaluation_mode"] != PUBLIC_EVIDENCE_LABEL
        or aggregate["evidence_label"] != "aggregate_only"
        or aggregate["status"] not in {"passed", "blocked"}
        or aggregate["record_count"] != REQUIRED_PUBLIC_RECORDS
        or aggregate["repeat_count"] != REQUIRED_REPEATS
        or aggregate["fold_count"] != REQUIRED_FOLDS
        or aggregate["evaluated_fold_count"]
        != REQUIRED_REPEATS * REQUIRED_FOLDS
        or aggregate["control_source_revision_sha"]
        != aggregate["control_revision_sha"]
        or aggregate["candidate_source_revision_sha"]
        != aggregate["candidate_revision_sha"]
        or aggregate["candidate_diff_manifest_sha256"]
        != aggregate["candidate_source_diff_sha256"]
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate evidence disagrees with the frozen experiment"
        )
    for name in (
        "base_revision_sha",
        "control_revision_sha",
        "candidate_revision_sha",
        "control_source_revision_sha",
        "candidate_source_revision_sha",
    ):
        _commit(aggregate[name], label=f"aggregate.{name}")
    if len(
        {
            aggregate["base_revision_sha"],
            aggregate["control_revision_sha"],
            aggregate["candidate_revision_sha"],
        }
    ) != 3:
        raise GroupedPolicyEvidenceBuildError(
            "aggregate A/P/C revisions must be distinct"
        )
    for name in (
        "experiment_plan_sha256",
        "runtime_contract_sha256",
        "split_manifest_sha256",
        "input_tree_sha256",
        "truth_sha256",
        "evaluator_sha256",
        "candidate_diff_manifest_sha256",
        "experiment_plan_record_sha256",
        "prereg_checkpoint_sha256",
        "prereg_governance_diff_sha256",
        "candidate_source_diff_sha256",
        "planned_scope_manifest_sha256",
        "hypothesis_sha256",
        "primary_variable_sha256",
        "capture_tool_sha256",
        "evidence_tool_sha256",
        "gate_tool_sha256",
        "control_capture_set_sha256",
        "candidate_capture_set_sha256",
        "control_producer_graph_sha256",
        "candidate_producer_graph_sha256",
    ):
        _digest(aggregate[name], label=f"aggregate.{name}")
    if (
        aggregate["control_producer_graph_sha256"]
        == aggregate["candidate_producer_graph_sha256"]
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate production graphs must differ"
        )

    for name in (
        "record_count",
        "layout_group_count",
        "repeat_count",
        "fold_count",
        "evaluated_fold_count",
        "catastrophic_false_approvals",
        "false_approvals",
        "missing_records",
        "invalid_records",
        "duplicate_records",
        "extra_records",
        "hard_gate_failure_count",
    ):
        _count(
            aggregate[name],
            label=f"aggregate.{name}",
            positive=name in {
                "record_count",
                "layout_group_count",
                "repeat_count",
                "fold_count",
                "evaluated_fold_count",
            },
        )
    if aggregate["layout_group_count"] < REQUIRED_FOLDS:
        raise GroupedPolicyEvidenceBuildError(
            "aggregate requires at least five layout groups"
        )
    for name in ("deterministic", "fold_consistent"):
        if not isinstance(aggregate[name], bool):
            raise GroupedPolicyEvidenceBuildError(
                f"aggregate.{name} must be a boolean"
            )

    gates = aggregate["gate_results"]
    checks = aggregate["checks"]
    counts = aggregate["counts"]
    metrics = aggregate["metrics"]
    components = aggregate["score_components"]
    confusion = aggregate["confusion_counts"]
    if (
        not isinstance(gates, Mapping)
        or set(gates) != _GATE_RESULT_NAMES
        or any(not isinstance(value, bool) for value in gates.values())
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate gate_results have an inexact schema"
        )
    failure_count = sum(not value for value in gates.values())
    if (
        aggregate["hard_gate_failure_count"] != failure_count
        or aggregate["status"]
        != ("passed" if failure_count == 0 else "blocked")
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate status does not match its hard gates"
        )
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _AGGREGATE_CHECK_NAMES
        or any(not isinstance(value, bool) for value in checks.values())
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate checks have an inexact schema"
        )
    if (
        not isinstance(counts, Mapping)
        or set(counts) != _AGGREGATE_COUNT_NAMES
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate counts have an inexact schema"
        )
    for name, value in counts.items():
        _count(value, label=f"aggregate.counts.{name}")
    if (
        counts["record_count"] != aggregate["record_count"]
        or counts["layout_group_count"] != aggregate["layout_group_count"]
        or counts["candidate_false_approval_count"]
        != aggregate["false_approvals"]
        or counts["candidate_catastrophic_false_approval_count"]
        != aggregate["catastrophic_false_approvals"]
        or counts["candidate_missing_record_count"]
        != aggregate["missing_records"]
        or counts["candidate_invalid_record_count"]
        != aggregate["invalid_records"]
        or counts["candidate_duplicate_record_count"]
        != aggregate["duplicate_records"]
        or counts["candidate_extra_record_count"]
        != aggregate["extra_records"]
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate top-level and nested counts disagree"
        )

    if (
        not isinstance(metrics, Mapping)
        or set(metrics) != _AGGREGATE_METRIC_NAMES
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate metrics have an inexact schema"
        )
    for name, value in metrics.items():
        if name.endswith("_positive_fold_count"):
            _count(value, label=f"aggregate.metrics.{name}")
            if value > REQUIRED_FOLDS:
                raise GroupedPolicyEvidenceBuildError(
                    "aggregate positive-fold count exceeds five"
                )
        else:
            score(value, label=f"aggregate.metrics.{name}")
    if (
        not isinstance(components, Mapping)
        or set(components) != _SCORE_COMPONENT_NAMES
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate score components have an inexact schema"
        )
    normalized_components = {
        name: score(value, label=f"aggregate.score_components.{name}")
        for name, value in components.items()
    }
    for arm in ("control", "candidate"):
        expected_total = (
            normalized_components[f"{arm}_extraction_score"]
            + normalized_components[f"{arm}_classification_score"]
            + normalized_components[f"{arm}_calibration_score"]
            - normalized_components[f"{arm}_missing_penalty"]
        )
        if abs(
            normalized_components[f"{arm}_total_score"] - expected_total
        ) > 1e-9:
            raise GroupedPolicyEvidenceBuildError(
                f"aggregate {arm} score components do not total"
            )
    control_score = score(
        aggregate["control_score"], label="aggregate.control_score"
    )
    candidate_score = score(
        aggregate["candidate_score"], label="aggregate.candidate_score"
    )
    delta = score(aggregate["score_delta"], label="aggregate.score_delta")
    if (
        control_score != normalized_components["control_total_score"]
        or candidate_score != normalized_components["candidate_total_score"]
        or abs(delta - (candidate_score - control_score)) > 1e-9
        or metrics["control_total_score"] != control_score
        or metrics["candidate_total_score"] != candidate_score
        or metrics["extraction_score"]
        != normalized_components["candidate_extraction_score"]
        or metrics["classification_score"]
        != normalized_components["candidate_classification_score"]
        or metrics["calibration_score"]
        != normalized_components["candidate_calibration_score"]
        or metrics["control_missing_penalty"]
        != normalized_components["control_missing_penalty"]
        or metrics["candidate_missing_penalty"]
        != normalized_components["candidate_missing_penalty"]
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate score redundancies disagree"
        )
    for component in (
        "extraction_score",
        "classification_score",
        "calibration_score",
    ):
        if abs(
            float(metrics[f"{component}_delta"])
            - (
                normalized_components[f"candidate_{component}"]
                - normalized_components[f"control_{component}"]
            )
        ) > 1e-9:
            raise GroupedPolicyEvidenceBuildError(
                "aggregate component delta is inconsistent"
            )

    if (
        not isinstance(confusion, Mapping)
        or set(confusion) != _CONFUSION_NAMES
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate confusion counts have an inexact schema"
        )
    for name, value in confusion.items():
        _count(value, label=f"aggregate.confusion_counts.{name}")
    for arm in ("control", "candidate"):
        if sum(
            value
            for name, value in confusion.items()
            if name.startswith(f"{arm}_")
        ) != REQUIRED_PUBLIC_RECORDS:
            raise GroupedPolicyEvidenceBuildError(
                f"aggregate {arm} confusion does not cover the population"
            )

    folds = aggregate["fold_metrics"]
    expected_fold_names = tuple(
        f"repeat_{repeat}_fold_{fold}"
        for repeat in range(1, REQUIRED_REPEATS + 1)
        for fold in range(1, REQUIRED_FOLDS + 1)
    )
    if not isinstance(folds, Mapping) or set(folds) != set(
        expected_fold_names
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate fold_metrics have an inexact schema"
        )
    observed_deltas: list[float] = []
    observed_weights: list[int] = []
    for name in expected_fold_names:
        fold = folds[name]
        if not isinstance(fold, Mapping) or set(fold) != {
            "record_count",
            "layout_group_count",
            "control_score",
            "candidate_score",
            "score_delta",
        }:
            raise GroupedPolicyEvidenceBuildError(
                f"aggregate fold {name} has an inexact schema"
            )
        record_count = _count(
            fold["record_count"],
            label=f"aggregate.fold_metrics.{name}.record_count",
            positive=True,
        )
        _count(
            fold["layout_group_count"],
            label=f"aggregate.fold_metrics.{name}.layout_group_count",
            positive=True,
        )
        fold_control = score(
            fold["control_score"],
            label=f"aggregate.fold_metrics.{name}.control_score",
        )
        fold_candidate = score(
            fold["candidate_score"],
            label=f"aggregate.fold_metrics.{name}.candidate_score",
        )
        fold_delta = score(
            fold["score_delta"],
            label=f"aggregate.fold_metrics.{name}.score_delta",
        )
        if abs(fold_delta - (fold_candidate - fold_control)) > 1e-9:
            raise GroupedPolicyEvidenceBuildError(
                f"aggregate fold {name} delta is inconsistent"
            )
        observed_weights.append(record_count)
        observed_deltas.append(fold_delta)
    for repeat in range(REQUIRED_REPEATS):
        start = repeat * REQUIRED_FOLDS
        stop = start + REQUIRED_FOLDS
        current = expected_fold_names[start:stop]
        if (
            sum(folds[name]["record_count"] for name in current)
            != REQUIRED_PUBLIC_RECORDS
            or sum(folds[name]["layout_group_count"] for name in current)
            != aggregate["layout_group_count"]
        ):
            raise GroupedPolicyEvidenceBuildError(
                "aggregate repeat folds do not cover the full population"
            )
    fold_deltas = aggregate["fold_deltas"]
    fold_weights = aggregate["fold_weights"]
    repeat_scores = aggregate["repeat_scores"]
    if (
        not isinstance(fold_deltas, list)
        or len(fold_deltas) != len(expected_fold_names)
        or not isinstance(fold_weights, list)
        or len(fold_weights) != len(expected_fold_names)
        or not isinstance(repeat_scores, list)
        or len(repeat_scores) != REQUIRED_REPEATS
    ):
        raise GroupedPolicyEvidenceBuildError(
            "aggregate fold vectors have an invalid shape"
        )
    if fold_deltas != observed_deltas or fold_weights != observed_weights:
        raise GroupedPolicyEvidenceBuildError(
            "aggregate fold vectors disagree with fold_metrics"
        )
    for repeat, value in enumerate(repeat_scores, start=1):
        repeat_score = score(
            value, label=f"aggregate.repeat_scores.{repeat}"
        )
        if repeat_score != metrics[
            f"repeat_{repeat}_weighted_score_delta"
        ]:
            raise GroupedPolicyEvidenceBuildError(
                "aggregate repeat score redundancies disagree"
            )

    # Reconstruct the exact gate input from the serialized primitive facts and
    # require byte-for-byte-equivalent derived output.  This prevents callers
    # from setting optimistic gate booleans, repeat summaries, safety counters,
    # or top-level redundancies by hand.
    control_arm = PolicyArmAggregate(
        total_score=normalized_components["control_total_score"],
        extraction_score=normalized_components["control_extraction_score"],
        classification_score=normalized_components[
            "control_classification_score"
        ],
        calibration_score=normalized_components["control_calibration_score"],
        missing_penalty=normalized_components["control_missing_penalty"],
        record_count=counts["control_record_count"],
        catastrophic_false_approvals=counts[
            "control_catastrophic_false_approval_count"
        ],
        false_approvals=counts["control_false_approval_count"],
        missing_records=counts["control_missing_record_count"],
        invalid_records=counts["control_invalid_record_count"],
        duplicate_records=counts["control_duplicate_record_count"],
        extra_records=counts["control_extra_record_count"],
        deterministic=checks["control_capture_deterministic"],
    )
    candidate_arm = PolicyArmAggregate(
        total_score=normalized_components["candidate_total_score"],
        extraction_score=normalized_components["candidate_extraction_score"],
        classification_score=normalized_components[
            "candidate_classification_score"
        ],
        calibration_score=normalized_components[
            "candidate_calibration_score"
        ],
        missing_penalty=normalized_components["candidate_missing_penalty"],
        record_count=counts["record_count"],
        catastrophic_false_approvals=counts[
            "candidate_catastrophic_false_approval_count"
        ],
        false_approvals=counts["candidate_false_approval_count"],
        missing_records=counts["candidate_missing_record_count"],
        invalid_records=counts["candidate_invalid_record_count"],
        duplicate_records=counts["candidate_duplicate_record_count"],
        extra_records=counts["candidate_extra_record_count"],
        deterministic=checks["candidate_capture_deterministic"],
    )
    fold_pairs = tuple(
        PolicyFoldPair(
            repeat=(index // REQUIRED_FOLDS),
            fold=(index % REQUIRED_FOLDS),
            record_count=folds[name]["record_count"],
            layout_group_count=folds[name]["layout_group_count"],
            control_score=folds[name]["control_score"],
            candidate_score=folds[name]["candidate_score"],
        )
        for index, name in enumerate(expected_fold_names)
    )
    capture_checks = {
        name: flag
        for name, flag in checks.items()
        if name not in _AGGREGATE_BINDING_CHECK_NAMES
    }
    reconstructed = GroupedPolicyEvidence(
        control_source_revision_sha=aggregate[
            "control_source_revision_sha"
        ],
        candidate_source_revision_sha=aggregate[
            "candidate_source_revision_sha"
        ],
        experiment_plan_sha256=aggregate["experiment_plan_sha256"],
        runtime_contract_sha256=aggregate["runtime_contract_sha256"],
        split_manifest_sha256=aggregate["split_manifest_sha256"],
        input_tree_sha256=aggregate["input_tree_sha256"],
        truth_sha256=aggregate["truth_sha256"],
        evaluator_sha256=aggregate["evaluator_sha256"],
        candidate_diff_manifest_sha256=aggregate[
            "candidate_diff_manifest_sha256"
        ],
        expected_record_count=aggregate["record_count"],
        expected_layout_group_count=aggregate["layout_group_count"],
        control=control_arm,
        candidate=candidate_arm,
        activity=PolicyActivityAggregate(
            eligible_guarded_initial_count=counts[
                "eligible_guarded_initial_count"
            ],
            guarded_initial_approval_count=counts[
                "guarded_initial_approval_count"
            ],
            unguarded_initial_approval_count=counts[
                "unguarded_initial_approval_count"
            ],
            late_revalidation_approval_count=counts[
                "late_revalidation_approval_count"
            ],
            legacy_forced_approval_count=counts[
                "legacy_forced_approval_count"
            ],
        ),
        folds=fold_pairs,
        new_false_approval_count=counts["new_false_approval_count"],
        new_catastrophic_false_approval_count=counts[
            "new_catastrophic_false_approval_count"
        ],
        non_decision_field_change_count=counts[
            "non_decision_field_change_count"
        ],
        decision_or_confidence_change_count=counts[
            "decision_or_confidence_change_count"
        ],
        source_and_diff_bound=checks["source_and_diff_bound"],
        population_bound=checks["population_bound"],
        group_exclusive=checks["group_exclusive"],
        paired_fold_members=checks["paired_fold_members"],
        split_deterministic=checks["split_deterministic"],
        runtime_contract_bound=checks["runtime_contract_bound"],
        capture_contract_checks=capture_checks,
        regression_counts={
            name: counts[f"regression_{name}_count"]
            for name in (
                "adversarial_failure",
                "focused_failure",
                "full_failure",
            )
        },
    )
    derived = GroupedPolicyRevalidationGate().evaluate(
        reconstructed
    ).to_aggregate_evidence()
    for name, expected_value in derived.items():
        if aggregate[name] != expected_value:
            raise GroupedPolicyEvidenceBuildError(
                f"aggregate {name} disagrees with recomputed gate evidence"
            )
    return json.loads(canonical_json(dict(aggregate)))


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
        }
    )
    return environment


def _git_bytes(
    repository_root: Path,
    *arguments: str,
) -> bytes:
    try:
        completed = subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", *arguments],
            cwd=repository_root,
            env=_git_environment(),
            check=False,
            capture_output=True,
        )
    except OSError as exc:
        raise GroupedPolicyEvidenceBuildError(
            "system Git is unavailable"
        ) from exc
    if completed.returncode != 0:
        raise GroupedPolicyEvidenceBuildError(
            "Git could not prove the requested experiment topology"
        )
    return completed.stdout


def _git_text(repository_root: Path, *arguments: str) -> str:
    raw = _git_bytes(repository_root, *arguments)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GroupedPolicyEvidenceBuildError(
            "Git returned non-UTF-8 topology metadata"
        ) from exc


def _require_unmodified_git_graph(repository_root: Path) -> None:
    if _git_text(
        repository_root, "rev-parse", "--is-shallow-repository"
    ).strip() != "false":
        raise GroupedPolicyEvidenceBuildError(
            "shallow repositories cannot prove governed ancestry"
        )
    if _git_text(
        repository_root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace/",
    ).strip():
        raise GroupedPolicyEvidenceBuildError(
            "Git replacement refs are forbidden"
        )
    graft_raw = _git_text(
        repository_root, "rev-parse", "--git-path", "info/grafts"
    ).strip()
    graft_path = Path(graft_raw)
    if not graft_path.is_absolute():
        graft_path = repository_root / graft_path
    try:
        if graft_path.exists() and graft_path.stat().st_size:
            raise GroupedPolicyEvidenceBuildError(
                "Git grafts are forbidden"
            )
    except OSError as exc:
        raise GroupedPolicyEvidenceBuildError(
            "Git graft state cannot be verified"
        ) from exc


def _require_exact_candidate_checkout(
    repository_root: Path,
    candidate_revision: str,
) -> None:
    """Bind every imported transitive byte to one ordinary clean C checkout."""

    _require_unmodified_git_graph(repository_root)
    if _git_text(repository_root, "rev-parse", "HEAD").strip().casefold() != (
        candidate_revision
    ):
        raise GroupedPolicyEvidenceBuildError(
            "evidence execution checkout is not candidate C"
        )
    if _git_bytes(
        repository_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ):
        raise GroupedPolicyEvidenceBuildError(
            "evidence execution requires an exact clean candidate C checkout"
        )
    flags = _git_text(repository_root, "ls-files", "-v", "-z")
    if any(
        len(entry) < 3 or entry[0] != "H" or entry[1] != " "
        for entry in flags.split("\0")
        if entry
    ):
        raise GroupedPolicyEvidenceBuildError(
            "evidence execution forbids exceptional Git index flags"
        )
    staged = _git_text(repository_root, "ls-files", "--stage", "-z")
    for entry in staged.split("\0"):
        if not entry:
            continue
        metadata = entry.split("\t", 1)[0].split()
        if (
            len(metadata) != 3
            or metadata[0] not in {"100644", "100755"}
            or metadata[2] != "0"
        ):
            raise GroupedPolicyEvidenceBuildError(
                "evidence execution requires ordinary stage-zero files"
            )


def _direct_parent(
    repository_root: Path,
    revision: str,
) -> str:
    line = _git_text(
        repository_root,
        "show",
        "-s",
        "--format=%H%x00%P",
        revision,
    ).strip("\n")
    parts = line.split("\0")
    if len(parts) != 2 or parts[0] != revision:
        raise GroupedPolicyEvidenceBuildError(
            "commit identity could not be proven"
        )
    parents = parts[1].split()
    if len(parents) != 1 or not _COMMIT_RE.fullmatch(parents[0]):
        raise GroupedPolicyEvidenceBuildError(
            "governed transitions must be direct non-merge children"
        )
    return parents[0]


@dataclass(frozen=True)
class _RawDiff:
    path: str
    old_mode: str
    new_mode: str
    old_object: str
    new_object: str
    status: str


def _raw_diff(
    repository_root: Path,
    parent: str,
    child: str,
) -> tuple[tuple[_RawDiff, ...], str]:
    raw = _git_bytes(
        repository_root,
        "diff",
        "--raw",
        "--full-index",
        "--abbrev=40",
        "-z",
        "--no-renames",
        "--no-ext-diff",
        "--ignore-submodules=none",
        parent,
        child,
        "--",
    )
    tokens = raw.split(b"\0")
    if tokens[-1] != b"" or len(tokens[:-1]) % 2:
        raise GroupedPolicyEvidenceBuildError(
            "Git raw diff has an invalid record shape"
        )
    records: list[_RawDiff] = []
    for index in range(0, len(tokens) - 1, 2):
        try:
            header = tokens[index].decode("ascii")
            path = tokens[index + 1].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GroupedPolicyEvidenceBuildError(
                "Git diff paths and headers must be UTF-8/ASCII"
            ) from exc
        fields = header.split()
        if (
            len(fields) != 5
            or not fields[0].startswith(":")
            or not re.fullmatch(r"[0-7]{6}", fields[0][1:])
            or not re.fullmatch(r"[0-7]{6}", fields[1])
            or not _COMMIT_RE.fullmatch(fields[2])
            or not _COMMIT_RE.fullmatch(fields[3])
            or fields[4] not in {"A", "D", "M"}
        ):
            raise GroupedPolicyEvidenceBuildError(
                "Git raw diff contains an unsupported transition"
            )
        pure = PurePosixPath(path)
        if (
            not path
            or "\\" in path
            or pure.is_absolute()
            or str(pure) != path
            or any(part in {"", ".", ".."} for part in pure.parts)
        ):
            raise GroupedPolicyEvidenceBuildError(
                "Git raw diff contains an unsafe path"
            )
        records.append(
            _RawDiff(
                path=path,
                old_mode=fields[0][1:],
                new_mode=fields[1],
                old_object=fields[2],
                new_object=fields[3],
                status=fields[4],
            )
        )
    if len({record.path for record in records}) != len(records):
        raise GroupedPolicyEvidenceBuildError(
            "Git raw diff contains duplicate paths"
        )
    return tuple(records), _sha256_bytes(raw)


def _git_blob(
    repository_root: Path,
    revision: str,
    path: str,
) -> bytes:
    return _git_bytes(repository_root, "cat-file", "blob", f"{revision}:{path}")


def _decode_canonical_git_object(
    raw: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroupedPolicyEvidenceBuildError(
            f"{label} is not canonical JSON"
        ) from exc
    if (
        not isinstance(value, dict)
        or raw != (canonical_json(value) + "\n").encode("utf-8")
    ):
        raise GroupedPolicyEvidenceBuildError(
            f"{label} is not canonical JSON"
        )
    return value


def _checkpoint_pointer(
    raw: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    value = _decode_canonical_git_object(raw, label=label)
    if set(value) != {
        "checkpoint_path",
        "checkpoint_sha256",
        "previous_checkpoint_sha256",
        "schema",
    } or value["schema"] != "mib-program-current-checkpoint/v1":
        raise GroupedPolicyEvidenceBuildError(
            f"{label} has an invalid schema"
        )
    digest = _digest(
        value["checkpoint_sha256"],
        label=f"{label}.checkpoint_sha256",
    )
    previous = value["previous_checkpoint_sha256"]
    if previous is not None:
        previous = _digest(previous, label=f"{label}.previous_checkpoint")
    expected_path = f"evaluation/program/{digest}.json"
    if value["checkpoint_path"] != expected_path:
        raise GroupedPolicyEvidenceBuildError(
            f"{label} checkpoint path is not digest-addressed"
        )
    return {
        "checkpoint_path": expected_path,
        "checkpoint_sha256": digest,
        "previous_checkpoint_sha256": previous,
        "schema": value["schema"],
    }


_GOVERNED_LEDGER_PATHS = {
    "candidate_state_ledger": "evaluation/program/candidate_state_ledger.jsonl",
    "experiment_ledger": "evaluation/program/experiment_ledger.jsonl",
    "protected_access_ledger": "evaluation/program/protected_access_ledger.jsonl",
    "taint_registry": "evaluation/program/taint_registry.jsonl",
}
_POINTER_PATH = "evaluation/program/current_checkpoint.json"
_EXPERIMENT_LEDGER_PATH = _GOVERNED_LEDGER_PATHS["experiment_ledger"]
_FORBIDDEN_CANDIDATE_CONTROL_PATHS = frozenset(
    {
        *_EVIDENCE_PREIMPORT_CONTROL_PATHS,
        "devtools/governed_experiment_cli.py",
    }
)


def _verified_checkpoint(
    repository_root: Path,
    revision: str,
    pointer: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    raw = _git_blob(
        repository_root,
        revision,
        str(pointer["checkpoint_path"]),
    )
    if _sha256_bytes(raw) != pointer["checkpoint_sha256"]:
        raise GroupedPolicyEvidenceBuildError(
            "checkpoint bytes do not match their pointer"
        )
    value = _decode_canonical_git_object(raw, label="program checkpoint")
    try:
        normalized = _validate_program_integrity_checkpoint(value)
    except IntegrityError as exc:
        raise GroupedPolicyEvidenceBuildError(
            "program checkpoint contract is invalid"
        ) from exc
    for name, expected_path in _GOVERNED_LEDGER_PATHS.items():
        anchor = normalized["stores"][name]
        if anchor.get("path") != Path(expected_path).name:
            raise GroupedPolicyEvidenceBuildError(
                "checkpoint ledger path binding is invalid"
            )
        ledger_raw = _git_blob(repository_root, revision, expected_path)
        if _sha256_bytes(ledger_raw) != anchor["sha256"]:
            raise GroupedPolicyEvidenceBuildError(
                "checkpoint ledger byte binding is invalid"
            )
        try:
            records = CanonicalHashChainStore._parse(
                ledger_raw.decode("utf-8")
            )
        except (UnicodeDecodeError, IntegrityError) as exc:
            raise GroupedPolicyEvidenceBuildError(
                "checkpoint ledger hash chain is invalid"
            ) from exc
        actual_head = (
            records[-1]["record_hash"] if records else "0" * 64
        )
        if (
            len(records) != anchor["expected_length"]
            or actual_head != anchor["expected_head"]
        ):
            raise GroupedPolicyEvidenceBuildError(
                "checkpoint ledger head/length binding is invalid"
            )
    return normalized, raw


def verify_governed_experiment_topology(
    repository_root: Path | str,
    *,
    plan: Mapping[str, Any],
    experiment_plan_sha256: str,
    control_revision_sha: str,
    candidate_revision_sha: str,
) -> GovernedExperimentTopology:
    """Prove exact A→P→C governance and source transitions without writes."""

    root = Path(repository_root).resolve(strict=True)
    normalized_plan = _normalize_experiment_plan(plan)
    plan_sha = _digest(
        experiment_plan_sha256, label="experiment_plan_sha256"
    )
    if _sha256_bytes(_canonical_bytes(normalized_plan)) != plan_sha:
        raise GroupedPolicyEvidenceBuildError(
            "experiment_plan_sha256 does not match canonical plan bytes"
        )
    control = _commit(
        control_revision_sha, label="control_revision_sha"
    )
    candidate = _commit(
        candidate_revision_sha, label="candidate_revision_sha"
    )
    base = _commit(
        normalized_plan["parent_commit_sha"],
        label="plan.parent_commit_sha",
    )
    _require_unmodified_git_graph(root)
    if _direct_parent(root, control) != base:
        raise GroupedPolicyEvidenceBuildError(
            "control P must be a direct child of preregistration base A"
        )
    if _direct_parent(root, candidate) != control:
        raise GroupedPolicyEvidenceBuildError(
            "candidate C must be a direct child of control P"
        )

    prereg_diff, prereg_diff_sha = _raw_diff(root, base, control)
    candidate_diff, candidate_diff_sha = _raw_diff(
        root, control, candidate
    )
    candidate_expected = set(normalized_plan["changed_files"])
    if any(
        path.startswith(("evaluation/program/", ".git", ".github/"))
        or path in _FORBIDDEN_CANDIDATE_CONTROL_PATHS
        for path in candidate_expected
    ):
        raise GroupedPolicyEvidenceBuildError(
            "plan.changed_files may not authorize governance or control paths"
        )
    if {record.path for record in candidate_diff} != candidate_expected:
        raise GroupedPolicyEvidenceBuildError(
            "candidate diff does not exactly match plan.changed_files"
        )
    if any(
        record.status != "M"
        or record.old_mode != record.new_mode
        or record.new_mode not in {"100644", "100755"}
        for record in candidate_diff
    ):
        raise GroupedPolicyEvidenceBuildError(
            "candidate diff must contain regular-file content modifications only"
        )

    control_pointer = _checkpoint_pointer(
        _git_blob(root, control, _POINTER_PATH),
        label="control checkpoint pointer",
    )
    base_pointer = _checkpoint_pointer(
        _git_blob(root, base, _POINTER_PATH),
        label="base checkpoint pointer",
    )
    checkpoint_path = control_pointer["checkpoint_path"]
    expected_prereg_paths = {
        _EXPERIMENT_LEDGER_PATH,
        _POINTER_PATH,
        checkpoint_path,
    }
    if {record.path for record in prereg_diff} != expected_prereg_paths:
        raise GroupedPolicyEvidenceBuildError(
            "P must contain exactly the ledger, pointer, and one checkpoint"
        )
    by_path = {record.path: record for record in prereg_diff}
    for modified_path in (_EXPERIMENT_LEDGER_PATH, _POINTER_PATH):
        record = by_path[modified_path]
        if (
            record.status != "M"
            or record.old_mode != "100644"
            or record.new_mode != "100644"
        ):
            raise GroupedPolicyEvidenceBuildError(
                "governance ledger/pointer must be regular-file modifications"
            )
    checkpoint_record = by_path[checkpoint_path]
    if (
        checkpoint_record.status != "A"
        or checkpoint_record.old_mode != "000000"
        or checkpoint_record.new_mode != "100644"
    ):
        raise GroupedPolicyEvidenceBuildError(
            "successor checkpoint must be one newly added regular file"
        )
    if (
        control_pointer["previous_checkpoint_sha256"]
        != base_pointer["checkpoint_sha256"]
    ):
        raise GroupedPolicyEvidenceBuildError(
            "successor pointer does not bind the base checkpoint"
        )

    base_checkpoint, _ = _verified_checkpoint(root, base, base_pointer)
    control_checkpoint, control_checkpoint_raw = _verified_checkpoint(
        root, control, control_pointer
    )
    for name in (
        "baseline_manifest_sha256",
        "promotion_population",
        "runtime_leakage_finding_count",
    ):
        if control_checkpoint.get(name) != base_checkpoint.get(name):
            raise GroupedPolicyEvidenceBuildError(
                "preregistration rewrote immutable checkpoint state"
            )
    for name in _GOVERNED_LEDGER_PATHS:
        before = base_checkpoint["stores"][name]
        after = control_checkpoint["stores"][name]
        if name != "experiment_ledger":
            if after != before:
                raise GroupedPolicyEvidenceBuildError(
                    "preregistration rewrote an unrelated ledger anchor"
                )
        elif (
            after["path"] != before["path"]
            or after["expected_length"] != before["expected_length"] + 1
        ):
            raise GroupedPolicyEvidenceBuildError(
                "preregistration checkpoint is not one ledger append"
            )

    base_ledger = _git_blob(root, base, _EXPERIMENT_LEDGER_PATH)
    control_ledger = _git_blob(root, control, _EXPERIMENT_LEDGER_PATH)
    if not control_ledger.startswith(base_ledger):
        raise GroupedPolicyEvidenceBuildError(
            "preregistration rewrote its ledger prefix"
        )
    try:
        before_records = CanonicalHashChainStore._parse(
            base_ledger.decode("utf-8")
        )
        after_records = CanonicalHashChainStore._parse(
            control_ledger.decode("utf-8")
        )
    except (UnicodeDecodeError, IntegrityError) as exc:
        raise GroupedPolicyEvidenceBuildError(
            "experiment ledger transition is invalid"
        ) from exc
    if len(after_records) != len(before_records) + 1:
        raise GroupedPolicyEvidenceBuildError(
            "preregistration did not append exactly one plan record"
        )
    plan_record = after_records[-1]
    payload = plan_record["payload"]
    if (
        set(payload) != {"event", "experiment_id", "plan"}
        or payload["event"] != "experiment_plan"
        or payload["plan"] != normalized_plan
        or not isinstance(payload["experiment_id"], str)
        or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", payload["experiment_id"])
    ):
        raise GroupedPolicyEvidenceBuildError(
            "appended experiment record does not contain the exact plan"
        )
    if (
        control_checkpoint["stores"]["experiment_ledger"]["expected_head"]
        != plan_record["record_hash"]
        or control_checkpoint["stores"]["experiment_ledger"]["sha256"]
        != _sha256_bytes(control_ledger)
        or _sha256_bytes(control_checkpoint_raw)
        != control_pointer["checkpoint_sha256"]
    ):
        raise GroupedPolicyEvidenceBuildError(
            "plan record is not bound by the successor checkpoint"
        )
    return GovernedExperimentTopology(
        base_revision_sha=base,
        control_revision_sha=control,
        candidate_revision_sha=candidate,
        experiment_plan_sha256=plan_sha,
        experiment_plan_record_sha256=plan_record["record_hash"],
        prereg_checkpoint_sha256=control_pointer["checkpoint_sha256"],
        prereg_governance_diff_sha256=prereg_diff_sha,
        candidate_source_diff_sha256=candidate_diff_sha,
        planned_scope_manifest_sha256=_sha256_bytes(
            (
                canonical_json(
                    sorted(normalized_plan["changed_files"])
                )
                + "\n"
            ).encode("utf-8")
        ),
    )


def _validate_runtime_contract(
    value: Mapping[str, Any],
    *,
    evaluator_sha256: str,
    input_tree_sha256: str,
    manifest_sha256: str,
    truth_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RUNTIME_CONTRACT_KEYS:
        raise GroupedPolicyEvidenceBuildError(
            "runtime contract has an inexact schema"
        )
    expected_evaluation = {
        "evaluator_sha256": evaluator_sha256,
        "evidence_label": PUBLIC_EVIDENCE_LABEL,
        "expected_record_count": REQUIRED_PUBLIC_RECORDS,
        "input_tree_sha256": input_tree_sha256,
        "layout_manifest_sha256": manifest_sha256,
        "truth_sha256": truth_sha256,
    }
    if (
        value["schema_version"] != "mib-wo17-runtime-contract/v1"
        or value["capture"] != _RUNTIME_CAPTURE
        or value["container_limits"] != _RUNTIME_CONTAINER_LIMITS
        or value["environment"] != _RUNTIME_ENVIRONMENT
        or value["evaluation"] != expected_evaluation
        or value["interface"] != _RUNTIME_INTERFACE
    ):
        raise GroupedPolicyEvidenceBuildError(
            "runtime contract bytes do not match the exact WO-17 contract"
        )
    return json.loads(canonical_json(dict(value)))


def _validate_observation(
    value: Mapping[str, Any],
    *,
    arm: str,
    revision: str,
    manifest_sha256: str,
    input_tree_sha256: str,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _OBSERVATION_ROOT_KEYS:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} observation has an inexact root schema"
        )
    require_aggregate_only(value)
    if (
        value["evaluation_mode"] != PUBLIC_EVIDENCE_LABEL
        or value["evidence_label"] != "aggregate_only"
        or value["status"] != arm
        or value["source_revision_sha"] != revision
        or value["layout_manifest_sha256"] != manifest_sha256
        or value["input_tree_sha256"] != input_tree_sha256
        or value["record_count"] != REQUIRED_PUBLIC_RECORDS
        or value["repeat_count"] != CAPTURE_REPEAT_COUNT
        or value["max_worker_count"] != runtime["capture"]["max_workers"]
        or value["sandbox_backend_sha256"]
        != _sha256_bytes(_SANDBOX_BACKEND.encode("utf-8"))
    ):
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} observation disagrees with the frozen experiment"
        )
    if value["deterministic"] is not True:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} observation is not deterministic"
        )
    checks = value["checks"]
    counts = value["counts"]
    metrics = value["metrics"]
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _OBSERVATION_CHECKS
        or any(flag is not True for flag in checks.values())
    ):
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} observation checks are incomplete or failed"
        )
    if not isinstance(counts, Mapping) or set(counts) != _OBSERVATION_COUNTS:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} observation counts have an inexact schema"
        )
    for name, count in counts.items():
        _count(count, label=f"{arm}.counts.{name}")
    if (
        counts["first_attempted_count"] != REQUIRED_PUBLIC_RECORDS
        or counts["first_answered_count"] != REQUIRED_PUBLIC_RECORDS
        or counts["second_attempted_count"] != REQUIRED_PUBLIC_RECORDS
        or counts["second_answered_count"] != REQUIRED_PUBLIC_RECORDS
        or counts["first_omitted_count"] != 0
        or counts["second_omitted_count"] != 0
        or counts["label_access_count"] != 0
    ):
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} observation does not cover the full truth-blind population"
        )
    if not isinstance(metrics, Mapping) or set(metrics) != _OBSERVATION_METRICS:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} observation metrics have an inexact schema"
        )
    for name, metric in metrics.items():
        if (
            isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or metric <= 0
        ):
            raise GroupedPolicyEvidenceBuildError(
                f"{arm} observation metric {name} must be positive"
            )
    for name in (
        "capture_tool_sha256",
        "first_audit_sha256",
        "first_predictions_sha256",
        "input_tree_sha256",
        "layout_manifest_sha256",
        "producer_graph_sha256",
        "sandbox_policy_sha256",
        "second_audit_sha256",
        "second_predictions_sha256",
        "source_archive_sha256",
        "source_tree_sha256",
    ):
        _digest(value[name], label=f"{arm}.{name}")
    return dict(value)


def _required_audit_count_names() -> frozenset[str]:
    return frozenset(
        {
            *(
                f"policy_{name}"
                for name in (*POLICY_AUDIT_COUNT_NAMES, "accepted_final_policy_result_count")
            ),
            *(f"matcher_{name}" for name in _MATCHER_COUNT_NAMES),
            *(f"contract_{name}" for name in CONTRACT_AUDIT_COUNTS),
        }
    )


def _validate_audit(
    value: Mapping[str, Any],
    *,
    arm: str,
    revision: str,
    manifest_sha256: str,
    input_tree_sha256: str,
    graph_sha256: str,
    prediction_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _AUDIT_ROOT_KEYS:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} audit has an inexact root schema"
        )
    require_aggregate_only(value)
    expected = {
        "evaluation_mode": PUBLIC_EVIDENCE_LABEL,
        "evidence_label": "aggregate_only",
        "status": arm,
        "source_revision_sha": revision,
        "layout_manifest_sha256": manifest_sha256,
        "input_tree_sha256": input_tree_sha256,
        "producer_graph_sha256": graph_sha256,
        "predictions_sha256": prediction_sha256,
    }
    if any(value[name] != expected_value for name, expected_value in expected.items()):
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} audit binding disagrees with its capture"
        )
    counts = value["counts"]
    checks = value["checks"]
    if not isinstance(counts, Mapping) or set(counts) != _required_audit_count_names():
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} audit counts have an inexact schema"
        )
    for name, count in counts.items():
        _count(count, label=f"{arm}.audit.{name}")
    if (
        counts["policy_accepted_final_policy_result_count"]
        != REQUIRED_PUBLIC_RECORDS
    ):
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} audit does not account for every accepted result"
        )
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(_CONTRACT_CHECK_NAMES)
        or any(not isinstance(flag, bool) for flag in checks.values())
    ):
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} matcher contract checks have an inexact schema"
        )
    return dict(value)


def _require_external(paths: Sequence[Path | str], *, label: str) -> None:
    forbidden_roots = {
        REPO_ROOT.resolve(),
        GIT_AUTHORITY_ROOT.resolve(),
    }
    resolved: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            raise GroupedPolicyEvidenceBuildError(
                f"{label} paths must be absolute"
            )
        try:
            value = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GroupedPolicyEvidenceBuildError(
                f"{label} path is unavailable"
            ) from exc
        for root in forbidden_roots:
            try:
                value.relative_to(root)
            except ValueError:
                continue
            raise GroupedPolicyEvidenceBuildError(
                f"{label} identity-bearing inputs must remain external"
            )
        resolved.append(value)
    if len(set(resolved)) != len(resolved):
        raise GroupedPolicyEvidenceBuildError(
            f"{label} paths must be distinct"
        )


def _prediction_rows(
    raw: bytes,
    *,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    try:
        rows = validate_prediction_bytes(
            raw,
            expected_count=REQUIRED_PUBLIC_RECORDS,
        )
    except PredictionContractError as exc:
        raise GroupedPolicyEvidenceBuildError(
            f"{label} predictions are invalid"
        ) from exc
    if any(
        not isinstance(row, Mapping) or tuple(row) != FIELD_NAMES
        for row in rows
    ):
        raise GroupedPolicyEvidenceBuildError(
            f"{label} predictions do not have the exact twelve fields"
        )
    return rows


def _truth_rows_from_bytes(
    raw: bytes,
    *,
    case_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    """Parse the exact truth bytes whose digest was checked by the caller."""

    try:
        text = raw.decode("utf-8")
        rows = tuple(csv.DictReader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise GroupedPolicyEvidenceBuildError(
            "truth is not a valid UTF-8 CSV"
        ) from exc
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id or case_id in indexed:
            raise GroupedPolicyEvidenceBuildError(
                "truth contains blank or duplicate case identities"
            )
        indexed[case_id] = row
    if set(indexed) != set(case_ids):
        raise GroupedPolicyEvidenceBuildError(
            "truth population does not exactly match the layout manifest"
        )
    return {
        case_id: indexed[case_id] for case_id in sorted(indexed)
    }


def _truth_rows(
    path: Path | str,
    *,
    case_ids: Sequence[str],
) -> dict[str, Mapping[str, Any]]:
    """Compatibility wrapper for callers that do not already hold the bytes."""

    return _truth_rows_from_bytes(
        _read_regular_bytes(path, label="truth"),
        case_ids=case_ids,
    )


def _invalid_record_count(official: Mapping[str, Any]) -> int:
    counts = official["counts"]
    return sum(
        int(counts[name])
        for name in (
            "blank_case_rows",
            "invalid_adjudication_records",
            "invalid_confidence_records",
            "invalid_fee_status_records",
        )
    )


def _false_approval_sets(
    evaluator: Any,
    truth: Mapping[str, Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[frozenset[str], frozenset[str]]:
    indexed, _, _ = evaluator.index_submission(rows)
    false_approvals = frozenset(
        case_id
        for case_id, truth_row in truth.items()
        if case_id in indexed
        and str(indexed[case_id].get("adjudication", "")).strip().upper()
        == "APPROVED"
        and str(truth_row.get("adjudication", "")).strip().upper()
        != "APPROVED"
    )
    catastrophic = frozenset(
        case_id
        for case_id, truth_row in truth.items()
        if case_id in indexed
        and evaluator.score_case(
            case_id, truth_row, indexed[case_id]
        )["catastrophic_false_approval"]
    )
    return false_approvals, catastrophic


def _arm_aggregate(
    evaluator: Any,
    truth: Mapping[str, Mapping[str, Any]],
    rows: tuple[Mapping[str, Any], ...],
    *,
    deterministic: bool,
) -> tuple[PolicyArmAggregate, Mapping[str, Any]]:
    official, _ = evaluator.build_results(truth, rows)
    counts = official["counts"]
    scores = official["scores"]
    false_approvals, _ = _false_approval_sets(evaluator, truth, rows)
    aggregate = PolicyArmAggregate(
        total_score=float(scores["total_score"]),
        extraction_score=float(scores["extraction_score"]),
        classification_score=float(scores["classification_score"]),
        calibration_score=float(scores["calibration_score"]),
        missing_penalty=float(scores["missing_penalty"]),
        record_count=int(counts["scored_predictions"]),
        catastrophic_false_approvals=int(
            official["raw"]["catastrophic_false_approvals"]
        ),
        false_approvals=len(false_approvals),
        missing_records=int(counts["missing_cases"]),
        invalid_records=_invalid_record_count(official),
        duplicate_records=int(counts["duplicate_case_ids"]),
        extra_records=int(counts["extra_cases"]),
        deterministic=deterministic,
    )
    return aggregate, official


def _artifact_set_sha256(digests: Sequence[str]) -> str:
    return _sha256_bytes(
        canonical_json(list(digests)).encode("utf-8")
    )


def _run_fresh_truth_blind_capture(
    *,
    arm: str,
    revision: str,
    input_dir: Path,
    layout_manifest_path: Path,
    manifest_sha256: str,
    input_tree_sha256: str,
    truth_path: Path,
) -> _FreshCapture:
    """Generate the authoritative capture under the OS-enforced sandbox."""

    capture_script = (
        GIT_AUTHORITY_ROOT / "devtools" / "policy_grouped_capture.py"
    ).resolve(strict=True)
    with tempfile.TemporaryDirectory(
        prefix=f"mib-wo17-evidence-{arm}-"
    ) as temporary_name:
        root = Path(temporary_name).resolve()
        predictions = (
            root / "predictions-1.jsonl",
            root / "predictions-2.jsonl",
        )
        audits = (
            root / "audit-1.json",
            root / "audit-2.json",
        )
        observation = root / "observation.json"
        arguments = (
            sys.executable,
            "-B",
            str(capture_script),
            "--arm",
            arm,
            "--source-revision-sha",
            revision,
            "--input-dir",
            str(input_dir),
            "--layout-manifest",
            str(layout_manifest_path),
            "--expected-layout-manifest-sha256",
            manifest_sha256,
            "--expected-input-tree-sha256",
            input_tree_sha256,
            "--denied-sensitive-path",
            str(truth_path),
            "--first-predictions",
            str(predictions[0]),
            "--second-predictions",
            str(predictions[1]),
            "--first-audit",
            str(audits[0]),
            "--second-audit",
            str(audits[1]),
            "--observation",
            str(observation),
            "--max-workers",
            "4",
        )
        try:
            completed = subprocess.run(
                arguments,
                cwd=GIT_AUTHORITY_ROOT,
                check=False,
                capture_output=True,
                timeout=31000,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GroupedPolicyEvidenceBuildError(
                f"{arm} sandboxed capture could not complete"
            ) from exc
        if completed.returncode != 0:
            raise GroupedPolicyEvidenceBuildError(
                f"{arm} sandboxed capture failed closed"
            )
        prediction_raw = tuple(
            _read_regular_bytes(path, label=f"{arm} fresh prediction")
            for path in predictions
        )
        audit_raw = tuple(
            _read_regular_bytes(path, label=f"{arm} fresh audit")
            for path in audits
        )
        observation_value, observation_raw = _canonical_object(
            observation,
            label=f"{arm} fresh observation",
        )
        return _FreshCapture(
            prediction_raw=(prediction_raw[0], prediction_raw[1]),
            audit_raw=(audit_raw[0], audit_raw[1]),
            observation=observation_value,
            observation_raw=observation_raw,
        )


def _load_arm(
    *,
    evaluator: Any,
    arm: str,
    revision: str,
    prediction_paths: Sequence[Path | str],
    audit_paths: Sequence[Path | str],
    observation_path: Path | str,
    manifest_sha256: str,
    input_tree_sha256: str,
    runtime: Mapping[str, Any],
    truth: Mapping[str, Mapping[str, Any]],
    fresh_capture: _FreshCapture,
) -> _Arm:
    if len(prediction_paths) != 2 or len(audit_paths) != 2:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} requires exactly two prediction and two audit files"
        )
    _require_external(
        (*prediction_paths, *audit_paths, observation_path),
        label=f"{arm} capture",
    )
    prediction_raw = tuple(
        _read_regular_bytes(path, label=f"{arm} prediction")
        for path in prediction_paths
    )
    if prediction_raw != fresh_capture.prediction_raw:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} supplied predictions differ from fresh sandbox capture"
        )
    prediction_hashes = tuple(
        _sha256_bytes(value) for value in prediction_raw
    )
    rows_by_run = tuple(
        _prediction_rows(raw, label=arm) for raw in prediction_raw
    )
    deterministic = (
        prediction_raw[0] == prediction_raw[1]
        and rows_by_run[0] == rows_by_run[1]
    )
    if not deterministic:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} prediction repeats are not byte deterministic"
        )
    supplied_observation, _ = _canonical_object(
        observation_path, label=f"{arm} observation"
    )
    # Validate the supplied observation for continuity, but only the freshly
    # generated sandbox observation is authoritative for gates and limits.
    _validate_observation(
        supplied_observation,
        arm=arm,
        revision=revision,
        manifest_sha256=manifest_sha256,
        input_tree_sha256=input_tree_sha256,
        runtime=runtime,
    )
    observation = fresh_capture.observation
    observation_raw = fresh_capture.observation_raw
    validated_observation = _validate_observation(
        observation,
        arm=arm,
        revision=revision,
        manifest_sha256=manifest_sha256,
        input_tree_sha256=input_tree_sha256,
        runtime=runtime,
    )
    if (
        validated_observation["first_predictions_sha256"]
        != prediction_hashes[0]
        or validated_observation["second_predictions_sha256"]
        != prediction_hashes[1]
    ):
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} observation does not bind its prediction bytes"
        )
    audit_raw = tuple(
        _read_regular_bytes(path, label=f"{arm} audit")
        for path in audit_paths
    )
    if audit_raw != fresh_capture.audit_raw:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} supplied audits differ from fresh sandbox capture"
        )
    audit_hashes = tuple(_sha256_bytes(value) for value in audit_raw)
    if (
        validated_observation["first_audit_sha256"] != audit_hashes[0]
        or validated_observation["second_audit_sha256"] != audit_hashes[1]
    ):
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} observation does not bind its audit bytes"
        )
    if audit_raw[0] != audit_raw[1]:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} aggregate audits are not deterministic"
        )
    audit_values = tuple(
        _canonical_object_bytes(raw, label=f"{arm} audit")
        for raw in audit_raw
    )
    validated_audits = tuple(
        _validate_audit(
            value,
            arm=arm,
            revision=revision,
            manifest_sha256=manifest_sha256,
            input_tree_sha256=input_tree_sha256,
            graph_sha256=validated_observation[
                "producer_graph_sha256"
            ],
            prediction_sha256=prediction_hashes[index],
        )
        for index, value in enumerate(audit_values)
    )
    if validated_audits[0] != validated_audits[1]:
        raise GroupedPolicyEvidenceBuildError(
            f"{arm} audit values disagree"
        )
    aggregate, official = _arm_aggregate(
        evaluator, truth, rows_by_run[0], deterministic=True
    )
    return _Arm(
        rows=rows_by_run[0],
        official=official,
        aggregate=aggregate,
        observation=validated_observation,
        audit=validated_audits[0],
        prediction_set_sha256=_artifact_set_sha256(
            prediction_hashes
        ),
        audit_set_sha256=_artifact_set_sha256(audit_hashes),
        observation_sha256=_sha256_bytes(observation_raw),
    )


def _folds(
    evaluator: Any,
    manifest: FrozenLayoutManifest,
    truth: Mapping[str, Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[PolicyFoldPair, ...], bool, bool, bool]:
    manager = RepeatedGroupedSplitManager(
        seed=manifest.split_seed,
        repeats=REQUIRED_REPEATS,
        folds=REQUIRED_FOLDS,
    )
    splits = manager.split_groups(manifest.groups)
    reversed_groups = dict(reversed(tuple(manifest.groups.items())))
    split_deterministic = splits == manager.split_groups(reversed_groups)
    group_exclusive = all(
        not set(split.tuning_groups) & set(split.validation_groups)
        and not set(split.tuning_case_ids) & set(split.validation_case_ids)
        for split in splits
    )
    control_index, _, _ = evaluator.index_submission(control_rows)
    candidate_index, _, _ = evaluator.index_submission(candidate_rows)
    pairs: list[PolicyFoldPair] = []
    paired = True
    for split in splits:
        members = set(split.validation_case_ids)
        fold_truth = {
            case_id: truth[case_id]
            for case_id in split.validation_case_ids
        }
        control_fold = tuple(
            control_index[case_id]
            for case_id in split.validation_case_ids
            if case_id in control_index
        )
        candidate_fold = tuple(
            candidate_index[case_id]
            for case_id in split.validation_case_ids
            if case_id in candidate_index
        )
        paired = paired and {
            str(row["case_id"]) for row in control_fold
        } == members == {
            str(row["case_id"]) for row in candidate_fold
        }
        control_result, _ = evaluator.build_results(
            fold_truth, control_fold
        )
        candidate_result, _ = evaluator.build_results(
            fold_truth, candidate_fold
        )
        pairs.append(
            PolicyFoldPair(
                repeat=split.repeat,
                fold=split.fold,
                record_count=len(split.validation_case_ids),
                layout_group_count=len(split.validation_groups),
                control_score=float(
                    control_result["scores"]["total_score"]
                ),
                candidate_score=float(
                    candidate_result["scores"]["total_score"]
                ),
            )
        )
    return tuple(pairs), group_exclusive, paired, split_deterministic


def _row_change_counts(
    evaluator: Any,
    control_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    case_ids: Sequence[str],
) -> tuple[int, int]:
    control, _, _ = evaluator.index_submission(control_rows)
    candidate, _, _ = evaluator.index_submission(candidate_rows)
    non_decision = 0
    decision = 0
    for case_id in case_ids:
        control_row = control.get(case_id, {})
        candidate_row = candidate.get(case_id, {})
        control_bytes = canonical_json(
            {
                name: control_row.get(name)
                for name in _NON_DECISION_FIELDS
            }
        ).encode("utf-8")
        candidate_bytes = canonical_json(
            {
                name: candidate_row.get(name)
                for name in _NON_DECISION_FIELDS
            }
        ).encode("utf-8")
        non_decision += control_bytes != candidate_bytes
        decision += any(
            control_row.get(name) != candidate_row.get(name)
            for name in ("adjudication", "confidence")
        )
    return non_decision, decision


def _complete_confusion(
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for arm, official in (("control", control), ("candidate", candidate)):
        observed = official["confusion"]
        for truth_value in _CONFUSION_TRUTH:
            for prediction_value in _CONFUSION_PREDICTION:
                key = (
                    f"{arm}_{truth_value.casefold()}_to_"
                    f"{prediction_value.casefold()}_count"
                )
                result[key] = int(
                    observed.get(
                        f"{truth_value}->{prediction_value}", 0
                    )
                )
    return result


def _capture_contract_checks(
    candidate_audit: Mapping[str, Any],
) -> dict[str, bool]:
    counts = candidate_audit["counts"]
    contract = {
        name: counts[f"contract_{name}"]
        for name in CONTRACT_AUDIT_COUNTS
    }
    nonvacuous = (
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
    )
    checks = dict(candidate_audit["checks"])
    checks.update(
        {
            "legacy_contract_nonvacuous": all(
                contract[name] > 0 for name in nonvacuous
            ),
            "legacy_contract_unsafe_counters_zero": all(
                contract[name] == 0
                for name in UNSAFE_CONTRACT_AUDIT_COUNTS
            ),
            "legacy_contract_order_accounted": (
                contract[
                    "candidate_late_recovery_before_revalidation_count"
                ]
                == contract[
                    "candidate_revalidation_after_late_recovery_count"
                ]
                == contract["normal_policy_rerun_count"]
            ),
            "legacy_contract_contradiction_accounted": (
                contract["contradicted_synthetic_reason_before_count"]
                == contract[
                    "contradicted_synthetic_reason_removed_count"
                ]
                + contract[
                    "contradicted_synthetic_reason_remaining_count"
                ]
            ),
        }
    )
    return dict(sorted(checks.items()))


def _verify_population(
    input_dir: Path | str,
    layout_manifest_path: Path | str,
    manifest_sha256: str,
    input_tree_sha256: str,
) -> FrozenLayoutManifest:
    snapshot = _strict_freezer_manifest(
        layout_manifest_path,
        expected_sha256=manifest_sha256,
    )
    observed = _verify_input_tree_and_recomputed_manifest(
        Path(input_dir).resolve(strict=True),
        snapshot,
        expected_sha256=input_tree_sha256,
    )
    if observed != input_tree_sha256:
        raise GroupedPolicyEvidenceBuildError(
            "recomputed input population digest changed"
        )
    return snapshot.manifest


def _git_bound_tool_sha256(
    repository_root: Path,
    candidate_revision: str,
    relative_path: str,
) -> str:
    disk_path = repository_root / relative_path
    disk_raw = _read_regular_bytes(
        disk_path.resolve(), label=f"tool {relative_path}"
    )
    committed = _git_blob(
        repository_root, candidate_revision, relative_path
    )
    if disk_raw != committed:
        raise GroupedPolicyEvidenceBuildError(
            f"tool bytes differ from candidate commit: {relative_path}"
        )
    return _sha256_bytes(committed)


def _require_evidence_archive_binding(candidate_revision: str) -> None:
    """Prove this process is executing the complete exact candidate archive."""

    contract = _EVIDENCE_ARCHIVE_CONTRACT
    if contract is None:
        raise GroupedPolicyEvidenceBuildError(
            "evidence builder must run from the verified candidate archive"
        )
    try:
        source_root = Path(
            str(contract["source_root"])
        ).resolve(strict=True)
        origin_root = Path(
            str(contract["origin_root"])
        ).resolve(strict=True)
        archive = _early_evidence_git(
            origin_root,
            "archive",
            "--format=tar",
            candidate_revision,
        )
        archive_sha = _sha256_bytes(archive)
        archive_tree_sha = _early_evidence_archive_tree_sha256(archive)
        source_tree_sha = _early_evidence_tree_sha256(source_root)
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        raise GroupedPolicyEvidenceBuildError(
            "candidate evidence archive binding cannot be reproduced"
        ) from exc
    if (
        REPO_ROOT.resolve() != source_root
        or GIT_AUTHORITY_ROOT.resolve() != origin_root
        or source_root == origin_root
        or contract["revision"] != candidate_revision
        or contract["archive_sha256"] != archive_sha
        or contract["tree_sha256"] != archive_tree_sha
        or source_tree_sha != archive_tree_sha
    ):
        raise GroupedPolicyEvidenceBuildError(
            "candidate evidence archive binding failed"
        )


def _verify_capture_archive_binding(
    repository_root: Path,
    *,
    revision: str,
    observation: Mapping[str, Any],
) -> None:
    raw = _git_bytes(
        repository_root,
        "archive",
        "--format=tar",
        revision,
    )
    try:
        tree_sha = _early_archive_tree_sha256(raw)
    except (OSError, RuntimeError) as exc:
        raise GroupedPolicyEvidenceBuildError(
            "capture source archive cannot be independently reproduced"
        ) from exc
    if (
        observation["source_archive_sha256"] != _sha256_bytes(raw)
        or observation["source_tree_sha256"] != tree_sha
        or observation["sandbox_backend_sha256"]
        != _sha256_bytes(_SANDBOX_BACKEND.encode("utf-8"))
    ):
        raise GroupedPolicyEvidenceBuildError(
            "capture observation does not bind the exact filtered source archive"
        )


def _full_regression_modules(source_root: Path) -> tuple[str, ...]:
    """Return the complete test-module set without recursive evidence builds."""

    modules = tuple(
        "tests." + path.stem
        for path in sorted((source_root / "tests").glob("test_*.py"))
        if path.stem != "test_grouped_policy_revalidation_evidence"
    )
    if not modules:
        raise GroupedPolicyEvidenceBuildError(
            "full regression suite contains no test modules"
        )
    return modules


def _run_regression_suites() -> dict[str, int]:
    """Run the pinned source-tree suites; callers cannot supply their counts."""

    suites = {
        **_REGRESSION_SUITE_MODULES,
        "full_failure": _full_regression_modules(REPO_ROOT),
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "DYLD_", "LD_", "GIT_"))
    }
    environment.update(
        {
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    counts: dict[str, int] = {}
    for name, modules in suites.items():
        try:
            completed = subprocess.run(
                (
                    sys.executable,
                    "-B",
                    "-m",
                    "unittest",
                    "-q",
                    *modules,
                ),
                cwd=REPO_ROOT,
                env=environment,
                check=False,
                capture_output=True,
                timeout=3600,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GroupedPolicyEvidenceBuildError(
                f"{name} regression suite could not complete"
            ) from exc
        counts[name] = 0 if completed.returncode == 0 else 1
    return counts


def build_aggregate_evidence(
    *,
    input_dir: Path | str,
    layout_manifest_path: Path | str,
    truth_path: Path | str,
    evaluator_path: Path | str,
    experiment_plan_path: Path | str,
    expected_experiment_plan_sha256: str,
    runtime_contract_path: Path | str,
    expected_runtime_contract_sha256: str,
    hypothesis_path: Path | str,
    primary_variable_path: Path | str,
    control_revision_sha: str,
    candidate_revision_sha: str,
    control_prediction_paths: Sequence[Path | str],
    candidate_prediction_paths: Sequence[Path | str],
    control_audit_paths: Sequence[Path | str],
    candidate_audit_paths: Sequence[Path | str],
    control_observation_path: Path | str,
    candidate_observation_path: Path | str,
    topology_verifier: Callable[..., GovernedExperimentTopology] = (
        verify_governed_experiment_topology
    ),
    population_verifier: Callable[
        [Path | str, Path | str, str, str], FrozenLayoutManifest
    ] = _verify_population,
) -> dict[str, Any]:
    """Build one source-bound, identity-free 3x5 gate decision."""

    control_revision = _commit(
        control_revision_sha, label="control_revision_sha"
    )
    candidate_revision = _commit(
        candidate_revision_sha, label="candidate_revision_sha"
    )
    _require_evidence_archive_binding(candidate_revision)
    _require_exact_candidate_checkout(
        GIT_AUTHORITY_ROOT, candidate_revision
    )
    expected_plan_sha = _digest(
        expected_experiment_plan_sha256,
        label="expected_experiment_plan_sha256",
    )
    expected_runtime_sha = _digest(
        expected_runtime_contract_sha256,
        label="expected_runtime_contract_sha256",
    )
    external_files = (
        layout_manifest_path,
        truth_path,
        experiment_plan_path,
        runtime_contract_path,
        hypothesis_path,
        primary_variable_path,
        *control_prediction_paths,
        *candidate_prediction_paths,
        *control_audit_paths,
        *candidate_audit_paths,
        control_observation_path,
        candidate_observation_path,
    )
    _require_external(external_files, label="experiment")
    input_root = Path(input_dir).resolve(strict=True)
    for repository_root in {
        REPO_ROOT.resolve(),
        GIT_AUTHORITY_ROOT.resolve(),
    }:
        try:
            input_root.relative_to(repository_root)
        except ValueError:
            continue
        raise GroupedPolicyEvidenceBuildError(
            "identity-bearing input PDFs must remain external"
        )

    plan, plan_raw = _canonical_object(
        experiment_plan_path, label="experiment plan"
    )
    if _sha256_bytes(plan_raw) != expected_plan_sha:
        raise GroupedPolicyEvidenceBuildError(
            "experiment plan bytes do not match the expected digest"
        )
    normalized_plan = _normalize_experiment_plan(plan)
    if (
        normalized_plan["evidence_label"] != PUBLIC_EVIDENCE_LABEL
        or normalized_plan["expected_record_count"]
        != REQUIRED_PUBLIC_RECORDS
    ):
        raise GroupedPolicyEvidenceBuildError(
            "plan is not the full public grouped WO-17 experiment"
        )
    manifest_sha = _digest(
        normalized_plan["split_manifest_sha256"],
        label="plan.split_manifest_sha256",
    )
    input_tree_sha = _digest(
        normalized_plan["input_tree_sha256"],
        label="plan.input_tree_sha256",
    )
    truth_sha = _digest(
        normalized_plan["truth_sha256"],
        label="plan.truth_sha256",
    )
    evaluator_sha = _digest(
        normalized_plan["evaluator_sha256"],
        label="plan.evaluator_sha256",
    )
    runtime_sha = _digest(
        normalized_plan["runtime_contract_sha256"],
        label="plan.runtime_contract_sha256",
    )
    if runtime_sha != expected_runtime_sha:
        raise GroupedPolicyEvidenceBuildError(
            "runtime digest expectation disagrees with the plan"
        )
    if _sha256_file(layout_manifest_path) != manifest_sha:
        raise GroupedPolicyEvidenceBuildError(
            "layout manifest bytes disagree with the plan"
        )
    expected_evaluator_path = (REPO_ROOT / "scripts" / "evaluate.py").resolve(
        strict=True
    )
    requested_evaluator_path = Path(evaluator_path)
    if (
        not requested_evaluator_path.is_absolute()
        or requested_evaluator_path.resolve(strict=True)
        != expected_evaluator_path
        or requested_evaluator_path != expected_evaluator_path
    ):
        raise GroupedPolicyEvidenceBuildError(
            "scored evaluator is not the repository official evaluator"
        )
    evaluator_raw = _read_regular_bytes(
        requested_evaluator_path, label="official evaluator"
    )
    if _sha256_bytes(evaluator_raw) != evaluator_sha:
        raise GroupedPolicyEvidenceBuildError(
            "evaluator bytes disagree with the plan"
        )
    bound_evaluator = _load_bound_evaluator(
        evaluator_raw, source_path=expected_evaluator_path
    )
    if _sha256_file(hypothesis_path) != normalized_plan["hypothesis_sha256"]:
        raise GroupedPolicyEvidenceBuildError(
            "hypothesis bytes disagree with the plan"
        )
    if (
        _sha256_file(primary_variable_path)
        != normalized_plan["primary_variable_sha256"]
    ):
        raise GroupedPolicyEvidenceBuildError(
            "primary-variable bytes disagree with the plan"
        )
    runtime, runtime_raw = _canonical_object(
        runtime_contract_path, label="runtime contract"
    )
    if _sha256_bytes(runtime_raw) != runtime_sha:
        raise GroupedPolicyEvidenceBuildError(
            "runtime contract bytes disagree with the plan"
        )
    validated_runtime = _validate_runtime_contract(
        runtime,
        evaluator_sha256=evaluator_sha,
        input_tree_sha256=input_tree_sha,
        manifest_sha256=manifest_sha,
        truth_sha256=truth_sha,
    )
    manifest = population_verifier(
        input_root,
        layout_manifest_path,
        manifest_sha,
        input_tree_sha,
    )
    if (
        manifest.sha256 != manifest_sha
        or len(manifest.case_ids) != REQUIRED_PUBLIC_RECORDS
        or len(manifest.groups) < REQUIRED_FOLDS
    ):
        raise GroupedPolicyEvidenceBuildError(
            "recomputed manifest is not the exact 1,000-case population"
        )

    topology = topology_verifier(
        GIT_AUTHORITY_ROOT,
        plan=normalized_plan,
        experiment_plan_sha256=expected_plan_sha,
        control_revision_sha=control_revision,
        candidate_revision_sha=candidate_revision,
    )
    fresh_control = _run_fresh_truth_blind_capture(
        arm="baseline",
        revision=control_revision,
        input_dir=input_root,
        layout_manifest_path=Path(layout_manifest_path),
        manifest_sha256=manifest_sha,
        input_tree_sha256=input_tree_sha,
        truth_path=Path(truth_path),
    )
    fresh_candidate = _run_fresh_truth_blind_capture(
        arm="candidate",
        revision=candidate_revision,
        input_dir=input_root,
        layout_manifest_path=Path(layout_manifest_path),
        manifest_sha256=manifest_sha,
        input_tree_sha256=input_tree_sha,
        truth_path=Path(truth_path),
    )
    # Truth is opened only after both fresh OS-sandboxed captures are frozen.
    truth_raw = _read_regular_bytes(truth_path, label="truth")
    if _sha256_bytes(truth_raw) != truth_sha:
        raise GroupedPolicyEvidenceBuildError(
            "truth bytes disagree with the plan"
        )
    truth = _truth_rows_from_bytes(
        truth_raw,
        case_ids=manifest.case_ids,
    )
    control = _load_arm(
        evaluator=bound_evaluator,
        arm="baseline",
        revision=control_revision,
        prediction_paths=control_prediction_paths,
        audit_paths=control_audit_paths,
        observation_path=control_observation_path,
        manifest_sha256=manifest_sha,
        input_tree_sha256=input_tree_sha,
        runtime=validated_runtime,
        truth=truth,
        fresh_capture=fresh_control,
    )
    candidate = _load_arm(
        evaluator=bound_evaluator,
        arm="candidate",
        revision=candidate_revision,
        prediction_paths=candidate_prediction_paths,
        audit_paths=candidate_audit_paths,
        observation_path=candidate_observation_path,
        manifest_sha256=manifest_sha,
        input_tree_sha256=input_tree_sha,
        runtime=validated_runtime,
        truth=truth,
        fresh_capture=fresh_candidate,
    )
    _verify_capture_archive_binding(
        GIT_AUTHORITY_ROOT,
        revision=control_revision,
        observation=control.observation,
    )
    _verify_capture_archive_binding(
        GIT_AUTHORITY_ROOT,
        revision=candidate_revision,
        observation=candidate.observation,
    )
    capture_tool_sha = _git_bound_tool_sha256(
        GIT_AUTHORITY_ROOT,
        candidate_revision,
        "devtools/policy_grouped_capture.py",
    )
    evidence_tool_sha = _git_bound_tool_sha256(
        GIT_AUTHORITY_ROOT,
        candidate_revision,
        "devtools/grouped_policy_revalidation_evidence.py",
    )
    gate_tool_sha = _git_bound_tool_sha256(
        GIT_AUTHORITY_ROOT,
        candidate_revision,
        "devtools/grouped_policy_revalidation_gate.py",
    )
    if (
        control.observation["capture_tool_sha256"] != capture_tool_sha
        or candidate.observation["capture_tool_sha256"] != capture_tool_sha
    ):
        raise GroupedPolicyEvidenceBuildError(
            "capture observations do not bind the committed capture tool"
        )
    if control.observation["producer_graph_sha256"] == candidate.observation[
        "producer_graph_sha256"
    ]:
        raise GroupedPolicyEvidenceBuildError(
            "candidate production graph did not change"
        )

    folds, group_exclusive, paired, split_deterministic = _folds(
        bound_evaluator,
        manifest,
        truth,
        control.rows,
        candidate.rows,
    )
    non_decision_changes, decision_changes = _row_change_counts(
        bound_evaluator,
        control.rows,
        candidate.rows,
        manifest.case_ids,
    )
    control_false, control_catastrophic = _false_approval_sets(
        bound_evaluator, truth, control.rows
    )
    candidate_false, candidate_catastrophic = _false_approval_sets(
        bound_evaluator, truth, candidate.rows
    )
    candidate_counts = candidate.audit["counts"]
    activity = PolicyActivityAggregate(
        eligible_guarded_initial_count=candidate_counts[
            "matcher_eligible_guarded_initial_count"
        ],
        guarded_initial_approval_count=candidate_counts[
            "matcher_guarded_initial_approval_count"
        ],
        unguarded_initial_approval_count=candidate_counts[
            "matcher_unguarded_initial_approval_count"
        ],
        late_revalidation_approval_count=candidate_counts[
            "matcher_late_revalidation_approval_count"
        ],
        legacy_forced_approval_count=candidate_counts[
            "policy_forced_approval_count"
        ],
    )
    expected_regressions = {
        "adversarial_failure": 0,
        "focused_failure": 0,
        "full_failure": 0,
    }
    regression_counts = _run_regression_suites()
    if set(regression_counts) != set(expected_regressions):
        raise GroupedPolicyEvidenceBuildError(
            "regression runner returned an inexact suite result set"
        )
    normalized_regressions = {
        name: _count(value, label=f"regression_counts.{name}")
        for name, value in regression_counts.items()
    }
    contract_checks = _capture_contract_checks(candidate.audit)
    limit = validated_runtime["container_limits"]
    runtime_limit_ok = all(
        observation["metrics"][f"{repeat}_output_bytes"]
        <= limit["output_bytes"]
        and observation["metrics"][f"{repeat}_peak_rss_bytes"]
        <= limit["peak_memory_bytes"]
        and observation["metrics"][f"{repeat}_runtime_seconds"]
        <= limit["runtime_seconds"]
        and observation["metrics"][f"{repeat}_runtime_seconds"]
        / REQUIRED_PUBLIC_RECORDS
        <= limit["per_record_runtime_seconds"]
        for observation in (control.observation, candidate.observation)
        for repeat in ("first", "second")
    )
    contract_checks["runtime_limits_satisfied"] = runtime_limit_ok

    evidence = GroupedPolicyEvidence(
        control_source_revision_sha=control_revision,
        candidate_source_revision_sha=candidate_revision,
        experiment_plan_sha256=expected_plan_sha,
        runtime_contract_sha256=runtime_sha,
        split_manifest_sha256=manifest_sha,
        input_tree_sha256=input_tree_sha,
        truth_sha256=truth_sha,
        evaluator_sha256=evaluator_sha,
        candidate_diff_manifest_sha256=(
            topology.candidate_source_diff_sha256
        ),
        expected_record_count=REQUIRED_PUBLIC_RECORDS,
        expected_layout_group_count=len(manifest.groups),
        control=control.aggregate,
        candidate=candidate.aggregate,
        activity=activity,
        folds=folds,
        new_false_approval_count=len(candidate_false - control_false),
        new_catastrophic_false_approval_count=len(
            candidate_catastrophic - control_catastrophic
        ),
        non_decision_field_change_count=non_decision_changes,
        decision_or_confidence_change_count=decision_changes,
        source_and_diff_bound=True,
        population_bound=True,
        group_exclusive=group_exclusive,
        paired_fold_members=paired,
        split_deterministic=split_deterministic,
        runtime_contract_bound=True,
        capture_contract_checks=contract_checks,
        regression_counts=normalized_regressions,
    )
    decision = GroupedPolicyRevalidationGate().evaluate(evidence)
    aggregate = decision.to_aggregate_evidence()
    aggregate.update(
        {
            "base_revision_sha": topology.base_revision_sha,
            "control_revision_sha": topology.control_revision_sha,
            "candidate_revision_sha": topology.candidate_revision_sha,
            "experiment_plan_record_sha256": (
                topology.experiment_plan_record_sha256
            ),
            "prereg_checkpoint_sha256": (
                topology.prereg_checkpoint_sha256
            ),
            "prereg_governance_diff_sha256": (
                topology.prereg_governance_diff_sha256
            ),
            "candidate_source_diff_sha256": (
                topology.candidate_source_diff_sha256
            ),
            "planned_scope_manifest_sha256": (
                topology.planned_scope_manifest_sha256
            ),
            "hypothesis_sha256": normalized_plan["hypothesis_sha256"],
            "primary_variable_sha256": (
                normalized_plan["primary_variable_sha256"]
            ),
            "capture_tool_sha256": capture_tool_sha,
            "evidence_tool_sha256": evidence_tool_sha,
            "gate_tool_sha256": gate_tool_sha,
            "control_capture_set_sha256": _artifact_set_sha256(
                (
                    control.prediction_set_sha256,
                    control.audit_set_sha256,
                    control.observation_sha256,
                )
            ),
            "candidate_capture_set_sha256": _artifact_set_sha256(
                (
                    candidate.prediction_set_sha256,
                    candidate.audit_set_sha256,
                    candidate.observation_sha256,
                )
            ),
            "control_producer_graph_sha256": control.observation[
                "producer_graph_sha256"
            ],
            "candidate_producer_graph_sha256": candidate.observation[
                "producer_graph_sha256"
            ],
            "confusion_counts": _complete_confusion(
                control.official, candidate.official
            ),
            "score_components": {
                "control_extraction_score": control.aggregate.extraction_score,
                "control_classification_score": (
                    control.aggregate.classification_score
                ),
                "control_calibration_score": (
                    control.aggregate.calibration_score
                ),
                "control_missing_penalty": (
                    control.aggregate.missing_penalty
                ),
                "control_total_score": control.aggregate.total_score,
                "candidate_extraction_score": (
                    candidate.aggregate.extraction_score
                ),
                "candidate_classification_score": (
                    candidate.aggregate.classification_score
                ),
                "candidate_calibration_score": (
                    candidate.aggregate.calibration_score
                ),
                "candidate_missing_penalty": (
                    candidate.aggregate.missing_penalty
                ),
                "candidate_total_score": candidate.aggregate.total_score,
            },
        }
    )
    validated = validate_aggregate_artifact(aggregate)
    _require_evidence_archive_binding(candidate_revision)
    _require_exact_candidate_checkout(
        GIT_AUTHORITY_ROOT, candidate_revision
    )
    return validated


def render_aggregate_markdown(aggregate: Mapping[str, Any]) -> str:
    """Render only the already-sanitized aggregate decision."""

    aggregate = validate_aggregate_artifact(aggregate)
    gates = aggregate["gate_results"]
    components = aggregate["score_components"]
    lines = [
        "# WO-17 full grouped policy revalidation",
        "",
        f"- Evidence class: `{aggregate['evaluation_mode']}`",
        f"- Status: **{str(aggregate['status']).upper()}**",
        f"- A / P / C revisions: `{aggregate['base_revision_sha']}` / "
        f"`{aggregate['control_revision_sha']}` / "
        f"`{aggregate['candidate_revision_sha']}`",
        f"- Plan / plan record: `{aggregate['experiment_plan_sha256']}` / "
        f"`{aggregate['experiment_plan_record_sha256']}`",
        f"- Runtime contract: `{aggregate['runtime_contract_sha256']}`",
        f"- Population: {aggregate['record_count']} records, "
        f"{aggregate['layout_group_count']} layout groups",
        "",
        "## Official evaluator",
        "",
        f"- Control: {float(components['control_total_score']):.9f}",
        f"- Candidate: {float(components['candidate_total_score']):.9f}",
        f"- Delta: {float(aggregate['score_delta']):+.9f}",
        f"- Candidate extraction / classification / calibration / missing: "
        f"{float(components['candidate_extraction_score']):.9f} / "
        f"{float(components['candidate_classification_score']):.9f} / "
        f"{float(components['candidate_calibration_score']):.9f} / "
        f"{float(components['candidate_missing_penalty']):.9f}",
        "",
        "## Repeated grouped robustness",
        "",
        "| Repeat | Weighted delta | Positive folds | Leave-best-out delta |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for repeat in range(1, REQUIRED_REPEATS + 1):
        metrics = aggregate["metrics"]
        lines.append(
            f"| {repeat} | "
            f"{float(metrics[f'repeat_{repeat}_weighted_score_delta']):+.9f} | "
            f"{int(metrics[f'repeat_{repeat}_positive_fold_count'])}/"
            f"{REQUIRED_FOLDS} | "
            f"{float(metrics[f'repeat_{repeat}_leave_best_fold_out_delta']):+.9f} |"
        )
    lines.extend(
        [
            "",
            "## Policy activity",
            "",
            f"- Eligible guarded initial cases: "
            f"{aggregate['counts']['eligible_guarded_initial_count']}",
            f"- Observed guarded approvals: "
            f"{aggregate['counts']['guarded_initial_approval_count']}",
            f"- Unguarded / late approvals: "
            f"{aggregate['counts']['unguarded_initial_approval_count']} / "
            f"{aggregate['counts']['late_revalidation_approval_count']}",
            f"- Explicit legacy `forced_approval_count`: "
            f"{aggregate['counts']['legacy_forced_approval_count']}",
            "",
            "## Hard gates",
            "",
            "| Gate | Result |",
            "| --- | :---: |",
        ]
    )
    lines.extend(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in sorted(gates.items())
    )
    lines.extend(
        [
            "",
            "> Public-label-exposed 1,000-case grouped robustness evidence; "
            "this is not an unseen holdout. No identities or paths are "
            "included in this report.",
            "",
        ]
    )
    return "\n".join(lines)


def write_aggregate_outputs(
    aggregate: Mapping[str, Any],
    *,
    output_json: Path | str,
    output_markdown: Path | str,
) -> None:
    """Create the canonical output pair once, with conflict detection."""

    aggregate = validate_aggregate_artifact(aggregate)
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    if (
        not json_path.is_absolute()
        or not markdown_path.is_absolute()
        or json_path == markdown_path
        or json_path.is_symlink()
        or markdown_path.is_symlink()
    ):
        raise GroupedPolicyEvidenceBuildError(
            "output paths must be distinct absolute non-symlink paths"
        )
    root = REPO_ROOT.resolve()
    for path in (json_path, markdown_path):
        parent = path.parent.resolve(strict=True)
        if parent != path.parent or not parent.is_dir():
            raise GroupedPolicyEvidenceBuildError(
                "output parents must be canonical existing directories"
            )
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GroupedPolicyEvidenceBuildError(
                "aggregate outputs must be committed inside the repository"
            ) from exc
    try:
        _write_output_pair(
            json_path,
            canonical_json(dict(aggregate)) + "\n",
            markdown_path,
            render_aggregate_markdown(aggregate),
        )
    except ExperimentControlError as exc:
        raise GroupedPolicyEvidenceBuildError(str(exc)) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build full 1,000-case aggregate WO-17 grouped evidence."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--experiment-plan", type=Path, required=True)
    parser.add_argument("--expected-experiment-plan-sha256", required=True)
    parser.add_argument("--runtime-contract", type=Path, required=True)
    parser.add_argument("--expected-runtime-contract-sha256", required=True)
    parser.add_argument("--hypothesis", type=Path, required=True)
    parser.add_argument("--primary-variable", type=Path, required=True)
    parser.add_argument("--control-revision-sha", required=True)
    parser.add_argument("--candidate-revision-sha", required=True)
    parser.add_argument(
        "--control-prediction", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--candidate-prediction", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--control-audit", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--candidate-audit", type=Path, action="append", required=True
    )
    parser.add_argument("--control-observation", type=Path, required=True)
    parser.add_argument("--candidate-observation", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        aggregate = build_aggregate_evidence(
            input_dir=arguments.input_dir,
            layout_manifest_path=arguments.layout_manifest,
            truth_path=arguments.truth,
            evaluator_path=arguments.evaluator,
            experiment_plan_path=arguments.experiment_plan,
            expected_experiment_plan_sha256=(
                arguments.expected_experiment_plan_sha256
            ),
            runtime_contract_path=arguments.runtime_contract,
            expected_runtime_contract_sha256=(
                arguments.expected_runtime_contract_sha256
            ),
            hypothesis_path=arguments.hypothesis,
            primary_variable_path=arguments.primary_variable,
            control_revision_sha=arguments.control_revision_sha,
            candidate_revision_sha=arguments.candidate_revision_sha,
            control_prediction_paths=arguments.control_prediction,
            candidate_prediction_paths=arguments.candidate_prediction,
            control_audit_paths=arguments.control_audit,
            candidate_audit_paths=arguments.candidate_audit,
            control_observation_path=arguments.control_observation,
            candidate_observation_path=arguments.candidate_observation,
        )
        write_aggregate_outputs(
            aggregate,
            output_json=arguments.output_json,
            output_markdown=arguments.output_markdown,
        )
    except (ExperimentControlError, OSError) as exc:
        print(
            f"grouped policy evidence error: {exc}",
            file=sys.stderr,
        )
        return 1
    print(
        "WO-17 full grouped evidence: "
        f"{aggregate['status']} "
        f"(delta {float(aggregate['score_delta']):+.9f})"
    )
    return 0 if aggregate["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
