#!/usr/bin/env python3
"""Authenticated checkpoint cutover and experiment preregistration.

The initial checkpoint cutover creates objects with the GitHub Git Database
API and publishes them with one GraphQL ``beforeOid`` compare-and-swap.
Experiment preregistration deliberately stops at a prepared local successor.
In both cases a remote GitHub branch, queried through the authenticated
``gh api`` transport, is the current checkpoint authority.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pwd
import re
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.experiment_control import (  # noqa: E402
    CanonicalHashChainStore,
    CheckpointAuthorityResolver,
    ExperimentControlError,
    ExperimentLedger,
    IntegrityError,
    ProgramIntegrityCheckpoint,
    PublishedCheckpointReference,
    RuntimeLeakageScanner,
    _normalize_experiment_plan,
    build_program_integrity_checkpoint,
    canonical_json,
)


POINTER_SCHEMA = "mib-program-current-checkpoint/v1"
CUTOVER_AUTHORITY_SCHEMA = "mib-program-cutover-authority/v1"
CUTOVER_RECOVERY_SCHEMA = "mib-github-cutover-recovery/v1"
CUTOVER_TRANSACTION_SCHEMA = "mib-github-cutover-transaction/v1"
CUTOVER_COMMIT_MESSAGE = "Publish governed checkpoint authority cutover"
CUTOVER_COMMIT_ACTOR = {
    "date": "2000-01-01T00:00:00Z",
    "email": "governed-publisher@users.noreply.github.com",
    "name": "Governed Checkpoint Publisher",
}
POINTER_RELATIVE_PATH = PurePosixPath(
    "evaluation/program/current_checkpoint.json"
)
CUTOVER_AUTHORITY_RELATIVE_PATH = PurePosixPath(
    "evaluation/program/cutover_authority.json"
)
CHECKPOINT_DIRECTORY_RELATIVE_PATH = PurePosixPath(
    "evaluation/program"
)
PROGRAM_RELATIVE_PATH = PurePosixPath("evaluation/program")
BASELINE_MANIFEST_RELATIVE_PATH = PurePosixPath(
    "evaluation/program/frozen_baseline_manifest.json"
)
LEDGER_FILENAMES = {
    "candidate_state_ledger": "candidate_state_ledger.jsonl",
    "experiment_ledger": "experiment_ledger.jsonl",
    "protected_access_ledger": "protected_access_ledger.jsonl",
    "taint_registry": "taint_registry.jsonl",
}
PROMOTION_POPULATION_KEYS = frozenset(
    {
        "evaluator_sha256",
        "expected_record_count",
        "input_tree_sha256",
        "runtime_contract_sha256",
        "split_manifest_sha256",
        "truth_sha256",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_GITHUB_REPOSITORY_RE = re.compile(
    r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"
)
_TRUSTED_GH_CANDIDATES = (
    Path("/opt/homebrew/bin/gh"),
    Path("/usr/local/bin/gh"),
    Path("/usr/bin/gh"),
)


class GovernedExperimentCLIError(ExperimentControlError):
    """The authenticated governed-experiment workflow failed closed."""


class GitHubBranchCASRejected(GovernedExperimentCLIError):
    """The authenticated branch was observed at a non-CAS commit."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(dict(value)) + "\n").encode("utf-8")


def _decode_canonical_object(
    raw: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GovernedExperimentCLIError(
            f"{label} is not valid UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise GovernedExperimentCLIError(f"{label} must be a JSON object")
    if raw != _canonical_bytes(value):
        raise GovernedExperimentCLIError(
            f"{label} must be canonical JSON with one trailing newline"
        )
    return value


def _read_regular_file(path: Path, *, label: str) -> bytes:
    try:
        canonical_parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise GovernedExperimentCLIError(
            f"{label} has an unreadable parent"
        ) from exc
    for parent in (canonical_parent, *canonical_parent.parents):
        try:
            parent_metadata = parent.lstat()
        except OSError as exc:
            raise GovernedExperimentCLIError(
                f"{label} has an unreadable parent"
            ) from exc
        if (
            stat.S_ISLNK(parent_metadata.st_mode)
            or not stat.S_ISDIR(parent_metadata.st_mode)
        ):
            raise GovernedExperimentCLIError(
                f"{label} has a symlink or non-directory parent"
            )
    path = canonical_parent / path.name
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GovernedExperimentCLIError(f"{label} is unreadable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise GovernedExperimentCLIError(
            f"{label} must be one regular, non-symlink file"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GovernedExperimentCLIError(
            f"{label} cannot be opened safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        current = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or metadata.st_dev != opened.st_dev
            or metadata.st_ino != opened.st_ino
            or opened.st_dev != current.st_dev
            or opened.st_ino != current.st_ino
            or (
                hasattr(os, "geteuid")
                and opened.st_uid != os.geteuid()
            )
        ):
            raise GovernedExperimentCLIError(
                f"{label} changed while it was opened"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final_descriptor = os.fstat(descriptor)
        final_path = path.lstat()
        identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if identity != (
            final_descriptor.st_dev,
            final_descriptor.st_ino,
            final_descriptor.st_size,
            final_descriptor.st_mtime_ns,
            final_descriptor.st_ctime_ns,
        ) or identity[:2] != (
            final_path.st_dev,
            final_path.st_ino,
        ):
            raise GovernedExperimentCLIError(
                f"{label} changed while it was read"
            )
        raw = b"".join(chunks)
        if len(raw) != final_descriptor.st_size:
            raise GovernedExperimentCLIError(
                f"{label} changed while it was read"
            )
        return raw
    finally:
        os.close(descriptor)


def _require_repository_path(
    repository_root: Path,
    relative_path: PurePosixPath | str,
    *,
    label: str,
) -> Path:
    raw = str(relative_path)
    pure = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or pure.is_absolute()
        or str(pure) != raw
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise GovernedExperimentCLIError(
            f"{label} must be a normalized repository-relative path"
        )
    root = repository_root.resolve()
    candidate = root.joinpath(*pure.parts)
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise GovernedExperimentCLIError(
            f"{label} escapes the repository"
        ) from exc
    current = root
    for part in pure.parts[:-1]:
        current = current / part
        if not current.exists():
            break
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise GovernedExperimentCLIError(
                f"{label} has an unreadable parent"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise GovernedExperimentCLIError(
                f"{label} has a symlink or non-directory parent"
            )
    return candidate


def _require_external_canonical_object(
    path: Path | str,
    *,
    repository_root: Path,
    label: str,
) -> dict[str, Any]:
    external = Path(path)
    try:
        supplied_metadata = external.lstat()
    except OSError as exc:
        raise GovernedExperimentCLIError(f"{label} is unreadable") from exc
    if stat.S_ISLNK(supplied_metadata.st_mode):
        raise GovernedExperimentCLIError(
            f"{label} may not be supplied through a symlink"
        )
    try:
        resolved = external.parent.resolve(strict=True) / external.name
    except OSError as exc:
        raise GovernedExperimentCLIError(f"{label} is unreadable") from exc
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError:
        pass
    else:
        raise GovernedExperimentCLIError(
            f"{label} must remain outside the repository"
        )
    return _decode_canonical_object(
        _read_regular_file(resolved, label=label),
        label=label,
    )


def _run_command(
    arguments: Sequence[str],
    *,
    cwd: Path,
    input_text: str | None = None,
) -> str:
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
    try:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            input=input_text,
        )
    except OSError as exc:
        raise GovernedExperimentCLIError(
            f"required command is unavailable: {arguments[0]}"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise GovernedExperimentCLIError(
            f"{arguments[0]} command failed"
            + (f": {detail}" if detail else "")
        )
    return completed.stdout


def _git(
    repository_root: Path,
    *arguments: str,
) -> str:
    return _run_command(
        (
            "/usr/bin/git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.ignoreStat=false",
            *arguments,
        ),
        cwd=repository_root,
    ).strip()


def _git_raw(
    repository_root: Path,
    *arguments: str,
) -> bytes:
    return _run_command(
        (
            "/usr/bin/git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.untrackedCache=false",
            "-c",
            "core.ignoreStat=false",
            *arguments,
        ),
        cwd=repository_root,
    ).encode("utf-8")


def _trusted_gh_executable() -> str:
    for candidate in _TRUSTED_GH_CANDIDATES:
        try:
            resolved = candidate.resolve(strict=True)
            metadata = resolved.stat()
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode) and os.access(
            resolved, os.X_OK
        ):
            return str(resolved)
    # Keep the invocation absolute and let subprocess report unavailability.
    return "/usr/bin/gh"


def _gh_json(
    repository_root: Path,
    endpoint: str,
    *,
    fields: Mapping[str, str] | None = None,
    method: str = "GET",
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if method not in {"GET", "POST", "PATCH"}:
        raise GovernedExperimentCLIError(
            "unsupported authenticated GitHub API method"
        )
    if payload is not None and (fields or method == "GET"):
        raise GovernedExperimentCLIError(
            "GitHub JSON payload is only valid for a write request"
        )
    arguments = [
        _trusted_gh_executable(),
        "api",
        "--hostname",
        "github.com",
        "--method",
        method,
        endpoint,
    ]
    input_text = None
    if payload is None:
        for name, value in sorted((fields or {}).items()):
            arguments.extend(("-f", f"{name}={value}"))
    else:
        arguments.extend(("--input", "-"))
        input_text = canonical_json(dict(payload))
    raw = _run_command(
        arguments,
        cwd=repository_root,
        input_text=input_text,
    )
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GovernedExperimentCLIError(
            "gh api returned invalid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise GovernedExperimentCLIError(
            "gh api response must be a JSON object"
        )
    return value


def _decode_github_content(
    response: Mapping[str, Any],
    *,
    label: str,
) -> bytes:
    if response.get("type") != "file":
        raise GovernedExperimentCLIError(
            f"remote {label} is not a regular file"
        )
    if response.get("encoding") != "base64":
        raise GovernedExperimentCLIError(
            f"remote {label} is not base64 encoded"
        )
    content = response.get("content")
    if not isinstance(content, str):
        raise GovernedExperimentCLIError(
            f"remote {label} has no content"
        )
    try:
        encoded = "".join(content.split())
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise GovernedExperimentCLIError(
            f"remote {label} has invalid base64 content"
        ) from exc


def _repository_head_safety(
    repository_root: Path,
    *,
    branch: str,
) -> str:
    root = repository_root.resolve()
    discovered = Path(
        _git(root, "rev-parse", "--show-toplevel")
    ).resolve()
    if discovered != root:
        raise GovernedExperimentCLIError(
            "repository root does not match the active Git worktree"
        )
    if _git(root, "rev-parse", "--is-shallow-repository") != "false":
        raise GovernedExperimentCLIError(
            "shallow Git history cannot establish checkpoint ancestry"
        )
    if _git(
        root,
        "for-each-ref",
        "--format=%(refname)",
        "refs/replace",
    ):
        raise GovernedExperimentCLIError(
            "Git replacement objects are forbidden"
        )
    common_directory = Path(
        _git(root, "rev-parse", "--git-common-dir")
    )
    if not common_directory.is_absolute():
        common_directory = root / common_directory
    grafts = common_directory.resolve() / "info/grafts"
    if grafts.exists() or grafts.is_symlink():
        if grafts.is_symlink() or grafts.stat().st_size:
            raise GovernedExperimentCLIError(
                "Git graft history is forbidden"
            )
    current_branch = _git(
        root,
        "symbolic-ref",
        "--quiet",
        "--short",
        "HEAD",
    )
    if current_branch != branch:
        raise GovernedExperimentCLIError(
            "local branch does not match the authority branch"
        )
    index_entries = _git(root, "ls-files", "-v", "-z")
    for entry in index_entries.split("\0"):
        if not entry:
            continue
        if len(entry) < 3 or entry[1] != " " or entry[0] != "H":
            raise GovernedExperimentCLIError(
                "tracked files may not use assume-unchanged, "
                "skip-worktree, or exceptional index flags"
            )
    head = _git(root, "rev-parse", "HEAD").lower()
    if not _COMMIT_RE.fullmatch(head):
        raise GovernedExperimentCLIError("local Git HEAD is invalid")
    return head


def _repository_state(
    repository_root: Path,
    *,
    branch: str,
) -> str:
    root = repository_root.resolve()
    head = _repository_head_safety(root, branch=branch)
    try:
        _git(
            root,
            "diff-index",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
        )
        _git(
            root,
            "diff-files",
            "--quiet",
            "--no-ext-diff",
            "--ignore-submodules=none",
            "--",
        )
    except GovernedExperimentCLIError as exc:
        raise GovernedExperimentCLIError(
            "tracked index and worktree bytes/modes must exactly match HEAD"
        ) from exc
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        raise GovernedExperimentCLIError(
            "worktree must be clean before checkpoint authority is resolved"
        )
    return head


def _github_repository_from_remote_url(value: str) -> str:
    raw = value.strip()
    path: str
    if raw.startswith("git@github.com:"):
        path = raw[len("git@github.com:") :]
    else:
        parsed = urlsplit(raw)
        if (
            parsed.scheme not in {"https", "ssh"}
            or parsed.hostname != "github.com"
            or parsed.port is not None
            or parsed.query
            or parsed.fragment
            or (
                parsed.scheme == "ssh"
                and parsed.username not in {None, "git"}
            )
            or (
                parsed.scheme == "https"
                and parsed.username is not None
            )
        ):
            raise GovernedExperimentCLIError(
                "authority upstream must be hosted on github.com"
            )
        path = parsed.path.lstrip("/")
    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if not _GITHUB_REPOSITORY_RE.fullmatch(path):
        raise GovernedExperimentCLIError(
            "authority upstream URL must identify one OWNER/REPOSITORY"
        )
    return path


def _require_authority_upstream(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
) -> str:
    if not _GITHUB_REPOSITORY_RE.fullmatch(github_repository):
        raise GovernedExperimentCLIError(
            "GitHub repository must be OWNER/REPOSITORY"
        )
    try:
        remote = _git(
            repository_root,
            "config",
            "--local",
            "--get",
            f"branch.{branch}.remote",
        )
        merge = _git(
            repository_root,
            "config",
            "--local",
            "--get",
            f"branch.{branch}.merge",
        )
        urls = _git(
            repository_root,
            "config",
            "--local",
            "--get-all",
            f"remote.{remote}.url",
        ).splitlines()
    except GovernedExperimentCLIError as exc:
        raise GovernedExperimentCLIError(
            "authority branch must have one configured GitHub upstream"
        ) from exc
    if (
        not remote
        or remote == "."
        or merge != f"refs/heads/{branch}"
        or len(urls) != 1
    ):
        raise GovernedExperimentCLIError(
            "authority branch must track its exact named upstream branch"
        )
    configured = _github_repository_from_remote_url(urls[0])
    if configured.casefold() != github_repository.casefold():
        raise GovernedExperimentCLIError(
            "GitHub authority repository does not match the branch upstream"
        )
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", remote)
        or ".." in remote.split("/")
    ):
        raise GovernedExperimentCLIError(
            "authority upstream remote name is unsafe"
        )
    return remote


def _require_exact_worktree_changes(
    repository_root: Path,
    *,
    expected_paths: set[str],
) -> None:
    status = _run_command(
        (
            "/usr/bin/git",
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        cwd=repository_root,
    ).rstrip("\n")
    observed: set[str] = set()
    for line in status.splitlines():
        if len(line) < 4:
            raise GovernedExperimentCLIError(
                "Git returned a malformed worktree status"
            )
        raw_path = line[3:]
        if " -> " in raw_path:
            raise GovernedExperimentCLIError(
                "renames are forbidden during a governed mutation"
            )
        observed.add(raw_path)
    if observed != expected_paths:
        raise GovernedExperimentCLIError(
            "worktree changed outside the exact governance publication set: "
            f"expected {sorted(expected_paths)}, observed {sorted(observed)}"
        )
    index_entries = _git(repository_root, "ls-files", "-v", "-z")
    if any(
        len(entry) < 3 or entry[1] != " " or entry[0] != "H"
        for entry in index_entries.split("\0")
        if entry
    ):
        raise GovernedExperimentCLIError(
            "governed transition introduced an exceptional Git index flag"
        )
    try:
        _git(
            repository_root,
            "diff-index",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
        )
    except GovernedExperimentCLIError as exc:
        raise GovernedExperimentCLIError(
            "governed transition may not stage any index change"
        ) from exc
    tracked_changes = {
        path
        for path in _git(
            repository_root,
            "diff-files",
            "--name-only",
            "-z",
            "--no-ext-diff",
            "--ignore-submodules=none",
            "--",
        ).split("\0")
        if path
    }
    if not tracked_changes.issubset(expected_paths):
        raise GovernedExperimentCLIError(
            "tracked candidate bytes/modes changed outside the exact "
            "governance publication set"
        )


def _remote_head(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
) -> str:
    response = _gh_json(
        repository_root,
        (
            f"repos/{github_repository}/branches/"
            f"{quote(branch, safe='')}"
        ),
    )
    if response.get("name") != branch:
        raise GovernedExperimentCLIError(
            "GitHub authority did not return the exact requested branch"
        )
    commit = response.get("commit")
    if not isinstance(commit, Mapping):
        raise GovernedExperimentCLIError(
            "GitHub authority returned an invalid branch object"
        )
    head = commit.get("sha")
    if not isinstance(head, str) or not _COMMIT_RE.fullmatch(
        head.casefold()
    ):
        raise GovernedExperimentCLIError(
            "GitHub authority returned an invalid branch HEAD"
        )
    return head.casefold()


def require_clean_remote_head(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
) -> str:
    _require_authority_upstream(
        repository_root,
        github_repository=github_repository,
        branch=branch,
    )
    local_head = _repository_state(repository_root, branch=branch)
    remote_head = _remote_head(
        repository_root,
        github_repository=github_repository,
        branch=branch,
    )
    if local_head != remote_head:
        raise GovernedExperimentCLIError(
            "local HEAD does not match authenticated GitHub branch HEAD"
        )
    return local_head


def _remote_file(
    repository_root: Path,
    *,
    github_repository: str,
    revision: str,
    relative_path: PurePosixPath,
    label: str,
) -> bytes:
    response = _gh_json(
        repository_root,
        (
            f"repos/{github_repository}/contents/"
            f"{quote(relative_path.as_posix(), safe='/')}"
        ),
        fields={"ref": revision},
    )
    response_path = response.get("path")
    if response_path != relative_path.as_posix():
        raise GovernedExperimentCLIError(
            f"remote {label} path does not match the request"
        )
    return _decode_github_content(response, label=label)


def _normalize_pointer(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "checkpoint_path",
        "checkpoint_sha256",
        "previous_checkpoint_sha256",
        "schema",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise GovernedExperimentCLIError(
            "current checkpoint pointer has an invalid schema"
        )
    if value["schema"] != POINTER_SCHEMA:
        raise GovernedExperimentCLIError(
            "current checkpoint pointer schema version is invalid"
        )
    digest = value["checkpoint_sha256"]
    previous = value["previous_checkpoint_sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise GovernedExperimentCLIError(
            "current checkpoint digest is invalid"
        )
    if previous is not None and (
        not isinstance(previous, str)
        or not _SHA256_RE.fullmatch(previous)
        or previous == digest
    ):
        raise GovernedExperimentCLIError(
            "previous checkpoint digest is invalid"
        )
    raw_path = value["checkpoint_path"]
    if not isinstance(raw_path, str):
        raise GovernedExperimentCLIError(
            "current checkpoint path is invalid"
        )
    expected_path = (
        CHECKPOINT_DIRECTORY_RELATIVE_PATH
        / f"{digest}.json"
    ).as_posix()
    if raw_path != expected_path:
        raise GovernedExperimentCLIError(
            "current checkpoint path is not the digest-addressed path"
        )
    normalized = {
        "checkpoint_path": expected_path,
        "checkpoint_sha256": digest,
        "previous_checkpoint_sha256": previous,
        "schema": POINTER_SCHEMA,
    }
    canonical_json(normalized)
    return normalized


def _pointer_for(
    checkpoint_sha256: str,
    *,
    previous_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    return _normalize_pointer(
        {
            "checkpoint_path": (
                CHECKPOINT_DIRECTORY_RELATIVE_PATH
                / f"{checkpoint_sha256}.json"
            ).as_posix(),
            "checkpoint_sha256": checkpoint_sha256,
            "previous_checkpoint_sha256": (
                previous_checkpoint_sha256
            ),
            "schema": POINTER_SCHEMA,
        }
    )


def _cutover_authority_for(
    *,
    parent_commit_sha: str,
    github_repository: str,
    branch: str,
) -> dict[str, Any]:
    value = {
        "branch": branch,
        "github_repository": github_repository,
        "parent_commit_sha": parent_commit_sha,
        "schema": CUTOVER_AUTHORITY_SCHEMA,
    }
    if (
        not _COMMIT_RE.fullmatch(parent_commit_sha)
        or not _GITHUB_REPOSITORY_RE.fullmatch(github_repository)
        or not branch
    ):
        raise GovernedExperimentCLIError(
            "cutover authority attestation is invalid"
        )
    canonical_json(value)
    return value


def _normalize_cutover_authority(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != {
        "branch",
        "github_repository",
        "parent_commit_sha",
        "schema",
    } or value.get("schema") != CUTOVER_AUTHORITY_SCHEMA:
        raise GovernedExperimentCLIError(
            "cutover authority attestation has an invalid schema"
        )
    return _cutover_authority_for(
        parent_commit_sha=value["parent_commit_sha"],
        github_repository=value["github_repository"],
        branch=value["branch"],
    )


def _program_paths(
    repository_root: Path,
) -> tuple[
    Path,
    Path,
    Path,
    dict[str, CanonicalHashChainStore],
]:
    program_root = _require_repository_path(
        repository_root,
        PROGRAM_RELATIVE_PATH,
        label="program root",
    )
    pointer_path = _require_repository_path(
        repository_root,
        POINTER_RELATIVE_PATH,
        label="current checkpoint pointer",
    )
    checkpoint_directory = _require_repository_path(
        repository_root,
        CHECKPOINT_DIRECTORY_RELATIVE_PATH,
        label="checkpoint directory",
    )
    stores = {
        name: CanonicalHashChainStore(program_root / filename)
        for name, filename in LEDGER_FILENAMES.items()
    }
    return program_root, pointer_path, checkpoint_directory, stores


def _ledger_bytes(
    stores: Mapping[str, CanonicalHashChainStore],
) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for name, store in stores.items():
        snapshot[name] = _read_regular_file(
            store.path,
            label=f"{name} ledger",
        )
        store.verify()
    return snapshot


def _require_unchanged_ledgers(
    before: Mapping[str, bytes],
    stores: Mapping[str, CanonicalHashChainStore],
) -> None:
    after = _ledger_bytes(stores)
    if dict(before) != after:
        raise IntegrityError(
            "cutover may not mutate any governed ledger"
        )


def _require_safe_directory(path: Path, *, parent: Path) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise GovernedExperimentCLIError(
            "checkpoint directory escapes its trusted root"
        ) from exc
    path.mkdir(mode=0o700, parents=False, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
        metadata.st_mode
    ):
        raise GovernedExperimentCLIError(
            "checkpoint directory must be a regular directory"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_once(
    path: Path,
    raw: bytes,
    *,
    label: str,
) -> bool:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        existing = _read_regular_file(path, label=label)
        if existing != raw:
            raise GovernedExperimentCLIError(
                f"{label} already exists with different bytes"
            )
        return False
    except OSError as exc:
        raise GovernedExperimentCLIError(
            f"{label} cannot be created safely"
        ) from exc
    created = os.fstat(descriptor)
    current = path.lstat()
    if (
        not stat.S_ISREG(created.st_mode)
        or created.st_nlink != 1
        or created.st_dev != current.st_dev
        or created.st_ino != current.st_ino
        or (
            hasattr(os, "geteuid")
            and created.st_uid != os.geteuid()
        )
    ):
        os.close(descriptor)
        raise GovernedExperimentCLIError(
            f"{label} was replaced while it was created"
        )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if _read_regular_file(path, label=label) != raw:
            raise GovernedExperimentCLIError(
                f"{label} changed while it was created"
            )
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            cleanup = path.lstat()
            if (
                cleanup.st_dev == created.st_dev
                and cleanup.st_ino == created.st_ino
            ):
                path.unlink()
                _fsync_directory(path.parent)
        except OSError:
            pass
        raise
    return True


def _unlink_exact(
    path: Path,
    *,
    expected: bytes,
    label: str,
) -> None:
    if _read_regular_file(path, label=label) != expected:
        raise IntegrityError(f"{label} changed before removal")
    path.unlink()
    _fsync_directory(path.parent)


def _atomic_replace_exact(
    path: Path,
    *,
    expected: bytes,
    replacement: bytes,
    label: str,
) -> None:
    current = _read_regular_file(path, label=label)
    if current != expected:
        raise GovernedExperimentCLIError(
            f"{label} changed before its atomic update"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        view = memoryview(replacement)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        if _read_regular_file(path, label=label) != expected:
            raise GovernedExperimentCLIError(
                f"{label} changed during its atomic update"
            )
        os.replace(temporary, path)
        try:
            _fsync_directory(path.parent)
        except BaseException as durability_error:
            if _read_regular_file(path, label=label) != replacement:
                raise IntegrityError(
                    f"{label} changed after its atomic replacement"
                ) from durability_error
            rollback_descriptor, rollback_name = tempfile.mkstemp(
                prefix=f".{path.name}.rollback.",
                dir=path.parent,
            )
            rollback = Path(rollback_name)
            try:
                os.chmod(rollback, 0o600)
                rollback_view = memoryview(expected)
                while rollback_view:
                    written = os.write(
                        rollback_descriptor,
                        rollback_view,
                    )
                    if written <= 0:
                        raise OSError("short rollback write")
                    rollback_view = rollback_view[written:]
                os.fsync(rollback_descriptor)
                os.close(rollback_descriptor)
                rollback_descriptor = -1
                os.replace(rollback, path)
                _fsync_directory(path.parent)
            except BaseException as rollback_error:
                raise IntegrityError(
                    f"{label} durability failed and rollback was incomplete"
                ) from rollback_error
            finally:
                if rollback_descriptor >= 0:
                    os.close(rollback_descriptor)
                if rollback.exists():
                    rollback.unlink()
            raise GovernedExperimentCLIError(
                f"{label} durability failed; original bytes were restored"
            ) from durability_error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _cutover_lock(program_root: Path):
    lock_path = program_root / ".program-integrity.lock"
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise GovernedExperimentCLIError(
            "program-integrity lock cannot be opened safely"
        ) from exc
    try:
        def require_same_owned_regular_lock() -> None:
            metadata = os.fstat(descriptor)
            try:
                current = lock_path.lstat()
            except OSError as exc:
                raise GovernedExperimentCLIError(
                    "program-integrity lock path was replaced"
                ) from exc
            if (
                not stat.S_ISREG(metadata.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or metadata.st_nlink != 1
                or current.st_nlink != 1
                or metadata.st_dev != current.st_dev
                or metadata.st_ino != current.st_ino
                or (
                    hasattr(os, "geteuid")
                    and metadata.st_uid != os.geteuid()
                )
            ):
                raise GovernedExperimentCLIError(
                    "program-integrity lock is not one owned regular file"
                )

        require_same_owned_regular_lock()
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except ImportError as exc:  # pragma: no cover
            raise GovernedExperimentCLIError(
                "program-integrity lock requires POSIX flock"
            ) from exc
        require_same_owned_regular_lock()
        try:
            yield
        finally:
            require_same_owned_regular_lock()
    finally:
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover
            pass
        os.close(descriptor)


def _runtime_leakage_finding_count(
    repository_root: Path,
) -> int:
    scanner = RuntimeLeakageScanner()
    targets = (
        repository_root / "solution.py",
        repository_root / "mib_pipeline",
    )
    return len(scanner.scan(targets))


def _normalize_promotion_population_file(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != PROMOTION_POPULATION_KEYS:
        raise GovernedExperimentCLIError(
            "promotion population must contain its exact six fields"
        )
    normalized: dict[str, Any] = {}
    for name in sorted(
        PROMOTION_POPULATION_KEYS - {"expected_record_count"}
    ):
        digest = value[name]
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(
            digest
        ):
            raise GovernedExperimentCLIError(
                f"promotion population {name} is invalid"
            )
        normalized[name] = digest
    count = value["expected_record_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise GovernedExperimentCLIError(
            "promotion population expected_record_count is invalid"
        )
    normalized["expected_record_count"] = count
    canonical_json(normalized)
    return normalized


def create_cutover(
    *,
    repository_root: Path | str,
    github_repository: str,
    branch: str,
    promotion_population_path: Path | str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    head = require_clean_remote_head(
        root,
        github_repository=github_repository,
        branch=branch,
    )
    (
        program_root,
        pointer_path,
        checkpoint_directory,
        stores,
    ) = _program_paths(root)
    if pointer_path.exists() or pointer_path.is_symlink():
        raise GovernedExperimentCLIError(
            "cutover requires an absent current checkpoint pointer"
        )
    promotion_population = _normalize_promotion_population_file(
        _require_external_canonical_object(
            promotion_population_path,
            repository_root=root,
            label="promotion population",
        )
    )
    baseline_path = _require_repository_path(
        root,
        BASELINE_MANIFEST_RELATIVE_PATH,
        label="frozen baseline manifest",
    )
    baseline_sha = _sha256_bytes(
        _read_regular_file(
            baseline_path,
            label="frozen baseline manifest",
        )
    )
    with _cutover_lock(program_root):
        if _repository_state(root, branch=branch) != head:
            raise GovernedExperimentCLIError(
                "local Git state changed during cutover"
            )
        if _remote_head(
            root,
            github_repository=github_repository,
            branch=branch,
        ) != head:
            raise GovernedExperimentCLIError(
                "authenticated GitHub branch moved during cutover"
            )
        before = _ledger_bytes(stores)
        _require_safe_directory(
            checkpoint_directory,
            parent=program_root,
        )
        raw, digest = build_program_integrity_checkpoint(
            stores=stores,
            baseline_manifest_sha256=baseline_sha,
            checkpoint_directory=checkpoint_directory,
            runtime_leakage_finding_count=(
                _runtime_leakage_finding_count(root)
            ),
            promotion_population=promotion_population,
        )
        pointer = _pointer_for(
            digest,
            previous_checkpoint_sha256=None,
        )
        pointer_raw = _canonical_bytes(pointer)
        authority_raw = _canonical_bytes(
            _cutover_authority_for(
                parent_commit_sha=head,
                github_repository=github_repository,
                branch=branch,
            )
        )
        authority_path = _require_repository_path(
            root,
            CUTOVER_AUTHORITY_RELATIVE_PATH,
            label="cutover authority attestation",
        )
        if authority_path.exists() or authority_path.is_symlink():
            raise GovernedExperimentCLIError(
                "cutover requires an absent authority attestation"
            )
        checkpoint_path = checkpoint_directory / f"{digest}.json"
        created_checkpoint = False
        created_pointer = False
        created_authority = False
        try:
            created_checkpoint = _create_once(
                checkpoint_path,
                raw,
                label="cutover checkpoint",
            )
            created_pointer = _create_once(
                pointer_path,
                pointer_raw,
                label="current checkpoint pointer",
            )
            created_authority = _create_once(
                authority_path,
                authority_raw,
                label="cutover authority attestation",
            )
            _require_unchanged_ledgers(before, stores)
            ProgramIntegrityCheckpoint(
                checkpoint_path,
                expected_sha256=digest,
                stores=stores,
            ).verify()
            _require_exact_worktree_changes(
                root,
                expected_paths={
                    pointer["checkpoint_path"],
                    POINTER_RELATIVE_PATH.as_posix(),
                    CUTOVER_AUTHORITY_RELATIVE_PATH.as_posix(),
                },
            )
            if _git(root, "rev-parse", "HEAD").casefold() != head:
                raise GovernedExperimentCLIError(
                    "local Git HEAD moved during cutover"
                )
            if _remote_head(
                root,
                github_repository=github_repository,
                branch=branch,
            ) != head:
                raise GovernedExperimentCLIError(
                    "authenticated GitHub branch moved during cutover"
                )
            if _git(root, "rev-parse", "HEAD").casefold() != head:
                raise GovernedExperimentCLIError(
                    "local Git HEAD moved during cutover"
                )
        except BaseException:
            if created_authority and authority_path.exists():
                _unlink_exact(
                    authority_path,
                    expected=authority_raw,
                    label="cutover authority rollback target",
                )
            if created_pointer and pointer_path.exists():
                _unlink_exact(
                    pointer_path,
                    expected=pointer_raw,
                    label="cutover pointer rollback target",
                )
            if created_checkpoint and checkpoint_path.exists():
                _unlink_exact(
                    checkpoint_path,
                    expected=raw,
                    label="cutover checkpoint rollback target",
                )
            raise
    return {
        "authority_path": CUTOVER_AUTHORITY_RELATIVE_PATH.as_posix(),
        "branch": branch,
        "checkpoint_path": pointer["checkpoint_path"],
        "checkpoint_sha256": digest,
        "github_repository": github_repository,
        "local_head": head,
        "next_required_action": (
            "publish these exact three prepared files with the authenticated "
            "GitHub compare-and-swap cutover publisher"
        ),
        "pointer_path": POINTER_RELATIVE_PATH.as_posix(),
        "status": "cutover_created_not_published",
    }


@dataclass(frozen=True)
class _PreparedCutover:
    parent_commit_sha: str
    pointer: dict[str, Any]
    checkpoint: ProgramIntegrityCheckpoint
    paths_to_bytes: dict[str, bytes]

    @property
    def changed_paths(self) -> set[str]:
        return set(self.paths_to_bytes)


def _git_blob_sha(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _load_prepared_cutover(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
) -> _PreparedCutover:
    (
        program_root,
        pointer_path,
        _checkpoint_directory,
        stores,
    ) = _program_paths(repository_root)
    pointer_raw = _read_regular_file(
        pointer_path,
        label="prepared current checkpoint pointer",
    )
    pointer = _normalize_pointer(
        _decode_canonical_object(
            pointer_raw,
            label="prepared current checkpoint pointer",
        )
    )
    if pointer["previous_checkpoint_sha256"] is not None:
        raise GovernedExperimentCLIError(
            "GitHub cutover publisher only accepts an initial checkpoint"
        )
    checkpoint_relative = PurePosixPath(pointer["checkpoint_path"])
    checkpoint_path = _require_repository_path(
        repository_root,
        checkpoint_relative,
        label="prepared cutover checkpoint",
    )
    try:
        checkpoint_path.resolve(strict=True).relative_to(
            program_root.resolve()
        )
    except (OSError, ValueError) as exc:
        raise GovernedExperimentCLIError(
            "prepared cutover checkpoint escapes its program root"
        ) from exc
    checkpoint_raw = _read_regular_file(
        checkpoint_path,
        label="prepared cutover checkpoint",
    )
    _decode_canonical_object(
        checkpoint_raw,
        label="prepared cutover checkpoint",
    )
    if _sha256_bytes(checkpoint_raw) != pointer["checkpoint_sha256"]:
        raise GovernedExperimentCLIError(
            "prepared checkpoint bytes do not match the pointer"
        )
    checkpoint = ProgramIntegrityCheckpoint(
        checkpoint_path,
        expected_sha256=pointer["checkpoint_sha256"],
        stores=stores,
    )
    checkpoint_value = checkpoint.verify()
    if "promotion_population" not in checkpoint_value:
        raise GovernedExperimentCLIError(
            "prepared cutover checkpoint has no promotion population"
        )
    authority_path = _require_repository_path(
        repository_root,
        CUTOVER_AUTHORITY_RELATIVE_PATH,
        label="prepared cutover authority attestation",
    )
    authority_raw = _read_regular_file(
        authority_path,
        label="prepared cutover authority attestation",
    )
    authority = _normalize_cutover_authority(
        _decode_canonical_object(
            authority_raw,
            label="prepared cutover authority attestation",
        )
    )
    if (
        authority["github_repository"].casefold()
        != github_repository.casefold()
        or authority["branch"] != branch
    ):
        raise GovernedExperimentCLIError(
            "prepared cutover authority names another GitHub branch"
        )
    paths_to_bytes = {
        POINTER_RELATIVE_PATH.as_posix(): pointer_raw,
        checkpoint_relative.as_posix(): checkpoint_raw,
        CUTOVER_AUTHORITY_RELATIVE_PATH.as_posix(): authority_raw,
    }
    if len(paths_to_bytes) != 3:
        raise IntegrityError(
            "prepared cutover does not contain exactly three paths"
        )
    return _PreparedCutover(
        parent_commit_sha=authority["parent_commit_sha"],
        pointer=pointer,
        checkpoint=checkpoint,
        paths_to_bytes=paths_to_bytes,
    )


def _git_database_ref(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
) -> str:
    response = _gh_json(
        repository_root,
        (
            f"repos/{github_repository}/git/ref/heads/"
            f"{quote(branch, safe='')}"
        ),
    )
    if response.get("ref") != f"refs/heads/{branch}":
        raise GovernedExperimentCLIError(
            "GitHub Git Database returned another branch ref"
        )
    target = response.get("object")
    if (
        not isinstance(target, Mapping)
        or target.get("type") != "commit"
        or not isinstance(target.get("sha"), str)
        or not _COMMIT_RE.fullmatch(target["sha"].casefold())
    ):
        raise GovernedExperimentCLIError(
            "GitHub branch ref does not identify one valid commit"
        )
    return target["sha"].casefold()


def _git_database_commit(
    repository_root: Path,
    *,
    github_repository: str,
    revision: str,
) -> dict[str, Any]:
    response = _gh_json(
        repository_root,
        f"repos/{github_repository}/git/commits/{revision}",
    )
    sha = response.get("sha")
    tree = response.get("tree")
    parents = response.get("parents")
    if (
        not isinstance(sha, str)
        or sha.casefold() != revision
        or not isinstance(tree, Mapping)
        or not isinstance(tree.get("sha"), str)
        or not _COMMIT_RE.fullmatch(tree["sha"].casefold())
        or not isinstance(parents, list)
        or any(
            not isinstance(parent, Mapping)
            or not isinstance(parent.get("sha"), str)
            or not _COMMIT_RE.fullmatch(parent["sha"].casefold())
            for parent in parents
        )
    ):
        raise GovernedExperimentCLIError(
            "GitHub Git Database returned an invalid commit object"
        )
    return {
        "message": response.get("message"),
        "parents": [
            parent["sha"].casefold()
            for parent in parents
        ],
        "sha": sha.casefold(),
        "tree_sha": tree["sha"].casefold(),
    }


def _git_database_tree(
    repository_root: Path,
    *,
    github_repository: str,
    tree_sha: str,
) -> dict[str, dict[str, str]]:
    response = _gh_json(
        repository_root,
        (
            f"repos/{github_repository}/git/trees/{tree_sha}"
            "?recursive=1"
        ),
    )
    response_sha = response.get("sha")
    if (
        not isinstance(response_sha, str)
        or response_sha.casefold() != tree_sha
        or response.get("truncated") is not False
        or not isinstance(response.get("tree"), list)
    ):
        raise GovernedExperimentCLIError(
            "GitHub returned an incomplete or mismatched recursive tree"
        )
    entries: dict[str, dict[str, str]] = {}
    for entry in response["tree"]:
        if not isinstance(entry, Mapping):
            raise GovernedExperimentCLIError(
                "GitHub recursive tree contains an invalid entry"
            )
        path = entry.get("path")
        mode = entry.get("mode")
        kind = entry.get("type")
        sha = entry.get("sha")
        if (
            not isinstance(path, str)
            or PurePosixPath(path).as_posix() != path
            or PurePosixPath(path).is_absolute()
            or any(
                part in {"", ".", ".."}
                for part in PurePosixPath(path).parts
            )
            or not isinstance(mode, str)
            or kind not in {"blob", "tree", "commit"}
            or not isinstance(sha, str)
            or not _COMMIT_RE.fullmatch(sha.casefold())
            or path in entries
        ):
            raise GovernedExperimentCLIError(
                "GitHub recursive tree contains an invalid entry"
            )
        entries[path] = {
            "mode": mode,
            "sha": sha.casefold(),
            "type": kind,
        }
    return entries


def _cutover_ancestor_paths(paths: Sequence[str]) -> set[str]:
    ancestors: set[str] = set()
    for raw in paths:
        pure = PurePosixPath(raw)
        for index in range(1, len(pure.parts)):
            ancestors.add(PurePosixPath(*pure.parts[:index]).as_posix())
    return ancestors


def _verify_remote_cutover(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    prepared: _PreparedCutover,
    published_commit_sha: str,
    require_branch_head: bool = True,
) -> dict[str, Any]:
    if require_branch_head and _git_database_ref(
        repository_root,
        github_repository=github_repository,
        branch=branch,
    ) != published_commit_sha:
        raise GovernedExperimentCLIError(
            "published cutover is not the authenticated branch HEAD"
        )
    parent = _git_database_commit(
        repository_root,
        github_repository=github_repository,
        revision=prepared.parent_commit_sha,
    )
    publication = _git_database_commit(
        repository_root,
        github_repository=github_repository,
        revision=published_commit_sha,
    )
    if (
        publication["parents"] != [prepared.parent_commit_sha]
        or publication["message"] != CUTOVER_COMMIT_MESSAGE
    ):
        raise GovernedExperimentCLIError(
            "published cutover is not the exact direct CAS child"
        )
    before = _git_database_tree(
        repository_root,
        github_repository=github_repository,
        tree_sha=parent["tree_sha"],
    )
    after = _git_database_tree(
        repository_root,
        github_repository=github_repository,
        tree_sha=publication["tree_sha"],
    )
    targets = prepared.changed_paths
    ancestors = _cutover_ancestor_paths(sorted(targets))
    expected_paths = set(before) | targets | ancestors
    if set(after) != expected_paths:
        raise GovernedExperimentCLIError(
            "published cutover tree has paths outside its exact topology"
        )
    for path, entry in before.items():
        if path in targets or path in ancestors:
            continue
        if after.get(path) != entry:
            raise GovernedExperimentCLIError(
                "published cutover rewrote a non-governance tree entry"
            )
    for path in ancestors:
        entry = after.get(path)
        if (
            entry is None
            or entry["type"] != "tree"
            or entry["mode"] != "040000"
        ):
            raise GovernedExperimentCLIError(
                "published cutover has an invalid ancestor topology"
            )
        before_entry = before.get(path)
        if before_entry is not None and (
            before_entry["type"] != "tree"
            or before_entry["mode"] != entry["mode"]
        ):
            raise GovernedExperimentCLIError(
                "published cutover changed an ancestor entry type or mode"
            )
    blob_shas = {
        path: _git_blob_sha(raw)
        for path, raw in prepared.paths_to_bytes.items()
    }
    for path, expected_raw in prepared.paths_to_bytes.items():
        entry = after.get(path)
        if entry != {
            "mode": "100644",
            "sha": blob_shas[path],
            "type": "blob",
        }:
            raise GovernedExperimentCLIError(
                "published cutover tree does not contain the exact blob"
            )
        response = _gh_json(
            repository_root,
            (
                f"repos/{github_repository}/contents/"
                f"{quote(path, safe='/')}"
            ),
            fields={"ref": published_commit_sha},
        )
        response_sha = response.get("sha")
        if (
            response.get("path") != path
            or not isinstance(response_sha, str)
            or response_sha.casefold() != blob_shas[path]
            or _decode_github_content(
                response,
                label=f"published cutover file {path}",
            )
            != expected_raw
        ):
            raise GovernedExperimentCLIError(
                "published cutover contents differ from prepared bytes"
            )
    return {
        "parent_commit_sha": prepared.parent_commit_sha,
        "published_commit_sha": published_commit_sha,
        "tree_sha": publication["tree_sha"],
    }


def _post_cutover_commit(
    repository_root: Path,
    *,
    github_repository: str,
    prepared: _PreparedCutover,
) -> tuple[str, str]:
    parent = _git_database_commit(
        repository_root,
        github_repository=github_repository,
        revision=prepared.parent_commit_sha,
    )
    before = _git_database_tree(
        repository_root,
        github_repository=github_repository,
        tree_sha=parent["tree_sha"],
    )
    if prepared.changed_paths & set(before):
        raise GovernedExperimentCLIError(
            "cutover parent already contains a prepared governance path"
        )
    blob_shas: dict[str, str] = {}
    for path in sorted(prepared.paths_to_bytes):
        raw = prepared.paths_to_bytes[path]
        expected = _git_blob_sha(raw)
        response = _gh_json(
            repository_root,
            f"repos/{github_repository}/git/blobs",
            method="POST",
            payload={
                "content": base64.b64encode(raw).decode("ascii"),
                "encoding": "base64",
            },
        )
        response_sha = response.get("sha")
        if (
            not isinstance(response_sha, str)
            or response_sha.casefold() != expected
        ):
            raise GovernedExperimentCLIError(
                "GitHub did not create the exact prepared blob"
            )
        blob_shas[path] = expected
    tree_response = _gh_json(
        repository_root,
        f"repos/{github_repository}/git/trees",
        method="POST",
        payload={
            "base_tree": parent["tree_sha"],
            "tree": [
                {
                    "mode": "100644",
                    "path": path,
                    "sha": blob_shas[path],
                    "type": "blob",
                }
                for path in sorted(blob_shas)
            ],
        },
    )
    tree_sha = tree_response.get("sha")
    if (
        not isinstance(tree_sha, str)
        or not _COMMIT_RE.fullmatch(tree_sha.casefold())
    ):
        raise GovernedExperimentCLIError(
            "GitHub did not create a valid cutover tree"
        )
    commit_response = _gh_json(
        repository_root,
        f"repos/{github_repository}/git/commits",
        method="POST",
        payload={
            "author": dict(CUTOVER_COMMIT_ACTOR),
            "committer": dict(CUTOVER_COMMIT_ACTOR),
            "message": CUTOVER_COMMIT_MESSAGE,
            "parents": [prepared.parent_commit_sha],
            "tree": tree_sha.casefold(),
        },
    )
    published = commit_response.get("sha")
    parents = commit_response.get("parents")
    response_tree = commit_response.get("tree")
    normalized_response_parents: list[str] | None = None
    if isinstance(parents, list) and all(
        isinstance(item, Mapping)
        and isinstance(item.get("sha"), str)
        for item in parents
    ):
        normalized_response_parents = [
            item["sha"].casefold()
            for item in parents
        ]
    response_tree_sha = (
        response_tree.get("sha")
        if isinstance(response_tree, Mapping)
        else None
    )
    if (
        not isinstance(published, str)
        or not _COMMIT_RE.fullmatch(published.casefold())
        or normalized_response_parents
        != [prepared.parent_commit_sha]
        or not isinstance(response_tree_sha, str)
        or response_tree_sha.casefold() != tree_sha.casefold()
    ):
        raise GovernedExperimentCLIError(
            "GitHub did not create the exact direct cutover commit"
        )
    return published.casefold(), tree_sha.casefold()


def _github_repository_node_id(
    repository_root: Path,
    *,
    github_repository: str,
) -> str:
    owner, name = github_repository.split("/", 1)
    response = _gh_json(
        repository_root,
        "graphql",
        method="POST",
        payload={
            "query": (
                "query CutoverRepositoryId($owner: String!, $name: String!) {"
                " repository(owner: $owner, name: $name) { id nameWithOwner }"
                " }"
            ),
            "variables": {
                "name": name,
                "owner": owner,
            },
        },
    )
    data = response.get("data")
    repository = (
        data.get("repository")
        if isinstance(data, Mapping)
        else None
    )
    node_id = (
        repository.get("id")
        if isinstance(repository, Mapping)
        else None
    )
    name_with_owner = (
        repository.get("nameWithOwner")
        if isinstance(repository, Mapping)
        else None
    )
    if (
        response.get("errors") not in (None, [])
        or not isinstance(node_id, str)
        or not node_id
        or node_id.strip() != node_id
        or not isinstance(name_with_owner, str)
        or name_with_owner.casefold() != github_repository.casefold()
    ):
        raise GovernedExperimentCLIError(
            "GitHub GraphQL did not identify the authority repository"
        )
    return node_id


def _compare_and_swap_cutover_ref(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    parent_commit_sha: str,
    published_commit_sha: str,
) -> None:
    client_mutation_id = (
        "governed-cutover-"
        + _sha256_bytes(
            (
                f"{github_repository.casefold()}\0{branch}\0"
                f"{parent_commit_sha}\0{published_commit_sha}"
            ).encode("utf-8")
        )
    )
    try:
        repository_id = _github_repository_node_id(
            repository_root,
            github_repository=github_repository,
        )
        response = _gh_json(
            repository_root,
            "graphql",
            method="POST",
            payload={
                "query": (
                    "mutation CutoverRefCAS($input: UpdateRefsInput!) {"
                    " updateRefs(input: $input) { clientMutationId }"
                    " }"
                ),
                "variables": {
                    "input": {
                        "clientMutationId": client_mutation_id,
                        "refUpdates": [
                            {
                                "afterOid": published_commit_sha,
                                "beforeOid": parent_commit_sha,
                                "force": False,
                                "name": f"refs/heads/{branch}",
                            }
                        ],
                        "repositoryId": repository_id,
                    }
                },
            },
        )
    except GovernedExperimentCLIError as exc:
        current = _git_database_ref(
            repository_root,
            github_repository=github_repository,
            branch=branch,
        )
        if current == published_commit_sha:
            return
        raise GitHubBranchCASRejected(
            "GitHub branch compare-and-swap rejected the cutover because "
            f"the branch moved from {parent_commit_sha}"
        ) from exc
    data = response.get("data")
    update_refs = (
        data.get("updateRefs")
        if isinstance(data, Mapping)
        else None
    )
    if (
        response.get("errors") not in (None, [])
        or not isinstance(update_refs, Mapping)
        or update_refs.get("clientMutationId") != client_mutation_id
    ):
        if _git_database_ref(
            repository_root,
            github_repository=github_repository,
            branch=branch,
        ) == published_commit_sha:
            return
        raise GitHubBranchCASRejected(
            "GitHub returned an invalid cutover GraphQL CAS response"
        )


def _local_status_paths(repository_root: Path) -> set[str]:
    raw = _git(
        repository_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    entries = raw.split("\0")
    paths: set[str] = set()
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2] != " ":
            raise GovernedExperimentCLIError(
                "Git returned a malformed NUL-delimited status"
            )
        status = entry[:2]
        path = entry[3:]
        if "R" in status or "C" in status:
            if index >= len(entries) or not entries[index]:
                raise GovernedExperimentCLIError(
                    "Git returned an incomplete rename status"
                )
            index += 1
            raise GovernedExperimentCLIError(
                "renames are forbidden during local cutover recovery"
            )
        paths.add(path)
    return paths


def _verify_local_cutover_commit(
    repository_root: Path,
    *,
    prepared: _PreparedCutover,
    published_commit_sha: str,
) -> None:
    lineage = _git(
        repository_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        published_commit_sha,
    ).split()
    if lineage != [
        published_commit_sha,
        prepared.parent_commit_sha,
    ]:
        raise GovernedExperimentCLIError(
            "fetched cutover commit has invalid local ancestry"
        )
    changed = {
        path
        for path in _git(
            repository_root,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            "-z",
            prepared.parent_commit_sha,
            published_commit_sha,
            "--",
        ).split("\0")
        if path
    }
    if changed != prepared.changed_paths:
        raise GovernedExperimentCLIError(
            "fetched cutover commit is not the exact three-file transition"
        )
    for path, raw in prepared.paths_to_bytes.items():
        if _git_blob(
            repository_root,
            revision=published_commit_sha,
            relative_path=path,
        ) != raw:
            raise GovernedExperimentCLIError(
                "fetched cutover commit has different governance bytes"
            )


def _ensure_local_cutover_commit(
    repository_root: Path,
    *,
    upstream_remote: str,
    prepared: _PreparedCutover,
    published_commit_sha: str,
) -> None:
    try:
        _git(
            repository_root,
            "cat-file",
            "-e",
            f"{published_commit_sha}^{{commit}}",
        )
    except GovernedExperimentCLIError:
        _git(
            repository_root,
            "fetch",
            "--no-tags",
            "--no-write-fetch-head",
            upstream_remote,
            published_commit_sha,
        )
    _verify_local_cutover_commit(
        repository_root,
        prepared=prepared,
        published_commit_sha=published_commit_sha,
    )


def _git_index_path(repository_root: Path) -> Path:
    git_directory_raw = _git(
        repository_root,
        "rev-parse",
        "--absolute-git-dir",
    )
    git_directory = Path(git_directory_raw)
    if not git_directory.is_absolute():
        raise GovernedExperimentCLIError(
            "local Git directory is not absolute"
        )
    try:
        git_directory = git_directory.resolve(strict=True)
        git_directory_metadata = git_directory.lstat()
    except OSError as exc:
        raise GovernedExperimentCLIError(
            "local Git directory is unreadable"
        ) from exc
    if not stat.S_ISDIR(git_directory_metadata.st_mode):
        raise GovernedExperimentCLIError(
            "local Git directory is not a directory"
        )

    raw = _git(repository_root, "rev-parse", "--git-path", "index")
    path = Path(raw)
    if not path.is_absolute():
        path = repository_root.resolve() / path
    # Normalize dots lexically without following the index leaf. Resolving the
    # leaf would turn a repository-local symlink into an external write target.
    path = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = path.relative_to(git_directory)
    except ValueError as exc:
        raise GovernedExperimentCLIError(
            "local Git index escapes the authenticated Git directory"
        ) from exc
    current = git_directory
    for part in relative.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise GovernedExperimentCLIError(
                "local Git index has an unreadable parent"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
            metadata.st_mode
        ):
            raise GovernedExperimentCLIError(
                "local Git index has a symlink or non-directory parent"
            )
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GovernedExperimentCLIError(
            "local Git index is unreadable"
        ) from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or (
            hasattr(os, "geteuid")
            and metadata.st_uid != os.geteuid()
        )
    ):
        raise GovernedExperimentCLIError(
            "local Git index must be one owned regular, non-symlink file"
        )
    return path


def _index_bytes_for_commit(
    repository_root: Path,
    *,
    revision: str,
    index_parent: Path,
) -> bytes:
    descriptor, name = tempfile.mkstemp(
        prefix=".governed-index.",
        dir=index_parent,
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    try:
        _git(
            repository_root,
            "read-tree",
            f"--index-output={temporary}",
            revision,
        )
        return _read_regular_file(
            temporary,
            label="prepared local cutover index",
        )
    finally:
        if temporary.exists():
            temporary.unlink()


def _owned_single_link_regular(metadata: os.stat_result) -> bool:
    return (
        stat.S_ISREG(metadata.st_mode)
        and metadata.st_nlink == 1
        and (
            not hasattr(os, "geteuid")
            or metadata.st_uid == os.geteuid()
        )
    )


def _detach_hardlinked_git_index(
    index_path: Path,
    *,
    compromised_descriptor: int,
    compromised_identity: tuple[int, int],
    replacement_index: bytes,
) -> None:
    """Replace a multiply-linked installed index with a fresh private inode."""

    repair_lock = index_path.with_name(f"{index_path.name}.lock")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(repair_lock, flags, 0o600)
    except OSError as exc:
        raise GovernedExperimentCLIError(
            "local Git index acquired an external hard-link alias and "
            "cannot be detached under a fresh Git lock"
        ) from exc
    repair_identity: tuple[int, int] | None = None
    installed = False
    try:
        opened = os.fstat(descriptor)
        current_lock = repair_lock.lstat()
        repair_identity = (opened.st_dev, opened.st_ino)
        if (
            not _owned_single_link_regular(opened)
            or not _owned_single_link_regular(current_lock)
            or (current_lock.st_dev, current_lock.st_ino)
            != repair_identity
        ):
            raise GovernedExperimentCLIError(
                "local Git index alias-repair lock is not one owned "
                "regular file"
            )
        view = memoryview(replacement_index)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short Git index alias-repair write")
            view = view[written:]
        os.fsync(descriptor)
        prepared = os.fstat(descriptor)
        current_lock = repair_lock.lstat()
        current_index = index_path.lstat()
        compromised = os.fstat(compromised_descriptor)
        if (
            not _owned_single_link_regular(prepared)
            or not _owned_single_link_regular(current_lock)
            or (prepared.st_dev, prepared.st_ino) != repair_identity
            or (current_lock.st_dev, current_lock.st_ino)
            != repair_identity
            or prepared.st_size != len(replacement_index)
            or (current_index.st_dev, current_index.st_ino)
            != compromised_identity
            or (compromised.st_dev, compromised.st_ino)
            != compromised_identity
        ):
            raise GovernedExperimentCLIError(
                "local Git index changed during hard-link alias repair"
            )
        os.replace(repair_lock, index_path)
        installed = True
        repaired_descriptor = os.fstat(descriptor)
        repaired_path = index_path.lstat()
        if (
            not _owned_single_link_regular(repaired_descriptor)
            or not _owned_single_link_regular(repaired_path)
            or (repaired_descriptor.st_dev, repaired_descriptor.st_ino)
            != repair_identity
            or (repaired_path.st_dev, repaired_path.st_ino)
            != repair_identity
            or repaired_descriptor.st_size != len(replacement_index)
        ):
            if (
                repaired_path.st_dev,
                repaired_path.st_ino,
            ) == repair_identity:
                index_path.unlink()
                _fsync_directory(index_path.parent)
            raise GovernedExperimentCLIError(
                "local Git index could not be safely detached from its "
                "external hard-link alias"
            )
        _fsync_directory(index_path.parent)
    finally:
        os.close(descriptor)
        if not installed:
            try:
                current = repair_lock.lstat()
                if repair_identity is not None and (
                    current.st_dev,
                    current.st_ino,
                ) == repair_identity:
                    repair_lock.unlink()
                    _fsync_directory(repair_lock.parent)
            except OSError:
                pass


def _install_git_index_with_lock(
    repository_root: Path,
    *,
    index_path: Path,
    expected_index: bytes,
    replacement_index: bytes,
    branch: str,
    current_head: str,
    parent_commit_sha: str,
    published_commit_sha: str,
) -> None:
    """Install an index under Git's O_EXCL lock and CAS the branch ref."""

    lock_path = index_path.with_name(f"{index_path.name}.lock")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise GovernedExperimentCLIError(
            "local Git index lock already exists; refusing to race "
            "another Git writer"
        ) from exc
    except OSError as exc:
        raise GovernedExperimentCLIError(
            "local Git index lock cannot be created safely"
        ) from exc
    installed = False
    lock_identity: tuple[int, int] | None = None
    try:
        opened = os.fstat(descriptor)
        current_lock = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_dev != current_lock.st_dev
            or opened.st_ino != current_lock.st_ino
            or (
                hasattr(os, "geteuid")
                and opened.st_uid != os.geteuid()
            )
        ):
            raise GovernedExperimentCLIError(
                "local Git index lock is not one owned regular file"
            )
        lock_identity = (opened.st_dev, opened.st_ino)
        if _read_regular_file(
            index_path,
            label="local Git index",
        ) != expected_index:
            raise GovernedExperimentCLIError(
                "local Git index changed before its lock was acquired"
            )
        view = memoryview(replacement_index)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short Git index lock write")
            view = view[written:]
        os.fsync(descriptor)
        final_lock = os.fstat(descriptor)
        current_lock = lock_path.lstat()
        if (
            not _owned_single_link_regular(final_lock)
            or not _owned_single_link_regular(current_lock)
            or (final_lock.st_dev, final_lock.st_ino) != lock_identity
            or (current_lock.st_dev, current_lock.st_ino) != lock_identity
            or final_lock.st_size != len(replacement_index)
        ):
            raise GovernedExperimentCLIError(
                "local Git index lock changed before commit"
            )
        if current_head == parent_commit_sha:
            _git(
                repository_root,
                "update-ref",
                f"refs/heads/{branch}",
                published_commit_sha,
                parent_commit_sha,
            )
        elif current_head != published_commit_sha:
            raise GovernedExperimentCLIError(
                "local branch moved outside the Git index transaction"
            )
        if _read_regular_file(
            index_path,
            label="local Git index",
        ) != expected_index:
            raise GovernedExperimentCLIError(
                "local Git index changed despite its exclusive lock"
            )
        # Revalidate the lock after the final index read and immediately before
        # rename. In particular, do not install an inode that gained a second
        # filesystem name while the transaction was in flight.
        final_lock = os.fstat(descriptor)
        current_lock = lock_path.lstat()
        if (
            not _owned_single_link_regular(final_lock)
            or not _owned_single_link_regular(current_lock)
            or (final_lock.st_dev, final_lock.st_ino) != lock_identity
            or (current_lock.st_dev, current_lock.st_ino) != lock_identity
            or final_lock.st_size != len(replacement_index)
        ):
            raise GovernedExperimentCLIError(
                "local Git index lock changed immediately before install"
            )
        os.replace(lock_path, index_path)
        installed = True
        installed_descriptor = os.fstat(descriptor)
        installed_path = index_path.lstat()
        installed_identity = (
            installed_descriptor.st_dev,
            installed_descriptor.st_ino,
        )
        if (
            not _owned_single_link_regular(installed_descriptor)
            or not _owned_single_link_regular(installed_path)
            or installed_identity != lock_identity
            or (installed_path.st_dev, installed_path.st_ino)
            != lock_identity
            or installed_descriptor.st_size != len(replacement_index)
        ):
            if (
                installed_identity == lock_identity
                and (
                    installed_path.st_dev,
                    installed_path.st_ino,
                )
                == lock_identity
                and (
                    installed_descriptor.st_nlink != 1
                    or installed_path.st_nlink != 1
                )
            ):
                _detach_hardlinked_git_index(
                    index_path,
                    compromised_descriptor=descriptor,
                    compromised_identity=lock_identity,
                    replacement_index=replacement_index,
                )
                # The detached replacement is a read-tree index without a
                # trusted worktree stat cache. Refresh it before surfacing the
                # fail-closed recovery result so the next immutable retry can
                # recognize the already-synchronized local transaction.
                _git(repository_root, "update-index", "--refresh")
                raise GovernedExperimentCLIError(
                    "local Git index acquired an external hard-link alias "
                    "during install; the live index was safely detached"
                )
            raise GovernedExperimentCLIError(
                "local Git index changed while its lock was installed"
            )
        os.close(descriptor)
        descriptor = -1
        _fsync_directory(index_path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not installed:
            try:
                current = lock_path.lstat()
                if lock_identity is not None and (
                    current.st_dev,
                    current.st_ino,
                ) == lock_identity:
                    lock_path.unlink()
                    _fsync_directory(lock_path.parent)
            except OSError:
                pass


def _synchronize_local_cutover(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    upstream_remote: str,
    prepared: _PreparedCutover,
    published_commit_sha: str,
) -> None:
    _ensure_local_cutover_commit(
        repository_root,
        upstream_remote=upstream_remote,
        prepared=prepared,
        published_commit_sha=published_commit_sha,
    )
    local_head = _repository_head_safety(
        repository_root,
        branch=branch,
    )
    if local_head == published_commit_sha:
        try:
            if (
                _repository_state(
                    repository_root,
                    branch=branch,
                )
                == published_commit_sha
            ):
                return
        except GovernedExperimentCLIError:
            pass
    elif local_head != prepared.parent_commit_sha:
        raise GovernedExperimentCLIError(
            "local branch moved outside the recoverable cutover CAS"
        )
    if _local_status_paths(repository_root) != prepared.changed_paths:
        raise GovernedExperimentCLIError(
            "local recovery found changes outside the three cutover files"
        )
    for path, expected in prepared.paths_to_bytes.items():
        local = _read_regular_file(
            _require_repository_path(
                repository_root,
                path,
                label="local prepared cutover file",
            ),
            label="local prepared cutover file",
        )
        if local != expected:
            raise GovernedExperimentCLIError(
                "local prepared cutover bytes changed before synchronization"
            )
    parent_tree = _git(
        repository_root,
        "rev-parse",
        f"{prepared.parent_commit_sha}^{{tree}}",
    )
    published_tree = _git(
        repository_root,
        "rev-parse",
        f"{published_commit_sha}^{{tree}}",
    )
    index_path = _git_index_path(repository_root)
    index_lock_path = index_path.with_name(f"{index_path.name}.lock")
    if index_lock_path.exists() or index_lock_path.is_symlink():
        raise GovernedExperimentCLIError(
            "local Git index lock already exists; refusing to race "
            "another Git writer"
        )
    current_index_tree = _git(repository_root, "write-tree")
    if current_index_tree not in {parent_tree, published_tree}:
        raise GovernedExperimentCLIError(
            "local index contains changes outside a recoverable cutover"
        )
    index_after = _index_bytes_for_commit(
        repository_root,
        revision=published_commit_sha,
        index_parent=index_path.parent,
    )
    if (
        _repository_head_safety(repository_root, branch=branch)
        != local_head
        or _local_status_paths(repository_root)
        != prepared.changed_paths
        or _git_database_ref(
            repository_root,
            github_repository=github_repository,
            branch=branch,
        )
        != published_commit_sha
    ):
        raise GovernedExperimentCLIError(
            "local or remote state changed before local cutover CAS"
        )
    # Status inspection may refresh stat data in the index.  Snapshot it only
    # after every read-only Git check and perform no index-touching Git command
    # between this byte snapshot and the atomic compare-and-replace.
    index_before = _read_regular_file(
        index_path,
        label="local Git index",
    )
    _install_git_index_with_lock(
        repository_root,
        index_path=index_path,
        expected_index=index_before,
        replacement_index=index_after,
        branch=branch,
        current_head=local_head,
        parent_commit_sha=prepared.parent_commit_sha,
        published_commit_sha=published_commit_sha,
    )
    # A read-tree index intentionally has no trusted worktree stat cache.
    # Refresh only those stat records after the byte-level index CAS; this
    # neither updates blobs nor touches worktree files.
    _git(repository_root, "update-index", "--refresh")
    if (
        _repository_state(repository_root, branch=branch)
        != published_commit_sha
    ):
        raise GovernedExperimentCLIError(
            "local cutover synchronization did not become clean"
        )


def _cutover_receipt_directory(repository_root: Path) -> Path:
    git_directory_raw = _git(
        repository_root,
        "rev-parse",
        "--git-dir",
    )
    git_directory = Path(git_directory_raw)
    if not git_directory.is_absolute():
        git_directory = repository_root / git_directory
    git_directory = git_directory.resolve(strict=True)
    receipt_directory = git_directory / "governed-experiment-recovery"
    _require_safe_directory(
        receipt_directory,
        parent=git_directory,
    )
    return receipt_directory


def _cutover_file_receipts(
    prepared: _PreparedCutover,
) -> dict[str, dict[str, str]]:
    return {
        path: {
            "git_blob_sha": _git_blob_sha(raw),
            "sha256": _sha256_bytes(raw),
        }
        for path, raw in sorted(prepared.paths_to_bytes.items())
    }


def _cutover_transaction_path(
    repository_root: Path,
    *,
    prepared: _PreparedCutover,
) -> Path:
    directory = _cutover_receipt_directory(repository_root)
    return directory / (
        "cutover-transaction-"
        f"{prepared.parent_commit_sha}-"
        f"{prepared.pointer['checkpoint_sha256']}.json"
    )


def _record_cutover_transaction(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    prepared: _PreparedCutover,
    published_commit_sha: str,
    tree_sha: str,
) -> dict[str, Any]:
    value = {
        "branch": branch,
        "files": _cutover_file_receipts(prepared),
        "github_repository": github_repository,
        "operation": "cutover",
        "parent_commit_sha": prepared.parent_commit_sha,
        "published_commit_sha": published_commit_sha,
        "schema": CUTOVER_TRANSACTION_SCHEMA,
        "status": "commit_created_pending_branch_cas",
        "tree_sha": tree_sha,
    }
    raw = _canonical_bytes(value)
    path = _cutover_transaction_path(
        repository_root,
        prepared=prepared,
    )
    _create_once(
        path,
        raw,
        label="cutover CAS transaction receipt",
    )
    return value


def _load_cutover_transaction(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    prepared: _PreparedCutover,
) -> dict[str, Any] | None:
    path = _cutover_transaction_path(
        repository_root,
        prepared=prepared,
    )
    if not path.exists() and not path.is_symlink():
        return None
    value = _decode_canonical_object(
        _read_regular_file(
            path,
            label="cutover CAS transaction receipt",
        ),
        label="cutover CAS transaction receipt",
    )
    expected_keys = {
        "branch",
        "files",
        "github_repository",
        "operation",
        "parent_commit_sha",
        "published_commit_sha",
        "schema",
        "status",
        "tree_sha",
    }
    receipt_repository = value.get("github_repository")
    if (
        set(value) != expected_keys
        or value.get("schema") != CUTOVER_TRANSACTION_SCHEMA
        or value.get("operation") != "cutover"
        or value.get("status")
        != "commit_created_pending_branch_cas"
        or value.get("branch") != branch
        or not isinstance(receipt_repository, str)
        or receipt_repository.casefold() != github_repository.casefold()
        or value.get("parent_commit_sha")
        != prepared.parent_commit_sha
        or value.get("files") != _cutover_file_receipts(prepared)
        or not isinstance(value.get("published_commit_sha"), str)
        or not _COMMIT_RE.fullmatch(
            value["published_commit_sha"].casefold()
        )
        or not isinstance(value.get("tree_sha"), str)
        or not _COMMIT_RE.fullmatch(value["tree_sha"].casefold())
    ):
        raise GovernedExperimentCLIError(
            "cutover CAS transaction receipt is invalid or mismatched"
        )
    value["published_commit_sha"] = value[
        "published_commit_sha"
    ].casefold()
    value["tree_sha"] = value["tree_sha"].casefold()
    return value


def _cutover_recovery_receipt(
    repository_root: Path,
    *,
    github_repository: str,
    branch: str,
    prepared: _PreparedCutover,
    published_commit_sha: str,
    tree_sha: str,
    status: str = "remote_published_local_unsynced",
) -> dict[str, Any]:
    if status not in {
        "branch_cas_outcome_ambiguous_local_unsynced",
        "remote_published_local_unsynced",
    }:
        raise GovernedExperimentCLIError(
            "cutover recovery receipt status is invalid"
        )
    receipt_directory = _cutover_receipt_directory(repository_root)
    value = {
        "branch": branch,
        "files": _cutover_file_receipts(prepared),
        "github_repository": github_repository,
        "operation": "cutover",
        "parent_commit_sha": prepared.parent_commit_sha,
        "published_commit_sha": published_commit_sha,
        "schema": CUTOVER_RECOVERY_SCHEMA,
        "status": status,
        "tree_sha": tree_sha,
    }
    raw = _canonical_bytes(value)
    receipt_path = (
        receipt_directory
        / f"cutover-{status}-{published_commit_sha}.json"
    )
    _create_once(
        receipt_path,
        raw,
        label="cutover recovery receipt",
    )
    return {
        "path": str(receipt_path),
        "sha256": _sha256_bytes(raw),
        "value": value,
    }


def publish_prepared_cutover(
    *,
    repository_root: Path | str,
    github_repository: str,
    branch: str,
) -> dict[str, Any]:
    """Publish exactly three prepared cutover files with a GitHub CAS."""

    root = Path(repository_root).resolve()
    upstream_remote = _require_authority_upstream(
        root,
        github_repository=github_repository,
        branch=branch,
    )
    prepared = _load_prepared_cutover(
        root,
        github_repository=github_repository,
        branch=branch,
    )
    local_head = _repository_head_safety(root, branch=branch)
    remote_head = _git_database_ref(
        root,
        github_repository=github_repository,
        branch=branch,
    )
    published_commit_sha: str
    tree_sha: str
    compare_and_swap_performed = False
    transaction = _load_cutover_transaction(
        root,
        github_repository=github_repository,
        branch=branch,
        prepared=prepared,
    )
    if remote_head == prepared.parent_commit_sha:
        if local_head != prepared.parent_commit_sha:
            raise GovernedExperimentCLIError(
                "local HEAD is not the authenticated cutover parent"
            )
        _require_exact_worktree_changes(
            root,
            expected_paths=prepared.changed_paths,
        )
        for path in prepared.changed_paths:
            if _git_tree_path_exists(
                root,
                revision=prepared.parent_commit_sha,
                relative_path=path,
            ):
                raise GovernedExperimentCLIError(
                    "cutover parent already contains a prepared path"
                )
    if transaction is not None:
        published_commit_sha = transaction["published_commit_sha"]
        tree_sha = transaction["tree_sha"]
        verified_orphan = _verify_remote_cutover(
            root,
            github_repository=github_repository,
            branch=branch,
            prepared=prepared,
            published_commit_sha=published_commit_sha,
            require_branch_head=False,
        )
        if verified_orphan["tree_sha"] != tree_sha:
            raise GovernedExperimentCLIError(
                "recorded cutover transaction names another tree"
            )
        if remote_head not in {
            prepared.parent_commit_sha,
            published_commit_sha,
        }:
            raise GitHubBranchCASRejected(
                "GitHub branch compare-and-swap rejected the cutover because "
                "the branch differs from both the immutable transaction "
                "parent and its recorded commit"
            )
    elif remote_head == prepared.parent_commit_sha:
        published_commit_sha, tree_sha = _post_cutover_commit(
            root,
            github_repository=github_repository,
            prepared=prepared,
        )
        # Persist immutable intent before the branch write. A retry must reuse
        # P; it may never manufacture or accept a different equivalent commit.
        _record_cutover_transaction(
            root,
            github_repository=github_repository,
            branch=branch,
            prepared=prepared,
            published_commit_sha=published_commit_sha,
            tree_sha=tree_sha,
        )
        verified_orphan = _verify_remote_cutover(
            root,
            github_repository=github_repository,
            branch=branch,
            prepared=prepared,
            published_commit_sha=published_commit_sha,
            require_branch_head=False,
        )
        if verified_orphan["tree_sha"] != tree_sha:
            raise GovernedExperimentCLIError(
                "new cutover transaction names another tree"
            )
    else:
        published_commit_sha = remote_head
        existing = _verify_remote_cutover(
            root,
            github_repository=github_repository,
            branch=branch,
            prepared=prepared,
            published_commit_sha=published_commit_sha,
        )
        tree_sha = existing["tree_sha"]

    if remote_head == prepared.parent_commit_sha:
        try:
            _compare_and_swap_cutover_ref(
                root,
                github_repository=github_repository,
                branch=branch,
                parent_commit_sha=prepared.parent_commit_sha,
                published_commit_sha=published_commit_sha,
            )
        except GitHubBranchCASRejected:
            raise
        except Exception as exc:
            receipt = _cutover_recovery_receipt(
                root,
                github_repository=github_repository,
                branch=branch,
                prepared=prepared,
                published_commit_sha=published_commit_sha,
                tree_sha=tree_sha,
                status=(
                    "branch_cas_outcome_ambiguous_local_unsynced"
                ),
            )
            return {
                "authority_verified": False,
                "branch": branch,
                "error": str(exc),
                "github_repository": github_repository,
                "parent_commit_sha": prepared.parent_commit_sha,
                "published_commit_sha": published_commit_sha,
                "recovery_receipt_path": receipt["path"],
                "recovery_receipt_sha256": receipt["sha256"],
                "status": (
                    "branch_cas_outcome_ambiguous_local_unsynced"
                ),
            }
        compare_and_swap_performed = True
    try:
        verified = _verify_remote_cutover(
            root,
            github_repository=github_repository,
            branch=branch,
            prepared=prepared,
            published_commit_sha=published_commit_sha,
        )
        if verified["tree_sha"] != tree_sha:
            raise GovernedExperimentCLIError(
                "published cutover tree differs from its transaction receipt"
            )
        _synchronize_local_cutover(
            root,
            github_repository=github_repository,
            branch=branch,
            upstream_remote=upstream_remote,
            prepared=prepared,
            published_commit_sha=published_commit_sha,
        )
        if _git_database_ref(
            root,
            github_repository=github_repository,
            branch=branch,
        ) != published_commit_sha:
            raise GovernedExperimentCLIError(
                "GitHub branch moved after cutover publication"
            )
        authority = verify_authority(
            repository_root=root,
            github_repository=github_repository,
            branch=branch,
        )
    except Exception as exc:
        receipt = _cutover_recovery_receipt(
            root,
            github_repository=github_repository,
            branch=branch,
            prepared=prepared,
            published_commit_sha=published_commit_sha,
            tree_sha=tree_sha,
        )
        return {
            "authority_verified": False,
            "branch": branch,
            "error": str(exc),
            "github_repository": github_repository,
            "parent_commit_sha": prepared.parent_commit_sha,
            "published_commit_sha": published_commit_sha,
            "recovery_receipt_path": receipt["path"],
            "recovery_receipt_sha256": receipt["sha256"],
            "status": "remote_published_local_unsynced",
        }
    return {
        "authority": authority,
        "authority_verified": True,
        "branch": branch,
        "checkpoint_path": prepared.pointer["checkpoint_path"],
        "checkpoint_sha256": prepared.pointer["checkpoint_sha256"],
        "compare_and_swap_performed": compare_and_swap_performed,
        "github_repository": github_repository,
        "parent_commit_sha": prepared.parent_commit_sha,
        "published_commit_sha": published_commit_sha,
        "status": (
            "authenticated_cutover_published"
            if compare_and_swap_performed
            else "authenticated_cutover_already_published"
        ),
    }


def execute_cutover(
    *,
    repository_root: Path | str,
    github_repository: str,
    branch: str,
    promotion_population_path: Path | str,
) -> dict[str, Any]:
    """Prepare an absent cutover, then publish or recover it atomically."""

    root = Path(repository_root).resolve()
    pointer = root.joinpath(*POINTER_RELATIVE_PATH.parts)
    if not pointer.exists() and not pointer.is_symlink():
        create_cutover(
            repository_root=root,
            github_repository=github_repository,
            branch=branch,
            promotion_population_path=promotion_population_path,
        )
    return publish_prepared_cutover(
        repository_root=root,
        github_repository=github_repository,
        branch=branch,
    )


class GitHubCheckpointAuthority:
    """Resolve one exact current checkpoint from authenticated GitHub bytes."""

    def __init__(
        self,
        *,
        repository_root: Path | str,
        github_repository: str,
        branch: str,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.github_repository = github_repository
        self.branch = branch

    def resolve(
        self,
        *,
        stores: Mapping[str, CanonicalHashChainStore],
    ) -> tuple[ProgramIntegrityCheckpoint, dict[str, Any], str]:
        revision = require_clean_remote_head(
            self.repository_root,
            github_repository=self.github_repository,
            branch=self.branch,
        )
        (
            program_root,
            pointer_path,
            _checkpoint_directory,
            expected_stores,
        ) = _program_paths(self.repository_root)
        for name, expected in expected_stores.items():
            supplied = stores.get(name)
            if (
                not isinstance(supplied, CanonicalHashChainStore)
                or Path(os.path.abspath(supplied.path))
                != Path(os.path.abspath(expected.path))
            ):
                raise GovernedExperimentCLIError(
                    "authority received an unexpected governed store"
                )
            _read_regular_file(
                expected.path,
                label=f"{name} governed ledger",
            )
        local_pointer_raw = _read_regular_file(
            pointer_path,
            label="local current checkpoint pointer",
        )
        authority_path = _require_repository_path(
            self.repository_root,
            CUTOVER_AUTHORITY_RELATIVE_PATH,
            label="cutover authority attestation",
        )
        local_authority_raw = _read_regular_file(
            authority_path,
            label="local cutover authority attestation",
        )
        remote_authority_raw = _remote_file(
            self.repository_root,
            github_repository=self.github_repository,
            revision=revision,
            relative_path=CUTOVER_AUTHORITY_RELATIVE_PATH,
            label="cutover authority attestation",
        )
        if local_authority_raw != remote_authority_raw:
            raise GovernedExperimentCLIError(
                "local and remote cutover authority attestations differ"
            )
        cutover_authority = _normalize_cutover_authority(
            _decode_canonical_object(
                local_authority_raw,
                label="cutover authority attestation",
            )
        )
        if (
            cutover_authority["github_repository"].casefold()
            != self.github_repository.casefold()
            or cutover_authority["branch"] != self.branch
        ):
            raise GovernedExperimentCLIError(
                "cutover authority attestation names another publisher"
            )
        remote_pointer_raw = _remote_file(
            self.repository_root,
            github_repository=self.github_repository,
            revision=revision,
            relative_path=POINTER_RELATIVE_PATH,
            label="current checkpoint pointer",
        )
        if local_pointer_raw != remote_pointer_raw:
            raise GovernedExperimentCLIError(
                "local and remote current checkpoint pointers differ"
            )
        pointer = _normalize_pointer(
            _decode_canonical_object(
                local_pointer_raw,
                label="current checkpoint pointer",
            )
        )
        checkpoint_relative = PurePosixPath(
            pointer["checkpoint_path"]
        )
        checkpoint_path = _require_repository_path(
            self.repository_root,
            checkpoint_relative,
            label="current checkpoint",
        )
        try:
            checkpoint_path.resolve(strict=True).relative_to(
                program_root.resolve()
            )
        except (OSError, ValueError) as exc:
            raise GovernedExperimentCLIError(
                "current checkpoint escapes the trusted program root"
            ) from exc
        local_checkpoint_raw = _read_regular_file(
            checkpoint_path,
            label="local current checkpoint",
        )
        remote_checkpoint_raw = _remote_file(
            self.repository_root,
            github_repository=self.github_repository,
            revision=revision,
            relative_path=checkpoint_relative,
            label="current checkpoint",
        )
        if local_checkpoint_raw != remote_checkpoint_raw:
            raise GovernedExperimentCLIError(
                "local and remote current checkpoints differ"
            )
        digest = pointer["checkpoint_sha256"]
        if _sha256_bytes(local_checkpoint_raw) != digest:
            raise GovernedExperimentCLIError(
                "current checkpoint bytes do not match the remote pointer"
            )
        checkpoint_value = _decode_canonical_object(
            local_checkpoint_raw,
            label="current checkpoint",
        )
        if "promotion_population" not in checkpoint_value:
            raise GovernedExperimentCLIError(
                "current checkpoint has no immutable promotion population"
            )
        baseline_path = _require_repository_path(
            self.repository_root,
            BASELINE_MANIFEST_RELATIVE_PATH,
            label="frozen baseline manifest",
        )
        if checkpoint_value.get(
            "baseline_manifest_sha256"
        ) != _sha256_bytes(
            _read_regular_file(
                baseline_path,
                label="frozen baseline manifest",
            )
        ):
            raise GovernedExperimentCLIError(
                "current checkpoint rewrote the frozen baseline binding"
            )
        anchors = checkpoint_value.get("stores")
        if not isinstance(anchors, Mapping) or set(anchors) != set(
            LEDGER_FILENAMES
        ):
            raise GovernedExperimentCLIError(
                "current checkpoint has invalid governed-store anchors"
            )
        for name, filename in LEDGER_FILENAMES.items():
            anchor = anchors[name]
            if (
                not isinstance(anchor, Mapping)
                or anchor.get("path") != filename
            ):
                raise GovernedExperimentCLIError(
                    f"{name} checkpoint anchor must name its exact ledger"
                )

        reference = PublishedCheckpointReference(
            path=checkpoint_path,
            sha256=digest,
        )
        checkpoint = CheckpointAuthorityResolver(
            lambda: reference,
            trusted_checkpoint_root=program_root,
        ).resolve(stores=stores)
        return checkpoint, pointer, revision


def _git_tree_path_exists(
    repository_root: Path,
    *,
    revision: str,
    relative_path: str,
) -> bool:
    return bool(
        _git_raw(
            repository_root,
            "ls-tree",
            "-z",
            revision,
            "--",
            relative_path,
        )
    )


def _git_blob(
    repository_root: Path,
    *,
    revision: str,
    relative_path: str,
) -> bytes:
    return _git_raw(
        repository_root,
        "cat-file",
        "blob",
        f"{revision}:{relative_path}",
    )


def _verify_publication_topology(
    repository_root: Path,
    *,
    revision: str,
    pointer: Mapping[str, Any],
    checkpoint: ProgramIntegrityCheckpoint,
) -> dict[str, Any]:
    lineage = _git(
        repository_root,
        "rev-list",
        "--parents",
        "-n",
        "1",
        revision,
    ).split()
    if len(lineage) != 2 or lineage[0] != revision:
        raise GovernedExperimentCLIError(
            "checkpoint publication must be one direct non-merge child"
        )
    parent = lineage[1]
    pointer_path = POINTER_RELATIVE_PATH.as_posix()
    checkpoint_path = str(pointer["checkpoint_path"])
    authority_path = CUTOVER_AUTHORITY_RELATIVE_PATH.as_posix()
    authority_raw = _git_blob(
        repository_root,
        revision=revision,
        relative_path=authority_path,
    )
    cutover_authority = _normalize_cutover_authority(
        _decode_canonical_object(
            authority_raw,
            label="published cutover authority attestation",
        )
    )
    local_pointer = _read_regular_file(
        repository_root.joinpath(*POINTER_RELATIVE_PATH.parts),
        label="published current checkpoint pointer",
    )
    local_checkpoint = _read_regular_file(
        checkpoint.path,
        label="published current checkpoint",
    )
    current_checkpoint_value = _decode_canonical_object(
        local_checkpoint,
        label="published current checkpoint",
    )
    if _git_blob(
        repository_root,
        revision=revision,
        relative_path=pointer_path,
    ) != local_pointer or _git_blob(
        repository_root,
        revision=revision,
        relative_path=checkpoint_path,
    ) != local_checkpoint:
        raise GovernedExperimentCLIError(
            "published governance blobs do not match authenticated HEAD"
        )
    if _git_tree_path_exists(
        repository_root,
        revision=parent,
        relative_path=checkpoint_path,
    ):
        raise GovernedExperimentCLIError(
            "successor checkpoint must be created once in its publication"
        )

    previous = pointer["previous_checkpoint_sha256"]
    expected_changes = {pointer_path, checkpoint_path}
    transition = "cutover"
    if previous is None:
        expected_changes.add(authority_path)
        if cutover_authority["parent_commit_sha"] != parent:
            raise GovernedExperimentCLIError(
                "cutover publication parent does not match its immutable "
                "authority attestation"
            )
        if _git_tree_path_exists(
            repository_root,
            revision=parent,
            relative_path=pointer_path,
        ):
            raise GovernedExperimentCLIError(
                "cutover parent may not already contain a current pointer"
            )
        if _git_tree_path_exists(
            repository_root,
            revision=parent,
            relative_path=authority_path,
        ):
            raise GovernedExperimentCLIError(
                "cutover parent may not contain an authority attestation"
            )
    else:
        transition = "experiment_preregistration"
        if _git_blob(
            repository_root,
            revision=parent,
            relative_path=authority_path,
        ) != authority_raw:
            raise GovernedExperimentCLIError(
                "successor publication changed cutover authority"
            )
        if not _git_tree_path_exists(
            repository_root,
            revision=parent,
            relative_path=pointer_path,
        ):
            raise GovernedExperimentCLIError(
                "successor publication parent has no current pointer"
            )
        parent_pointer_raw = _git_blob(
            repository_root,
            revision=parent,
            relative_path=pointer_path,
        )
        parent_pointer = _normalize_pointer(
            _decode_canonical_object(
                parent_pointer_raw,
                label="parent current checkpoint pointer",
            )
        )
        if parent_pointer["checkpoint_sha256"] != previous:
            raise GovernedExperimentCLIError(
                "successor pointer does not name its direct parent checkpoint"
            )
        parent_checkpoint_path = parent_pointer["checkpoint_path"]
        parent_checkpoint_raw = _git_blob(
            repository_root,
            revision=parent,
            relative_path=parent_checkpoint_path,
        )
        if _sha256_bytes(parent_checkpoint_raw) != previous:
            raise GovernedExperimentCLIError(
                "parent checkpoint bytes do not match the predecessor pointer"
            )
        parent_checkpoint_value = _decode_canonical_object(
            parent_checkpoint_raw,
            label="parent checkpoint",
        )
        for name, filename in LEDGER_FILENAMES.items():
            parent_ledger_raw = _git_blob(
                repository_root,
                revision=parent,
                relative_path=(
                    PROGRAM_RELATIVE_PATH / filename
                ).as_posix(),
            )
            try:
                parent_records_for_anchor = (
                    CanonicalHashChainStore._parse(
                        parent_ledger_raw.decode("utf-8")
                    )
                )
            except (UnicodeDecodeError, IntegrityError) as exc:
                raise GovernedExperimentCLIError(
                    f"parent {name} ledger is invalid"
                ) from exc
            parent_anchor = parent_checkpoint_value["stores"][name]
            if (
                parent_anchor.get("path") != filename
                or parent_anchor.get("sha256")
                != _sha256_bytes(parent_ledger_raw)
                or parent_anchor.get("expected_length")
                != len(parent_records_for_anchor)
                or parent_anchor.get("expected_head")
                != CanonicalHashChainStore._head(
                    parent_records_for_anchor
                )
            ):
                raise GovernedExperimentCLIError(
                    f"parent {name} anchor does not match its ledger"
                )
        for key in (
            "baseline_manifest_sha256",
            "promotion_population",
            "runtime_leakage_finding_count",
        ):
            if current_checkpoint_value.get(
                key
            ) != parent_checkpoint_value.get(key):
                raise GovernedExperimentCLIError(
                    "successor publication rewrote immutable checkpoint "
                    f"field {key}"
                )
        for name in LEDGER_FILENAMES:
            before_anchor = parent_checkpoint_value["stores"][name]
            after_anchor = current_checkpoint_value["stores"][name]
            if name != "experiment_ledger":
                if after_anchor != before_anchor:
                    raise GovernedExperimentCLIError(
                        f"successor publication rewrote {name}"
                    )
            elif (
                after_anchor.get("path") != before_anchor.get("path")
                or after_anchor.get("expected_length")
                != before_anchor.get("expected_length") + 1
            ):
                raise GovernedExperimentCLIError(
                    "successor publication is not one exact experiment append"
                )

        experiment_path = (
            PROGRAM_RELATIVE_PATH
            / LEDGER_FILENAMES["experiment_ledger"]
        ).as_posix()
        expected_changes.add(experiment_path)
        parent_ledger_raw = _git_blob(
            repository_root,
            revision=parent,
            relative_path=experiment_path,
        )
        current_ledger_raw = _read_regular_file(
            checkpoint.stores["experiment_ledger"].path,
            label="published experiment ledger",
        )
        if not current_ledger_raw.startswith(parent_ledger_raw):
            raise GovernedExperimentCLIError(
                "experiment publication rewrote its predecessor ledger"
            )
        try:
            parent_records = CanonicalHashChainStore._parse(
                parent_ledger_raw.decode("utf-8")
            )
            current_records = CanonicalHashChainStore._parse(
                current_ledger_raw.decode("utf-8")
            )
        except (UnicodeDecodeError, IntegrityError) as exc:
            raise GovernedExperimentCLIError(
                "experiment publication ledger transition is invalid"
            ) from exc
        if len(current_records) != len(parent_records) + 1:
            raise GovernedExperimentCLIError(
                "experiment publication must append exactly one ledger record"
            )
        appended = current_records[-1]["payload"]
        if (
            appended.get("event") != "experiment_plan"
            or set(appended)
            != {"event", "experiment_id", "plan"}
            or not isinstance(appended.get("plan"), Mapping)
        ):
            raise GovernedExperimentCLIError(
                "publication append is not one canonical experiment plan"
            )
        normalized_plan = _normalize_experiment_plan(appended["plan"])
        if (
            normalized_plan != appended["plan"]
            or normalized_plan["parent_commit_sha"] != parent
        ):
            raise GovernedExperimentCLIError(
                "published experiment plan is not bound to its direct parent"
            )
        if current_checkpoint_value["stores"][
            "experiment_ledger"
        ].get("expected_head") != current_records[-1]["record_hash"]:
            raise GovernedExperimentCLIError(
                "successor checkpoint does not anchor the appended plan"
            )

    changed_raw = _git_raw(
        repository_root,
        "diff-tree",
        "--no-commit-id",
        "--name-only",
        "-r",
        "-z",
        parent,
        revision,
        "--",
    )
    try:
        changed_paths = {
            path
            for path in changed_raw.decode("utf-8").split("\0")
            if path
        }
    except UnicodeDecodeError as exc:
        raise GovernedExperimentCLIError(
            "checkpoint publication contains a non-UTF-8 path"
        ) from exc
    if changed_paths != expected_changes:
        raise GovernedExperimentCLIError(
            "checkpoint publication is not the exact governance-only "
            f"transition: expected {sorted(expected_changes)}, "
            f"observed {sorted(changed_paths)}"
        )
    return {
        "changed_paths": sorted(changed_paths),
        "parent_commit_sha": parent,
        "transition": transition,
    }


def _require_exact_experiment_successor(
    current: ProgramIntegrityCheckpoint,
    receipt: Any,
    *,
    previous: Mapping[str, Any],
) -> None:
    successor_raw = receipt.next_checkpoint_bytes
    successor = _decode_canonical_object(
        successor_raw,
        label="experiment successor checkpoint",
    )
    immutable_keys = {
        "baseline_manifest_sha256",
        "promotion_population",
        "runtime_leakage_finding_count",
    }
    if not immutable_keys.issubset(previous) or any(
        successor.get(key) != previous.get(key)
        for key in immutable_keys
    ):
        raise IntegrityError(
            "experiment successor rewrote immutable checkpoint authority"
        )
    if set(successor) != set(previous):
        raise IntegrityError(
            "experiment successor changed checkpoint root fields"
        )
    previous_stores = previous["stores"]
    successor_stores = successor.get("stores")
    if not isinstance(successor_stores, Mapping) or set(
        successor_stores
    ) != set(previous_stores):
        raise IntegrityError(
            "experiment successor changed governed-store anchors"
        )
    for name in previous_stores:
        before = previous_stores[name]
        after = successor_stores[name]
        if name != "experiment_ledger":
            if after != before:
                raise IntegrityError(
                    f"experiment successor rewrote {name}"
                )
            continue
        if (
            not isinstance(after, Mapping)
            or after.get("path") != before.get("path")
            or after.get("expected_length")
            != before.get("expected_length") + 1
            or after.get("expected_head") != receipt.record_hash
        ):
            raise IntegrityError(
                "experiment successor is not the exact one-record append"
            )
    if (
        receipt.previous_checkpoint_sha256
        != current.expected_sha256
        or _sha256_bytes(successor_raw)
        != receipt.next_checkpoint_sha256
    ):
        raise IntegrityError(
            "experiment successor transition digest is inconsistent"
        )


def verify_authority(
    *,
    repository_root: Path | str,
    github_repository: str,
    branch: str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    _program, _pointer, _directory, stores = _program_paths(root)
    checkpoint, pointer, revision = GitHubCheckpointAuthority(
        repository_root=root,
        github_repository=github_repository,
        branch=branch,
    ).resolve(stores=stores)
    checkpoint.verify()
    publication = _verify_publication_topology(
        root,
        revision=revision,
        pointer=pointer,
        checkpoint=checkpoint,
    )
    if _repository_state(root, branch=branch) != revision:
        raise GovernedExperimentCLIError(
            "local Git state moved during authority verification"
        )
    if (
        _remote_head(
            root,
            github_repository=github_repository,
            branch=branch,
        )
        != revision
        or _repository_state(root, branch=branch) != revision
    ):
        raise GovernedExperimentCLIError(
            "authenticated GitHub branch moved during authority verification"
        )
    return {
        "branch": branch,
        "checkpoint_path": pointer["checkpoint_path"],
        "checkpoint_sha256": checkpoint.expected_sha256,
        "github_repository": github_repository,
        "publication": publication,
        "remote_head": revision,
        "status": "authenticated_current_checkpoint_verified",
    }


def preregister(
    *,
    repository_root: Path | str,
    github_repository: str,
    branch: str,
    experiment_id: str,
    plan_path: Path | str,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    (
        program_root,
        pointer_path,
        checkpoint_directory,
        stores,
    ) = _program_paths(root)
    plan = _require_external_canonical_object(
        plan_path,
        repository_root=root,
        label="experiment plan",
    )
    authority = GitHubCheckpointAuthority(
        repository_root=root,
        github_repository=github_repository,
        branch=branch,
    )
    current, current_pointer, revision = authority.resolve(
        stores=stores
    )
    if plan.get("parent_commit_sha") != revision:
        raise GovernedExperimentCLIError(
            "experiment plan parent_commit_sha must equal authenticated HEAD"
        )
    ledger = ExperimentLedger(
        stores["experiment_ledger"].path,
        protected_access_path=stores[
            "protected_access_ledger"
        ].path,
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
            raise GovernedExperimentCLIError(
                "authenticated checkpoint authority changed before "
                "preregistration"
            )
        current = locked_current
        previous_checkpoint_value = current.verify()
        before = _ledger_bytes(stores)
        pointer_before = _read_regular_file(
            pointer_path,
            label="current checkpoint pointer",
        )
        next_pointer_raw: bytes | None = None
        after: dict[str, bytes] | None = None
        try:
            receipt = ledger.preregister(
                experiment_id,
                plan,
                integrity_checkpoint=current,
                expected_head=current.verify()["stores"][
                    "experiment_ledger"
                ]["expected_head"],
            )
            after = _ledger_bytes(stores)
            CheckpointAuthorityResolver.validate_successor(
                current,
                receipt.integrity,
            )
            _require_exact_experiment_successor(
                current,
                receipt,
                previous=previous_checkpoint_value,
            )
            for name in sorted(stores):
                if name == "experiment_ledger":
                    continue
                if after[name] != before[name]:
                    raise IntegrityError(
                        f"{name} changed during experiment preregistration"
                    )
            if not receipt.mutated:
                raise GovernedExperimentCLIError(
                    "experiment was already preregistered; no pointer update "
                    "was performed"
                )
            if after["experiment_ledger"] == before["experiment_ledger"]:
                raise IntegrityError(
                    "preregistration did not append its experiment plan"
                )
            next_digest = receipt.next_checkpoint_sha256
            checkpoint_path = (
                checkpoint_directory / f"{next_digest}.json"
            )
            checkpoint_created = _create_once(
                checkpoint_path,
                receipt.next_checkpoint_bytes,
                label="successor checkpoint",
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
                    "current checkpoint pointer update is incomplete"
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
            if _git(root, "rev-parse", "HEAD").casefold() != revision:
                raise GovernedExperimentCLIError(
                    "local Git HEAD moved during preregistration"
                )
            if _remote_head(
                root,
                github_repository=github_repository,
                branch=branch,
            ) != revision:
                raise GovernedExperimentCLIError(
                    "authenticated GitHub branch moved during "
                    "preregistration"
                )
            if _git(root, "rev-parse", "HEAD").casefold() != revision:
                raise GovernedExperimentCLIError(
                    "local Git HEAD moved during preregistration"
                )
        except BaseException as operation_error:
            cleanup_errors: list[str] = []
            try:
                current_pointer = _read_regular_file(
                    pointer_path,
                    label="partial current checkpoint pointer",
                )
                pointer_was_replaced = (
                    next_pointer_raw is not None
                    and current_pointer == next_pointer_raw
                )
                if pointer_updated or pointer_was_replaced:
                    _atomic_replace_exact(
                        pointer_path,
                        expected=next_pointer_raw,
                        replacement=pointer_before,
                        label="current checkpoint pointer rollback",
                    )
                elif current_pointer != pointer_before:
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
                        label="successor checkpoint rollback target",
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
                        if (
                            after is None
                            or current_ledger != after[name]
                        ):
                            raise IntegrityError(
                                f"{name} changed to bytes not owned by "
                                "this preregistration"
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
                    "preregistration cleanup was incomplete: "
                    + "; ".join(cleanup_errors)
                ) from operation_error
            raise
    if receipt is None:  # pragma: no cover - defensive
        raise IntegrityError("preregistration produced no receipt")
    return {
        "branch": branch,
        "checkpoint_path": (
            CHECKPOINT_DIRECTORY_RELATIVE_PATH
            / f"{receipt.next_checkpoint_sha256}.json"
        ).as_posix(),
        "checkpoint_sha256": receipt.next_checkpoint_sha256,
        "experiment_id": experiment_id,
        "github_repository": github_repository,
        "next_required_action": (
            "commit and push the experiment-ledger append, successor "
            "checkpoint, and current pointer as one Git commit; then run "
            "verify-authority before editing or executing candidate code"
        ),
        "plan_record_hash": receipt.record_hash,
        "previous_checkpoint_sha256": (
            receipt.previous_checkpoint_sha256
        ),
        "remote_parent_head": revision,
        "status": "preregistered_locally_not_published",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPO_ROOT,
    )
    parser.add_argument("--github-repository", required=True)
    parser.add_argument("--branch", required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    cutover = commands.add_parser("cutover")
    cutover.add_argument(
        "--promotion-population",
        type=Path,
        required=True,
    )

    commands.add_parser("verify-authority")

    register = commands.add_parser("preregister")
    register.add_argument("--experiment-id", required=True)
    register.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    common = {
        "repository_root": arguments.repository_root,
        "github_repository": arguments.github_repository,
        "branch": arguments.branch,
    }
    try:
        if arguments.command == "cutover":
            result = execute_cutover(
                **common,
                promotion_population_path=(
                    arguments.promotion_population
                ),
            )
        elif arguments.command == "verify-authority":
            result = verify_authority(**common)
        else:
            result = preregister(
                **common,
                experiment_id=arguments.experiment_id,
                plan_path=arguments.plan,
            )
    except (ExperimentControlError, OSError) as exc:
        print(
            canonical_json(
                {
                    "error": str(exc),
                    "status": "blocked",
                }
            ),
            file=sys.stderr,
        )
        return 2
    print(canonical_json(result))
    if (
        arguments.command == "cutover"
        and result.get("authority_verified") is not True
    ):
        # The immutable recovery receipt has been emitted, but callers must
        # not mistake an ambiguous or locally unsynchronized publication for
        # an authoritative cutover.
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
