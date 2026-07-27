#!/usr/bin/env python3
"""Capture two truth-blind production runs for a grouped WO-17 experiment.

The capture command never accepts a truth/label argument.  Identity-bearing
prediction rows remain in explicitly external files.  The observation and
policy-audit files are path-free aggregate objects suitable for later binding
by :mod:`devtools.grouped_policy_revalidation_evidence`.

The production source, producer graph, v2 layout manifest, and input PDF tree
are verified before execution and rechecked afterward.  Every final output
case contributes exactly one historical policy-audit mapping.  A separate
observer classifies review-to-approval transitions with the exact frozen
``ReviewDenialRecoveryAdjudicator._matching_approval_rules`` implementation;
it does not reinterpret the historical ``forced_approval_count``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import resource
import socket
import stat
import subprocess
import sys
import tempfile
import tarfile
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


_ARCHIVE_CONTRACT_FD_ENV = "MIB_GROUPED_CAPTURE_CONTRACT_FD"
_ARCHIVE_ROOT_ENV = "MIB_GROUPED_CAPTURE_ARCHIVE_ROOT"
_ARCHIVE_DIGEST_ENV = "MIB_GROUPED_CAPTURE_ARCHIVE_DIGEST"
_ARCHIVE_REVISION_ENV = "MIB_GROUPED_CAPTURE_ARCHIVE_REVISION"
_LEGACY_ARCHIVE_ENV_KEYS = frozenset(
    {
        _ARCHIVE_ROOT_ENV,
        _ARCHIVE_DIGEST_ENV,
        _ARCHIVE_REVISION_ENV,
    }
)
_ARCHIVE_CONTRACT_KEYS = frozenset(
    {
        "archive_sha256",
        "bootstrap_pid",
        "denied_sensitive_paths",
        "origin_root",
        "revision",
        "sandbox_backend",
        "sandbox_canary_path",
        "sandbox_profile_sha256",
        "schema",
        "source_root",
        "tree_sha256",
    }
)
_ARCHIVE_CHILD_FLAG = "--_mib-wo17-sandbox-child"
_SANDBOX_EXECUTABLE = Path("/usr/bin/sandbox-exec")
_SANDBOX_PATH = (
    "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
)
_FORBIDDEN_ARCHIVE_PATHS = frozenset({"data/train_labels.csv"})
_SANDBOX_BACKEND = "macos_sandbox_exec_v1"


def _early_canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _consume_archive_contract() -> Mapping[str, Any] | None:
    """Consume the one-shot parent/child pipe contract, if this is the child."""

    descriptor_text = os.environ.pop(_ARCHIVE_CONTRACT_FD_ENV, None)
    if descriptor_text is None:
        return None
    if len(sys.argv) < 2 or sys.argv[1] != _ARCHIVE_CHILD_FLAG:
        raise RuntimeError("caller-supplied archive child state is forbidden")
    del sys.argv[1]
    try:
        descriptor = int(descriptor_text)
    except ValueError as exc:
        raise RuntimeError("archive contract descriptor is invalid") from exc
    if descriptor <= 2:
        raise RuntimeError("archive contract descriptor is invalid")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISFIFO(opened.st_mode):
            raise RuntimeError("archive contract must arrive over a pipe")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            total += len(chunk)
            if total > 8192:
                raise RuntimeError("archive contract is oversized")
            chunks.append(chunk)
    except OSError as exc:
        raise RuntimeError("archive contract cannot be consumed") from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    raw = b"".join(chunks)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("archive contract is not canonical JSON") from exc
    if (
        not isinstance(value, dict)
        or set(value) != _ARCHIVE_CONTRACT_KEYS
        or raw != _early_canonical_bytes(value)
        or value["schema"] != "mib-wo17-archive-child/v2"
    ):
        raise RuntimeError("archive contract has an inexact schema")
    for name in ("archive_sha256", "tree_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(value[name])) is None:
            raise RuntimeError(f"archive contract {name} is invalid")
    if re.fullmatch(r"[0-9a-f]{40}", str(value["revision"])) is None:
        raise RuntimeError("archive contract revision is invalid")
    if (
        isinstance(value["bootstrap_pid"], bool)
        or not isinstance(value["bootstrap_pid"], int)
        or value["bootstrap_pid"] <= 1
        or value["bootstrap_pid"] != os.getppid()
        or value["sandbox_backend"] != _SANDBOX_BACKEND
    ):
        raise RuntimeError("archive contract parent/backend binding is invalid")
    if re.fullmatch(
        r"[0-9a-f]{64}", str(value["sandbox_profile_sha256"])
    ) is None:
        raise RuntimeError("archive contract sandbox digest is invalid")
    denied = value["denied_sensitive_paths"]
    if (
        not isinstance(denied, list)
        or not denied
        or len(set(denied)) != len(denied)
        or any(
            not isinstance(item, str)
            or not Path(item).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(item).parts)
            or str(Path(item)) != item
            for item in denied
        )
    ):
        raise RuntimeError(
            "archive contract denied-sensitive-path set is invalid"
        )
    for name in ("source_root",):
        requested = Path(str(value[name]))
        try:
            resolved = requested.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise RuntimeError(
                f"archive contract {name} is unavailable"
            ) from exc
        if not requested.is_absolute() or requested != resolved:
            raise RuntimeError(
                f"archive contract {name} must be canonical"
            )
    for name in ("origin_root", "sandbox_canary_path"):
        requested = Path(str(value[name]))
        if (
            not requested.is_absolute()
            or any(part in {"", ".", ".."} for part in requested.parts)
            or str(requested) != str(value[name])
        ):
            raise RuntimeError(
                f"archive contract {name} must be canonical"
            )
    return dict(value)


try:
    _ARCHIVE_CONTRACT = _consume_archive_contract()
except RuntimeError as _archive_contract_error:
    if __name__ == "__main__":
        print(
            f"grouped policy capture error: {_archive_contract_error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    raise


def _early_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for path in root.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        if path.relative_to(root).as_posix() in _FORBIDDEN_ARCHIVE_PATHS:
            raise RuntimeError(
                "archive source contains a forbidden truth/label path"
            )
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise RuntimeError(
                "archive source entry cannot be inspected"
            ) from exc
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError(
                "archive source contains a non-regular entry"
            )
        paths.append(path)
    paths.sort()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _early_archive_tree_sha256(raw: bytes) -> str:
    """Compute the extracted-tree digest directly from ordinary tar members."""

    files: list[tuple[bytes, bytes]] = []
    seen: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as bundle:
        for member in bundle.getmembers():
            pure = Path(member.name)
            if (
                pure.is_absolute()
                or "\\" in member.name
                or any(part in {"", ".", ".."} for part in pure.parts)
                or member.name in seen
                or not (member.isdir() or member.isfile())
            ):
                raise RuntimeError("Git archive contains an unsafe entry")
            seen.add(member.name)
            if (
                member.isdir()
                or "__pycache__" in pure.parts
                or pure.as_posix() in _FORBIDDEN_ARCHIVE_PATHS
            ):
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError("Git archive regular file has no bytes")
            files.append(
                (pure.as_posix().encode("utf-8"), source.read())
            )
    digest = hashlib.sha256()
    for relative, content in sorted(files):
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(content)
    return digest.hexdigest()


def _sandbox_profile(
    *,
    origin_root: Path,
    source_root: Path,
    input_root: Path,
    denied_sensitive_paths: Sequence[Path],
) -> str:
    """Return the exact Seatbelt policy used for untrusted production code."""

    def literal(path: Path) -> str:
        return json.dumps(str(path.resolve(strict=True)))

    # The platform runtime needs broad system reads, but the evaluated code is
    # denied every canonical repository/label byte, all network access, and
    # writes to source or inputs.  Capture outputs are written by this same
    # sandboxed process only after all measurements and are create-once.
    rules = [
            "(version 1)",
            "(allow default)",
            "(deny network*)",
            f"(deny file-read* (subpath {literal(origin_root)}))",
            f"(deny file-write* (subpath {literal(origin_root)}))",
            f"(deny file-write* (subpath {literal(source_root)}))",
            f"(deny file-write* (subpath {literal(input_root)}))",
    ]
    rules.extend(
        f"(deny file-read* (literal {literal(path)}))"
        for path in denied_sensitive_paths
    )
    rules.append("")
    return "\n".join(rules)


def _sandbox_limit_preexec() -> None:
    """Install inherited hard limits before sandbox-exec starts Python."""

    limits = (
        (getattr(resource, "RLIMIT_CPU", None), 30000),
        (getattr(resource, "RLIMIT_FSIZE", None), 26214400),
        (getattr(resource, "RLIMIT_NOFILE", None), 1024),
    )
    for kind, requested in limits:
        if kind is None:
            continue
        soft, hard = resource.getrlimit(kind)
        bounded = min(
            requested,
            hard if hard != resource.RLIM_INFINITY else requested,
        )
        resource.setrlimit(kind, (bounded, bounded))


def _sandbox_runtime_environment() -> dict[str, str]:
    """Return the exact macOS capture environment allowed inside Seatbelt."""

    return {
        "LC_ALL": "C",
        "MIB_MAX_WORKERS": "4",
        "MKL_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "PATH": _SANDBOX_PATH,
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _sandbox_denial_probe(path: Path, *, label: str) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except PermissionError:
        return
    except OSError as exc:
        if exc.errno in {1, 13}:
            return
        raise RuntimeError(
            f"sandbox {label} denial could not be verified"
        ) from exc
    else:
        os.close(descriptor)
    raise RuntimeError(f"sandbox did not deny {label}")


def _verify_enforced_sandbox(source_root: Path) -> None:
    """Actively prove the two critical Seatbelt denials inside the child."""

    if _ARCHIVE_CONTRACT is None:
        raise RuntimeError("sandbox child contract is unavailable")
    _sandbox_denial_probe(
        Path(str(_ARCHIVE_CONTRACT["sandbox_canary_path"])),
        label="canonical truth/repository access",
    )
    for raw in _ARCHIVE_CONTRACT["denied_sensitive_paths"]:
        _sandbox_denial_probe(
            Path(str(raw)),
            label="external sensitive input access",
        )
    for relative in _FORBIDDEN_ARCHIVE_PATHS:
        if (source_root / relative).exists():
            raise RuntimeError(
                "filtered source archive still contains truth/label bytes"
            )
    expected_environment = _sandbox_runtime_environment()
    if dict(os.environ) != expected_environment:
        raise RuntimeError(
            "sandbox runtime environment is not the exact allowlist"
        )
    probe: socket.socket | None = None
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.05)
        probe.connect(("127.0.0.1", 9))
    except PermissionError:
        return
    except OSError as exc:
        if exc.errno in {1, 13}:
            return
        raise RuntimeError("sandbox network denial is not enforced") from exc
    finally:
        if probe is not None:
            probe.close()
    raise RuntimeError("sandbox network denial is not enforced")


def _bootstrap_archive_child(arguments: Sequence[str]) -> int:
    """Run the CLI from a fresh read-only archive before production imports."""

    if _ARCHIVE_CONTRACT_FD_ENV in os.environ:
        print(
            "grouped policy capture error: caller-supplied child mode is forbidden",
            file=sys.stderr,
        )
        return 1
    if any(name in os.environ for name in _LEGACY_ARCHIVE_ENV_KEYS):
        print(
            "grouped policy capture error: caller-supplied archive state is forbidden",
            file=sys.stderr,
        )
        return 1
    try:
        revision_index = list(arguments).index("--source-revision-sha") + 1
        revision = str(arguments[revision_index]).strip().casefold()
    except (ValueError, IndexError):
        print(
            "grouped policy capture error: --source-revision-sha is required",
            file=sys.stderr,
        )
        return 1
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        print(
            "grouped policy capture error: source revision must be a full SHA",
            file=sys.stderr,
        )
        return 1
    try:
        input_index = list(arguments).index("--input-dir") + 1
        input_root = Path(arguments[input_index]).resolve(strict=True)
    except (ValueError, IndexError, OSError, RuntimeError):
        print(
            "grouped policy capture error: --input-dir must be canonical",
            file=sys.stderr,
        )
        return 1
    try:
        denied_sensitive_paths = tuple(
            Path(arguments[index + 1]).resolve(strict=True)
            for index, value in enumerate(arguments)
            if value == "--denied-sensitive-path"
        )
    except (IndexError, OSError, RuntimeError):
        denied_sensitive_paths = ()
    if (
        not denied_sensitive_paths
        or len(set(denied_sensitive_paths)) != len(denied_sensitive_paths)
        or any(
            not path.is_file() or path.is_symlink()
            for path in denied_sensitive_paths
        )
    ):
        print(
            "grouped policy capture error: at least one canonical sensitive "
            "deny path is required",
            file=sys.stderr,
        )
        return 1
    original_root = Path(__file__).resolve().parents[1]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update({"GIT_NO_REPLACE_OBJECTS": "1", "LC_ALL": "C"})

    def git(*values: str) -> bytes:
        completed = subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", *values],
            cwd=original_root,
            env=environment,
            check=False,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError("Git archive source could not be verified")
        return completed.stdout

    try:
        bootstrap_head = (
            git("rev-parse", "HEAD").decode("ascii").strip().casefold()
        )
        if (
            git(
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            )
            or git("rev-parse", "--is-shallow-repository").strip()
            != b"false"
            or git(
                "for-each-ref",
                "--format=%(refname)",
                "refs/replace/",
            )
        ):
            raise RuntimeError(
                "archive capture requires an exact clean ordinary checkout"
            )
        if (
            git(
                "cat-file",
                "blob",
                f"{revision}:devtools/policy_grouped_capture.py",
            )
            != Path(__file__).read_bytes()
        ):
            raise RuntimeError(
                "capture bootstrap tool differs from the requested revision"
            )
        index_flags = git("ls-files", "-v", "-z").decode("utf-8")
        if any(
            len(entry) < 3 or entry[0] != "H" or entry[1] != " "
            for entry in index_flags.split("\0")
            if entry
        ):
            raise RuntimeError(
                "archive capture forbids exceptional Git index flags"
            )
        staged_entries = git("ls-files", "--stage", "-z").decode("utf-8")
        for entry in staged_entries.split("\0"):
            if not entry:
                continue
            metadata = entry.split("\t", 1)[0].split()
            if (
                len(metadata) != 3
                or metadata[0] not in {"100644", "100755"}
                or metadata[2] != "0"
            ):
                raise RuntimeError(
                    "archive capture requires ordinary stage-zero files"
                )
        graft_name = git(
            "rev-parse", "--git-path", "info/grafts"
        ).decode("utf-8").strip()
        graft_path = Path(graft_name)
        if not graft_path.is_absolute():
            graft_path = original_root / graft_path
        if graft_path.exists() and graft_path.stat().st_size:
            raise RuntimeError("archive capture forbids grafted history")
        archive = git("archive", "--format=tar", revision)
        archive_sha256 = hashlib.sha256(archive).hexdigest()
        with tempfile.TemporaryDirectory(
            prefix="mib-wo17-archive-"
        ) as temporary_name:
            archive_root = Path(temporary_name) / "source"
            archive_root.mkdir(mode=0o700)
            seen: set[str] = set()
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
                for member in bundle.getmembers():
                    pure = Path(member.name)
                    if (
                        pure.is_absolute()
                        or "\\" in member.name
                        or any(part in {"", ".", ".."} for part in pure.parts)
                        or member.name in seen
                        or not (member.isdir() or member.isfile())
                    ):
                        raise RuntimeError(
                            "Git archive contains an unsafe entry"
                        )
                    seen.add(member.name)
                    target = archive_root.joinpath(*pure.parts)
                    if pure.as_posix() in _FORBIDDEN_ARCHIVE_PATHS:
                        continue
                    if member.isdir():
                        target.mkdir(mode=0o700, parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(
                        mode=0o700, parents=True, exist_ok=True
                    )
                    source = bundle.extractfile(member)
                    if source is None:
                        raise RuntimeError(
                            "Git archive regular file has no bytes"
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
                            handle.flush()
                            os.fsync(handle.fileno())
                    finally:
                        os.close(descriptor)
            digest = _early_tree_sha256(archive_root)
            for path in sorted(
                archive_root.rglob("*"),
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                os.chmod(path, 0o444 if path.is_file() else 0o555)
            os.chmod(archive_root, 0o555)
            canary = original_root / "data" / "train_labels.csv"
            if (
                not canary.is_file()
                or canary.is_symlink()
                or canary.resolve(strict=True) != canary
            ):
                raise RuntimeError(
                    "canonical truth canary is unavailable for sandbox proof"
                )
            try:
                sandbox_metadata = _SANDBOX_EXECUTABLE.resolve(
                    strict=True
                ).stat()
            except OSError as exc:
                raise RuntimeError(
                    "trusted no-network sandbox backend is unavailable"
                ) from exc
            if not stat.S_ISREG(sandbox_metadata.st_mode) or not os.access(
                _SANDBOX_EXECUTABLE, os.X_OK
            ):
                raise RuntimeError(
                    "trusted no-network sandbox backend is unavailable"
                )
            profile = _sandbox_profile(
                origin_root=original_root,
                source_root=archive_root,
                input_root=input_root,
                denied_sensitive_paths=denied_sensitive_paths,
            )
            profile_sha256 = hashlib.sha256(
                profile.encode("utf-8")
            ).hexdigest()
            child_environment = _sandbox_runtime_environment()
            contract = {
                "archive_sha256": archive_sha256,
                "bootstrap_pid": os.getpid(),
                "denied_sensitive_paths": [
                    str(path) for path in denied_sensitive_paths
                ],
                "origin_root": str(original_root),
                "revision": revision,
                "sandbox_backend": _SANDBOX_BACKEND,
                "sandbox_canary_path": str(canary),
                "sandbox_profile_sha256": profile_sha256,
                "schema": "mib-wo17-archive-child/v2",
                "source_root": str(archive_root),
                "tree_sha256": digest,
            }
            read_fd, write_fd = os.pipe()
            try:
                os.set_inheritable(read_fd, True)
                with os.fdopen(write_fd, "wb", closefd=True) as handle:
                    handle.write(_early_canonical_bytes(contract))
                    handle.flush()
                write_fd = -1
                child_environment[_ARCHIVE_CONTRACT_FD_ENV] = str(read_fd)
                child = subprocess.run(
                    [
                        str(_SANDBOX_EXECUTABLE),
                        "-p",
                        profile,
                        sys.executable,
                        "-I",
                        "-B",
                        str(
                            archive_root
                            / "devtools"
                            / "policy_grouped_capture.py"
                        ),
                        _ARCHIVE_CHILD_FLAG,
                        *arguments,
                    ],
                    cwd=archive_root,
                    env=child_environment,
                    check=False,
                    capture_output=True,
                    pass_fds=(read_fd,),
                    preexec_fn=_sandbox_limit_preexec,
                    timeout=30100,
                )
                if (
                    git("rev-parse", "HEAD")
                    .decode("ascii")
                    .strip()
                    .casefold()
                    != bootstrap_head
                    or hashlib.sha256(
                        git("archive", "--format=tar", revision)
                    ).hexdigest()
                    != archive_sha256
                ):
                    raise RuntimeError(
                        "archive origin changed during sandboxed capture"
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
                        value
                        for value in archive_root.rglob("*")
                        if value.is_dir()
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
        UnicodeDecodeError,
        tarfile.TarError,
    ) as exc:
        print(f"grouped policy capture error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__" and _ARCHIVE_CONTRACT is None:
    raise SystemExit(_bootstrap_archive_child(sys.argv[1:]))


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.experiment_control import (  # noqa: E402
    ExperimentControlError,
    canonical_json,
    require_aggregate_only,
)
from devtools.grouped_split_evidence import (  # noqa: E402
    _input_tree_sha256,
    _strict_freezer_manifest,
    _verify_input_tree_and_recomputed_manifest,
)
from devtools.policy_grouped_capture_contract import (  # noqa: E402
    AUDIT_ROOT_KEYS as _AUDIT_ROOT_KEYS,
    CAPTURE_REPEAT_COUNT,
    CONTRACT_CHECK_NAMES as _CONTRACT_CHECK_NAMES,
    MATCHER_COUNT_NAMES as _MATCHER_COUNT_NAMES,
    MAX_WORKERS,
    OBSERVATION_ROOT_KEYS as _OBSERVATION_ROOT_KEYS,
    POLICY_AUDIT_COUNT_NAMES as _CONTRACT_POLICY_AUDIT_COUNT_NAMES,
    PredictionContractError,
    SANDBOX_BACKEND as _CONTRACT_SANDBOX_BACKEND,
    grouped_producer_graph_sha256 as producer_graph_sha256,
    validate_prediction_bytes as _contract_validate_prediction_bytes,
)
from devtools.policy_revalidation_audit_contract import (  # noqa: E402
    CONTRACT_AUDIT_COUNTS,
)
from devtools.policy_revalidation_contract_probe import (  # noqa: E402
    _candidate,
    _outcome,
    _output_candidates,
    _resolved,
    run_contract_probes,
)
from mib_pipeline import BatchRunner, build_production_processor  # noqa: E402
from mib_pipeline.batch import discover_case_pdfs  # noqa: E402
from mib_pipeline.decision_recovery import (  # noqa: E402
    POLICY_AUDIT_COUNT_NAMES as _RUNTIME_POLICY_AUDIT_COUNT_NAMES,
    ReviewDenialRecoveryAdjudicator,
)
from mib_pipeline.extraction import EvidenceType  # noqa: E402


_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PolicyGroupedCaptureError(RuntimeError):
    """The truth-blind grouped production capture failed closed."""


if (
    tuple(_RUNTIME_POLICY_AUDIT_COUNT_NAMES)
    != _CONTRACT_POLICY_AUDIT_COUNT_NAMES
):
    raise RuntimeError(
        "runtime policy audit counters differ from the frozen capture contract"
    )
if _SANDBOX_BACKEND != _CONTRACT_SANDBOX_BACKEND:
    raise RuntimeError(
        "bootstrap sandbox backend differs from the frozen capture contract"
    )
POLICY_AUDIT_COUNT_NAMES = _CONTRACT_POLICY_AUDIT_COUNT_NAMES


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json(dict(value)) + "\n").encode("utf-8")


def _git_environment() -> dict[str, str]:
    """Return a minimal environment without caller-controlled Git routing."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment["LC_ALL"] = "C"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment


def verify_clean_source_revision(
    source_revision_sha: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Require an exact clean committed checkout using the system Git binary."""

    revision = str(source_revision_sha).strip().casefold()
    if not _COMMIT_RE.fullmatch(revision):
        raise PolicyGroupedCaptureError(
            "source revision must be a full lowercase Git commit SHA"
        )
    if _ARCHIVE_CONTRACT is not None:
        expected_root = Path(
            _ARCHIVE_CONTRACT["source_root"]
        ).resolve(strict=True)
        origin_root = Path(str(_ARCHIVE_CONTRACT["origin_root"]))
        expected_digest = _ARCHIVE_CONTRACT["tree_sha256"]
        expected_archive = _ARCHIVE_CONTRACT["archive_sha256"]
        expected_revision = _ARCHIVE_CONTRACT["revision"]
        _verify_enforced_sandbox(expected_root)
        if (
            Path(__file__).resolve().parents[1] != expected_root
            or repo_root.resolve(strict=True) != expected_root
            or origin_root == expected_root
            or revision != expected_revision
            or not _SHA256_RE.fullmatch(expected_digest)
            or not _SHA256_RE.fullmatch(expected_archive)
            or _early_tree_sha256(expected_root) != expected_digest
        ):
            raise PolicyGroupedCaptureError(
                "immutable Git archive source binding failed"
            )
        return
    try:
        head = subprocess.run(
            ["/usr/bin/git", "rev-parse", "HEAD"],
            cwd=repo_root,
            env=_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().casefold()
        status_output = subprocess.run(
            ["/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repo_root,
            env=_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        index_flags = subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", "ls-files", "-v", "-z"],
            cwd=repo_root,
            env=_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        staged_entries = subprocess.run(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "ls-files",
                "--stage",
                "-z",
            ],
            cwd=repo_root,
            env=_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        shallow = subprocess.run(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "rev-parse",
                "--is-shallow-repository",
            ],
            cwd=repo_root,
            env=_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        replacement_refs = subprocess.run(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "for-each-ref",
                "--format=%(refname)",
                "refs/replace/",
            ],
            cwd=repo_root,
            env=_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        graft_name = subprocess.run(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "rev-parse",
                "--git-path",
                "info/grafts",
            ],
            cwd=repo_root,
            env=_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PolicyGroupedCaptureError(
            "cannot verify the production source revision"
        ) from exc
    if head != revision:
        raise PolicyGroupedCaptureError(
            "capture source revision does not match repository HEAD"
        )
    if status_output:
        raise PolicyGroupedCaptureError(
            "production capture requires a clean committed checkout"
        )
    if any(
        len(entry) < 3 or entry[0] != "H" or entry[1] != " "
        for entry in index_flags.split("\0")
        if entry
    ):
        raise PolicyGroupedCaptureError(
            "production capture forbids exceptional Git index flags"
        )
    for entry in staged_entries.split("\0"):
        if not entry:
            continue
        metadata = entry.split("\t", 1)[0].split()
        if (
            len(metadata) != 3
            or metadata[0] not in {"100644", "100755"}
            or metadata[2] != "0"
        ):
            raise PolicyGroupedCaptureError(
                "production capture requires ordinary stage-zero files"
            )
    if shallow != "false" or replacement_refs:
        raise PolicyGroupedCaptureError(
            "production capture forbids shallow or replaced history"
        )
    graft_path = Path(graft_name)
    if not graft_path.is_absolute():
        graft_path = repo_root / graft_path
    try:
        graft_present = graft_path.exists() and graft_path.stat().st_size > 0
    except OSError as exc:
        raise PolicyGroupedCaptureError(
            "cannot verify Git graft state"
        ) from exc
    if graft_present:
        raise PolicyGroupedCaptureError(
            "production capture forbids grafted history"
        )


def _peak_memory_mib() -> float:
    usages = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )
    divisor = (
        1024.0
        if sys.platform.startswith("linux")
        else 1024.0 * 1024.0
    )
    return max(float(value) / divisor for value in usages)


def _cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (
        own.ru_utime
        + own.ru_stime
        + children.ru_utime
        + children.ru_stime
    )


def _validate_policy_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(
        POLICY_AUDIT_COUNT_NAMES
    ):
        raise PolicyGroupedCaptureError(
            "accepted result policy counts do not match the frozen contract"
        )
    result: dict[str, int] = {}
    for name in POLICY_AUDIT_COUNT_NAMES:
        count = value[name]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise PolicyGroupedCaptureError(
                f"policy counter {name} must be non-negative"
            )
        result[name] = count
    return result


class _AcceptedPolicyCollector:
    """Collect one historical policy mapping per accepted final output."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: list[dict[str, int]] = []
        self._errors: list[str] = []

    def record(self, value: Any) -> None:
        try:
            normalized = _validate_policy_counts(value)
        except PolicyGroupedCaptureError as exc:
            with self._lock:
                self._errors.append(str(exc))
            return
        with self._lock:
            self._values.append(normalized)

    def aggregate(self, *, expected_count: int) -> dict[str, int]:
        with self._lock:
            values = tuple(dict(value) for value in self._values)
            errors = tuple(self._errors)
        if errors:
            raise PolicyGroupedCaptureError(errors[0])
        if len(values) != expected_count:
            raise PolicyGroupedCaptureError(
                "accepted policy result count does not match output population"
            )
        aggregate = {
            name: sum(value[name] for value in values)
            for name in POLICY_AUDIT_COUNT_NAMES
        }
        aggregate["accepted_final_policy_result_count"] = len(values)
        return aggregate


class _ExactMatcherObserver:
    """Classify actual rewrites without changing their production result."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts = {name: 0 for name in _MATCHER_COUNT_NAMES}

    def observe(
        self,
        *,
        allow_approval_recovery: bool,
        matching_rules: Sequence[str],
        baseline: Any,
        result: Any,
    ) -> None:
        baseline_review = bool(
            baseline.row.adjudication == "NEEDS_REVIEW"
            and baseline.trace.decision == "NEEDS_REVIEW"
        )
        result_approval = bool(
            result.row.adjudication == "APPROVED"
            and result.trace.decision == "APPROVED"
        )
        with self._lock:
            if (
                baseline_review
                and allow_approval_recovery
                and matching_rules
            ):
                self._counts["eligible_guarded_initial_count"] += 1
            if not baseline_review or not result_approval:
                return
            if not allow_approval_recovery:
                self._counts["late_revalidation_approval_count"] += 1
            elif matching_rules:
                self._counts["guarded_initial_approval_count"] += 1
            else:
                self._counts["unguarded_initial_approval_count"] += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)


def _instrument_processor(
    production: Any,
) -> tuple[
    _AcceptedPolicyCollector,
    _ExactMatcherObserver,
    Callable[[], None],
]:
    """Attach truth-blind observers and return a restoration callback."""

    rapid = getattr(production, "processor", None)
    adjudicator = getattr(rapid, "_adjudicator", None)
    if (
        rapid is None
        or not isinstance(
            adjudicator, ReviewDenialRecoveryAdjudicator
        )
    ):
        raise PolicyGroupedCaptureError(
            "production graph does not expose the WO-17 adjudicator"
        )
    policy_collector = _AcceptedPolicyCollector()
    matcher_observer = _ExactMatcherObserver()
    previous_policy_observer = getattr(
        rapid, "_policy_audit_observer", None
    )
    policy_observer_existed = hasattr(rapid, "_policy_audit_observer")
    original_apply = adjudicator._apply_synthetic_rules

    def observed_apply(
        resolved_case: Any,
        baseline: Any,
        *,
        allow_approval_recovery: bool = True,
    ) -> Any:
        matching_rules = tuple(
            adjudicator._matching_approval_rules(
                resolved_case,
                baseline,
            )
        )
        result = original_apply(
            resolved_case,
            baseline,
            allow_approval_recovery=allow_approval_recovery,
        )
        matcher_observer.observe(
            allow_approval_recovery=allow_approval_recovery,
            matching_rules=matching_rules,
            baseline=baseline,
            result=result,
        )
        return result

    try:
        setattr(rapid, "_policy_audit_observer", policy_collector.record)
        setattr(adjudicator, "_apply_synthetic_rules", observed_apply)
    except (AttributeError, TypeError) as exc:
        raise PolicyGroupedCaptureError(
            "production policy graph cannot be instrumented"
        ) from exc

    def restore() -> None:
        setattr(adjudicator, "_apply_synthetic_rules", original_apply)
        if policy_observer_existed:
            setattr(
                rapid,
                "_policy_audit_observer",
                previous_policy_observer,
            )
        else:
            try:
                delattr(rapid, "_policy_audit_observer")
            except AttributeError:
                pass

    return policy_collector, matcher_observer, restore


def _complete_xw1_fixture() -> tuple[Any, Any]:
    output = _output_candidates(overrides={"visa_class": "XW-1"})
    marker = _candidate(
        "page_type_present_sponsor_attestation",
        "present",
        evidence_type=EvidenceType.SPONSOR_ATTESTATION,
        cues=("packet_page_type:sponsor_attestation",),
        page_index=20,
        left=20,
    )
    return _resolved(*output, marker), _outcome(confidence=0.20)


def matcher_contract_checks() -> Mapping[str, bool]:
    """Exercise exact matcher vetoes without asserting candidate activation."""

    resolved, review = _complete_xw1_fixture()
    adjudicator = ReviewDenialRecoveryAdjudicator(
        type(
            "_FixedReview",
            (),
            {"adjudicate_case": lambda self, case: review},
        )()
    )

    def matches(case: Any, outcome: Any = review) -> bool:
        return bool(
            adjudicator._matching_approval_rules(case, outcome)
        )

    output = list(_output_candidates(overrides={"visa_class": "XW-1"}))
    marker = _candidate(
        "page_type_present_sponsor_attestation",
        "present",
        evidence_type=EvidenceType.SPONSOR_ATTESTATION,
        cues=("packet_page_type:sponsor_attestation",),
        page_index=20,
        left=20,
    )

    def replace_output(field_name: str, replacement: Any) -> Any:
        return _resolved(
            *(
                replacement
                if candidate.field_name == field_name
                else candidate
                for candidate in output
            ),
            marker,
        )

    wrong_case_marker = replace(marker, case_id_hint="MIB-999999")
    wrong_applicant_marker = replace(
        marker,
        applicant_hint="Different Applicant",
        ocr_provenance=(),
    )
    authoritative = replace(
        review,
        trace=replace(
            review.trace,
            authoritative_source=True,
            review_reasons=("authoritative_visible_decision",),
        ),
    )
    explicit_denial = _outcome(
        denial_reasons=("ordinary_policy_denial",),
        confidence=0.20,
    )
    default_case = _resolved(
        *(candidate for candidate in output if candidate.field_name != "home_world"),
        marker,
    )
    placeholder_case = replace_output(
        "home_world",
        _candidate(
            "home_world",
            "unknown",
            cues=("synthetic_default",),
        ),
    )
    sentinel_case = replace_output(
        "sponsor_id",
        _candidate(
            "sponsor_id",
            "SPN-0000",
            cues=("synthetic_default",),
        ),
    )
    text_layer_case = replace_output(
        "home_world",
        _candidate(
            "home_world",
            "Earth",
            evidence_type=EvidenceType.TEXT_LAYER,
        ),
    )
    superseded_case = replace_output(
        "home_world",
        replace(
            _candidate("home_world", "Earth"),
            superseded=True,
        ),
    )
    struck_case = replace_output(
        "home_world",
        _candidate(
            "home_world",
            "Earth",
            cues=("strikethrough",),
        ),
    )
    late = adjudicator._apply_synthetic_rules(
        resolved,
        review,
        allow_approval_recovery=False,
    )
    checks = {
        "authoritative_review_veto": not matches(
            resolved, authoritative
        ),
        "explicit_denial_veto": not matches(resolved, explicit_denial),
        "late_revalidation_false_preserves_review": (
            late.row.adjudication == "NEEDS_REVIEW"
            and late.trace.decision == "NEEDS_REVIEW"
        ),
        "matcher_positive_control": matches(resolved),
        "placeholder_veto": not matches(placeholder_case),
        "sentinel_veto": not matches(sentinel_case),
        "serialization_default_veto": not matches(default_case),
        "strikethrough_veto": not matches(struck_case),
        "superseded_veto": not matches(superseded_case),
        "text_layer_veto": not matches(text_layer_case),
        "wrong_applicant_scope_veto": not matches(
            _resolved(*output, wrong_applicant_marker)
        ),
        "wrong_record_scope_veto": not matches(
            _resolved(*output, wrong_case_marker)
        ),
    }
    if set(checks) != set(_CONTRACT_CHECK_NAMES):
        raise PolicyGroupedCaptureError(
            "matcher contract check set is incomplete"
        )
    return dict(sorted(checks.items()))


def _validate_prediction_bytes(
    content: bytes,
    *,
    expected_count: int,
) -> None:
    try:
        _contract_validate_prediction_bytes(
            content,
            expected_count=expected_count,
        )
    except PredictionContractError as exc:
        raise PolicyGroupedCaptureError(str(exc)) from exc


def _run_once(
    *,
    input_dir: Path,
    expected_count: int,
    max_workers: int,
    processor_factory: Callable[[], Any],
) -> tuple[bytes, dict[str, int], dict[str, int], dict[str, float | int]]:
    production = processor_factory()
    policy_collector, matcher_observer, restore = _instrument_processor(
        production
    )
    cpu_before = _cpu_seconds()
    wall_before = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(
            prefix="mib-wo17-grouped-capture-"
        ) as temporary_name:
            prediction_path = Path(temporary_name) / "predictions.jsonl"
            report = BatchRunner(
                production, max_workers=max_workers
            ).run(input_dir, prediction_path)
            prediction_bytes = prediction_path.read_bytes()
    finally:
        restore()
    wall_seconds = time.monotonic() - wall_before
    cpu_seconds = _cpu_seconds() - cpu_before
    if (
        report.attempted != expected_count
        or report.answered != expected_count
        or report.omitted != 0
        or report.failures
    ):
        raise PolicyGroupedCaptureError(
            "production batch did not answer the exact input population"
        )
    if wall_seconds <= 0 or cpu_seconds <= 0:
        raise PolicyGroupedCaptureError(
            "production runtime clocks did not advance"
        )
    _validate_prediction_bytes(
        prediction_bytes, expected_count=expected_count
    )
    policy_counts = policy_collector.aggregate(
        expected_count=expected_count
    )
    matcher_counts = matcher_observer.snapshot()
    if set(matcher_counts) != set(_MATCHER_COUNT_NAMES):
        raise PolicyGroupedCaptureError(
            "exact matcher observer returned an incomplete count set"
        )
    metrics: dict[str, float | int] = {
        "answered_count": report.answered,
        "attempted_count": report.attempted,
        "omitted_count": report.omitted,
        "output_bytes": len(prediction_bytes),
        "peak_rss_bytes": int(_peak_memory_mib() * 1024 * 1024),
        "process_cpu_seconds": cpu_seconds,
        "runtime_seconds": wall_seconds,
    }
    return prediction_bytes, policy_counts, matcher_counts, metrics


def _audit_payload(
    *,
    arm: str,
    source_revision_sha: str,
    manifest_sha256: str,
    input_tree_sha256: str,
    graph_sha256: str,
    predictions_sha256: str,
    policy_counts: Mapping[str, int],
    matcher_counts: Mapping[str, int],
    contract_counts: Mapping[str, int],
    contract_checks: Mapping[str, bool],
) -> dict[str, Any]:
    counts = {
        **{
            f"policy_{name}": int(value)
            for name, value in policy_counts.items()
        },
        **{
            f"matcher_{name}": int(value)
            for name, value in matcher_counts.items()
        },
        **{
            f"contract_{name}": int(value)
            for name, value in contract_counts.items()
        },
    }
    payload: dict[str, Any] = {
        "evaluation_mode": "public_grouped_robustness_not_unseen",
        "evidence_label": "aggregate_only",
        "status": arm,
        "source_revision_sha": source_revision_sha,
        "layout_manifest_sha256": manifest_sha256,
        "input_tree_sha256": input_tree_sha256,
        "producer_graph_sha256": graph_sha256,
        "predictions_sha256": predictions_sha256,
        "counts": counts,
        "checks": dict(contract_checks),
    }
    if set(payload) != _AUDIT_ROOT_KEYS:
        raise PolicyGroupedCaptureError(
            "internal audit payload schema is incomplete"
        )
    require_aggregate_only(payload)
    return payload


def _validate_external_output_paths(paths: Sequence[Path]) -> None:
    resolved_repo = REPO_ROOT.resolve()
    normalized: list[Path] = []
    for raw in paths:
        path = raw.expanduser()
        if not path.is_absolute():
            raise PolicyGroupedCaptureError(
                "capture output paths must be absolute"
            )
        if path.exists() or path.is_symlink():
            raise PolicyGroupedCaptureError(
                "capture outputs are create-once and must not already exist"
            )
        parent = path.parent.resolve(strict=True)
        if parent != path.parent or path.parent.is_symlink():
            raise PolicyGroupedCaptureError(
                "capture output directories must be canonical and symlink-free"
            )
        if not parent.is_dir():
            raise PolicyGroupedCaptureError(
                "capture output parent must be a directory"
            )
        if parent == resolved_repo or resolved_repo in parent.parents:
            raise PolicyGroupedCaptureError(
                "identity-bearing capture outputs must remain external"
            )
        if path.name in {"", ".", ".."}:
            raise PolicyGroupedCaptureError(
                "capture output filename is invalid"
            )
        normalized.append(parent / path.name)
    if len(set(normalized)) != len(normalized):
        raise PolicyGroupedCaptureError(
            "capture output paths must be distinct"
        )


def _write_output_set(values: Mapping[Path, bytes]) -> None:
    """Create all outputs once with no-follow directory-relative writes."""

    paths = tuple(values)
    _validate_external_output_paths(paths)
    created: list[Path] = []
    directory_handles: list[int] = []
    try:
        for path, content in values.items():
            parent = path.parent.resolve(strict=True)
            directory_flags = os.O_RDONLY
            directory_flags |= getattr(os, "O_DIRECTORY", 0)
            directory_flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fd = os.open(parent, directory_flags)
            directory_handles.append(directory_fd)
            before = os.fstat(directory_fd)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(
                path.name,
                flags,
                0o600,
                dir_fd=directory_fd,
            )
            created.append(path)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                written = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            after = os.fstat(directory_fd)
            if (
                (before.st_dev, before.st_ino)
                != (after.st_dev, after.st_ino)
                or not stat.S_ISREG(written.st_mode)
                or written.st_nlink != 1
                or written.st_size != len(content)
            ):
                raise PolicyGroupedCaptureError(
                    "capture output binding changed during create-once write"
                )
        for path, expected in values.items():
            if path.is_symlink() or path.read_bytes() != expected:
                raise PolicyGroupedCaptureError(
                    "capture output bytes changed after write"
                )
    except BaseException:
        for path in reversed(created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for directory_fd in directory_handles:
            os.close(directory_fd)


def run_grouped_capture(
    *,
    arm: str,
    source_revision_sha: str,
    input_dir: Path | str,
    layout_manifest_path: Path | str,
    expected_layout_manifest_sha256: str,
    expected_input_tree_sha256: str,
    denied_sensitive_paths: Sequence[Path | str],
    first_predictions_path: Path | str,
    second_predictions_path: Path | str,
    first_audit_path: Path | str,
    second_audit_path: Path | str,
    observation_path: Path | str,
    max_workers: int = MAX_WORKERS,
    processor_factory: Callable[[], Any] = build_production_processor,
    source_verifier: Callable[[str], None] = verify_clean_source_revision,
    graph_provider: Callable[[], str] = producer_graph_sha256,
) -> dict[str, Any]:
    """Run two truth-blind captures and persist external bound artifacts."""

    if _ARCHIVE_CONTRACT is None:
        raise PolicyGroupedCaptureError(
            "grouped capture must run inside the verified sandbox archive child"
        )
    denied_paths = tuple(str(Path(path)) for path in denied_sensitive_paths)
    contract_denied_paths = tuple(
        str(path) for path in _ARCHIVE_CONTRACT["denied_sensitive_paths"]
    )
    if (
        not denied_paths
        or denied_paths != contract_denied_paths
        or any(
            not Path(path).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(path).parts)
            for path in denied_paths
        )
    ):
        raise PolicyGroupedCaptureError(
            "capture sensitive deny paths disagree with the enforced sandbox"
        )
    normalized_arm = str(arm).strip().casefold()
    if normalized_arm not in {"baseline", "candidate"}:
        raise PolicyGroupedCaptureError(
            "arm must be baseline or candidate"
        )
    revision = str(source_revision_sha).strip().casefold()
    if not _COMMIT_RE.fullmatch(revision):
        raise PolicyGroupedCaptureError(
            "source revision must be a full lowercase Git SHA"
        )
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or max_workers != MAX_WORKERS
    ):
        raise PolicyGroupedCaptureError(
            "max_workers must equal the frozen runtime value four"
        )
    expected_manifest = str(
        expected_layout_manifest_sha256
    ).strip().casefold()
    expected_tree = str(
        expected_input_tree_sha256
    ).strip().casefold()
    if (
        not _SHA256_RE.fullmatch(expected_manifest)
        or not _SHA256_RE.fullmatch(expected_tree)
    ):
        raise PolicyGroupedCaptureError(
            "manifest and input expectations must be full SHA-256 digests"
        )

    output_paths = tuple(
        Path(path)
        for path in (
            first_predictions_path,
            second_predictions_path,
            first_audit_path,
            second_audit_path,
            observation_path,
        )
    )
    _validate_external_output_paths(output_paths)
    source_verifier(revision)
    graph_sha = str(graph_provider()).strip().casefold()
    if not _SHA256_RE.fullmatch(graph_sha):
        raise PolicyGroupedCaptureError(
            "producer graph provider returned an invalid digest"
        )

    snapshot = _strict_freezer_manifest(
        layout_manifest_path,
        expected_sha256=expected_manifest,
    )
    observed_tree = _verify_input_tree_and_recomputed_manifest(
        Path(input_dir).resolve(strict=True),
        snapshot,
        expected_sha256=expected_tree,
    )
    expected_count = len(snapshot.manifest.case_ids)
    discovered = discover_case_pdfs(Path(input_dir))
    if len(discovered) != expected_count:
        raise PolicyGroupedCaptureError(
            "input directory does not match the manifest population"
        )

    try:
        contract_counts = dict(run_contract_probes())
    except Exception as exc:
        raise PolicyGroupedCaptureError(
            "existing WO-17 contract probes failed"
        ) from exc
    if set(contract_counts) != set(CONTRACT_AUDIT_COUNTS):
        raise PolicyGroupedCaptureError(
            "existing contract probes returned an incomplete count set"
        )
    contract_checks = dict(matcher_contract_checks())

    first = _run_once(
        input_dir=Path(input_dir),
        expected_count=expected_count,
        max_workers=max_workers,
        processor_factory=processor_factory,
    )
    second = _run_once(
        input_dir=Path(input_dir),
        expected_count=expected_count,
        max_workers=max_workers,
        processor_factory=processor_factory,
    )
    first_predictions, first_policy, first_matcher, first_metrics = first
    second_predictions, second_policy, second_matcher, second_metrics = second
    for label, metrics in (
        ("first", first_metrics),
        ("second", second_metrics),
    ):
        if (
            int(metrics["output_bytes"]) > 26214400
            or int(metrics["peak_rss_bytes"]) > 8589934592
            or float(metrics["runtime_seconds"]) > 30000
            or float(metrics["runtime_seconds"]) / expected_count > 6
        ):
            raise PolicyGroupedCaptureError(
                f"{label} capture exceeded the enforced runtime envelope"
            )
    deterministic = (
        first_predictions == second_predictions
        and first_policy == second_policy
        and first_matcher == second_matcher
    )
    first_prediction_sha = _sha256_bytes(first_predictions)
    second_prediction_sha = _sha256_bytes(second_predictions)
    first_audit = _audit_payload(
        arm=normalized_arm,
        source_revision_sha=revision,
        manifest_sha256=snapshot.manifest.sha256,
        input_tree_sha256=observed_tree,
        graph_sha256=graph_sha,
        predictions_sha256=first_prediction_sha,
        policy_counts=first_policy,
        matcher_counts=first_matcher,
        contract_counts=contract_counts,
        contract_checks=contract_checks,
    )
    second_audit = _audit_payload(
        arm=normalized_arm,
        source_revision_sha=revision,
        manifest_sha256=snapshot.manifest.sha256,
        input_tree_sha256=observed_tree,
        graph_sha256=graph_sha,
        predictions_sha256=second_prediction_sha,
        policy_counts=second_policy,
        matcher_counts=second_matcher,
        contract_counts=contract_counts,
        contract_checks=contract_checks,
    )
    first_audit_bytes = _canonical_bytes(first_audit)
    second_audit_bytes = _canonical_bytes(second_audit)

    # Recheck all mutable bindings after both long-running passes.
    source_verifier(revision)
    final_graph_sha = str(graph_provider()).strip().casefold()
    if final_graph_sha != graph_sha:
        raise PolicyGroupedCaptureError(
            "producer graph changed during capture"
        )
    final_snapshot = _strict_freezer_manifest(
        layout_manifest_path,
        expected_sha256=expected_manifest,
    )
    final_tree = _verify_input_tree_and_recomputed_manifest(
        Path(input_dir).resolve(strict=True),
        final_snapshot,
        expected_sha256=expected_tree,
    )
    final_discovered = discover_case_pdfs(Path(input_dir))
    if (
        final_tree != expected_tree
        or _input_tree_sha256(final_discovered) != expected_tree
        or len(final_discovered) != expected_count
        or final_snapshot.manifest.case_ids != snapshot.manifest.case_ids
        or _sha256_file(layout_manifest_path) != expected_manifest
    ):
        raise PolicyGroupedCaptureError(
            "input tree or layout manifest changed during capture"
        )

    observation: dict[str, Any] = {
        "evaluation_mode": "public_grouped_robustness_not_unseen",
        "evidence_label": "aggregate_only",
        "status": normalized_arm,
        "source_revision_sha": revision,
        "layout_manifest_sha256": expected_manifest,
        "input_tree_sha256": expected_tree,
        "max_worker_count": max_workers,
        "producer_graph_sha256": graph_sha,
        "sandbox_backend_sha256": _sha256_bytes(
            str(_ARCHIVE_CONTRACT["sandbox_backend"]).encode("utf-8")
        ),
        "sandbox_policy_sha256": str(
            _ARCHIVE_CONTRACT["sandbox_profile_sha256"]
        ),
        "source_archive_sha256": str(
            _ARCHIVE_CONTRACT["archive_sha256"]
        ),
        "source_tree_sha256": str(_ARCHIVE_CONTRACT["tree_sha256"]),
        "capture_tool_sha256": _sha256_file(Path(__file__)),
        "first_predictions_sha256": first_prediction_sha,
        "second_predictions_sha256": second_prediction_sha,
        "first_audit_sha256": _sha256_bytes(first_audit_bytes),
        "second_audit_sha256": _sha256_bytes(second_audit_bytes),
        "record_count": expected_count,
        "repeat_count": CAPTURE_REPEAT_COUNT,
        "deterministic": deterministic,
        "checks": {
            "audit_deterministic": (
                first_policy == second_policy
                and first_matcher == second_matcher
            ),
            "byte_deterministic": (
                first_predictions == second_predictions
            ),
            "input_recomputed": observed_tree == expected_tree,
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
            "first_answered_count": int(
                first_metrics["answered_count"]
            ),
            "first_attempted_count": int(
                first_metrics["attempted_count"]
            ),
            "first_omitted_count": int(first_metrics["omitted_count"]),
            "second_answered_count": int(
                second_metrics["answered_count"]
            ),
            "second_attempted_count": int(
                second_metrics["attempted_count"]
            ),
            "second_omitted_count": int(
                second_metrics["omitted_count"]
            ),
            "label_access_count": 0,
        },
        "metrics": {
            "first_output_bytes": int(first_metrics["output_bytes"]),
            "first_peak_rss_bytes": int(
                first_metrics["peak_rss_bytes"]
            ),
            "first_process_cpu_seconds": float(
                first_metrics["process_cpu_seconds"]
            ),
            "first_runtime_seconds": float(
                first_metrics["runtime_seconds"]
            ),
            "second_output_bytes": int(second_metrics["output_bytes"]),
            "second_peak_rss_bytes": int(
                second_metrics["peak_rss_bytes"]
            ),
            "second_process_cpu_seconds": float(
                second_metrics["process_cpu_seconds"]
            ),
            "second_runtime_seconds": float(
                second_metrics["runtime_seconds"]
            ),
        },
    }
    if set(observation) != _OBSERVATION_ROOT_KEYS:
        raise PolicyGroupedCaptureError(
            "internal observation schema is incomplete"
        )
    require_aggregate_only(observation)
    observation_bytes = _canonical_bytes(observation)
    _write_output_set(
        {
            output_paths[0]: first_predictions,
            output_paths[1]: second_predictions,
            output_paths[2]: first_audit_bytes,
            output_paths[3]: second_audit_bytes,
            output_paths[4]: observation_bytes,
        }
    )
    return observation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture two truth-blind full production runs for grouped WO-17 "
            "policy evidence."
        )
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-layout-manifest-sha256", required=True
    )
    parser.add_argument("--expected-input-tree-sha256", required=True)
    parser.add_argument(
        "--denied-sensitive-path",
        type=Path,
        action="append",
        required=True,
        help=(
            "Existing truth/label path that the sandbox must prove unreadable; "
            "the file is never opened by capture."
        ),
    )
    parser.add_argument("--first-predictions", type=Path, required=True)
    parser.add_argument("--second-predictions", type=Path, required=True)
    parser.add_argument("--first-audit", type=Path, required=True)
    parser.add_argument("--second-audit", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        observation = run_grouped_capture(
            arm=arguments.arm,
            source_revision_sha=arguments.source_revision_sha,
            input_dir=arguments.input_dir,
            layout_manifest_path=arguments.layout_manifest,
            expected_layout_manifest_sha256=(
                arguments.expected_layout_manifest_sha256
            ),
            expected_input_tree_sha256=(
                arguments.expected_input_tree_sha256
            ),
            denied_sensitive_paths=arguments.denied_sensitive_path,
            first_predictions_path=arguments.first_predictions,
            second_predictions_path=arguments.second_predictions,
            first_audit_path=arguments.first_audit,
            second_audit_path=arguments.second_audit,
            observation_path=arguments.observation,
            max_workers=arguments.max_workers,
        )
    except (
        ExperimentControlError,
        OSError,
        PolicyGroupedCaptureError,
    ) as exc:
        print(f"grouped policy capture error: {exc}", file=sys.stderr)
        return 1
    print(
        "WO-17 grouped capture: "
        f"{observation['status']} "
        f"records={observation['record_count']} "
        f"deterministic={str(observation['deterministic']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
