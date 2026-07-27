#!/usr/bin/env python3
"""Build, constrain, repeat, and measure one offline Docker submission.

The aggregate evidence emitted by this harness deliberately contains counts,
timings, sizes, and output hashes only.  It never persists case identifiers,
filenames, prediction rows, truth, or labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts import validate_submission
except ImportError:  # Direct execution adds scripts/, not the repository root.
    import validate_submission  # type: ignore[no-redef]


EVIDENCE_SCHEMA = "mib-wo20-docker-runtime-envelope/v1"
PEAK_RSS_SOURCE = "in_container_procfs_summed_process_tree_vmrss"
PEAK_CONTAINER_MEMORY_SOURCE = "docker_stats_mem_usage_cgroup"
IMAGE_MODEL_SCAN_SCOPE = "fixed_runtime_roots"
MODEL_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".h5",
    ".joblib",
    ".mar",
    ".mlmodel",
    ".onnx",
    ".pb",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
    ".traineddata",
}
IMAGE_MODEL_EVIDENCE_EXTENSIONS = MODEL_EXTENSIONS | {".json"}
_MEMORY_RE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b)\b",
    re.IGNORECASE,
)
_MEMORY_MULTIPLIERS = {
    "b": 1,
    "kb": 1000,
    "kib": 1024,
    "mb": 1000**2,
    "mib": 1024**2,
    "gb": 1000**3,
    "gib": 1024**3,
    "tb": 1000**4,
    "tib": 1024**4,
}


class RuntimeEnvelopeError(RuntimeError):
    """A Docker/runtime constraint could not be proven."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class OutputSummary:
    attempted: int
    answered: int
    omitted: int
    invalid: int
    rows_emitted: int

    def __post_init__(self) -> None:
        values = (
            self.attempted,
            self.answered,
            self.omitted,
            self.invalid,
            self.rows_emitted,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        ):
            raise RuntimeEnvelopeError(
                "invalid_output_counts",
                "output counts must be non-negative integers",
            )
        if self.answered + self.omitted != self.attempted:
            raise RuntimeEnvelopeError(
                "invalid_output_counts",
                "answered plus omitted must equal attempted",
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "attempted": self.attempted,
            "answered": self.answered,
            "omitted": self.omitted,
            "invalid": self.invalid,
            "rows_emitted": self.rows_emitted,
        }


@dataclass(frozen=True)
class ContainerRun:
    repeat_index: int
    elapsed_seconds: float
    peak_process_tree_rss_bytes: int
    peak_container_memory_bytes: int
    output_sha256: str
    output_bytes: int
    output: OutputSummary

    def __post_init__(self) -> None:
        if self.repeat_index < 1:
            raise RuntimeEnvelopeError(
                "invalid_repeat_index",
                "repeat index must be positive",
            )
        if (
            not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds <= 0
        ):
            raise RuntimeEnvelopeError(
                "invalid_elapsed_time",
                "elapsed runtime must be finite and positive",
            )
        if self.peak_process_tree_rss_bytes <= 0:
            raise RuntimeEnvelopeError(
                "peak_rss_unavailable",
                "container process-tree RSS was not observed",
            )
        if self.peak_container_memory_bytes <= 0:
            raise RuntimeEnvelopeError(
                "peak_container_memory_unavailable",
                "Docker container memory was not observed",
            )
        if not re.fullmatch(r"[0-9a-f]{64}", self.output_sha256):
            raise RuntimeEnvelopeError(
                "invalid_output_hash",
                "output SHA-256 is invalid",
            )
        if self.output_bytes < 0:
            raise RuntimeEnvelopeError(
                "invalid_output_size",
                "output size cannot be negative",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "repeat_index": self.repeat_index,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_process_tree_rss_bytes": self.peak_process_tree_rss_bytes,
            "peak_process_tree_rss_mib": (
                self.peak_process_tree_rss_bytes / (1024 * 1024)
            ),
            "peak_process_tree_rss_source": PEAK_RSS_SOURCE,
            "peak_container_memory_bytes": self.peak_container_memory_bytes,
            "peak_container_memory_mib": (
                self.peak_container_memory_bytes / (1024 * 1024)
            ),
            "peak_container_memory_source": PEAK_CONTAINER_MEMORY_SOURCE,
            "output_sha256": self.output_sha256,
            "output_bytes": self.output_bytes,
            "coverage": self.output.to_dict(),
        }


def run(
    cmd: Sequence[str],
    *,
    timeout: float | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(
        list(cmd),
        cwd=cwd,
        timeout=timeout,
        check=True,
        text=True,
    )


def docker_output(
    cmd: Sequence[str],
    *,
    timeout: float = 15.0,
) -> str:
    return subprocess.check_output(
        list(cmd),
        text=True,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    ).strip()


def docker_status() -> tuple[bool, str]:
    """Return daemon availability and a non-host-identifying version string."""

    if shutil.which("docker") is None:
        return False, "docker_cli_missing"
    try:
        version = docker_output(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "docker_daemon_unavailable"
    if not version:
        return False, "docker_server_version_unavailable"
    return True, version


def image_size_bytes(image_tag: str) -> int:
    try:
        raw = docker_output(
            [
                "docker",
                "image",
                "inspect",
                image_tag,
                "--format",
                "{{.Size}}",
            ]
        )
        size = int(raw)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeEnvelopeError(
            "image_inspect_failed",
            "Docker image size could not be inspected",
        ) from exc
    if size <= 0:
        raise RuntimeEnvelopeError(
            "invalid_image_size",
            "Docker image size must be positive",
        )
    return size


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_tree_sha256(root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        {item.resolve() for item in paths},
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        if not path.is_file():
            raise RuntimeEnvelopeError(
                "source_binding_missing",
                "a source-binding file is absent",
            )
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_bindings(repo: Path) -> dict[str, object]:
    """Bind evidence to one clean Git revision and exact producer graph."""

    try:
        status = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
        revision = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip().casefold()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeEnvelopeError(
            "git_source_binding_failed",
            "Git source binding could not be established",
        ) from exc
    if status.strip():
        raise RuntimeEnvelopeError(
            "git_worktree_not_clean",
            "runtime evidence requires a clean Git worktree",
        )
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeEnvelopeError(
            "git_source_binding_failed",
            "Git revision is not a full commit SHA",
        )

    required = {
        "dockerfile_sha256": repo / "Dockerfile",
        "requirements_lock_sha256": repo / "requirements.lock",
        "run_sh_sha256": repo / "run.sh",
        "solution_sha256": repo / "solution.py",
        "harness_sha256": repo / "scripts" / "run_docker_submission.py",
    }
    individual = {
        name: _sha256_file(path)
        for name, path in required.items()
        if path.is_file()
    }
    if set(individual) != set(required):
        raise RuntimeEnvelopeError(
            "source_binding_missing",
            "a required runtime source file is absent",
        )
    graph_paths = [
        repo / "solution.py",
        repo / "run.sh",
        repo / "requirements.lock",
        repo / "scripts" / "run_docker_submission.py",
        *(
            path
            for path in (repo / "mib_pipeline").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ),
    ]
    return {
        "git_revision": revision,
        **individual,
        "producer_graph_sha256": _canonical_tree_sha256(
            repo,
            graph_paths,
        ),
        "clean_worktree": True,
    }


def input_tree_sha256(input_dir: Path) -> str:
    pdfs = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not pdfs:
        raise RuntimeEnvelopeError(
            "expected_case_inventory_invalid",
            "input directory contains no PDF cases",
        )
    digest = hashlib.sha256()
    for path in pdfs:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def scan_repo_model_artifacts(
    repo: Path,
    max_model_bytes: int,
    max_total_bytes: int,
) -> None:
    """Cheap pre-build check; the authoritative check scans the final image."""

    oversized: list[tuple[Path, int]] = []
    total = 0
    for path in Path(repo).rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        is_runtime_json = (
            path.suffix.casefold() == ".json"
            and "mib_pipeline" in path.parts
            and "artifacts" in path.parts
        )
        if path.suffix.lower() in MODEL_EXTENSIONS or is_runtime_json:
            size = path.stat().st_size
            total += size
            if size > max_model_bytes:
                oversized.append((path, size))
    if oversized:
        raise RuntimeEnvelopeError(
            "repo_model_artifact_too_large",
            "a repository model artifact exceeds the per-file limit",
        )
    if total > max_total_bytes:
        raise RuntimeEnvelopeError(
            "repo_model_artifacts_too_large",
            "repository model artifacts exceed the aggregate limit",
        )


_IMAGE_MODEL_SCAN_SCRIPT = """
import json
import os

extensions = set(json.loads(os.environ["MIB_MODEL_EXTENSIONS_JSON"]))
roots = (
    "/app",
    "/opt",
    "/usr/local/lib/python3.12/site-packages",
    "/usr/share/tesseract-ocr",
)
artifacts = []
seen = set()
for root in roots:
    if not os.path.isdir(root):
        continue
    for directory, child_dirs, filenames in os.walk(root, topdown=True):
        child_dirs[:] = sorted(
            name for name in child_dirs
            if name not in {"__pycache__", ".cache"}
        )
        for filename in sorted(filenames):
            path = os.path.join(directory, filename)
            extension = os.path.splitext(filename)[1].lower()
            is_runtime_json = (
                extension == ".json"
                and path.startswith("/app/mib_pipeline/artifacts/")
            )
            if extension not in extensions and not is_runtime_json:
                continue
            try:
                stat = os.stat(path)
            except OSError:
                continue
            inode = (stat.st_dev, stat.st_ino)
            if inode in seen:
                continue
            seen.add(inode)
            artifacts.append({"path": path, "bytes": stat.st_size})
print(json.dumps(
    {
        "scan_roots": list(roots),
        "artifacts": sorted(artifacts, key=lambda item: item["path"]),
    },
    sort_keys=True,
    separators=(",", ":"),
))
""".strip()


def _resource_flags(*, cpus: str, memory: str) -> list[str]:
    return [
        "--network",
        "none",
        "--cpus",
        str(cpus),
        "--memory",
        str(memory),
        "--memory-swap",
        str(memory),
        "--pids-limit",
        "512",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=2g",
    ]


def scan_image_model_artifacts(
    image_tag: str,
    *,
    cpus: str,
    memory: str,
    max_model_bytes: int,
    max_total_bytes: int,
) -> dict[str, object]:
    """Scan installed model files inside the final image, not just the repo."""

    command = [
        "docker",
        "run",
        "--rm",
        *_resource_flags(cpus=cpus, memory=memory),
        "--env",
        "MIB_MODEL_EXTENSIONS_JSON="
        + json.dumps(sorted(MODEL_EXTENSIONS), separators=(",", ":")),
        "--entrypoint",
        "python3",
        image_tag,
        "-c",
        _IMAGE_MODEL_SCAN_SCRIPT,
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeEnvelopeError(
            "image_model_scan_failed",
            "installed model artifacts could not be scanned",
        ) from exc
    if completed.returncode != 0:
        raise RuntimeEnvelopeError(
            "image_model_scan_failed",
            "installed model artifact scan exited unsuccessfully",
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeEnvelopeError(
            "image_model_scan_invalid",
            "installed model artifact scan returned invalid JSON",
        ) from exc
    raw_artifacts = payload.get("artifacts")
    scan_roots = payload.get("scan_roots")
    if (
        not isinstance(raw_artifacts, list)
        or scan_roots
        != [
            "/app",
            "/opt",
            "/usr/local/lib/python3.12/site-packages",
            "/usr/share/tesseract-ocr",
        ]
    ):
        raise RuntimeEnvelopeError(
            "image_model_scan_invalid",
            "installed model artifact scan scope is absent or invalid",
        )

    artifacts: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise RuntimeEnvelopeError(
                "image_model_scan_invalid",
                "installed model artifact entry is invalid",
            )
        path = raw.get("path")
        size = raw.get("bytes")
        is_runtime_json = bool(
            isinstance(path, str)
            and path.startswith("/app/mib_pipeline/artifacts/")
            and Path(path).suffix.casefold() == ".json"
        )
        if (
            not isinstance(path, str)
            or not path.startswith("/")
            or path in seen_paths
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or (
                Path(path).suffix.lower() not in MODEL_EXTENSIONS
                and not is_runtime_json
            )
        ):
            raise RuntimeEnvelopeError(
                "image_model_scan_invalid",
                "installed model artifact entry violates the schema",
            )
        seen_paths.add(path)
        artifacts.append({"path": path, "bytes": size})

    artifacts.sort(key=lambda item: str(item["path"]))
    total = sum(int(item["bytes"]) for item in artifacts)
    maximum = max((int(item["bytes"]) for item in artifacts), default=0)
    if maximum > max_model_bytes:
        raise RuntimeEnvelopeError(
            "image_model_artifact_too_large",
            "an installed model artifact exceeds the per-file limit",
        )
    if total > max_total_bytes:
        raise RuntimeEnvelopeError(
            "image_model_artifacts_too_large",
            "installed model artifacts exceed the aggregate limit",
        )
    extension_counts: dict[str, int] = {}
    sanitized_artifacts: list[dict[str, object]] = []
    for item in artifacts:
        extension = Path(str(item["path"])).suffix.lower()
        extension_counts[extension] = extension_counts.get(extension, 0) + 1
        sanitized_artifacts.append(
            {
                "extension": extension,
                "bytes": int(item["bytes"]),
            }
        )
    return {
        "scan_scope": IMAGE_MODEL_SCAN_SCOPE,
        # Runtime calibration/policy JSON artifacts are deliberately counted
        # alongside binary OCR/model files, so the declared extension
        # inventory must include their sanitized `.json` entries too.
        "extensions": sorted(IMAGE_MODEL_EVIDENCE_EXTENSIONS),
        "artifact_count": len(artifacts),
        "extension_counts": dict(sorted(extension_counts.items())),
        "total_bytes": total,
        "maximum_artifact_bytes": maximum,
        # Paths and filenames are intentionally omitted from repository-safe
        # aggregate evidence.  Per-artifact sizes still prove both gates.
        "artifacts": sanitized_artifacts,
    }


def image_identity(
    image_tag: str,
    *,
    expected_source_revision: str,
    expected_producer_graph_sha256: str,
) -> dict[str, object]:
    """Bind a built or reused image to its exact source and platform."""

    try:
        image_id = docker_output(
            [
                "docker",
                "image",
                "inspect",
                image_tag,
                "--format",
                "{{.Id}}",
            ]
        )
        platform = docker_output(
            [
                "docker",
                "image",
                "inspect",
                image_tag,
                "--format",
                "{{.Os}}/{{.Architecture}}",
            ]
        )
        raw_labels = docker_output(
            [
                "docker",
                "image",
                "inspect",
                image_tag,
                "--format",
                "{{json .Config.Labels}}",
            ]
        )
        labels = json.loads(raw_labels)
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        raise RuntimeEnvelopeError(
            "image_identity_unavailable",
            "Docker image identity could not be inspected",
        ) from exc
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise RuntimeEnvelopeError(
            "image_identity_unavailable",
            "Docker image ID is not a full SHA-256 identifier",
        )
    if not re.fullmatch(r"[a-z0-9_+-]+/[a-z0-9_+-]+", platform):
        raise RuntimeEnvelopeError(
            "image_identity_unavailable",
            "Docker image platform is invalid",
        )
    if not isinstance(labels, Mapping):
        raise RuntimeEnvelopeError(
            "image_source_binding_mismatch",
            "Docker image has no source-binding labels",
        )
    expected_labels = {
        "mib.wo20.source_revision": expected_source_revision,
        "mib.wo20.producer_graph_sha256": expected_producer_graph_sha256,
    }
    if any(labels.get(name) != value for name, value in expected_labels.items()):
        raise RuntimeEnvelopeError(
            "image_source_binding_mismatch",
            "Docker image labels do not match the measured source",
        )
    operating_system, architecture = platform.split("/", 1)
    return {
        "image_id": image_id,
        "operating_system": operating_system,
        "architecture": architecture,
        "source_binding_labels_match": True,
    }


def parse_memory_bytes(value: str) -> int:
    """Parse the used-memory side of Docker's ``MemUsage`` display."""

    used = value.split("/", 1)[0].strip()
    match = _MEMORY_RE.match(used)
    if match is None:
        raise RuntimeEnvelopeError(
            "docker_stats_parse_failed",
            "Docker memory usage has an unsupported format",
        )
    magnitude = float(match.group(1))
    unit = match.group(2).casefold()
    multiplier = _MEMORY_MULTIPLIERS.get(unit)
    if (
        multiplier is None
        or not math.isfinite(magnitude)
        or magnitude < 0
    ):
        raise RuntimeEnvelopeError(
            "docker_stats_parse_failed",
            "Docker memory usage is invalid",
        )
    return int(magnitude * multiplier)


_RUNTIME_RSS_WRAPPER_SCRIPT = """
import json
import os
import subprocess
import sys
import time

launcher, input_dir, output_path, metrics_path = sys.argv[1:5]

def children(pid):
    try:
        raw = open(
            f"/proc/{pid}/task/{pid}/children",
            encoding="utf-8",
        ).read()
    except OSError:
        return ()
    return tuple(int(value) for value in raw.split() if value.isdigit())

def process_tree(root_pid):
    found = []
    pending = [root_pid]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        found.append(pid)
        pending.extend(children(pid))
    return found

def rss_bytes(pid):
    try:
        lines = open(
            f"/proc/{pid}/status",
            encoding="utf-8",
        ).read().splitlines()
    except OSError:
        return 0
    for line in lines:
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return 0

process = subprocess.Popen([launcher, input_dir, output_path])
peak = 0
samples = 0
while process.poll() is None:
    peak = max(peak, sum(rss_bytes(pid) for pid in process_tree(process.pid)))
    samples += 1
    time.sleep(0.1)
peak = max(peak, sum(rss_bytes(pid) for pid in process_tree(process.pid)))
payload = {
    "peak_process_tree_rss_bytes": peak,
    "rss_sample_count": samples,
    "source": "in_container_procfs_summed_process_tree_vmrss",
}
temporary = metrics_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
    handle.write("\\n")
os.replace(temporary, metrics_path)
os.chmod(metrics_path, 0o644)
if os.path.isfile(output_path):
    os.chmod(output_path, 0o644)
raise SystemExit(process.returncode)
""".strip()


def build_container_command(
    *,
    image_tag: str,
    container_name: str,
    input_dir: Path,
    output_dir: Path,
    output_name: str,
    metrics_name: str,
    cpus: str,
    memory: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--name",
        container_name,
        *_resource_flags(cpus=cpus, memory=memory),
        "--mount",
        f"type=bind,src={input_dir},dst=/input,readonly",
        "--mount",
        f"type=bind,src={output_dir},dst=/output",
        "--entrypoint",
        "python3",
        image_tag,
        "-c",
        _RUNTIME_RSS_WRAPPER_SCRIPT,
        "/app/run.sh",
        "/input",
        f"/output/{output_name}",
        f"/output/{metrics_name}",
    ]


def _stream_docker_memory(
    container_name: str,
    *,
    stop: threading.Event,
    samples: list[int],
    retry_seconds: float,
) -> None:
    while not stop.is_set():
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                [
                    "docker",
                    "stats",
                    "--format",
                    "{{.MemUsage}}",
                    container_name,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if process.stdout is None:
                raise OSError("docker stats stdout is unavailable")
            for raw in process.stdout:
                if stop.is_set():
                    break
                try:
                    samples.append(parse_memory_bytes(raw))
                except RuntimeEnvelopeError:
                    continue
        except OSError:
            pass
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        stop.wait(retry_seconds)


def _inspect_container_state(container_name: str) -> Mapping[str, object]:
    try:
        raw = docker_output(
            [
                "docker",
                "inspect",
                container_name,
                "--format",
                "{{json .State}}",
            ]
        )
        state = json.loads(raw)
    except (OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
        raise RuntimeEnvelopeError(
            "container_state_unavailable",
            "container state could not be inspected",
        ) from exc
    if not isinstance(state, Mapping):
        raise RuntimeEnvelopeError(
            "container_state_unavailable",
            "container state is not an object",
        )
    return state


def _best_effort_remove_container(
    container_name: str,
    *,
    timeout_seconds: float = 15.0,
) -> None:
    """Bound cleanup even when the Docker daemon is unhealthy."""

    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError):
        return


def execute_container(
    command: Sequence[str],
    *,
    container_name: str,
    metrics_path: Path,
    timeout_seconds: int,
    stats_interval_seconds: float,
) -> tuple[float, int, int]:
    """Run one retained container and capture elapsed, RSS, and cgroup memory."""

    print("+ " + " ".join(str(part) for part in command), flush=True)
    samples: list[int] = []
    stop = threading.Event()
    monitor = threading.Thread(
        target=_stream_docker_memory,
        kwargs={
            "container_name": container_name,
            "stop": stop,
            "samples": samples,
            "retry_seconds": stats_interval_seconds,
        },
        daemon=True,
    )
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        monitor.start()
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _best_effort_remove_container(container_name)
            try:
                process.communicate(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            raise RuntimeEnvelopeError(
                "container_timeout",
                f"container exceeded {timeout_seconds} seconds",
            ) from exc
        elapsed = time.monotonic() - started
        stop.set()
        monitor.join(timeout=6.0)
        state = _inspect_container_state(container_name)
        if state.get("OOMKilled") is True:
            raise RuntimeEnvelopeError(
                "container_oom_killed",
                "container exceeded its memory envelope",
            )
        state_exit_code = state.get("ExitCode")
        if (
            process.returncode != 0
            or isinstance(state_exit_code, bool)
            or not isinstance(state_exit_code, int)
            or state_exit_code != 0
        ):
            detail = (stderr or stdout).strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise RuntimeEnvelopeError(
                "container_exit_nonzero",
                "container exited unsuccessfully" + suffix,
            )
        if not samples or max(samples) <= 0:
            raise RuntimeEnvelopeError(
                "peak_container_memory_unavailable",
                "Docker stats did not yield a positive memory sample",
            )
        try:
            rss_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            peak_rss = rss_payload["peak_process_tree_rss_bytes"]
            rss_samples = rss_payload["rss_sample_count"]
            rss_source = rss_payload["source"]
        except (KeyError, OSError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeEnvelopeError(
                "peak_rss_unavailable",
                "process-tree RSS metrics were not emitted",
            ) from exc
        if (
            isinstance(peak_rss, bool)
            or not isinstance(peak_rss, int)
            or peak_rss <= 0
            or isinstance(rss_samples, bool)
            or not isinstance(rss_samples, int)
            or rss_samples <= 0
            or rss_source != PEAK_RSS_SOURCE
        ):
            raise RuntimeEnvelopeError(
                "peak_rss_unavailable",
                "process-tree RSS metrics are invalid",
            )
        return elapsed, peak_rss, max(samples)
    except OSError as exc:
        raise RuntimeEnvelopeError(
            "container_start_failed",
            "Docker container could not be started",
        ) from exc
    finally:
        stop.set()
        if monitor.is_alive():
            monitor.join(timeout=6.0)
        _best_effort_remove_container(container_name)


def _expected_ids(input_dir: Path, manifest: Path | None) -> tuple[str, ...]:
    try:
        pdf_values = sorted(
            path.stem
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".pdf"
        )
        manifest_values = (
            validate_submission.expected_ids_from_manifest(manifest)
            if manifest is not None
            else pdf_values
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise RuntimeEnvelopeError(
            "expected_case_inventory_invalid",
            "expected case inventory could not be read",
        ) from exc
    normalized = tuple(str(value).strip() for value in pdf_values)
    normalized_manifest = tuple(
        str(value).strip() for value in manifest_values
    )
    if (
        not normalized
        or any(not value for value in normalized)
        or len(set(normalized)) != len(normalized)
        or any(not value for value in normalized_manifest)
        or len(set(normalized_manifest)) != len(normalized_manifest)
    ):
        raise RuntimeEnvelopeError(
            "expected_case_inventory_invalid",
            "expected case inventory must be non-empty and unique",
        )
    if set(normalized_manifest) != set(normalized):
        raise RuntimeEnvelopeError(
            "manifest_pdf_inventory_mismatch",
            "manifest IDs must exactly match actual PDF stems",
        )
    return normalized


def summarize_output(
    output_path: Path,
    *,
    expected_ids: Sequence[str],
) -> OutputSummary:
    expected = set(expected_ids)
    try:
        rows, submission_format = validate_submission.read_submission(
            output_path
        )
    except (OSError, SystemExit):
        return OutputSummary(
            attempted=len(expected),
            answered=0,
            omitted=len(expected),
            invalid=1,
            rows_emitted=0,
        )

    invalid = 0
    seen: set[str] = set()
    answered: set[str] = set()
    for index, row in enumerate(rows, start=1):
        row_errors = validate_submission.validate_row(
            row,
            f"record {index}",
            submission_format,
        )
        case_id = str(row.get("case_id", "")).strip()
        duplicate = bool(case_id and case_id in seen)
        if case_id:
            seen.add(case_id)
        unexpected = case_id not in expected
        if row_errors or duplicate or unexpected:
            invalid += 1
            continue
        answered.add(case_id)
    return OutputSummary(
        attempted=len(expected),
        answered=len(answered),
        omitted=len(expected - answered),
        invalid=invalid,
        rows_emitted=len(rows),
    )


def require_canonical_jsonl(output_path: Path) -> None:
    """Require the exact compact, ordered JSONL form emitted by production."""

    try:
        raw = output_path.read_bytes()
    except OSError as exc:
        raise RuntimeEnvelopeError(
            "output_unreadable",
            "prediction output could not be read",
        ) from exc
    if raw and not raw.endswith(b"\n"):
        raise RuntimeEnvelopeError(
            "noncanonical_prediction_output",
            "canonical JSONL must end every record with a newline",
        )
    previous_case_id: str | None = None
    for raw_line in raw.splitlines(keepends=True):
        if raw_line in {b"", b"\n", b"\r\n"}:
            raise RuntimeEnvelopeError(
                "noncanonical_prediction_output",
                "canonical JSONL cannot contain blank records",
            )
        try:
            line = raw_line.decode("utf-8")
            payload = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeEnvelopeError(
                "noncanonical_prediction_output",
                "canonical JSONL contains an invalid record",
            ) from exc
        if not isinstance(payload, Mapping):
            raise RuntimeEnvelopeError(
                "noncanonical_prediction_output",
                "each canonical JSONL record must be an object",
            )
        if tuple(payload) != tuple(validate_submission.FIELDNAMES):
            raise RuntimeEnvelopeError(
                "noncanonical_prediction_output",
                "canonical JSONL fields are missing, extra, or out of order",
            )
        canonical_line = (
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        if line != canonical_line:
            raise RuntimeEnvelopeError(
                "noncanonical_prediction_output",
                "prediction record is not compact canonical JSON",
            )
        case_id = payload.get("case_id")
        if not isinstance(case_id, str) or (
            previous_case_id is not None and case_id <= previous_case_id
        ):
            raise RuntimeEnvelopeError(
                "noncanonical_prediction_output",
                "canonical JSONL records must be strictly case-ID ordered",
            )
        previous_case_id = case_id


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise RuntimeEnvelopeError(
            "latency_samples_unavailable",
            "latency percentile requires samples",
        )
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(runs: Sequence[ContainerRun]) -> dict[str, object]:
    if not runs:
        raise RuntimeEnvelopeError(
            "runtime_samples_unavailable",
            "at least one runtime sample is required",
        )
    attempted = runs[0].output.attempted
    if attempted <= 0 or any(run.output.attempted != attempted for run in runs):
        raise RuntimeEnvelopeError(
            "runtime_denominator_invalid",
            "runtime samples need one positive, stable attempted count",
        )
    per_pdf = [run.elapsed_seconds / attempted for run in runs]
    return {
        "measurement_basis": (
            "full_container_elapsed_divided_by_attempted_pdf_count; "
            "percentiles_across_repeat_normalized_runs_not_individual_pdf_latency"
        ),
        "repeat_sample_count": len(per_pdf),
        "total_elapsed_seconds": sum(run.elapsed_seconds for run in runs),
        "mean_run_elapsed_seconds": (
            sum(run.elapsed_seconds for run in runs) / len(runs)
        ),
        "per_pdf_seconds": {
            "average": sum(per_pdf) / len(per_pdf),
            "p50": _percentile(per_pdf, 0.50),
            "p90": _percentile(per_pdf, 0.90),
            "p95": _percentile(per_pdf, 0.95),
            "max": max(per_pdf),
        },
    }


def require_each_repeat_within_average_limit(
    runs: Sequence[ContainerRun],
    *,
    max_seconds_per_pdf: float,
) -> dict[str, object]:
    measured = latency_summary(runs)
    if any(
        run.elapsed_seconds / run.output.attempted > max_seconds_per_pdf
        for run in runs
    ):
        raise RuntimeEnvelopeError(
            "average_runtime_limit_exceeded",
            "at least one repeat exceeds the per-PDF runtime limit",
        )
    return measured


def determinism_summary(runs: Sequence[ContainerRun]) -> dict[str, object]:
    if len(runs) < 2:
        raise RuntimeEnvelopeError(
            "repeat_count_too_small",
            "determinism requires at least two complete runs",
        )
    hashes = {run.output_sha256 for run in runs}
    sizes = {run.output_bytes for run in runs}
    coverages = {json.dumps(run.output.to_dict(), sort_keys=True) for run in runs}
    byte_identical = len(hashes) == 1 and len(sizes) == 1
    if not byte_identical or len(coverages) != 1:
        raise RuntimeEnvelopeError(
            "nondeterministic_output",
            "repeated container outputs are not exactly identical",
        )
    return {
        "repeat_count": len(runs),
        "byte_identical": True,
        "unique_output_hash_count": 1,
        "coverage_identical": True,
    }


def determinism_evidence(
    runs: Sequence[ContainerRun],
    *,
    single_run_capture: bool,
) -> dict[str, object]:
    if single_run_capture:
        if len(runs) != 1:
            raise RuntimeEnvelopeError(
                "single_run_capture_count_invalid",
                "single-run capture must contain exactly one run",
            )
        return {
            "evaluated": False,
            "reason": (
                "single capture requires an external comparison against an "
                "independently executed capture"
            ),
        }
    return {
        "evaluated": True,
        **determinism_summary(runs),
    }


def completed_status(
    *,
    single_run_capture: bool,
    warnings: Sequence[str],
) -> str:
    if single_run_capture:
        return (
            "CAPTURE_COMPLETE_WITH_WARNINGS"
            if warnings
            else "CAPTURE_COMPLETE"
        )
    return "PASS_WITH_WARNINGS" if warnings else "PASS"


def _write_evidence(path: Path | None, payload: Mapping[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(rendered)
    os.replace(temporary, path)


def _base_evidence(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": EVIDENCE_SCHEMA,
        "mode": (
            "single_run_capture"
            if args.single_run_capture
            else "repeat_certification"
        ),
        "status": "BLOCKED",
        "blocking_reasons": [],
        "warnings": [],
        "environment": {
            "docker_available": False,
            "docker_status": "not_checked",
        },
        "limits": {
            "cpus": str(args.cpus),
            "memory": str(args.memory),
            "network": "none",
            "read_only_root": True,
            "read_only_input": True,
            "tmpfs": "/tmp:rw,nosuid,nodev,size=2g",
            "pids_limit": 512,
            "max_image_bytes": int(
                args.max_image_gib * 1024 * 1024 * 1024
            ),
            "max_model_artifact_bytes": int(
                args.max_model_mib * 1024 * 1024
            ),
            "max_total_model_bytes": int(
                args.max_total_model_mib * 1024 * 1024
            ),
            "max_output_bytes": int(
                args.max_output_mib * 1024 * 1024
            ),
            "timeout_seconds": args.timeout_seconds,
            "max_average_seconds_per_pdf": (
                args.max_average_seconds_per_pdf
            ),
        },
        "requested_repeat_count": args.repeat_count,
        "deadline_controls": {
            "whole_run_timeout_seconds": args.timeout_seconds,
            "per_case_deadline_enforced": False,
        },
        "resilience": {
            "output_commit_strategy": "batch_end_atomic",
            "partial_progress_recovery": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and measure an offline Docker submission."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Candidate repository containing Dockerfile.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="PDF input directory.",
    )
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument(
        "--output",
        dest="output_path",
        help="Host path where predictions should be written.",
    )
    output_group.add_argument(
        "--output-csv",
        dest="output_path",
        help="Compatibility alias for --output.",
    )
    parser.add_argument(
        "--manifest",
        help="Optional manifest used to validate output case ids.",
    )
    parser.add_argument(
        "--evidence-json",
        help=(
            "Optional aggregate WO20 evidence JSON path. Evidence mode "
            "requires complete output even without --require-complete."
        ),
    )
    parser.add_argument("--image-tag", default=None)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=30000)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument(
        "--single-run-capture",
        action="store_true",
        help=(
            "Allow exactly one fully measured run for an externally compared "
            "parallel capture. This mode never claims determinism by itself."
        ),
    )
    parser.add_argument("--stats-interval-seconds", type=float, default=1.0)
    parser.add_argument("--cpus", default="4")
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--max-image-gib", type=float, default=4.0)
    parser.add_argument("--max-model-mib", type=float, default=250.0)
    parser.add_argument("--max-total-model-mib", type=float, default=1024.0)
    parser.add_argument("--max-output-mib", type=float, default=25.0)
    parser.add_argument(
        "--max-average-seconds-per-pdf",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail when the output omits an expected case.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_path = Path(args.output_path).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else None
    evidence_path = (
        Path(args.evidence_json).resolve() if args.evidence_json else None
    )
    numeric_limits = (
        args.stats_interval_seconds,
        args.max_image_gib,
        args.max_model_mib,
        args.max_total_model_mib,
        args.max_output_mib,
        args.max_average_seconds_per_pdf,
    )
    if (
        args.timeout_seconds <= 0
        or args.stats_interval_seconds < 1.0
        or any(
            not math.isfinite(value) or value <= 0
            for value in numeric_limits
        )
    ):
        evidence = {
            "schema_version": EVIDENCE_SCHEMA,
            "status": "BLOCKED",
            "blocking_reasons": ["invalid_runtime_limits"],
            "warnings": [],
            "environment": {
                "docker_available": False,
                "docker_status": "not_checked",
            },
            "requested_repeat_count": args.repeat_count,
        }
        _write_evidence(evidence_path, evidence)
        print(
            "error: runtime limits must be positive and finite; "
            "Docker stats retry interval must be at least 1 second",
            file=sys.stderr,
        )
        return 2
    evidence = _base_evidence(args)

    invalid_repeat_configuration = (
        args.repeat_count != 1
        if args.single_run_capture
        else args.repeat_count < 2
    )
    if invalid_repeat_configuration:
        evidence["blocking_reasons"] = [
            (
                "single_run_capture_repeat_count_invalid"
                if args.single_run_capture
                else "repeat_count_too_small"
            )
        ]
        _write_evidence(evidence_path, evidence)
        print(
            (
                "error: --single-run-capture requires repeat-count 1"
                if args.single_run_capture
                else "error: repeat-count must be at least 2"
            ),
            file=sys.stderr,
        )
        return 2
    if not (repo / "Dockerfile").is_file():
        evidence["blocking_reasons"] = ["dockerfile_missing"]
        _write_evidence(evidence_path, evidence)
        print(f"error: no Dockerfile found in {repo}", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        evidence["blocking_reasons"] = ["input_directory_missing"]
        _write_evidence(evidence_path, evidence)
        print(f"error: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2
    if str(args.cpus) not in {"4", "4.0"} or str(args.memory).casefold() != "8g":
        evidence["blocking_reasons"] = ["noncanonical_resource_envelope"]
        _write_evidence(evidence_path, evidence)
        print(
            "error: WO20 evidence requires exactly 4 CPUs and 8g memory",
            file=sys.stderr,
        )
        return 2

    available, docker_detail = docker_status()
    evidence["environment"] = {
        "docker_available": available,
        "docker_status": (
            "available" if available else docker_detail
        ),
        **(
            {"docker_server_version": docker_detail}
            if available
            else {}
        ),
    }
    if not available:
        evidence["status"] = "DOCKER_UNAVAILABLE"
        evidence["blocking_reasons"] = [docker_detail]
        _write_evidence(evidence_path, evidence)
        print(
            "error: Docker is unavailable; no runtime-envelope claim was made",
            file=sys.stderr,
        )
        return 3

    max_image_bytes = int(args.max_image_gib * 1024 * 1024 * 1024)
    max_model_bytes = int(args.max_model_mib * 1024 * 1024)
    max_total_model_bytes = int(
        args.max_total_model_mib * 1024 * 1024
    )
    max_output_bytes = int(args.max_output_mib * 1024 * 1024)
    image_tag = args.image_tag or f"mib-submission-{int(time.time())}"

    try:
        expected_ids = _expected_ids(input_dir, manifest)
        measured_source = source_bindings(repo)
        evidence["source_binding"] = measured_source
        measured_input_binding = {
            "pdf_count": len(expected_ids),
            "input_tree_sha256": input_tree_sha256(input_dir),
            "manifest_sha256": (
                _sha256_file(manifest) if manifest is not None else None
            ),
            "manifest_matches_pdf_inventory": True,
        }
        evidence["input_binding"] = measured_input_binding
        scan_repo_model_artifacts(
            repo,
            max_model_bytes,
            max_total_model_bytes,
        )
        if args.skip_build:
            if args.image_tag is None:
                raise RuntimeEnvelopeError(
                    "image_tag_required",
                    "--skip-build requires --image-tag",
                )
        else:
            try:
                run(
                    [
                        "docker",
                        "build",
                        "--pull=false",
                        "--label",
                        "mib.wo20.source_revision="
                        + str(measured_source["git_revision"]),
                        "--label",
                        "mib.wo20.producer_graph_sha256="
                        + str(measured_source["producer_graph_sha256"]),
                        "-t",
                        image_tag,
                        str(repo),
                    ],
                    timeout=args.timeout_seconds,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeEnvelopeError(
                    "docker_build_failed",
                    "Docker image build failed",
                ) from exc

        actual_image_size = image_size_bytes(image_tag)
        if actual_image_size > max_image_bytes:
            raise RuntimeEnvelopeError(
                "image_size_limit_exceeded",
                "Docker image exceeds the configured size limit",
            )
        measured_image_identity = image_identity(
            image_tag,
            expected_source_revision=str(
                measured_source["git_revision"]
            ),
            expected_producer_graph_sha256=str(
                measured_source["producer_graph_sha256"]
            ),
        )
        runtime_image_ref = str(measured_image_identity["image_id"])
        evidence["image"] = {
            "size_bytes": actual_image_size,
            "within_limit": True,
            "build_mode": "reused" if args.skip_build else "fresh",
            **measured_image_identity,
        }
        evidence["installed_model_artifacts"] = scan_image_model_artifacts(
            runtime_image_ref,
            cpus=str(args.cpus),
            memory=str(args.memory),
            max_model_bytes=max_model_bytes,
            max_total_bytes=max_total_model_bytes,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()
        runs: list[ContainerRun] = []
        with tempfile.TemporaryDirectory(
            prefix=".mib-wo20-runs-",
            dir=output_path.parent,
        ) as temporary_directory:
            run_output_dir = Path(temporary_directory)
            # A host-created TemporaryDirectory is mode 0700. The sticky
            # world-writable mode keeps the output bind compatible with both
            # the image's portable root-user contract and future explicit
            # runtime UID overrides, while preventing unrelated host users
            # from deleting another user's files. The in-container wrapper
            # makes its own outputs host-readable before it exits.
            run_output_dir.chmod(0o1777)
            first_output: Path | None = None
            for repeat_index in range(1, args.repeat_count + 1):
                suffix = output_path.suffix or ".jsonl"
                repeated_output = (
                    run_output_dir / f"predictions-run-{repeat_index:02d}{suffix}"
                )
                metrics_path = (
                    run_output_dir / f"runtime-metrics-{repeat_index:02d}.json"
                )
                container_name = (
                    f"mib-score-{os.getpid()}-{time.time_ns()}-{repeat_index}"
                )
                command = build_container_command(
                    image_tag=runtime_image_ref,
                    container_name=container_name,
                    input_dir=input_dir,
                    output_dir=run_output_dir,
                    output_name=repeated_output.name,
                    metrics_name=metrics_path.name,
                    cpus=str(args.cpus),
                    memory=str(args.memory),
                )
                elapsed, peak_rss, peak_container_memory = execute_container(
                    command,
                    container_name=container_name,
                    metrics_path=metrics_path,
                    timeout_seconds=args.timeout_seconds,
                    stats_interval_seconds=args.stats_interval_seconds,
                )
                if not repeated_output.is_file():
                    raise RuntimeEnvelopeError(
                        "output_missing",
                        "container did not write the expected output",
                    )
                output_size = repeated_output.stat().st_size
                if output_size > max_output_bytes:
                    raise RuntimeEnvelopeError(
                        "output_size_limit_exceeded",
                        "prediction output exceeds the configured size limit",
                    )
                require_canonical_jsonl(repeated_output)
                summary = summarize_output(
                    repeated_output,
                    expected_ids=expected_ids,
                )
                if summary.invalid:
                    raise RuntimeEnvelopeError(
                        "invalid_prediction_output",
                        "prediction output contains invalid records",
                    )
                if summary.omitted and (
                    args.require_complete or evidence_path is not None
                ):
                    raise RuntimeEnvelopeError(
                        "incomplete_prediction_output",
                        "certification evidence cannot omit expected cases",
                    )
                runs.append(
                    ContainerRun(
                        repeat_index=repeat_index,
                        elapsed_seconds=elapsed,
                        peak_process_tree_rss_bytes=peak_rss,
                        peak_container_memory_bytes=peak_container_memory,
                        output_sha256=_sha256_file(repeated_output),
                        output_bytes=output_size,
                        output=summary,
                    )
                )
                if first_output is None:
                    first_output = repeated_output

            deterministic = determinism_evidence(
                runs,
                single_run_capture=args.single_run_capture,
            )
            try:
                measured_latency = require_each_repeat_within_average_limit(
                    runs,
                    max_seconds_per_pdf=args.max_average_seconds_per_pdf,
                )
            except RuntimeEnvelopeError:
                evidence["runs"] = [item.to_dict() for item in runs]
                evidence["runtime"] = latency_summary(runs)
                raise
            if first_output is None:
                raise RuntimeEnvelopeError(
                    "output_missing",
                    "no repeated output was produced",
                )
            if source_bindings(repo) != measured_source:
                raise RuntimeEnvelopeError(
                    "source_binding_changed_during_run",
                    "source files changed during the runtime measurement",
                )
            _expected_ids(input_dir, manifest)
            final_input_binding = {
                "pdf_count": len(expected_ids),
                "input_tree_sha256": input_tree_sha256(input_dir),
                "manifest_sha256": (
                    _sha256_file(manifest) if manifest is not None else None
                ),
                "manifest_matches_pdf_inventory": True,
            }
            if final_input_binding != measured_input_binding:
                raise RuntimeEnvelopeError(
                    "input_binding_changed_during_run",
                    "input files or manifest changed during the runtime measurement",
                )
            shutil.copyfile(first_output, output_path)

        warnings: list[str] = ["per_case_deadline_not_enforced"]
        if runs[0].output.omitted:
            warnings.append("prediction_output_contains_omissions")
        status = completed_status(
            single_run_capture=args.single_run_capture,
            warnings=warnings,
        )
        evidence.update(
            {
                "status": status,
                "blocking_reasons": [],
                "warnings": warnings,
                "runs": [item.to_dict() for item in runs],
                "coverage": runs[0].output.to_dict(),
                "determinism": deterministic,
                "bindings_reverified_after_runs": True,
                "runtime": {
                    **measured_latency,
                    "all_repeats_within_average_limit": True,
                    "peak_process_tree_rss_bytes": max(
                        item.peak_process_tree_rss_bytes for item in runs
                    ),
                    "peak_process_tree_rss_mib": max(
                        item.peak_process_tree_rss_bytes for item in runs
                    )
                    / (1024 * 1024),
                    "peak_process_tree_rss_source": PEAK_RSS_SOURCE,
                    "peak_container_memory_bytes": max(
                        item.peak_container_memory_bytes for item in runs
                    ),
                    "peak_container_memory_mib": max(
                        item.peak_container_memory_bytes for item in runs
                    )
                    / (1024 * 1024),
                    "peak_container_memory_source": (
                        PEAK_CONTAINER_MEMORY_SOURCE
                    ),
                },
            }
        )
        _write_evidence(evidence_path, evidence)
        print(
            (
                "Offline Docker single-run capture completed; external "
                f"determinism comparison required: {output_path}"
                if args.single_run_capture
                else (
                    "Offline Docker submission completed with "
                    f"{len(runs)} byte-identical runs: {output_path}"
                )
            )
        )
        return 0
    except RuntimeEnvelopeError as exc:
        evidence["status"] = "BLOCKED"
        evidence["blocking_reasons"] = [exc.code]
        _write_evidence(evidence_path, evidence)
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
