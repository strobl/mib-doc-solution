#!/usr/bin/env python3
"""Compare two independent WO20 single-run Docker evidence captures.

The runtime harness deliberately refuses to certify determinism from one
capture.  This tool closes that gap for environments, such as GitHub-hosted
runners, where two full 5,000-case runs must execute in parallel to fit the
per-job time limit.

Only aggregate evidence is accepted or emitted.  Prediction rows, case
identifiers, filenames, labels, truth, and local paths never enter the output.
Every binding and envelope check is fail-closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


CAPTURE_SCHEMA = "mib-wo20-docker-runtime-envelope/v1"
OUTPUT_SCHEMA = "mib-wo20-parallel-determinism/v1"
ALLOWED_WARNINGS = frozenset({"per_case_deadline_not_enforced"})
COMPLETE_STATUSES = frozenset(
    {"CAPTURE_COMPLETE", "CAPTURE_COMPLETE_WITH_WARNINGS"}
)

GIB = 1024**3
MIB = 1024**2
OFFICIAL_CPU_COUNT = "4"
OFFICIAL_MEMORY = "8g"
OFFICIAL_IMAGE_LIMIT = 4 * GIB
OFFICIAL_MODEL_LIMIT = 250 * MIB
OFFICIAL_TOTAL_MODEL_LIMIT = 1024 * MIB
OFFICIAL_OUTPUT_LIMIT = 25 * MIB
OFFICIAL_RUNTIME_LIMIT_SECONDS = 30_000
OFFICIAL_SECONDS_PER_PDF = 6.0
IMAGE_MODEL_SCAN_SCOPE = "fixed_runtime_roots"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PLATFORM_PART_RE = re.compile(r"^[a-z0-9_+-]+$")
_EXTENSION_RE = re.compile(r"^\.[a-z0-9_+-]+$")
_CASE_ID_RE = re.compile(r"\bMIB-[0-9]{6}\b", re.IGNORECASE)
_ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|[\s\"'])(?:/(?:Users|home|private|tmp|var)/|"
    r"[A-Za-z]:\\|\\\\)"
)

_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "mode",
        "status",
        "blocking_reasons",
        "warnings",
        "environment",
        "limits",
        "requested_repeat_count",
        "deadline_controls",
        "resilience",
        "source_binding",
        "input_binding",
        "image",
        "installed_model_artifacts",
        "runs",
        "coverage",
        "determinism",
        "bindings_reverified_after_runs",
        "runtime",
    }
)
_SOURCE_KEYS = frozenset(
    {
        "git_revision",
        "dockerfile_sha256",
        "requirements_lock_sha256",
        "run_sh_sha256",
        "solution_sha256",
        "harness_sha256",
        "producer_graph_sha256",
        "clean_worktree",
    }
)
_INPUT_KEYS = frozenset(
    {
        "pdf_count",
        "input_tree_sha256",
        "manifest_sha256",
        "manifest_matches_pdf_inventory",
    }
)
_LIMIT_KEYS = frozenset(
    {
        "cpus",
        "memory",
        "network",
        "read_only_root",
        "read_only_input",
        "tmpfs",
        "pids_limit",
        "max_image_bytes",
        "max_model_artifact_bytes",
        "max_total_model_bytes",
        "max_output_bytes",
        "timeout_seconds",
        "max_average_seconds_per_pdf",
    }
)
_COVERAGE_KEYS = frozenset(
    {"attempted", "answered", "omitted", "invalid", "rows_emitted"}
)
_RUN_KEYS = frozenset(
    {
        "repeat_index",
        "elapsed_seconds",
        "peak_process_tree_rss_bytes",
        "peak_process_tree_rss_mib",
        "peak_process_tree_rss_source",
        "peak_container_memory_bytes",
        "peak_container_memory_mib",
        "peak_container_memory_source",
        "output_sha256",
        "output_bytes",
        "coverage",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "measurement_basis",
        "repeat_sample_count",
        "total_elapsed_seconds",
        "mean_run_elapsed_seconds",
        "per_pdf_seconds",
        "all_repeats_within_average_limit",
        "peak_process_tree_rss_bytes",
        "peak_process_tree_rss_mib",
        "peak_process_tree_rss_source",
        "peak_container_memory_bytes",
        "peak_container_memory_mib",
        "peak_container_memory_source",
    }
)
_PER_PDF_KEYS = frozenset({"average", "p50", "p90", "p95", "max"})
_IMAGE_KEYS = frozenset(
    {
        "size_bytes",
        "within_limit",
        "build_mode",
        "image_id",
        "operating_system",
        "architecture",
        "source_binding_labels_match",
    }
)
_MODEL_KEYS = frozenset(
    {
        "scan_scope",
        "extensions",
        "artifact_count",
        "extension_counts",
        "total_bytes",
        "maximum_artifact_bytes",
        "artifacts",
    }
)
_MODEL_ITEM_KEYS = frozenset({"extension", "bytes"})


class WO20ParallelCompareError(RuntimeError):
    """A capture is incomplete, unsafe, malformed, or inconsistently bound."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ValidatedCapture:
    """The repository-safe subset needed for cross-capture comparison."""

    evidence_sha256: str
    source_binding: Mapping[str, Any]
    input_binding: Mapping[str, Any]
    limits: Mapping[str, Any]
    coverage: Mapping[str, Any]
    installed_model_artifacts: Mapping[str, Any]
    warnings: tuple[str, ...]
    output_sha256: str
    output_bytes: int
    elapsed_seconds: float
    peak_process_tree_rss_bytes: int
    peak_container_memory_bytes: int
    image_id: str
    image_size_bytes: int
    image_build_mode: str
    operating_system: str
    architecture: str


def _fail(code: str, message: str) -> None:
    raise WO20ParallelCompareError(code, message)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        _fail(code, message)


def _pairs_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(
                "duplicate_json_key",
                "capture JSON contains a duplicate object key",
            )
        result[key] = value
    return result


def _mapping(value: Any, *, code: str, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code, f"{label} must be a JSON object")
    return value


def _sequence(value: Any, *, code: str, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(code, f"{label} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    code: str,
    label: str,
) -> None:
    if set(value) != expected:
        _fail(code, f"{label} does not match the aggregate evidence schema")


def _integer(
    value: Any,
    *,
    code: str,
    label: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(code, f"{label} must be an integer of at least {minimum}")
    return value


def _number(
    value: Any,
    *,
    code: str,
    label: str,
    minimum_exclusive: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(code, f"{label} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered):
        _fail(code, f"{label} must be finite")
    if minimum_exclusive is not None and rendered <= minimum_exclusive:
        _fail(code, f"{label} must be greater than {minimum_exclusive}")
    return rendered


def _digest(value: Any, *, code: str, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(code, f"{label} must be a lowercase SHA-256 digest")
    return value


def _revision(value: Any, *, code: str, label: str) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        _fail(code, f"{label} must be a full lowercase Git revision")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_capture(path: Path, *, capture_index: int) -> tuple[Mapping[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_object,
        )
    except WO20ParallelCompareError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WO20ParallelCompareError(
            "capture_json_unreadable",
            f"capture {capture_index} must be readable UTF-8 JSON",
        ) from exc
    return (
        _mapping(
            payload,
            code="capture_schema_invalid",
            label=f"capture {capture_index}",
        ),
        _sha256_bytes(raw),
    )


def _validate_source_binding(
    payload: Any,
    *,
    expected_revision: str,
) -> Mapping[str, Any]:
    source = _mapping(
        payload,
        code="source_binding_invalid",
        label="source binding",
    )
    _exact_keys(
        source,
        _SOURCE_KEYS,
        code="source_binding_invalid",
        label="source binding",
    )
    _require(
        _revision(
            source["git_revision"],
            code="source_binding_invalid",
            label="source revision",
        )
        == expected_revision,
        "source_revision_mismatch",
        "capture source revision does not match the requested revision",
    )
    for key in _SOURCE_KEYS - {"git_revision", "clean_worktree"}:
        _digest(
            source[key],
            code="source_binding_invalid",
            label=key,
        )
    _require(
        source["clean_worktree"] is True,
        "source_worktree_not_clean",
        "capture was not bound to a clean worktree",
    )
    return source


def _validate_input_binding(
    payload: Any,
    *,
    expected_pdf_count: int,
) -> Mapping[str, Any]:
    binding = _mapping(
        payload,
        code="input_binding_invalid",
        label="input binding",
    )
    _exact_keys(
        binding,
        _INPUT_KEYS,
        code="input_binding_invalid",
        label="input binding",
    )
    _require(
        _integer(
            binding["pdf_count"],
            code="input_binding_invalid",
            label="PDF count",
            minimum=1,
        )
        == expected_pdf_count,
        "input_pdf_count_mismatch",
        "capture does not cover the required PDF count",
    )
    _digest(
        binding["input_tree_sha256"],
        code="input_binding_invalid",
        label="input tree digest",
    )
    _digest(
        binding["manifest_sha256"],
        code="input_binding_invalid",
        label="manifest digest",
    )
    _require(
        binding["manifest_matches_pdf_inventory"] is True,
        "input_manifest_mismatch",
        "manifest does not match the measured PDF inventory",
    )
    return binding


def _validate_limits(payload: Any) -> Mapping[str, Any]:
    limits = _mapping(
        payload,
        code="runtime_limits_invalid",
        label="runtime limits",
    )
    _exact_keys(
        limits,
        _LIMIT_KEYS,
        code="runtime_limits_invalid",
        label="runtime limits",
    )
    _require(
        str(limits["cpus"]) in {"4", "4.0"},
        "noncanonical_resource_envelope",
        "capture did not use exactly four CPUs",
    )
    _require(
        str(limits["memory"]).casefold() == OFFICIAL_MEMORY,
        "noncanonical_resource_envelope",
        "capture did not use exactly 8g memory",
    )
    _require(
        limits["network"] == "none"
        and limits["read_only_root"] is True
        and limits["read_only_input"] is True
        and limits["tmpfs"] == "/tmp:rw,nosuid,nodev,size=2g"
        and limits["pids_limit"] == 512,
        "noncanonical_resource_envelope",
        "capture Docker isolation flags are incomplete",
    )
    expected_sizes = {
        "max_image_bytes": OFFICIAL_IMAGE_LIMIT,
        "max_model_artifact_bytes": OFFICIAL_MODEL_LIMIT,
        "max_total_model_bytes": OFFICIAL_TOTAL_MODEL_LIMIT,
        "max_output_bytes": OFFICIAL_OUTPUT_LIMIT,
    }
    for key, expected in expected_sizes.items():
        _require(
            _integer(
                limits[key],
                code="runtime_limits_invalid",
                label=key,
                minimum=1,
            )
            == expected,
            "noncanonical_resource_envelope",
            f"{key} is not the official limit",
        )
    timeout_seconds = _integer(
        limits["timeout_seconds"],
        code="runtime_limits_invalid",
        label="whole-run timeout",
        minimum=1,
    )
    _require(
        timeout_seconds <= OFFICIAL_RUNTIME_LIMIT_SECONDS,
        "runtime_timeout_too_large",
        "whole-run timeout exceeds the official runtime limit",
    )
    _require(
        math.isclose(
            _number(
                limits["max_average_seconds_per_pdf"],
                code="runtime_limits_invalid",
                label="average runtime limit",
                minimum_exclusive=0,
            ),
            OFFICIAL_SECONDS_PER_PDF,
            rel_tol=0,
            abs_tol=1e-12,
        ),
        "runtime_average_limit_invalid",
        "average runtime limit is not six seconds per PDF",
    )
    return limits


def _validate_coverage(
    payload: Any,
    *,
    expected_pdf_count: int,
) -> Mapping[str, Any]:
    coverage = _mapping(
        payload,
        code="coverage_invalid",
        label="coverage",
    )
    _exact_keys(
        coverage,
        _COVERAGE_KEYS,
        code="coverage_invalid",
        label="coverage",
    )
    for key in _COVERAGE_KEYS:
        _integer(
            coverage[key],
            code="coverage_invalid",
            label=key,
        )
    _require(
        coverage
        == {
            "attempted": expected_pdf_count,
            "answered": expected_pdf_count,
            "omitted": 0,
            "invalid": 0,
            "rows_emitted": expected_pdf_count,
        },
        "coverage_incomplete",
        "capture output is incomplete or invalid",
    )
    return coverage


def _validate_image(
    payload: Any,
    *,
    limits: Mapping[str, Any],
) -> tuple[str, int, str, str, str]:
    image = _mapping(
        payload,
        code="image_evidence_invalid",
        label="image evidence",
    )
    _exact_keys(
        image,
        _IMAGE_KEYS,
        code="image_evidence_invalid",
        label="image evidence",
    )
    size_bytes = _integer(
        image["size_bytes"],
        code="image_evidence_invalid",
        label="image size",
        minimum=1,
    )
    _require(
        size_bytes <= limits["max_image_bytes"]
        and image["within_limit"] is True,
        "image_size_limit_exceeded",
        "image exceeds the official size limit",
    )
    image_id = image["image_id"]
    _require(
        isinstance(image_id, str) and bool(_IMAGE_ID_RE.fullmatch(image_id)),
        "image_evidence_invalid",
        "image ID must be a full SHA-256 identifier",
    )
    build_mode = image["build_mode"]
    _require(
        build_mode in {"fresh", "reused"},
        "image_evidence_invalid",
        "image build mode is invalid",
    )
    operating_system = image["operating_system"]
    architecture = image["architecture"]
    _require(
        isinstance(operating_system, str)
        and bool(_PLATFORM_PART_RE.fullmatch(operating_system))
        and isinstance(architecture, str)
        and bool(_PLATFORM_PART_RE.fullmatch(architecture)),
        "image_evidence_invalid",
        "image platform is invalid",
    )
    _require(
        image["source_binding_labels_match"] is True,
        "image_source_binding_mismatch",
        "image labels do not match the measured source",
    )
    return (
        image_id,
        size_bytes,
        str(build_mode),
        operating_system,
        architecture,
    )


def _validate_models(
    payload: Any,
    *,
    limits: Mapping[str, Any],
) -> Mapping[str, Any]:
    models = _mapping(
        payload,
        code="model_inventory_invalid",
        label="model inventory",
    )
    _exact_keys(
        models,
        _MODEL_KEYS,
        code="model_inventory_invalid",
        label="model inventory",
    )
    _require(
        models["scan_scope"] == IMAGE_MODEL_SCAN_SCOPE,
        "model_inventory_invalid",
        "model scan did not cover the source-bound fixed runtime roots",
    )
    extensions = _sequence(
        models["extensions"],
        code="model_inventory_invalid",
        label="model extensions",
    )
    _require(
        all(
            isinstance(item, str) and bool(_EXTENSION_RE.fullmatch(item))
            for item in extensions
        )
        and list(extensions) == sorted(set(extensions)),
        "model_inventory_invalid",
        "model extension inventory is invalid",
    )
    artifact_count = _integer(
        models["artifact_count"],
        code="model_inventory_invalid",
        label="model artifact count",
    )
    extension_counts = _mapping(
        models["extension_counts"],
        code="model_inventory_invalid",
        label="model extension counts",
    )
    _require(
        all(
            isinstance(key, str)
            and key in extensions
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for key, value in extension_counts.items()
        )
        and sum(extension_counts.values()) == artifact_count,
        "model_inventory_invalid",
        "model extension counts are inconsistent",
    )
    artifacts = _sequence(
        models["artifacts"],
        code="model_inventory_invalid",
        label="model artifacts",
    )
    _require(
        len(artifacts) == artifact_count,
        "model_inventory_invalid",
        "model artifact count does not match inventory",
    )
    sizes: list[int] = []
    counted_extensions: dict[str, int] = {}
    for item in artifacts:
        artifact = _mapping(
            item,
            code="model_inventory_invalid",
            label="model artifact",
        )
        _exact_keys(
            artifact,
            _MODEL_ITEM_KEYS,
            code="model_inventory_invalid",
            label="model artifact",
        )
        extension = artifact["extension"]
        _require(
            isinstance(extension, str) and extension in extensions,
            "model_inventory_invalid",
            "model artifact extension is outside the scan inventory",
        )
        size = _integer(
            artifact["bytes"],
            code="model_inventory_invalid",
            label="model artifact size",
        )
        sizes.append(size)
        counted_extensions[extension] = (
            counted_extensions.get(extension, 0) + 1
        )
    total = _integer(
        models["total_bytes"],
        code="model_inventory_invalid",
        label="total model bytes",
    )
    maximum = _integer(
        models["maximum_artifact_bytes"],
        code="model_inventory_invalid",
        label="maximum model artifact bytes",
    )
    _require(
        total == sum(sizes)
        and maximum == (max(sizes) if sizes else 0)
        and counted_extensions == dict(extension_counts),
        "model_inventory_invalid",
        "model artifact aggregates do not match the inventory",
    )
    _require(
        maximum <= limits["max_model_artifact_bytes"],
        "model_artifact_size_limit_exceeded",
        "an installed model artifact exceeds the official limit",
    )
    _require(
        total <= limits["max_total_model_bytes"],
        "model_total_size_limit_exceeded",
        "installed model artifacts exceed the official total limit",
    )
    return models


def _validate_runtime(
    payload: Any,
    *,
    run: Mapping[str, Any],
    expected_pdf_count: int,
) -> None:
    runtime = _mapping(
        payload,
        code="runtime_evidence_invalid",
        label="runtime evidence",
    )
    _exact_keys(
        runtime,
        _RUNTIME_KEYS,
        code="runtime_evidence_invalid",
        label="runtime evidence",
    )
    _require(
        isinstance(runtime["measurement_basis"], str)
        and "attempted_pdf_count" in runtime["measurement_basis"],
        "runtime_evidence_invalid",
        "runtime measurement basis is invalid",
    )
    _require(
        runtime["repeat_sample_count"] == 1
        and runtime["all_repeats_within_average_limit"] is True,
        "runtime_evidence_invalid",
        "single-capture runtime summary is inconsistent",
    )
    elapsed = float(run["elapsed_seconds"])
    expected_per_pdf = elapsed / expected_pdf_count
    for key in ("total_elapsed_seconds", "mean_run_elapsed_seconds"):
        _require(
            math.isclose(
                _number(
                    runtime[key],
                    code="runtime_evidence_invalid",
                    label=key,
                    minimum_exclusive=0,
                ),
                elapsed,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ),
            "runtime_evidence_invalid",
            f"{key} does not match the measured run",
        )
    per_pdf = _mapping(
        runtime["per_pdf_seconds"],
        code="runtime_evidence_invalid",
        label="per-PDF runtime",
    )
    _exact_keys(
        per_pdf,
        _PER_PDF_KEYS,
        code="runtime_evidence_invalid",
        label="per-PDF runtime",
    )
    for key in _PER_PDF_KEYS:
        _require(
            math.isclose(
                _number(
                    per_pdf[key],
                    code="runtime_evidence_invalid",
                    label=f"per-PDF {key}",
                    minimum_exclusive=0,
                ),
                expected_per_pdf,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ),
            "runtime_evidence_invalid",
            f"per-PDF {key} does not match the measured run",
        )
    expected_pairs = (
        ("peak_process_tree_rss_bytes", "peak_process_tree_rss_mib"),
        ("peak_container_memory_bytes", "peak_container_memory_mib"),
    )
    for bytes_key, mib_key in expected_pairs:
        _require(
            runtime[bytes_key] == run[bytes_key],
            "runtime_evidence_invalid",
            f"{bytes_key} does not match the measured run",
        )
        _require(
            math.isclose(
                _number(
                    runtime[mib_key],
                    code="runtime_evidence_invalid",
                    label=mib_key,
                    minimum_exclusive=0,
                ),
                run[bytes_key] / MIB,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ),
            "runtime_evidence_invalid",
            f"{mib_key} does not match the measured run",
        )
    _require(
        runtime["peak_process_tree_rss_source"]
        == run["peak_process_tree_rss_source"]
        and runtime["peak_container_memory_source"]
        == run["peak_container_memory_source"],
        "runtime_evidence_invalid",
        "runtime memory sources do not match the measured run",
    )


def _validate_capture(
    payload: Mapping[str, Any],
    *,
    evidence_sha256: str,
    expected_revision: str,
    expected_pdf_count: int,
) -> ValidatedCapture:
    _exact_keys(
        payload,
        _TOP_LEVEL_KEYS,
        code="capture_schema_invalid",
        label="capture",
    )
    _require(
        payload["schema_version"] == CAPTURE_SCHEMA,
        "capture_schema_invalid",
        "capture schema version is unsupported",
    )
    _require(
        payload["mode"] == "single_run_capture"
        and payload["requested_repeat_count"] == 1,
        "capture_mode_invalid",
        "evidence is not an explicit one-run parallel capture",
    )
    blocking = _sequence(
        payload["blocking_reasons"],
        code="capture_status_invalid",
        label="blocking reasons",
    )
    _require(
        not blocking,
        "capture_status_invalid",
        "capture contains a blocking reason",
    )
    warnings_value = _sequence(
        payload["warnings"],
        code="capture_status_invalid",
        label="warnings",
    )
    _require(
        all(isinstance(item, str) for item in warnings_value)
        and len(set(warnings_value)) == len(warnings_value),
        "capture_status_invalid",
        "capture warnings are invalid",
    )
    warnings = tuple(sorted(warnings_value))
    _require(
        set(warnings) <= ALLOWED_WARNINGS,
        "unexpected_capture_warning",
        "capture contains an unsupported warning",
    )
    expected_status = (
        "CAPTURE_COMPLETE_WITH_WARNINGS"
        if warnings
        else "CAPTURE_COMPLETE"
    )
    _require(
        payload["status"] == expected_status
        and payload["status"] in COMPLETE_STATUSES,
        "capture_status_invalid",
        "capture status is inconsistent with its warnings",
    )
    environment = _mapping(
        payload["environment"],
        code="docker_environment_invalid",
        label="Docker environment",
    )
    _exact_keys(
        environment,
        frozenset(
            {"docker_available", "docker_status", "docker_server_version"}
        ),
        code="docker_environment_invalid",
        label="Docker environment",
    )
    _require(
        environment["docker_available"] is True
        and environment["docker_status"] == "available"
        and isinstance(environment["docker_server_version"], str)
        and bool(environment["docker_server_version"].strip()),
        "docker_environment_invalid",
        "capture did not run against an available Docker server",
    )
    limits = _validate_limits(payload["limits"])
    deadlines = _mapping(
        payload["deadline_controls"],
        code="deadline_controls_invalid",
        label="deadline controls",
    )
    _require(
        deadlines
        == {
            "whole_run_timeout_seconds": limits["timeout_seconds"],
            "per_case_deadline_enforced": False,
        },
        "deadline_controls_invalid",
        "deadline controls do not match the measured limits",
    )
    _require(
        payload["resilience"]
        == {
            "output_commit_strategy": "batch_end_atomic",
            "partial_progress_recovery": False,
        },
        "resilience_evidence_invalid",
        "runtime resilience evidence is unexpected",
    )
    source_binding = _validate_source_binding(
        payload["source_binding"],
        expected_revision=expected_revision,
    )
    input_binding = _validate_input_binding(
        payload["input_binding"],
        expected_pdf_count=expected_pdf_count,
    )
    coverage = _validate_coverage(
        payload["coverage"],
        expected_pdf_count=expected_pdf_count,
    )
    (
        image_id,
        image_size,
        image_build_mode,
        operating_system,
        architecture,
    ) = _validate_image(payload["image"], limits=limits)
    models = _validate_models(
        payload["installed_model_artifacts"],
        limits=limits,
    )
    determinism = _mapping(
        payload["determinism"],
        code="capture_determinism_invalid",
        label="single-capture determinism",
    )
    _exact_keys(
        determinism,
        frozenset({"evaluated", "reason"}),
        code="capture_determinism_invalid",
        label="single-capture determinism",
    )
    _require(
        determinism["evaluated"] is False
        and isinstance(determinism["reason"], str)
        and "external comparison" in determinism["reason"],
        "capture_self_certified_determinism",
        "single capture incorrectly claims determinism",
    )
    _require(
        payload["bindings_reverified_after_runs"] is True,
        "bindings_not_reverified",
        "capture bindings were not reverified after execution",
    )
    runs = _sequence(
        payload["runs"],
        code="run_evidence_invalid",
        label="runs",
    )
    _require(
        len(runs) == 1,
        "single_run_count_invalid",
        "single-run capture must contain exactly one run",
    )
    run = _mapping(
        runs[0],
        code="run_evidence_invalid",
        label="run",
    )
    _exact_keys(
        run,
        _RUN_KEYS,
        code="run_evidence_invalid",
        label="run",
    )
    _require(
        run["repeat_index"] == 1,
        "run_evidence_invalid",
        "single-run capture has an invalid repeat index",
    )
    elapsed = _number(
        run["elapsed_seconds"],
        code="run_evidence_invalid",
        label="elapsed runtime",
        minimum_exclusive=0,
    )
    _require(
        elapsed <= OFFICIAL_RUNTIME_LIMIT_SECONDS
        and elapsed / expected_pdf_count <= OFFICIAL_SECONDS_PER_PDF,
        "runtime_limit_exceeded",
        "capture exceeds the official runtime envelope",
    )
    peak_rss = _integer(
        run["peak_process_tree_rss_bytes"],
        code="run_evidence_invalid",
        label="peak process-tree RSS",
        minimum=1,
    )
    peak_container = _integer(
        run["peak_container_memory_bytes"],
        code="run_evidence_invalid",
        label="peak container memory",
        minimum=1,
    )
    _require(
        peak_container <= 8 * GIB,
        "container_memory_limit_exceeded",
        "capture exceeds the 8 GiB container memory limit",
    )
    _require(
        math.isclose(
            _number(
                run["peak_process_tree_rss_mib"],
                code="run_evidence_invalid",
                label="peak process-tree RSS MiB",
                minimum_exclusive=0,
            ),
            peak_rss / MIB,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        and math.isclose(
            _number(
                run["peak_container_memory_mib"],
                code="run_evidence_invalid",
                label="peak container memory MiB",
                minimum_exclusive=0,
            ),
            peak_container / MIB,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ),
        "run_evidence_invalid",
        "run memory units are inconsistent",
    )
    _require(
        run["peak_process_tree_rss_source"]
        == "in_container_procfs_summed_process_tree_vmrss"
        and run["peak_container_memory_source"]
        == "docker_stats_mem_usage_cgroup",
        "run_evidence_invalid",
        "run memory sources are invalid",
    )
    output_sha256 = _digest(
        run["output_sha256"],
        code="run_evidence_invalid",
        label="output digest",
    )
    output_bytes = _integer(
        run["output_bytes"],
        code="run_evidence_invalid",
        label="output bytes",
        minimum=1,
    )
    _require(
        output_bytes <= limits["max_output_bytes"],
        "output_size_limit_exceeded",
        "prediction output exceeds the official size limit",
    )
    _require(
        run["coverage"] == coverage,
        "coverage_binding_mismatch",
        "run coverage does not match top-level coverage",
    )
    _validate_runtime(
        payload["runtime"],
        run=run,
        expected_pdf_count=expected_pdf_count,
    )
    return ValidatedCapture(
        evidence_sha256=evidence_sha256,
        source_binding=source_binding,
        input_binding=input_binding,
        limits=limits,
        coverage=coverage,
        installed_model_artifacts=models,
        warnings=warnings,
        output_sha256=output_sha256,
        output_bytes=output_bytes,
        elapsed_seconds=elapsed,
        peak_process_tree_rss_bytes=peak_rss,
        peak_container_memory_bytes=peak_container,
        image_id=image_id,
        image_size_bytes=image_size,
        image_build_mode=image_build_mode,
        operating_system=operating_system,
        architecture=architecture,
    )


def _same(left: Any, right: Any, *, code: str, message: str) -> None:
    if left != right:
        _fail(code, message)


def _assert_aggregate_only(value: Any) -> None:
    """Reject accidental identity-bearing material before serialization."""

    forbidden_keys = {
        "case_id",
        "filename",
        "file_path",
        "prediction",
        "predictions",
        "truth",
        "labels",
    }

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, nested in item.items():
                if str(key).casefold() in forbidden_keys:
                    _fail(
                        "aggregate_only_violation",
                        "final evidence contains an identity-bearing key",
                    )
                visit(nested)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes)
        ):
            for nested in item:
                visit(nested)
        elif isinstance(item, str):
            if _CASE_ID_RE.search(item) or _ABSOLUTE_PATH_RE.search(item):
                _fail(
                    "aggregate_only_violation",
                    "final evidence contains identity-bearing text",
                )

    visit(value)


def compare_capture_evidence(
    capture_paths: Sequence[Path | str],
    *,
    expected_source_revision: str,
    expected_pdf_count: int = 5000,
) -> Mapping[str, Any]:
    """Validate and compare exactly two independent aggregate captures."""

    expected_revision = _revision(
        expected_source_revision,
        code="expected_source_revision_invalid",
        label="expected source revision",
    )
    _require(
        isinstance(expected_pdf_count, int)
        and not isinstance(expected_pdf_count, bool)
        and expected_pdf_count > 0,
        "expected_pdf_count_invalid",
        "expected PDF count must be a positive integer",
    )
    _require(
        len(capture_paths) == 2,
        "capture_count_invalid",
        "exactly two single-run capture files are required",
    )
    _require(
        Path(capture_paths[0]).resolve() != Path(capture_paths[1]).resolve(),
        "capture_files_not_independent",
        "the two capture arguments must name different files",
    )
    captures: list[ValidatedCapture] = []
    for index, raw_path in enumerate(capture_paths, start=1):
        payload, evidence_sha = _read_capture(
            Path(raw_path),
            capture_index=index,
        )
        captures.append(
            _validate_capture(
                payload,
                evidence_sha256=evidence_sha,
                expected_revision=expected_revision,
                expected_pdf_count=expected_pdf_count,
            )
        )
    first, second = captures
    _same(
        first.source_binding,
        second.source_binding,
        code="source_binding_mismatch",
        message="capture source bindings differ",
    )
    _same(
        first.input_binding,
        second.input_binding,
        code="input_binding_mismatch",
        message="capture input bindings differ",
    )
    _same(
        first.limits,
        second.limits,
        code="runtime_limits_mismatch",
        message="capture runtime limits differ",
    )
    _same(
        first.coverage,
        second.coverage,
        code="coverage_binding_mismatch",
        message="capture coverage differs",
    )
    _same(
        first.installed_model_artifacts,
        second.installed_model_artifacts,
        code="model_inventory_mismatch",
        message="installed model inventories differ",
    )
    _same(
        first.warnings,
        second.warnings,
        code="capture_warnings_mismatch",
        message="capture warnings differ",
    )
    _same(
        (first.operating_system, first.architecture),
        (second.operating_system, second.architecture),
        code="image_platform_mismatch",
        message="capture image platforms differ",
    )
    _same(
        first.output_sha256,
        second.output_sha256,
        code="output_hash_mismatch",
        message="prediction output SHA-256 differs between captures",
    )
    _same(
        first.output_bytes,
        second.output_bytes,
        code="output_size_mismatch",
        message="prediction output byte size differs between captures",
    )
    warnings = sorted(set(first.warnings) | set(second.warnings))
    result: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "comparator_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "status": "PASS_WITH_WARNINGS" if warnings else "PASS",
        "blocking_reasons": [],
        "warnings": warnings,
        "aggregate_only": True,
        "source_binding": dict(first.source_binding),
        "input_binding": dict(first.input_binding),
        "limits": dict(first.limits),
        "coverage": dict(first.coverage),
        "installed_model_artifacts": dict(
            first.installed_model_artifacts
        ),
        "capture_evidence_sha256": [
            first.evidence_sha256,
            second.evidence_sha256,
        ],
        "determinism": {
            "evaluated": True,
            "method": (
                "two_independent_single_run_captures_compared_by_"
                "sha256_size_and_coverage"
            ),
            "independent_capture_count": 2,
            "byte_identical": True,
            "coverage_identical": True,
            "unique_output_hash_count": 1,
            "output_sha256": first.output_sha256,
            "output_bytes": first.output_bytes,
        },
        "runtime": {
            "capture_count": 2,
            "elapsed_seconds": [
                first.elapsed_seconds,
                second.elapsed_seconds,
            ],
            "seconds_per_pdf": [
                first.elapsed_seconds / expected_pdf_count,
                second.elapsed_seconds / expected_pdf_count,
            ],
            "max_elapsed_seconds": max(
                first.elapsed_seconds,
                second.elapsed_seconds,
            ),
            "max_seconds_per_pdf": max(
                first.elapsed_seconds,
                second.elapsed_seconds,
            )
            / expected_pdf_count,
            "peak_process_tree_rss_bytes": max(
                first.peak_process_tree_rss_bytes,
                second.peak_process_tree_rss_bytes,
            ),
            "peak_container_memory_bytes": max(
                first.peak_container_memory_bytes,
                second.peak_container_memory_bytes,
            ),
            "all_captures_within_official_runtime_limit": True,
            "all_captures_within_official_memory_limit": True,
        },
        "images": {
            "same_platform": True,
            "operating_system": first.operating_system,
            "architecture": first.architecture,
            "source_binding_verified": True,
            "image_ids": [first.image_id, second.image_id],
            "size_bytes": [
                first.image_size_bytes,
                second.image_size_bytes,
            ],
            "build_modes": [
                first.image_build_mode,
                second.image_build_mode,
            ],
        },
    }
    _assert_aggregate_only(result)
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _blocked_payload(
    *,
    code: str,
    expected_source_revision: str,
    expected_pdf_count: int,
) -> Mapping[str, Any]:
    revision = (
        expected_source_revision
        if _REVISION_RE.fullmatch(expected_source_revision)
        else None
    )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "comparator_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "status": "BLOCKED",
        "blocking_reasons": [code],
        "warnings": [],
        "aggregate_only": True,
        "source_revision": revision,
        "expected_pdf_count": (
            expected_pdf_count
            if isinstance(expected_pdf_count, int)
            and not isinstance(expected_pdf_count, bool)
            and expected_pdf_count > 0
            else None
        ),
        "determinism": {
            "evaluated": False,
            "independent_capture_count": 0,
            "byte_identical": False,
        },
    }
    _assert_aggregate_only(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed comparison of two aggregate WO20 single-run "
            "Docker evidence files."
        )
    )
    parser.add_argument(
        "--capture",
        action="append",
        default=[],
        help="Aggregate single-run evidence JSON; provide exactly twice.",
    )
    parser.add_argument(
        "--expected-source-revision",
        required=True,
        help="Full Git SHA that both captures must bind.",
    )
    parser.add_argument(
        "--expected-pdf-count",
        type=int,
        default=5000,
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_path = Path(args.output)
    try:
        result = compare_capture_evidence(
            tuple(Path(value) for value in args.capture),
            expected_source_revision=args.expected_source_revision,
            expected_pdf_count=args.expected_pdf_count,
        )
    except WO20ParallelCompareError as exc:
        result = _blocked_payload(
            code=exc.code,
            expected_source_revision=args.expected_source_revision,
            expected_pdf_count=args.expected_pdf_count,
        )
        _atomic_write_json(output_path, result)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _atomic_write_json(output_path, result)
    print(
        "WO20 parallel captures are byte-identical and aggregate-only: "
        f"{output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
