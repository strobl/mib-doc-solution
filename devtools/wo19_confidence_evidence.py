#!/usr/bin/env python3
"""Build aggregate-only WO19 confidence-refit evidence.

The builder deliberately separates three phases:

1. validate a truth-blind, byte-repeated S0 production capture;
2. run the normal ``solution.py`` batch path itself and prove byte equality;
3. join public truth only after the capture is frozen, run grouped OOF
   comparison, and shadow-apply the selected confidence-only calibrator.

Identity-bearing rows, layout memberships, contexts, truth, and grouped-CV
samples remain in memory or in an explicitly external sample file.  The JSON
and Markdown returned for repository storage contain aggregates and digests
only.  A candidate that passes the statistical promotion gates is *not*
reported as deployed: the result becomes ``needs_s1`` and requires a separate
S1 runtime integration (code and artifact as needed), followed by a fresh
double production capture.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.confidence_refit_cv import (  # noqa: E402
    FOLD_COUNT,
    PROMOTION_TOLERANCE,
    REPEAT_SEEDS,
    REPORT_SCHEMA,
    SAMPLE_SCHEMA,
    GroupedCalibrationExample,
    compare_confidence_families,
    grouped_folds,
    load_grouped_examples,
)
from devtools.experiment_control import canonical_json  # noqa: E402
from devtools.final_confidence_capture import (  # noqa: E402
    CAPTURE_SCHEMA,
    OBSERVATION_SCHEMA,
    CaptureBindings,
    context_contract_sha256,
    non_confidence_jsonl,
    prediction_jsonl,
    producer_graph_sha256,
)
from devtools.grouped_recovery_evidence import (  # noqa: E402
    FrozenLayoutManifest,
    load_layout_manifest,
)
from devtools.ocr_ablation import _input_tree_sha256  # noqa: E402
from mib_pipeline import final_confidence as final_confidence_module  # noqa: E402
from mib_pipeline.confidence_refit import (  # noqa: E402
    CALIBRATION_FAMILIES,
    CalibrationExample,
    FittedConfidenceCalibrator,
)
from mib_pipeline.final_confidence import FinalConfidenceContext  # noqa: E402
from mib_pipeline.models import CASE_ID_PATTERN, FIELD_NAMES, PredictionRow  # noqa: E402
from mib_pipeline.output_confidence import (  # noqa: E402
    PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH,
)
from scripts import evaluate as official_evaluate  # noqa: E402


EVIDENCE_SCHEMA = "mib-wo19-confidence-evidence/v1"
BOUND_SAMPLE_SCHEMA = "mib-wo19-bound-confidence-samples/v1"
EXPECTED_RECORD_COUNT = 32
TRUTH_FIELDS = (
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|[\s\"'])(?:/(?:Users|private|tmp)/|[A-Za-z]:\\|\\\\)"
)
_CASE_ID_TEXT_RE = re.compile(r"\bMIB-[0-9]{6}\b")

_CAPTURE_KEYS = {
    "schema_version",
    "frozen_before_truth_join",
    "confidence_topology",
    "bindings",
    "record_count",
    "full_predictions_jsonl_sha256",
    "non_confidence_jsonl_sha256",
    "rows",
}
_CAPTURE_BINDING_KEYS = {
    field.name for field in dataclasses.fields(CaptureBindings)
}
_OBSERVATION_KEYS = {
    "schema_version",
    "source_revision_sha",
    "layout_manifest_sha256",
    "input_tree_sha256",
    "producer_graph_sha256",
    "producer_source_sha256",
    "runtime_confidence_artifact_sha256",
    "runtime_confidence_artifact_file_sha256",
    "context_contract_sha256",
    "context_contract_source_sha256",
    "confidence_topology",
    "capture_sha256",
    "rerun_capture_sha256",
    "full_predictions_jsonl_sha256",
    "rerun_full_predictions_jsonl_sha256",
    "non_confidence_jsonl_sha256",
    "rerun_non_confidence_jsonl_sha256",
    "record_count",
    "capture_run_count",
    "missing_record_count",
    "duplicate_record_count",
    "invalid_record_count",
    "out_of_range_confidence_count",
    "byte_deterministic",
    "truth_or_label_input_count",
    "identity_bearing_capture_storage",
    "aggregate_only",
}
_BOUND_BINDING_KEYS = {
    "source_revision_sha",
    "capture_sha256",
    "full_output_sha256",
    "non_confidence_output_sha256",
    "truth_sha256",
    "layout_manifest_sha256",
    "input_tree_sha256",
    "context_contract_sha256",
    "runtime_confidence_artifact_sha256",
    "runtime_confidence_artifact_file_sha256",
    "producer_graph_sha256",
    "producer_source_sha256",
    "cv_source_sha256",
    "evaluator_source_sha256",
    "evidence_builder_source_sha256",
}
_EXPECTED_CONFIDENCE_TOPOLOGY = {
    "captured_probability": "post_current_outer_recalibrator",
    "candidate_position": "downstream_confidence_only",
    "parallel_batch_runtime_equivalence": (
        "not_claimed_without_shadow_solution_proof"
    ),
}


class WO19EvidenceError(RuntimeError):
    """An identity, byte, source, schema, or evaluation binding failed."""


@dataclass(frozen=True)
class WO19EvidenceResult:
    """Repository-safe evidence plus explicitly external private artifacts."""

    evidence: Mapping[str, Any]
    markdown: str
    bound_samples_bytes: bytes
    selected_artifact_bytes: bytes
    needs_s1: bool


@dataclass(frozen=True)
class _ValidatedCapture:
    payload: Mapping[str, Any]
    rows: tuple[PredictionRow, ...]
    contexts: tuple[FinalConfidenceContext, ...]
    layout: FrozenLayoutManifest
    prediction_bytes: bytes
    non_confidence_bytes: bytes
    capture_sha256: str
    bindings: Mapping[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _artifact_bytes(calibrator: FittedConfidenceCalibrator) -> bytes:
    return _canonical_bytes(calibrator.to_runtime_mapping())


def _require_digest(value: Any, *, label: str) -> str:
    rendered = str(value).strip().casefold()
    if not _SHA256_RE.fullmatch(rendered):
        raise WO19EvidenceError(f"{label} must be a lowercase SHA-256 digest")
    return rendered


def _require_revision(value: Any) -> str:
    rendered = str(value).strip().casefold()
    if not _REVISION_RE.fullmatch(rendered):
        raise WO19EvidenceError(
            "source revision must be a full lowercase Git commit SHA"
        )
    return rendered


def _pairs_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WO19EvidenceError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_canonical_object(
    path: Path | str,
    *,
    label: str,
) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = Path(path).read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WO19EvidenceError(
            f"{label} must be a readable canonical JSON object"
        ) from exc
    if not isinstance(value, Mapping):
        raise WO19EvidenceError(f"{label} must be a JSON object")
    if raw != _canonical_bytes(value):
        raise WO19EvidenceError(f"{label} bytes are not canonical")
    return value, raw


def _same(label: str, left: Any, right: Any) -> None:
    if left != right:
        raise WO19EvidenceError(f"{label} binding mismatch")


def _non_confidence_projection(row: PredictionRow) -> dict[str, Any]:
    values = row.to_dict()
    return {
        field: values[field]
        for field in FIELD_NAMES
        if field != "confidence"
    }


def _production_jsonl(rows: Sequence[PredictionRow]) -> bytes:
    return "".join(
        json.dumps(
            row.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in sorted(rows, key=lambda value: value.case_id)
    ).encode("utf-8")


def _non_confidence_jsonl(rows: Sequence[PredictionRow]) -> bytes:
    return "".join(
        json.dumps(
            _non_confidence_projection(row),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in sorted(rows, key=lambda value: value.case_id)
    ).encode("utf-8")


def assert_non_confidence_unchanged(
    baseline: PredictionRow,
    candidate: PredictionRow,
) -> None:
    """Fail closed if any canonical output field except confidence changed."""

    if _non_confidence_projection(candidate) != _non_confidence_projection(
        baseline
    ):
        raise WO19EvidenceError(
            "shadow calibrator changed a non-confidence field"
        )


def _parse_capture_rows(
    payload: Mapping[str, Any],
) -> tuple[tuple[PredictionRow, ...], tuple[FinalConfidenceContext, ...]]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_RECORD_COUNT:
        raise WO19EvidenceError("capture must contain exactly 32 rows")
    rows: list[PredictionRow] = []
    contexts: list[FinalConfidenceContext] = []
    previous_case_id: str | None = None
    seen: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "case_id",
            "layout_group",
            "prediction",
            "context",
        }:
            raise WO19EvidenceError("capture row fields are not exact")
        case_id = raw.get("case_id")
        group_id = raw.get("layout_group")
        if (
            not isinstance(case_id, str)
            or not CASE_ID_PATTERN.fullmatch(case_id)
            or not isinstance(group_id, str)
            or not group_id.strip()
        ):
            raise WO19EvidenceError("capture identity or layout group is invalid")
        if case_id in seen or (
            previous_case_id is not None and case_id <= previous_case_id
        ):
            raise WO19EvidenceError(
                "capture cases must be unique and strictly ordered"
            )
        prediction = raw.get("prediction")
        context_value = raw.get("context")
        if (
            not isinstance(prediction, Mapping)
            or set(prediction) != set(FIELD_NAMES)
            or not isinstance(context_value, Mapping)
        ):
            raise WO19EvidenceError("capture prediction or context is malformed")
        try:
            row = PredictionRow.from_mapping(prediction)
            context = FinalConfidenceContext(**context_value)
        except (TypeError, ValueError) as exc:
            raise WO19EvidenceError(
                "capture prediction or context is invalid"
            ) from exc
        if row.to_dict() != dict(prediction):
            raise WO19EvidenceError("capture prediction is not canonical")
        if row.case_id != case_id or context.final_class != row.adjudication:
            raise WO19EvidenceError(
                "capture identity, decision, and context are not aligned"
            )
        rows.append(row)
        contexts.append(context)
        seen.add(case_id)
        previous_case_id = case_id
    return tuple(rows), tuple(contexts)


def _validate_capture_bundle(
    *,
    capture_path: Path,
    rerun_capture_path: Path,
    capture_predictions_path: Path,
    rerun_capture_predictions_path: Path,
    observation_path: Path,
    layout_manifest_path: Path,
    input_dir: Path,
    source_revision_sha: str,
) -> _ValidatedCapture:
    capture, capture_bytes = _read_canonical_object(
        capture_path,
        label="capture",
    )
    rerun, rerun_bytes = _read_canonical_object(
        rerun_capture_path,
        label="rerun capture",
    )
    if capture_bytes != rerun_bytes or capture != rerun:
        raise WO19EvidenceError("the two production captures are not identical")
    if set(capture) != _CAPTURE_KEYS:
        raise WO19EvidenceError("capture top-level fields are not exact")
    if (
        capture.get("schema_version") != CAPTURE_SCHEMA
        or capture.get("frozen_before_truth_join") is not True
        or capture.get("record_count") != EXPECTED_RECORD_COUNT
    ):
        raise WO19EvidenceError("capture contract is not the frozen 32-case S0")
    _same(
        "capture confidence topology",
        capture.get("confidence_topology"),
        _EXPECTED_CONFIDENCE_TOPOLOGY,
    )
    bindings = capture.get("bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != _CAPTURE_BINDING_KEYS:
        raise WO19EvidenceError("capture bindings are not exact")
    revision = _require_revision(source_revision_sha)
    _same("capture source revision", bindings["source_revision_sha"], revision)
    for key in _CAPTURE_BINDING_KEYS - {
        "source_revision_sha",
        "runtime_confidence_artifact_id",
    }:
        _require_digest(bindings[key], label=f"capture {key}")
    if not isinstance(bindings["runtime_confidence_artifact_id"], str) or not str(
        bindings["runtime_confidence_artifact_id"]
    ).strip():
        raise WO19EvidenceError("capture runtime artifact ID is empty")

    rows, contexts = _parse_capture_rows(capture)
    predicted = _production_jsonl(rows)
    projected = _non_confidence_jsonl(rows)
    _same(
        "capture full-output hash",
        _sha256_bytes(predicted),
        capture["full_predictions_jsonl_sha256"],
    )
    _same(
        "capture non-confidence hash",
        _sha256_bytes(projected),
        capture["non_confidence_jsonl_sha256"],
    )
    try:
        first_predictions = capture_predictions_path.read_bytes()
        second_predictions = rerun_capture_predictions_path.read_bytes()
    except OSError as exc:
        raise WO19EvidenceError(
            "capture prediction JSONL files are unreadable"
        ) from exc
    if first_predictions != predicted or second_predictions != predicted:
        raise WO19EvidenceError(
            "capture prediction files do not equal production-writer bytes"
        )

    observation, observation_bytes = _read_canonical_object(
        observation_path,
        label="capture observation",
    )
    if set(observation) != _OBSERVATION_KEYS:
        raise WO19EvidenceError("capture observation fields are not exact")
    expected_observation = {
        "schema_version": OBSERVATION_SCHEMA,
        "source_revision_sha": revision,
        "layout_manifest_sha256": bindings["layout_manifest_sha256"],
        "input_tree_sha256": bindings["input_tree_sha256"],
        "producer_graph_sha256": bindings["producer_graph_sha256"],
        "producer_source_sha256": bindings["producer_source_sha256"],
        "runtime_confidence_artifact_sha256": (
            bindings["runtime_confidence_artifact_sha256"]
        ),
        "runtime_confidence_artifact_file_sha256": (
            bindings["runtime_confidence_artifact_file_sha256"]
        ),
        "context_contract_sha256": bindings["context_contract_sha256"],
        "context_contract_source_sha256": (
            bindings["context_contract_source_sha256"]
        ),
        "confidence_topology": capture["confidence_topology"],
        "capture_sha256": _sha256_bytes(capture_bytes),
        "rerun_capture_sha256": _sha256_bytes(rerun_bytes),
        "full_predictions_jsonl_sha256": _sha256_bytes(predicted),
        "rerun_full_predictions_jsonl_sha256": _sha256_bytes(predicted),
        "non_confidence_jsonl_sha256": _sha256_bytes(projected),
        "rerun_non_confidence_jsonl_sha256": _sha256_bytes(projected),
        "record_count": EXPECTED_RECORD_COUNT,
        "capture_run_count": 2,
        "missing_record_count": 0,
        "duplicate_record_count": 0,
        "invalid_record_count": 0,
        "out_of_range_confidence_count": 0,
        "byte_deterministic": True,
        "truth_or_label_input_count": 0,
        "identity_bearing_capture_storage": "external",
        "aggregate_only": True,
    }
    _same("capture observation", dict(observation), expected_observation)

    layout = load_layout_manifest(layout_manifest_path)
    if (
        not layout.frozen_before_scoring
        or len(layout.case_ids) != EXPECTED_RECORD_COUNT
        or len(layout.groups) < FOLD_COUNT
    ):
        raise WO19EvidenceError(
            "layout manifest is not a frozen 32-case, five-fold cohort"
        )
    _same(
        "layout manifest hash",
        layout.sha256,
        bindings["layout_manifest_sha256"],
    )
    case_groups = {
        case_id: group_id
        for group_id, case_ids in layout.groups.items()
        for case_id in case_ids
    }
    case_ids = tuple(row.case_id for row in rows)
    _same("capture/layout case set", set(case_ids), set(layout.case_ids))
    for raw in capture["rows"]:
        _same(
            "capture layout membership",
            raw["layout_group"],
            case_groups[raw["case_id"]],
        )

    input_tree_sha, input_count = _input_tree_sha256(input_dir)
    if input_count != EXPECTED_RECORD_COUNT:
        raise WO19EvidenceError("input tree must contain exactly 32 PDFs")
    _same("input tree", input_tree_sha, bindings["input_tree_sha256"])
    input_cases = {
        path.stem
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.casefold() == ".pdf"
    }
    _same("input/capture case set", input_cases, set(case_ids))
    _same(
        "observation bytes hash",
        _sha256_bytes(observation_bytes),
        _sha256_file(observation_path),
    )
    return _ValidatedCapture(
        payload=capture,
        rows=rows,
        contexts=contexts,
        layout=layout,
        prediction_bytes=predicted,
        non_confidence_bytes=projected,
        capture_sha256=_sha256_bytes(capture_bytes),
        bindings=dict(bindings),
    )


def _git_bytes(revision: str, relative_path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{revision}:{relative_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise WO19EvidenceError(
            f"source revision does not contain {relative_path}"
        ) from exc


def _git_paths(revision: str) -> tuple[str, ...]:
    try:
        output = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", revision],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise WO19EvidenceError("cannot inspect source revision tree") from exc
    return tuple(line for line in output.splitlines() if line)


def _git_producer_graph_sha256(revision: str) -> str:
    graph_paths = []
    for path in _git_paths(revision):
        if (
            path.startswith("mib_pipeline/")
            and (
                path.endswith(".py")
                or path.startswith("mib_pipeline/artifacts/")
            )
        ) or path in {"requirements.lock", "Dockerfile", "run.sh"}:
            graph_paths.append(path)
    if not graph_paths:
        raise WO19EvidenceError("source revision has no production graph")
    entries = []
    for path in sorted(graph_paths):
        data = _git_bytes(revision, path)
        entries.append(
            {
                "path": path,
                "sha256": _sha256_bytes(data),
                "size_bytes": len(data),
            }
        )
    return _sha256_bytes(_canonical_bytes(entries))


def _runtime_artifact_binding() -> tuple[str, str, str]:
    try:
        raw = PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WO19EvidenceError(
            "pinned runtime confidence artifact is unreadable"
        ) from exc
    if not isinstance(value, Mapping):
        raise WO19EvidenceError(
            "pinned runtime confidence artifact must be an object"
        )
    canonical_sha = _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    artifact_id = value.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise WO19EvidenceError("pinned runtime artifact has no ID")
    return canonical_sha, _sha256_bytes(raw), artifact_id


def _validate_source_and_runtime_bindings(
    capture: _ValidatedCapture,
    *,
    source_revision_sha: str,
    verify_repository: bool,
) -> dict[str, str]:
    revision = _require_revision(source_revision_sha)
    capture_source_path = "devtools/final_confidence_capture.py"
    context_source_path = "mib_pipeline/final_confidence.py"
    cv_source_path = "devtools/confidence_refit_cv.py"
    evaluator_source_path = "scripts/evaluate.py"
    evidence_builder_source_path = "devtools/wo19_confidence_evidence.py"
    if verify_repository:
        try:
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise WO19EvidenceError("cannot inspect repository state") from exc
        _same("repository HEAD", head, revision)
        if status.strip():
            raise WO19EvidenceError(
                "evidence generation requires a clean S0 checkout"
            )
        _same(
            "committed capture producer source",
            _sha256_bytes(_git_bytes(revision, capture_source_path)),
            capture.bindings["producer_source_sha256"],
        )
        _same(
            "committed context source",
            _sha256_bytes(_git_bytes(revision, context_source_path)),
            capture.bindings["context_contract_source_sha256"],
        )
        _same(
            "committed production graph",
            _git_producer_graph_sha256(revision),
            capture.bindings["producer_graph_sha256"],
        )
        _same(
            "live production graph",
            producer_graph_sha256(),
            capture.bindings["producer_graph_sha256"],
        )
        _same(
            "live capture producer source",
            _sha256_file(REPO_ROOT / capture_source_path),
            capture.bindings["producer_source_sha256"],
        )
        _same(
            "live context source",
            _sha256_file(Path(final_confidence_module.__file__)),
            capture.bindings["context_contract_source_sha256"],
        )
        _same(
            "live context contract",
            context_contract_sha256(),
            capture.bindings["context_contract_sha256"],
        )
        artifact_sha, artifact_file_sha, artifact_id = (
            _runtime_artifact_binding()
        )
        _same(
            "live runtime artifact",
            artifact_sha,
            capture.bindings["runtime_confidence_artifact_sha256"],
        )
        _same(
            "live runtime artifact file",
            artifact_file_sha,
            capture.bindings["runtime_confidence_artifact_file_sha256"],
        )
        _same(
            "live runtime artifact ID",
            artifact_id,
            capture.bindings["runtime_confidence_artifact_id"],
        )
        cv_sha = _sha256_bytes(_git_bytes(revision, cv_source_path))
        evaluator_sha = _sha256_bytes(
            _git_bytes(revision, evaluator_source_path)
        )
        evidence_builder_sha = _sha256_bytes(
            _git_bytes(revision, evidence_builder_source_path)
        )
    else:
        cv_sha = _sha256_file(REPO_ROOT / cv_source_path)
        evaluator_sha = _sha256_file(REPO_ROOT / evaluator_source_path)
        evidence_builder_sha = _sha256_file(
            REPO_ROOT / evidence_builder_source_path
        )
    return {
        "cv_source_sha256": cv_sha,
        "evaluator_source_sha256": evaluator_sha,
        "evidence_builder_source_sha256": evidence_builder_sha,
    }


def _default_solution_runner(input_dir: Path, output_path: Path) -> int:
    from solution import configured_worker_limit, main as solution_main

    if configured_worker_limit() != 4:
        raise WO19EvidenceError(
            "normal equivalence proof requires the four-worker solution path"
        )

    return solution_main(
        ["solution.py", str(input_dir), str(output_path)]
    )


def _run_normal_solution_equivalence(
    *,
    input_dir: Path,
    expected_prediction_bytes: bytes,
    runner: Callable[[Path, Path], int] | None,
) -> str:
    selected_runner = runner or _default_solution_runner
    with tempfile.TemporaryDirectory(prefix="mib-wo19-normal-batch-") as directory:
        output_path = Path(directory) / "predictions.jsonl"
        exit_code = selected_runner(input_dir, output_path)
        if exit_code != 0 or not output_path.is_file():
            raise WO19EvidenceError(
                "normal solution.py batch path did not complete successfully"
            )
        produced = output_path.read_bytes()
    if produced != expected_prediction_bytes:
        raise WO19EvidenceError(
            "normal solution.py/BatchRunner bytes differ from capture bytes"
        )
    try:
        parsed = [
            json.loads(line)
            for line in produced.decode("utf-8").splitlines()
            if line
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WO19EvidenceError(
            "normal solution.py batch output is invalid JSONL"
        ) from exc
    if len(parsed) != EXPECTED_RECORD_COUNT:
        raise WO19EvidenceError(
            "normal solution.py batch output is not exactly 32 records"
        )
    return _sha256_bytes(produced)


def _load_truth(
    *,
    path: Path,
    expected_sha256: str,
    expected_case_ids: Sequence[str],
) -> tuple[dict[str, Mapping[str, Any]], str]:
    expected_sha = _require_digest(expected_sha256, label="expected truth")
    actual_sha = _sha256_file(path)
    _same("truth file hash", actual_sha, expected_sha)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != TRUTH_FIELDS:
                raise WO19EvidenceError("truth CSV fields are not exact")
            raw_rows = list(reader)
    except (OSError, csv.Error) as exc:
        raise WO19EvidenceError("truth CSV is unreadable") from exc
    if len(raw_rows) != EXPECTED_RECORD_COUNT:
        raise WO19EvidenceError("truth CSV must contain exactly 32 rows")
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        if set(row) != set(TRUTH_FIELDS) or None in row:
            raise WO19EvidenceError("truth row schema is not exact")
        case_id = str(row["case_id"]).strip()
        if not CASE_ID_PATTERN.fullmatch(case_id) or case_id in indexed:
            raise WO19EvidenceError(
                "truth case identities are invalid or duplicated"
            )
        adjudication = str(row["adjudication"]).strip()
        if adjudication not in {"APPROVED", "DENIED", "NEEDS_REVIEW"}:
            raise WO19EvidenceError("truth adjudication is invalid")
        indexed[case_id] = dict(row)
    _same("truth/capture case set", set(indexed), set(expected_case_ids))
    return {
        case_id: indexed[case_id] for case_id in sorted(indexed)
    }, actual_sha


def _bound_samples(
    *,
    capture: _ValidatedCapture,
    truth: Mapping[str, Mapping[str, Any]],
    truth_sha256: str,
    source_hashes: Mapping[str, str],
) -> tuple[
    Mapping[str, Any],
    bytes,
    tuple[GroupedCalibrationExample, ...],
]:
    groups = {
        case_id: group_id
        for group_id, case_ids in capture.layout.groups.items()
        for case_id in case_ids
    }
    samples = []
    for row, context in zip(capture.rows, capture.contexts, strict=True):
        samples.append(
            {
                "sample_key": row.case_id,
                "group_key": groups[row.case_id],
                "input_confidence": row.confidence,
                "correct": (
                    row.adjudication
                    == str(truth[row.case_id]["adjudication"]).strip()
                ),
                "context": context.to_dict(),
            }
        )
    bindings = {
        "source_revision_sha": capture.bindings["source_revision_sha"],
        "capture_sha256": capture.capture_sha256,
        "full_output_sha256": _sha256_bytes(capture.prediction_bytes),
        "non_confidence_output_sha256": _sha256_bytes(
            capture.non_confidence_bytes
        ),
        "truth_sha256": truth_sha256,
        "layout_manifest_sha256": capture.layout.sha256,
        "input_tree_sha256": capture.bindings["input_tree_sha256"],
        "context_contract_sha256": (
            capture.bindings["context_contract_sha256"]
        ),
        "runtime_confidence_artifact_sha256": (
            capture.bindings["runtime_confidence_artifact_sha256"]
        ),
        "runtime_confidence_artifact_file_sha256": (
            capture.bindings["runtime_confidence_artifact_file_sha256"]
        ),
        "producer_graph_sha256": capture.bindings["producer_graph_sha256"],
        "producer_source_sha256": capture.bindings["producer_source_sha256"],
        "cv_source_sha256": source_hashes["cv_source_sha256"],
        "evaluator_source_sha256": source_hashes[
            "evaluator_source_sha256"
        ],
        "evidence_builder_source_sha256": source_hashes[
            "evidence_builder_source_sha256"
        ],
    }
    value = {
        "schema_version": BOUND_SAMPLE_SCHEMA,
        "frozen_capture_before_truth_join": True,
        "bindings": bindings,
        "sample_count": EXPECTED_RECORD_COUNT,
        "cv_payload": {
            "schema_version": SAMPLE_SCHEMA,
            "samples": samples,
        },
    }
    encoded = _canonical_bytes(value)
    examples = validate_bound_samples(
        value,
        expected_bindings=bindings,
    )
    return value, encoded, examples


def validate_bound_samples(
    value: Mapping[str, Any],
    *,
    expected_bindings: Mapping[str, Any],
) -> tuple[GroupedCalibrationExample, ...]:
    """Validate the private joined schema and feed its exact CV payload loader."""

    if set(value) != {
        "schema_version",
        "frozen_capture_before_truth_join",
        "bindings",
        "sample_count",
        "cv_payload",
    }:
        raise WO19EvidenceError("bound-sample top-level fields are not exact")
    if (
        value.get("schema_version") != BOUND_SAMPLE_SCHEMA
        or value.get("frozen_capture_before_truth_join") is not True
        or value.get("sample_count") != EXPECTED_RECORD_COUNT
    ):
        raise WO19EvidenceError("bound-sample freeze contract is invalid")
    bindings = value.get("bindings")
    if (
        not isinstance(bindings, Mapping)
        or set(bindings) != _BOUND_BINDING_KEYS
        or dict(bindings) != dict(expected_bindings)
    ):
        raise WO19EvidenceError("bound-sample bindings are not exact")
    _require_revision(bindings["source_revision_sha"])
    for key in _BOUND_BINDING_KEYS - {"source_revision_sha"}:
        _require_digest(bindings[key], label=f"sample {key}")
    payload = value.get("cv_payload")
    if not isinstance(payload, Mapping):
        raise WO19EvidenceError("bound CV payload is malformed")
    with tempfile.TemporaryDirectory(prefix="mib-wo19-cv-loader-") as directory:
        path = Path(directory) / "samples.json"
        path.write_bytes(_canonical_bytes(payload))
        try:
            examples = load_grouped_examples(path)
        except (TypeError, ValueError) as exc:
            raise WO19EvidenceError("bound CV samples are invalid") from exc
    if len(examples) != EXPECTED_RECORD_COUNT:
        raise WO19EvidenceError("bound CV sample count is not exactly 32")
    return examples


def _fold_membership_digest(
    examples: Sequence[GroupedCalibrationExample],
) -> str:
    ordered = tuple(sorted(examples, key=lambda item: item.sample_key))
    private_membership = []
    for repeat_index, seed in enumerate(REPEAT_SEEDS):
        folds = grouped_folds(ordered, seed=seed)
        for fold_index, indices in enumerate(folds):
            members = [ordered[index] for index in indices]
            test_groups = sorted({member.group_key for member in members})
            train_groups = {
                item.group_key
                for index, item in enumerate(ordered)
                if index not in set(indices)
            }
            if set(test_groups) & train_groups:
                raise WO19EvidenceError("grouped CV membership leaked a group")
            private_membership.append(
                {
                    "repeat": repeat_index,
                    "fold": fold_index,
                    "sample_keys": sorted(
                        member.sample_key for member in members
                    ),
                    "group_keys": test_groups,
                }
            )
    if len(private_membership) != len(REPEAT_SEEDS) * FOLD_COUNT:
        raise WO19EvidenceError("CV membership is not exact 3x5")
    return _sha256_bytes(_canonical_bytes(private_membership))


def _validate_refit_artifacts(
    artifacts: Mapping[str, FittedConfidenceCalibrator],
    *,
    forbidden_values: Sequence[str],
) -> dict[str, bytes]:
    if set(artifacts) != set(CALIBRATION_FAMILIES):
        raise WO19EvidenceError("refit did not produce all four families")
    encoded: dict[str, bytes] = {}
    for family in CALIBRATION_FAMILIES:
        calibrator = artifacts[family]
        if not isinstance(calibrator, FittedConfidenceCalibrator):
            raise WO19EvidenceError("refit artifact has the wrong runtime type")
        mapping = calibrator.to_runtime_mapping()
        try:
            reconstructed = FittedConfidenceCalibrator.from_runtime_mapping(
                mapping
            )
        except (TypeError, ValueError) as exc:
            raise WO19EvidenceError(
                "refit artifact violates the runtime contract"
            ) from exc
        if reconstructed.to_runtime_mapping() != mapping:
            raise WO19EvidenceError("refit artifact is not canonical")
        raw = _artifact_bytes(calibrator)
        text = raw.decode("utf-8")
        if (
            _CASE_ID_TEXT_RE.search(text)
            or _ABSOLUTE_PATH_RE.search(text)
            or any(value and value in text for value in forbidden_values)
            or any(
                forbidden in mapping
                for forbidden in (
                    "sample_key",
                    "group_key",
                    "correct",
                    "label",
                    "labels",
                    "prediction",
                    "predictions",
                    "context",
                )
            )
        ):
            raise WO19EvidenceError(
                "refit artifact contains identity or training-only data"
            )
        encoded[family] = raw
    return encoded


def _repeat_refit(
    examples: Sequence[GroupedCalibrationExample],
    *,
    comparison_runner: Callable[
        [Sequence[GroupedCalibrationExample]],
        tuple[dict[str, Any], dict[str, FittedConfidenceCalibrator]],
    ],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, FittedConfidenceCalibrator],
    Mapping[str, bytes],
]:
    first_report, first_artifacts = comparison_runner(examples)
    second_report, second_artifacts = comparison_runner(examples)
    if first_report.get("schema_version") != REPORT_SCHEMA:
        raise WO19EvidenceError("CV report schema is unsupported")
    if _canonical_bytes(first_report) != _canonical_bytes(second_report):
        raise WO19EvidenceError("grouped OOF report is not deterministic")
    forbidden = [
        item.sample_key for item in examples
    ] + [
        item.group_key for item in examples
    ]
    first_bytes = _validate_refit_artifacts(
        first_artifacts,
        forbidden_values=forbidden,
    )
    second_bytes = _validate_refit_artifacts(
        second_artifacts,
        forbidden_values=forbidden,
    )
    if first_bytes != second_bytes:
        raise WO19EvidenceError("full-data refit artifacts are not deterministic")
    return first_report, first_artifacts, first_bytes


def _shadow_apply(
    *,
    capture: _ValidatedCapture,
    calibrator: FittedConfidenceCalibrator,
) -> tuple[tuple[PredictionRow, ...], bytes, bytes]:
    shadow_rows = []
    for baseline, context in zip(
        capture.rows,
        capture.contexts,
        strict=True,
    ):
        confidence = calibrator.predict(baseline.confidence, context)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise WO19EvidenceError(
                "selected calibrator returned an invalid confidence"
            )
        shadow = dataclasses.replace(
            baseline,
            confidence=float(confidence),
        )
        assert_non_confidence_unchanged(baseline, shadow)
        shadow_rows.append(shadow)
    shadow_payload = {
        "rows": [
            {
                "case_id": row.case_id,
                "layout_group": raw["layout_group"],
                "prediction": row.to_dict(),
                "context": context.to_dict(),
            }
            for row, raw, context in zip(
                shadow_rows,
                capture.payload["rows"],
                capture.contexts,
                strict=True,
            )
        ]
    }
    shadow_full = prediction_jsonl(shadow_payload)
    shadow_non_confidence = non_confidence_jsonl(shadow_payload)
    if (
        shadow_non_confidence != capture.non_confidence_bytes
        or _sha256_bytes(shadow_non_confidence)
        != capture.payload["non_confidence_jsonl_sha256"]
    ):
        raise WO19EvidenceError(
            "shadow canonical non-confidence bytes changed"
        )
    return tuple(shadow_rows), shadow_full, shadow_non_confidence


def _official_aggregate(
    truth: Mapping[str, Mapping[str, Any]],
    rows: Sequence[PredictionRow],
) -> Mapping[str, Any]:
    results, _ = official_evaluate.build_results(
        dict(truth),
        [row.to_dict() for row in rows],
    )
    counts = results["counts"]
    if (
        counts["truth_cases"] != EXPECTED_RECORD_COUNT
        or counts["submitted_records"] != EXPECTED_RECORD_COUNT
        or counts["scored_predictions"] != EXPECTED_RECORD_COUNT
        or any(
            counts[name]
            for name in (
                "missing_cases",
                "extra_cases",
                "duplicate_case_ids",
                "blank_case_rows",
                "invalid_adjudication_records",
                "invalid_confidence_records",
                "invalid_fee_status_records",
            )
        )
    ):
        raise WO19EvidenceError(
            "official evaluator did not receive a complete valid 32-case run"
        )
    return {
        "record_count": EXPECTED_RECORD_COUNT,
        "total_score": results["scores"]["total_score"],
        "extraction_score": results["scores"]["extraction_score"],
        "classification_score": results["scores"][
            "classification_score"
        ],
        "calibration_score": results["scores"]["calibration_score"],
        "mean_brier": results["raw"]["mean_confidence_brier"],
        "catastrophic_false_approvals": results["raw"][
            "catastrophic_false_approvals"
        ],
        "missing_record_count": 0,
        "invalid_record_count": 0,
        "duplicate_record_count": 0,
        "extra_record_count": 0,
    }


def _cv_aggregate_summary(report: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = report["candidates"]
    return {
        family: {
            "mean_oof_brier": candidates[family]["aggregate"]["overall"][
                "brier"
            ],
            "mean_oof_calibration_score": candidates[family]["aggregate"][
                "overall"
            ]["calibration_score"],
            "ece_10_bin": candidates[family]["aggregate"]["overall"][
                "ece_10_bin"
            ],
            "repeat_briers": [
                repeat["metrics"]["overall"]["brier"]
                for repeat in candidates[family]["repeats"]
            ],
            "fold_count": len(candidates[family]["folds"]),
            "leave_one_group_out_non_regression": candidates[family][
                "leave_one_group_out"
            ]["non_regression"],
        }
        for family in CALIBRATION_FAMILIES
    }


def _assert_repository_safe(
    evidence: Mapping[str, Any],
    *,
    private_values: Sequence[str],
) -> None:
    text = canonical_json(evidence)
    if _CASE_ID_TEXT_RE.search(text) or _ABSOLUTE_PATH_RE.search(text):
        raise WO19EvidenceError(
            "repository evidence contains identity or filesystem data"
        )
    for value in private_values:
        if value and value in text:
            raise WO19EvidenceError(
                "repository evidence contains private sample/group data"
            )
    banned_keys = {
        "case_id",
        "case_ids",
        "sample_key",
        "sample_keys",
        "group_key",
        "group_keys",
        "rows",
        "samples",
        "contexts",
        "labels",
        "predictions",
        "paths",
    }

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if set(value) & banned_keys:
                raise WO19EvidenceError(
                    "repository evidence contains identity-bearing fields"
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(evidence)


def render_markdown(evidence: Mapping[str, Any]) -> str:
    """Render the already-sanitized aggregate evidence."""

    decision = evidence["decision"]
    baseline = evidence["official_evaluator"]["baseline"]
    shadow = evidence["official_evaluator"]["shadow"]
    selection = evidence["selection"]
    checks = evidence["checks"]
    lines = [
        "# WO19 final-confidence refit evidence",
        "",
        f"- Status: `{evidence['status']}`",
        f"- Evidence scope: `{evidence['comparison_scope']}`",
        f"- S0 revision: `{evidence['source_revision_sha']}`",
        (
            "- Records / layout groups: "
            f"{evidence['counts']['record_count']} / "
            f"{evidence['counts']['layout_group_count']}"
        ),
        (
            "- Selected shadow family: "
            f"`{selection['selected_family']}`"
        ),
        (
            "- Promotion recommendation: "
            f"`{str(selection['promotion_recommended']).lower()}`"
        ),
        (
            "- Runtime action: "
            f"`{decision['runtime_artifact_action']}`"
        ),
        "",
        "## Exact runtime and byte gates",
        "",
        (
            "- Double truth-blind capture deterministic: "
            f"`{str(checks['capture_byte_deterministic']).lower()}`"
        ),
        (
            "- Normal solution.py/BatchRunner equals capture bytes: "
            f"`{str(checks['normal_batch_equals_capture']).lower()}`"
        ),
        (
            "- All non-confidence bytes unchanged in shadow: "
            f"`{str(checks['non_confidence_bytes_unchanged']).lower()}`"
        ),
        (
            "- Refit and artifact bytes deterministic: "
            f"`{str(checks['refit_byte_deterministic']).lower()}`"
        ),
        "",
        "## Official evaluator",
        "",
        "| Arm | Total | Calibration | Mean Brier |",
        "|---|---:|---:|---:|",
        (
            f"| S0 baseline | {baseline['total_score']:.12f} | "
            f"{baseline['calibration_score']:.12f} | "
            f"{baseline['mean_brier']:.12f} |"
        ),
        (
            f"| Selected shadow | {shadow['total_score']:.12f} | "
            f"{shadow['calibration_score']:.12f} | "
            f"{shadow['mean_brier']:.12f} |"
        ),
        "",
        "## Decision",
        "",
        decision["statement"],
        "",
        (
            "The Brier target of 0.01 is tracked and reported, but it is not "
            "forced as a promotion gate. This is public-label-exposed grouped "
            "robustness evidence, not an unseen holdout result."
        ),
        "",
    ]
    return "\n".join(lines)


def build_wo19_evidence(
    *,
    capture_path: Path,
    rerun_capture_path: Path,
    capture_predictions_path: Path,
    rerun_capture_predictions_path: Path,
    observation_path: Path,
    layout_manifest_path: Path,
    input_dir: Path,
    truth_path: Path,
    expected_truth_sha256: str,
    source_revision_sha: str,
    verify_repository: bool = True,
    solution_runner: Callable[[Path, Path], int] | None = None,
    comparison_runner: Callable[
        [Sequence[GroupedCalibrationExample]],
        tuple[dict[str, Any], dict[str, FittedConfidenceCalibrator]],
    ] = compare_confidence_families,
) -> WO19EvidenceResult:
    """Validate all external inputs and return aggregate-only WO19 evidence."""

    revision = _require_revision(source_revision_sha)
    capture = _validate_capture_bundle(
        capture_path=capture_path,
        rerun_capture_path=rerun_capture_path,
        capture_predictions_path=capture_predictions_path,
        rerun_capture_predictions_path=rerun_capture_predictions_path,
        observation_path=observation_path,
        layout_manifest_path=layout_manifest_path,
        input_dir=input_dir,
        source_revision_sha=revision,
    )
    source_hashes = _validate_source_and_runtime_bindings(
        capture,
        source_revision_sha=revision,
        verify_repository=verify_repository,
    )
    normal_batch_sha = _run_normal_solution_equivalence(
        input_dir=input_dir,
        expected_prediction_bytes=capture.prediction_bytes,
        runner=solution_runner,
    )
    truth, truth_sha = _load_truth(
        path=truth_path,
        expected_sha256=expected_truth_sha256,
        expected_case_ids=[row.case_id for row in capture.rows],
    )
    bound_value, bound_bytes, examples = _bound_samples(
        capture=capture,
        truth=truth,
        truth_sha256=truth_sha,
        source_hashes=source_hashes,
    )
    fold_membership_sha = _fold_membership_digest(examples)
    report, artifacts, artifact_bytes = _repeat_refit(
        examples,
        comparison_runner=comparison_runner,
    )
    selected_family = report["selection"]["family"]
    if selected_family not in CALIBRATION_FAMILIES:
        raise WO19EvidenceError("CV selected an unsupported family")
    selected = artifacts[selected_family]
    shadow_rows, shadow_full, shadow_non_confidence = _shadow_apply(
        capture=capture,
        calibrator=selected,
    )
    baseline_score = _official_aggregate(truth, capture.rows)
    shadow_score = _official_aggregate(truth, shadow_rows)
    cv_promotion_recommended = report["selection"]["promotion_recommended"]
    if not isinstance(cv_promotion_recommended, bool):
        raise WO19EvidenceError("CV promotion decision is not boolean")
    realized_shadow_checks = {
        "mean_brier_strictly_improved": (
            shadow_score["mean_brier"]
            < baseline_score["mean_brier"] - PROMOTION_TOLERANCE
        ),
        "calibration_score_strictly_improved": (
            shadow_score["calibration_score"]
            > baseline_score["calibration_score"] + PROMOTION_TOLERANCE
        ),
        "total_score_strictly_improved": (
            shadow_score["total_score"]
            > baseline_score["total_score"] + PROMOTION_TOLERANCE
        ),
        "non_confidence_bytes_unchanged": (
            shadow_non_confidence == capture.non_confidence_bytes
        ),
    }
    promotion_recommended = (
        cv_promotion_recommended
        and all(realized_shadow_checks.values())
    )
    needs_s1 = promotion_recommended
    if needs_s1:
        status = "needs_s1"
        artifact_action = (
            "requires_separate_s1_runtime_integration_and_fresh_double_capture"
        )
        statement = (
            "The selected calibrator passed both the grouped-CV gates and "
            "the realized official-evaluator shadow gates. It is not deployed "
            "or certified here; create a separate S1 runtime integration "
            "(code and artifact as needed) and repeat the complete production "
            "and byte gates."
        )
    else:
        status = "evaluated_no_promotion"
        artifact_action = "retained_current_s0_artifact"
        statement = (
            "At least one promotion gate did not pass. The current S0 "
            "confidence artifact remains pinned and the shadow candidate "
            "remains outside the runtime composition."
        )
    selection = report["selection"]
    evidence = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": status,
        "comparison_scope": "public_grouped_robustness_not_unseen",
        "source_revision_sha": revision,
        "counts": {
            "record_count": EXPECTED_RECORD_COUNT,
            "layout_group_count": len(capture.layout.groups),
            "repeat_count": len(REPEAT_SEEDS),
            "fold_count": len(REPEAT_SEEDS) * FOLD_COUNT,
            "family_count": len(CALIBRATION_FAMILIES),
            "missing_record_count": 0,
            "duplicate_record_count": 0,
            "invalid_record_count": 0,
        },
        "bindings": {
            "capture_sha256": capture.capture_sha256,
            "normal_batch_output_sha256": normal_batch_sha,
            "full_output_sha256": _sha256_bytes(capture.prediction_bytes),
            "non_confidence_output_sha256": _sha256_bytes(
                capture.non_confidence_bytes
            ),
            "shadow_full_output_sha256": _sha256_bytes(shadow_full),
            "shadow_non_confidence_output_sha256": _sha256_bytes(
                shadow_non_confidence
            ),
            "truth_sha256": truth_sha,
            "layout_manifest_sha256": capture.layout.sha256,
            "input_tree_sha256": capture.bindings["input_tree_sha256"],
            "producer_graph_sha256": capture.bindings[
                "producer_graph_sha256"
            ],
            "producer_source_sha256": capture.bindings[
                "producer_source_sha256"
            ],
            "context_contract_sha256": capture.bindings[
                "context_contract_sha256"
            ],
            "context_contract_source_sha256": capture.bindings[
                "context_contract_source_sha256"
            ],
            "current_runtime_artifact_sha256": capture.bindings[
                "runtime_confidence_artifact_sha256"
            ],
            "current_runtime_artifact_file_sha256": capture.bindings[
                "runtime_confidence_artifact_file_sha256"
            ],
            "cv_source_sha256": source_hashes["cv_source_sha256"],
            "evaluator_source_sha256": source_hashes[
                "evaluator_source_sha256"
            ],
            "evidence_builder_source_sha256": source_hashes[
                "evidence_builder_source_sha256"
            ],
            "bound_sample_set_sha256": _sha256_bytes(bound_bytes),
            "fold_membership_sha256": fold_membership_sha,
            "cv_report_sha256": _sha256_bytes(_canonical_bytes(report)),
            "selected_shadow_artifact_sha256": _sha256_bytes(
                artifact_bytes[selected_family]
            ),
        },
        "checks": {
            "capture_frozen_before_truth_join": True,
            "capture_byte_deterministic": True,
            "capture_bindings_verified": True,
            "source_blob_bindings_verified": verify_repository,
            "input_layout_artifact_context_bindings_verified": True,
            "normal_batch_generated_by_tool": True,
            "normal_batch_worker_limit": 4,
            "normal_batch_equals_capture": True,
            "truth_join_after_frozen_capture": True,
            "truth_schema_and_hash_verified": True,
            "exact_case_set_verified": True,
            "grouped_oof_exact_3x5": True,
            "group_exclusive": True,
            "refit_byte_deterministic": True,
            "runtime_artifacts_identity_free": True,
            "shadow_downstream_confidence_only": True,
            "non_confidence_bytes_unchanged": True,
            "official_evaluator_verified": True,
            "candidate_runtime_composed": False,
            "fresh_s1_capture_verified": False,
        },
        "official_evaluator": {
            "score_version": official_evaluate.SCORE_VERSION,
            "baseline": baseline_score,
            "shadow": shadow_score,
            "delta": {
                "total_score": (
                    shadow_score["total_score"]
                    - baseline_score["total_score"]
                ),
                "calibration_score": (
                    shadow_score["calibration_score"]
                    - baseline_score["calibration_score"]
                ),
                "mean_brier": (
                    shadow_score["mean_brier"]
                    - baseline_score["mean_brier"]
                ),
            },
        },
        "cv_family_metrics": _cv_aggregate_summary(report),
        "acceptance_diagnostics": {
            "baseline": report["baseline"]["aggregate"],
            "selected": report["candidates"][selected_family]["aggregate"],
            "dimensions": [
                "final_class",
                "policy_route",
                "recovery_route",
            ],
            "reliability_bin_count": 10,
            "includes_ece": True,
            "includes_correct_incorrect_distributions": True,
        },
        "selection": {
            "selected_family": selected_family,
            "cv_promotion_checks": dict(selection["promotion_checks"]),
            "cv_promotion_recommended": cv_promotion_recommended,
            "realized_shadow_checks": realized_shadow_checks,
            "promotion_recommended": promotion_recommended,
            "baseline_oof_brier": selection["baseline_brier"],
            "selected_oof_brier": selection["selected_brier"],
            "oof_brier_delta": selection["brier_delta"],
            "repeat_brier_deltas": list(
                selection["repeat_brier_deltas"]
            ),
            "tracked_brier_target": dict(
                selection["tracked_brier_target"]
            ),
        },
        "decision": {
            "candidate_runtime_composed": False,
            "runtime_artifact_action": artifact_action,
            "requires_s1": needs_s1,
            "deployment_certified": False,
            "statement": statement,
        },
        "privacy": {
            "repository_artifact": "aggregate_only",
            "identity_bearing_capture_storage": "external",
            "identity_bearing_bound_samples_storage": "external",
            "raw_contexts_serialized_to_repository": False,
            "labels_serialized_to_repository": False,
            "prediction_rows_serialized_to_repository": False,
            "layout_memberships_serialized_to_repository": False,
        },
    }
    final_capture = _validate_capture_bundle(
        capture_path=capture_path,
        rerun_capture_path=rerun_capture_path,
        capture_predictions_path=capture_predictions_path,
        rerun_capture_predictions_path=rerun_capture_predictions_path,
        observation_path=observation_path,
        layout_manifest_path=layout_manifest_path,
        input_dir=input_dir,
        source_revision_sha=revision,
    )
    final_source_hashes = _validate_source_and_runtime_bindings(
        final_capture,
        source_revision_sha=revision,
        verify_repository=verify_repository,
    )
    _same("post-run capture", final_capture.capture_sha256, capture.capture_sha256)
    _same("post-run source bindings", final_source_hashes, source_hashes)
    private_values = [
        row.case_id for row in capture.rows
    ] + [
        group_id for group_id in capture.layout.groups
    ]
    _assert_repository_safe(evidence, private_values=private_values)
    markdown = render_markdown(evidence)
    if any(value and value in markdown for value in private_values):
        raise WO19EvidenceError(
            "repository Markdown contains private sample/group data"
        )
    return WO19EvidenceResult(
        evidence=evidence,
        markdown=markdown,
        bound_samples_bytes=bound_bytes,
        selected_artifact_bytes=artifact_bytes[selected_family],
        needs_s1=needs_s1,
    )


def _require_external_path(path: Path) -> None:
    resolved = path.expanduser().resolve(strict=False)
    repository = REPO_ROOT.resolve()
    if resolved == repository or repository in resolved.parents:
        raise WO19EvidenceError(
            "identity-bearing/CV artifacts must remain outside the repository"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--rerun-capture", type=Path, required=True)
    parser.add_argument("--capture-predictions", type=Path, required=True)
    parser.add_argument(
        "--rerun-capture-predictions",
        type=Path,
        required=True,
    )
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--expected-truth-sha256", required=True)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--external-bound-samples", type=Path)
    parser.add_argument("--external-selected-artifact", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.external_bound_samples is not None:
        _require_external_path(arguments.external_bound_samples)
    if arguments.external_selected_artifact is not None:
        _require_external_path(arguments.external_selected_artifact)
    result = build_wo19_evidence(
        capture_path=arguments.capture,
        rerun_capture_path=arguments.rerun_capture,
        capture_predictions_path=arguments.capture_predictions,
        rerun_capture_predictions_path=(
            arguments.rerun_capture_predictions
        ),
        observation_path=arguments.observation,
        layout_manifest_path=arguments.layout_manifest,
        input_dir=arguments.input_dir,
        truth_path=arguments.truth,
        expected_truth_sha256=arguments.expected_truth_sha256,
        source_revision_sha=arguments.source_revision_sha,
    )
    arguments.output_json.write_bytes(_canonical_bytes(result.evidence))
    arguments.output_markdown.write_text(
        result.markdown,
        encoding="utf-8",
    )
    if arguments.external_bound_samples is not None:
        arguments.external_bound_samples.write_bytes(
            result.bound_samples_bytes
        )
    if result.needs_s1 and arguments.external_selected_artifact is not None:
        arguments.external_selected_artifact.write_bytes(
            result.selected_artifact_bytes
        )
    print(
        canonical_json(
            {
                "status": result.evidence["status"],
                "needs_s1": result.needs_s1,
                "selected_family": result.evidence["selection"][
                    "selected_family"
                ],
            }
        )
    )
    return 3 if result.needs_s1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
