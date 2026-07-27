#!/usr/bin/env python3
"""Truth-blind WO19 capture of final rows and confidence contexts.

The executable accepts no truth, labels, correctness flags, or prediction
overrides.  It calls the exact production composition root, obtains the
accepted :class:`FinalPredictionWithConfidenceContext` from the inner
processor, and then applies the current outer confidence recalibrator.  All
identity-bearing outputs must be written outside the repository.  Only the
aggregate observation is suitable for later evidence packaging.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.experiment_control import canonical_json  # noqa: E402
from devtools.grouped_recovery_evidence import (  # noqa: E402
    FrozenLayoutManifest,
    load_layout_manifest,
)
from devtools.ocr_ablation import _input_tree_sha256  # noqa: E402
from devtools.wo18_production_capture import (  # noqa: E402
    producer_graph_sha256,
)
from mib_pipeline import final_confidence as final_confidence_module  # noqa: E402
from mib_pipeline.final_confidence import (  # noqa: E402
    FINAL_CLASSES,
    FINAL_CONFIDENCE_CONTEXT_SCHEMA_VERSION,
    FINAL_POLICY_ROUTES,
    FINAL_RECOVERY_ROUTES,
    FinalConfidenceContext,
    FinalPredictionWithConfidenceContext,
)
from mib_pipeline.models import (  # noqa: E402
    CASE_ID_PATTERN,
    FIELD_NAMES,
    PredictionRow,
    RowValidationError,
)
from mib_pipeline.output_confidence import (  # noqa: E402
    PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH,
    PINNED_OUTPUT_CONFIDENCE_ARTIFACT_SHA256,
    OutputConfidenceRecalibrationProcessor,
)
from mib_pipeline.production import build_production_processor  # noqa: E402


CAPTURE_SCHEMA = "mib-final-confidence-production-capture/v1"
OBSERVATION_SCHEMA = "mib-final-confidence-capture-observation/v1"
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CONTEXT_FIELDS = tuple(
    field.name for field in dataclasses.fields(FinalConfidenceContext)
)
_NON_CONFIDENCE_FIELDS = tuple(
    field_name for field_name in FIELD_NAMES if field_name != "confidence"
)


class FinalConfidenceCaptureError(RuntimeError):
    """The frozen production graph did not yield an exact valid capture."""


@dataclass(frozen=True)
class CapturedFinalPrediction:
    """One accepted inner result and its current serialized production row."""

    accepted: FinalPredictionWithConfidenceContext
    final_row: PredictionRow

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, FinalPredictionWithConfidenceContext):
            raise TypeError("accepted result must include final confidence context")
        if not isinstance(self.final_row, PredictionRow):
            raise TypeError("final_row must be PredictionRow")
        _validate_prediction(self.accepted.row)
        _validate_prediction(self.final_row)
        _validate_context(self.accepted.context)
        if self.accepted.context.final_class != self.final_row.adjudication:
            raise FinalConfidenceCaptureError(
                "outer confidence stage changed the accepted decision"
            )
        if _non_confidence_projection(
            self.accepted.row
        ) != _non_confidence_projection(self.final_row):
            raise FinalConfidenceCaptureError(
                "outer confidence stage changed a non-confidence field"
            )


@dataclass(frozen=True)
class CaptureBindings:
    """Immutable source and runtime bindings copied into both capture runs."""

    source_revision_sha: str
    layout_manifest_sha256: str
    input_tree_sha256: str
    producer_graph_sha256: str
    producer_source_sha256: str
    runtime_confidence_artifact_sha256: str
    runtime_confidence_artifact_file_sha256: str
    runtime_confidence_artifact_id: str
    context_contract_sha256: str
    context_contract_source_sha256: str

    def __post_init__(self) -> None:
        if not _GIT_COMMIT_RE.fullmatch(self.source_revision_sha):
            raise FinalConfidenceCaptureError(
                "source revision must be a full lowercase Git commit SHA"
            )
        for name in (
            "layout_manifest_sha256",
            "input_tree_sha256",
            "producer_graph_sha256",
            "producer_source_sha256",
            "runtime_confidence_artifact_sha256",
            "runtime_confidence_artifact_file_sha256",
            "context_contract_sha256",
            "context_contract_source_sha256",
        ):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise FinalConfidenceCaptureError(
                    f"{name} must be a full lowercase SHA-256 digest"
                )
        if (
            not isinstance(self.runtime_confidence_artifact_id, str)
            or not self.runtime_confidence_artifact_id.strip()
        ):
            raise FinalConfidenceCaptureError(
                "runtime confidence artifact ID must be non-empty"
            )

    def to_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


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


def _production_prediction_jsonl(rows: Sequence[PredictionRow]) -> bytes:
    """Mirror ``CanonicalJsonlWriter`` without touching the filesystem."""

    case_ids = [row.case_id for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise FinalConfidenceCaptureError(
            "full prediction serialization contains duplicate case identity"
        )
    ordered = sorted(rows, key=lambda row: row.case_id)
    return "".join(
        json.dumps(
            row.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in ordered
    ).encode("utf-8")


def _non_confidence_projection_jsonl(
    rows: Sequence[PredictionRow],
) -> bytes:
    """Serialize FIELD_NAMES-minus-confidence in production field order."""

    ordered = sorted(rows, key=lambda row: row.case_id)
    return "".join(
        json.dumps(
            _non_confidence_projection(row),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in ordered
    ).encode("utf-8")


def _non_confidence_projection(row: PredictionRow) -> dict[str, object]:
    values = row.to_dict()
    return {
        field_name: values[field_name]
        for field_name in _NON_CONFIDENCE_FIELDS
    }


def _validate_prediction(row: PredictionRow) -> None:
    if not isinstance(row, PredictionRow):
        raise FinalConfidenceCaptureError(
            "production graph returned a non-PredictionRow value"
        )
    values = row.to_dict()
    if tuple(values) != FIELD_NAMES or not CASE_ID_PATTERN.fullmatch(row.case_id):
        raise FinalConfidenceCaptureError(
            "production prediction violates the exact output schema"
        )
    if (
        isinstance(row.confidence, bool)
        or not isinstance(row.confidence, (int, float))
        or not math.isfinite(float(row.confidence))
        or not 0.0 <= float(row.confidence) <= 1.0
    ):
        raise FinalConfidenceCaptureError(
            "production confidence is outside [0, 1]"
        )
    try:
        normalized = PredictionRow.from_mapping(values)
    except RowValidationError as exc:
        raise FinalConfidenceCaptureError(
            "production prediction is invalid"
        ) from exc
    if normalized != row:
        raise FinalConfidenceCaptureError(
            "production prediction is not canonically schema-valid"
        )


def _validate_context(context: FinalConfidenceContext) -> None:
    if not isinstance(context, FinalConfidenceContext):
        raise FinalConfidenceCaptureError(
            "inner processor returned no typed confidence context"
        )
    values = context.to_dict()
    if tuple(values) != _CONTEXT_FIELDS:
        raise FinalConfidenceCaptureError(
            "final-confidence context fields are not exact"
        )
    try:
        reconstructed = FinalConfidenceContext(**values)
    except (TypeError, ValueError) as exc:
        raise FinalConfidenceCaptureError(
            "final-confidence context is invalid"
        ) from exc
    if reconstructed != context:
        raise FinalConfidenceCaptureError(
            "final-confidence context is not canonical"
        )


def context_contract_mapping() -> dict[str, object]:
    """Return the fixed identity-free context contract, without case data."""

    return {
        "schema_version": FINAL_CONFIDENCE_CONTEXT_SCHEMA_VERSION,
        "fields": list(_CONTEXT_FIELDS),
        "final_classes": list(FINAL_CLASSES),
        "policy_routes": list(FINAL_POLICY_ROUTES),
        "recovery_routes": list(FINAL_RECOVERY_ROUTES),
        "numeric_range": [0.0, 1.0],
        "model_diagnostics_pair_required": True,
        "free_form_values_allowed": False,
    }


def context_contract_sha256() -> str:
    return _sha256_bytes(_canonical_bytes(context_contract_mapping()))


def _runtime_confidence_binding() -> tuple[str, str, str]:
    try:
        value = json.loads(
            PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalConfidenceCaptureError(
            "cannot read the runtime confidence artifact"
        ) from exc
    if not isinstance(value, dict):
        raise FinalConfidenceCaptureError(
            "runtime confidence artifact must be an object"
        )
    canonical_sha = _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    if canonical_sha != PINNED_OUTPUT_CONFIDENCE_ARTIFACT_SHA256:
        raise FinalConfidenceCaptureError(
            "runtime confidence artifact does not match its pinned digest"
        )
    artifact_id = value.get("artifact_id")
    if not isinstance(artifact_id, str) or not artifact_id.strip():
        raise FinalConfidenceCaptureError(
            "runtime confidence artifact has no artifact ID"
        )
    return (
        canonical_sha,
        _sha256_file(PINNED_OUTPUT_CONFIDENCE_ARTIFACT_PATH),
        artifact_id,
    )


def _capture_with_outer(
    pdf_path: Path,
    outer: OutputConfidenceRecalibrationProcessor,
) -> CapturedFinalPrediction:
    if not isinstance(outer, OutputConfidenceRecalibrationProcessor):
        raise FinalConfidenceCaptureError(
            "production composition root has an unsupported outer stage"
        )
    inner_method = getattr(
        outer.processor,
        "process_case_with_confidence_context",
        None,
    )
    if not callable(inner_method):
        raise FinalConfidenceCaptureError(
            "production inner processor has no confidence-context API"
        )
    accepted = inner_method(pdf_path)
    if not isinstance(accepted, FinalPredictionWithConfidenceContext):
        raise FinalConfidenceCaptureError(
            "production inner processor returned an invalid contextual result"
        )
    final_row = outer.recalibrator.recalibrate(
        accepted.row,
        context=accepted.context,
    )
    return CapturedFinalPrediction(
        accepted=accepted,
        final_row=final_row,
    )


def capture_production_case(pdf_path: Path) -> CapturedFinalPrediction:
    """Capture one PDF through the exact submitted production composition."""

    return _capture_with_outer(pdf_path, build_production_processor())


def _layout_group_by_case(
    layout_manifest: FrozenLayoutManifest,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for group_id, case_ids in layout_manifest.groups.items():
        for case_id in case_ids:
            if case_id in result:
                raise FinalConfidenceCaptureError(
                    "layout manifest contains duplicate case identity"
                )
            result[case_id] = group_id
    return result


def build_capture_payload(
    *,
    input_dir: Path,
    layout_manifest: FrozenLayoutManifest,
    bindings: CaptureBindings,
    capture_case: Callable[[Path], CapturedFinalPrediction] = (
        capture_production_case
    ),
) -> Mapping[str, object]:
    """Capture the exact frozen cohort once, without consulting truth."""

    if bindings.layout_manifest_sha256 != layout_manifest.sha256:
        raise FinalConfidenceCaptureError(
            "layout manifest does not match the frozen binding"
        )
    actual_tree, pdf_count = _input_tree_sha256(input_dir)
    expected_cases = set(layout_manifest.case_ids)
    if (
        actual_tree != bindings.input_tree_sha256
        or pdf_count != len(expected_cases)
    ):
        raise FinalConfidenceCaptureError(
            "input tree does not match the frozen cohort binding"
        )
    group_by_case = _layout_group_by_case(layout_manifest)
    pdf_paths = tuple(
        sorted(
            (
                path
                for path in input_dir.iterdir()
                if path.is_file() and path.suffix.casefold() == ".pdf"
            ),
            key=lambda path: (path.name.casefold(), path.name),
        )
    )
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for pdf_path in pdf_paths:
        captured = capture_case(pdf_path)
        case_id = captured.final_row.case_id
        if case_id != pdf_path.stem:
            raise FinalConfidenceCaptureError(
                "production case identity does not match the frozen input"
            )
        if case_id in seen:
            raise FinalConfidenceCaptureError(
                "production capture returned duplicate case identity"
            )
        if case_id not in expected_cases:
            raise FinalConfidenceCaptureError(
                "production capture returned unexpected case identity"
            )
        seen.add(case_id)
        rows.append(
            {
                "case_id": case_id,
                "layout_group": group_by_case[case_id],
                "prediction": captured.final_row.to_dict(),
                "context": captured.accepted.context.to_dict(),
            }
        )
    missing = expected_cases - seen
    if missing or len(rows) != len(expected_cases):
        raise FinalConfidenceCaptureError(
            "production capture is missing frozen cohort cases"
        )
    ordered = sorted(rows, key=lambda row: str(row["case_id"]))
    predictions = [
        PredictionRow.from_mapping(row["prediction"])  # type: ignore[arg-type]
        for row in ordered
    ]
    return {
        "schema_version": CAPTURE_SCHEMA,
        "frozen_before_truth_join": True,
        "confidence_topology": {
            "captured_probability": "post_current_outer_recalibrator",
            "candidate_position": "downstream_confidence_only",
            "parallel_batch_runtime_equivalence": (
                "not_claimed_without_shadow_solution_proof"
            ),
        },
        "bindings": bindings.to_dict(),
        "record_count": len(ordered),
        "full_predictions_jsonl_sha256": _sha256_bytes(
            _production_prediction_jsonl(predictions)
        ),
        "non_confidence_jsonl_sha256": _sha256_bytes(
            _non_confidence_projection_jsonl(predictions)
        ),
        "rows": ordered,
    }


def _prediction_rows(
    payload: Mapping[str, object],
) -> tuple[PredictionRow, ...]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise FinalConfidenceCaptureError("capture rows are malformed")
    result: list[PredictionRow] = []
    seen: set[str] = set()
    for value in raw_rows:
        if not isinstance(value, dict) or set(value) != {
            "case_id",
            "layout_group",
            "prediction",
            "context",
        }:
            raise FinalConfidenceCaptureError("capture row is malformed")
        prediction = value.get("prediction")
        if (
            not isinstance(prediction, dict)
            or tuple(prediction) != FIELD_NAMES
        ):
            raise FinalConfidenceCaptureError(
                "captured prediction fields are not exact"
            )
        try:
            row = PredictionRow.from_mapping(prediction)
        except (RowValidationError, TypeError, ValueError) as exc:
            raise FinalConfidenceCaptureError(
                "captured prediction is invalid"
            ) from exc
        if row.to_dict() != prediction:
            raise FinalConfidenceCaptureError(
                "captured prediction is not canonical"
            )
        _validate_prediction(row)
        context_value = value.get("context")
        if not isinstance(context_value, dict):
            raise FinalConfidenceCaptureError(
                "captured confidence context is malformed"
            )
        try:
            context = FinalConfidenceContext(**context_value)
        except (TypeError, ValueError) as exc:
            raise FinalConfidenceCaptureError(
                "captured confidence context is invalid"
            ) from exc
        _validate_context(context)
        if context.final_class != row.adjudication:
            raise FinalConfidenceCaptureError(
                "captured context does not match the final prediction"
            )
        if value["case_id"] != row.case_id:
            raise FinalConfidenceCaptureError(
                "captured row identity does not match its prediction"
            )
        if row.case_id in seen:
            raise FinalConfidenceCaptureError(
                "captured rows contain duplicate case identity"
            )
        seen.add(row.case_id)
        result.append(row)
    return tuple(result)


def prediction_jsonl(payload: Mapping[str, object]) -> bytes:
    """Serialize with the exact ``CanonicalJsonlWriter`` byte contract."""

    return _production_prediction_jsonl(_prediction_rows(payload))


def non_confidence_jsonl(payload: Mapping[str, object]) -> bytes:
    """Serialize the canonical confidence-excluded projection."""

    return _non_confidence_projection_jsonl(_prediction_rows(payload))


def _verify_clean_revision(source_revision_sha: str) -> None:
    revision = source_revision_sha.strip().casefold()
    if not _GIT_COMMIT_RE.fullmatch(revision):
        raise FinalConfidenceCaptureError(
            "source_revision_sha must be a full Git commit SHA"
        )
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
        raise FinalConfidenceCaptureError(
            "cannot verify the production source revision"
        ) from exc
    if head != revision:
        raise FinalConfidenceCaptureError(
            "capture source revision does not match repository HEAD"
        )
    if status.strip():
        raise FinalConfidenceCaptureError(
            "production capture requires a clean committed checkout"
        )


def _require_external_outputs(paths: Sequence[Path]) -> None:
    repository = REPO_ROOT.resolve()
    resolved = [path.expanduser().resolve(strict=False) for path in paths]
    if len(set(resolved)) != len(resolved):
        raise FinalConfidenceCaptureError(
            "capture output paths must be distinct"
        )
    for path in resolved:
        if path == repository or repository in path.parents:
            raise FinalConfidenceCaptureError(
                "identity-bearing capture outputs must remain outside the repository"
            )


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _verify_frozen_inputs_unchanged(
    *,
    input_dir: Path,
    layout_manifest_path: Path,
    layout_manifest: FrozenLayoutManifest,
    expected_input_tree_sha256: str,
) -> None:
    final_tree, final_pdf_count = _input_tree_sha256(input_dir)
    if (
        final_tree != expected_input_tree_sha256
        or final_pdf_count != len(layout_manifest.case_ids)
    ):
        raise FinalConfidenceCaptureError(
            "frozen input tree changed during capture"
        )
    if _sha256_file(layout_manifest_path) != layout_manifest.sha256:
        raise FinalConfidenceCaptureError(
            "frozen layout manifest changed during capture"
        )


def _build_bindings(
    *,
    source_revision_sha: str,
    layout_manifest: FrozenLayoutManifest,
    input_tree_sha256: str,
) -> CaptureBindings:
    artifact_sha, artifact_file_sha, artifact_id = (
        _runtime_confidence_binding()
    )
    return CaptureBindings(
        source_revision_sha=source_revision_sha,
        layout_manifest_sha256=layout_manifest.sha256,
        input_tree_sha256=input_tree_sha256,
        producer_graph_sha256=producer_graph_sha256(),
        producer_source_sha256=_sha256_file(Path(__file__)),
        runtime_confidence_artifact_sha256=artifact_sha,
        runtime_confidence_artifact_file_sha256=artifact_file_sha,
        runtime_confidence_artifact_id=artifact_id,
        context_contract_sha256=context_contract_sha256(),
        context_contract_source_sha256=_sha256_file(
            Path(final_confidence_module.__file__)
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument("--expected-input-tree-sha256", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--rerun-capture", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--rerun-predictions", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    arguments = parser.parse_args(argv)

    output_paths = (
        arguments.capture,
        arguments.rerun_capture,
        arguments.predictions,
        arguments.rerun_predictions,
        arguments.observation,
    )
    _require_external_outputs(output_paths)
    revision = arguments.source_revision_sha.strip().casefold()
    expected_tree = (
        arguments.expected_input_tree_sha256.strip().casefold()
    )
    if not _SHA256_RE.fullmatch(expected_tree):
        raise FinalConfidenceCaptureError(
            "expected_input_tree_sha256 must be a full SHA-256 digest"
        )
    _verify_clean_revision(revision)
    manifest = load_layout_manifest(arguments.layout_manifest)
    bindings = _build_bindings(
        source_revision_sha=revision,
        layout_manifest=manifest,
        input_tree_sha256=expected_tree,
    )
    first = build_capture_payload(
        input_dir=arguments.input_dir,
        layout_manifest=manifest,
        bindings=bindings,
    )
    second = build_capture_payload(
        input_dir=arguments.input_dir,
        layout_manifest=manifest,
        bindings=bindings,
    )
    first_bytes = _canonical_bytes(first)
    second_bytes = _canonical_bytes(second)
    first_predictions = prediction_jsonl(first)
    second_predictions = prediction_jsonl(second)
    first_non_confidence = non_confidence_jsonl(first)
    second_non_confidence = non_confidence_jsonl(second)
    if (
        first_bytes != second_bytes
        or first_predictions != second_predictions
        or first_non_confidence != second_non_confidence
    ):
        raise FinalConfidenceCaptureError(
            "production confidence capture is not byte-deterministic"
        )
    if (
        _sha256_bytes(first_predictions)
        != first["full_predictions_jsonl_sha256"]
        or _sha256_bytes(first_non_confidence)
        != first["non_confidence_jsonl_sha256"]
    ):
        raise FinalConfidenceCaptureError(
            "capture serialization hashes are inconsistent"
        )

    # Re-check both the checkout and the graph after the two potentially long
    # production passes so a mid-capture source mutation cannot be certified.
    _verify_clean_revision(revision)
    if producer_graph_sha256() != bindings.producer_graph_sha256:
        raise FinalConfidenceCaptureError(
            "production graph changed during capture"
        )
    _verify_frozen_inputs_unchanged(
        input_dir=arguments.input_dir,
        layout_manifest_path=arguments.layout_manifest,
        layout_manifest=manifest,
        expected_input_tree_sha256=expected_tree,
    )

    _write_bytes(arguments.capture, first_bytes)
    _write_bytes(arguments.rerun_capture, second_bytes)
    _write_bytes(arguments.predictions, first_predictions)
    _write_bytes(arguments.rerun_predictions, second_predictions)
    observation = {
        "schema_version": OBSERVATION_SCHEMA,
        "source_revision_sha": revision,
        "layout_manifest_sha256": manifest.sha256,
        "input_tree_sha256": expected_tree,
        "producer_graph_sha256": bindings.producer_graph_sha256,
        "producer_source_sha256": bindings.producer_source_sha256,
        "runtime_confidence_artifact_sha256": (
            bindings.runtime_confidence_artifact_sha256
        ),
        "runtime_confidence_artifact_file_sha256": (
            bindings.runtime_confidence_artifact_file_sha256
        ),
        "context_contract_sha256": bindings.context_contract_sha256,
        "context_contract_source_sha256": (
            bindings.context_contract_source_sha256
        ),
        "confidence_topology": first["confidence_topology"],
        "capture_sha256": _sha256_bytes(first_bytes),
        "rerun_capture_sha256": _sha256_bytes(second_bytes),
        "full_predictions_jsonl_sha256": _sha256_bytes(
            first_predictions
        ),
        "rerun_full_predictions_jsonl_sha256": _sha256_bytes(
            second_predictions
        ),
        "non_confidence_jsonl_sha256": _sha256_bytes(
            first_non_confidence
        ),
        "rerun_non_confidence_jsonl_sha256": _sha256_bytes(
            second_non_confidence
        ),
        "record_count": first["record_count"],
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
    _write_bytes(arguments.observation, _canonical_bytes(observation))
    print(canonical_json(observation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
