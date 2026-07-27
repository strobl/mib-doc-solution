#!/usr/bin/env python3
"""Fixed three-by-five grouped OOF runner for the four WO-18 approaches.

The runner consumes frozen, identity-bearing feature rows outside the
repository.  It fits the compact head on training groups only, predicts each
validation group exactly once per repeat, and executes the exact production
feature-level comparators.  It emits byte-pinned arm manifests consumed and
independently recomputed by :mod:`devtools.decision_recovery_evidence`.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.decision_recovery_gate import (  # noqa: E402
    APPROACHES,
    REQUIRED_FOLDS,
    REQUIRED_REPEATS,
)
from devtools.experiment_control import (  # noqa: E402
    RepeatedGroupedSplitManager,
    canonical_json,
)
from devtools.grouped_recovery_evidence import (  # noqa: E402
    FrozenLayoutManifest,
    _read_json_object,
    _read_truth_subset,
    load_layout_manifest,
)
from mib_pipeline.model_recovery import (  # noqa: E402
    FEATURE_NAMES,
    MODEL_CLASSES,
    CompactThreeClassModel,
    EvidenceCompletionOnlyModel,
    GatedHybridDecisionRule,
    IdentityFreeDecisionFeatures,
)
from mib_pipeline.models import FIELD_NAMES, PredictionRow  # noqa: E402
from scripts import evaluate as official_evaluate  # noqa: E402


FEATURE_ROWS_SCHEMA = "mib-wo18-frozen-feature-rows/v1"
ARM_MANIFEST_SCHEMA = "mib-wo18-oof-arm/v1"
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class DecisionRecoveryCVError(ValueError):
    """The frozen feature cohort or OOF execution contract is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path | str) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _hash_values(values: Sequence[str]) -> str:
    return _sha256_bytes(
        canonical_json(sorted(str(value) for value in values)).encode("utf-8")
    )


def feature_schema_sha256() -> str:
    return _sha256_bytes(
        canonical_json(list(FEATURE_NAMES)).encode("utf-8")
    )


def runner_source_sha256() -> str:
    return _sha256_file(Path(__file__))


def approach_graph_sha256(approach: str) -> str:
    """Hash the exact callable graph used for an approach."""

    if approach not in APPROACHES:
        raise DecisionRecoveryCVError("unknown WO-18 approach")
    graph: dict[str, str] = {
        "deterministic_engine": "frozen_baseline_prediction_passthrough_v1",
        "evidence_completion_only": (
            inspect.getsource(EvidenceCompletionOnlyModel.predict)
            + inspect.getsource(GatedHybridDecisionRule.decide)
        ),
        "compact_identity_free_model": (
            inspect.getsource(CompactThreeClassModel.fit)
            + inspect.getsource(CompactThreeClassModel.predict)
        ),
        "gated_hybrid": (
            inspect.getsource(CompactThreeClassModel.fit)
            + inspect.getsource(CompactThreeClassModel.predict)
            + inspect.getsource(GatedHybridDecisionRule.decide)
        ),
    }
    return _sha256_bytes(graph[approach].encode("utf-8"))


@dataclass(frozen=True)
class FrozenFeatureCase:
    case_id: str
    baseline_prediction: PredictionRow
    baseline_decision: str
    recovery_route: str
    features: IdentityFreeDecisionFeatures


@dataclass(frozen=True)
class FrozenFeatureCohort:
    cases: Mapping[str, FrozenFeatureCase]
    sha256: str
    source_revision_sha: str
    input_tree_sha256: str


@dataclass(frozen=True)
class FoldExecution:
    repeat: int
    fold: int
    training_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    training_case_ids: tuple[str, ...]
    validation_case_ids: tuple[str, ...]
    model_artifact: Mapping[str, object]
    validation_rows: Mapping[str, tuple[Mapping[str, Any], ...]]


@dataclass(frozen=True)
class ComparisonExecution:
    rows: Mapping[str, tuple[tuple[Mapping[str, Any], ...], ...]]
    folds: tuple[FoldExecution, ...]


def load_frozen_feature_rows(
    path: Path | str,
    *,
    layout_manifest: FrozenLayoutManifest,
    expected_source_revision_sha: str,
    expected_input_tree_sha256: str,
) -> FrozenFeatureCohort:
    """Load exact numeric features without labels or arbitrary metadata."""

    feature_path = Path(path).resolve()
    raw = _read_json_object(feature_path, label="frozen feature rows")
    if set(raw) != {
        "schema_version",
        "frozen_before_fit",
        "source_revision_sha",
        "layout_manifest_sha256",
        "input_tree_sha256",
        "feature_schema_sha256",
        "rows",
    }:
        raise DecisionRecoveryCVError(
            "frozen feature rows must contain the exact contract keys"
        )
    if (
        raw.get("schema_version") != FEATURE_ROWS_SCHEMA
        or raw.get("frozen_before_fit") is not True
    ):
        raise DecisionRecoveryCVError(
            "feature rows require the frozen WO-18 schema"
        )
    source_revision = str(raw.get("source_revision_sha", "")).casefold()
    if (
        not _GIT_COMMIT_RE.fullmatch(source_revision)
        or source_revision
        != str(expected_source_revision_sha).strip().casefold()
    ):
        raise DecisionRecoveryCVError(
            "feature rows source revision binding mismatch"
        )
    for name, actual, expected in (
        (
            "layout",
            raw.get("layout_manifest_sha256"),
            layout_manifest.sha256,
        ),
        (
            "input tree",
            raw.get("input_tree_sha256"),
            expected_input_tree_sha256,
        ),
        (
            "feature schema",
            raw.get("feature_schema_sha256"),
            feature_schema_sha256(),
        ),
    ):
        normalized = str(actual).strip().casefold()
        if not _SHA256_RE.fullmatch(normalized) or normalized != str(
            expected
        ).strip().casefold():
            raise DecisionRecoveryCVError(
                f"feature rows {name} binding mismatch"
            )
    rows = raw.get("rows")
    if not isinstance(rows, list) or not rows:
        raise DecisionRecoveryCVError("frozen feature rows must be non-empty")
    cases: dict[str, FrozenFeatureCase] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "case_id",
            "baseline_prediction",
            "baseline_decision",
            "recovery_route",
            "features",
        }:
            raise DecisionRecoveryCVError(
                f"feature row {index} has an invalid schema"
            )
        case_id = str(row.get("case_id", "")).strip()
        if case_id in cases:
            raise DecisionRecoveryCVError(
                "feature rows contain duplicate case identity"
            )
        baseline_raw = row.get("baseline_prediction")
        features_raw = row.get("features")
        if not isinstance(baseline_raw, Mapping) or set(
            baseline_raw
        ) != set(FIELD_NAMES):
            raise DecisionRecoveryCVError(
                "baseline prediction must have the exact output schema"
            )
        if not isinstance(features_raw, Mapping) or set(
            features_raw
        ) != set(FEATURE_NAMES):
            raise DecisionRecoveryCVError(
                "feature row must use the exact ordered feature schema"
            )
        baseline = PredictionRow.from_mapping(baseline_raw)
        decision = str(row.get("baseline_decision", "")).strip().upper()
        route = str(row.get("recovery_route", "")).strip()
        if (
            case_id != baseline.case_id
            or decision != baseline.adjudication
            or decision not in MODEL_CLASSES
            or route not in {"primary", "late_visible", "rapid_visible"}
        ):
            raise DecisionRecoveryCVError(
                "feature row baseline/route binding is invalid"
            )
        features = IdentityFreeDecisionFeatures.from_mapping(
            {name: features_raw[name] for name in FEATURE_NAMES}
        )
        baseline_flags = (
            features.values["baseline_approved"],
            features.values["baseline_denied"],
            features.values["baseline_review"],
        )
        expected_flags = tuple(
            float(decision == class_name) for class_name in MODEL_CLASSES
        )
        if baseline_flags != expected_flags:
            raise DecisionRecoveryCVError(
                "feature row baseline one-hot does not match its decision"
            )
        cases[case_id] = FrozenFeatureCase(
            case_id=case_id,
            baseline_prediction=baseline,
            baseline_decision=decision,
            recovery_route=route,
            features=features,
        )
    if set(cases) != set(layout_manifest.case_ids):
        raise DecisionRecoveryCVError(
            "feature rows must cover the exact frozen cohort"
        )
    return FrozenFeatureCohort(
        cases=MappingProxyType(dict(sorted(cases.items()))),
        sha256=_sha256_file(feature_path),
        source_revision_sha=source_revision,
        input_tree_sha256=str(expected_input_tree_sha256).casefold(),
    )


def _prediction_row(
    sample: FrozenFeatureCase,
    decision: str,
) -> Mapping[str, Any]:
    return replace(
        sample.baseline_prediction,
        adjudication=decision,
    ).to_dict()


def execute_grouped_oof(
    *,
    layout_manifest: FrozenLayoutManifest,
    truth: Mapping[str, Mapping[str, Any]],
    features: FrozenFeatureCohort,
) -> ComparisonExecution:
    """Fit/predict exact validation groups and assemble complete OOF rows."""

    manager = RepeatedGroupedSplitManager(
        seed=layout_manifest.split_seed,
        repeats=REQUIRED_REPEATS,
        folds=REQUIRED_FOLDS,
    )
    splits = manager.split_groups(layout_manifest.groups)
    rows: dict[str, list[dict[str, Mapping[str, Any]]]] = {
        approach: [dict() for _ in range(REQUIRED_REPEATS)]
        for approach in APPROACHES
    }
    fold_executions: list[FoldExecution] = []
    rule = GatedHybridDecisionRule()
    for split in splits:
        training_features = [
            features.cases[case_id].features
            for case_id in split.tuning_case_ids
        ]
        training_labels = [
            str(truth[case_id].get("adjudication", "")).strip().upper()
            for case_id in split.tuning_case_ids
        ]
        if any(label not in MODEL_CLASSES for label in training_labels):
            raise DecisionRecoveryCVError(
                "truth labels must use the frozen three-class schema"
            )
        model = CompactThreeClassModel.fit(
            training_features,
            training_labels,
        )
        validation_rows: dict[str, list[Mapping[str, Any]]] = {
            approach: [] for approach in APPROACHES
        }
        for case_id in split.validation_case_ids:
            sample = features.cases[case_id]
            evidence_prediction = EvidenceCompletionOnlyModel.predict(
                sample.features
            )
            model_prediction = model.predict(sample.features)
            decisions = {
                "deterministic_engine": sample.baseline_decision,
                "evidence_completion_only": rule.decide(
                    sample.baseline_decision,
                    sample.features,
                    evidence_prediction,
                ).decision,
                "compact_identity_free_model": model_prediction.decision,
                "gated_hybrid": rule.decide(
                    sample.baseline_decision,
                    sample.features,
                    model_prediction,
                ).decision,
            }
            for approach in APPROACHES:
                row = _prediction_row(sample, decisions[approach])
                if case_id in rows[approach][split.repeat]:
                    raise DecisionRecoveryCVError(
                        "OOF validation case was predicted more than once"
                    )
                rows[approach][split.repeat][case_id] = row
                validation_rows[approach].append(row)
        fold_executions.append(
            FoldExecution(
                repeat=split.repeat,
                fold=split.fold,
                training_groups=split.tuning_groups,
                validation_groups=split.validation_groups,
                training_case_ids=split.tuning_case_ids,
                validation_case_ids=split.validation_case_ids,
                model_artifact=MappingProxyType(
                    model.members[0].to_dict()
                ),
                validation_rows=MappingProxyType(
                    {
                        approach: tuple(
                            sorted(
                                validation_rows[approach],
                                key=lambda row: str(row["case_id"]),
                            )
                        )
                        for approach in APPROACHES
                    }
                ),
            )
        )
    expected = set(layout_manifest.case_ids)
    normalized_rows: dict[
        str, tuple[tuple[Mapping[str, Any], ...], ...]
    ] = {}
    for approach in APPROACHES:
        repeats: list[tuple[Mapping[str, Any], ...]] = []
        for repeat in range(REQUIRED_REPEATS):
            if set(rows[approach][repeat]) != expected:
                raise DecisionRecoveryCVError(
                    "OOF execution did not cover the exact cohort"
                )
            repeats.append(
                tuple(
                    rows[approach][repeat][case_id]
                    for case_id in sorted(expected)
                )
            )
        normalized_rows[approach] = tuple(repeats)
    return ComparisonExecution(
        rows=MappingProxyType(normalized_rows),
        folds=tuple(fold_executions),
    )


def execution_fingerprint(execution: ComparisonExecution) -> str:
    """Hash every prediction and fold artifact, excluding filesystem paths."""

    payload = {
        "rows": {
            approach: [list(run) for run in execution.rows[approach]]
            for approach in APPROACHES
        },
        "folds": [
            {
                "repeat": fold.repeat,
                "fold": fold.fold,
                "training_groups_sha256": _hash_values(
                    fold.training_groups
                ),
                "validation_groups_sha256": _hash_values(
                    fold.validation_groups
                ),
                "training_case_set_sha256": _hash_values(
                    fold.training_case_ids
                ),
                "validation_case_set_sha256": _hash_values(
                    fold.validation_case_ids
                ),
                "model_artifact": dict(fold.model_artifact),
                "validation_rows": {
                    approach: list(fold.validation_rows[approach])
                    for approach in APPROACHES
                },
            }
            for fold in execution.folds
        ],
    }
    return _sha256_bytes(canonical_json(payload).encode("utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(canonical_json(dict(row)) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_comparison_artifacts(
    *,
    output_dir: Path | str,
    execution: ComparisonExecution,
    rerun: ComparisonExecution,
    layout_manifest: FrozenLayoutManifest,
    feature_rows_sha256: str,
    protected_role_manifest_sha256: str,
    truth_sha256: str,
    input_tree_sha256: str,
    source_revision_sha: str,
) -> Mapping[str, Path]:
    """Persist exact OOF outputs; refuse a non-deterministic in-process rerun."""

    if execution_fingerprint(execution) != execution_fingerprint(rerun):
        raise DecisionRecoveryCVError(
            "grouped comparison is not deterministic"
        )
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    evaluator_sha = _sha256_file(Path(official_evaluate.__file__))
    source_sha = runner_source_sha256()
    fold_by_key = {
        (fold.repeat, fold.fold): fold for fold in execution.folds
    }
    model_pins: dict[tuple[int, int], tuple[Path, str]] = {}
    for key, fold in fold_by_key.items():
        repeat, fold_index = key
        path = root / "models" / f"repeat-{repeat}-fold-{fold_index}.json"
        _write_json(path, dict(fold.model_artifact))
        model_pins[key] = (path, _sha256_file(path))
    manifest_paths: dict[str, Path] = {}
    for approach in APPROACHES:
        runs: list[dict[str, Any]] = []
        for repeat in range(REQUIRED_REPEATS):
            prediction_path = (
                root / "predictions" / f"{approach}-repeat-{repeat}.jsonl"
            )
            rerun_path = (
                root
                / "predictions"
                / f"{approach}-repeat-{repeat}-rerun.jsonl"
            )
            _write_jsonl(prediction_path, execution.rows[approach][repeat])
            _write_jsonl(rerun_path, rerun.rows[approach][repeat])
            folds: list[dict[str, Any]] = []
            for fold_index in range(REQUIRED_FOLDS):
                fold = fold_by_key[(repeat, fold_index)]
                subset_path = (
                    root
                    / "fold-predictions"
                    / f"{approach}-repeat-{repeat}-fold-{fold_index}.jsonl"
                )
                _write_jsonl(subset_path, fold.validation_rows[approach])
                model_path, model_sha = model_pins[(repeat, fold_index)]
                folds.append(
                    {
                        "fold_index": fold_index,
                        "training_groups_sha256": _hash_values(
                            fold.training_groups
                        ),
                        "validation_groups_sha256": _hash_values(
                            fold.validation_groups
                        ),
                        "training_case_set_sha256": _hash_values(
                            fold.training_case_ids
                        ),
                        "validation_case_set_sha256": _hash_values(
                            fold.validation_case_ids
                        ),
                        "fold_model_artifact_path": str(
                            model_path.resolve()
                        ),
                        "fold_model_artifact_sha256": model_sha,
                        "validation_predictions_path": str(
                            subset_path.resolve()
                        ),
                        "validation_predictions_sha256": _sha256_file(
                            subset_path
                        ),
                    }
                )
            runs.append(
                {
                    "repeat_index": repeat,
                    "predictions_path": str(prediction_path.resolve()),
                    "predictions_sha256": _sha256_file(prediction_path),
                    "rerun_predictions_path": str(rerun_path.resolve()),
                    "rerun_predictions_sha256": _sha256_file(rerun_path),
                    "folds": folds,
                }
            )
        manifest = {
            "schema_version": ARM_MANIFEST_SCHEMA,
            "approach": approach,
            "source_revision_sha": source_revision_sha,
            "layout_manifest_sha256": layout_manifest.sha256,
            "protected_role_manifest_sha256": (
                protected_role_manifest_sha256
            ),
            "truth_sha256": truth_sha256,
            "input_tree_sha256": input_tree_sha256,
            "feature_schema_sha256": feature_schema_sha256(),
            "frozen_feature_rows_sha256": feature_rows_sha256,
            "official_evaluator_sha256": evaluator_sha,
            "runner_source_sha256": source_sha,
            "approach_graph_sha256": approach_graph_sha256(approach),
            "runs": runs,
        }
        path = root / "manifests" / f"{approach}.json"
        _write_json(path, manifest)
        manifest_paths[approach] = path
    return MappingProxyType(manifest_paths)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--feature-rows", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--protected-role-manifest-sha256", required=True)
    parser.add_argument("--input-tree-sha256", required=True)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    manifest = load_layout_manifest(arguments.layout_manifest)
    features = load_frozen_feature_rows(
        arguments.feature_rows,
        layout_manifest=manifest,
        expected_source_revision_sha=arguments.source_revision_sha,
        expected_input_tree_sha256=arguments.input_tree_sha256,
    )
    truth = _read_truth_subset(arguments.truth, manifest.case_ids)
    first = execute_grouped_oof(
        layout_manifest=manifest,
        truth=truth,
        features=features,
    )
    second = execute_grouped_oof(
        layout_manifest=manifest,
        truth=truth,
        features=features,
    )
    paths = write_comparison_artifacts(
        output_dir=arguments.output_dir,
        execution=first,
        rerun=second,
        layout_manifest=manifest,
        feature_rows_sha256=features.sha256,
        protected_role_manifest_sha256=(
            arguments.protected_role_manifest_sha256
        ),
        truth_sha256=_sha256_file(arguments.truth),
        input_tree_sha256=arguments.input_tree_sha256,
        source_revision_sha=arguments.source_revision_sha,
    )
    print(
        canonical_json(
            {
                approach: str(path.resolve())
                for approach, path in paths.items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
