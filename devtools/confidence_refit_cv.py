"""Group-exclusive repeated OOF comparison for WO19 confidence candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mib_pipeline.confidence_refit import (
    CALIBRATION_FAMILIES,
    CalibrationExample,
    FittedConfidenceCalibrator,
    fit_confidence_calibrator,
)
from mib_pipeline.final_confidence import FinalConfidenceContext


SAMPLE_SCHEMA = "mib-confidence-refit-samples/v1"
REPORT_SCHEMA = "mib-confidence-refit-grouped-oof/v1"
REPEAT_SEEDS = (19001, 19002, 19003)
FOLD_COUNT = 5
MINIMUM_SLICE_SUPPORT = 5
MINIMUM_GROUP_COUNT = 5
PROMOTION_TOLERANCE = 1e-12
TRACKED_BRIER_TARGET = 0.01
SELECTION_TIE_BREAK = (
    "temperature",
    "beta",
    "hierarchical_shrunk",
    "isotonic",
)
FAMILY_TIE_BREAK = {
    family: index for index, family in enumerate(SELECTION_TIE_BREAK)
}


class ConfidenceCVError(ValueError):
    """Raised when the grouped comparison inputs are incomplete or unsafe."""


@dataclass(frozen=True)
class GroupedCalibrationExample:
    """Training-only sample whose identity/group keys are never serialized."""

    sample_key: str
    group_key: str
    calibration: CalibrationExample

    def __post_init__(self) -> None:
        if not isinstance(self.sample_key, str) or not self.sample_key:
            raise ConfidenceCVError("sample_key must be non-empty")
        if not isinstance(self.group_key, str) or not self.group_key:
            raise ConfidenceCVError("group_key must be non-empty")
        if not isinstance(self.calibration, CalibrationExample):
            raise TypeError("calibration must be CalibrationExample")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _finite_probability(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ConfidenceCVError(f"{name} must be finite and within [0, 1]")
    return float(value)


def grouped_folds(
    examples: Sequence[GroupedCalibrationExample],
    *,
    seed: int,
    fold_count: int = FOLD_COUNT,
) -> tuple[tuple[int, ...], ...]:
    """Assign whole groups to deterministically size-balanced folds."""

    if (
        isinstance(fold_count, bool)
        or not isinstance(fold_count, int)
        or fold_count < 2
    ):
        raise ConfidenceCVError("fold_count must be at least two")
    groups: dict[str, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(example.group_key, []).append(index)
    minimum_groups = max(MINIMUM_GROUP_COUNT, fold_count)
    if len(groups) < minimum_groups:
        raise ConfidenceCVError(
            f"grouped CV requires at least {minimum_groups} distinct groups"
        )
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
            group,
        ),
    )
    fold_indices: list[list[int]] = [[] for _ in range(fold_count)]
    fold_sizes = [0] * fold_count
    for group in ordered_groups:
        target = min(range(fold_count), key=lambda index: (fold_sizes[index], index))
        fold_indices[target].extend(groups[group])
        fold_sizes[target] += len(groups[group])
    folds = tuple(tuple(sorted(indices)) for indices in fold_indices)
    if any(not fold for fold in folds):
        raise ConfidenceCVError("every grouped CV fold must have test samples")
    if any(len(fold) == len(examples) for fold in folds):
        raise ConfidenceCVError("every grouped CV fold must have train samples")
    return folds


def _reliability_bins(
    predictions: Sequence[float],
    labels: Sequence[bool],
    *,
    bins: int,
) -> list[dict[str, Any]]:
    if len(predictions) != len(labels):
        raise ConfidenceCVError("reliability inputs are not aligned")
    total = len(predictions)
    result = []
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, prediction in enumerate(predictions)
            if lower <= prediction < upper
            or (bin_index == bins - 1 and prediction == 1.0)
        ]
        if not members:
            result.append(
                {
                    "bin": bin_index,
                    "lower_inclusive": lower,
                    "upper_inclusive": upper if bin_index == bins - 1 else None,
                    "upper_exclusive": None if bin_index == bins - 1 else upper,
                    "support": 0,
                    "weight": 0.0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "absolute_gap": None,
                }
            )
            continue
        mean_confidence = (
            sum(predictions[index] for index in members) / len(members)
        )
        accuracy = sum(labels[index] for index in members) / len(members)
        result.append(
            {
                "bin": bin_index,
                "lower_inclusive": lower,
                "upper_inclusive": upper if bin_index == bins - 1 else None,
                "upper_exclusive": None if bin_index == bins - 1 else upper,
                "support": len(members),
                "weight": len(members) / total if total else 0.0,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": abs(mean_confidence - accuracy),
            }
        )
    return result


def _ece(reliability: Sequence[Mapping[str, Any]]) -> float | None:
    if not reliability or not any(bin_value["support"] for bin_value in reliability):
        return None
    return sum(
        float(bin_value["weight"]) * float(bin_value["absolute_gap"])
        for bin_value in reliability
        if bin_value["support"]
    )


def _metric_summary(
    predictions: Sequence[float],
    labels: Sequence[bool],
    *,
    unique_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    if len(predictions) != len(labels):
        raise ConfidenceCVError("predictions and labels are not aligned")
    if unique_keys is None:
        unique_keys = tuple(str(index) for index in range(len(predictions)))
    if len(unique_keys) != len(predictions):
        raise ConfidenceCVError("metric unique keys are not aligned")
    unique_support = len(set(unique_keys))
    if not predictions:
        return {
            "support": 0,
            "unique_support": 0,
            "brier": None,
            "calibration_score": None,
            "ece_5_bin": None,
            "ece_10_bin": None,
            "reliability_bins_10": _reliability_bins([], [], bins=10),
            "accuracy": None,
            "mean_confidence": None,
        }
    rendered = [
        _finite_probability(value, name="prediction") for value in predictions
    ]
    brier = sum(
        (prediction - float(label)) ** 2
        for prediction, label in zip(rendered, labels, strict=True)
    ) / len(rendered)
    reliability_5 = _reliability_bins(rendered, labels, bins=5)
    reliability_10 = _reliability_bins(rendered, labels, bins=10)
    return {
        "support": len(rendered),
        "unique_support": unique_support,
        "brier": brier,
        "calibration_score": 20.0 * max(0.0, 1.0 - 2.0 * brier),
        "ece_5_bin": _ece(reliability_5),
        "ece_10_bin": _ece(reliability_10),
        "reliability_bins_10": reliability_10,
        "accuracy": sum(labels) / len(labels),
        "mean_confidence": sum(rendered) / len(rendered),
    }


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _confidence_distribution(
    values: Sequence[float],
    *,
    unique_keys: Sequence[str],
) -> dict[str, Any]:
    if len(values) != len(unique_keys):
        raise ConfidenceCVError("distribution unique keys are not aligned")
    if not values:
        return {
            "support": 0,
            "unique_support": 0,
            "mean": None,
            "median": None,
            "standard_deviation": None,
            "p10": None,
            "p90": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "support": len(values),
        "unique_support": len(set(unique_keys)),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "standard_deviation": statistics.pstdev(values),
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
        "minimum": min(values),
        "maximum": max(values),
    }


def _slice_metrics(
    predictions: Sequence[float],
    labels: Sequence[bool],
    values: Sequence[str],
    unique_keys: Sequence[str],
) -> dict[str, Any]:
    if not (
        len(predictions)
        == len(labels)
        == len(values)
        == len(unique_keys)
    ):
        raise ConfidenceCVError("slice inputs are not aligned")
    buckets: dict[str, list[int]] = {}
    for index, value in enumerate(values):
        buckets.setdefault(value, []).append(index)
    retained: dict[str, Any] = {}
    sparse: list[int] = []
    for value in sorted(buckets):
        members = buckets[value]
        unique_support = len({unique_keys[index] for index in members})
        if unique_support < MINIMUM_SLICE_SUPPORT:
            sparse.extend(members)
            continue
        retained[value] = _metric_summary(
            [predictions[index] for index in members],
            [labels[index] for index in members],
            unique_keys=[unique_keys[index] for index in members],
        )
    if sparse:
        retained["OTHER_SPARSE"] = _metric_summary(
            [predictions[index] for index in sparse],
            [labels[index] for index in sparse],
            unique_keys=[unique_keys[index] for index in sparse],
        )
        retained["OTHER_SPARSE"]["category_count"] = sum(
            len({unique_keys[index] for index in members})
            < MINIMUM_SLICE_SUPPORT
            for members in buckets.values()
        )
    return retained


def _evaluation(
    predictions: Sequence[float],
    examples: Sequence[GroupedCalibrationExample],
) -> dict[str, Any]:
    labels = [example.calibration.correct for example in examples]
    unique_keys = [example.sample_key for example in examples]
    return {
        "overall": _metric_summary(
            predictions,
            labels,
            unique_keys=unique_keys,
        ),
        "slices": {
            "final_class": _slice_metrics(
                predictions,
                labels,
                [example.calibration.context.final_class for example in examples],
                unique_keys,
            ),
            "policy_route": _slice_metrics(
                predictions,
                labels,
                [example.calibration.context.policy_route for example in examples],
                unique_keys,
            ),
            "recovery_route": _slice_metrics(
                predictions,
                labels,
                [example.calibration.context.recovery_route for example in examples],
                unique_keys,
            ),
        },
        "confidence_distribution": {
            "correct": _confidence_distribution(
                [
                    prediction
                    for prediction, label in zip(
                        predictions,
                        labels,
                        strict=True,
                    )
                    if label
                ],
                unique_keys=[
                    unique_key
                    for unique_key, label in zip(
                        unique_keys,
                        labels,
                        strict=True,
                    )
                    if label
                ],
            ),
            "incorrect": _confidence_distribution(
                [
                    prediction
                    for prediction, label in zip(
                        predictions,
                        labels,
                        strict=True,
                    )
                    if not label
                ],
                unique_keys=[
                    unique_key
                    for unique_key, label in zip(
                        unique_keys,
                        labels,
                        strict=True,
                    )
                    if not label
                ],
            ),
        },
    }


def _family_oof(
    family: str,
    examples: Sequence[GroupedCalibrationExample],
) -> tuple[dict[str, Any], list[float]]:
    all_predictions: list[float] = []
    all_examples: list[GroupedCalibrationExample] = []
    repeat_reports = []
    fold_reports = []
    for repeat_index, seed in enumerate(REPEAT_SEEDS):
        repeat_predictions: list[float] = []
        repeat_examples: list[GroupedCalibrationExample] = []
        folds = grouped_folds(examples, seed=seed)
        for fold_index, test_indices in enumerate(folds):
            test_set = set(test_indices)
            train = [
                example.calibration
                for index, example in enumerate(examples)
                if index not in test_set
            ]
            test = [examples[index] for index in test_indices]
            if not train or not test:
                raise ConfidenceCVError(
                    "every grouped CV fold requires nonempty train and test data"
                )
            calibrator = fit_confidence_calibrator(family, train)
            predictions = [
                calibrator.predict(
                    example.calibration.input_confidence,
                    example.calibration.context,
                )
                for example in test
            ]
            train_groups = {
                example.group_key
                for index, example in enumerate(examples)
                if index not in test_set
            }
            test_groups = {example.group_key for example in test}
            overlap = train_groups & test_groups
            if overlap:
                raise AssertionError("group-exclusive split leaked a group")
            fold_reports.append(
                {
                    "repeat": repeat_index,
                    "fold": fold_index,
                    "train_support": len(train),
                    "test_support": len(test),
                    "train_group_count": len(train_groups),
                    "test_group_count": len(test_groups),
                    "group_overlap_count": 0,
                    "metrics": _metric_summary(
                        predictions,
                        [example.calibration.correct for example in test],
                        unique_keys=[example.sample_key for example in test],
                    ),
                }
            )
            repeat_predictions.extend(predictions)
            repeat_examples.extend(test)
        repeat_reports.append(
            {
                "repeat": repeat_index,
                "metrics": _evaluation(repeat_predictions, repeat_examples),
            }
        )
        all_predictions.extend(repeat_predictions)
        all_examples.extend(repeat_examples)
    return (
        {
            "aggregate": _evaluation(all_predictions, all_examples),
            "repeats": repeat_reports,
            "folds": fold_reports,
        },
        all_predictions,
    )


def _baseline_report(
    examples: Sequence[GroupedCalibrationExample],
) -> dict[str, Any]:
    repeated_examples = list(examples) * len(REPEAT_SEEDS)
    predictions = [
        example.calibration.input_confidence for example in repeated_examples
    ]
    repeats = []
    for repeat_index in range(len(REPEAT_SEEDS)):
        repeats.append(
            {
                "repeat": repeat_index,
                "metrics": _evaluation(
                    [example.calibration.input_confidence for example in examples],
                    examples,
                ),
            }
        )
    return {
        "aggregate": _evaluation(predictions, repeated_examples),
        "repeats": repeats,
    }


def _supported_slice_regressions(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    regressions = []
    baseline_slices = baseline["aggregate"]["slices"]
    candidate_slices = candidate["aggregate"]["slices"]
    for dimension in ("final_class", "policy_route", "recovery_route"):
        for value in sorted(set(baseline_slices[dimension]) & set(candidate_slices[dimension])):
            baseline_metric = baseline_slices[dimension][value]
            candidate_metric = candidate_slices[dimension][value]
            if (
                baseline_metric["unique_support"] < MINIMUM_SLICE_SUPPORT
                or candidate_metric["unique_support"] < MINIMUM_SLICE_SUPPORT
            ):
                continue
            delta = candidate_metric["brier"] - baseline_metric["brier"]
            if delta > PROMOTION_TOLERANCE:
                regressions.append(
                    {
                        "dimension": dimension,
                        "value": value,
                        "unique_support": candidate_metric["unique_support"],
                        "pooled_support": candidate_metric["support"],
                        "brier_delta": delta,
                    }
                )
    return regressions


def _parameter_count(calibrator: FittedConfidenceCalibrator) -> int:
    if calibrator.family == "temperature":
        return 1
    if calibrator.family == "beta":
        return len(calibrator.parameters["coefficients"])
    if calibrator.family == "isotonic":
        return 2 * len(calibrator.parameters["values"])
    return 1 + sum(
        len(calibrator.parameters[node_name])
        for node_name in (
            "class_nodes",
            "policy_nodes",
            "recovery_nodes",
            "evidence_nodes",
        )
    )


def _leave_one_group_out_guard(
    family: str,
    examples: Sequence[GroupedCalibrationExample],
) -> dict[str, Any]:
    groups: dict[str, list[GroupedCalibrationExample]] = {}
    for example in examples:
        groups.setdefault(example.group_key, []).append(example)
    if len(groups) < MINIMUM_GROUP_COUNT:
        raise ConfidenceCVError(
            f"leave-one-group-out requires at least {MINIMUM_GROUP_COUNT} groups"
        )
    candidate_predictions: list[float] = []
    baseline_predictions: list[float] = []
    held_out_examples: list[GroupedCalibrationExample] = []
    group_deltas = []
    holdout_supports = []
    for group_key in sorted(groups):
        test = groups[group_key]
        train = [
            example.calibration
            for example in examples
            if example.group_key != group_key
        ]
        if not train or not test:
            raise ConfidenceCVError(
                "leave-one-group-out requires nonempty train and test data"
            )
        calibrator = fit_confidence_calibrator(family, train)
        candidate = [
            calibrator.predict(
                example.calibration.input_confidence,
                example.calibration.context,
            )
            for example in test
        ]
        baseline = [
            example.calibration.input_confidence for example in test
        ]
        labels = [example.calibration.correct for example in test]
        candidate_brier = _metric_summary(
            candidate,
            labels,
            unique_keys=[example.sample_key for example in test],
        )["brier"]
        baseline_brier = _metric_summary(
            baseline,
            labels,
            unique_keys=[example.sample_key for example in test],
        )["brier"]
        group_deltas.append(candidate_brier - baseline_brier)
        holdout_supports.append(len(test))
        candidate_predictions.extend(candidate)
        baseline_predictions.extend(baseline)
        held_out_examples.extend(test)
    regression_count = sum(
        delta > PROMOTION_TOLERANCE for delta in group_deltas
    )
    return {
        "holdout_count": len(groups),
        "minimum_holdout_support": min(holdout_supports),
        "maximum_holdout_support": max(holdout_supports),
        "baseline": _evaluation(baseline_predictions, held_out_examples),
        "candidate": _evaluation(candidate_predictions, held_out_examples),
        "mean_group_brier_delta": sum(group_deltas) / len(group_deltas),
        "worst_group_brier_delta": max(group_deltas),
        "regression_count": regression_count,
        "non_regression": regression_count == 0,
    }


def compare_confidence_families(
    examples: Iterable[GroupedCalibrationExample],
) -> tuple[dict[str, Any], dict[str, FittedConfidenceCalibrator]]:
    """Run exact 3x5 grouped OOF and fit identity-free full-data artifacts."""

    ordered = tuple(sorted(examples, key=lambda example: example.sample_key))
    if not ordered:
        raise ConfidenceCVError("at least one calibration example is required")
    if any(not isinstance(example, GroupedCalibrationExample) for example in ordered):
        raise TypeError("examples must contain GroupedCalibrationExample values")
    sample_keys = [example.sample_key for example in ordered]
    if len(set(sample_keys)) != len(sample_keys):
        raise ConfidenceCVError("sample_key values must be unique")
    group_count = len({example.group_key for example in ordered})
    if group_count < MINIMUM_GROUP_COUNT:
        raise ConfidenceCVError(
            f"grouped comparison requires at least {MINIMUM_GROUP_COUNT} groups"
        )

    baseline = _baseline_report(ordered)
    candidates: dict[str, Any] = {}
    artifacts: dict[str, FittedConfidenceCalibrator] = {}
    parameter_counts: dict[str, int] = {}
    for family in CALIBRATION_FAMILIES:
        report, _ = _family_oof(family, ordered)
        report["leave_one_group_out"] = _leave_one_group_out_guard(
            family,
            ordered,
        )
        candidates[family] = report
        artifacts[family] = fit_confidence_calibrator(
            family,
            [example.calibration for example in ordered],
        )
        parameter_counts[family] = _parameter_count(artifacts[family])
    selected_family = min(
        CALIBRATION_FAMILIES,
        key=lambda family: (
            candidates[family]["aggregate"]["overall"]["brier"],
            parameter_counts[family],
            FAMILY_TIE_BREAK[family],
        ),
    )
    baseline_brier = baseline["aggregate"]["overall"]["brier"]
    selected_brier = candidates[selected_family]["aggregate"]["overall"]["brier"]
    baseline_score = baseline["aggregate"]["overall"]["calibration_score"]
    selected_score = candidates[selected_family]["aggregate"]["overall"][
        "calibration_score"
    ]
    slice_regressions = _supported_slice_regressions(
        baseline,
        candidates[selected_family],
    )
    repeat_brier_deltas = [
        candidate_repeat["metrics"]["overall"]["brier"]
        - baseline_repeat["metrics"]["overall"]["brier"]
        for candidate_repeat, baseline_repeat in zip(
            candidates[selected_family]["repeats"],
            baseline["repeats"],
            strict=True,
        )
    ]
    overall_improved = (
        selected_brier < baseline_brier - PROMOTION_TOLERANCE
    )
    every_repeat_improved = all(
        delta < -PROMOTION_TOLERANCE for delta in repeat_brier_deltas
    )
    score_improved = (
        selected_score > baseline_score + PROMOTION_TOLERANCE
    )
    logo_non_regression = candidates[selected_family][
        "leave_one_group_out"
    ]["non_regression"]
    promotion_checks = {
        "mean_brier_strictly_improved": overall_improved,
        "every_repeat_brier_strictly_improved": every_repeat_improved,
        "calibration_score_strictly_improved": score_improved,
        "supported_slices_non_regressing": not slice_regressions,
        "leave_one_group_out_non_regressing": logo_non_regression,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "protocol": {
            "repeats": len(REPEAT_SEEDS),
            "folds": FOLD_COUNT,
            "minimum_group_count": MINIMUM_GROUP_COUNT,
            "group_exclusive": True,
            "repeat_seeds_sha256": hashlib.sha256(
                _canonical_json(list(REPEAT_SEEDS)).encode("utf-8")
            ).hexdigest(),
            "slice_minimum_support": MINIMUM_SLICE_SUPPORT,
            "slice_support_unit": "unique_samples",
            "sparse_slice_policy": "aggregate_as_OTHER_SPARSE",
            "ece_bins": [5, 10],
            "official_calibration_formula": (
                "20 * max(0, 1 - 2 * mean_brier)"
            ),
            "tracked_brier_target": TRACKED_BRIER_TARGET,
            "tracked_target_policy": "reported_not_forced",
        },
        "support": {
            "samples": len(ordered),
            "groups": group_count,
        },
        "baseline": baseline,
        "candidates": candidates,
        "selection": {
            "family": selected_family,
            "selection_order": (
                "mean_oof_brier",
                "parameter_count",
                "fixed_family_order",
            ),
            "tie_break_order": list(SELECTION_TIE_BREAK),
            "parameter_counts": parameter_counts,
            "baseline_brier": baseline_brier,
            "selected_brier": selected_brier,
            "brier_delta": selected_brier - baseline_brier,
            "calibration_score_delta": selected_score - baseline_score,
            "repeat_brier_deltas": repeat_brier_deltas,
            "supported_slice_regressions": slice_regressions,
            "promotion_checks": promotion_checks,
            "promotion_recommended": all(promotion_checks.values()),
            "tracked_brier_target": {
                "threshold": TRACKED_BRIER_TARGET,
                "selected_mean_brier": selected_brier,
                "delta_to_target": selected_brier - TRACKED_BRIER_TARGET,
                "met": selected_brier <= TRACKED_BRIER_TARGET,
                "promotion_gate": False,
                "policy": "tracked_not_forced",
            },
        },
    }
    return report, artifacts


def load_grouped_examples(path: Path) -> tuple[GroupedCalibrationExample, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfidenceCVError(f"cannot read calibration samples: {path}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "samples",
    } or value.get("schema_version") != SAMPLE_SCHEMA:
        raise ConfidenceCVError("unsupported calibration sample schema")
    samples = value["samples"]
    if not isinstance(samples, list):
        raise ConfidenceCVError("samples must be a list")
    result = []
    expected = {
        "sample_key",
        "group_key",
        "input_confidence",
        "correct",
        "context",
    }
    for sample in samples:
        if not isinstance(sample, dict) or set(sample) != expected:
            raise ConfidenceCVError("calibration sample fields are not exact")
        context = sample["context"]
        if not isinstance(context, dict):
            raise ConfidenceCVError("sample context must be an object")
        result.append(
            GroupedCalibrationExample(
                sample_key=sample["sample_key"],
                group_key=sample["group_key"],
                calibration=CalibrationExample(
                    input_confidence=sample["input_confidence"],
                    correct=sample["correct"],
                    context=FinalConfidenceContext(**context),
                ),
            )
        )
    return tuple(result)


def _write_outputs(
    output_dir: Path,
    report: Mapping[str, Any],
    artifacts: Mapping[str, FittedConfidenceCalibrator],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "grouped-oof-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for family in CALIBRATION_FAMILIES:
        (output_dir / f"{family}-runtime-artifact.json").write_text(
            json.dumps(
                artifacts[family].to_runtime_mapping(),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    arguments = parser.parse_args(argv)
    examples = load_grouped_examples(arguments.samples)
    report, artifacts = compare_confidence_families(examples)
    _write_outputs(arguments.output_dir, report, artifacts)
    print(_canonical_json(report["selection"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
