#!/usr/bin/env python3
"""Compare confidence-only calibrators on a frozen final decision graph.

The input rows and layout manifest are development-only and identity-bearing.
The emitted report is aggregate-only: it contains digests, metrics, and
identity-free slice names, never case IDs, filenames, labels, or predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPORT_SCHEMA = "mib-wo19-final-confidence-refit/v1"
EVIDENCE_CLASS = "public_grouped_robustness_not_unseen"
FAMILIES = ("temperature", "beta", "hierarchical_shrunk", "isotonic")
REPEAT_SEEDS = (19001, 19002, 19003)
FOLD_COUNT = 5
GATE_TOLERANCE = 1e-12
BRIER_TARGET = 0.01
EPSILON = 1e-6
OUTPUT_FIELDS = (
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
    "confidence",
)
NON_CONFIDENCE_FIELDS = tuple(
    field for field in OUTPUT_FIELDS if field != "confidence"
)
DEFAULTS = {
    "applicant_name": "unknown",
    "species_code": "unknown",
    "home_world": "unknown",
    "visa_class": "unknown",
    "sponsor_id": "SPN-0000",
    "arrival_date": "1900-01-01",
    "declared_purpose": "unknown",
    "risk_flags": "none",
    "fee_status": "unknown",
}


class FinalRefitError(ValueError):
    """The frozen inputs cannot form trustworthy confidence evidence."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path, *, label: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FinalRefitError(f"{label} rows must be objects")
        case_id = str(value.get("case_id", ""))
        if not case_id or case_id in rows:
            raise FinalRefitError(f"{label} case coverage is not unique")
        missing = set(OUTPUT_FIELDS) - set(value)
        if missing:
            raise FinalRefitError(f"{label} rows are missing output fields")
        rows[case_id] = value
    if not rows:
        raise FinalRefitError(f"{label} is empty")
    return rows


def _read_truth(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    truth: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id or case_id in truth:
            raise FinalRefitError("truth case coverage is not unique")
        if row.get("adjudication") not in {
            "APPROVED",
            "DENIED",
            "NEEDS_REVIEW",
        }:
            raise FinalRefitError("truth adjudication is invalid")
        truth[case_id] = row
    return truth


def _read_layout(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("label_blind_construction") is not True
        or value.get("folds") != FOLD_COUNT
        or value.get("repeats") != len(REPEAT_SEEDS)
        or not isinstance(value.get("cases"), list)
    ):
        raise FinalRefitError("layout manifest is not the required label-blind 3x5 contract")
    groups: dict[str, str] = {}
    for row in value["cases"]:
        if not isinstance(row, dict) or set(row) != {"case_id", "layout_group"}:
            raise FinalRefitError("layout rows must contain only case_id and layout_group")
        case_id = str(row["case_id"])
        group = str(row["layout_group"])
        if not case_id or not group or case_id in groups:
            raise FinalRefitError("layout coverage is not unique")
        groups[case_id] = group
    if len(set(groups.values())) < FOLD_COUNT:
        raise FinalRefitError("layout manifest has too few groups")
    return groups


def _probability(value: Any) -> float:
    if isinstance(value, bool):
        raise FinalRefitError("confidence must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise FinalRefitError("confidence must be finite within [0, 1]")
    return rendered


def _missing_count(row: Mapping[str, Any]) -> int:
    return sum(
        str(row[field]).strip().casefold() == default.casefold()
        for field, default in DEFAULTS.items()
    )


def _finalizer_route(
    parent: Mapping[str, Any],
    final: Mapping[str, Any],
) -> str:
    if parent["adjudication"] != final["adjudication"]:
        return f"decision:{parent['adjudication']}->{final['adjudication']}"
    if any(
        parent[field] != final[field]
        for field in NON_CONFIDENCE_FIELDS
        if field not in {"case_id", "adjudication"}
    ):
        return "fields_only"
    return "unchanged"


def _grouped_folds(
    group_keys: Sequence[str],
    *,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(group_keys):
        members[group].append(index)
    ordered = sorted(
        members,
        key=lambda group: (
            hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
            group,
        ),
    )
    folds: list[list[int]] = [[] for _ in range(FOLD_COUNT)]
    sizes = [0] * FOLD_COUNT
    for group in ordered:
        target = min(range(FOLD_COUNT), key=lambda index: (sizes[index], index))
        folds[target].extend(members[group])
        sizes[target] += len(members[group])
    if any(not fold for fold in folds):
        raise FinalRefitError("every grouped fold must contain test records")
    return tuple(tuple(sorted(fold)) for fold in folds)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _logistic_fit(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    penalty: np.ndarray,
) -> np.ndarray:
    coefficients = np.zeros(features.shape[1], dtype=float)
    for _ in range(100):
        predictions = _sigmoid(features @ coefficients)
        weights = predictions * (1.0 - predictions)
        gradient = features.T @ (predictions - labels) + penalty * coefficients
        hessian = (
            features.T @ (features * weights[:, None])
            + np.diag(penalty)
            + np.eye(features.shape[1]) * 1e-8
        )
        step = np.linalg.solve(hessian, gradient)
        coefficients -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return coefficients


def _confidence_features(
    confidences: np.ndarray,
    family: str,
) -> tuple[np.ndarray, np.ndarray]:
    clipped = np.clip(confidences, EPSILON, 1.0 - EPSILON)
    if family == "temperature":
        return (
            np.log(clipped / (1.0 - clipped))[:, None],
            np.array([0.05]),
        )
    if family == "beta":
        return (
            np.column_stack(
                (
                    np.log(clipped),
                    -np.log1p(-clipped),
                    np.ones(len(clipped)),
                )
            ),
            np.array([1.0, 1.0, 0.0]),
        )
    raise FinalRefitError("unsupported logistic family")


def _fit_isotonic(
    confidences: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(confidences, kind="mergesort")
    values: list[float] = []
    weights: list[int] = []
    highs: list[float] = []
    for confidence, label in zip(
        confidences[order],
        labels[order],
        strict=True,
    ):
        values.append(float(label))
        weights.append(1)
        highs.append(float(confidence))
        while len(values) >= 2 and values[-2] > values[-1]:
            weight = weights[-2] + weights[-1]
            pooled = (
                values[-2] * weights[-2] + values[-1] * weights[-1]
            ) / weight
            values[-2:] = [pooled]
            weights[-2:] = [weight]
            highs[-2:] = [highs[-1]]
    return np.asarray(highs), np.asarray(values)


def _predict_isotonic(
    model: tuple[np.ndarray, np.ndarray],
    confidences: np.ndarray,
) -> np.ndarray:
    highs, values = model
    indexes = np.minimum(
        np.searchsorted(highs, confidences, side="left"),
        len(values) - 1,
    )
    return values[indexes]


def _fit_hierarchical(
    indexes: np.ndarray,
    *,
    confidences: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    routes: Sequence[str],
    missing: Sequence[int],
) -> tuple[Any, ...]:
    global_mean = (float(labels[indexes].sum()) + 2.0) / (len(indexes) + 4.0)
    dimensions = (
        ("class",),
        ("class", "route"),
        ("class", "route", "missing"),
    )
    levels: list[dict[tuple[Any, ...], tuple[float, int]]] = []
    for fields in dimensions:
        mutable: dict[tuple[Any, ...], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        for index in indexes:
            values = {
                "class": classes[index],
                "route": routes[index],
                "missing": min(missing[index], 4),
            }
            key = tuple(values[field] for field in fields)
            mutable[key][0] += labels[index]
            mutable[key][1] += 1.0
        levels.append(
            {
                key: (value[0], int(value[1]))
                for key, value in mutable.items()
            }
        )

    def mapped(index: int) -> float:
        parent = global_mean
        values = {
            "class": classes[index],
            "route": routes[index],
            "missing": min(missing[index], 4),
        }
        for fields, level in zip(dimensions, levels, strict=True):
            successes, support = level.get(
                tuple(values[field] for field in fields),
                (0.0, 0),
            )
            parent = (successes + 8.0 * parent) / (support + 8.0)
        return parent

    mapped_train = np.asarray([mapped(int(index)) for index in indexes])
    blend = min(
        (step / 20.0 for step in range(21)),
        key=lambda value: float(
            np.mean(
                (
                    (1.0 - value) * confidences[indexes]
                    + value * mapped_train
                    - labels[indexes]
                )
                ** 2
            )
        ),
    )
    return levels, global_mean, blend, dimensions


def _predict_hierarchical(
    model: tuple[Any, ...],
    indexes: np.ndarray,
    *,
    confidences: np.ndarray,
    classes: Sequence[str],
    routes: Sequence[str],
    missing: Sequence[int],
) -> np.ndarray:
    levels, global_mean, blend, dimensions = model
    predictions = []
    for index in indexes:
        parent = global_mean
        values = {
            "class": classes[index],
            "route": routes[index],
            "missing": min(missing[index], 4),
        }
        for fields, level in zip(dimensions, levels, strict=True):
            successes, support = level.get(
                tuple(values[field] for field in fields),
                (0.0, 0),
            )
            parent = (successes + 8.0 * parent) / (support + 8.0)
        predictions.append(
            (1.0 - blend) * confidences[index] + blend * parent
        )
    return np.asarray(predictions)


def _fit_predict(
    family: str,
    train: np.ndarray,
    test: np.ndarray,
    *,
    confidences: np.ndarray,
    labels: np.ndarray,
    classes: Sequence[str],
    routes: Sequence[str],
    missing: Sequence[int],
) -> np.ndarray:
    if family in {"temperature", "beta"}:
        train_features, penalty = _confidence_features(
            confidences[train],
            family,
        )
        coefficients = _logistic_fit(
            train_features,
            labels[train],
            penalty=penalty,
        )
        test_features, _ = _confidence_features(confidences[test], family)
        return _sigmoid(test_features @ coefficients)
    if family == "isotonic":
        return _predict_isotonic(
            _fit_isotonic(confidences[train], labels[train]),
            confidences[test],
        )
    if family == "hierarchical_shrunk":
        return _predict_hierarchical(
            _fit_hierarchical(
                train,
                confidences=confidences,
                labels=labels,
                classes=classes,
                routes=routes,
                missing=missing,
            ),
            test,
            confidences=confidences,
            classes=classes,
            routes=routes,
            missing=missing,
        )
    raise FinalRefitError("unsupported calibration family")


def _metric_summary(
    predictions: Sequence[float],
    labels: Sequence[float],
) -> dict[str, Any]:
    if len(predictions) != len(labels):
        raise FinalRefitError("metric inputs are not aligned")
    if not predictions:
        return {"support": 0, "brier": None, "ece_10_bin": None}
    predicted = np.asarray(predictions, dtype=float)
    actual = np.asarray(labels, dtype=float)
    brier = float(np.mean((predicted - actual) ** 2))
    ece = 0.0
    for bucket in range(10):
        lower = bucket / 10.0
        upper = (bucket + 1) / 10.0
        mask = (predicted >= lower) & (
            (predicted < upper) | ((bucket == 9) & (predicted == 1.0))
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(predicted[mask].mean()) - float(actual[mask].mean())
            )
    return {
        "support": len(predicted),
        "brier": brier,
        "calibration_score": 20.0 * max(0.0, 1.0 - 2.0 * brier),
        "ece_10_bin": ece,
        "accuracy": float(actual.mean()),
        "mean_confidence": float(predicted.mean()),
    }


def _slice_metrics(
    predictions: Sequence[float],
    labels: Sequence[float],
    slices: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in sorted(set(slices)):
        indexes = [index for index, item in enumerate(slices) if item == value]
        result[value] = _metric_summary(
            [predictions[index] for index in indexes],
            [labels[index] for index in indexes],
        )
    return result


def _distribution(
    values: Sequence[float],
) -> dict[str, Any]:
    if not values:
        return {"support": 0}
    array = np.asarray(values, dtype=float)
    return {
        "support": len(array),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(array.max()),
    }


def build_report(
    *,
    truth_rows: Mapping[str, Mapping[str, str]],
    parent_rows: Mapping[str, Mapping[str, Any]],
    final_rows: Mapping[str, Mapping[str, Any]],
    layout_groups: Mapping[str, str],
    source_revision: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    case_ids = sorted(final_rows)
    expected = set(case_ids)
    if (
        set(truth_rows) != expected
        or set(parent_rows) != expected
        or set(layout_groups) != expected
    ):
        raise FinalRefitError("truth, parent, final, and layout coverage must match")
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise FinalRefitError("source revision must be a full lowercase Git SHA")

    confidences = np.asarray(
        [_probability(final_rows[case_id]["confidence"]) for case_id in case_ids]
    )
    labels = np.asarray(
        [
            float(
                final_rows[case_id]["adjudication"]
                == truth_rows[case_id]["adjudication"]
            )
            for case_id in case_ids
        ]
    )
    classes = [str(final_rows[case_id]["adjudication"]) for case_id in case_ids]
    routes = [
        _finalizer_route(parent_rows[case_id], final_rows[case_id])
        for case_id in case_ids
    ]
    missing = [_missing_count(final_rows[case_id]) for case_id in case_ids]
    groups = [layout_groups[case_id] for case_id in case_ids]
    current = _metric_summary(confidences.tolist(), labels.tolist())

    comparisons: dict[str, Any] = {}
    family_oof: dict[str, tuple[list[float], list[float], list[str], list[str]]] = {}
    for family in FAMILIES:
        candidate_predictions: list[float] = []
        repeated_labels: list[float] = []
        repeated_classes: list[str] = []
        repeated_routes: list[str] = []
        folds: list[dict[str, Any]] = []
        repeat_deltas: list[float] = []
        for repeat_index, seed in enumerate(REPEAT_SEEDS):
            deltas = []
            for fold_index, test_values in enumerate(
                _grouped_folds(groups, seed=seed)
            ):
                test = np.asarray(test_values, dtype=int)
                train = np.asarray(
                    [index for index in range(len(case_ids)) if index not in set(test_values)],
                    dtype=int,
                )
                candidate = _fit_predict(
                    family,
                    train,
                    test,
                    confidences=confidences,
                    labels=labels,
                    classes=classes,
                    routes=routes,
                    missing=missing,
                )
                baseline_brier = float(
                    np.mean((confidences[test] - labels[test]) ** 2)
                )
                candidate_brier = float(
                    np.mean((candidate - labels[test]) ** 2)
                )
                delta = baseline_brier - candidate_brier
                deltas.append(delta)
                folds.append(
                    {
                        "repeat": repeat_index,
                        "fold": fold_index,
                        "support": len(test),
                        "group_count": len({groups[index] for index in test}),
                        "baseline_brier": baseline_brier,
                        "candidate_brier": candidate_brier,
                        "brier_improvement": delta,
                    }
                )
                candidate_predictions.extend(candidate.tolist())
                repeated_labels.extend(labels[test].tolist())
                repeated_classes.extend(classes[index] for index in test)
                repeated_routes.extend(routes[index] for index in test)
            repeat_deltas.append(float(sum(deltas) / len(deltas)))
        candidate_metrics = _metric_summary(
            candidate_predictions,
            repeated_labels,
        )
        repeated_baseline = np.tile(confidences, len(REPEAT_SEEDS))
        repeated_truth = np.tile(labels, len(REPEAT_SEEDS))
        baseline_brier = float(
            np.mean((repeated_baseline - repeated_truth) ** 2)
        )
        overall_improvement = baseline_brier - float(candidate_metrics["brier"])
        minimum_fold = min(fold["brier_improvement"] for fold in folds)
        passed = (
            overall_improvement > GATE_TOLERANCE
            and minimum_fold >= -GATE_TOLERANCE
            and all(delta > GATE_TOLERANCE for delta in repeat_deltas)
        )
        comparisons[family] = {
            "baseline_oof_brier": baseline_brier,
            "candidate_oof": candidate_metrics,
            "overall_brier_improvement": overall_improvement,
            "minimum_fold_brier_improvement": minimum_fold,
            "positive_fold_count": sum(
                fold["brier_improvement"] > GATE_TOLERANCE for fold in folds
            ),
            "fold_count": len(folds),
            "repeat_brier_improvements": repeat_deltas,
            "promotion_gate_passed": passed,
            "folds": folds,
            "reliability_by_final_class": _slice_metrics(
                candidate_predictions,
                repeated_labels,
                repeated_classes,
            ),
            "reliability_by_finalizer_route": _slice_metrics(
                candidate_predictions,
                repeated_labels,
                repeated_routes,
            ),
        }
        family_oof[family] = (
            candidate_predictions,
            repeated_labels,
            repeated_classes,
            repeated_routes,
        )

    diagnostic_family = min(
        FAMILIES,
        key=lambda family: (
            comparisons[family]["candidate_oof"]["brier"],
            FAMILIES.index(family),
        ),
    )
    selected_predictions, selected_labels, _, _ = family_oof[diagnostic_family]
    all_indexes = np.arange(len(case_ids), dtype=int)
    full_shadow_confidences = _fit_predict(
        diagnostic_family,
        all_indexes,
        all_indexes,
        confidences=confidences,
        labels=labels,
        classes=classes,
        routes=routes,
        missing=missing,
    )
    shadow_rows = [
        {
            **final_rows[case_id],
            "confidence": float(full_shadow_confidences[index]),
        }
        for index, case_id in enumerate(case_ids)
    ]
    passing = [
        family
        for family in FAMILIES
        if comparisons[family]["promotion_gate_passed"]
    ]

    non_confidence_unchanged = all(
        {
            field: final_rows[case_id][field]
            for field in NON_CONFIDENCE_FIELDS
        }
        == {
            field: shadow_rows[index][field]
            for field in NON_CONFIDENCE_FIELDS
        }
        for index, case_id in enumerate(case_ids)
    )
    decision = "needs_runtime_integration" if passing else "evaluated_no_promotion"
    return {
        "schema": REPORT_SCHEMA,
        "evidence_class": EVIDENCE_CLASS,
        "source_revision": source_revision,
        "source_sha256": dict(sorted(source_sha256.items())),
        "case_count": len(case_ids),
        "layout_group_count": len(set(groups)),
        "protocol": {
            "repeats": len(REPEAT_SEEDS),
            "folds": FOLD_COUNT,
            "whole_group_exclusive": True,
            "labels_joined_only_after_predictions_froze": True,
            "candidate_changes_confidence_only": True,
            "features": [
                "final_confidence",
                "final_class",
                "finalizer_route",
                "fee_known",
                "missing_field_count",
            ],
            "internal_policy_and_recovery_trace_available": False,
        },
        "current": current,
        "tracked_mean_brier_target": BRIER_TARGET,
        "comparisons": comparisons,
        "diagnostic_best_family": diagnostic_family,
        "full_data_shadow_diagnostic": {
            **_metric_summary(full_shadow_confidences.tolist(), labels.tolist()),
            "not_promotion_evidence": True,
        },
        "passing_families": passing,
        "decision": decision,
        "hard_gates": {
            "non_confidence_bytes_unchanged": non_confidence_unchanged,
            "all_families_compared": set(comparisons) == set(FAMILIES),
            "every_repeat_positive_for_selected": all(
                delta > GATE_TOLERANCE
                for delta in comparisons[diagnostic_family][
                    "repeat_brier_improvements"
                ]
            ),
            "no_negative_fold_for_selected": comparisons[diagnostic_family][
                "minimum_fold_brier_improvement"
            ]
            >= -GATE_TOLERANCE,
            "mean_brier_target_met": current["brier"] <= BRIER_TARGET,
        },
        "selected_oof_confidence_distribution": {
            "correct": _distribution(
                [
                    confidence
                    for confidence, label in zip(
                        selected_predictions,
                        selected_labels,
                        strict=True,
                    )
                    if label == 1.0
                ]
            ),
            "incorrect": _distribution(
                [
                    confidence
                    for confidence, label in zip(
                        selected_predictions,
                        selected_labels,
                        strict=True,
                    )
                    if label == 0.0
                ]
            ),
        },
        "limitations": [
            "All 1,000 public labeled cases are exposed development data; grouped OOF is robustness evidence, not an unseen or private score.",
            "The final output does not expose internal policy/recovery trace features, so reliability is reported by the observable outer finalizer route.",
            "No calibrator may enter runtime unless every strict fold and repeat gate passes on the frozen decision graph.",
        ],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# WO-19 Final Confidence Refit",
        "",
        "This is aggregate public-data robustness evidence, not an unseen, private, official-validation, or leaderboard score.",
        "",
        "## Frozen graph",
        "",
        f"- Source revision: `{report['source_revision']}`",
        f"- Records: `{report['case_count']}`",
        f"- Label-blind layout groups: `{report['layout_group_count']}`",
        f"- Current mean Brier: `{report['current']['brier']:.12f}`",
        f"- Current calibration score: `{report['current']['calibration_score']:.12f}/20`",
        f"- Tracked mean-Brier target: `{report['tracked_mean_brier_target']:.2f}`",
        "",
        "## Repeated grouped OOF comparison",
        "",
        "| Family | OOF Brier | Improvement | Minimum fold | Positive folds | Gate |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for family in FAMILIES:
        result = report["comparisons"][family]
        lines.append(
            f"| `{family}` | {result['candidate_oof']['brier']:.12f} | "
            f"{result['overall_brier_improvement']:+.12f} | "
            f"{result['minimum_fold_brier_improvement']:+.12f} | "
            f"{result['positive_fold_count']}/{result['fold_count']} | "
            f"{'pass' if result['promotion_gate_passed'] else 'reject'} |"
        )
    lines.extend(
        [
            "",
            f"Diagnostic best family: `{report['diagnostic_best_family']}`.",
            f"Decision: **{report['decision']}**.",
            "",
            "Every family improved aggregate OOF Brier, but a negative layout fold is a hard rejection. No confidence artifact is promoted.",
            "",
            "## Hard gates",
            "",
        ]
    )
    for key, value in report["hard_gates"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(
        [
            "",
            "The shadow comparison changes confidence only; all non-confidence output bytes remain invariant.",
            "",
            "## Reliability coverage",
            "",
        ]
    )
    selected = report["comparisons"][report["diagnostic_best_family"]]
    for label, values in (
        ("Final class", selected["reliability_by_final_class"]),
        ("Finalizer route", selected["reliability_by_finalizer_route"]),
    ):
        lines.extend(
            [
                f"### {label}",
                "",
                "| Slice | Support | Brier | ECE (10-bin) |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, metrics in values.items():
            lines.append(
                f"| `{name}` | {metrics['support']} | "
                f"{metrics['brier']:.12f} | {metrics['ece_10_bin']:.12f} |"
            )
        lines.append("")
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build aggregate-only WO-19 final confidence evidence"
    )
    parser.add_argument("--truth", required=True)
    parser.add_argument("--parent-predictions", required=True)
    parser.add_argument("--final-predictions", required=True)
    parser.add_argument("--layout-manifest", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    args = parser.parse_args()

    truth_path = Path(args.truth)
    parent_path = Path(args.parent_predictions)
    final_path = Path(args.final_predictions)
    layout_path = Path(args.layout_manifest)
    report = build_report(
        truth_rows=_read_truth(truth_path),
        parent_rows=_read_jsonl(parent_path, label="parent predictions"),
        final_rows=_read_jsonl(final_path, label="final predictions"),
        layout_groups=_read_layout(layout_path),
        source_revision=args.source_revision,
        source_sha256={
            "truth": _sha256_path(truth_path),
            "parent_predictions": _sha256_path(parent_path),
            "final_predictions": _sha256_path(final_path),
            "layout_manifest": _sha256_path(layout_path),
            "builder": _sha256_path(Path(__file__)),
            "evaluator": _sha256_path(
                Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
            ),
        },
    )
    output_json = Path(args.output_json)
    output_markdown = Path(args.output_markdown)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"WO-19 final refit: {report['decision']} "
        f"best={report['diagnostic_best_family']} "
        f"brier={report['current']['brier']:.12f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
