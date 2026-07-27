"""Deterministic, identity-free confidence calibration candidates.

This module contains the training and runtime portions of the WO19 comparison.
Every candidate consumes the same closed :class:`FinalConfidenceContext`
contract.  Fitted runtime artifacts contain only fixed feature names, generic
enumeration values, and numeric parameters; training labels and grouping keys
never cross the artifact boundary.
"""

from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .final_confidence import (
    FINAL_CLASSES,
    FINAL_POLICY_ROUTES,
    FINAL_RECOVERY_ROUTES,
    FinalConfidenceContext,
)


CALIBRATOR_SCHEMA = "mib-confidence-refit-calibrator/v1"
CALIBRATION_FAMILIES = (
    "temperature",
    "beta",
    "hierarchical_shrunk",
    "isotonic",
)
PROBABILITY_EPSILON = 1e-6
BETA_L2_STRENGTH = 1.0
HIERARCHICAL_SHRINKAGE = 4.0
EVIDENCE_STATES = (
    "clean",
    "conflict",
    "incomplete",
    "model_uncertain",
    "ocr_disagreement",
    "resolution_uncertain",
)

CONFIDENCE_FEATURE_ORDER = (
    "bias",
    "input_confidence",
    "log_input_confidence",
    "negative_log_one_minus_input_confidence",
    *(f"final_class={value}" for value in FINAL_CLASSES),
    *(f"policy_route={value}" for value in FINAL_POLICY_ROUTES),
    "authoritative",
    "visible_completeness",
    "has_conflict",
    "ocr_disagreement",
    *(f"recovery_route={value}" for value in FINAL_RECOVERY_ROUTES),
    "model_diagnostics_missing",
    "model_margin",
    "ensemble_agreement",
    "resolution_entropy",
)

_SENSITIVE_VALUE = re.compile(
    r"(?:\bMIB-[0-9]{6}\b|\bSPN-[0-9]{4}\b|"
    r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b|\.pdf\b|"
    r"^/(?:Users|private|tmp)/|^[A-Za-z]:\\|^\\\\)",
    re.IGNORECASE,
)


class ConfidenceRefitError(ValueError):
    """Raised when a candidate or its identity-free inputs are malformed."""


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    rendered = float(value)
    if not math.isfinite(rendered) or not 0.0 <= rendered <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return rendered


def _clipped_probability(value: float) -> float:
    return min(1.0 - PROBABILITY_EPSILON, max(PROBABILITY_EPSILON, value))


def _logit(value: float) -> float:
    probability = _clipped_probability(value)
    return math.log(probability) - math.log1p(-probability)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        factor = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + factor)
    factor = math.exp(max(value, -700.0))
    return factor / (1.0 + factor)


def identity_free_feature_vector(
    input_confidence: float,
    context: FinalConfidenceContext,
) -> tuple[float, ...]:
    """Return the one and only ordered WO19 runtime feature vector."""

    probability = _probability(input_confidence, name="input_confidence")
    if not isinstance(context, FinalConfidenceContext):
        raise TypeError("context must be FinalConfidenceContext")
    clipped = _clipped_probability(probability)
    diagnostics_missing = context.model_margin is None
    values = (
        1.0,
        probability,
        math.log(clipped),
        -math.log1p(-clipped),
        *(float(context.final_class == value) for value in FINAL_CLASSES),
        *(float(context.policy_route == value) for value in FINAL_POLICY_ROUTES),
        float(context.authoritative),
        context.visible_completeness,
        float(context.has_conflict),
        context.ocr_disagreement,
        *(float(context.recovery_route == value) for value in FINAL_RECOVERY_ROUTES),
        float(diagnostics_missing),
        0.0 if context.model_margin is None else context.model_margin,
        (
            0.0
            if context.ensemble_agreement is None
            else context.ensemble_agreement
        ),
        context.resolution_entropy,
    )
    if len(values) != len(CONFIDENCE_FEATURE_ORDER):
        raise AssertionError("confidence feature contract length drifted")
    if any(not math.isfinite(value) for value in values):
        raise ConfidenceRefitError("confidence feature vector is not finite")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class CalibrationExample:
    """A label-bearing training-only record."""

    input_confidence: float
    correct: bool
    context: FinalConfidenceContext

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_confidence",
            _probability(self.input_confidence, name="input_confidence"),
        )
        if not isinstance(self.correct, bool):
            raise TypeError("correct must be a boolean")
        if not isinstance(self.context, FinalConfidenceContext):
            raise TypeError("context must be FinalConfidenceContext")
        identity_free_feature_vector(self.input_confidence, self.context)


def _contains_sensitive_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_sensitive_value(key) or _contains_sensitive_value(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_sensitive_value(child) for child in value)
    return isinstance(value, str) and _SENSITIVE_VALUE.search(value) is not None


def _validate_finite_tree(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ConfidenceRefitError("calibrator artifact contains non-finite values")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfidenceRefitError("calibrator artifact keys must be strings")
            _validate_finite_tree(child)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _validate_finite_tree(child)
        return
    raise ConfidenceRefitError("calibrator artifact contains unsupported values")


def _is_exact_numeric_constant(value: object, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == expected
    )


def _evidence_state(context: FinalConfidenceContext) -> str:
    if context.has_conflict:
        return "conflict"
    if context.visible_completeness < 0.75:
        return "incomplete"
    if context.ocr_disagreement >= 0.25:
        return "ocr_disagreement"
    if context.model_margin is not None and (
        context.model_margin < 0.20 or context.ensemble_agreement < 0.67
    ):
        return "model_uncertain"
    if context.resolution_entropy >= 0.50:
        return "resolution_uncertain"
    return "clean"


_HIERARCHICAL_NODE_CONTRACT = (
    ("class_nodes", (frozenset(FINAL_CLASSES),)),
    (
        "policy_nodes",
        (frozenset(FINAL_CLASSES), frozenset(FINAL_POLICY_ROUTES)),
    ),
    (
        "recovery_nodes",
        (
            frozenset(FINAL_CLASSES),
            frozenset(FINAL_POLICY_ROUTES),
            frozenset(FINAL_RECOVERY_ROUTES),
        ),
    ),
    (
        "evidence_nodes",
        (
            frozenset(FINAL_CLASSES),
            frozenset(FINAL_POLICY_ROUTES),
            frozenset(FINAL_RECOVERY_ROUTES),
            frozenset(EVIDENCE_STATES),
        ),
    ),
)


@dataclass(frozen=True)
class FittedConfidenceCalibrator:
    """Validated runtime calibrator with no training identities or labels."""

    family: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.family not in CALIBRATION_FAMILIES:
            raise ConfidenceRefitError("unsupported confidence calibration family")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("calibrator parameters must be a mapping")
        rendered = json.loads(json.dumps(dict(self.parameters), sort_keys=True))
        _validate_finite_tree(rendered)
        if _contains_sensitive_value(rendered):
            raise ConfidenceRefitError(
                "calibrator artifact contains identity-bearing values"
            )
        object.__setattr__(self, "parameters", rendered)
        self._validate_shape()

    def _validate_shape(self) -> None:
        if self.family == "temperature":
            if set(self.parameters) != {"temperature"}:
                raise ConfidenceRefitError("temperature parameters are malformed")
            temperature = self.parameters["temperature"]
            if (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or float(temperature) <= 0.0
            ):
                raise ConfidenceRefitError("temperature must be positive")
            return
        if self.family == "isotonic":
            if set(self.parameters) != {"thresholds", "values"}:
                raise ConfidenceRefitError("isotonic parameters are malformed")
            thresholds = self.parameters["thresholds"]
            values = self.parameters["values"]
            if (
                not isinstance(thresholds, list)
                or not thresholds
                or not isinstance(values, list)
                or len(thresholds) != len(values)
                or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in [*thresholds, *values]
                )
                or any(
                    float(left) >= float(right)
                    for left, right in zip(thresholds, thresholds[1:])
                )
                or any(
                    float(left) > float(right)
                    for left, right in zip(values, values[1:])
                )
                or any(
                    not 0.0 <= float(value) <= 1.0 for value in thresholds
                )
                or any(not 0.0 <= float(value) <= 1.0 for value in values)
            ):
                raise ConfidenceRefitError("isotonic map is not monotone")
            return
        if self.family == "beta":
            if set(self.parameters) != {"coefficients", "l2_strength"}:
                raise ConfidenceRefitError("beta parameters are malformed")
            coefficients = self.parameters["coefficients"]
            if (
                not isinstance(coefficients, list)
                or len(coefficients) != 3
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in coefficients
                )
                or not _is_exact_numeric_constant(
                    self.parameters["l2_strength"],
                    BETA_L2_STRENGTH,
                )
            ):
                raise ConfidenceRefitError("beta coefficients are malformed")
            return
        expected = {
            "global_probability",
            "shrinkage",
            "class_nodes",
            "policy_nodes",
            "recovery_nodes",
            "evidence_nodes",
        }
        if set(self.parameters) != expected:
            raise ConfidenceRefitError("hierarchical parameters are malformed")
        if not _is_exact_numeric_constant(
            self.parameters["shrinkage"],
            HIERARCHICAL_SHRINKAGE,
        ):
            raise ConfidenceRefitError("hierarchical shrinkage is not frozen")
        _probability(
            self.parameters["global_probability"],
            name="global_probability",
        )
        parent_tokens: set[tuple[str, ...]] = {()}
        for node_name, domains in _HIERARCHICAL_NODE_CONTRACT:
            nodes = self.parameters[node_name]
            if not isinstance(nodes, list):
                raise ConfidenceRefitError("hierarchical nodes must be lists")
            previous: tuple[str, ...] | None = None
            current_tokens: set[tuple[str, ...]] = set()
            for node in nodes:
                if not isinstance(node, dict) or set(node) != {
                    "tokens",
                    "probability",
                    "support",
                }:
                    raise ConfidenceRefitError("hierarchical node is malformed")
                tokens = node["tokens"]
                if (
                    not isinstance(tokens, list)
                    or len(tokens) != len(domains)
                    or any(
                        not isinstance(token, str) or token not in domain
                        for token, domain in zip(tokens, domains, strict=True)
                    )
                ):
                    raise ConfidenceRefitError(
                        "hierarchical node tokens violate the closed contract"
                    )
                token_tuple = tuple(tokens)
                if previous is not None and token_tuple <= previous:
                    raise ConfidenceRefitError("hierarchical nodes are not ordered")
                if token_tuple[:-1] not in parent_tokens:
                    raise ConfidenceRefitError(
                        "hierarchical node has no exact parent ancestry"
                    )
                previous = token_tuple
                current_tokens.add(token_tuple)
                _probability(node["probability"], name="node probability")
                if (
                    isinstance(node["support"], bool)
                    or not isinstance(node["support"], int)
                    or node["support"] <= 0
                ):
                    raise ConfidenceRefitError("hierarchical support is invalid")
            parent_tokens = current_tokens

    def to_runtime_mapping(self) -> dict[str, Any]:
        """Serialize only the fixed contract and fitted numeric parameters."""

        return {
            "schema_version": CALIBRATOR_SCHEMA,
            "family": self.family,
            "feature_order": list(CONFIDENCE_FEATURE_ORDER),
            "parameters": json.loads(json.dumps(self.parameters, sort_keys=True)),
        }

    @classmethod
    def from_runtime_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "FittedConfidenceCalibrator":
        if set(value) != {
            "schema_version",
            "family",
            "feature_order",
            "parameters",
        } or value.get("schema_version") != CALIBRATOR_SCHEMA:
            raise ConfidenceRefitError("unsupported confidence calibrator schema")
        if tuple(value.get("feature_order", ())) != CONFIDENCE_FEATURE_ORDER:
            raise ConfidenceRefitError("confidence feature order is not frozen")
        if _contains_sensitive_value(value):
            raise ConfidenceRefitError(
                "confidence calibrator contains identity-bearing values"
            )
        return cls(
            family=str(value.get("family", "")),
            parameters=value.get("parameters", {}),
        )

    def predict(
        self,
        input_confidence: float,
        context: FinalConfidenceContext,
    ) -> float:
        probability = _probability(input_confidence, name="input_confidence")
        identity_free_feature_vector(probability, context)
        if self.family == "temperature":
            result = _sigmoid(
                _logit(probability) / float(self.parameters["temperature"])
            )
        elif self.family == "isotonic":
            thresholds = [float(value) for value in self.parameters["thresholds"]]
            values = [float(value) for value in self.parameters["values"]]
            index = bisect.bisect_left(thresholds, probability)
            result = values[min(index, len(values) - 1)]
        elif self.family == "beta":
            coefficients = [
                float(value) for value in self.parameters["coefficients"]
            ]
            clipped = _clipped_probability(probability)
            beta_features = (
                1.0,
                math.log(clipped),
                -math.log1p(-clipped),
            )
            result = _sigmoid(
                sum(
                    coefficient * feature
                    for coefficient, feature in zip(
                        coefficients,
                        beta_features,
                        strict=True,
                    )
                )
            )
        else:
            tokens = (
                context.final_class,
                context.policy_route,
                context.recovery_route,
                _evidence_state(context),
            )
            result = float(self.parameters["global_probability"])
            for node_name, depth in (
                ("class_nodes", 1),
                ("policy_nodes", 2),
                ("recovery_nodes", 3),
                ("evidence_nodes", 4),
            ):
                target = list(tokens[:depth])
                match = next(
                    (
                        node
                        for node in self.parameters[node_name]
                        if node["tokens"] == target
                    ),
                    None,
                )
                if match is None:
                    break
                result = float(match["probability"])
        if not math.isfinite(result):
            raise ConfidenceRefitError("calibrator produced a non-finite value")
        return _clipped_probability(result)


def _mean_brier(
    predictions: Sequence[float],
    labels: Sequence[float],
) -> float:
    if not predictions:
        return 0.0
    return sum(
        (prediction - label) ** 2
        for prediction, label in zip(predictions, labels, strict=True)
    ) / len(predictions)


def _fit_temperature(
    examples: Sequence[CalibrationExample],
) -> FittedConfidenceCalibrator:
    if not examples:
        return FittedConfidenceCalibrator(
            family="temperature",
            parameters={"temperature": 1.0},
        )
    labels = [float(example.correct) for example in examples]
    logits = [_logit(example.input_confidence) for example in examples]

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        predictions = [_sigmoid(logit / temperature) for logit in logits]
        return _mean_brier(predictions, labels)

    lower = -3.0
    upper = 3.0
    best_log_temperature = 0.0
    best_key = (objective(0.0), abs(0.0), 0.0)
    for _ in range(8):
        step = (upper - lower) / 20.0
        candidates = [lower + index * step for index in range(21)]
        for candidate in candidates:
            key = (objective(candidate), abs(candidate), candidate)
            if key < best_key:
                best_key = key
                best_log_temperature = candidate
        lower = max(-3.0, best_log_temperature - step)
        upper = min(3.0, best_log_temperature + step)
    return FittedConfidenceCalibrator(
        family="temperature",
        parameters={"temperature": math.exp(best_log_temperature)},
    )


def _fit_isotonic(
    examples: Sequence[CalibrationExample],
) -> FittedConfidenceCalibrator:
    if not examples:
        return FittedConfidenceCalibrator(
            family="isotonic",
            parameters={"thresholds": [1.0], "values": [0.5]},
        )
    correct_count = sum(example.correct for example in examples)
    if correct_count in {0, len(examples)}:
        probability = (correct_count + 1.0) / (len(examples) + 2.0)
        return FittedConfidenceCalibrator(
            family="isotonic",
            parameters={"thresholds": [1.0], "values": [probability]},
        )
    aggregated: list[list[float]] = []
    for example in sorted(
        examples,
        key=lambda item: (item.input_confidence, int(item.correct)),
    ):
        if aggregated and aggregated[-1][0] == example.input_confidence:
            aggregated[-1][1] += float(example.correct)
            aggregated[-1][2] += 1.0
        else:
            aggregated.append(
                [example.input_confidence, float(example.correct), 1.0]
            )
    blocks: list[list[float]] = []
    for threshold, successes, weight in aggregated:
        blocks.append([threshold, successes, weight])
        while (
            len(blocks) >= 2
            and blocks[-2][1] / blocks[-2][2]
            > blocks[-1][1] / blocks[-1][2]
        ):
            right = blocks.pop()
            left = blocks.pop()
            blocks.append(
                [
                    right[0],
                    left[1] + right[1],
                    left[2] + right[2],
                ]
            )
    return FittedConfidenceCalibrator(
        family="isotonic",
        parameters={
            "thresholds": [block[0] for block in blocks],
            "values": [
                _clipped_probability(block[1] / block[2]) for block in blocks
            ],
        },
    )


def _solve_linear_system(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    size = len(vector)
    augmented = [
        [float(value) for value in row] + [float(vector[index])]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: (abs(augmented[row][column]), -row),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            return [0.0] * size
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                left - factor * right
                for left, right in zip(
                    augmented[row],
                    augmented[column],
                    strict=True,
                )
            ]
    return [augmented[index][-1] for index in range(size)]


def _beta_objective(
    coefficients: Sequence[float],
    rows: Sequence[Sequence[float]],
    labels: Sequence[float],
) -> float:
    objective = 0.0
    for row, label in zip(rows, labels, strict=True):
        score = sum(
            coefficient * value
            for coefficient, value in zip(coefficients, row, strict=True)
        )
        softplus = (
            score + math.log1p(math.exp(-score))
            if score >= 0.0
            else math.log1p(math.exp(score))
        )
        objective += softplus - label * score
    objective += 0.5 * BETA_L2_STRENGTH * sum(
        coefficient * coefficient for coefficient in coefficients[1:]
    )
    return objective


def _fit_beta(
    examples: Sequence[CalibrationExample],
) -> FittedConfidenceCalibrator:
    if not examples:
        coefficients = [0.0, 1.0, 1.0]
    else:
        correct_count = sum(example.correct for example in examples)
        if correct_count in {0, len(examples)}:
            prior = (correct_count + 1.0) / (len(examples) + 2.0)
            coefficients = [_logit(prior), 0.0, 0.0]
        else:
            rows = []
            labels = []
            for example in examples:
                probability = _clipped_probability(example.input_confidence)
                rows.append(
                    [1.0, math.log(probability), -math.log1p(-probability)]
                )
                labels.append(float(example.correct))
            coefficients = [_logit(correct_count / len(examples)), 0.0, 0.0]
            for _ in range(75):
                predictions = [
                    _sigmoid(
                        sum(
                            coefficient * value
                            for coefficient, value in zip(
                                coefficients,
                                row,
                                strict=True,
                            )
                        )
                    )
                    for row in rows
                ]
                gradient = [
                    sum(
                        row[column] * (label - prediction)
                        for row, label, prediction in zip(
                            rows,
                            labels,
                            predictions,
                            strict=True,
                        )
                    )
                    - (
                        BETA_L2_STRENGTH * coefficients[column]
                        if column
                        else 0.0
                    )
                    for column in range(3)
                ]
                hessian = [
                    [
                        sum(
                            row[left]
                            * row[right]
                            * prediction
                            * (1.0 - prediction)
                            for row, prediction in zip(
                                rows,
                                predictions,
                                strict=True,
                            )
                        )
                        + (
                            BETA_L2_STRENGTH
                            if left == right and left
                            else 0.0
                        )
                        + (1e-9 if left == right else 0.0)
                        for right in range(3)
                    ]
                    for left in range(3)
                ]
                update = _solve_linear_system(hessian, gradient)
                if max(abs(value) for value in update) < 1e-10:
                    break
                directional_improvement = sum(
                    gradient_value * update_value
                    for gradient_value, update_value in zip(
                        gradient,
                        update,
                        strict=True,
                    )
                )
                if (
                    not math.isfinite(directional_improvement)
                    or directional_improvement <= 0.0
                ):
                    break
                current_objective = _beta_objective(
                    coefficients,
                    rows,
                    labels,
                )
                step = 1.0
                accepted: list[float] | None = None
                for _ in range(30):
                    candidate = [
                        coefficient + step * update_value
                        for coefficient, update_value in zip(
                            coefficients,
                            update,
                            strict=True,
                        )
                    ]
                    if all(math.isfinite(value) for value in candidate):
                        candidate_objective = _beta_objective(
                            candidate,
                            rows,
                            labels,
                        )
                        if candidate_objective <= (
                            current_objective
                            - 1e-4 * step * directional_improvement
                        ):
                            accepted = candidate
                            break
                    step *= 0.5
                if accepted is None:
                    break
                coefficients = accepted
    return FittedConfidenceCalibrator(
        family="beta",
        parameters={
            "coefficients": coefficients,
            "l2_strength": BETA_L2_STRENGTH,
        },
    )


def _fit_hierarchical(
    examples: Sequence[CalibrationExample],
) -> FittedConfidenceCalibrator:
    correct_count = sum(example.correct for example in examples)
    global_probability = (correct_count + 1.0) / (len(examples) + 2.0)
    levels = (
        ("class_nodes", lambda example: (example.context.final_class,)),
        (
            "policy_nodes",
            lambda example: (
                example.context.final_class,
                example.context.policy_route,
            ),
        ),
        (
            "recovery_nodes",
            lambda example: (
                example.context.final_class,
                example.context.policy_route,
                example.context.recovery_route,
            ),
        ),
        (
            "evidence_nodes",
            lambda example: (
                example.context.final_class,
                example.context.policy_route,
                example.context.recovery_route,
                _evidence_state(example.context),
            ),
        ),
    )
    parent_probabilities: dict[tuple[str, ...], float] = {
        (): global_probability
    }
    parameters: dict[str, Any] = {
        "global_probability": global_probability,
        "shrinkage": HIERARCHICAL_SHRINKAGE,
    }
    for node_name, token_builder in levels:
        counts: dict[tuple[str, ...], list[int]] = {}
        for example in examples:
            tokens = token_builder(example)
            bucket = counts.setdefault(tokens, [0, 0])
            bucket[0] += int(example.correct)
            bucket[1] += 1
        nodes = []
        for tokens in sorted(counts):
            successes, support = counts[tokens]
            parent = parent_probabilities[tokens[:-1]]
            probability = (
                successes + HIERARCHICAL_SHRINKAGE * parent
            ) / (support + HIERARCHICAL_SHRINKAGE)
            parent_probabilities[tokens] = probability
            nodes.append(
                {
                    "tokens": list(tokens),
                    "probability": probability,
                    "support": support,
                }
            )
        parameters[node_name] = nodes
    return FittedConfidenceCalibrator(
        family="hierarchical_shrunk",
        parameters=parameters,
    )


def fit_confidence_calibrator(
    family: str,
    examples: Iterable[CalibrationExample],
) -> FittedConfidenceCalibrator:
    """Fit one deterministic candidate, including sparse-fold fallbacks."""

    if family not in CALIBRATION_FAMILIES:
        raise ConfidenceRefitError("unsupported confidence calibration family")
    supplied_examples = tuple(examples)
    if any(not isinstance(example, CalibrationExample) for example in supplied_examples):
        raise TypeError("examples must contain CalibrationExample values")
    frozen_examples = tuple(
        sorted(
            supplied_examples,
            key=lambda example: (
                identity_free_feature_vector(
                    example.input_confidence,
                    example.context,
                ),
                int(example.correct),
            ),
        )
    )
    fitters = {
        "temperature": _fit_temperature,
        "isotonic": _fit_isotonic,
        "beta": _fit_beta,
        "hierarchical_shrunk": _fit_hierarchical,
    }
    return fitters[family](frozen_examples)
