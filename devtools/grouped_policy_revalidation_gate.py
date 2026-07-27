"""Strict aggregate gate for one grouped WO-17 policy experiment.

The gate consumes no case identity.  It requires two deterministic full
production captures per arm and scores those already-frozen predictions over
exactly three repeats of five layout-group-exclusive validation folds.

The historical ``forced_approval_count`` is retained as a separately reported
legacy counter.  It is never silently renamed.  A capture-time observer uses
the production class's exact frozen matcher to classify an observed initial
review-to-approval transition as guarded or unguarded.  The gate requires all
legacy forced events to be accounted for by that observer, at least one
guarded candidate event, and zero unguarded or late-revalidation events.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from devtools.evaluation import PUBLIC_ROBUSTNESS_EVIDENCE
from devtools.experiment_control import (
    ExperimentControlError,
    require_aggregate_only,
)


REQUIRED_REPEATS = 3
REQUIRED_FOLDS = 5
PUBLIC_EVIDENCE_LABEL = PUBLIC_ROBUSTNESS_EVIDENCE
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class GroupedPolicyGateError(ExperimentControlError):
    """Grouped policy evidence is malformed or internally inconsistent."""


def _finite_score(name: str, value: Any) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise GroupedPolicyGateError(f"{name} must be a finite score")
    return float(value)


def _count(name: str, value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        qualifier = "positive" if positive else "non-negative"
        raise GroupedPolicyGateError(
            f"{name} must be a {qualifier} integer"
        )
    return value


def _flag(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise GroupedPolicyGateError(f"{name} must be a boolean")
    return value


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _float(value: Decimal) -> float:
    return 0.0 if value == 0 else float(value)


@dataclass(frozen=True)
class PolicyArmAggregate:
    """Official-evaluator and capture facts for one full prediction arm."""

    total_score: float
    extraction_score: float
    classification_score: float
    calibration_score: float
    missing_penalty: float
    record_count: int
    catastrophic_false_approvals: int
    false_approvals: int
    missing_records: int
    invalid_records: int
    duplicate_records: int
    extra_records: int
    deterministic: bool

    def __post_init__(self) -> None:
        for name in (
            "total_score",
            "extraction_score",
            "classification_score",
            "calibration_score",
            "missing_penalty",
        ):
            object.__setattr__(
                self, name, _finite_score(name, getattr(self, name))
            )
        for name in (
            "record_count",
            "catastrophic_false_approvals",
            "false_approvals",
            "missing_records",
            "invalid_records",
            "duplicate_records",
            "extra_records",
        ):
            object.__setattr__(
                self, name, _count(name, getattr(self, name))
            )
        object.__setattr__(
            self,
            "deterministic",
            _flag("deterministic", self.deterministic),
        )
        if self.missing_penalty < 0:
            raise GroupedPolicyGateError(
                "missing_penalty must be non-negative"
            )
        expected_total = (
            self.extraction_score
            + self.classification_score
            + self.calibration_score
            - self.missing_penalty
        )
        if abs(self.total_score - expected_total) > 1e-9:
            raise GroupedPolicyGateError(
                "arm total_score must equal its three score components"
            )


@dataclass(frozen=True)
class PolicyFoldPair:
    """One paired candidate/control score over identical validation members."""

    repeat: int
    fold: int
    record_count: int
    layout_group_count: int
    control_score: float
    candidate_score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "repeat", _count("repeat", self.repeat))
        object.__setattr__(self, "fold", _count("fold", self.fold))
        object.__setattr__(
            self,
            "record_count",
            _count("record_count", self.record_count, positive=True),
        )
        object.__setattr__(
            self,
            "layout_group_count",
            _count(
                "layout_group_count",
                self.layout_group_count,
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "control_score",
            _finite_score("control_score", self.control_score),
        )
        object.__setattr__(
            self,
            "candidate_score",
            _finite_score("candidate_score", self.candidate_score),
        )
        if self.layout_group_count > self.record_count:
            raise GroupedPolicyGateError(
                "fold layout-group count cannot exceed its record count"
            )

    @property
    def delta_decimal(self) -> Decimal:
        return _decimal(self.candidate_score) - _decimal(
            self.control_score
        )

    @property
    def delta(self) -> float:
        return _float(self.delta_decimal)


@dataclass(frozen=True)
class PolicyActivityAggregate:
    """Exact-matcher activity plus the disclosed legacy forced counter."""

    eligible_guarded_initial_count: int
    guarded_initial_approval_count: int
    unguarded_initial_approval_count: int
    late_revalidation_approval_count: int
    legacy_forced_approval_count: int

    def __post_init__(self) -> None:
        for name in (
            "eligible_guarded_initial_count",
            "guarded_initial_approval_count",
            "unguarded_initial_approval_count",
            "late_revalidation_approval_count",
            "legacy_forced_approval_count",
        ):
            object.__setattr__(
                self, name, _count(name, getattr(self, name))
            )


@dataclass(frozen=True)
class GroupedPolicyEvidence:
    """All aggregate facts used by the strict WO-17 experiment gate."""

    control_source_revision_sha: str
    candidate_source_revision_sha: str
    experiment_plan_sha256: str
    runtime_contract_sha256: str
    split_manifest_sha256: str
    input_tree_sha256: str
    truth_sha256: str
    evaluator_sha256: str
    candidate_diff_manifest_sha256: str
    expected_record_count: int
    expected_layout_group_count: int
    control: PolicyArmAggregate
    candidate: PolicyArmAggregate
    activity: PolicyActivityAggregate
    folds: tuple[PolicyFoldPair, ...]
    new_false_approval_count: int
    new_catastrophic_false_approval_count: int
    non_decision_field_change_count: int
    decision_or_confidence_change_count: int
    source_and_diff_bound: bool
    population_bound: bool
    group_exclusive: bool
    paired_fold_members: bool
    split_deterministic: bool
    runtime_contract_bound: bool
    capture_contract_checks: Mapping[str, bool]
    regression_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in (
            "control_source_revision_sha",
            "candidate_source_revision_sha",
        ):
            value = str(getattr(self, name)).strip().casefold()
            if not _COMMIT_RE.fullmatch(value):
                raise GroupedPolicyGateError(
                    f"{name} must be a full Git commit SHA"
                )
            object.__setattr__(self, name, value)
        for name in (
            "experiment_plan_sha256",
            "runtime_contract_sha256",
            "split_manifest_sha256",
            "input_tree_sha256",
            "truth_sha256",
            "evaluator_sha256",
            "candidate_diff_manifest_sha256",
        ):
            value = str(getattr(self, name)).strip().casefold()
            if not _SHA256_RE.fullmatch(value):
                raise GroupedPolicyGateError(
                    f"{name} must be a full SHA-256 digest"
                )
            object.__setattr__(self, name, value)
        for name in (
            "expected_record_count",
            "expected_layout_group_count",
        ):
            object.__setattr__(
                self,
                name,
                _count(name, getattr(self, name), positive=True),
            )
        if self.expected_layout_group_count < REQUIRED_FOLDS:
            raise GroupedPolicyGateError(
                "at least five layout groups are required"
            )
        for name in (
            "new_false_approval_count",
            "new_catastrophic_false_approval_count",
            "non_decision_field_change_count",
            "decision_or_confidence_change_count",
        ):
            object.__setattr__(
                self, name, _count(name, getattr(self, name))
            )
        for name in (
            "source_and_diff_bound",
            "population_bound",
            "group_exclusive",
            "paired_fold_members",
            "split_deterministic",
            "runtime_contract_bound",
        ):
            object.__setattr__(
                self, name, _flag(name, getattr(self, name))
            )
        if (
            not isinstance(self.folds, tuple)
            or not all(
                isinstance(value, PolicyFoldPair) for value in self.folds
            )
        ):
            raise GroupedPolicyGateError(
                "folds must be a tuple of PolicyFoldPair values"
            )
        checks = self._boolean_mapping(
            "capture_contract_checks", self.capture_contract_checks
        )
        regressions = self._count_mapping(
            "regression_counts", self.regression_counts
        )
        object.__setattr__(self, "capture_contract_checks", checks)
        object.__setattr__(self, "regression_counts", regressions)

    @staticmethod
    def _boolean_mapping(
        name: str, value: Mapping[str, bool]
    ) -> Mapping[str, bool]:
        if not isinstance(value, Mapping) or not value:
            raise GroupedPolicyGateError(f"{name} must be non-empty")
        normalized: dict[str, bool] = {}
        for key, flag in value.items():
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", key)
            ):
                raise GroupedPolicyGateError(
                    f"{name} contains an invalid dimension"
                )
            normalized[key] = _flag(f"{name}.{key}", flag)
        return dict(sorted(normalized.items()))

    @staticmethod
    def _count_mapping(
        name: str, value: Mapping[str, int]
    ) -> Mapping[str, int]:
        if not isinstance(value, Mapping) or not value:
            raise GroupedPolicyGateError(f"{name} must be non-empty")
        normalized: dict[str, int] = {}
        for key, count in value.items():
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", key)
            ):
                raise GroupedPolicyGateError(
                    f"{name} contains an invalid dimension"
                )
            normalized[key] = _count(f"{name}.{key}", count)
        return dict(sorted(normalized.items()))


@dataclass(frozen=True)
class GroupedPolicyGateDecision:
    """Fail-closed decision and repository-safe aggregate serialization."""

    passed: bool
    evidence: GroupedPolicyEvidence
    ordered_folds: tuple[PolicyFoldPair, ...]
    repeat_weighted_deltas: tuple[float, ...]
    repeat_positive_fold_counts: tuple[int, ...]
    repeat_leave_best_fold_out_deltas: tuple[float, ...]
    gate_results: tuple[tuple[str, bool], ...]

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        return tuple(
            name for name, passed in self.gate_results if not passed
        )

    def to_aggregate_evidence(self) -> dict[str, Any]:
        """Return the strict identity-free aggregate accepted by the ledger."""

        evidence = self.evidence
        control = evidence.control
        candidate = evidence.candidate
        activity = evidence.activity
        fold_metrics = {
            f"repeat_{pair.repeat + 1}_fold_{pair.fold + 1}": {
                "record_count": pair.record_count,
                "layout_group_count": pair.layout_group_count,
                "control_score": pair.control_score,
                "candidate_score": pair.candidate_score,
                "score_delta": pair.delta,
            }
            for pair in self.ordered_folds
        }
        metrics: dict[str, float | int] = {
            "control_total_score": control.total_score,
            "candidate_total_score": candidate.total_score,
            "extraction_score": candidate.extraction_score,
            "classification_score": candidate.classification_score,
            "calibration_score": candidate.calibration_score,
            "control_missing_penalty": control.missing_penalty,
            "candidate_missing_penalty": candidate.missing_penalty,
            "extraction_score_delta": (
                candidate.extraction_score - control.extraction_score
            ),
            "classification_score_delta": (
                candidate.classification_score
                - control.classification_score
            ),
            "calibration_score_delta": (
                candidate.calibration_score - control.calibration_score
            ),
        }
        for repeat, value in enumerate(
            self.repeat_weighted_deltas, start=1
        ):
            metrics[f"repeat_{repeat}_weighted_score_delta"] = value
            metrics[f"repeat_{repeat}_positive_fold_count"] = (
                self.repeat_positive_fold_counts[repeat - 1]
            )
            metrics[f"repeat_{repeat}_leave_best_fold_out_delta"] = (
                self.repeat_leave_best_fold_out_deltas[repeat - 1]
            )
        counts = {
            "record_count": candidate.record_count,
            "layout_group_count": evidence.expected_layout_group_count,
            "control_record_count": control.record_count,
            "control_false_approval_count": control.false_approvals,
            "control_catastrophic_false_approval_count": (
                control.catastrophic_false_approvals
            ),
            "control_missing_record_count": control.missing_records,
            "control_invalid_record_count": control.invalid_records,
            "control_duplicate_record_count": control.duplicate_records,
            "control_extra_record_count": control.extra_records,
            "new_false_approval_count": evidence.new_false_approval_count,
            "new_catastrophic_false_approval_count": (
                evidence.new_catastrophic_false_approval_count
            ),
            "candidate_false_approval_count": candidate.false_approvals,
            "candidate_catastrophic_false_approval_count": (
                candidate.catastrophic_false_approvals
            ),
            "candidate_missing_record_count": candidate.missing_records,
            "candidate_invalid_record_count": candidate.invalid_records,
            "candidate_duplicate_record_count": candidate.duplicate_records,
            "candidate_extra_record_count": candidate.extra_records,
            "non_decision_field_change_count": (
                evidence.non_decision_field_change_count
            ),
            "decision_or_confidence_change_count": (
                evidence.decision_or_confidence_change_count
            ),
            "guarded_initial_approval_count": (
                activity.guarded_initial_approval_count
            ),
            "eligible_guarded_initial_count": (
                activity.eligible_guarded_initial_count
            ),
            "unguarded_initial_approval_count": (
                activity.unguarded_initial_approval_count
            ),
            "late_revalidation_approval_count": (
                activity.late_revalidation_approval_count
            ),
            # This remains explicitly named as the legacy counter.  It may be
            # non-zero only when fully accounted for by exact-matcher events.
            "legacy_forced_approval_count": (
                activity.legacy_forced_approval_count
            ),
            **{
                f"regression_{name}_count": value
                for name, value in evidence.regression_counts.items()
            },
        }
        gates = dict(self.gate_results)
        aggregate: dict[str, Any] = {
            "evaluation_mode": PUBLIC_EVIDENCE_LABEL,
            "evidence_label": "aggregate_only",
            "status": "passed" if self.passed else "blocked",
            "control_source_revision_sha": (
                evidence.control_source_revision_sha
            ),
            "candidate_source_revision_sha": (
                evidence.candidate_source_revision_sha
            ),
            "experiment_plan_sha256": evidence.experiment_plan_sha256,
            "runtime_contract_sha256": evidence.runtime_contract_sha256,
            "split_manifest_sha256": evidence.split_manifest_sha256,
            "input_tree_sha256": evidence.input_tree_sha256,
            "truth_sha256": evidence.truth_sha256,
            "evaluator_sha256": evidence.evaluator_sha256,
            "candidate_diff_manifest_sha256": (
                evidence.candidate_diff_manifest_sha256
            ),
            "record_count": candidate.record_count,
            "layout_group_count": evidence.expected_layout_group_count,
            "repeat_count": REQUIRED_REPEATS,
            "fold_count": REQUIRED_FOLDS,
            "evaluated_fold_count": REQUIRED_REPEATS * REQUIRED_FOLDS,
            "control_score": control.total_score,
            "candidate_score": candidate.total_score,
            "score_delta": candidate.total_score - control.total_score,
            "fold_deltas": [
                pair.delta for pair in self.ordered_folds
            ],
            "fold_weights": [
                pair.record_count for pair in self.ordered_folds
            ],
            "repeat_scores": list(self.repeat_weighted_deltas),
            "deterministic": (
                control.deterministic
                and candidate.deterministic
                and evidence.split_deterministic
            ),
            "fold_consistent": (
                gates["no_negative_folds"]
                and gates["leave_best_fold_out_positive"]
            ),
            "catastrophic_false_approvals": (
                candidate.catastrophic_false_approvals
            ),
            "false_approvals": candidate.false_approvals,
            "missing_records": candidate.missing_records,
            "invalid_records": candidate.invalid_records,
            "duplicate_records": candidate.duplicate_records,
            "extra_records": candidate.extra_records,
            "hard_gate_failure_count": len(self.blocking_reasons),
            "counts": counts,
            "checks": {
                **dict(evidence.capture_contract_checks),
                "source_and_diff_bound": evidence.source_and_diff_bound,
                "population_bound": evidence.population_bound,
                "group_exclusive": evidence.group_exclusive,
                "paired_fold_members": evidence.paired_fold_members,
                "split_deterministic": evidence.split_deterministic,
                "runtime_contract_bound": evidence.runtime_contract_bound,
                "control_capture_deterministic": control.deterministic,
                "candidate_capture_deterministic": candidate.deterministic,
            },
            "metrics": metrics,
            "fold_metrics": fold_metrics,
            "gate_results": gates,
        }
        require_aggregate_only(aggregate)
        return aggregate


class GroupedPolicyRevalidationGate:
    """Compute strict fold metrics and apply all WO-17 hard gates."""

    @staticmethod
    def _ordered_folds(
        evidence: GroupedPolicyEvidence,
    ) -> tuple[PolicyFoldPair, ...]:
        expected = {
            (repeat, fold)
            for repeat in range(REQUIRED_REPEATS)
            for fold in range(REQUIRED_FOLDS)
        }
        keyed: dict[tuple[int, int], PolicyFoldPair] = {}
        for pair in evidence.folds:
            key = (pair.repeat, pair.fold)
            if key in keyed:
                raise GroupedPolicyGateError(
                    "duplicate repeat/fold coordinate"
                )
            keyed[key] = pair
        if set(keyed) != expected:
            raise GroupedPolicyGateError(
                "evidence must contain exactly three repeats of five folds"
            )
        ordered = tuple(keyed[key] for key in sorted(expected))
        for repeat in range(REQUIRED_REPEATS):
            current = tuple(
                pair for pair in ordered if pair.repeat == repeat
            )
            if sum(pair.record_count for pair in current) != (
                evidence.expected_record_count
            ):
                raise GroupedPolicyGateError(
                    "each repeat must cover the complete record population"
                )
            if sum(pair.layout_group_count for pair in current) != (
                evidence.expected_layout_group_count
            ):
                raise GroupedPolicyGateError(
                    "each repeat must cover every layout group exactly once"
                )
        return ordered

    @staticmethod
    def _repeat_metrics(
        pairs: Sequence[PolicyFoldPair],
    ) -> tuple[Decimal, int, Decimal]:
        total_weight = sum(pair.record_count for pair in pairs)
        numerator = sum(
            (
                pair.delta_decimal * Decimal(pair.record_count)
                for pair in pairs
            ),
            Decimal(0),
        )
        weighted = numerator / Decimal(total_weight)
        positive_count = sum(pair.delta_decimal > 0 for pair in pairs)
        leave_one_out: list[Decimal] = []
        for pair in pairs:
            remaining_weight = total_weight - pair.record_count
            if remaining_weight <= 0:
                raise GroupedPolicyGateError(
                    "leave-one-fold-out population must be non-empty"
                )
            leave_one_out.append(
                (
                    numerator
                    - pair.delta_decimal * Decimal(pair.record_count)
                )
                / Decimal(remaining_weight)
            )
        return weighted, positive_count, min(leave_one_out)

    @staticmethod
    def _complete(
        arm: PolicyArmAggregate, expected_record_count: int
    ) -> bool:
        return (
            arm.record_count == expected_record_count
            and arm.missing_records == 0
            and arm.invalid_records == 0
            and arm.duplicate_records == 0
            and arm.extra_records == 0
        )

    def evaluate(
        self, evidence: GroupedPolicyEvidence
    ) -> GroupedPolicyGateDecision:
        ordered = self._ordered_folds(evidence)
        repeat_weighted: list[Decimal] = []
        repeat_positive: list[int] = []
        repeat_leave_best: list[Decimal] = []
        for repeat in range(REQUIRED_REPEATS):
            metrics = self._repeat_metrics(
                tuple(
                    pair for pair in ordered if pair.repeat == repeat
                )
            )
            repeat_weighted.append(metrics[0])
            repeat_positive.append(metrics[1])
            repeat_leave_best.append(metrics[2])

        control = evidence.control
        candidate = evidence.candidate
        activity = evidence.activity
        no_negative = all(
            pair.delta_decimal >= 0 for pair in ordered
        )
        leave_best_positive = all(
            value > 0 for value in repeat_leave_best
        )
        legacy_accounted = (
            activity.legacy_forced_approval_count
            == activity.guarded_initial_approval_count
            + activity.unguarded_initial_approval_count
            + activity.late_revalidation_approval_count
        )
        gates: Mapping[str, bool] = {
            "public_exposed_evidence": (
                PUBLIC_EVIDENCE_LABEL
                == "public_grouped_robustness_not_unseen"
            ),
            "source_and_diff_bound": evidence.source_and_diff_bound,
            "population_bound": evidence.population_bound,
            "runtime_contract_bound": evidence.runtime_contract_bound,
            "control_complete": self._complete(
                control, evidence.expected_record_count
            ),
            "candidate_complete": self._complete(
                candidate, evidence.expected_record_count
            ),
            "capture_deterministic": (
                control.deterministic and candidate.deterministic
            ),
            "group_exclusive": evidence.group_exclusive,
            "paired_fold_members": evidence.paired_fold_members,
            "split_deterministic": evidence.split_deterministic,
            "full_score_positive": (
                candidate.total_score > control.total_score
            ),
            "extraction_score_unchanged": (
                candidate.extraction_score == control.extraction_score
            ),
            "non_decision_fields_unchanged": (
                evidence.non_decision_field_change_count == 0
            ),
            "decision_variable_nonvacuous": (
                evidence.decision_or_confidence_change_count > 0
            ),
            "repeat_weighted_deltas_positive": all(
                value > 0 for value in repeat_weighted
            ),
            "no_negative_folds": no_negative,
            "leave_best_fold_out_positive": leave_best_positive,
            "no_new_false_approvals": (
                evidence.new_false_approval_count == 0
                and candidate.false_approvals <= control.false_approvals
            ),
            "no_catastrophic_false_approvals": (
                candidate.catastrophic_false_approvals == 0
                and evidence.new_catastrophic_false_approval_count == 0
            ),
            "guarded_initial_approval_nonvacuous": (
                activity.guarded_initial_approval_count > 0
            ),
            "eligible_guarded_activity_nonvacuous": (
                activity.eligible_guarded_initial_count > 0
                and activity.guarded_initial_approval_count
                <= activity.eligible_guarded_initial_count
            ),
            "no_unguarded_initial_approval": (
                activity.unguarded_initial_approval_count == 0
            ),
            "no_late_revalidation_approval": (
                activity.late_revalidation_approval_count == 0
            ),
            "legacy_forced_counter_fully_accounted": legacy_accounted,
            "contract_probes_pass": all(
                evidence.capture_contract_checks.values()
            ),
            "regression_suites_clean": not any(
                evidence.regression_counts.values()
            ),
        }
        blocking = tuple(
            name for name, passed in gates.items() if not passed
        )
        return GroupedPolicyGateDecision(
            passed=not blocking,
            evidence=evidence,
            ordered_folds=ordered,
            repeat_weighted_deltas=tuple(
                _float(value) for value in repeat_weighted
            ),
            repeat_positive_fold_counts=tuple(repeat_positive),
            repeat_leave_best_fold_out_deltas=tuple(
                _float(value) for value in repeat_leave_best
            ),
            gate_results=tuple(gates.items()),
        )
