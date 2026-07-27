"""Aggregate-only robustness gate for visible-evidence recovery candidates.

This development-only module evaluates a full public-exposed comparison plus
three repeats of five group-exclusive folds.  It deliberately accepts no case
identifiers or row-shaped outcomes.  Case-level manifests and evaluator output
must remain in the external evaluation directory; only their frozen manifest
hash and aggregate paired scores enter this gate.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from devtools.evaluation import PUBLIC_ROBUSTNESS_EVIDENCE
from devtools.experiment_control import ExperimentControlError, require_aggregate_only


PUBLIC_EXPOSED_SCOPE_LABEL = PUBLIC_ROBUSTNESS_EVIDENCE
REQUIRED_REPEATS = 3
REQUIRED_FOLDS = 5
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class GroupedRecoveryEvidenceError(ExperimentControlError):
    """The aggregate evidence is incomplete, unpaired, or malformed."""


def _require_count(name: str, value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise GroupedRecoveryEvidenceError(f"{name} must be a {qualifier} integer")
    return value


def _require_flag(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise GroupedRecoveryEvidenceError(f"{name} must be a boolean")
    return value


def _require_score(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroupedRecoveryEvidenceError(f"{name} must be a finite score")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise GroupedRecoveryEvidenceError(
            f"{name} must be finite and non-negative"
        )
    return normalized


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _float(value: Decimal) -> float:
    return 0.0 if value == 0 else float(value)


@dataclass(frozen=True)
class FullRunAggregate:
    """Aggregate metrics for one full control or candidate run."""

    total_score: float
    record_count: int
    catastrophic_false_approvals: int
    missing_records: int
    invalid_records: int
    false_positive_denial_recoveries: int
    deterministic: bool
    duplicate_records: int = 0
    extra_records: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "total_score", _require_score("total_score", self.total_score)
        )
        for name in (
            "record_count",
            "catastrophic_false_approvals",
            "missing_records",
            "invalid_records",
            "false_positive_denial_recoveries",
            "duplicate_records",
            "extra_records",
        ):
            object.__setattr__(
                self, name, _require_count(name, getattr(self, name))
            )
        object.__setattr__(
            self,
            "deterministic",
            _require_flag("deterministic", self.deterministic),
        )


@dataclass(frozen=True)
class RecoveryAuditAggregate:
    """Candidate-only audit counters for recovered visible evidence."""

    recovered_field_count: int
    recovered_field_complete_provenance_count: int
    serialization_default_used_as_evidence_count: int

    def __post_init__(self) -> None:
        for name in (
            "recovered_field_count",
            "recovered_field_complete_provenance_count",
            "serialization_default_used_as_evidence_count",
        ):
            object.__setattr__(
                self, name, _require_count(name, getattr(self, name))
            )
        if (
            self.recovered_field_complete_provenance_count
            > self.recovered_field_count
        ):
            raise GroupedRecoveryEvidenceError(
                "complete provenance count cannot exceed recovered field count"
            )

    @property
    def provenance_coverage_fraction(self) -> float:
        if self.recovered_field_count == 0:
            return 0.0
        return (
            self.recovered_field_complete_provenance_count
            / self.recovered_field_count
        )


@dataclass(frozen=True)
class FoldScorePair:
    """One candidate/control score pair over identical validation members."""

    repeat: int
    fold: int
    control_score: float
    candidate_score: float
    control_record_count: int
    candidate_record_count: int
    layout_group_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repeat", _require_count("repeat", self.repeat)
        )
        object.__setattr__(self, "fold", _require_count("fold", self.fold))
        object.__setattr__(
            self,
            "control_score",
            _require_score("control_score", self.control_score),
        )
        object.__setattr__(
            self,
            "candidate_score",
            _require_score("candidate_score", self.candidate_score),
        )
        for name in (
            "control_record_count",
            "candidate_record_count",
            "layout_group_count",
        ):
            object.__setattr__(
                self,
                name,
                _require_count(name, getattr(self, name), positive=True),
            )
        if self.control_record_count != self.candidate_record_count:
            raise GroupedRecoveryEvidenceError(
                "candidate and control fold record counts must match"
            )
        if self.layout_group_count > self.candidate_record_count:
            raise GroupedRecoveryEvidenceError(
                "a fold cannot contain more layout groups than records"
            )

    @property
    def record_count(self) -> int:
        return self.candidate_record_count

    @property
    def score_delta_decimal(self) -> Decimal:
        return _decimal(self.candidate_score) - _decimal(self.control_score)

    @property
    def score_delta(self) -> float:
        return _float(self.score_delta_decimal)


@dataclass(frozen=True)
class GroupedRecoveryEvidence:
    """All aggregate inputs required for a WO-15 recovery decision."""

    layout_manifest_sha256: str
    expected_record_count: int
    expected_layout_group_count: int
    control_full: FullRunAggregate
    candidate_full: FullRunAggregate
    candidate_recovery_audit: RecoveryAuditAggregate
    folds: tuple[FoldScorePair, ...]
    manifest_frozen_before_scoring: bool
    group_exclusive: bool
    paired_fold_members: bool
    split_deterministic: bool

    def __post_init__(self) -> None:
        digest = str(self.layout_manifest_sha256).strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            raise GroupedRecoveryEvidenceError(
                "layout_manifest_sha256 must be a full SHA-256 digest"
            )
        object.__setattr__(self, "layout_manifest_sha256", digest)
        object.__setattr__(
            self,
            "expected_record_count",
            _require_count(
                "expected_record_count", self.expected_record_count, positive=True
            ),
        )
        object.__setattr__(
            self,
            "expected_layout_group_count",
            _require_count(
                "expected_layout_group_count",
                self.expected_layout_group_count,
                positive=True,
            ),
        )
        if self.expected_layout_group_count < REQUIRED_FOLDS:
            raise GroupedRecoveryEvidenceError(
                "layout group count must be at least the fold count"
            )
        if not isinstance(self.folds, tuple) or not all(
            isinstance(fold, FoldScorePair) for fold in self.folds
        ):
            raise GroupedRecoveryEvidenceError(
                "folds must be a tuple of FoldScorePair values"
            )
        for name in (
            "manifest_frozen_before_scoring",
            "group_exclusive",
            "paired_fold_members",
            "split_deterministic",
        ):
            object.__setattr__(
                self, name, _require_flag(name, getattr(self, name))
            )


@dataclass(frozen=True)
class EvaluatedFold:
    repeat: int
    fold: int
    record_count: int
    layout_group_count: int
    control_score: float
    candidate_score: float
    score_delta: float


@dataclass(frozen=True)
class GroupedRecoveryGateDecision:
    """Immutable decision with a separately serializable aggregate view."""

    passed: bool
    blocking_reasons: tuple[str, ...]
    layout_manifest_sha256: str
    expected_record_count: int
    expected_layout_group_count: int
    control_full: FullRunAggregate
    candidate_full: FullRunAggregate
    candidate_recovery_audit: RecoveryAuditAggregate
    folds: tuple[EvaluatedFold, ...]
    full_score_delta: float
    repeat_weighted_deltas: tuple[float, ...]
    repeat_positive_fold_counts: tuple[int, ...]
    repeat_leave_best_fold_out_deltas: tuple[float, ...]
    gate_results: tuple[tuple[str, bool], ...]
    diagnostic_results: tuple[tuple[str, bool], ...]
    concentration_warning: bool

    @property
    def decision(self) -> str:
        return "PASSED" if self.passed else "BLOCKED"

    def to_aggregate_evidence(self) -> dict[str, Any]:
        """Return a broker-safe aggregate object with no case-level identity."""

        fold_metrics = {
            f"repeat_{fold.repeat}_fold_{fold.fold}": {
                "record_count": fold.record_count,
                "layout_group_count": fold.layout_group_count,
                "control_score": fold.control_score,
                "candidate_score": fold.candidate_score,
                "score_delta": fold.score_delta,
            }
            for fold in self.folds
        }
        repeat_metrics: dict[str, int | float] = {}
        for repeat in range(REQUIRED_REPEATS):
            repeat_metrics[f"repeat_{repeat}_weighted_score_delta"] = (
                self.repeat_weighted_deltas[repeat]
            )
            repeat_metrics[f"repeat_{repeat}_positive_fold_count"] = (
                self.repeat_positive_fold_counts[repeat]
            )
            repeat_metrics[f"repeat_{repeat}_leave_best_fold_out_delta"] = (
                self.repeat_leave_best_fold_out_deltas[repeat]
            )

        gate_results = dict(self.gate_results)
        diagnostic_results = dict(self.diagnostic_results)
        aggregate = {
            "evaluation_mode": PUBLIC_EXPOSED_SCOPE_LABEL,
            "status": self.decision.casefold(),
            "layout_manifest_sha256": self.layout_manifest_sha256,
            "expected_record_count": self.expected_record_count,
            "layout_group_count": self.expected_layout_group_count,
            "repeat_count": REQUIRED_REPEATS,
            "fold_count": REQUIRED_REPEATS * REQUIRED_FOLDS,
            "full_control_score": self.control_full.total_score,
            "full_candidate_score": self.candidate_full.total_score,
            "score_delta": self.full_score_delta,
            "fold_deltas": [fold.score_delta for fold in self.folds],
            "repeat_scores": list(self.repeat_weighted_deltas),
            "deterministic": gate_results["run_deterministic"]
            and gate_results["split_deterministic"],
            "fold_consistent": gate_results["repeat_weighted_deltas_positive"]
            and gate_results["at_least_one_positive_fold_per_repeat"]
            and gate_results["no_negative_folds"]
            and gate_results["leave_best_fold_out_nonnegative"],
            "catastrophic_false_approvals": (
                self.candidate_full.catastrophic_false_approvals
            ),
            "missing_records": self.candidate_full.missing_records,
            "invalid_records": self.candidate_full.invalid_records,
            "duplicate_records": self.candidate_full.duplicate_records,
            "extra_records": self.candidate_full.extra_records,
            "false_positive_denial_recoveries_delta": (
                self.candidate_full.false_positive_denial_recoveries
                - self.control_full.false_positive_denial_recoveries
            ),
            "recovered_field_count": (
                self.candidate_recovery_audit.recovered_field_count
            ),
            "recovered_field_complete_provenance_count": (
                self.candidate_recovery_audit
                .recovered_field_complete_provenance_count
            ),
            "provenance_coverage_fraction": (
                self.candidate_recovery_audit.provenance_coverage_fraction
            ),
            "serialization_default_used_as_evidence_count": (
                self.candidate_recovery_audit
                .serialization_default_used_as_evidence_count
            ),
            "hard_gate_failure_count": len(self.blocking_reasons),
            "warning_count": int(self.concentration_warning),
            "gate_results": gate_results,
            "checks": {
                **diagnostic_results,
                "score_gain_concentration_warning": self.concentration_warning,
            },
            "metrics": repeat_metrics,
            "fold_metrics": fold_metrics,
        }
        require_aggregate_only(aggregate)
        return aggregate


class GroupedRecoveryGate:
    """Compute paired deltas and fail closed on every WO-15 hard gate."""

    @staticmethod
    def _ordered_folds(
        evidence: GroupedRecoveryEvidence,
    ) -> tuple[FoldScorePair, ...]:
        expected_keys = {
            (repeat, fold)
            for repeat in range(REQUIRED_REPEATS)
            for fold in range(REQUIRED_FOLDS)
        }
        keyed: dict[tuple[int, int], FoldScorePair] = {}
        for pair in evidence.folds:
            key = (pair.repeat, pair.fold)
            if key in keyed:
                raise GroupedRecoveryEvidenceError(
                    f"duplicate aggregate fold: repeat={pair.repeat}, fold={pair.fold}"
                )
            keyed[key] = pair
        if set(keyed) != expected_keys:
            raise GroupedRecoveryEvidenceError(
                "evidence must contain exactly three repeats of five folds"
            )

        ordered = tuple(keyed[key] for key in sorted(expected_keys))
        for repeat in range(REQUIRED_REPEATS):
            current = tuple(pair for pair in ordered if pair.repeat == repeat)
            record_count = sum(pair.record_count for pair in current)
            group_count = sum(pair.layout_group_count for pair in current)
            if record_count != evidence.expected_record_count:
                raise GroupedRecoveryEvidenceError(
                    f"repeat {repeat} does not cover the expected record count"
                )
            if group_count != evidence.expected_layout_group_count:
                raise GroupedRecoveryEvidenceError(
                    f"repeat {repeat} does not cover the expected layout group count"
                )
        return ordered

    @staticmethod
    def _repeat_metrics(
        folds: Sequence[FoldScorePair],
    ) -> tuple[Decimal, int, Decimal]:
        weighted_numerator = sum(
            (
                pair.score_delta_decimal * Decimal(pair.record_count)
                for pair in folds
            ),
            Decimal(0),
        )
        total_weight = sum(pair.record_count for pair in folds)
        weighted_delta = weighted_numerator / Decimal(total_weight)
        positive_fold_count = sum(
            pair.score_delta_decimal > 0 for pair in folds
        )

        leave_one_out: list[Decimal] = []
        for omitted in folds:
            remaining_weight = total_weight - omitted.record_count
            remaining_numerator = (
                weighted_numerator
                - omitted.score_delta_decimal * Decimal(omitted.record_count)
            )
            leave_one_out.append(
                remaining_numerator / Decimal(remaining_weight)
            )
        # The smallest residual delta omits the strongest weighted contributor.
        leave_best_fold_out_delta = min(leave_one_out)
        return weighted_delta, positive_fold_count, leave_best_fold_out_delta

    def evaluate(
        self, evidence: GroupedRecoveryEvidence
    ) -> GroupedRecoveryGateDecision:
        ordered = self._ordered_folds(evidence)
        repeat_weighted: list[Decimal] = []
        repeat_positive_counts: list[int] = []
        repeat_leave_best_out: list[Decimal] = []
        for repeat in range(REQUIRED_REPEATS):
            current = tuple(pair for pair in ordered if pair.repeat == repeat)
            weighted, positives, leave_best_out = self._repeat_metrics(current)
            repeat_weighted.append(weighted)
            repeat_positive_counts.append(positives)
            repeat_leave_best_out.append(leave_best_out)

        full_delta = (
            _decimal(evidence.candidate_full.total_score)
            - _decimal(evidence.control_full.total_score)
        )
        candidate_complete = (
            evidence.candidate_full.record_count == evidence.expected_record_count
            and evidence.candidate_full.missing_records == 0
            and evidence.candidate_full.invalid_records == 0
            and evidence.candidate_full.duplicate_records == 0
            and evidence.candidate_full.extra_records == 0
        )
        control_complete = (
            evidence.control_full.record_count == evidence.expected_record_count
            and evidence.control_full.missing_records == 0
            and evidence.control_full.invalid_records == 0
            and evidence.control_full.duplicate_records == 0
            and evidence.control_full.extra_records == 0
        )
        gate_results: Mapping[str, bool] = {
            "public_exposed_evidence": (
                PUBLIC_EXPOSED_SCOPE_LABEL
                == "public_grouped_robustness_not_unseen"
            ),
            "manifest_frozen_before_scoring": (
                evidence.manifest_frozen_before_scoring
            ),
            "group_exclusive": evidence.group_exclusive,
            "paired_fold_members": evidence.paired_fold_members,
            "split_deterministic": evidence.split_deterministic,
            "run_deterministic": (
                evidence.control_full.deterministic
                and evidence.candidate_full.deterministic
            ),
            "full_score_positive": full_delta > 0,
            "repeat_weighted_deltas_positive": all(
                delta > 0 for delta in repeat_weighted
            ),
            "at_least_one_positive_fold_per_repeat": all(
                count >= 1 for count in repeat_positive_counts
            ),
            "no_negative_folds": all(
                pair.score_delta_decimal >= 0 for pair in ordered
            ),
            "leave_best_fold_out_nonnegative": all(
                delta >= 0 for delta in repeat_leave_best_out
            ),
            "candidate_complete": candidate_complete,
            "control_complete": control_complete,
            "no_catastrophic_false_approvals": (
                evidence.candidate_full.catastrophic_false_approvals == 0
            ),
            "no_increased_false_positive_denial_recoveries": (
                evidence.candidate_full.false_positive_denial_recoveries
                <= evidence.control_full.false_positive_denial_recoveries
            ),
            "provenance_complete": (
                evidence.candidate_recovery_audit.recovered_field_count > 0
                and evidence.candidate_recovery_audit
                .recovered_field_complete_provenance_count
                == evidence.candidate_recovery_audit.recovered_field_count
            ),
            "no_serialization_default_as_evidence": (
                evidence.candidate_recovery_audit
                .serialization_default_used_as_evidence_count
                == 0
            ),
        }
        diagnostic_results: Mapping[str, bool] = {
            "fold_majority_positive": all(
                count >= 3 for count in repeat_positive_counts
            ),
            "leave_best_fold_out_positive": all(
                delta > 0 for delta in repeat_leave_best_out
            ),
        }
        concentration_warning = not all(diagnostic_results.values())
        blocking_reasons = tuple(
            name for name, passed in gate_results.items() if not passed
        )
        evaluated_folds = tuple(
            EvaluatedFold(
                repeat=pair.repeat,
                fold=pair.fold,
                record_count=pair.record_count,
                layout_group_count=pair.layout_group_count,
                control_score=pair.control_score,
                candidate_score=pair.candidate_score,
                score_delta=pair.score_delta,
            )
            for pair in ordered
        )
        return GroupedRecoveryGateDecision(
            passed=not blocking_reasons,
            blocking_reasons=blocking_reasons,
            layout_manifest_sha256=evidence.layout_manifest_sha256,
            expected_record_count=evidence.expected_record_count,
            expected_layout_group_count=evidence.expected_layout_group_count,
            control_full=evidence.control_full,
            candidate_full=evidence.candidate_full,
            candidate_recovery_audit=evidence.candidate_recovery_audit,
            folds=evaluated_folds,
            full_score_delta=_float(full_delta),
            repeat_weighted_deltas=tuple(
                _float(delta) for delta in repeat_weighted
            ),
            repeat_positive_fold_counts=tuple(repeat_positive_counts),
            repeat_leave_best_fold_out_deltas=tuple(
                _float(delta) for delta in repeat_leave_best_out
            ),
            gate_results=tuple(gate_results.items()),
            diagnostic_results=tuple(diagnostic_results.items()),
            concentration_warning=concentration_warning,
        )
