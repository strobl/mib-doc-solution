"""Aggregate-only repeated-group gate for applicant-aware evidence fusion.

The gate consumes no case identifiers.  It compares a frozen legacy-fusion
control with the candidate over the full public corpus and three deterministic
repeats of five layout-group-exclusive folds.  Every fold must be non-negative
and every repeat must remain positive after its strongest fold is removed.
Fold-majority remains a concentration diagnostic; every acceptance condition
is fail-closed.
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


class GroupedFusionEvidenceError(ExperimentControlError):
    """The aggregate fusion evidence is incomplete, unpaired, or malformed."""


def _require_count(name: str, value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise GroupedFusionEvidenceError(f"{name} must be a {qualifier} integer")
    return value


def _require_flag(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise GroupedFusionEvidenceError(f"{name} must be a boolean")
    return value


def _require_score(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroupedFusionEvidenceError(f"{name} must be a finite score")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise GroupedFusionEvidenceError(
            f"{name} must be finite and non-negative"
        )
    return normalized


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _float(value: Decimal) -> float:
    return 0.0 if value == 0 else float(value)


@dataclass(frozen=True)
class FusionRunAggregate:
    """Official-evaluator aggregates for one repeated prediction arm."""

    total_score: float
    record_count: int
    catastrophic_false_approvals: int
    false_positive_denials: int
    missing_records: int
    invalid_records: int
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
            "false_positive_denials",
            "missing_records",
            "invalid_records",
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
class FusionAuditAggregate:
    """Candidate-only, revision-bound aggregate evidence-fusion counters."""

    changed_field_count: int
    changed_field_complete_provenance_count: int
    clean_higher_authority_override_count: int
    binding_authority_override_count: int
    text_layer_winner_count: int
    serialization_default_used_as_evidence_count: int
    correlated_views_collapsed: int
    independent_agreement_resolutions: int
    same_rank_contested_count: int
    cross_applicant_candidates_excluded: int

    def __post_init__(self) -> None:
        for name in (
            "changed_field_count",
            "changed_field_complete_provenance_count",
            "clean_higher_authority_override_count",
            "binding_authority_override_count",
            "text_layer_winner_count",
            "serialization_default_used_as_evidence_count",
            "correlated_views_collapsed",
            "independent_agreement_resolutions",
            "same_rank_contested_count",
            "cross_applicant_candidates_excluded",
        ):
            object.__setattr__(
                self, name, _require_count(name, getattr(self, name))
            )
        if (
            self.changed_field_complete_provenance_count
            > self.changed_field_count
        ):
            raise GroupedFusionEvidenceError(
                "complete provenance count cannot exceed changed field count"
            )

    @property
    def provenance_coverage_fraction(self) -> float:
        if self.changed_field_count == 0:
            return 0.0
        return (
            self.changed_field_complete_provenance_count
            / self.changed_field_count
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
        object.__setattr__(self, "repeat", _require_count("repeat", self.repeat))
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
            raise GroupedFusionEvidenceError(
                "candidate and control fold record counts must match"
            )
        if self.layout_group_count > self.candidate_record_count:
            raise GroupedFusionEvidenceError(
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
class GroupedFusionEvidence:
    """All aggregate inputs required for a WO-16 fusion decision."""

    layout_manifest_sha256: str
    expected_record_count: int
    expected_layout_group_count: int
    legacy_control_full: FusionRunAggregate
    candidate_full: FusionRunAggregate
    candidate_fusion_audit: FusionAuditAggregate
    new_catastrophic_false_approval_count: int
    new_false_positive_denial_count: int
    folds: tuple[FoldScorePair, ...]
    manifest_frozen_before_scoring: bool
    group_exclusive: bool
    paired_fold_members: bool
    split_deterministic: bool

    def __post_init__(self) -> None:
        digest = str(self.layout_manifest_sha256).strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            raise GroupedFusionEvidenceError(
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
            raise GroupedFusionEvidenceError(
                "layout group count must be at least the fold count"
            )
        for name in (
            "new_catastrophic_false_approval_count",
            "new_false_positive_denial_count",
        ):
            object.__setattr__(
                self, name, _require_count(name, getattr(self, name))
            )
        if not isinstance(self.folds, tuple) or not all(
            isinstance(fold, FoldScorePair) for fold in self.folds
        ):
            raise GroupedFusionEvidenceError(
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
class GroupedFusionGateDecision:
    """Immutable decision with a separately serializable aggregate view."""

    passed: bool
    blocking_reasons: tuple[str, ...]
    layout_manifest_sha256: str
    expected_record_count: int
    expected_layout_group_count: int
    legacy_control_full: FusionRunAggregate
    candidate_full: FusionRunAggregate
    candidate_fusion_audit: FusionAuditAggregate
    new_catastrophic_false_approval_count: int
    new_false_positive_denial_count: int
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

        gates = dict(self.gate_results)
        diagnostics = dict(self.diagnostic_results)
        audit = self.candidate_fusion_audit
        aggregate = {
            "evaluation_mode": PUBLIC_EXPOSED_SCOPE_LABEL,
            "status": self.decision.casefold(),
            "layout_manifest_sha256": self.layout_manifest_sha256,
            "expected_record_count": self.expected_record_count,
            "layout_group_count": self.expected_layout_group_count,
            "repeat_count": REQUIRED_REPEATS,
            "fold_count": REQUIRED_FOLDS,
            "evaluated_fold_count": REQUIRED_REPEATS * REQUIRED_FOLDS,
            "full_control_score": self.legacy_control_full.total_score,
            "full_candidate_score": self.candidate_full.total_score,
            "score_delta": self.full_score_delta,
            "fold_deltas": [fold.score_delta for fold in self.folds],
            "fold_weights": [fold.record_count for fold in self.folds],
            "repeat_scores": list(self.repeat_weighted_deltas),
            "deterministic": (
                gates["run_deterministic"]
                and gates["split_deterministic"]
            ),
            "fold_consistent": (
                gates["repeat_weighted_deltas_positive"]
                and gates["at_least_one_positive_fold_per_repeat"]
                and gates["no_negative_folds"]
                and gates["leave_best_fold_out_positive"]
            ),
            "catastrophic_false_approvals": (
                self.candidate_full.catastrophic_false_approvals
            ),
            "catastrophic_false_approvals_delta": (
                self.candidate_full.catastrophic_false_approvals
                - self.legacy_control_full.catastrophic_false_approvals
            ),
            "false_positive_denials_delta": (
                self.candidate_full.false_positive_denials
                - self.legacy_control_full.false_positive_denials
            ),
            "new_catastrophic_false_approval_count": (
                self.new_catastrophic_false_approval_count
            ),
            "new_false_positive_denial_count": (
                self.new_false_positive_denial_count
            ),
            "missing_records": self.candidate_full.missing_records,
            "invalid_records": self.candidate_full.invalid_records,
            "duplicate_records": self.candidate_full.duplicate_records,
            "extra_records": self.candidate_full.extra_records,
            "provenance_coverage_fraction": audit.provenance_coverage_fraction,
            "counts": {
                "changed_field_count": audit.changed_field_count,
                "changed_field_complete_provenance_count": (
                    audit.changed_field_complete_provenance_count
                ),
                "clean_higher_authority_override_count": (
                    audit.clean_higher_authority_override_count
                ),
                "binding_authority_override_count": (
                    audit.binding_authority_override_count
                ),
                "text_layer_winner_count": audit.text_layer_winner_count,
                "serialization_default_used_as_evidence_count": (
                    audit.serialization_default_used_as_evidence_count
                ),
                "correlated_views_collapsed": (
                    audit.correlated_views_collapsed
                ),
                "independent_agreement_resolutions": (
                    audit.independent_agreement_resolutions
                ),
                "same_rank_contested_count": (
                    audit.same_rank_contested_count
                ),
                "cross_applicant_candidates_excluded": (
                    audit.cross_applicant_candidates_excluded
                ),
            },
            "hard_gate_failure_count": len(self.blocking_reasons),
            "warning_count": int(self.concentration_warning),
            "gate_results": gates,
            "checks": {
                **diagnostics,
                "score_gain_concentration_warning": self.concentration_warning,
            },
            "metrics": repeat_metrics,
            "fold_metrics": fold_metrics,
        }
        require_aggregate_only(aggregate)
        return aggregate


class GroupedFusionGate:
    """Compute paired deltas and fail closed on every WO-16 hard gate."""

    @staticmethod
    def _ordered_folds(
        evidence: GroupedFusionEvidence,
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
                raise GroupedFusionEvidenceError(
                    f"duplicate aggregate fold: repeat={pair.repeat}, fold={pair.fold}"
                )
            keyed[key] = pair
        if set(keyed) != expected_keys:
            raise GroupedFusionEvidenceError(
                "evidence must contain exactly three repeats of five folds"
            )

        ordered = tuple(keyed[key] for key in sorted(expected_keys))
        for repeat in range(REQUIRED_REPEATS):
            current = tuple(pair for pair in ordered if pair.repeat == repeat)
            if sum(pair.record_count for pair in current) != (
                evidence.expected_record_count
            ):
                raise GroupedFusionEvidenceError(
                    f"repeat {repeat} does not cover the expected record count"
                )
            if sum(pair.layout_group_count for pair in current) != (
                evidence.expected_layout_group_count
            ):
                raise GroupedFusionEvidenceError(
                    f"repeat {repeat} does not cover the expected layout group count"
                )
        return ordered

    @staticmethod
    def _repeat_metrics(
        folds: Sequence[FoldScorePair],
    ) -> tuple[Decimal, int, Decimal]:
        numerator = sum(
            (
                pair.score_delta_decimal * Decimal(pair.record_count)
                for pair in folds
            ),
            Decimal(0),
        )
        total_weight = sum(pair.record_count for pair in folds)
        weighted_delta = numerator / Decimal(total_weight)
        positive_count = sum(pair.score_delta_decimal > 0 for pair in folds)

        leave_one_out: list[Decimal] = []
        for omitted in folds:
            remaining_weight = total_weight - omitted.record_count
            remaining_numerator = (
                numerator
                - omitted.score_delta_decimal * Decimal(omitted.record_count)
            )
            leave_one_out.append(
                remaining_numerator / Decimal(remaining_weight)
            )
        return weighted_delta, positive_count, min(leave_one_out)

    def evaluate(
        self, evidence: GroupedFusionEvidence
    ) -> GroupedFusionGateDecision:
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
            - _decimal(evidence.legacy_control_full.total_score)
        )

        def complete(run: FusionRunAggregate) -> bool:
            return (
                run.record_count == evidence.expected_record_count
                and run.missing_records == 0
                and run.invalid_records == 0
                and run.duplicate_records == 0
                and run.extra_records == 0
            )

        audit = evidence.candidate_fusion_audit
        no_negative_folds = all(
            pair.score_delta_decimal >= 0 for pair in ordered
        )
        leave_best_fold_out_positive = all(
            delta > 0 for delta in repeat_leave_best_out
        )
        gates: Mapping[str, bool] = {
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
                evidence.legacy_control_full.deterministic
                and evidence.candidate_full.deterministic
            ),
            "full_score_positive": full_delta > 0,
            "repeat_weighted_deltas_positive": all(
                delta > 0 for delta in repeat_weighted
            ),
            "at_least_one_positive_fold_per_repeat": all(
                count >= 1 for count in repeat_positive_counts
            ),
            "no_negative_folds": no_negative_folds,
            "leave_best_fold_out_positive": leave_best_fold_out_positive,
            "candidate_complete": complete(evidence.candidate_full),
            "legacy_control_complete": complete(evidence.legacy_control_full),
            "no_increased_catastrophic_false_approvals": (
                evidence.candidate_full.catastrophic_false_approvals
                <= evidence.legacy_control_full.catastrophic_false_approvals
            ),
            "no_increased_false_positive_denials": (
                evidence.candidate_full.false_positive_denials
                <= evidence.legacy_control_full.false_positive_denials
            ),
            "no_new_catastrophic_false_approvals": (
                evidence.new_catastrophic_false_approval_count == 0
            ),
            "no_new_false_positive_denials": (
                evidence.new_false_positive_denial_count == 0
            ),
            "changed_fields_nonvacuous": audit.changed_field_count > 0,
            "changed_field_provenance_complete": (
                audit.changed_field_count > 0
                and audit.changed_field_complete_provenance_count
                == audit.changed_field_count
            ),
            "no_clean_higher_authority_overrides": (
                audit.clean_higher_authority_override_count == 0
            ),
            "no_binding_authority_overrides": (
                audit.binding_authority_override_count == 0
            ),
            "no_text_layer_winners": audit.text_layer_winner_count == 0,
            "no_serialization_default_as_evidence": (
                audit.serialization_default_used_as_evidence_count == 0
            ),
        }
        diagnostics: Mapping[str, bool] = {
            "no_negative_folds": no_negative_folds,
            "leave_best_fold_out_nonnegative": all(
                delta >= 0 for delta in repeat_leave_best_out
            ),
            "fold_majority_positive": all(
                count >= 3 for count in repeat_positive_counts
            ),
            "leave_best_fold_out_positive": leave_best_fold_out_positive,
        }
        concentration_warning = not all(diagnostics.values())
        blocking_reasons = tuple(
            name for name, passed in gates.items() if not passed
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
        return GroupedFusionGateDecision(
            passed=not blocking_reasons,
            blocking_reasons=blocking_reasons,
            layout_manifest_sha256=evidence.layout_manifest_sha256,
            expected_record_count=evidence.expected_record_count,
            expected_layout_group_count=evidence.expected_layout_group_count,
            legacy_control_full=evidence.legacy_control_full,
            candidate_full=evidence.candidate_full,
            candidate_fusion_audit=evidence.candidate_fusion_audit,
            new_catastrophic_false_approval_count=(
                evidence.new_catastrophic_false_approval_count
            ),
            new_false_positive_denial_count=(
                evidence.new_false_positive_denial_count
            ),
            folds=evaluated_folds,
            full_score_delta=_float(full_delta),
            repeat_weighted_deltas=tuple(
                _float(delta) for delta in repeat_weighted
            ),
            repeat_positive_fold_counts=tuple(repeat_positive_counts),
            repeat_leave_best_fold_out_deltas=tuple(
                _float(delta) for delta in repeat_leave_best_out
            ),
            gate_results=tuple(gates.items()),
            diagnostic_results=tuple(diagnostics.items()),
            concentration_warning=concentration_warning,
        )
