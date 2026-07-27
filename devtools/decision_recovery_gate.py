"""Fail-closed aggregate acceptance gate for WO-18 decision recovery.

The gate deliberately knows nothing about case identity, filenames, model
weights, or individual predictions.  It consumes official-evaluator
aggregates produced by :mod:`devtools.decision_recovery_evidence` and promotes
the gated hybrid only when its gain over the deterministic policy engine is
strictly positive in every repeated grouped fold and every protected role.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from devtools.evaluation import PUBLIC_ROBUSTNESS_EVIDENCE
from devtools.experiment_control import (
    ExperimentControlError,
    require_aggregate_only,
)


PUBLIC_EXPOSED_SCOPE_LABEL = PUBLIC_ROBUSTNESS_EVIDENCE
APPROACHES = (
    "deterministic_engine",
    "evidence_completion_only",
    "compact_identity_free_model",
    "gated_hybrid",
)
CONTROL_APPROACH = "deterministic_engine"
CANDIDATE_APPROACH = "gated_hybrid"
PROTECTED_ROLES = (
    "binding_authority",
    "visible_disqualifier",
    "approval_guard",
    "denial_guard",
    "uncertainty",
)
REQUIRED_REPEATS = 3
REQUIRED_FOLDS = 5
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class DecisionRecoveryEvidenceError(ExperimentControlError):
    """WO-18 aggregate evidence is incomplete, unsafe, or inconsistent."""


def _count(name: str, value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise DecisionRecoveryEvidenceError(
            f"{name} must be a {qualifier} integer"
        )
    return value


def _score(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionRecoveryEvidenceError(f"{name} must be a finite score")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise DecisionRecoveryEvidenceError(
            f"{name} must be a finite non-negative score"
        )
    return normalized


def _flag(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise DecisionRecoveryEvidenceError(f"{name} must be a boolean")
    return value


def _digest(name: str, value: Any) -> str:
    normalized = str(value).strip().casefold()
    if not _SHA256_RE.fullmatch(normalized):
        raise DecisionRecoveryEvidenceError(
            f"{name} must be a full SHA-256 digest"
        )
    return normalized


def _revision(value: Any) -> str:
    normalized = str(value).strip().casefold()
    if not _GIT_COMMIT_RE.fullmatch(normalized):
        raise DecisionRecoveryEvidenceError(
            "source_revision_sha must be a full Git commit SHA"
        )
    return normalized


def _approach_mapping(
    name: str,
    value: Mapping[str, Any],
    factory: Any,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(APPROACHES):
        raise DecisionRecoveryEvidenceError(
            f"{name} must contain the exact four WO-18 approaches"
        )
    return MappingProxyType(
        {
            approach: factory(f"{name}.{approach}", value[approach])
            for approach in APPROACHES
        }
    )


@dataclass(frozen=True)
class ApproachRunAggregate:
    """Official-evaluator metrics for one approach over one exact slice."""

    total_score: float
    classification_score: float
    record_count: int
    catastrophic_false_approvals: int
    false_positive_denials: int
    missing_records: int
    invalid_records: int
    duplicate_records: int
    extra_records: int
    deterministic: bool

    def __post_init__(self) -> None:
        for name in ("total_score", "classification_score"):
            object.__setattr__(self, name, _score(name, getattr(self, name)))
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
                self, name, _count(name, getattr(self, name))
            )
        object.__setattr__(
            self,
            "deterministic",
            _flag("deterministic", self.deterministic),
        )

    @property
    def complete_and_valid(self) -> bool:
        return (
            self.record_count > 0
            and self.missing_records == 0
            and self.invalid_records == 0
            and self.duplicate_records == 0
            and self.extra_records == 0
        )


@dataclass(frozen=True)
class ScoreSlice:
    """One paired repeat, grouped fold, or protected-role comparison."""

    repeat: int
    fold: int | None
    record_count: int
    layout_group_count: int
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repeat", _count("repeat", self.repeat)
        )
        if self.fold is not None:
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
        if self.layout_group_count > self.record_count:
            raise DecisionRecoveryEvidenceError(
                "a score slice cannot contain more groups than records"
            )
        object.__setattr__(
            self,
            "scores",
            _approach_mapping("scores", self.scores, _score),
        )

    @property
    def candidate_delta(self) -> float:
        return float(
            Decimal(str(self.scores[CANDIDATE_APPROACH]))
            - Decimal(str(self.scores[CONTROL_APPROACH]))
        )


@dataclass(frozen=True)
class ProtectedRoleScore:
    """One role-specific score per repeat, with identity kept external."""

    role: str
    repeat: int
    record_count: int
    layout_group_count: int
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        role = str(self.role).strip()
        if role not in PROTECTED_ROLES:
            raise DecisionRecoveryEvidenceError(
                "protected role is outside the frozen WO-18 role taxonomy"
            )
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self, "repeat", _count("repeat", self.repeat)
        )
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
        if self.layout_group_count > self.record_count:
            raise DecisionRecoveryEvidenceError(
                "a protected role cannot contain more groups than records"
            )
        object.__setattr__(
            self,
            "scores",
            _approach_mapping("scores", self.scores, _score),
        )

    @property
    def candidate_delta(self) -> float:
        return float(
            Decimal(str(self.scores[CANDIDATE_APPROACH]))
            - Decimal(str(self.scores[CONTROL_APPROACH]))
        )


@dataclass(frozen=True)
class DecisionRecoveryContractAudit:
    """Aggregate hard-ordering, recovery-guard, and leakage probes."""

    counts: Mapping[str, int]
    leakage_clean: bool
    feature_schema_exact: bool
    deterministic: bool

    REQUIRED_POSITIVE_COUNTS = (
        "binding_authority_probe_count",
        "visible_disqualifier_probe_count",
        "deterministic_policy_probe_count",
        "residual_recovery_probe_count",
        "uncertainty_review_probe_count",
        "approval_complete_visible_probe_count",
        "approval_scope_probe_count",
        "approval_conflict_probe_count",
        "approval_watermark_probe_count",
        "denial_visible_violation_probe_count",
        "denial_binding_authority_probe_count",
        "denial_topology_only_probe_count",
        "denial_missingness_only_probe_count",
        "model_margin_probe_count",
        "ensemble_disagreement_probe_count",
        "probability_simplex_probe_count",
        "true_margin_probe_count",
    )
    REQUIRED_ZERO_COUNTS = (
        "hard_ordering_failure_count",
        "approval_guard_failure_count",
        "denial_guard_failure_count",
        "denial_without_visible_violation_count",
        "forbidden_feature_finding_count",
        "hidden_content_dependency_count",
        "identity_feature_count",
        "nonfinite_probability_count",
        "unnormalized_probability_count",
        "incorrect_margin_count",
        "invalid_disagreement_count",
    )
    REQUIRED_COUNTS = REQUIRED_POSITIVE_COUNTS + REQUIRED_ZERO_COUNTS

    def __post_init__(self) -> None:
        if not isinstance(self.counts, Mapping) or set(self.counts) != set(
            self.REQUIRED_COUNTS
        ):
            raise DecisionRecoveryEvidenceError(
                "contract audit must contain the exact required counters"
            )
        object.__setattr__(
            self,
            "counts",
            MappingProxyType(
                {
                    name: _count(f"counts.{name}", self.counts[name])
                    for name in self.REQUIRED_COUNTS
                }
            ),
        )
        for name in (
            "leakage_clean",
            "feature_schema_exact",
            "deterministic",
        ):
            object.__setattr__(
                self, name, _flag(name, getattr(self, name))
            )

    @property
    def probes_exercised(self) -> bool:
        return all(
            self.counts[name] > 0 for name in self.REQUIRED_POSITIVE_COUNTS
        )

    @property
    def unsafe_counts_zero(self) -> bool:
        return all(
            self.counts[name] == 0 for name in self.REQUIRED_ZERO_COUNTS
        )


@dataclass(frozen=True)
class DecisionRecoveryEvidence:
    """All aggregate, revision-bound inputs required for WO-18 promotion."""

    source_revision_sha: str
    layout_manifest_sha256: str
    protected_role_manifest_sha256: str
    input_tree_sha256: str
    cohort_set_sha256: str
    truth_sha256: str
    official_evaluator_sha256: str
    feature_schema_sha256: str
    artifact_set_sha256: str
    expected_record_count: int
    expected_layout_group_count: int
    full_runs: Mapping[str, ApproachRunAggregate]
    repeat_scores: tuple[ScoreSlice, ...]
    fold_scores: tuple[ScoreSlice, ...]
    protected_role_scores: tuple[ProtectedRoleScore, ...]
    contract_audit: DecisionRecoveryContractAudit
    new_catastrophic_false_approval_count: int
    new_false_positive_denial_count: int
    manifest_frozen_before_scoring: bool
    roles_frozen_before_scoring: bool
    group_exclusive: bool
    paired_fold_members: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_revision_sha",
            _revision(self.source_revision_sha),
        )
        for name in (
            "layout_manifest_sha256",
            "protected_role_manifest_sha256",
            "input_tree_sha256",
            "cohort_set_sha256",
            "truth_sha256",
            "official_evaluator_sha256",
            "feature_schema_sha256",
            "artifact_set_sha256",
        ):
            object.__setattr__(
                self, name, _digest(name, getattr(self, name))
            )
        for name in (
            "expected_record_count",
            "expected_layout_group_count",
        ):
            object.__setattr__(
                self, name, _count(name, getattr(self, name), positive=True)
            )
        object.__setattr__(
            self,
            "full_runs",
            _approach_mapping(
                "full_runs",
                self.full_runs,
                lambda name, value: (
                    value
                    if isinstance(value, ApproachRunAggregate)
                    else (_ for _ in ()).throw(
                        DecisionRecoveryEvidenceError(
                            f"{name} must be an ApproachRunAggregate"
                        )
                    )
                ),
            ),
        )
        if not isinstance(self.repeat_scores, tuple) or not all(
            isinstance(value, ScoreSlice) for value in self.repeat_scores
        ):
            raise DecisionRecoveryEvidenceError(
                "repeat_scores must contain ScoreSlice values"
            )
        if not isinstance(self.fold_scores, tuple) or not all(
            isinstance(value, ScoreSlice) for value in self.fold_scores
        ):
            raise DecisionRecoveryEvidenceError(
                "fold_scores must contain ScoreSlice values"
            )
        if not isinstance(self.protected_role_scores, tuple) or not all(
            isinstance(value, ProtectedRoleScore)
            for value in self.protected_role_scores
        ):
            raise DecisionRecoveryEvidenceError(
                "protected_role_scores must contain ProtectedRoleScore values"
            )
        for name in (
            "new_catastrophic_false_approval_count",
            "new_false_positive_denial_count",
        ):
            object.__setattr__(
                self, name, _count(name, getattr(self, name))
            )
        for name in (
            "manifest_frozen_before_scoring",
            "roles_frozen_before_scoring",
            "group_exclusive",
            "paired_fold_members",
        ):
            object.__setattr__(
                self, name, _flag(name, getattr(self, name))
            )


@dataclass(frozen=True)
class DecisionRecoveryGateDecision:
    """Immutable gate result with aggregate-only persistence."""

    passed: bool
    gate_results: tuple[tuple[str, bool], ...]
    evidence: DecisionRecoveryEvidence

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.gate_results if not passed)

    def to_aggregate_evidence(self) -> dict[str, Any]:
        evidence = self.evidence
        full = evidence.full_runs
        control = full[CONTROL_APPROACH]
        candidate = full[CANDIDATE_APPROACH]
        full_delta = float(
            Decimal(str(candidate.total_score))
            - Decimal(str(control.total_score))
        )
        repeat_deltas = [value.candidate_delta for value in evidence.repeat_scores]
        fold_deltas = [value.candidate_delta for value in evidence.fold_scores]
        role_deltas = [
            value.candidate_delta for value in evidence.protected_role_scores
        ]
        class_metrics = {
            approach: {
                "total_score": run.total_score,
                "classification_score": run.classification_score,
                "record_count": run.record_count,
                "catastrophic_false_approvals": (
                    run.catastrophic_false_approvals
                ),
                "false_positive_denial_count": run.false_positive_denials,
                "missing_records": run.missing_records,
                "invalid_records": run.invalid_records,
                "deterministic": run.deterministic,
            }
            for approach, run in full.items()
        }
        role_metrics: dict[str, dict[str, float | int]] = {}
        for role in PROTECTED_ROLES:
            values = [
                score
                for score in evidence.protected_role_scores
                if score.role == role
            ]
            deltas = [value.candidate_delta for value in values]
            role_metrics[role] = {
                "repeat_count": len(values),
                "record_count": min(value.record_count for value in values),
                "score_delta_min": min(deltas),
                "score_delta_mean": sum(deltas) / len(deltas),
            }
        gates = dict(self.gate_results)
        aggregate: dict[str, Any] = {
            "evaluation_mode": PUBLIC_EXPOSED_SCOPE_LABEL,
            "evidence_label": "aggregate_only",
            "status": "passed" if self.passed else "blocked",
            "source_revision_sha": evidence.source_revision_sha,
            "layout_manifest_sha256": evidence.layout_manifest_sha256,
            "protected_role_manifest_sha256": (
                evidence.protected_role_manifest_sha256
            ),
            "input_tree_sha256": evidence.input_tree_sha256,
            "cohort_set_sha256": evidence.cohort_set_sha256,
            "truth_sha256": evidence.truth_sha256,
            "official_evaluator_sha256": evidence.official_evaluator_sha256,
            "feature_schema_sha256": evidence.feature_schema_sha256,
            "artifact_set_sha256": evidence.artifact_set_sha256,
            "approach_count": len(APPROACHES),
            "expected_record_count": evidence.expected_record_count,
            "expected_layout_group_count": (
                evidence.expected_layout_group_count
            ),
            "repeat_count": REQUIRED_REPEATS,
            "fold_count": REQUIRED_REPEATS * REQUIRED_FOLDS,
            "folds_per_repeat_count": REQUIRED_FOLDS,
            "protected_role_count": len(PROTECTED_ROLES),
            "full_control_score": control.total_score,
            "full_candidate_score": candidate.total_score,
            "score_delta": full_delta,
            "repeat_score_delta_min": min(repeat_deltas),
            "fold_score_delta_min": min(fold_deltas),
            "protected_role_score_delta_min": min(role_deltas),
            "positive_repeat_count": sum(value > 0 for value in repeat_deltas),
            "positive_fold_count": sum(value > 0 for value in fold_deltas),
            "positive_protected_role_repeat_count": sum(
                value > 0 for value in role_deltas
            ),
            "new_catastrophic_false_approval_count": (
                evidence.new_catastrophic_false_approval_count
            ),
            "new_false_positive_denial_count": (
                evidence.new_false_positive_denial_count
            ),
            "leakage_finding_count": (
                evidence.contract_audit.counts[
                    "forbidden_feature_finding_count"
                ]
            ),
            "hard_gate_failure_count": len(self.blocking_reasons),
            "class_metrics": class_metrics,
            "field_metrics": role_metrics,
            "counts": dict(evidence.contract_audit.counts),
            "gate_results": gates,
            "checks": {
                "all_four_approaches_compared": len(full) == len(APPROACHES),
                "positive_every_repeat": all(value > 0 for value in repeat_deltas),
                "positive_every_fold": all(value > 0 for value in fold_deltas),
                "positive_every_protected_role": all(
                    value > 0 for value in role_deltas
                ),
                "leakage_clean": evidence.contract_audit.leakage_clean,
                "feature_schema_exact": (
                    evidence.contract_audit.feature_schema_exact
                ),
            },
        }
        require_aggregate_only(aggregate)
        return aggregate


class DecisionRecoveryGate:
    """Apply every hard WO-18 comparison, safety, and leakage rule."""

    @staticmethod
    def _complete(evidence: DecisionRecoveryEvidence) -> bool:
        return all(
            run.complete_and_valid
            and run.deterministic
            and run.record_count == evidence.expected_record_count
            for run in evidence.full_runs.values()
        )

    @staticmethod
    def _repeat_shape(evidence: DecisionRecoveryEvidence) -> bool:
        return (
            len(evidence.repeat_scores) == REQUIRED_REPEATS
            and {
                value.repeat for value in evidence.repeat_scores
            }
            == set(range(REQUIRED_REPEATS))
            and all(
                value.fold is None
                and value.record_count == evidence.expected_record_count
                and value.layout_group_count
                == evidence.expected_layout_group_count
                for value in evidence.repeat_scores
            )
        )

    @staticmethod
    def _fold_shape(evidence: DecisionRecoveryEvidence) -> bool:
        expected = {
            (repeat, fold)
            for repeat in range(REQUIRED_REPEATS)
            for fold in range(REQUIRED_FOLDS)
        }
        actual = {
            (value.repeat, value.fold) for value in evidence.fold_scores
        }
        if len(evidence.fold_scores) != len(expected) or actual != expected:
            return False
        return all(
            sum(
                value.record_count
                for value in evidence.fold_scores
                if value.repeat == repeat
            )
            == evidence.expected_record_count
            and sum(
                value.layout_group_count
                for value in evidence.fold_scores
                if value.repeat == repeat
            )
            == evidence.expected_layout_group_count
            for repeat in range(REQUIRED_REPEATS)
        )

    @staticmethod
    def _role_shape(evidence: DecisionRecoveryEvidence) -> bool:
        expected = {
            (role, repeat)
            for role in PROTECTED_ROLES
            for repeat in range(REQUIRED_REPEATS)
        }
        actual = {
            (value.role, value.repeat)
            for value in evidence.protected_role_scores
        }
        return (
            len(evidence.protected_role_scores) == len(expected)
            and actual == expected
        )

    def evaluate(
        self, evidence: DecisionRecoveryEvidence
    ) -> DecisionRecoveryGateDecision:
        repeat_shape = self._repeat_shape(evidence)
        fold_shape = self._fold_shape(evidence)
        role_shape = self._role_shape(evidence)
        repeat_positive = repeat_shape and all(
            value.candidate_delta > 0 for value in evidence.repeat_scores
        )
        fold_positive = fold_shape and all(
            value.candidate_delta > 0 for value in evidence.fold_scores
        )
        role_positive = role_shape and all(
            value.candidate_delta > 0
            for value in evidence.protected_role_scores
        )
        control = evidence.full_runs[CONTROL_APPROACH]
        candidate = evidence.full_runs[CANDIDATE_APPROACH]
        gates = (
            (
                "frozen_manifests",
                evidence.manifest_frozen_before_scoring
                and evidence.roles_frozen_before_scoring,
            ),
            (
                "grouped_split_integrity",
                evidence.group_exclusive
                and evidence.paired_fold_members
                and repeat_shape
                and fold_shape,
            ),
            ("all_four_approaches_complete", self._complete(evidence)),
            (
                "positive_full_gain",
                candidate.total_score > control.total_score,
            ),
            ("positive_gain_every_repeat", repeat_positive),
            ("positive_gain_every_fold", fold_positive),
            ("positive_gain_every_protected_role", role_positive),
            (
                "no_new_catastrophic_false_approval",
                evidence.new_catastrophic_false_approval_count == 0,
            ),
            (
                "no_new_false_positive_denial",
                evidence.new_false_positive_denial_count == 0,
            ),
            (
                "candidate_has_no_catastrophic_false_approval",
                candidate.catastrophic_false_approvals == 0,
            ),
            (
                "contract_probes_exercised",
                evidence.contract_audit.probes_exercised,
            ),
            (
                "contract_unsafe_counts_zero",
                evidence.contract_audit.unsafe_counts_zero,
            ),
            (
                "identity_free_feature_schema",
                evidence.contract_audit.feature_schema_exact
                and evidence.contract_audit.leakage_clean,
            ),
            (
                "contract_audit_deterministic",
                evidence.contract_audit.deterministic,
            ),
            ("protected_role_shape_complete", role_shape),
        )
        return DecisionRecoveryGateDecision(
            passed=all(passed for _, passed in gates),
            gate_results=gates,
            evidence=evidence,
        )
