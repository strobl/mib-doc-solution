"""Fail-closed aggregate acceptance gate for WO-17 policy revalidation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping

from devtools.evaluation import PUBLIC_ROBUSTNESS_EVIDENCE
from devtools.experiment_control import (
    ExperimentControlError,
    require_aggregate_only,
)
from devtools.policy_revalidation_audit_contract import (
    CONTRACT_AUDIT_COUNTS,
    COHORT_AUDIT_COUNTS,
    UNSAFE_COHORT_AUDIT_COUNTS,
    UNSAFE_CONTRACT_AUDIT_COUNTS,
)


PUBLIC_EXPOSED_SCOPE_LABEL = PUBLIC_ROBUSTNESS_EVIDENCE
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class PolicyRevalidationEvidenceError(ExperimentControlError):
    """WO-17 evidence is incomplete, unsafe, or internally inconsistent."""


def _count(name: str, value: Any, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise PolicyRevalidationEvidenceError(
            f"{name} must be a {qualifier} integer"
        )
    return value


def _score(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyRevalidationEvidenceError(f"{name} must be a finite score")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise PolicyRevalidationEvidenceError(f"{name} must be a finite score")
    return normalized


def _flag(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise PolicyRevalidationEvidenceError(f"{name} must be a boolean")
    return value


@dataclass(frozen=True)
class PolicyRunAggregate:
    """Official-evaluator and schema metrics for one prediction arm."""

    total_score: float
    extraction_score: float
    classification_score: float
    calibration_score: float
    record_count: int
    catastrophic_false_approvals: int
    false_positive_denials: int
    missing_records: int
    invalid_records: int
    duplicate_records: int
    extra_records: int
    approved_to_needs_review_count: int
    denied_to_needs_review_count: int
    deterministic: bool

    def __post_init__(self) -> None:
        for name in (
            "total_score",
            "extraction_score",
            "classification_score",
            "calibration_score",
        ):
            object.__setattr__(self, name, _score(name, getattr(self, name)))
        for name in (
            "record_count",
            "catastrophic_false_approvals",
            "false_positive_denials",
            "missing_records",
            "invalid_records",
            "duplicate_records",
            "extra_records",
            "approved_to_needs_review_count",
            "denied_to_needs_review_count",
        ):
            object.__setattr__(self, name, _count(name, getattr(self, name)))
        object.__setattr__(
            self, "deterministic", _flag("deterministic", self.deterministic)
        )


@dataclass(frozen=True)
class PolicyExecutionAudit:
    """Deterministic cohort occurrence counters plus non-vacuous probes."""

    cohort_counts: Mapping[str, int]
    contract_counts: Mapping[str, int]
    deterministic: bool

    def __post_init__(self) -> None:
        cohort = self._validate_exact_counts(
            "cohort", self.cohort_counts, COHORT_AUDIT_COUNTS
        )
        contract = self._validate_exact_counts(
            "contract", self.contract_counts, CONTRACT_AUDIT_COUNTS
        )
        object.__setattr__(self, "cohort_counts", cohort)
        object.__setattr__(self, "contract_counts", contract)
        object.__setattr__(
            self, "deterministic", _flag("deterministic", self.deterministic)
        )

    @staticmethod
    def _validate_exact_counts(
        label: str,
        values: Mapping[str, int],
        required: tuple[str, ...],
    ) -> Mapping[str, int]:
        if not isinstance(values, Mapping) or set(values) != set(required):
            raise PolicyRevalidationEvidenceError(
                f"{label} audit counts must contain the exact required keys"
            )
        return {
            name: _count(f"{label}.{name}", values[name])
            for name in required
        }


@dataclass(frozen=True)
class PolicyRevalidationEvidence:
    """All aggregate inputs required for the WO-17 hard gate."""

    layout_manifest_sha256: str
    expected_record_count: int
    layout_group_count: int
    legacy_control: PolicyRunAggregate
    candidate: PolicyRunAggregate
    audit: PolicyExecutionAudit
    new_catastrophic_false_approval_count: int
    new_false_positive_denial_count: int
    new_missing_record_count: int
    new_invalid_record_count: int
    non_policy_field_change_count: int
    manifest_frozen_before_scoring: bool

    def __post_init__(self) -> None:
        digest = str(self.layout_manifest_sha256).strip().casefold()
        if not _SHA256_RE.fullmatch(digest):
            raise PolicyRevalidationEvidenceError(
                "layout_manifest_sha256 must be a full SHA-256 digest"
            )
        object.__setattr__(self, "layout_manifest_sha256", digest)
        for name in (
            "expected_record_count",
            "layout_group_count",
        ):
            object.__setattr__(
                self, name, _count(name, getattr(self, name), positive=True)
            )
        for name in (
            "new_catastrophic_false_approval_count",
            "new_false_positive_denial_count",
            "new_missing_record_count",
            "new_invalid_record_count",
            "non_policy_field_change_count",
        ):
            object.__setattr__(self, name, _count(name, getattr(self, name)))
        object.__setattr__(
            self,
            "manifest_frozen_before_scoring",
            _flag(
                "manifest_frozen_before_scoring",
                self.manifest_frozen_before_scoring,
            ),
        )


@dataclass(frozen=True)
class PolicyRevalidationGateDecision:
    """Immutable decision with a repository-safe aggregate serialization."""

    passed: bool
    gate_results: tuple[tuple[str, bool], ...]
    evidence: PolicyRevalidationEvidence

    @property
    def blocking_reasons(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.gate_results if not passed)

    def to_aggregate_evidence(self) -> dict[str, Any]:
        evidence = self.evidence
        control = evidence.legacy_control
        candidate = evidence.candidate
        cohort = evidence.audit.cohort_counts
        contract = evidence.audit.contract_counts
        approved_delta = (
            candidate.approved_to_needs_review_count
            - control.approved_to_needs_review_count
        )
        denied_delta = (
            candidate.denied_to_needs_review_count
            - control.denied_to_needs_review_count
        )
        gates = dict(self.gate_results)
        aggregate: dict[str, Any] = {
            "evaluation_mode": PUBLIC_EXPOSED_SCOPE_LABEL,
            "evidence_label": "aggregate_only",
            "status": "passed" if self.passed else "blocked",
            "layout_manifest_sha256": evidence.layout_manifest_sha256,
            "expected_record_count": evidence.expected_record_count,
            "layout_group_count": evidence.layout_group_count,
            "full_control_score": control.total_score,
            "full_candidate_score": candidate.total_score,
            "score_delta": float(
                Decimal(str(candidate.total_score))
                - Decimal(str(control.total_score))
            ),
            "extraction_score_delta": float(
                Decimal(str(candidate.extraction_score))
                - Decimal(str(control.extraction_score))
            ),
            "classification_score_delta": float(
                Decimal(str(candidate.classification_score))
                - Decimal(str(control.classification_score))
            ),
            "calibration_score_delta": float(
                Decimal(str(candidate.calibration_score))
                - Decimal(str(control.calibration_score))
            ),
            "catastrophic_false_approvals": (
                candidate.catastrophic_false_approvals
            ),
            "catastrophic_false_approvals_delta": (
                candidate.catastrophic_false_approvals
                - control.catastrophic_false_approvals
            ),
            "false_positive_denials_delta": (
                candidate.false_positive_denials
                - control.false_positive_denials
            ),
            "new_catastrophic_false_approval_count": (
                evidence.new_catastrophic_false_approval_count
            ),
            "new_false_positive_denial_count": (
                evidence.new_false_positive_denial_count
            ),
            "new_missing_record_count": evidence.new_missing_record_count,
            "new_invalid_record_count": evidence.new_invalid_record_count,
            "missing_records": candidate.missing_records,
            "invalid_records": candidate.invalid_records,
            "duplicate_records": candidate.duplicate_records,
            "extra_records": candidate.extra_records,
            "non_policy_field_change_count": (
                evidence.non_policy_field_change_count
            ),
            "legacy_control_deterministic": control.deterministic,
            "candidate_deterministic": candidate.deterministic,
            "audit_deterministic": evidence.audit.deterministic,
            "hard_gate_failure_count": len(self.blocking_reasons),
            "confusion_counts": {
                "approved_to_needs_review_control_count": (
                    control.approved_to_needs_review_count
                ),
                "approved_to_needs_review_candidate_count": (
                    candidate.approved_to_needs_review_count
                ),
                "approved_to_needs_review_delta": approved_delta,
                "denied_to_needs_review_control_count": (
                    control.denied_to_needs_review_count
                ),
                "denied_to_needs_review_candidate_count": (
                    candidate.denied_to_needs_review_count
                ),
                "denied_to_needs_review_delta": denied_delta,
            },
            "counts": {
                **{
                    f"cohort_{name}": value
                    for name, value in cohort.items()
                },
                **{
                    f"contract_{name}": value
                    for name, value in contract.items()
                },
            },
            "gate_results": gates,
            "checks": {
                "targeted_confusion_changed": (
                    approved_delta != 0 or denied_delta != 0
                ),
                "cohort_revalidation_observed": (
                    cohort["revalidation_after_late_recovery_count"] > 0
                ),
                "cohort_contradiction_observed": (
                    cohort[
                        "contradicted_synthetic_reason_before_count"
                    ]
                    > 0
                ),
            },
        }
        require_aggregate_only(aggregate)
        return aggregate


class PolicyRevalidationGate:
    """Apply every hard WO-17 execution-order and safety acceptance rule."""

    @staticmethod
    def _complete(
        run: PolicyRunAggregate, expected_record_count: int
    ) -> bool:
        return (
            run.record_count == expected_record_count
            and run.missing_records == 0
            and run.invalid_records == 0
            and run.duplicate_records == 0
            and run.extra_records == 0
        )

    def evaluate(
        self, evidence: PolicyRevalidationEvidence
    ) -> PolicyRevalidationGateDecision:
        control = evidence.legacy_control
        candidate = evidence.candidate
        cohort = evidence.audit.cohort_counts
        contract = evidence.audit.contract_counts
        approved_delta = (
            candidate.approved_to_needs_review_count
            - control.approved_to_needs_review_count
        )
        denied_delta = (
            candidate.denied_to_needs_review_count
            - control.denied_to_needs_review_count
        )

        cohort_order_consistent = (
            cohort["late_recovery_before_revalidation_count"]
            == cohort["revalidation_after_late_recovery_count"]
            == cohort["normal_policy_rerun_count"]
        )
        cohort_contradiction_accounted = (
            cohort["contradicted_synthetic_reason_before_count"]
            == cohort["contradicted_synthetic_reason_removed_count"]
            + cohort["contradicted_synthetic_reason_remaining_count"]
        )
        contract_order_consistent = (
            contract["candidate_late_recovery_before_revalidation_count"]
            == contract["candidate_revalidation_after_late_recovery_count"]
            == contract["normal_policy_rerun_count"]
        )
        contract_contradiction_accounted = (
            contract["contradicted_synthetic_reason_before_count"]
            == contract["contradicted_synthetic_reason_removed_count"]
            + contract["contradicted_synthetic_reason_remaining_count"]
        )

        gates: Mapping[str, bool] = {
            "public_exposed_evidence": (
                PUBLIC_EXPOSED_SCOPE_LABEL
                == "public_grouped_robustness_not_unseen"
            ),
            "manifest_frozen_before_scoring": (
                evidence.manifest_frozen_before_scoring
            ),
            "candidate_complete": self._complete(
                candidate, evidence.expected_record_count
            ),
            "legacy_control_complete": self._complete(
                control, evidence.expected_record_count
            ),
            "repeated_runs_deterministic": (
                control.deterministic and candidate.deterministic
            ),
            "execution_audits_deterministic": evidence.audit.deterministic,
            "score_non_regression": candidate.total_score >= control.total_score,
            "extraction_score_unchanged": (
                candidate.extraction_score == control.extraction_score
            ),
            "non_policy_fields_unchanged": (
                evidence.non_policy_field_change_count == 0
            ),
            "approved_to_review_non_regression": approved_delta <= 0,
            "denied_to_review_non_regression": denied_delta <= 0,
            "no_new_catastrophic_false_approvals": (
                evidence.new_catastrophic_false_approval_count == 0
                and candidate.catastrophic_false_approvals
                <= control.catastrophic_false_approvals
            ),
            "no_new_false_positive_denials": (
                evidence.new_false_positive_denial_count == 0
                and candidate.false_positive_denials
                <= control.false_positive_denials
            ),
            "no_new_missing_records": evidence.new_missing_record_count == 0,
            "no_new_invalid_records": evidence.new_invalid_record_count == 0,
            "accepted_final_policy_results_complete": (
                cohort["accepted_final_policy_result_count"]
                == evidence.expected_record_count
            ),
            "cohort_execution_order_accounted": cohort_order_consistent,
            "cohort_contradictions_accounted": (
                cohort_contradiction_accounted
            ),
            "cohort_unsafe_counters_zero": all(
                cohort[name] == 0 for name in UNSAFE_COHORT_AUDIT_COUNTS
            ),
            "contract_legacy_order_exercised": (
                contract[
                    "legacy_synthetic_before_late_recovery_count"
                ]
                > 0
            ),
            "contract_candidate_order_exercised": (
                contract[
                    "candidate_late_recovery_before_revalidation_count"
                ]
                > 0
                and contract[
                    "candidate_revalidation_after_late_recovery_count"
                ]
                > 0
            ),
            "contract_execution_order_accounted": (
                contract_order_consistent
            ),
            "contract_contradiction_removed": (
                contract["contradicted_synthetic_reason_before_count"] > 0
                and contract["contradicted_synthetic_reason_removed_count"] > 0
                and contract_contradiction_accounted
            ),
            "contract_independent_denial_retained": (
                contract["independent_denial_reason_retained_count"] > 0
            ),
            "contract_review_confidence_restored": (
                contract["review_confidence_restored_count"] > 0
            ),
            "contract_signed_late_authority_recovered": (
                contract["signed_late_authority_recovery_count"] >= 3
            ),
            "contract_late_decision_evidence_preserved": (
                contract["late_adjudication_evidence_preserved_count"] >= 3
            ),
            "contract_late_biohazard_evidence_preserved": (
                contract["late_biohazard_evidence_preserved_count"] > 0
            ),
            "contract_placeholder_guards_exercised": (
                contract["placeholder_guard_probe_count"] >= 35
                and contract["sentinel_guard_probe_count"] >= 2
                and contract[
                    "serialization_default_guard_probe_count"
                ]
                >= 35
            ),
            "contract_policy_safety_guards_exercised": (
                contract["stale_threshold_guard_probe_count"] >= 2
                and contract["forced_approval_guard_probe_count"] >= 35
                and contract[
                    "direct_approval_head_guard_probe_count"
                ]
                >= 1
            ),
            "contract_unsafe_counters_zero": all(
                contract[name] == 0
                for name in UNSAFE_CONTRACT_AUDIT_COUNTS
            ),
        }
        return PolicyRevalidationGateDecision(
            passed=all(gates.values()),
            gate_results=tuple(gates.items()),
            evidence=evidence,
        )
