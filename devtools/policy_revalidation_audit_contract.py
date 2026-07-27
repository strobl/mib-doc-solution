"""Aggregate audit contract for WO-17 late-recovery policy revalidation.

The production-cohort counters describe occurrences in the frozen public
cohort and may legitimately be zero.  The contract-probe counters are
different: they come from identity-free, purpose-built fixtures and must
exercise every required execution-order and safety branch.  Keeping these
names in one development-only module prevents the producer and evidence gate
from silently disagreeing about what was measured.
"""

from __future__ import annotations


POLICY_REVALIDATION_AUDIT_SCHEMA = "mib_policy_revalidation_audit_v1"

COHORT_AUDIT_COUNTS = (
    "accepted_final_policy_result_count",
    "late_recovery_before_revalidation_count",
    "revalidation_after_late_recovery_count",
    "contradicted_synthetic_reason_before_count",
    "contradicted_synthetic_reason_removed_count",
    "contradicted_synthetic_reason_remaining_count",
    "independent_denial_reason_retained_count",
    "review_confidence_restored_count",
    "normal_policy_rerun_count",
    "signed_late_authority_recovery_count",
    "late_adjudication_evidence_preserved_count",
    "late_biohazard_evidence_preserved_count",
    "forced_approval_count",
    "sentinel_value_used_as_evidence_count",
    "placeholder_value_used_as_evidence_count",
    "serialization_default_used_as_evidence_count",
    "stale_threshold_mismatch_count",
)

CONTRACT_AUDIT_COUNTS = (
    "legacy_synthetic_before_late_recovery_count",
    "candidate_late_recovery_before_revalidation_count",
    "candidate_revalidation_after_late_recovery_count",
    "contradicted_synthetic_reason_before_count",
    "contradicted_synthetic_reason_removed_count",
    "contradicted_synthetic_reason_remaining_count",
    "independent_denial_reason_retained_count",
    "review_confidence_restored_count",
    "normal_policy_rerun_count",
    "signed_late_authority_recovery_count",
    "late_adjudication_evidence_preserved_count",
    "late_biohazard_evidence_preserved_count",
    "placeholder_guard_probe_count",
    "sentinel_guard_probe_count",
    "serialization_default_guard_probe_count",
    "stale_threshold_guard_probe_count",
    "forced_approval_guard_probe_count",
    "direct_approval_head_guard_probe_count",
    "forced_approval_count",
    "sentinel_value_used_as_evidence_count",
    "placeholder_value_used_as_evidence_count",
    "serialization_default_used_as_evidence_count",
    "stale_threshold_mismatch_count",
)

UNSAFE_COHORT_AUDIT_COUNTS = (
    "contradicted_synthetic_reason_remaining_count",
    "forced_approval_count",
    "sentinel_value_used_as_evidence_count",
    "placeholder_value_used_as_evidence_count",
    "serialization_default_used_as_evidence_count",
    "stale_threshold_mismatch_count",
)

UNSAFE_CONTRACT_AUDIT_COUNTS = (
    "contradicted_synthetic_reason_remaining_count",
    "forced_approval_count",
    "sentinel_value_used_as_evidence_count",
    "placeholder_value_used_as_evidence_count",
    "serialization_default_used_as_evidence_count",
    "stale_threshold_mismatch_count",
)
