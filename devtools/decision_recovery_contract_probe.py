#!/usr/bin/env python3
"""Execute the identity-free WO-18 hard-ordering and model contract probes."""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from devtools.decision_recovery_gate import (  # noqa: E402
    DecisionRecoveryContractAudit,
)
from devtools.experiment_control import canonical_json
from mib_pipeline.model_recovery import (
    FEATURE_NAMES,
    MODEL_CLASSES,
    GatedHybridDecisionRule,
    IdentityFreeDecisionFeatures,
    ModelPrediction,
)


CONTRACT_AUDIT_SCHEMA = "mib-wo18-contract-audit/v1"
CONTRACT_FIXTURE_SCHEMA = "mib-wo18-contract-fixture/v1"


def contract_fixture_payload() -> dict[str, Any]:
    """Return the exact identity-free synthetic scenarios executed below."""

    return {
        "schema_version": CONTRACT_FIXTURE_SCHEMA,
        "feature_schema_sha256": hashlib.sha256(
            canonical_json(list(FEATURE_NAMES)).encode("utf-8")
        ).hexdigest(),
        "scenarios": [
            "binding_authority_precedes_model",
            "visible_disqualifier_precedes_model",
            "deterministic_policy_precedes_residual",
            "complete_visible_approval_recovery",
            "approval_scope_conflict_watermark_vetoes",
            "denial_requires_visible_violation_or_binding",
            "topology_and_missingness_never_create_denial",
            "uncertainty_margin_and_disagreement_return_review",
            "three_class_probability_simplex_margin_disagreement",
        ],
    }


def _features(**changes: float) -> IdentityFreeDecisionFeatures:
    values = {name: 0.0 for name in FEATURE_NAMES}
    values.update(
        {
            "baseline_review": 1.0,
            "resolved_fraction": 1.0,
            "visible_fraction": 1.0,
            "exact_case_scope_fraction": 1.0,
            "exact_subject_scope_fraction": 1.0,
            "clean_fraction": 1.0,
            "provenance_complete_fraction": 1.0,
            "link_confidence": 1.0,
            "route_primary": 1.0,
        }
    )
    values.update(changes)
    return IdentityFreeDecisionFeatures.from_mapping(
        {name: values[name] for name in FEATURE_NAMES}
    )


def _prediction(
    approved: float,
    denied: float,
    review: float,
    *,
    members: Sequence[Mapping[str, float]] | None = None,
) -> ModelPrediction:
    probabilities = {
        "APPROVED": approved,
        "DENIED": denied,
        "NEEDS_REVIEW": review,
    }
    member_values = tuple(members or (probabilities,))
    ordered = sorted(probabilities.values(), reverse=True)
    disagreement = max(
        max(member[name] for member in member_values)
        - min(member[name] for member in member_values)
        for name in MODEL_CLASSES
    )
    return ModelPrediction(
        decision=max(
            MODEL_CLASSES,
            key=lambda name: (
                probabilities[name],
                {"APPROVED": 0, "DENIED": 1, "NEEDS_REVIEW": 2}[name],
            ),
        ),
        probabilities=probabilities,
        margin=ordered[0] - ordered[1],
        disagreement=disagreement,
        member_probabilities=member_values,
    )


def run_contract_probes() -> dict[str, int]:
    """Run every named probe and return only reproducible aggregate counts."""

    counts = {
        name: 0 for name in DecisionRecoveryContractAudit.REQUIRED_COUNTS
    }
    rule = GatedHybridDecisionRule()
    approve = _prediction(0.96, 0.01, 0.03)
    deny = _prediction(0.01, 0.96, 0.03)
    uncertain = _prediction(0.34, 0.33, 0.33)

    for decision, flag in (
        ("APPROVED", "binding_approval"),
        ("DENIED", "binding_denial"),
        ("NEEDS_REVIEW", "binding_review"),
    ):
        counts["binding_authority_probe_count"] += 1
        result = rule.decide(
            "NEEDS_REVIEW",
            _features(**{flag: 1.0}),
            deny if decision != "DENIED" else approve,
        )
        counts["hard_ordering_failure_count"] += int(
            result.decision != decision
        )

    counts["visible_disqualifier_probe_count"] += 1
    visible_denial = rule.decide(
        "NEEDS_REVIEW",
        _features(policy_explicit_violation=1.0),
        approve,
    )
    counts["hard_ordering_failure_count"] += int(
        visible_denial.decision != "DENIED"
    )

    for baseline in ("APPROVED", "DENIED"):
        counts["deterministic_policy_probe_count"] += 1
        result = rule.decide(
            baseline,
            _features(
                baseline_review=0.0,
                baseline_approved=float(baseline == "APPROVED"),
                baseline_denied=float(baseline == "DENIED"),
            ),
            deny if baseline == "APPROVED" else approve,
        )
        counts["hard_ordering_failure_count"] += int(
            result.decision != baseline
        )

    counts["residual_recovery_probe_count"] += 1
    complete_approval = rule.decide(
        "NEEDS_REVIEW", _features(), approve
    )
    counts["approval_complete_visible_probe_count"] += 1
    counts["approval_guard_failure_count"] += int(
        complete_approval.decision != "APPROVED"
    )

    approval_vetoes = (
        (
            "approval_scope_probe_count",
            {"exact_subject_scope_fraction": 0.0},
        ),
        (
            "approval_scope_probe_count",
            {"link_confidence": 0.0, "unresolved_linkage": 1.0},
        ),
        (
            "approval_conflict_probe_count",
            {"packet_conflict": 1.0},
        ),
        (
            "approval_watermark_probe_count",
            {"packet_watermark": 1.0},
        ),
        (
            "approval_complete_visible_probe_count",
            {"visible_fraction": 0.8},
        ),
    )
    for counter, changes in approval_vetoes:
        counts[counter] += 1
        result = rule.decide(
            "NEEDS_REVIEW", _features(**changes), approve
        )
        counts["approval_guard_failure_count"] += int(
            result.decision != "NEEDS_REVIEW"
        )

    counts["denial_visible_violation_probe_count"] += 1
    counts["denial_binding_authority_probe_count"] += 1
    if visible_denial.decision != "DENIED":
        counts["denial_guard_failure_count"] += 1
    binding_denial = rule.decide(
        "NEEDS_REVIEW", _features(binding_denial=1.0), approve
    )
    counts["denial_guard_failure_count"] += int(
        binding_denial.decision != "DENIED"
    )

    topology_only = rule.decide(
        "NEEDS_REVIEW",
        _features(
            fee_page_present=1.0,
            attestation_page_present=1.0,
            biometric_evidence_present=1.0,
        ),
        deny,
    )
    counts["denial_topology_only_probe_count"] += 1
    counts["denial_guard_failure_count"] += int(
        topology_only.decision != "NEEDS_REVIEW"
    )
    counts["denial_without_visible_violation_count"] += int(
        topology_only.decision == "DENIED"
    )

    missingness_only = rule.decide(
        "NEEDS_REVIEW",
        _features(
            resolved_fraction=0.4,
            visible_fraction=0.4,
            clean_fraction=0.4,
            unknown_fraction=0.6,
            policy_review_gap=1.0,
        ),
        deny,
    )
    counts["denial_missingness_only_probe_count"] += 1
    counts["denial_guard_failure_count"] += int(
        missingness_only.decision != "NEEDS_REVIEW"
    )
    counts["denial_without_visible_violation_count"] += int(
        missingness_only.decision == "DENIED"
    )

    counts["uncertainty_review_probe_count"] += 1
    counts["model_margin_probe_count"] += 1
    margin_result = rule.decide(
        "NEEDS_REVIEW", _features(), uncertain
    )
    counts["hard_ordering_failure_count"] += int(
        margin_result.decision != "NEEDS_REVIEW"
    )

    members = (
        {"APPROVED": 0.91, "DENIED": 0.01, "NEEDS_REVIEW": 0.08},
        {"APPROVED": 0.61, "DENIED": 0.01, "NEEDS_REVIEW": 0.38},
    )
    average = {
        name: sum(member[name] for member in members) / len(members)
        for name in MODEL_CLASSES
    }
    disagreement_prediction = _prediction(
        average["APPROVED"],
        average["DENIED"],
        average["NEEDS_REVIEW"],
        members=members,
    )
    counts["ensemble_disagreement_probe_count"] += 1
    disagreement_result = rule.decide(
        "NEEDS_REVIEW", _features(), disagreement_prediction
    )
    counts["hard_ordering_failure_count"] += int(
        disagreement_result.decision != "NEEDS_REVIEW"
    )

    counts["probability_simplex_probe_count"] += 4
    counts["true_margin_probe_count"] += 2
    for scenario, expected_fragment in (
        (
            {
                "decision": "APPROVED",
                "probabilities": {
                    "APPROVED": math.nan,
                    "DENIED": 0.0,
                    "NEEDS_REVIEW": 1.0,
                },
                "margin": 1.0,
                "disagreement": 0.0,
            },
            "finite",
        ),
        (
            {
                "decision": "APPROVED",
                "probabilities": {
                    "APPROVED": 0.8,
                    "DENIED": 0.1,
                    "NEEDS_REVIEW": 0.2,
                },
                "margin": 0.6,
                "disagreement": 0.0,
            },
            "sum",
        ),
        (
            {
                "decision": "APPROVED",
                "probabilities": {
                    "APPROVED": 0.8,
                    "DENIED": 0.1,
                    "NEEDS_REVIEW": 0.1,
                },
                "margin": 0.1,
                "disagreement": 0.0,
            },
            "margin",
        ),
    ):
        try:
            ModelPrediction(
                **scenario,
                member_probabilities=(scenario["probabilities"],),
            )
        except (TypeError, ValueError) as exc:
            message = str(exc).casefold()
            accepted_invalid = expected_fragment not in message
        else:
            accepted_invalid = True
        if expected_fragment == "finite":
            counts["nonfinite_probability_count"] += int(accepted_invalid)
        elif expected_fragment == "sum":
            counts["unnormalized_probability_count"] += int(
                accepted_invalid
            )
        else:
            counts["incorrect_margin_count"] += int(accepted_invalid)

    try:
        ModelPrediction(
            decision=disagreement_prediction.decision,
            probabilities=disagreement_prediction.probabilities,
            margin=disagreement_prediction.margin,
            disagreement=0.0,
            member_probabilities=members,
        )
    except ValueError as exc:
        accepted_invalid_disagreement = (
            "disagreement" not in str(exc).casefold()
        )
    else:
        accepted_invalid_disagreement = True
    counts["invalid_disagreement_count"] += int(
        accepted_invalid_disagreement
    )
    return counts


def build_contract_audit(
    *,
    repeat_index: int,
    source_revision_sha: str,
    layout_manifest_sha256: str,
    protected_role_manifest_sha256: str,
    input_tree_sha256: str,
    contract_fixture_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_AUDIT_SCHEMA,
        "repeat_index": repeat_index,
        "source_revision_sha": source_revision_sha,
        "layout_manifest_sha256": layout_manifest_sha256,
        "protected_role_manifest_sha256": (
            protected_role_manifest_sha256
        ),
        "input_tree_sha256": input_tree_sha256,
        "feature_schema_sha256": contract_fixture_payload()[
            "feature_schema_sha256"
        ],
        "feature_names": list(FEATURE_NAMES),
        "contract_fixture_sha256": contract_fixture_sha256,
        "counts": run_contract_probes(),
        "deterministic": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision-sha", required=True)
    parser.add_argument("--layout-manifest-sha256", required=True)
    parser.add_argument("--protected-role-manifest-sha256", required=True)
    parser.add_argument("--input-tree-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = arguments.output_dir / "contract-fixture.json"
    fixture_path.write_text(
        canonical_json(contract_fixture_payload()) + "\n",
        encoding="utf-8",
    )
    fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    for repeat in (1, 2):
        audit = build_contract_audit(
            repeat_index=repeat,
            source_revision_sha=arguments.source_revision_sha,
            layout_manifest_sha256=arguments.layout_manifest_sha256,
            protected_role_manifest_sha256=(
                arguments.protected_role_manifest_sha256
            ),
            input_tree_sha256=arguments.input_tree_sha256,
            contract_fixture_sha256=fixture_sha,
        )
        (arguments.output_dir / f"contract-audit-{repeat}.json").write_text(
            canonical_json(audit) + "\n",
            encoding="utf-8",
        )
    print(str(arguments.output_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
