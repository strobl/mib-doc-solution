from __future__ import annotations

import json
import re

import pytest

from devtools.wo19_final_refit import (
    FAMILIES,
    FinalRefitError,
    _grouped_folds,
    build_report,
    render_markdown,
)


def _rows(count: int = 50):
    truth = {}
    parent = {}
    final = {}
    layout = {}
    classes = ("APPROVED", "DENIED", "NEEDS_REVIEW")
    for index in range(count):
        case_id = f"MIB-{index + 1:06d}"
        adjudication = classes[index % len(classes)]
        row = {
            "case_id": case_id,
            "applicant_name": "unknown",
            "species_code": "ARCTURIAN",
            "home_world": "Mars",
            "visa_class": "MED-3",
            "sponsor_id": "SPN-0000",
            "arrival_date": "1900-01-01",
            "declared_purpose": "unknown",
            "risk_flags": "none",
            "fee_status": "unknown",
            "adjudication": adjudication,
            "confidence": 0.55 + (index % 5) * 0.08,
        }
        truth[case_id] = {
            "case_id": case_id,
            "adjudication": (
                adjudication if index % 4 else classes[(index + 1) % len(classes)]
            ),
        }
        parent[case_id] = dict(row)
        final[case_id] = dict(row)
        layout[case_id] = f"layout-{index % 10}"
    return truth, parent, final, layout


def test_grouped_folds_are_complete_and_exclusive() -> None:
    groups = [f"group-{index % 10}" for index in range(50)]
    folds = _grouped_folds(groups, seed=19001)
    assert sorted(index for fold in folds for index in fold) == list(range(50))
    ownership = {}
    for fold_index, fold in enumerate(folds):
        for index in fold:
            ownership.setdefault(groups[index], set()).add(fold_index)
    assert all(len(folds_for_group) == 1 for folds_for_group in ownership.values())


def test_report_is_aggregate_only_and_compares_every_family() -> None:
    truth, parent, final, layout = _rows()
    report = build_report(
        truth_rows=truth,
        parent_rows=parent,
        final_rows=final,
        layout_groups=layout,
        source_revision="a" * 40,
        source_sha256={"truth": "b" * 64},
    )
    rendered = json.dumps(report, sort_keys=True)
    markdown = render_markdown(report)
    assert set(report["comparisons"]) == set(FAMILIES)
    assert report["hard_gates"]["non_confidence_bytes_unchanged"] is True
    assert not re.search(r"MIB-[0-9]{6}", rendered)
    assert not re.search(r"MIB-[0-9]{6}", markdown)
    assert "public-data robustness evidence" in markdown


def test_coverage_mismatch_fails_closed() -> None:
    truth, parent, final, layout = _rows()
    layout.pop(next(iter(layout)))
    with pytest.raises(FinalRefitError, match="coverage"):
        build_report(
            truth_rows=truth,
            parent_rows=parent,
            final_rows=final,
            layout_groups=layout,
            source_revision="a" * 40,
            source_sha256={"truth": "b" * 64},
        )
