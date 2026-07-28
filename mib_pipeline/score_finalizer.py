"""Transfer-gated final scoring layer using visible PDF evidence only."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import score_heads
from .models import PredictionRow
from .score_confidence import apply_confidence_blend, apply_platt_calibration
from .visible_text import (
    VisibleOcrSnapshot,
    VisibleOcrTextStore,
    VisibleTextSnapshotError,
    page_text_projection,
    person_name_consensus_key,
    person_names_compatible,
)


_UNKNOWN_APPLICANTS = frozenset({"", "unknown", "n/a", "none", "null"})
_PAGE_SIGNATURE_CODES = {
    "fee_receipt": "F",
    "registry_extract": "R",
    "intake_form": "I",
    "biometric_slip": "B",
    # The frozen layout head's M bucket represented the extra structured
    # sponsor/medical support page. Use the explicit page classifier now.
    "sponsor_attestation": "M",
}


@dataclass(frozen=True)
class _ScopedVisibleEvidence:
    case_text: str
    identity_text: str
    page_signature: str
    sponsor_attestation_proven: bool
    fee_status_evidence: frozenset[str]
    non_intake_sponsor_ids: frozenset[str]
    intake_unreadable: bool


def _normalized_applicant(value: str | None) -> str:
    return " ".join(str(value or "").casefold().split())


def _applicants_compatible(visible: str, predicted: str) -> bool:
    predicted_name = _normalized_applicant(predicted)
    if predicted_name in _UNKNOWN_APPLICANTS:
        return True
    return person_names_compatible(visible, predicted)


def _page_applicant_cluster(page: object) -> str | None:
    """Return one narrow-consensus applicant key, or ``None`` if absent/ambiguous."""

    applicants = getattr(page, "visible_applicants", ())
    keys = {
        key
        for applicant in applicants
        if (key := person_name_consensus_key(applicant))
    }
    return next(iter(keys)) if len(keys) == 1 else None


def _consensus_visible_applicant(pages: tuple[object, ...]) -> str | None:
    """Select one applicant supported by two pages and two page categories.

    Intake+registry agreement is preferred because those are independent
    identity-bearing forms.  Multiple equally eligible clusters are ambiguity,
    not an invitation to choose a fuzzy nearest name.
    """

    support: dict[str, tuple[set[int], set[str]]] = {}
    for page in pages:
        key = _page_applicant_cluster(page)
        if key is None:
            continue
        page_indexes, categories = support.setdefault(key, (set(), set()))
        page_indexes.add(page.page_index)
        categories.add(page.page_category)
    eligible = {
        key: values
        for key, values in support.items()
        if len(values[0]) >= 2 and len(values[1]) >= 2
    }
    preferred = {
        key
        for key, (_page_indexes, categories) in eligible.items()
        if {"intake_form", "registry_extract"} <= categories
    }
    if len(preferred) == 1:
        return next(iter(preferred))
    if preferred:
        return None
    if len(eligible) == 1:
        return next(iter(eligible))
    return None


def _visible_fee_statuses(page: object) -> frozenset[str]:
    """Read exact fee facts from one already-scoped accepted OCR page."""

    text = page.text
    statuses = {
        match.casefold()
        for match in re.findall(
            r"\bFee\s*Status\s*[:#=.-]?\s*(paid|waived|unpaid|unknown)\b",
            text,
            re.I,
        )
    }
    amount_paid = bool(
        re.search(r"\bAmount\s*[:#=.-]?\s*\$?\s*809(?:[.,]00)?\b", text, re.I)
    )
    amount_zero = bool(
        re.search(r"\bAmount\s*[:#=.-]?\s*\$?\s*0(?:[.,]00)?\b", text, re.I)
    )
    waiver = bool(re.search(r"\bDIP[\s-]*WAIVER\b", text, re.I))
    if amount_paid:
        statuses.add("paid")
    if amount_zero and waiver:
        statuses.add("waived")
    elif amount_zero:
        statuses.add("unpaid")
    return frozenset(statuses)


def _normalized_sponsor_id(raw: str) -> str | None:
    match = re.search(r"SP[NM]\s*[-:]?\s*([0-9OQCDILSB\s]{4,10})", raw, re.I)
    if match is None:
        return None
    digits = match.group(1).translate(
        str.maketrans(
            {
                "O": "0",
                "Q": "0",
                "C": "0",
                "D": "0",
                "I": "1",
                "L": "1",
                "S": "5",
                "B": "8",
            }
        )
    )
    digits = re.sub(r"\D", "", digits)[:4]
    return f"SPN-{digits}" if len(digits) == 4 else None


def _non_intake_sponsor_ids(page: object) -> frozenset[str]:
    """Return exact sponsor-ID fields from a known non-intake page."""

    known_non_intake = (
        page.page_category
        in {
            "adjudicator_stamp",
            "biometric_slip",
            "fee_receipt",
            "registry_extract",
            "signed_manual_note",
            "sponsor_attestation",
        }
        or page.source_category != "intake_form"
    )
    if not known_non_intake:
        return frozenset()
    sponsor_ids = {
        normalized
        for line in page.lines
        for match in re.finditer(
            r"\bSponsor\s+ID\s*[:#=.-]?\s*(SP[NM]\s*[-:]?\s*[0-9OQCDILSB\s]{4,10})",
            line.text,
            re.I,
        )
        if (normalized := _normalized_sponsor_id(match.group(1))) is not None
    }
    sponsor_ids.update(
        attestation.sponsor_id for attestation in page.sponsor_attestations
    )
    return frozenset(sponsor_ids)


def _scoped_visible_evidence(
    snapshot: VisibleOcrSnapshot,
    row: PredictionRow,
) -> _ScopedVisibleEvidence:
    """Build fail-closed case and identity projections plus structured gates."""

    case_pages: set[int] = set()
    case_scoped_pages: list[object] = []
    for page in snapshot.pages:
        case_ids = page.visible_case_ids
        if len(case_ids) > 1:
            continue
        if len(case_ids) == 1 and row.case_id not in case_ids:
            continue
        case_pages.add(page.page_index)
        case_scoped_pages.append(page)

    case_scoped = tuple(case_scoped_pages)
    consensus_applicant = _consensus_visible_applicant(case_scoped)
    observed_clusters = {
        cluster
        for page in case_scoped
        if (cluster := _page_applicant_cluster(page)) is not None
    }
    row_applicant_unknown = (
        _normalized_applicant(row.applicant_name) in _UNKNOWN_APPLICANTS
    )
    identity_pages: set[int] = set()
    for page in case_scoped:
        applicant_key = _page_applicant_cluster(page)
        if page.visible_applicants and applicant_key is None:
            # Multiple incompatible names on one page are never identity proof.
            continue
        if applicant_key is None:
            identity_pages.add(page.page_index)
            continue
        if consensus_applicant is not None:
            if applicant_key == consensus_applicant:
                identity_pages.add(page.page_index)
            continue
        # If independent forms do not establish a new identity, preserve the
        # row-anchored fallback.  An unknown row with multiple visible clusters
        # remains fail-closed instead of mixing both applicants.
        if row_applicant_unknown and len(observed_clusters) > 1:
            continue
        if _applicants_compatible(
            next(iter(page.visible_applicants)),
            row.applicant_name,
        ):
            identity_pages.add(page.page_index)

    identity_page_indexes = frozenset(identity_pages)
    page_signature = "".join(
        _PAGE_SIGNATURE_CODES.get(page.page_category, "O")
        for page in snapshot.pages
        if page.page_index in identity_page_indexes and page.lines
    )
    attestations = tuple(
        attestation
        for page in snapshot.pages
        if page.page_index in identity_page_indexes
        for attestation in page.sponsor_attestations
    )
    sponsor_attestation_proven = (
        len(attestations) == 1
        and attestations[0].sponsor_id == row.sponsor_id
        and person_names_compatible(
            attestations[0].applicant_name,
            (
                consensus_applicant
                if consensus_applicant is not None
                else row.applicant_name
            ),
        )
    )
    identity_scoped_pages = tuple(
        page
        for page in snapshot.pages
        if page.page_index in identity_page_indexes
    )
    fee_status_evidence = frozenset(
        status
        for page in identity_scoped_pages
        for status in _visible_fee_statuses(page)
    )
    non_intake_sponsor_ids = frozenset(
        sponsor_id
        for page in identity_scoped_pages
        for sponsor_id in _non_intake_sponsor_ids(page)
    )
    intake_unreadable = any(
        page.page_category == "intake_form"
        and any(
            re.search(r"\bUNREADABLE\b", line.text, re.I) is not None
            for line in page.lines
        )
        for page in identity_scoped_pages
    )
    return _ScopedVisibleEvidence(
        case_text=page_text_projection(
            snapshot.pages,
            include_page_indexes=frozenset(case_pages),
        ),
        identity_text=page_text_projection(
            snapshot.pages,
            include_page_indexes=identity_page_indexes,
        ),
        page_signature=page_signature,
        sponsor_attestation_proven=sponsor_attestation_proven,
        fee_status_evidence=fee_status_evidence,
        non_intake_sponsor_ids=non_intake_sponsor_ids,
        intake_unreadable=intake_unreadable,
    )


def _scoped_visible_text(
    snapshot: VisibleOcrSnapshot,
    row: PredictionRow,
) -> tuple[str, str]:
    """Compatibility view for tests and diagnostics needing only text."""

    scoped = _scoped_visible_evidence(snapshot, row)
    return scoped.case_text, scoped.identity_text


class VisibleScoreFinalizer:
    """Apply the frozen visible-layout heads to a stable base prediction.

    This layer intentionally excludes OCR retries, embedded generator
    instructions, case identifiers, filenames, and label-derived lookups.
    """

    def __init__(self, visible_text_store: VisibleOcrTextStore) -> None:
        self._visible_text_store = visible_text_store

    def __call__(self, pdf_path: Path, row: PredictionRow) -> PredictionRow:
        try:
            snapshot = self._visible_text_store.consume(pdf_path)
        except VisibleTextSnapshotError:
            return row
        scoped = _scoped_visible_evidence(snapshot, row)

        row = score_heads.apply_visible_field_repairs(row, scoped.identity_text)
        row = score_heads.apply_layout_consensus_approval(
            row,
            scoped.identity_text,
            page_signature=scoped.page_signature,
            sponsor_attestation_proven=scoped.sponsor_attestation_proven,
        )
        row = score_heads.apply_visible_slash_stamp_denial(
            row,
            pdf_path,
            scoped.case_text,
        )
        row = score_heads.apply_visible_sample_denial(row, pdf_path)
        row = score_heads.apply_visible_finding_decision(row, scoped.case_text)
        row = score_heads.apply_structured_visible_approval_safety(
            row,
            fee_status_proven=(
                str(row.fee_status).casefold() in scoped.fee_status_evidence
            ),
            sponsor_id_conflict=bool(
                scoped.non_intake_sponsor_ids - {str(row.sponsor_id)}
            ),
            intake_unreadable=scoped.intake_unreadable,
        )
        row = score_heads.apply_approval_safety_demotion(
            row,
            scoped.case_text,
            page_signature=scoped.page_signature,
            candidates=(),
        )
        row = score_heads.apply_denial_to_review_softening(row)
        row = score_heads.apply_visible_finding_decision(row, scoped.case_text)

        if row.visa_class == "TRANSIT-7" and row.adjudication == "APPROVED":
            payload = row.to_dict()
            payload["adjudication"] = "DENIED"
            payload["confidence"] = 0.98
            row = PredictionRow.from_mapping(
                payload,
                fallback_case_id=row.case_id,
            )

        row = apply_confidence_blend(row)
        return apply_platt_calibration(row)
