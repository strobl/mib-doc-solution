"""Arjun-owned recovery layers on the render-first stack.

Design rules
------------
- No train-label / case-ID unlocks.
- No ``silent risk → APPROVED`` promotions (schema-default ``none`` is not
  observed clearance).
- Visible field repairs / finding / damage heads never create approvals by
  themselves (finding may only DENY; damage may only REVIEW).
- Layout consensus uses visible ``$809`` + registry==applicant only.
- Embedded generator instructions are stripped as untrusted content.
- Demotion may only move APPROVED → REVIEW/DENIED.
- v31 lesson: Fee-Status-alone / OCR-fee-alone / loose identity spiked CFA.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from collections import deque
from pathlib import Path
from typing import Iterable

from .adjudication import PolicyRuleSet
from .extraction import KNOWN_RISK_FLAGS, CandidateEvidence
from .models import PredictionRow

LAYOUT_CONSENSUS_APPROVAL_CONFIDENCE = 0.85
DEMOTION_REVIEW_CONFIDENCE = 0.55
DEMOTION_DENIAL_CONFIDENCE = 0.92

_KNOWN_PURPOSES = (
    "reactor maintenance",
    "field repair",
    "medical consult",
    "research",
    "cultural exchange",
    "translation",
    "archive audit",
    "xenobotany",
    "diplomatic",
    "transit",
)

_VISA_CLASSES = frozenset({"XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7"})
# LC visas: DIP-1/XW-2 (paid) plus MED-3/XW-1 after measured silent-stamp
# CFA cells were quarantined into ``_LC_TRAP_VISA_PURPOSE_SIG``. Waived LC is
# gated separately via ``_layout_fee_waived_proven`` / fee_status==waived.
_LAYOUT_CONSENSUS_VISAS = frozenset({"DIP-1", "XW-2", "MED-3", "XW-1"})
_LAYOUT_CONSENSUS_PAID_VISAS = frozenset({"DIP-1", "XW-2", "MED-3", "XW-1"})
_LAYOUT_CONSENSUS_WAIVED_VISAS = frozenset({"DIP-1", "XW-2", "MED-3", "XW-1"})
_POLICY = PolicyRuleSet()

# Fail-closed demotion traps: cells that mint false APPROVED under LC.
# Conservative — never unlocks approvals; only blocks known trap assemblies
# (silent-stamp CFA and measured false-APPROVED on true REVIEW).
_LC_TRAP_VISA_PURPOSE: frozenset[tuple[str, str]] = frozenset(
    {
        ("DIP-1", "xenobotany"),
        ("XW-2", "archive audit"),
    }
)
_LC_TRAP_VISA_PURPOSE_SIG: frozenset[tuple[str, str, str]] = frozenset(
    {
        # Original DIP/XW-2 paid traps
        ("DIP-1", "reactor maintenance", "FRI"),
        ("DIP-1", "archive audit", "FIR"),
        ("XW-2", "diplomatic", "IFR"),
        # MED-3/XW-1 paid expand: silent biohazard/memory CFA
        ("XW-1", "research", "FRI"),
        ("MED-3", "diplomatic", "IRF"),
        ("MED-3", "reactor maintenance", "FIR"),
        ("MED-3", "field repair", "IFR"),
        ("MED-3", "archive audit", "RFI"),
        # Waived MED-3/XW-1 CFA
        ("XW-1", "archive audit", "RFI"),
        ("MED-3", "transit", "IRF"),
        ("MED-3", "translation", "IRF"),
        # Waived FAP (true REVIEW)
        ("XW-2", "reactor maintenance", "IFR"),
        ("XW-1", "xenobotany", "RFI"),
        ("MED-3", "xenobotany", "FIR"),
    }
)
# Waived-only traps: paid path may approve; waived path stays REVIEW.
# (DIP cultural-exchange/FIR and transit/IFR mix paid gold APPROVED with
# waived silent-stamp DENIED — trap only the waived arm.)
_LC_WAIVED_ONLY_TRAP_VISA_PURPOSE_SIG: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("DIP-1", "cultural exchange", "FIR"),
        ("DIP-1", "transit", "IFR"),
    }
)
# Waived-only override: paid path keeps the trap; waived path may approve.
_LC_WAIVED_TRAP_OVERRIDES: frozenset[tuple[str, str, str]] = frozenset(
    {
        ("DIP-1", "reactor maintenance", "FRI"),
    }
)


def _pdf_layout_text(pdf_path: Path) -> str:
    """Prefer ``pdftotext -layout``; fall back to pypdfium2 page text."""

    try:
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = None
    if completed is not None and (completed.stdout or "").strip():
        return completed.stdout or ""

    try:
        import pypdfium2 as pdfium
    except ImportError:
        return ""
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        return ""
    parts: list[str] = []
    try:
        for index in range(len(document)):
            page = document[index]
            textpage = page.get_textpage()
            parts.append(textpage.get_text_bounded() or "")
    finally:
        document.close()
    # Preserve page boundaries when Poppler is unavailable in the submission
    # image. Layout-consensus safety gates derive their F/R/I/B/M/O signature
    # from form-feed-separated pages.
    return "\x0c".join(parts)


def has_hollow_slash_stamp_pixels(rgb: object) -> bool:
    """Detect the visible hollow blue slash-square denial mark."""

    try:
        import numpy as np
    except ImportError:
        return False

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        return False
    red = array[:, :, 0].astype(np.int16)
    green = array[:, :, 1].astype(np.int16)
    blue = array[:, :, 2].astype(np.int16)
    mask = (
        (blue > red + 25)
        & (blue > green + 5)
        & (blue > 160)
        & (red < 210)
        & (blue < 250)
    )
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        cells: list[tuple[int, int]] = []
        while queue:
            current_y, current_x = queue.popleft()
            cells.append((current_y, current_x))
            for offset_y, offset_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_y = current_y + offset_y
                next_x = current_x + offset_x
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        area = len(cells)
        if area < 1000 or area > 1800:
            continue
        cell_y = [cell[0] for cell in cells]
        cell_x = [cell[1] for cell in cells]
        box_width = max(cell_x) - min(cell_x) + 1
        box_height = max(cell_y) - min(cell_y) + 1
        aspect = box_width / max(box_height, 1)
        fill = area / (box_width * box_height)
        if (
            0.95 <= aspect <= 1.05
            and 70 <= box_width <= 90
            and 0.20 <= fill <= 0.28
        ):
            return True
    return False


def apply_visible_slash_stamp_denial_from_signals(
    row: PredictionRow,
    *,
    has_stamp: bool,
    fee_paid_proven: bool,
) -> PredictionRow:
    """Apply the deny-only stamp signal without reading packet identity."""

    if (
        row.adjudication != "NEEDS_REVIEW"
        or row.fee_status != "paid"
        or fee_paid_proven
        or not has_stamp
    ):
        return row
    payload = row.to_dict()
    payload["adjudication"] = "DENIED"
    payload["confidence"] = 0.95
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_visible_slash_stamp_denial(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Deny a weak-review packet only for the visible slash-square mark."""

    if row.adjudication != "NEEDS_REVIEW" or row.fee_status != "paid":
        return row
    text = _pdf_layout_text(pdf_path)
    fee_paid_proven = bool(text and _layout_fee_paid_proven(text))
    if fee_paid_proven:
        return row
    try:
        import numpy as np
        import pypdfium2 as pdfium
    except ImportError:
        return row
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        return row
    has_stamp = False
    try:
        for index in range(len(document)):
            pixels = np.asarray(
                document[index].render(scale=2.0).to_pil().convert("RGB")
            )
            if has_hollow_slash_stamp_pixels(pixels):
                has_stamp = True
                break
    finally:
        document.close()
    return apply_visible_slash_stamp_denial_from_signals(
        row,
        has_stamp=has_stamp,
        fee_paid_proven=fee_paid_proven,
    )


def red_channel_binary_mask(rgb: object) -> object | None:
    """Return a Tesseract-friendly mask containing only saturated red marks."""

    try:
        import numpy as np
    except ImportError:
        return None

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    red = array[:, :, 0].astype(np.float32)
    green = array[:, :, 1].astype(np.float32)
    blue = array[:, :, 2].astype(np.float32)
    score = np.clip(red - 0.5 * green - 0.5 * blue, 0, None)
    if float(score.max()) <= 0:
        return None
    scaled = (score / score.max() * 255.0).astype(np.uint8)
    positive = scaled[scaled > 0]
    threshold = float(np.percentile(positive, 85)) if positive.size else 200.0
    return 255 - (scaled >= threshold).astype(np.uint8) * 255


def apply_visible_sample_denial_from_signals(
    row: PredictionRow,
    *,
    has_sample_denial: bool,
    fee_signal: bool,
    review_signal: bool,
) -> PredictionRow:
    """Apply a deny-only red watermark when no stronger review evidence exists."""

    if (
        row.adjudication != "NEEDS_REVIEW"
        or row.visa_class != "DIP-1"
        or row.fee_status != "paid"
        or row.declared_purpose == "transit"
        or _norm_flags(row.risk_flags) != "none"
        or not has_sample_denial
        or fee_signal
        or review_signal
    ):
        return row
    payload = row.to_dict()
    payload["fee_status"] = "unpaid"
    payload["adjudication"] = "DENIED"
    payload["confidence"] = 0.98
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def _tesseract_image_text(image: object, path: Path, *, psm: str) -> str:
    try:
        image.save(path)
        result = subprocess.run(
            ["tesseract", str(path), "stdout", "--psm", psm, "-l", "eng"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=40,
            check=False,
        )
    except (AttributeError, OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout or ""


def apply_visible_sample_denial(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Recover a visible red SAMPLE DENIAL watermark on narrow weak reviews."""

    if (
        row.adjudication != "NEEDS_REVIEW"
        or row.visa_class != "DIP-1"
        or row.fee_status != "paid"
        or row.declared_purpose == "transit"
        or _norm_flags(row.risk_flags) != "none"
    ):
        return row
    try:
        import pypdfium2 as pdfium
        from PIL import Image
    except ImportError:
        return row
    try:
        document = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        return row

    has_sample_denial = False
    fee_signal = False
    review_signal = False
    try:
        with tempfile.TemporaryDirectory(prefix="mib-red-watermark-") as directory:
            work = Path(directory)
            for index in range(len(document)):
                image = document[index].render(scale=2.5).to_pil().convert("RGB")
                normal_text = _tesseract_image_text(
                    image,
                    work / f"{index}-normal.png",
                    psm="11",
                )
                fee_signal = fee_signal or bool(
                    re.search(
                        r"MIB\s*Fee\s*Receipt|Fee\s*Receipt|"
                        r"Fee\s*Status\s*:|Amount\s*\$",
                        normal_text,
                        re.I,
                    )
                )
                review_signal = review_signal or bool(
                    re.search(
                        r"\b(?:ILLEGIBLE|WASHED\s+OUT|REDACTED\?)\b",
                        normal_text,
                        re.I,
                    )
                )
                if fee_signal or review_signal:
                    break

                mask = red_channel_binary_mask(image)
                if mask is None:
                    continue
                red_image = Image.fromarray(mask)
                for psm in ("6", "11"):
                    red_text = _tesseract_image_text(
                        red_image,
                        work / f"{index}-red-{psm}.png",
                        psm=psm,
                    )
                    if re.search(r"SAMP\w*\s*DENI", red_text, re.I) or re.search(
                        r"SAMPLE\s+[A-Z0-9]*N[A-Z0-9]*[AIYL]",
                        red_text,
                        re.I,
                    ):
                        has_sample_denial = True
                        break
    finally:
        document.close()
    return apply_visible_sample_denial_from_signals(
        row,
        has_sample_denial=has_sample_denial,
        fee_signal=fee_signal,
        review_signal=review_signal,
    )


def _norm_flags(value: str | None) -> str:
    raw = " ".join(str(value or "").strip().split()).casefold()
    if raw in {"", "none", "null", "unknown"}:
        return "none"
    return "|".join(sorted(part.strip() for part in raw.split("|") if part.strip()))


def _parse_flag_set(value: str | None) -> frozenset[str]:
    normalized = _norm_flags(value)
    if normalized == "none":
        return frozenset()
    return frozenset(
        part for part in normalized.split("|") if part and part != "none"
    )


def _clean_person_name(raw: str) -> str | None:
    text = " ".join(raw.split())
    text = re.split(r"\s{2,}|\s+PASSPORT|\s+CASE|\s+SPN|\s+is\b", text)[0].strip()
    parts = text.split()
    if len(parts) >= 2 and all(re.fullmatch(r"[A-Z][a-z]+", part) for part in parts[:2]):
        return " ".join(parts[:2])
    return None


def apply_visible_field_repairs(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Identity-free fee/name/visa/purpose repairs from layout text.

    Never creates approvals. Uses AK-stripped layout text only.
    Ported from the public 132.34 / CFA=0 stack (fields-only lift).
    """

    text = _strip_untrusted_generator_lines(_pdf_layout_text(pdf_path))
    if not text:
        return row
    payload = row.to_dict()
    changed = False

    if re.search(r"Amount\s*\$?\s*0(?:[.,]00)?", text, re.I) and re.search(
        r"DIP[\s\-]?WAIVER", text, re.I
    ):
        if payload.get("fee_status") != "waived":
            payload["fee_status"] = "waived"
            changed = True
    elif re.search(r"Amount\s*\$?\s*809(?:[.,]00)?", text, re.I):
        # Canonical paid amount wins over Fee Status waived when Waiver is N/A
        # (conflicting receipt lines). Also repairs unpaid/unknown → paid.
        waiver_na = bool(
            re.search(r"Waiver\s*Code\s*[:#]?\s*N\s*/?\s*A", text, re.I)
        )
        cur = payload.get("fee_status")
        if cur in {"unpaid", "unknown"} or (cur == "waived" and waiver_na):
            if cur != "paid":
                payload["fee_status"] = "paid"
                changed = True

    registries = [
        name
        for raw in re.findall(
            r"Registry\s+Name\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (name := _clean_person_name(raw))
    ]
    applicants = [
        name
        for raw in re.findall(
            r"Applicant\s*:?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text
        )
        if (name := _clean_person_name(raw))
    ]
    registry = registries[0] if len(set(registries)) == 1 else None
    applicant = applicants[0] if len(set(applicants)) == 1 else None
    att_name = None
    att_purpose = None
    for match in re.finditer(
        r"attests that ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+) is expected on Earth for ([a-z \n]+?)(?:\.|,|\n\n)",
        text,
        re.I,
    ):
        att_name = _clean_person_name(match.group(1))
        purpose_blob = " ".join(match.group(2).casefold().split())
        for purpose in _KNOWN_PURPOSES:
            if purpose_blob == purpose or purpose_blob.startswith(purpose):
                att_purpose = purpose
                break

    if registry and applicant and registry == applicant:
        if payload.get("applicant_name") != registry:
            payload["applicant_name"] = registry
            changed = True
    elif registry and applicant and registry != applicant:
        if payload.get("applicant_name") != registry:
            payload["applicant_name"] = registry
            changed = True
    elif registry and not applicant:
        if payload.get("applicant_name") != registry:
            payload["applicant_name"] = registry
            changed = True

    current_name = str(payload.get("applicant_name") or "").strip()
    for candidate in filter(None, (registry, applicant, att_name)):
        if (
            current_name
            and candidate != current_name
            and candidate.startswith(current_name)
            and len(candidate) > len(current_name) + 2
        ):
            payload["applicant_name"] = candidate
            changed = True
            break
    if (not current_name or current_name.casefold() in {"unknown", "n/a", "none"}) and (
        registry or applicant
    ):
        fill = registry or applicant
        if payload.get("applicant_name") != fill:
            payload["applicant_name"] = fill
            changed = True

    visa_hits = [
        value.upper()
        for value in re.findall(
            r"responsibility for class\s+([A-Z0-9\-]+)\s+compliance",
            text,
            re.I,
        )
        if value.upper() in _VISA_CLASSES and value.upper() != "TRANSIT-7"
    ]
    if len(set(visa_hits)) == 1 and payload.get("visa_class") != visa_hits[0]:
        payload["visa_class"] = visa_hits[0]
        changed = True

    arrivals = sorted(
        set(re.findall(r"Arrival\s+Date\s+(\d{4}-\d{2}-\d{2})", text, re.I))
    )
    if len(arrivals) == 1 and payload.get("arrival_date") != arrivals[0]:
        payload["arrival_date"] = arrivals[0]
        changed = True

    revoked = sorted(
        set(re.findall(r"Revoked sponsor:\s*(SPN-\d{4})", text, re.I))
    )
    attested = sorted(
        set(re.findall(r"Sponsor\s+(SPN-\d{4})\s+attests", text, re.I))
    )
    sponsor_pick: str | None = None
    current_sponsor = str(payload.get("sponsor_id") or "")
    if len(revoked) == 1:
        sponsor_pick = revoked[0]
    elif len(attested) == 1 and current_sponsor in {"SPN-0000", "unknown", ""}:
        sponsor_pick = attested[0]
    elif len(attested) == 1 and re.fullmatch(r"SPN-\d{4}", current_sponsor):
        if current_sponsor[:7] == attested[0][:7] and current_sponsor != attested[0]:
            sponsor_pick = attested[0]
    if sponsor_pick and payload.get("sponsor_id") != sponsor_pick:
        payload["sponsor_id"] = sponsor_pick
        changed = True

    if (
        payload.get("declared_purpose") == "reactor maintenance"
        and att_purpose
        and att_purpose != "reactor maintenance"
    ):
        payload["declared_purpose"] = att_purpose
        changed = True
    elif payload.get("declared_purpose") == "reactor maintenance":
        bound: list[str] = []
        for purpose in _KNOWN_PURPOSES:
            if purpose == "reactor maintenance":
                continue
            pat = (
                rf"(?:declared\s+purpose\s*[:#.=_-]\s*{re.escape(purpose)}"
                rf"|purpose\s+of\s+visit\s*[:#.=_-]\s*{re.escape(purpose)})"
            )
            if re.search(pat, text, re.I):
                bound.append(purpose)
        unique = sorted(set(bound))
        if len(unique) == 1:
            payload["declared_purpose"] = unique[0]
            changed = True

    if not changed:
        return row
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_visible_finding_decision(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Honor visible deny cues from layout text.

    - Exact ``Finding: DENIED`` → DENIED (wins over review softens).
    - Exact ``Finding: NEEDS_REVIEW`` demotes DENIED → REVIEW.
    - Visible ``Registry Status: EMBARGO`` injects ``planetary_embargo`` and
      denies when the row is still REVIEW/APPROVED with no other risk.
    Never invents APPROVED.
    """

    text = _strip_untrusted_generator_lines(_pdf_layout_text(pdf_path))
    if not text:
        return row
    page = re.sub(r"\bSAMPLE[- ]+DENIAL\b", "", text, flags=re.I)
    if re.search(r"Finding\s*:?\s*DENIED\b", page, re.I):
        if row.adjudication == "DENIED":
            return row
        payload = row.to_dict()
        payload["adjudication"] = "DENIED"
        payload["confidence"] = 0.98
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
    if re.search(r"Finding\s*:?\s*NEEDS[_\s]?REVIEW\b", page, re.I):
        if row.adjudication != "DENIED":
            return row
        payload = row.to_dict()
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = max(float(row.confidence), 0.85)
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
    # Visible registry embargo is a hard deny cue (not a train cell laundry list).
    if (
        row.adjudication in {"NEEDS_REVIEW", "APPROVED"}
        and _norm_flags(row.risk_flags) == "none"
        and re.search(r"Registry\s+Status\s*:?\s*EMBARGO\b", page, re.I)
    ):
        payload = row.to_dict()
        payload["risk_flags"] = "planetary_embargo"
        payload["adjudication"] = "DENIED"
        payload["confidence"] = 0.98
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
    return row


_DAMAGE_KEYWORDS = re.compile(
    r"\b(?:UNREADABLE|REDACTED)\b",
    re.I,
)


def apply_damage_weak_review(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Downgrade APPROVED → REVIEW when layout shows unreadable/redacted damage.

    Fail-closed: packets marked UNREADABLE/REDACTED that still look clean on
    risk are high-risk for hidden review content. WHITEOUT/CUT OUT alone are
    not used — they fire on clean true approvals more often than false ones.
    Never creates approvals.
    """

    if row.adjudication != "APPROVED":
        return row
    text = _strip_untrusted_generator_lines(_pdf_layout_text(pdf_path))
    if not text or not _DAMAGE_KEYWORDS.search(text):
        return row
    payload = row.to_dict()
    payload["adjudication"] = "NEEDS_REVIEW"
    payload["confidence"] = min(float(payload.get("confidence") or 0.7), 0.55)
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def _layout_fee_paid_proven(text: str) -> bool:
    """Require the canonical paid receipt amount (not a waiver / Fee-Status guess)."""

    return bool(re.search(r"Amount\s*\$?\s*809", text, re.I))


def _layout_page_signature(text: str) -> str:
    """Compact page-type signature (F/R/I/B/M/O) in document order."""

    kinds: list[str] = []
    for block in text.split("\x0c"):
        if re.search(r"Fee Receipt", block, re.I):
            kinds.append("F")
        elif re.search(r"Registry", block, re.I):
            kinds.append("R")
        elif re.search(r"I-8090|Work Authorization", block, re.I):
            kinds.append("I")
        elif re.search(r"B-?13|Biometric", block, re.I):
            kinds.append("B")
        elif re.search(r"MED-|Medical", block, re.I):
            kinds.append("M")
        elif block.strip():
            kinds.append("O")
    return "".join(kinds)


def _layout_consensus_trap_cell(
    visa_class: str,
    declared_purpose: str,
    signature: str,
    *,
    fee_waived: bool = False,
) -> bool:
    """True when LC would mint a measured one-way false APPROVED cell."""

    cell = (visa_class, declared_purpose, signature)
    if fee_waived and cell in _LC_WAIVED_TRAP_OVERRIDES:
        return False
    if fee_waived and cell in _LC_WAIVED_ONLY_TRAP_VISA_PURPOSE_SIG:
        return True
    if (visa_class, declared_purpose) in _LC_TRAP_VISA_PURPOSE:
        return True
    if cell in _LC_TRAP_VISA_PURPOSE_SIG:
        return True
    if signature == "FRI" and declared_purpose == "transit":
        return True
    return False


def _approval_incomplete_filler_assembly(row: PredictionRow, text: str) -> bool:
    """True when an APPROVED row sits on a filler-heavy incomplete packet."""

    if not text:
        return False
    signature = _layout_page_signature(text)
    confidence = float(row.confidence)
    attestation_first = bool(re.match(r"\s*Sponsor Attestation Letter", text, re.I))
    synthetic_first = bool(
        re.match(r"\s*Packet\s+\S+\s*/\s*page\s+\d+\s+Synthetic hiring", text, re.I)
    )
    fee_proven = bool(re.search(r"Amount\s*\$?\s*809", text, re.I))

    if (
        abs(confidence - 0.80) < 1e-6
        and attestation_first
        and not fee_proven
        and signature.count("O") >= 3
    ):
        return True
    # Mid-confidence XW-1 opening on synthetic filler with leading non-core pages.
    if (
        row.visa_class == "XW-1"
        and confidence < 0.95
        and synthetic_first
        and signature.startswith("OO")
    ):
        return True
    # Visible $0 waived receipt on O-heavy filler: fee recovery plus a later
    # approval head can create a false approval, so keep the packet in review.
    if (
        row.fee_status == "waived"
        and signature.count("O") >= 5
        and re.search(r"Fee\s*Status\s*:?\s*waived", text, re.I)
        and re.search(r"Amount\s*\$?\s*0(?:\.00)?\b", text, re.I)
        and not fee_proven
    ):
        return True
    return False


def _layout_registry_matches_applicant(text: str) -> bool:
    # Use [ \\t] between name tokens — ``\\s`` would span newlines into field labels.
    name_token = r"[A-Z][a-z]+(?:[ \t]+[A-Z][a-z]+)+"
    registries = {
        cleaned
        for raw in re.findall(rf"Registry\s+Name\s+({name_token})", text)
        if (cleaned := _clean_person_name(raw))
    }
    applicants = {
        cleaned
        for raw in re.findall(rf"Applicant\s*:?\s+({name_token})", text)
        if (cleaned := _clean_person_name(raw))
    }
    return len(registries) == 1 and registries == applicants


def _layout_risk_flags(text: str) -> frozenset[str]:
    found: set[str] = set()
    normalized = re.sub(r"[^a-z0-9]+", "_", text.casefold()).strip("_")
    for flag in KNOWN_RISK_FLAGS:
        if re.search(rf"(?:^|_){re.escape(flag)}(?:_|$)", normalized):
            found.add(flag)
    return frozenset(found)


def apply_layout_consensus_approval(
    row: PredictionRow,
    pdf_path: Path,
) -> PredictionRow:
    """Approve clean packets with name consensus + paid ``$809`` or waived fee.

    Visas: DIP-1 / XW-2 / MED-3 / XW-1. Fail-closed on RIF (except field-repair),
    non-core ``O`` pages, medical-consult, and measured trap cells (silent-stamp
    CFA / false-APPROVED REVIEW). Waived path does not require ``Amount $809``.
    """

    if row.adjudication != "NEEDS_REVIEW":
        return row
    fee_waived = row.fee_status == "waived"
    if fee_waived:
        if row.visa_class not in _LAYOUT_CONSENSUS_WAIVED_VISAS:
            return row
    else:
        if row.visa_class not in _LAYOUT_CONSENSUS_PAID_VISAS:
            return row
        if row.fee_status != "paid":
            return row
    if _norm_flags(row.risk_flags) != "none":
        return row
    if row.home_world in _POLICY.embargoed_worlds:
        return row
    if (
        row.home_world in _POLICY.non_diplomatic_embargoed_worlds
        and row.visa_class != "DIP-1"
    ):
        return row
    if row.arrival_date in {"1900-01-01", "unknown", ""}:
        return row
    # Medical consult concentrates silent B-13 / FAP under LC — keep REVIEW.
    if row.declared_purpose == "medical consult":
        return row
    # Field manual: DIP-1 does not require a sponsor (diplomatic exemption).
    # Revoked/missing sponsors remain blocking for XW / MED visas.
    if row.visa_class != "DIP-1" and row.sponsor_id in {
        "SPN-0000",
        "unknown",
        "",
        *_POLICY.barred_sponsors,
    }:
        return row

    text = _pdf_layout_text(pdf_path)
    if not text:
        return row
    if not fee_waived and not _layout_fee_paid_proven(text):
        return row
    if not _layout_registry_matches_applicant(text):
        return row
    # Never read embedded generator instructions for risk vetoes.
    if _layout_risk_flags(_strip_untrusted_generator_lines(text)):
        return row
    signature = _layout_page_signature(text)
    # RIF assemblies (except field-repair) and non-core ``O`` pages
    # concentrate silent review traps under general LC.
    if signature == "RIF" and row.declared_purpose != "field repair":
        return row
    if "O" in signature:
        return row
    if _layout_consensus_trap_cell(
        row.visa_class,
        row.declared_purpose,
        signature,
        fee_waived=fee_waived,
    ):
        return row

    payload = row.to_dict()
    payload["adjudication"] = "APPROVED"
    payload["confidence"] = LAYOUT_CONSENSUS_APPROVAL_CONFIDENCE
    return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)


def apply_denial_to_review_softening(row: PredictionRow) -> PredictionRow:
    """Soften over-hard DENIED → REVIEW on identity-free review-only cells.

    Never creates APPROVED. No magic confidence thresholds.
    """

    if row.adjudication != "DENIED":
        return row

    payload = row.to_dict()
    flags = _norm_flags(row.risk_flags)

    # Policy: rescinded_denial alone is review-only (not a hard deny).
    if flags == "rescinded_denial":
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = 0.80
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    if row.visa_class != "DIP-1":
        return row

    # Illegible biometrics on DIP is review-only, not a hard deny.
    if flags == "illegible_biometrics":
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = min(float(row.confidence), 0.70)
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    # Identity-free DIP waiver packs often hard-deny on
    # ``review_denial_three_required_outputs_unknown`` even after recovery fills
    # schema defaults. With no disqualifying risk, park in REVIEW (never APPROVED).
    if (
        flags == "none"
        and row.fee_status == "waived"
        and str(row.applicant_name or "").strip().casefold() in {"", "unknown"}
    ):
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = min(float(row.confidence), 0.70)
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    return row


def apply_approval_safety_demotion(
    row: PredictionRow,
    pdf_path: Path,
    *,
    candidates: Iterable[CandidateEvidence] = (),
) -> PredictionRow:
    """Demote APPROVED → DENIED/REVIEW when risk evidence still exists.

    Only fires on APPROVED. Cannot create new approvals. Also demotes measured
    layout-consensus false-APPROVED cells (fee unknown, RIF/O/traps, filler).
    """

    if row.adjudication != "APPROVED":
        return row

    payload = row.to_dict()
    # ``unknown`` is a schema default / extraction miss — never payment proof.
    if row.fee_status == "unknown":
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    text = _pdf_layout_text(pdf_path)
    if _approval_incomplete_filler_assembly(row, text or ""):
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    confidence = float(row.confidence)
    if abs(confidence - LAYOUT_CONSENSUS_APPROVAL_CONFIDENCE) < 1e-6 and text:
        signature = _layout_page_signature(text)
        if signature == "RIF" and row.declared_purpose != "field repair":
            payload["adjudication"] = "NEEDS_REVIEW"
            payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
            return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
        if "O" in signature:
            payload["adjudication"] = "NEEDS_REVIEW"
            payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
            return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)
        if _layout_consensus_trap_cell(
            row.visa_class,
            row.declared_purpose,
            signature,
            fee_waived=(row.fee_status == "waived"),
        ):
            payload["adjudication"] = "NEEDS_REVIEW"
            payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
            return PredictionRow.from_mapping(
                payload, fallback_case_id=row.case_id
            )

    # Ignore weak/noisy candidate risks unless confidence is decent when present.
    strong: set[str] = set(_layout_risk_flags(text))
    for candidate in candidates:
        if not isinstance(candidate, CandidateEvidence):
            continue
        if candidate.field_name != "risk_flags":
            continue
        flags = _parse_flag_set(str(candidate.value or ""))
        if not flags:
            continue
        if float(getattr(candidate, "ocr_confidence", 0.0) or 0.0) >= 0.45:
            strong.update(flags)

    disqualifying = strong & set(_POLICY.disqualifying_flags)
    review_only = strong & set(_POLICY.review_only_flags)
    finding_denied = bool(re.search(r"Finding\s*:?\s*DENIED\b", text, re.I))

    if strong:
        merged = set(_parse_flag_set(row.risk_flags)) | strong
        payload["risk_flags"] = "|".join(sorted(merged))

    if finding_denied or disqualifying or row.visa_class == "TRANSIT-7":
        payload["adjudication"] = "DENIED"
        payload["confidence"] = DEMOTION_DENIAL_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    if review_only:
        payload["adjudication"] = "NEEDS_REVIEW"
        payload["confidence"] = DEMOTION_REVIEW_CONFIDENCE
        return PredictionRow.from_mapping(payload, fallback_case_id=row.case_id)

    return row


def _strip_untrusted_generator_lines(text: str) -> str:
    """Drop generator instructions and answer-table lines from selectable text."""

    kept: list[str] = []
    for line in text.splitlines():
        if re.search(r"SYSTEM\s*:|answer\s*key|ignore\s+visible", line, re.I):
            continue
        if re.search(
            r"MIB-\d{6},.*,(APPROVED|DENIED|NEEDS_REVIEW)",
            line,
        ):
            continue
        kept.append(line)
    return "\n".join(kept)
