"""Deterministic legal-record renderer (spec section 11).

Rules are the source of truth; records are deterministic prose projections.
The renderer:
  - takes a world with structured ground truth (records with section stubs);
  - expands each section into realistic but synthetic legal prose;
  - keeps source_rule_ids attached to the structured world but excluded
    from the model-visible record text;
  - never includes hidden rule IDs in rendered prose.
"""

from __future__ import annotations

import hashlib
import random
from datetime import date
from decimal import Decimal
from typing import Any

from .models import (
    ApprovalMethod,
    LegalRecord,
    RecordSection,
    RecordType,
    ValidActionWorld,
)


WORDING_VARIANTS = {
    "committee_authority": [
        "The {body} may approve equity grants to employees of up to {units} units per recipient.",
        "Authority is delegated to the {body} for grants to employees not exceeding {units} units per award.",
        "{body} may act upon employee equity grants within a per-recipient cap of {units} units.",
    ],
    "committee_exclusion": [
        "This delegation excludes grants to directors and Section 16 officers.",
        "Awards to directors are reserved to the full board.",
        "Director and officer awards are not within the {body} scope.",
    ],
    "board_quorum_majority": [
        "A majority of seated directors constitutes a quorum.",
        "Quorum requires a majority of the directors then in office.",
        "Board action requires a quorum of seated directors.",
    ],
    "board_vote_majority": [
        "Action by meeting requires a majority of directors present.",
        "Resolutions adopted at a meeting pass by majority of those present.",
    ],
    "written_consent_unanimous": [
        "Action may be taken by unanimous written consent in lieu of a meeting.",
        "Any action that may be taken at a meeting may be taken by unanimous written consent.",
    ],
    "delegated_signature_cap": [
        "Officers may execute contracts not exceeding {amount} within their delegation.",
        "Contracts above the officer delegation require board authorization.",
    ],
    "plan_reserve": [
        "The {plan} reserves {reserved} units for awards; {available} remain available.",
        "Reserved capacity under the {plan} is {reserved} units; available is {available}.",
    ],
    "class_consent": [
        "Issuance of preferred shares requires the consent of the holders of a majority of the {class}.",
        "The {class} holders must consent to issuances that dilute their position.",
    ],
    "debt_covenant": [
        "Material contracts with payment terms exceeding {threshold} require lender consent.",
        "The credit agreement restricts commitments above {threshold} absent lender consent.",
    ],
    "conflict_recusal": [
        "A director with a material interest must disclose the interest and recuse from the vote.",
        "Directors with a conflicting interest shall be recused and excluded from the disinterested quorum.",
    ],
    "subsidiary_authority": [
        "The subsidiary board must approve any borrowing on behalf of the subsidiary.",
        "Borrowings require subsidiary board approval before parent action.",
    ],
    "parent_guarantee": [
        "A parent guarantee of subsidiary debt requires parent board approval.",
        "Parent guarantees are reserved matters requiring board action.",
    ],
    "treasury_floor": [
        "Treasury token sales must not be priced below {floor} per token.",
        "Treasury transactions below the floor price are prohibited.",
    ],
    "token_lockup": [
        "Tokens sold from treasury carry a lockup of {lockup_months} months.",
        "Counterparties must accept a {lockup_months}-month lockup on treasury token sales.",
    ],
    "amendment_marker": [
        "This record supersedes {supersedes}.",
        "Effective on and after {effective_date}, this record supersedes {supersedes}.",
    ],
}


def render_world(world: ValidActionWorld, seed: int) -> list[LegalRecord]:
    """Render deterministic prose for every record and section in the world.

    Mutates each record's section text in place (records are shared with the
    world). source_rule_ids on sections is preserved for tests and trace
    rendering but excluded from the model-visible record.
    """
    rng = random.Random(seed)
    for record in world.records:
        for section in record.sections:
            section.text = _render_section(rng, record, section, world)
        record.search_text = _search_text(record)
    return world.records


def _render_section(
    rng: random.Random,
    record: LegalRecord,
    section: RecordSection,
    world: ValidActionWorld,
) -> str:
    template_key = _template_key(section, record)
    variants = WORDING_VARIANTS.get(template_key)
    if variants is None:
        return section.text or _generic(record, section)
    template = rng.choice(variants)
    return _fill_template(template, record, section, world, rng)


def _template_key(section: RecordSection, record: LegalRecord) -> str:
    heading = section.heading.lower()
    if "delegation" in heading and "committee" in heading:
        if "exclude" in heading or "director" in heading:
            return "committee_exclusion"
        return "committee_authority"
    if "quorum" in heading:
        return "board_quorum_majority"
    if "written consent" in heading or "unanimous" in heading:
        return "written_consent_unanimous"
    if "signature" in heading or "delegation" in heading and "contract" in record.title.lower():
        return "delegated_signature_cap"
    if "reserve" in heading or "capacity" in heading:
        return "plan_reserve"
    if "class" in heading and "consent" in heading:
        return "class_consent"
    if "covenant" in heading or "lender" in heading and "consent" in heading:
        return "debt_covenant"
    if "recusal" in heading or "conflict" in heading:
        return "conflict_recusal"
    if "subsidiary" in heading and "borrow" in heading:
        return "subsidiary_authority"
    if "guarantee" in heading:
        return "parent_guarantee"
    if "floor" in heading:
        return "treasury_floor"
    if "lockup" in heading:
        return "token_lockup"
    if "supersede" in heading:
        return "amendment_marker"
    return ""


def _fill_template(
    template: str,
    record: LegalRecord,
    section: RecordSection,
    world: ValidActionWorld,
    rng: random.Random,
) -> str:
    payload = section.model_dump(mode="json", exclude={"source_rule_ids"})
    body = world.body_by_id(record.entity_id or "") if record.entity_id else None
    replacements: dict[str, str] = {
        "body": body.display_name if body else "the committee",
        "units": str(payload.get("units", 0)),
        "amount": str(payload.get("amount", 0)),
        "plan": payload.get("plan", "the equity plan"),
        "reserved": str(payload.get("reserved", 0)),
        "available": str(payload.get("available", 0)),
        "class": payload.get("class", "preferred stock"),
        "threshold": str(payload.get("threshold", 0)),
        "lockup_months": str(payload.get("lockup_months", 0)),
        "floor": str(payload.get("floor", 0)),
        "supersedes": ", ".join(record.supersedes_record_ids) or "the prior record",
        "effective_date": record.effective_date.isoformat(),
    }
    text = template
    for key, value in replacements.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def _generic(record: LegalRecord, section: RecordSection) -> str:
    return f"[{record.record_type.value}] {section.heading}: {section.text or ''}".strip()


def _search_text(record: LegalRecord) -> str:
    parts = [record.title, record.record_type.value]
    for section in record.sections:
        parts.append(section.heading)
        parts.append(section.text)
    return " \n ".join(parts)


def visible_record(record: LegalRecord) -> dict[str, Any]:
    """Project a record for the model: strip source_rule_ids from sections."""
    rec = record.model_dump(mode="json")
    rec["sections"] = [
        {k: v for k, v in section.items() if k != "source_rule_ids"}
        for section in rec["sections"]
    ]
    return rec


def snippet(text: str, query: str, max_length: int = 160) -> str:
    """Return a short snippet of text centered on the first query match."""
    if not text or not query:
        return text[:max_length]
    lower = text.lower()
    idx = lower.find(query.lower())
    if idx < 0:
        return text[:max_length]
    start = max(0, idx - max_length // 2)
    end = min(len(text), start + max_length)
    snippet_text = text[start:end]
    if start > 0:
        snippet_text = "..." + snippet_text
    if end < len(text):
        snippet_text = snippet_text + "..."
    return snippet_text


def fingerprint_text(text: str) -> str:
    """Deterministic short hash for log deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "render_world",
    "visible_record",
    "snippet",
    "fingerprint_text",
]
