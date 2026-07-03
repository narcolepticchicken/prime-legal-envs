"""Canonical JSON, fingerprinting, and task-row conversion for valid-action worlds.

The world is the source of truth. Records and prompts are deterministic projections.
The fingerprint is a stable structural hash used to keep train and eval disjoint.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from .models import (
    Difficulty,
    RecordSection,
    RecordType,
    ValidActionWorld,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(value: Any) -> str:
    """Return a deterministic JSON string. Lists are order-stable by construction
    because generators and fixtures build them in fixed order."""
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def world_to_canonical(world: ValidActionWorld) -> dict[str, Any]:
    """Serialize a world to a plain dict using pydantic's mode='json' so Decimals
    become strings and dates become ISO strings. Order-stable via canonical_json."""
    payload = world.model_dump(mode="json")
    return json.loads(canonical_json(payload))


def visible_records(world: ValidActionWorld) -> list[dict[str, Any]]:
    """Serialize records for the model: strips source_rule_ids from sections."""
    out: list[dict[str, Any]] = []
    for record in world.records:
        rec = record.model_dump(mode="json", exclude={"sections"})
        rec["sections"] = [
            RecordSection.model_validate(s).model_dump(
                mode="json", exclude={"source_rule_ids"}
            )
            for s in record.sections
        ]
        out.append(rec)
    return out


def compute_fingerprint(world: ValidActionWorld) -> str:
    """Hash the canonical structural shape of the world.

    Excludes names, dates, and prose wording so train/eval worlds can share
    structural patterns without colliding. Includes seed and instance index
    via entity/record id ordering so two instances of the same template
    produce distinct fingerprints.
    """
    shape: dict[str, Any] = {
        "schema_version": world.schema_version,
        "template_id": world.world_template_id,
        "seed": world.seed,
        "difficulty": world.difficulty.value,
        "action_type": world.action_request.action_type.value,
        "entity_ids": sorted({e.entity_id for e in world.entities}),
        "body_types": sorted({b.body_type.value for b in world.bodies}),
        "record_types": sorted({r.record_type.value for r in world.records}),
        "record_statuses": sorted({r.status for r in world.records}),
        "n_records": len(world.records),
        "n_roles": len(world.roles),
        "n_holders": len(world.holder_positions),
        "requirements": _requirement_shape(world),
    }
    payload = canonical_json(shape)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _requirement_shape(world: ValidActionWorld) -> dict[str, Any]:
    """Project the requirement graph to a fingerprint-safe structural summary."""
    graph = world.requirements
    return _node_shape(graph)


def _node_shape(node: Any) -> dict[str, Any]:
    if node is None:
        return {"type": "none"}
    if hasattr(node, "model_dump"):
        data = node.model_dump(mode="json")
    else:
        data = dict(node)
    if "children" in data:
        data["children"] = [_node_shape(child) for child in data["children"]]
    if "source_record_id" in data:
        data.pop("source_record_id", None)
        data.pop("source_section_id", None)
    return data


def world_to_task_payload(world: ValidActionWorld) -> dict[str, Any]:
    """Convert a world into a serializable Task row mapping.

    The payload is what the agent sees: prompt, info, example_id, max_turns.
    Hidden fields (oracle path, source_rule_ids) stay on the world but never
    appear here.
    """
    return {
        "info": {
            "world": world_to_canonical(world),
            "records": visible_records(world),
            "registers": _register_payload(world),
            "fingerprint": compute_fingerprint(world),
        },
        "max_turns": _max_turns_for(world.difficulty),
    }


def _register_payload(world: ValidActionWorld) -> dict[str, Any]:
    """Project the structured registers for the agent. Each is a JSON-friendly list."""
    return {
        "directors": [r.model_dump(mode="json") for r in world.roles if r.role_type == "director"],
        "officers": [r.model_dump(mode="json") for r in world.roles if r.role_type == "officer"],
        "committee_members": [
            r.model_dump(mode="json") for r in world.roles if r.role_type == "committee_member"
        ],
        "bodies": [b.model_dump(mode="json") for b in world.bodies],
        "security_classes": [s.model_dump(mode="json") for s in world.security_classes],
        "holder_positions": [h.model_dump(mode="json") for h in world.holder_positions],
        "plan_capacities": [p.model_dump(mode="json") for p in world.plan_capacities],
        "entities": [e.model_dump(mode="json") for e in world.entities],
        "people": [p.model_dump(mode="json") for p in world.people],
    }


def _max_turns_for(difficulty: Difficulty) -> int:
    return {"easy": 10, "medium": 14, "hard": 18}[difficulty.value]


def assert_json_serializable(value: Any) -> None:
    """Raise if value contains types Pydantic / verifiers can't serialize."""
    json.dumps(value, default=_json_default)


def deep_round_trip(model: BaseModel) -> BaseModel:
    """Re-parse a model after dumping to JSON to confirm canonical stability."""
    payload = model.model_dump(mode="json")
    return type(model).model_validate(payload)


def has_record_type(world: ValidActionWorld, record_type: RecordType) -> bool:
    return any(r.record_type == record_type for r in world.records)


def find_section(world: ValidActionWorld, record_id: str, section_id: str) -> RecordSection | None:
    for record in world.records:
        if record.record_id != record_id:
            continue
        for section in record.sections:
            if section.section_id == section_id:
                return section
    return None


__all__ = [
    "canonical_json",
    "world_to_canonical",
    "visible_records",
    "compute_fingerprint",
    "world_to_task_payload",
    "assert_json_serializable",
    "deep_round_trip",
    "has_record_type",
    "find_section",
]
