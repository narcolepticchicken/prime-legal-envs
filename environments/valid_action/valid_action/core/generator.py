"""Seeded world and task generation (spec section 10).

The generator produces solvable worlds from structured ground truth and
runs the oracle to verify the world is solvable. Rejection sampling is
bounded to avoid runaway generation.

Templates are sourced from `fixtures.GOLDEN_FIXTURES` for the MVP.
A separate compositional generator will be added after the fixtures cover
all difficulty bands.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from copy import deepcopy
from datetime import date
from decimal import Decimal
from typing import Any

from .fixtures import GOLDEN_FIXTURES
from .models import (
    Difficulty,
    EquityGrantRequest,
    MaterialContractRequest,
    RelatedPartyTransactionRequest,
    SecurityIssuanceRequest,
    SubsidiaryFinancingRequest,
    TokenTreasuryTransactionRequest,
    ValidActionWorld,
)
from .oracle import build_oracle_action, replay_oracle_as_attempts, solve_oracle
from .render import render_world
from .requirements import RequirementGraph
from .serialization import compute_fingerprint
from .validator import validate_final_execution


class GenerationError(RuntimeError):
    pass


def _derive_seed(base_seed: int, index: int, salt: str = "") -> int:
    """Stable sub-seed for an indexed task within a split."""
    payload = f"{base_seed}:{index}:{salt}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def _stable_id(prefix: str, base: int) -> str:
    digest = hashlib.sha256(f"{prefix}:{base}".encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:8]}"


def _retag_ids(world: ValidActionWorld, salt: str) -> None:
    """Re-stamp IDs in the world to vary per seed+split without changing semantics.

    Maps each existing id through a salted hash so the same fixture template
    produces disjoint entity/role/body/record ids per seed.
    """
    id_map: dict[str, str] = {}

    def remap(prefix: str, old: str) -> str:
        if old not in id_map:
            digest = hashlib.sha256(f"{prefix}:{salt}:{old}".encode("utf-8")).hexdigest()[:8]
            id_map[old] = f"{prefix}_{digest}"
        return id_map[old]

    for entity in world.entities:
        new_id = remap("ent", entity.entity_id)
        entity.entity_id = new_id
    for person in world.people:
        new_id = remap("prs", person.person_id)
        person.person_id = new_id
    for role in world.roles:
        new_id = remap("role", role.role_id)
        old_role_id = role.role_id
        role.role_id = new_id
        if role.body_id is not None:
            role.body_id = id_map.get(role.body_id, remap("body", role.body_id))
        role.entity_id = id_map.get(role.entity_id, remap("ent", role.entity_id))
        role.person_id = id_map.get(role.person_id, remap("prs", role.person_id))
        # rewrite references in id_map
        id_map[old_role_id] = new_id
    for body in world.bodies:
        new_id = remap("body", body.body_id)
        old_body_id = body.body_id
        body.body_id = new_id
        body.entity_id = id_map.get(body.entity_id, remap("ent", body.entity_id))
        body.member_role_ids = [id_map.get(r, remap("role", r)) for r in body.member_role_ids]
        id_map[old_body_id] = new_id
    for sec in world.security_classes:
        sec.class_id = remap("cls", sec.class_id)
        sec.entity_id = id_map.get(sec.entity_id, remap("ent", sec.entity_id))
    for holder in world.holder_positions:
        holder.holder_id = remap("holder", holder.holder_id)
        holder.class_id = id_map.get(holder.class_id, remap("cls", holder.class_id))
    for plan in world.plan_capacities:
        plan.plan_id = remap("plan", plan.plan_id)
        plan.entity_id = id_map.get(plan.entity_id, remap("ent", plan.entity_id))
    for record in world.records:
        record.record_id = remap("rec", record.record_id)
        if record.entity_id is not None:
            record.entity_id = id_map.get(record.entity_id, remap("ent", record.entity_id))
        record.supersedes_record_ids = [
            id_map.get(s, remap("rec", s)) for s in record.supersedes_record_ids
        ]

    req = world.action_request
    request_field_map = {
        MaterialContractRequest: ["entity_id", "counterparty_id"],
        EquityGrantRequest: ["entity_id", "recipient_person_id"],
        SecurityIssuanceRequest: ["entity_id", "purchaser_id", "class_id"],
        RelatedPartyTransactionRequest: ["entity_id", "counterparty_id", "related_person_id"],
        TokenTreasuryTransactionRequest: ["entity_id", "counterparty_id", "related_person_id"],
        SubsidiaryFinancingRequest: ["entity_id", "subsidiary_id", "lender_id"],
    }
    for klass, fields in request_field_map.items():
        if isinstance(req, klass):
            for field in fields:
                value = getattr(req, field)
                if value is None:
                    continue
                prefix_map = {
                    "entity_id": "ent",
                    "counterparty_id": "cp",
                    "recipient_person_id": "prs",
                    "purchaser_id": "cp",
                    "class_id": "cls",
                    "related_person_id": "prs",
                    "subsidiary_id": "ent",
                    "lender_id": "cp",
                }
                prefix = prefix_map[field]
                setattr(req, field, id_map.get(value, remap(prefix, value)))
            break

    graph = world.requirements
    _retag_requirement_graph(graph, id_map)

    if isinstance(req, MaterialContractRequest):
        req.entity_id = id_map.get(req.entity_id, req.entity_id)
    world.oracle_solution.signatory_person_ids = [
        id_map.get(s, remap("prs", s)) for s in world.oracle_solution.signatory_person_ids
    ]


def _retag_requirement_graph(graph: RequirementGraph, id_map: dict[str, str]) -> None:
    """Rewrite authorizer_id / consent_holder_id / body_id / entity_id refs in the graph."""
    from .requirements import (
        AuthorizationRequirement,
        CapacityRequirement,
        ConsentRequirement,
        SignatoryRequirement,
        TermsRequirement,
        AllOf,
        AnyOf,
    )

    capacity_prefix = {
        "authorized_shares": "cls",
        "plan_pool": "plan",
        "treasury_tokens": "treasury",
        "borrowing_headroom": "debt",
    }

    def walk(node) -> None:
        if isinstance(node, (AllOf, AnyOf)):
            for child in node.children:
                walk(child)
        if isinstance(node, AuthorizationRequirement):
            node.authorizer_id = id_map.get(node.authorizer_id, node.authorizer_id)
            node.eligible_voter_role_ids = [
                id_map.get(r, r) for r in node.eligible_voter_role_ids
            ]
            node.eligible_voter_person_ids = [
                id_map.get(p, p) for p in node.eligible_voter_person_ids
            ]
        elif isinstance(node, ConsentRequirement):
            node.consent_holder_id = id_map.get(node.consent_holder_id, node.consent_holder_id)
            if node.participant_id is not None:
                node.participant_id = id_map.get(node.participant_id, node.participant_id)
        elif isinstance(node, SignatoryRequirement):
            node.entity_id = id_map.get(node.entity_id, node.entity_id) if node.entity_id else None
            node.eligible_role_ids = [id_map.get(r, r) for r in node.eligible_role_ids]
            node.eligible_person_ids = [id_map.get(p, p) for p in node.eligible_person_ids]
        elif isinstance(node, CapacityRequirement):
            kind_value = node.capacity_kind.value if hasattr(node.capacity_kind, "value") else str(node.capacity_kind)
            prefix = capacity_prefix.get(kind_value, "cap")
            node.target_id = id_map.get(node.target_id, node.target_id)
            if node.target_id not in id_map.values():
                digest = hashlib.sha256(f"cap:{node.target_id}".encode("utf-8")).hexdigest()[:8]
                node.target_id = f"{prefix}_{digest}"
        elif isinstance(node, TermsRequirement):
            updated = {}
            for key, value in node.expected_payload.items():
                if isinstance(value, str) and value in id_map:
                    updated[key] = id_map[value]
                elif isinstance(value, dict) and "value" in value:
                    inner = value["value"]
                    if isinstance(inner, str) and inner in id_map:
                        new_val = dict(value)
                        new_val["value"] = id_map[inner]
                        updated[key] = new_val
                    else:
                        updated[key] = value
                else:
                    updated[key] = value
            node.expected_payload = updated

    walk(graph.root)


def generate_world(
    *,
    split: str,
    difficulty: Difficulty,
    seed: int,
    index: int,
    max_attempts: int = 5,
) -> ValidActionWorld:
    """Generate one solvable world. Raises GenerationError if all attempts fail."""
    templates = list(GOLDEN_FIXTURES.items())
    if not templates:
        raise GenerationError("no templates available")
    rng = random.Random(_derive_seed(seed, index, split))
    attempts = 0
    last_error: Exception | None = None
    while attempts < max_attempts:
        attempts += 1
        name, builder = rng.choice(templates)
        try:
            world = builder()
            world.seed = seed
            world.difficulty = difficulty
            salt = f"{split}:{seed}:{index}:{attempts}"
            _retag_ids(world, salt)
            _distribute_extra_records(rng, world, salt)
            render_world(world, seed=seed + attempts * 31)
            oracle = solve_oracle(world)
            if not oracle.feasible:
                raise GenerationError(f"{name} not solvable after render")
            world.oracle_solution = oracle
            _ = compute_fingerprint(world)
            _verify_oracle_replay(world)
            return world
        except Exception as exc:
            last_error = exc
            continue
    raise GenerationError(f"failed after {attempts} attempts: {last_error}")


def _distribute_extra_records(
    rng: random.Random, world: ValidActionWorld, salt: str = ""
) -> None:
    """Add small distractor records to keep composition honest."""
    if world.difficulty == Difficulty.EASY:
        return
    from .models import LegalRecord, RecordSection, RecordType

    n_extra = {"medium": 2, "hard": 4}[world.difficulty.value]
    entity_id = world.action_request.entity_id
    for i in range(n_extra):
        suffix = hashlib.sha256(f"{salt}:{i}".encode("utf-8")).hexdigest()[:8]
        record = LegalRecord(
            record_id=f"rec_dist_{suffix}",
            entity_id=entity_id,
            record_type=RecordType.BYLAWS,
            title=f"Distractor Note {i+1}",
            effective_date=date(2020, 1, 1),
            status="active",
            sections=[
                RecordSection(
                    section_id=f"s_dist_{i}",
                    heading=f"Miscellaneous provision {i+1}",
                    text="This provision is for context and does not authorize any specific action.",
                    source_rule_ids=[],
                )
            ],
        )
        world.records.append(record)


def _verify_oracle_replay(world: ValidActionWorld) -> None:
    """Round-trip the oracle through the normal validator to catch shared bugs."""
    action_id = _stable_id("act", world.seed)
    created = build_oracle_action(world, action_id, timestamp_step=1)
    attempts = replay_oracle_as_attempts(world, world.oracle_solution, action_id)
    from .models import ExecutionAttempt

    signatory_id = world.oracle_solution.signatory_person_ids[0] if world.oracle_solution.signatory_person_ids else "prs_unknown"
    execution = ExecutionAttempt(
        execution_id=_stable_id("exec", world.seed),
        action_id=action_id,
        signatory_person_id=signatory_id,
        timestamp_step=len(attempts) + 2,
        valid=True,
        defect_codes=[],
    )
    result = validate_final_execution(
        world=world,
        created_action=created,
        authorization_attempts=attempts,
        execution=execution,
    )
    if not result.valid:
        raise GenerationError(
            "oracle replay did not validate: "
            + ", ".join(d.value for d in result.defect_codes)
        )


def generate_dataset(
    *,
    split: str,
    difficulty: Difficulty,
    seed: int,
    num_examples: int,
) -> list[ValidActionWorld]:
    """Generate a deterministic list of worlds for the given split."""
    worlds: list[ValidActionWorld] = []
    for index in range(num_examples):
        worlds.append(
            generate_world(
                split=split,
                difficulty=difficulty,
                seed=seed,
                index=index,
            )
        )
    return worlds


def split_seeds(base_seed: int) -> dict[str, int]:
    """Recommended train/eval split seeds (spec section 10.3)."""
    return {
        "train": base_seed,
        "eval": base_seed + 1_000_003,
    }


def split_stride() -> int:
    return 1009


def reproducible_rng(seed: int) -> random.Random:
    return random.Random(seed)


__all__ = [
    "GenerationError",
    "generate_dataset",
    "generate_world",
    "reproducible_rng",
    "split_seeds",
    "split_stride",
    "_stable_id",
    "_derive_seed",
]
