"""Minimum-valid-process oracle (spec section 14).

DP over the requirement graph:
  - Atomic terms/prohibition/capacity/conflict nodes are feasibility checks (cost 0).
  - Authorization/consent nodes each contribute cost 1 (one valid artifact).
  - AllOf concatenates child plans; total cost is sum.
  - AnyOf picks the lowest-cost valid child.
  - Sequence nodes order the predecessor before the successor.

The oracle produces an OracleSolution used for reward calculation and tests.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from datetime import date
from decimal import Decimal
from typing import Any

from .models import (
    ActionType,
    ApprovalMethod,
    AuthorizationAttempt,
    CreatedAction,
    MaterialContractRequest,
    OracleAuthorizationStep,
    OracleSolution,
    Person,
    RecordType,
    RoleAppointment,
    ValidActionWorld,
    action_request_discriminator,
)
from .requirements import (
    AllOf,
    AnyOf,
    AuthorizationRequirement,
    CapacityRequirement,
    ConflictRequirement,
    ConsentRequirement,
    ProhibitionRequirement,
    RequirementGraph,
    RequirementNode,
    SequenceRequirement,
    SignatoryRequirement,
    TermsRequirement,
    applies_to_action,
    collect_authorization_requirements,
    collect_capacity_requirements,
    collect_conflict_requirements,
    collect_consent_requirements,
    collect_signatory_requirements,
    collect_terms_requirements,
)
from .validator import (
    AuthorizationAttempt,
    body_active,
    body_size,
    body_voters,
    active_roles,
)


def solve_oracle(world: ValidActionWorld) -> OracleSolution:
    plan = _plan_node(world, world.requirements.root, world.action_date, timestamp=0)
    if plan is None:
        return OracleSolution(
            feasible=False,
            minimum_process_cost=0,
            action_payload=world.action_request.model_dump(mode="json"),
            authorization_steps=[],
            signatory_person_ids=[],
            requirement_path_ids=[],
        )

    auth_steps, signatory_ids, requirement_ids = plan
    cost = len(auth_steps)
    return OracleSolution(
        feasible=True,
        minimum_process_cost=cost,
        action_payload=world.action_request.model_dump(mode="json"),
        authorization_steps=auth_steps,
        signatory_person_ids=sorted(set(signatory_ids)),
        requirement_path_ids=requirement_ids,
    )


_PlanStep = tuple[list[OracleAuthorizationStep], list[str], list[str]]


def _plan_node(
    world: ValidActionWorld,
    node: RequirementNode,
    as_of: date,
    timestamp: int,
) -> _PlanStep | None:
    if not node.is_operative(as_of):
        return None
    if isinstance(node, AllOf):
        merged_steps: list[OracleAuthorizationStep] = []
        merged_signatories: list[str] = []
        merged_requirements: list[str] = []
        ts = timestamp
        for child in node.children:
            child_plan = _plan_node(world, child, as_of, ts)
            if child_plan is None:
                return None
            child_steps, child_signatories, child_reqs = child_plan
            merged_steps.extend(child_steps)
            merged_signatories.extend(child_signatories)
            merged_requirements.extend(child_reqs)
            ts += len(child_steps)
        if not node.children:
            return merged_steps, merged_signatories, [node.requirement_id]
        return merged_steps, merged_signatories, [node.requirement_id] + merged_requirements
    if isinstance(node, AnyOf):
        best: _PlanStep | None = None
        for child in node.children:
            child_plan = _plan_node(world, child, as_of, timestamp)
            if child_plan is None:
                continue
            if best is None or len(child_plan[0]) < len(best[0]):
                best = child_plan
        if best is None:
            return None
        return best[0], best[1], [node.requirement_id] + best[2]
    if isinstance(node, AuthorizationRequirement):
        if not applies_to_action(node, world.action_request.action_type):
            return None
        step = _build_authorization_step(world, node, as_of, timestamp)
        if step is None:
            return None
        return [step], [], [node.requirement_id]
    if isinstance(node, ConsentRequirement):
        if not applies_to_action(node, world.action_request.action_type):
            return None
        step = _build_consent_step(world, node, as_of, timestamp)
        if step is None:
            return None
        return [step], [], [node.requirement_id]
    if isinstance(node, TermsRequirement):
        if _terms_match(node, world):
            return [], [], [node.requirement_id]
        return None
    if isinstance(node, CapacityRequirement):
        if _capacity_available(node, world):
            return [], [], [node.requirement_id]
        return None
    if isinstance(node, ConflictRequirement):
        return [], [], [node.requirement_id]
    if isinstance(node, SequenceRequirement):
        return [], [], [node.requirement_id]
    if isinstance(node, SignatoryRequirement):
        signatories = _find_signatories(world, node, as_of)
        if not signatories:
            return None
        return [], signatories, [node.requirement_id]
    if isinstance(node, ProhibitionRequirement):
        if _prohibition_clear(node, world):
            return [], [], [node.requirement_id]
        return None
    return None


def _build_authorization_step(
    world: ValidActionWorld,
    req: AuthorizationRequirement,
    as_of: date,
    timestamp: int,
) -> OracleAuthorizationStep | None:
    body = world.body_by_id(req.authorizer_id)
    if body is None or not body_active(world, req.authorizer_id, as_of):
        return None
    voters = body_voters(world, req.authorizer_id, as_of)
    if not voters:
        return None

    conflict_req = _matching_conflict(world, req.authorizer_id)

    if conflict_req is not None and conflict_req.requires_disclosure_record:
        disclosure_ids = _operative_disclosure_record_ids(world, as_of)
    else:
        disclosure_ids = []

    recused_role_ids: list[str] = []
    eligible_role_ids: list[str] = []
    for role in voters:
        if role.role_id in set(req.eligible_voter_role_ids) or not req.eligible_voter_role_ids:
            if _is_related_party(world, role.person_id):
                recused_role_ids.append(role.role_id)
            else:
                eligible_role_ids.append(role.role_id)

    if not eligible_role_ids and req.eligible_voter_role_ids:
        for role in voters:
            if role.role_id in set(req.eligible_voter_role_ids):
                eligible_role_ids.append(role.role_id)

    if not eligible_role_ids and not recused_role_ids:
        eligible_role_ids = [v.role_id for v in voters]

    method = _preferred_method(req, body)
    return OracleAuthorizationStep(
        requirement_id=req.requirement_id,
        authorizer_id=req.authorizer_id,
        method=method,
        participant_ids=eligible_role_ids,
        recused_ids=recused_role_ids,
        disclosure_record_ids=disclosure_ids,
        timestamp_step=timestamp,
    )


def _build_consent_step(
    world: ValidActionWorld,
    req: ConsentRequirement,
    as_of: date,
    timestamp: int,
) -> OracleAuthorizationStep | None:
    holder_body = world.body_by_id(req.consent_holder_id)
    if holder_body is None:
        return None
    voters = body_voters(world, req.consent_holder_id, as_of)
    if voters:
        return OracleAuthorizationStep(
            requirement_id=req.requirement_id,
            authorizer_id=req.consent_holder_id,
            method=ApprovalMethod.CONTRACTUAL_CONSENT,
            participant_ids=[v.role_id for v in voters],
            timestamp_step=timestamp,
        )
    if holder_body.body_type.value == "security_class":
        class_ids = [s.class_id for s in world.security_classes if s.entity_id == holder_body.entity_id]
        if class_ids:
            holders = [
                h.holder_id
                for h in world.holder_positions
                if h.class_id in class_ids
                and h.effective_from <= as_of
                and (h.effective_until is None or as_of <= h.effective_until)
            ]
            if holders:
                return OracleAuthorizationStep(
                    requirement_id=req.requirement_id,
                    authorizer_id=req.consent_holder_id,
                    method=ApprovalMethod.CONTRACTUAL_CONSENT,
                    participant_ids=holders,
                    timestamp_step=timestamp,
                )
    if req.participant_id is not None:
        return OracleAuthorizationStep(
            requirement_id=req.requirement_id,
            authorizer_id=req.consent_holder_id,
            method=ApprovalMethod.CONTRACTUAL_CONSENT,
            participant_ids=[req.participant_id],
            timestamp_step=timestamp,
        )
    return None


def _preferred_method(req: AuthorizationRequirement, body) -> ApprovalMethod:
    if ApprovalMethod.WRITTEN_CONSENT in req.permitted_methods:
        return ApprovalMethod.WRITTEN_CONSENT
    if ApprovalMethod.MEETING in req.permitted_methods:
        return ApprovalMethod.MEETING
    if ApprovalMethod.DELEGATED_APPROVAL in req.permitted_methods:
        return ApprovalMethod.DELEGATED_APPROVAL
    return req.permitted_methods[0] if req.permitted_methods else ApprovalMethod.MEETING


def _matching_conflict(world: ValidActionWorld, authorizer_id: str) -> ConflictRequirement | None:
    for conflict in collect_conflict_requirements(world.requirements):
        if conflict.required_approval_body_id == authorizer_id:
            return conflict
    return None


def _operative_disclosure_record_ids(world: ValidActionWorld, as_of: date) -> list[str]:
    return [
        r.record_id
        for r in world.records
        if r.record_type == RecordType.CONFLICT_DISCLOSURE
        and r.status == "active"
        and r.effective_date <= as_of
    ]


def _is_related_party(world: ValidActionWorld, person_id: str) -> bool:
    person = world.person_by_id(person_id)
    if person is None:
        return False
    return bool(person.relationships)


def _terms_match(req: TermsRequirement, world: ValidActionWorld) -> bool:
    """A TermsRequirement with op=='ge' or 'le' uses Decimal comparison; otherwise exact equality."""
    request = world.action_request.model_dump(mode="json")
    for key, expected in req.expected_payload.items():
        actual = request.get(key)
        if isinstance(expected, dict) and "op" in expected and "value" in expected:
            try:
                left = Decimal(str(actual))
                right = Decimal(str(expected["value"]))
            except (TypeError, ValueError, ArithmeticError):
                return False
            op = expected["op"]
            if op == "ge" and not (left >= right):
                return False
            if op == "le" and not (left <= right):
                return False
            if op == "eq" and not (left == right):
                return False
            continue
        if actual != expected:
            return False
    return True


def _capacity_available(req: CapacityRequirement, world: ValidActionWorld) -> bool:
    if req.capacity_kind == "authorized_shares":
        sec = next((s for s in world.security_classes if s.class_id == req.target_id), None)
        return sec is not None and sec.available >= req.min_available
    if req.capacity_kind == "plan_pool":
        plan = next((p for p in world.plan_capacities if p.plan_id == req.target_id), None)
        return plan is not None and plan.available >= req.min_available
    return True


def _find_signatories(
    world: ValidActionWorld,
    req: SignatoryRequirement,
    as_of: date,
) -> list[str]:
    roles = active_roles(world, as_of)
    candidates: list[RoleAppointment] = []
    for role in roles:
        if req.eligible_role_ids and role.role_id not in req.eligible_role_ids:
            continue
        if req.eligible_titles and role.title not in req.eligible_titles:
            continue
        if req.entity_id is not None and role.entity_id != req.entity_id:
            continue
        candidates.append(role)
    if not candidates:
        return []
    request_value = _request_max_commitment(world.action_request)
    if req.max_commitment is not None and request_value is not None:
        candidates = [r for r in candidates if request_value <= req.max_commitment]
    return [r.person_id for r in candidates]


def _request_max_commitment(request) -> Any:
    if isinstance(request, MaterialContractRequest):
        return request.total_commitment
    return None


def _prohibition_clear(req: ProhibitionRequirement, world: ValidActionWorld) -> bool:
    request = world.action_request
    if req.predicate == "treasury_floor_breach":
        if not isinstance(request, MaterialContractRequest):
            return True
        return request.total_commitment >= 100
    return True


def replay_oracle_as_attempts(
    world: ValidActionWorld,
    oracle: OracleSolution,
    action_id: str,
) -> list[AuthorizationAttempt]:
    """Convert the oracle's authorization steps into AuthorizationAttempt objects
    suitable for replaying through the validator."""
    out: list[AuthorizationAttempt] = []
    for idx, step in enumerate(oracle.authorization_steps):
        out.append(
            AuthorizationAttempt(
                attempt_id=f"oracle_{action_id}_{idx}",
                action_id=action_id,
                authorizer_id=step.authorizer_id,
                method=step.method,
                participant_ids=list(step.participant_ids),
                recused_ids=list(step.recused_ids),
                disclosure_record_ids=list(step.disclosure_record_ids),
                timestamp_step=step.timestamp_step,
                approved=True,
                defect_codes=[],
            )
        )
    return out


def build_oracle_action(
    world: ValidActionWorld,
    action_id: str,
    timestamp_step: int,
) -> CreatedAction:
    """Construct the created-action draft that the oracle would produce."""
    request = world.action_request
    payload = request.model_dump(mode="json", exclude={"request_id", "business_purpose"})
    return CreatedAction(
        action_id=action_id,
        request_id=request.request_id,
        entity_id=request.entity_id,
        action_date=request.action_date,
        payload=payload,
        action_type=ActionType(request.action_type.value),
        created_step=timestamp_step,
    )


__all__ = [
    "build_oracle_action",
    "replay_oracle_as_attempts",
    "solve_oracle",
]
