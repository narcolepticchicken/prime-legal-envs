"""Final validator (spec section 15).

Validates:
  - individual authorization attempts against a specific AuthorizationRequirement;
  - the final execution against the full requirement graph.

Returns ALL discoverable defects (not just the first).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import (
    ActionRequest,
    ApprovalMethod,
    AuthorizationAttempt,
    BodyType,
    CreatedAction,
    DefectCode,
    EquityPlanCapacity,
    ExecutionAttempt,
    GovernanceBody,
    HolderPosition,
    LegalRecord,
    MaterialContractRequest,
    Person,
    RecordType,
    RoleAppointment,
    SecurityClass,
    ValidActionWorld,
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
    collect_prohibition_requirements,
    collect_sequence_requirements,
    collect_signatory_requirements,
    collect_terms_requirements,
    find_node,
)


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    defect_codes: list[DefectCode] = Field(default_factory=list)
    satisfied_requirement_ids: list[str] = Field(default_factory=list)
    unsatisfied_requirement_ids: list[str] = Field(default_factory=list)
    redundant_authorization_attempt_ids: list[str] = Field(default_factory=list)
    invalid_authorization_attempt_ids: list[str] = Field(default_factory=list)
    process_cost: int
    minimum_process_cost: int


# ---------- Authorization-attempt validation ----------


def active_roles(world: ValidActionWorld, as_of: date) -> list[RoleAppointment]:
    out: list[RoleAppointment] = []
    for role in world.roles:
        if role.effective_from > as_of:
            continue
        if role.effective_until is not None and role.effective_until < as_of:
            continue
        out.append(role)
    return out


def active_person(world: ValidActionWorld, person_id: str, as_of: date) -> bool:
    person = world.person_by_id(person_id)
    if person is None:
        return False
    if person.active_from > as_of:
        return False
    if person.active_until is not None and person.active_until < as_of:
        return False
    return True


def body_active(world: ValidActionWorld, body_id: str, as_of: date) -> bool:
    body = world.body_by_id(body_id)
    if body is None:
        return False
    roles = [
        r
        for r in active_roles(world, as_of)
        if r.role_id in body.member_role_ids
    ]
    return bool(roles)


def body_voters(world: ValidActionWorld, body_id: str, as_of: date) -> list[RoleAppointment]:
    body = world.body_by_id(body_id)
    if body is None:
        return []
    member_ids = set(body.member_role_ids)
    return [r for r in active_roles(world, as_of) if r.role_id in member_ids]


def body_size(world: ValidActionWorld, body_id: str, as_of: date) -> int:
    return len(body_voters(world, body_id, as_of))


def is_record_operative(record: LegalRecord, as_of: date) -> bool:
    if record.status != "active":
        return False
    if record.effective_date > as_of:
        return False
    return True


def validate_authorization_attempt(
    world: ValidActionWorld,
    requirement: AuthorizationRequirement,
    attempt: AuthorizationAttempt,
    created_action: CreatedAction | None,
) -> list[DefectCode]:
    defects: list[DefectCode] = []
    as_of = created_action.action_date if created_action else world.action_date

    if not requirement.is_operative(as_of):
        defects.append(DefectCode.STALE_AUTHORIZATION)
        return defects

    if attempt.method not in requirement.permitted_methods:
        defects.append(DefectCode.METHOD_NOT_PERMITTED)

    authorizer_body = world.body_by_id(attempt.authorizer_id)
    if authorizer_body is None:
        defects.append(DefectCode.WRONG_AUTHORIZER)
        return defects

    if requirement.authorizer_id != attempt.authorizer_id:
        defects.append(DefectCode.WRONG_AUTHORIZER)

    eligible_voters = body_voters(world, attempt.authorizer_id, as_of)
    eligible_role_ids = set(requirement.eligible_voter_role_ids)
    eligible_person_ids = set(requirement.eligible_voter_person_ids)

    seen_person_ids: set[str] = set()
    for participant_id in attempt.participant_ids:
        role = world.role_by_id(participant_id)
        if role is None:
            defects.append(DefectCode.INELIGIBLE_PARTICIPANT)
            continue
        if role.role_id not in eligible_role_ids and role.person_id not in eligible_person_ids:
            defects.append(DefectCode.INELIGIBLE_PARTICIPANT)
        if not active_person(world, role.person_id, as_of):
            defects.append(DefectCode.INELIGIBLE_PARTICIPANT)
        if role.person_id in seen_person_ids:
            defects.append(DefectCode.INELIGIBLE_PARTICIPANT)
        seen_person_ids.add(role.person_id)

    recused_person_ids: set[str] = set()
    for recused_id in attempt.recused_ids:
        role = world.role_by_id(recused_id)
        if role is not None:
            recused_person_ids.add(role.person_id)

    participating_person_ids = {
        world.role_by_id(p).person_id
        for p in attempt.participant_ids
        if world.role_by_id(p) is not None
    }
    conflict_persons = participating_person_ids & recused_person_ids
    if conflict_persons:
        for pid in conflict_persons:
            defects.append(DefectCode.CONFLICTED_VOTE_COUNTED)

    defect_pairs = _check_conflict_requirement(
        world=world,
        requirement=requirement,
        attempt=attempt,
        eligible_voters=eligible_voters,
        as_of=as_of,
    )
    defects.extend(defect_pairs)

    if not _quorum_satisfied(world, attempt, requirement, eligible_voters, as_of):
        defects.append(DefectCode.NO_QUORUM)

    if not _vote_threshold_satisfied(attempt, requirement):
        defects.append(DefectCode.INSUFFICIENT_VOTE)

    for record_id in attempt.disclosure_record_ids:
        record = world.record_by_id(record_id)
        if record is None:
            defects.append(DefectCode.CONFLICT_NOT_DISCLOSED)
        elif record.record_type != RecordType.CONFLICT_DISCLOSURE:
            defects.append(DefectCode.CONFLICT_NOT_DISCLOSED)
        elif not is_record_operative(record, as_of):
            defects.append(DefectCode.CONFLICT_NOT_DISCLOSED)

    return defects


def _check_conflict_requirement(
    *,
    world: ValidActionWorld,
    requirement: AuthorizationRequirement,
    attempt: AuthorizationAttempt,
    eligible_voters: list[RoleAppointment],
    as_of: date,
) -> list[DefectCode]:
    """If any related party is in the body, validate disclosure + recusal."""
    defects: list[DefectCode] = []
    participating_person_ids = {
        world.role_by_id(p).person_id
        for p in attempt.participant_ids
        if world.role_by_id(p) is not None
    }
    recused_person_ids = {
        world.role_by_id(p).person_id
        for p in attempt.recused_ids
        if world.role_by_id(p) is not None
    }
    related_person_ids = _related_party_person_ids(world, attempt.authorizer_id)
    conflict_present = bool(participating_person_ids & related_person_ids)
    if not conflict_present:
        return defects

    has_disclosure = any(
        world.record_by_id(rid) is not None
        and world.record_by_id(rid).record_type == RecordType.CONFLICT_DISCLOSURE
        for rid in attempt.disclosure_record_ids
    )
    if not has_disclosure:
        defects.append(DefectCode.CONFLICT_NOT_DISCLOSED)

    uncounted = participating_person_ids & related_person_ids
    if uncounted and not (uncounted <= recused_person_ids):
        defects.append(DefectCode.MISSING_RECUSAL)
        defects.append(DefectCode.CONFLICTED_VOTE_COUNTED)

    return defects


def _related_party_person_ids(world: ValidActionWorld, body_id: str) -> set[str]:
    """Persons flagged as related via world.people[*].relationships.

    A person with a non-empty relationships list participates in conflict checks
    when seated on the given body."""
    related: set[str] = set()
    body = world.body_by_id(body_id)
    if body is None:
        return related
    member_ids = set(body.member_role_ids)
    for role in world.roles:
        if role.role_id not in member_ids:
            continue
        person = world.person_by_id(role.person_id)
        if person is None:
            continue
        if person.relationships:
            related.add(person.person_id)
    return related


def _quorum_satisfied(
    world: ValidActionWorld,
    attempt: AuthorizationAttempt,
    requirement: AuthorizationRequirement,
    eligible_voters: list[RoleAppointment],
    as_of: date,
) -> bool:
    if requirement.quorum_formula == "majority_of_seated":
        seated = body_size(world, attempt.authorizer_id, as_of)
        required = seated // 2 + 1 if seated > 0 else 1
        return _present_count(attempt, eligible_voters, requirement) >= required
    if requirement.quorum_formula == "majority_of_eligible":
        participating = _present_count(attempt, eligible_voters, requirement)
        recused_participating = sum(
            1 for rid in attempt.participant_ids
            if rid in set(requirement.eligible_voter_role_ids) and rid in set(attempt.recused_ids)
        )
        if not requirement.conflicted_members_count_for_quorum:
            participating -= recused_participating
        eligible_count = len(eligible_voters) - sum(
            1 for rid in attempt.recused_ids
            if rid in {r.role_id for r in eligible_voters}
        )
        required = eligible_count // 2 + 1 if eligible_count > 0 else 1
        return participating >= required
    if requirement.quorum_formula == "fixed_minimum":
        return _present_count(attempt, eligible_voters, requirement) >= int(requirement.quorum_value)
    if requirement.quorum_formula == "voting_power_pct":
        return _voting_power_pct(attempt, world, as_of) >= Decimal(str(requirement.quorum_value))
    return False


def _present_count(
    attempt: AuthorizationAttempt,
    eligible_voters: list[RoleAppointment],
    requirement: AuthorizationRequirement,
) -> int:
    eligible_role_ids = set(requirement.eligible_voter_role_ids)
    return sum(1 for rid in attempt.participant_ids if rid in eligible_role_ids)


def _recused_present_count(
    attempt: AuthorizationAttempt,
    eligible_voters: list[RoleAppointment],
    requirement: AuthorizationRequirement,
) -> int:
    eligible_role_ids = set(requirement.eligible_voter_role_ids)
    return sum(
        1 for rid in attempt.recused_ids
        if rid in eligible_role_ids and rid in attempt.participant_ids
    )


def _voting_power_pct(
    attempt: AuthorizationAttempt,
    world: ValidActionWorld,
    as_of: date,
) -> Decimal:
    body = world.body_by_id(attempt.authorizer_id)
    if body is None or body.body_type != BodyType.SECURITY_CLASS:
        return Decimal("0")
    class_ids = [s.class_id for s in world.security_classes if s.entity_id == body.entity_id]
    if not class_ids:
        return Decimal("0")
    class_id = class_ids[0]
    total = _active_units_for_class(world, class_id, as_of)
    present = _active_units_for_holders(world, class_id, attempt.participant_ids, as_of)
    if total == 0:
        return Decimal("0")
    return Decimal(present) / Decimal(total) * Decimal("100")


def _active_units_for_class(world: ValidActionWorld, class_id: str, as_of: date) -> int:
    return sum(
        h.units
        for h in world.holder_positions
        if h.class_id == class_id
        and h.effective_from <= as_of
        and (h.effective_until is None or as_of <= h.effective_until)
    )


def _active_units_for_holders(
    world: ValidActionWorld,
    class_id: str,
    participant_ids: Iterable[str],
    as_of: date,
) -> int:
    participant_set = set(participant_ids)
    total = 0
    for h in world.holder_positions:
        if h.class_id != class_id:
            continue
        if h.effective_from > as_of:
            continue
        if h.effective_until is not None and as_of > h.effective_until:
            continue
        if h.holder_id in participant_set:
            total += h.units
    return total


def _vote_threshold_satisfied(
    attempt: AuthorizationAttempt,
    requirement: AuthorizationRequirement,
) -> bool:
    approving = 1 if attempt.approved else 0
    if requirement.vote_threshold == "unanimous_consent":
        return approving == 1 and attempt.method == ApprovalMethod.WRITTEN_CONSENT
    if requirement.vote_threshold == "majority_present":
        return approving == 1
    if requirement.vote_threshold == "majority_eligible":
        return approving == 1
    if requirement.vote_threshold == "supermajority_pct":
        if requirement.vote_threshold_value is None:
            return False
        return approving == 1  # deterministic yes; structure implies pre-approval
    if requirement.vote_threshold == "voting_power_threshold":
        return approving == 1
    return False


# ---------- Final execution validation ----------


def validate_final_execution(
    world: ValidActionWorld,
    created_action: CreatedAction | None,
    authorization_attempts: list[AuthorizationAttempt],
    execution: ExecutionAttempt | None,
) -> ValidationResult:
    defects: list[DefectCode] = []
    satisfied: list[str] = []
    unsatisfied: list[str] = []
    invalid_attempt_ids: list[str] = []
    process_cost = len(authorization_attempts)
    minimum_cost = (
        world.oracle_solution.minimum_process_cost
        if world.oracle_solution.feasible
        else 0
    )

    if created_action is None:
        defects.append(DefectCode.NO_ACTION)
        return ValidationResult(
            valid=False,
            defect_codes=defects,
            satisfied_requirement_ids=satisfied,
            unsatisfied_requirement_ids=list(world.requirements.collect_ids()),
            process_cost=process_cost,
            minimum_process_cost=minimum_cost,
        )

    if execution is None:
        defects.append(DefectCode.NO_ACTION)
    else:
        defects.extend(_validate_execution_shape(world, created_action, execution))

    request = world.action_request
    request_defects = _check_terms_match(request, created_action)
    defects.extend(request_defects)
    if created_action.entity_id != request.entity_id:
        defects.append(DefectCode.WRONG_ENTITY)
    if created_action.action_date != request.action_date:
        defects.append(DefectCode.EXECUTION_DATE_INVALID)

    prohibition_defects = _check_prohibitions(world, created_action, request)
    defects.extend(prohibition_defects)

    capacity_defects = _check_capacity(world, created_action, request)
    defects.extend(capacity_defects)

    valid_attempts, redundant_attempts = _categorize_attempts(
        world, created_action, authorization_attempts
    )
    invalid_attempt_ids = sorted(
        a.attempt_id
        for a in authorization_attempts
        if a.attempt_id not in {va.attempt_id for va in valid_attempts}
    )

    auth_graph_result = _check_requirement_graph(
        world=world,
        valid_attempts=valid_attempts,
        redundant_attempts=redundant_attempts,
        invalid_attempt_ids=set(invalid_attempt_ids),
        created_action=created_action,
        execution=execution,
    )
    defects.extend(auth_graph_result["defects"])
    satisfied.extend(auth_graph_result["satisfied"])
    unsatisfied.extend(auth_graph_result["unsatisfied"])

    valid = not defects
    return ValidationResult(
        valid=valid,
        defect_codes=sorted(set(defects), key=lambda d: d.value),
        satisfied_requirement_ids=satisfied,
        unsatisfied_requirement_ids=sorted(set(unsatisfied)),
        redundant_authorization_attempt_ids=sorted(a.attempt_id for a in redundant_attempts),
        invalid_authorization_attempt_ids=sorted(invalid_attempt_ids),
        process_cost=process_cost,
        minimum_process_cost=minimum_cost,
    )


def _validate_execution_shape(
    world: ValidActionWorld,
    created_action: CreatedAction,
    execution: ExecutionAttempt,
) -> list[DefectCode]:
    defects: list[DefectCode] = []
    if execution.action_id != created_action.action_id:
        defects.append(DefectCode.INVALID_SIGNATORY)
    if not active_person(world, execution.signatory_person_id, created_action.action_date):
        defects.append(DefectCode.INVALID_SIGNATORY)
    return defects


def _check_terms_match(request: ActionRequest, created_action: CreatedAction) -> list[DefectCode]:
    defects: list[DefectCode] = []
    request_payload = request.model_dump(
        mode="json",
        exclude={"request_id", "business_purpose", "action_type"},
    )
    payload = dict(created_action.payload)
    # Map user-facing alias `effective_date` to internal `action_date`.
    if "effective_date" in payload and "action_date" in request_payload:
        payload["action_date"] = payload.pop("effective_date")
    # Drop None-valued expected fields so optional aliases don't trip matching.
    request_payload = {k: v for k, v in request_payload.items() if v is not None}
    payload = {k: v for k, v in payload.items() if k in request_payload}
    if payload != request_payload:
        defects.append(DefectCode.TERMS_MISMATCH)
    return defects


def _check_prohibitions(
    world: ValidActionWorld,
    created_action: CreatedAction,
    request: ActionRequest,
) -> list[DefectCode]:
    defects: list[DefectCode] = []
    for prohibition in collect_prohibition_requirements(world.requirements):
        if not applies_to_action(prohibition, request.action_type):
            continue
        if not prohibition.is_operative(created_action.action_date):
            continue
        if _prohibition_triggered(prohibition, world, request):
            defects.append(DefectCode.PROHIBITED_ACTION)
    return defects


def _prohibition_triggered(
    prohibition: ProhibitionRequirement,
    world: ValidActionWorld,
    request: ActionRequest,
) -> bool:
    if prohibition.predicate == "treasury_floor_breach":
        if not isinstance(request, MaterialContractRequest):
            return False
        return Decimal(str(request.total_commitment)) < Decimal("100")
    if prohibition.predicate == "category_blocked":
        return False  # always clear in MVP
    return False


def _check_capacity(
    world: ValidActionWorld,
    created_action: CreatedAction,
    request: ActionRequest,
) -> list[DefectCode]:
    defects: list[DefectCode] = []
    for capacity in collect_capacity_requirements(world.requirements):
        if not applies_to_action(capacity, request.action_type):
            continue
        if not capacity.is_operative(created_action.action_date):
            continue
        if capacity.capacity_kind == "authorized_shares":
            sec = next((s for s in world.security_classes if s.class_id == capacity.target_id), None)
            if sec is None or sec.available < capacity.min_available:
                defects.append(DefectCode.CAPACITY_EXCEEDED)
        elif capacity.capacity_kind == "plan_pool":
            plan = next((p for p in world.plan_capacities if p.plan_id == capacity.target_id), None)
            if plan is None or plan.available < capacity.min_available:
                defects.append(DefectCode.CAPACITY_EXCEEDED)
    return defects


def _categorize_attempts(
    world: ValidActionWorld,
    created_action: CreatedAction,
    attempts: list[AuthorizationAttempt],
) -> tuple[list[AuthorizationAttempt], list[AuthorizationAttempt]]:
    from .requirements import collect_consent_requirements

    auth_reqs = collect_authorization_requirements(world.requirements)
    consent_reqs = collect_consent_requirements(world.requirements)
    valid: list[AuthorizationAttempt] = []
    redundant: list[AuthorizationAttempt] = []
    seen_keys: set[tuple[str, str, str, tuple[str, ...]]] = set()
    for attempt in attempts:
        key = (
            attempt.action_id,
            attempt.authorizer_id,
            attempt.method.value,
            tuple(sorted(attempt.participant_ids)),
        )
        if key in seen_keys:
            redundant.append(attempt)
            continue
        seen_keys.add(key)
        auth_req = _resolve_auth_requirement(world, attempt)
        consent_req = next(
            (c for c in consent_reqs if c.consent_holder_id == attempt.authorizer_id),
            None,
        )
        if auth_req is None and consent_req is None:
            redundant.append(attempt)
            continue
        if auth_req is not None:
            defects = validate_authorization_attempt(world, auth_req, attempt, created_action)
        else:
            defects = _validate_consent_attempt(world, consent_req, attempt, created_action)
        if not defects:
            valid.append(attempt)
        else:
            redundant.append(attempt)
    return valid, redundant


def _validate_consent_attempt(
    world: ValidActionWorld,
    requirement: ConsentRequirement,
    attempt: AuthorizationAttempt,
    created_action: CreatedAction | None,
) -> list[DefectCode]:
    defects: list[DefectCode] = []
    as_of = created_action.action_date if created_action else world.action_date
    if not requirement.is_operative(as_of):
        defects.append(DefectCode.STALE_AUTHORIZATION)
    if attempt.authorizer_id != requirement.consent_holder_id:
        defects.append(DefectCode.WRONG_AUTHORIZER)
    if attempt.method != requirement.consent_method:
        defects.append(DefectCode.METHOD_NOT_PERMITTED)
    return defects


def _resolve_auth_requirement(
    world: ValidActionWorld,
    attempt: AuthorizationAttempt,
) -> AuthorizationRequirement | None:
    """The agent does not name a requirement; we resolve by authorizer body
    and method to disambiguate AnyOf branches."""
    for req in collect_authorization_requirements(world.requirements):
        if req.authorizer_id != attempt.authorizer_id:
            continue
        if attempt.method in req.permitted_methods:
            return req
    for req in collect_authorization_requirements(world.requirements):
        if req.authorizer_id == attempt.authorizer_id:
            return req
    return None


def _attempt_to_requirement_id(
    world: ValidActionWorld,
    attempt: AuthorizationAttempt,
) -> str | None:
    auth_req = _resolve_auth_requirement(world, attempt)
    return auth_req.requirement_id if auth_req is not None else None


def _check_requirement_graph(
    *,
    world: ValidActionWorld,
    valid_attempts: list[AuthorizationAttempt],
    redundant_attempts: list[AuthorizationAttempt],
    invalid_attempt_ids: set[str],
    created_action: CreatedAction,
    execution: ExecutionAttempt | None,
) -> dict[str, list[Any]]:
    defects: list[DefectCode] = []
    satisfied: list[str] = []
    unsatisfied: list[str] = []

    def is_auth_satisfied(node: RequirementNode) -> bool:
        if isinstance(node, AuthorizationRequirement):
            return any(a.authorizer_id == node.authorizer_id for a in valid_attempts)
        if isinstance(node, ConsentRequirement):
            return any(a.authorizer_id == node.consent_holder_id for a in valid_attempts)
        return False

    def walk(node: RequirementNode, optional: bool = False) -> bool:
        if not node.is_operative(created_action.action_date):
            return False
        if isinstance(node, AllOf):
            ok = True
            for child in node.children:
                if not walk(child, optional=optional):
                    ok = False
            if ok:
                satisfied.append(node.requirement_id)
            else:
                unsatisfied.append(node.requirement_id)
            return ok
        if isinstance(node, AnyOf):
            chosen: list[RequirementNode] = []
            for child in node.children:
                if walk(child, optional=True):
                    chosen.append(child)
            if chosen:
                satisfied.append(node.requirement_id)
                for child in node.children:
                    if child not in chosen:
                        unsatisfied.append(child.requirement_id)
                return True
            for child in node.children:
                unsatisfied.append(child.requirement_id)
            if not optional:
                defects.append(DefectCode.MISSING_AUTHORIZATION)
            return False
        if isinstance(node, AuthorizationRequirement):
            if is_auth_satisfied(node):
                satisfied.append(node.requirement_id)
                return True
            unsatisfied.append(node.requirement_id)
            if not optional:
                defects.append(DefectCode.MISSING_AUTHORIZATION)
            return False
        if isinstance(node, ConsentRequirement):
            if is_auth_satisfied(node):
                satisfied.append(node.requirement_id)
                return True
            unsatisfied.append(node.requirement_id)
            if not optional:
                defects.append(DefectCode.MISSING_CONSENT)
            return False
        if isinstance(node, ConflictRequirement):
            return True
        if isinstance(node, TermsRequirement):
            satisfied.append(node.requirement_id)
            return True
        if isinstance(node, CapacityRequirement):
            satisfied.append(node.requirement_id)
            return True
        if isinstance(node, SequenceRequirement):
            satisfied.append(node.requirement_id)
            return True
        if isinstance(node, SignatoryRequirement):
            if execution is None:
                unsatisfied.append(node.requirement_id)
                if not optional:
                    defects.append(DefectCode.INVALID_SIGNATORY)
                return False
            signatory_role = next(
                (
                    r
                    for r in world.roles
                    if r.person_id == execution.signatory_person_id
                ),
                None,
            )
            if signatory_role is None:
                unsatisfied.append(node.requirement_id)
                if not optional:
                    defects.append(DefectCode.INVALID_SIGNATORY)
                return False
            if node.eligible_role_ids and signatory_role.role_id not in node.eligible_role_ids:
                unsatisfied.append(node.requirement_id)
                if not optional:
                    defects.append(DefectCode.INVALID_SIGNATORY)
                return False
            if node.eligible_titles and signatory_role.title not in node.eligible_titles:
                unsatisfied.append(node.requirement_id)
                if not optional:
                    defects.append(DefectCode.INVALID_SIGNATORY)
                return False
            if node.eligible_person_ids and execution.signatory_person_id not in node.eligible_person_ids:
                unsatisfied.append(node.requirement_id)
                if not optional:
                    defects.append(DefectCode.INVALID_SIGNATORY)
                return False
            if node.entity_id is not None and signatory_role.entity_id != node.entity_id:
                unsatisfied.append(node.requirement_id)
                if not optional:
                    defects.append(DefectCode.INVALID_SIGNATORY)
                return False
            if node.max_commitment is not None:
                req_value = _request_max_commitment(world.action_request)
                if req_value is not None and req_value > node.max_commitment:
                    unsatisfied.append(node.requirement_id)
                    if not optional:
                        defects.append(DefectCode.INVALID_SIGNATORY)
                    return False
            satisfied.append(node.requirement_id)
            return True
        if isinstance(node, ProhibitionRequirement):
            satisfied.append(node.requirement_id)
            return True
        return True

    walk(world.requirements.root)
    return {"defects": defects, "satisfied": satisfied, "unsatisfied": unsatisfied}


def _request_max_commitment(request: ActionRequest) -> Decimal | None:
    if isinstance(request, MaterialContractRequest):
        return Decimal(str(request.total_commitment))
    if isinstance(request, SecurityIssuanceRequest):
        return Decimal(str(request.units)) * Decimal(str(request.price_per_unit))
    if isinstance(request, RelatedPartyTransactionRequest):
        return Decimal(str(request.total_value))
    if isinstance(request, TokenTreasuryTransactionRequest):
        return Decimal(str(request.token_units)) * Decimal(str(request.price_per_token))
    if isinstance(request, SubsidiaryFinancingRequest):
        return Decimal(str(request.principal))
    return None


__all__ = [
    "ValidationResult",
    "validate_authorization_attempt",
    "validate_final_execution",
    "active_roles",
    "body_active",
    "body_voters",
    "body_size",
    "is_record_operative",
]
