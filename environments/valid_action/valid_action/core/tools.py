"""Tool implementations for valid-action (spec section 12).

Each tool returns a JSON-compatible dict with the envelope:
    { tool, remaining_turns, preflight_checks_remaining, action_status, terminal, ... }

Tools are bound to `state` and `task` via the Taskset Toolset. The agent-visible
schema excludes `task` and `state`; only model-supplied arguments appear.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from decimal import Decimal
from typing import Any

from .models import (
    ActionType,
    ApprovalMethod,
    AuthorizationAttempt,
    CreatedAction,
    EquityGrantRequest,
    EquityPlanCapacity,
    ExecutionAttempt,
    MaterialContractRequest,
    RelatedPartyTransactionRequest,
    RecordType,
    SecurityClass,
    SecurityIssuanceRequest,
    SubsidiaryFinancingRequest,
    TokenTreasuryTransactionRequest,
    ValidActionWorld,
    action_request_discriminator,
)
from .oracle import solve_oracle
from .requirements import (
    AuthorizationRequirement,
    CapacityRequirement,
    collect_authorization_requirements,
    collect_capacity_requirements,
)
from .search import LexicalIndex
from .validator import (
    DefectCode,
    active_roles,
    body_voters,
    validate_authorization_attempt,
    validate_final_execution,
)


SCHEMA_VERSION = "valid-action.tools.v1"


def _envelope(
    tool: str,
    state: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
    terminal: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    runtime = state.setdefault("runtime", {})
    max_turns = int(runtime.get("max_turns", 14))
    used_turns = int(state.get("turn_count", 0))
    remaining = max(0, max_turns - used_turns)
    preflight = int(state.get("preflight_remaining", 1))
    if state.get("execution_attempt") is not None:
        action_status = "executed"
    elif state.get("created_action") is not None:
        action_status = "authorized" if state.get("authorization_attempts") else "draft"
    else:
        action_status = "none"
    out: dict[str, Any] = {
        "tool": tool,
        "schema_version": SCHEMA_VERSION,
        "remaining_turns": remaining,
        "preflight_checks_remaining": preflight,
        "action_status": action_status,
        "terminal": terminal,
    }
    if error is not None:
        out["error"] = error
    if payload is not None:
        out.update(payload)
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=_json_default, ensure_ascii=False)


def _world(state: dict[str, Any]) -> ValidActionWorld:
    w = state.get("world")
    if isinstance(w, ValidActionWorld):
        return w
    # state["world"] is a canonical dict (set up by the rollout setup)
    if isinstance(w, Mapping):
        return ValidActionWorld.model_validate(w)
    raise RuntimeError("state['world'] missing or not a Mapping/ValidActionWorld")


def _record_event(state: dict[str, Any], event: str) -> None:
    state.setdefault("event_log", []).append(event)


def _increment_tool_error(state: dict[str, Any], code: str) -> None:
    state.setdefault("tool_errors", []).append(code)


def _bump_turn(state: dict[str, Any]) -> None:
    state["turn_count"] = int(state.get("turn_count", 0)) + 1


def _index(state: dict[str, Any]) -> LexicalIndex:
    cached = state.get("search_index")
    if cached is None:
        cached = LexicalIndex(_world(state).records)
        state["search_index"] = cached
    return cached


def tool_search_records(
    query: str,
    entity_id: str | None = None,
    record_type: str | None = None,
    *,
    task: dict[str, Any],
    state: dict[str, Any],
) -> str:
    _bump_turn(state)
    state.setdefault("search_queries", []).append(query)
    if not query.strip():
        _increment_tool_error(state, "empty_query")
        return _serialize(_envelope("search_records", state, error="empty query"))
    if record_type is not None and record_type not in {rt.value for rt in RecordType}:
        _increment_tool_error(state, "unknown_record_type")
        return _serialize(
            _envelope(
                "search_records",
                state,
                error=f"unknown record_type {record_type}",
            )
        )
    try:
        results = _index(state).search(
            query,
            entity_id=entity_id,
            record_type=record_type,
            max_results=int(state.get("max_search_results", 5)),
        )
    except ValueError as exc:
        _increment_tool_error(state, "search_value_error")
        return _serialize(_envelope("search_records", state, error=str(exc)))
    _record_event(state, f"search_records({query!r}) -> {len(results)}")
    return _serialize(_envelope("search_records", state, payload={"results": results}))


def tool_read_record(record_id: str, *, task: dict[str, Any], state: dict[str, Any]) -> str:
    _bump_turn(state)
    if state.get("execution_attempt") is not None:
        return _serialize(
            _envelope(
                "read_record",
                state,
                error="rollout already terminated; cannot read after execution",
                terminal=True,
            )
        )
    state.setdefault("documents_read", []).append(record_id)
    record = _world(state).record_by_id(record_id)
    if record is None:
        suggestions = [r.record_id for r in _world(state).records][:5]
        _increment_tool_error(state, "unknown_record_id")
        return _serialize(
            _envelope(
                "read_record",
                state,
                error=f"unknown record {record_id}",
                payload={"suggestions": suggestions},
            )
        )
    out = record.model_dump(mode="json")
    out["sections"] = [
        {k: v for k, v in section.items() if k != "source_rule_ids"}
        for section in out["sections"]
    ]
    _record_event(state, f"read_record({record_id})")
    return _serialize(_envelope("read_record", state, payload={"record": out}))


def tool_inspect_register(
    entity_id: str,
    register_type: str,
    as_of_date: str,
    *,
    task: dict[str, Any],
    state: dict[str, Any],
) -> str:
    _bump_turn(state)
    try:
        as_of = date.fromisoformat(as_of_date)
    except ValueError:
        _increment_tool_error(state, "bad_date")
        return _serialize(_envelope("inspect_register", state, error=f"bad date {as_of_date}"))
    world = _world(state)
    if world.entity_by_id(entity_id) is None:
        _increment_tool_error(state, "unknown_entity")
        return _serialize(
            _envelope(
                "inspect_register",
                state,
                error=f"unknown entity {entity_id}",
            )
        )
    state.setdefault("register_inspections", []).append(
        {"entity_id": entity_id, "register_type": register_type, "as_of": as_of_date}
    )
    payload: dict[str, Any] = {"as_of_date": as_of_date}

    def _active_role(role):
        if role.entity_id != entity_id:
            return False
        if role.effective_from > as_of:
            return False
        if role.effective_until is not None and role.effective_until < as_of:
            return False
        return True

    if register_type == "directors":
        payload["members"] = [
            r.model_dump(mode="json")
            for r in world.roles
            if r.role_type == "director" and _active_role(r)
        ]
    elif register_type == "officers":
        payload["members"] = [
            r.model_dump(mode="json")
            for r in world.roles
            if r.role_type == "officer" and _active_role(r)
        ]
    elif register_type == "committee_members":
        payload["members"] = [
            r.model_dump(mode="json")
            for r in world.roles
            if r.role_type == "committee_member" and _active_role(r)
        ]
    elif register_type == "cap_table":
        payload["positions"] = [
            h.model_dump(mode="json")
            for h in world.holder_positions
            if h.effective_from <= as_of
            and (h.effective_until is None or as_of <= h.effective_until)
        ]
        payload["classes"] = [s.model_dump(mode="json") for s in world.security_classes]
    elif register_type == "security_capacity":
        payload["classes"] = [
            {
                **s.model_dump(mode="json"),
                "available": s.available,
            }
            for s in world.security_classes
            if s.entity_id == entity_id
        ]
    elif register_type == "equity_plan_capacity":
        payload["plans"] = [
            {
                **p.model_dump(mode="json"),
                "available": p.available,
            }
            for p in world.plan_capacities
            if p.entity_id == entity_id
        ]
    elif register_type == "subsidiaries":
        payload["entities"] = [
            e.model_dump(mode="json")
            for e in world.entities
            if e.parent_entity_id == entity_id
        ]
    else:
        _increment_tool_error(state, "unknown_register_type")
        return _serialize(
            _envelope(
                "inspect_register",
                state,
                error=f"unknown register_type {register_type}",
            )
        )
    _record_event(state, f"inspect_register({entity_id},{register_type},{as_of_date})")
    return _serialize(_envelope("inspect_register", state, payload=payload))


def _create_action_draft(
    *,
    world: ValidActionWorld,
    action_type: ActionType,
    payload: dict[str, Any],
    created_step: int,
) -> CreatedAction:
    request = world.action_request
    if request.action_type != action_type:
        raise ValueError(
            f"create tool {action_type.value} does not match task action {request.action_type.value}"
        )
    action_id = _next_action_id(world)
    return CreatedAction(
        action_id=action_id,
        request_id=request.request_id,
        entity_id=payload["entity_id"],
        action_date=payload.get("effective_date") or payload.get("action_date") or request.action_date,
        payload=payload,
        action_type=action_type,
        created_step=created_step,
    )


def _next_action_id(world: ValidActionWorld) -> str:
    return f"act_{world.world_template_id.lower()}_{world.seed}"


def _check_payload_match(request, payload: dict[str, Any]) -> list[DefectCode]:
    expected = request.model_dump(
        mode="json",
        exclude={"request_id", "business_purpose", "action_type"},
    )
    normalized = dict(payload)
    # Map user-facing alias `effective_date` to internal `action_date`.
    if "effective_date" in normalized and "action_date" in expected:
        normalized["action_date"] = normalized.pop("effective_date")
    # Drop None-valued expected fields so optional aliases don't trip matching.
    expected = {k: v for k, v in expected.items() if v is not None}
    fields = {k: v for k, v in normalized.items() if k in expected}
    return [] if fields == expected else [DefectCode.TERMS_MISMATCH]


def _handle_create(
    state: dict[str, Any],
    action_type: ActionType,
    payload: dict[str, Any],
) -> str:
    """Shared create-tool logic. Each create_X function passes its signature dict here."""
    _bump_turn(state)
    if state.get("created_action") is not None:
        _increment_tool_error(state, "duplicate_create")
        return _serialize(
            _envelope(
                f"create_{action_type.value}",
                state,
                error="action already created; one action per rollout",
            )
        )
    world = _world(state)
    try:
        defects = _check_payload_match(world.action_request, payload)
    except (ValueError, KeyError, TypeError) as exc:
        _increment_tool_error(state, "bad_create_args")
        return _serialize(_envelope(f"create_{action_type.value}", state, error=str(exc)))
    if defects:
        _increment_tool_error(state, "terms_mismatch")
        return _serialize(
            _envelope(
                f"create_{action_type.value}",
                state,
                error=f"terms mismatch ({', '.join(d.value for d in defects)})",
            )
        )
    created = _create_action_draft(
        world=world,
        action_type=action_type,
        payload=payload,
        created_step=int(state.get("turn_count", 1)),
    )
    state["created_action"] = created
    _record_event(state, f"create_{action_type.value}({created.action_id})")
    return _serialize(
        _envelope(
            f"create_{action_type.value}",
            state,
            payload={"action_id": created.action_id, "defect_codes": []},
        )
    )


def _decimal_str(value: Any) -> Any:
    if value is None:
        return None
    return str(Decimal(str(value)))


def create_equity_grant(
    entity_id: str,
    recipient_person_id: str,
    award_type: str,
    units: int,
    vesting_months: int,
    cliff_months: int,
    effective_date: str,
    strike_price: str | None = None,
    *,
    state: dict[str, Any],
    task: dict[str, Any],
) -> str:
    return _handle_create(
        state,
        ActionType.EQUITY_GRANT,
        {
            "entity_id": entity_id,
            "recipient_person_id": recipient_person_id,
            "award_type": award_type,
            "units": units,
            "vesting_months": vesting_months,
            "cliff_months": cliff_months,
            "effective_date": effective_date,
            "strike_price": _decimal_str(strike_price),
        },
    )


def create_material_contract(
    entity_id: str,
    counterparty_id: str,
    contract_category: str,
    total_commitment: str,
    term_months: int,
    effective_date: str,
    includes_exclusivity: bool = False,
    includes_data_processing: bool = False,
    *,
    state: dict[str, Any],
    task: dict[str, Any],
) -> str:
    return _handle_create(
        state,
        ActionType.MATERIAL_CONTRACT,
        {
            "entity_id": entity_id,
            "counterparty_id": counterparty_id,
            "contract_category": contract_category,
            "total_commitment": _decimal_str(total_commitment),
            "term_months": term_months,
            "effective_date": effective_date,
            "includes_exclusivity": includes_exclusivity,
            "includes_data_processing": includes_data_processing,
        },
    )


def create_security_issuance(
    entity_id: str,
    purchaser_id: str,
    class_id: str,
    units: int,
    price_per_unit: str,
    financing_round: str,
    effective_date: str,
    *,
    state: dict[str, Any],
    task: dict[str, Any],
) -> str:
    return _handle_create(
        state,
        ActionType.SECURITY_ISSUANCE,
        {
            "entity_id": entity_id,
            "purchaser_id": purchaser_id,
            "class_id": class_id,
            "units": units,
            "price_per_unit": _decimal_str(price_per_unit),
            "financing_round": financing_round,
            "effective_date": effective_date,
        },
    )


def create_related_party_transaction(
    entity_id: str,
    counterparty_id: str,
    related_person_id: str,
    transaction_category: str,
    total_value: str,
    effective_date: str,
    *,
    state: dict[str, Any],
    task: dict[str, Any],
) -> str:
    return _handle_create(
        state,
        ActionType.RELATED_PARTY_TRANSACTION,
        {
            "entity_id": entity_id,
            "counterparty_id": counterparty_id,
            "related_person_id": related_person_id,
            "transaction_category": transaction_category,
            "total_value": _decimal_str(total_value),
            "effective_date": effective_date,
        },
    )


def create_token_treasury_transaction(
    entity_id: str,
    counterparty_id: str,
    token_units: str,
    price_per_token: str,
    lockup_months: int,
    effective_date: str,
    related_person_id: str | None = None,
    *,
    state: dict[str, Any],
    task: dict[str, Any],
) -> str:
    return _handle_create(
        state,
        ActionType.TOKEN_TREASURY_TRANSACTION,
        {
            "entity_id": entity_id,
            "counterparty_id": counterparty_id,
            "token_units": _decimal_str(token_units),
            "price_per_token": _decimal_str(price_per_token),
            "lockup_months": lockup_months,
            "effective_date": effective_date,
            "related_person_id": related_person_id,
        },
    )


def create_subsidiary_financing(
    entity_id: str,
    subsidiary_id: str,
    lender_id: str,
    principal: str,
    maturity_months: int,
    effective_date: str,
    parent_guarantee: bool = False,
    *,
    state: dict[str, Any],
    task: dict[str, Any],
) -> str:
    return _handle_create(
        state,
        ActionType.SUBSIDIARY_FINANCING,
        {
            "entity_id": entity_id,
            "subsidiary_id": subsidiary_id,
            "lender_id": lender_id,
            "principal": _decimal_str(principal),
            "maturity_months": maturity_months,
            "effective_date": effective_date,
            "parent_guarantee": parent_guarantee,
        },
    )


def tool_record_authorization(
    action_id: str,
    authorizer_id: str,
    method: str,
    participant_ids: list[str],
    recused_ids: list[str] | None = None,
    disclosure_record_ids: list[str] | None = None,
    *,
    task: dict[str, Any],
    state: dict[str, Any],
) -> str:
    _bump_turn(state)
    if state.get("created_action") is None:
        _increment_tool_error(state, "auth_before_action")
        return _serialize(
            _envelope(
                "record_authorization",
                state,
                error="authorization cannot be recorded before action is created",
            )
        )
    if state.get("execution_attempt") is not None:
        return _serialize(
            _envelope(
                "record_authorization",
                state,
                error="rollout already terminated; cannot record after execution",
                terminal=True,
            )
        )
    created = state["created_action"]
    if action_id != created.action_id:
        _increment_tool_error(state, "auth_wrong_action")
        return _serialize(
            _envelope(
                "record_authorization",
                state,
                error=f"action_id {action_id} does not match created action",
            )
        )
    try:
        method_enum = ApprovalMethod(method)
    except ValueError:
        _increment_tool_error(state, "bad_method")
        return _serialize(
            _envelope("record_authorization", state, error=f"unknown method {method}")
        )
    world = _world(state)
    requirement = _match_authorization_requirement(world, authorizer_id)
    if requirement is None:
        # Fall back to consent requirement matching
        consent_req = _match_consent_requirement(world, authorizer_id)
        if consent_req is not None:
            return _handle_record_consent(
                action_id=action_id,
                consent_req=consent_req,
                method_enum=method_enum,
                participant_ids=list(participant_ids),
                recused_ids=list(recused_ids or []),
                disclosure_record_ids=list(disclosure_record_ids or []),
                state=state,
            )
        _increment_tool_error(state, "unknown_authorizer")
        return _serialize(
            _envelope(
                "record_authorization",
                state,
                error=f"unknown authorizer {authorizer_id}",
            )
        )
    attempt = AuthorizationAttempt(
        attempt_id=f"auth_{len(state.get('authorization_attempts', [])) + 1}",
        action_id=action_id,
        authorizer_id=authorizer_id,
        method=method_enum,
        participant_ids=list(participant_ids),
        recused_ids=list(recused_ids or []),
        disclosure_record_ids=list(disclosure_record_ids or []),
        timestamp_step=int(state.get("turn_count", 1)),
        approved=True,
        defect_codes=[],
    )
    defects = validate_authorization_attempt(
        world=world,
        requirement=requirement,
        attempt=attempt,
        created_action=created,
    )
    attempt.defect_codes = defects
    state.setdefault("authorization_attempts", []).append(attempt)
    if defects:
        _record_event(
            state,
            f"record_authorization({authorizer_id}) invalid: {', '.join(d.value for d in defects)}",
        )
    else:
        _record_event(state, f"record_authorization({authorizer_id}) approved")
    return _serialize(
        _envelope(
            "record_authorization",
            state,
            payload={
                "attempt_id": attempt.attempt_id,
                "approved": not bool(defects),
                "defect_codes": [d.value for d in defects],
            },
        )
    )


def _match_authorization_requirement(world: ValidActionWorld, authorizer_id: str) -> AuthorizationRequirement | None:
    for req in collect_authorization_requirements(world.requirements):
        if req.authorizer_id == authorizer_id:
            return req
    return None


def _match_consent_requirement(world: ValidActionWorld, holder_id: str) -> ConsentRequirement | None:
    from .requirements import ConsentRequirement, collect_consent_requirements

    for req in collect_consent_requirements(world.requirements):
        if req.consent_holder_id == holder_id:
            return req
    return None


def _handle_record_consent(
    *,
    action_id: str,
    consent_req: "ConsentRequirement",
    method_enum: ApprovalMethod,
    participant_ids: list[str],
    recused_ids: list[str],
    disclosure_record_ids: list[str],
    state: dict[str, Any],
) -> str:
    """Record a consent step (counterparty, security class, etc.)."""
    attempt = AuthorizationAttempt(
        attempt_id=f"consent_{len(state.get('authorization_attempts', [])) + 1}",
        action_id=action_id,
        authorizer_id=consent_req.consent_holder_id,
        method=method_enum,
        participant_ids=participant_ids,
        recused_ids=recused_ids,
        disclosure_record_ids=disclosure_record_ids,
        timestamp_step=int(state.get("turn_count", 1)),
        approved=True,
        defect_codes=[],
    )
    state.setdefault("authorization_attempts", []).append(attempt)
    _record_event(state, f"record_consent(holder={consent_req.consent_holder_id})")
    return _serialize(
        _envelope(
            "record_authorization",
            state,
            payload={
                "attempt_id": attempt.attempt_id,
                "kind": "consent",
                "requirement_id": consent_req.requirement_id,
            },
        )
    )


def tool_preflight_action(action_id: str, *, task: dict[str, Any], state: dict[str, Any]) -> str:
    _bump_turn(state)
    if int(state.get("preflight_remaining", 0)) <= 0:
        _increment_tool_error(state, "preflight_exhausted")
        return _serialize(
            _envelope("preflight_action", state, error="preflight budget exhausted")
        )
    state["preflight_remaining"] = int(state["preflight_remaining"]) - 1
    world = _world(state)
    created = state.get("created_action")
    if created is None or created.action_id != action_id:
        _increment_tool_error(state, "preflight_wrong_action")
        return _serialize(
            _envelope(
                "preflight_action",
                state,
                error=f"action {action_id} not created",
            )
        )
    synthetic_execution = ExecutionAttempt(
        execution_id="preflight",
        action_id=action_id,
        signatory_person_id=(
            world.oracle_solution.signatory_person_ids[0]
            if world.oracle_solution.signatory_person_ids
            else "prs_unknown"
        ),
        timestamp_step=int(state.get("turn_count", 1)),
        valid=True,
        defect_codes=[],
    )
    result = validate_final_execution(
        world=world,
        created_action=created,
        authorization_attempts=state.get("authorization_attempts", []),
        execution=synthetic_execution,
    )
    state.setdefault("preflight_used", 0)
    state["preflight_used"] += 1
    _record_event(state, "preflight_action")
    return _serialize(
        _envelope(
            "preflight_action",
            state,
            payload={
                "valid_if_executed_now": bool(result.valid),
                "defect_codes": [d.value for d in result.defect_codes],
            },
        )
    )


def tool_execute_action(
    action_id: str,
    signatory_person_id: str,
    *,
    task: dict[str, Any],
    state: dict[str, Any],
) -> str:
    _bump_turn(state)
    if state.get("execution_attempt") is not None:
        _increment_tool_error(state, "duplicate_execution")
        return _serialize(
            _envelope(
                "execute_action",
                state,
                error="execution already attempted; one execution per rollout",
                terminal=True,
            )
        )
    created = state.get("created_action")
    if created is None or created.action_id != action_id:
        _increment_tool_error(state, "execute_without_action")
        return _serialize(
            _envelope(
                "execute_action",
                state,
                error=f"action {action_id} not created",
            )
        )
    execution = ExecutionAttempt(
        execution_id=f"exec_{action_id}",
        action_id=action_id,
        signatory_person_id=signatory_person_id,
        timestamp_step=int(state.get("turn_count", 1)),
        valid=True,
        defect_codes=[],
    )
    result = validate_final_execution(
        world=_world(state),
        created_action=created,
        authorization_attempts=state.get("authorization_attempts", []),
        execution=execution,
    )
    execution.valid = result.valid
    execution.defect_codes = result.defect_codes
    state["execution_attempt"] = execution
    state["score_breakdown"] = {
        "process_cost": result.process_cost,
        "minimum_process_cost": result.minimum_process_cost,
    }
    state["validation_result"] = result.model_dump(mode="json")
    _record_event(state, f"execute_action(approved={result.valid})")
    state["done"] = True
    if hasattr(state, "stop"):
        try:
            state.stop("execution")
        except Exception:
            pass
    return _serialize(
        _envelope(
            "execute_action",
            state,
            payload={
                "execution_id": execution.execution_id,
                "valid": result.valid,
                "defect_codes": [d.value for d in result.defect_codes],
                "process_cost": result.process_cost,
                "minimum_process_cost": result.minimum_process_cost,
            },
            terminal=True,
        )
    )


ALL_CREATE_TOOL_FUNCTIONS = {
    "create_equity_grant": create_equity_grant,
    "create_material_contract": create_material_contract,
    "create_security_issuance": create_security_issuance,
    "create_related_party_transaction": create_related_party_transaction,
    "create_token_treasury_transaction": create_token_treasury_transaction,
    "create_subsidiary_financing": create_subsidiary_financing,
}


def make_tools_for_action(action_type: ActionType) -> dict[str, Callable[..., str]]:
    return {f"create_{action_type.value}": ALL_CREATE_TOOL_FUNCTIONS[f"create_{action_type.value}"]}


ALL_CREATE_TOOLS = set(ALL_CREATE_TOOL_FUNCTIONS.keys())


def search_read_toolset() -> list[Callable[..., str]]:
    return [tool_search_records, tool_read_record, tool_inspect_register]


def auth_toolset() -> list[Callable[..., str]]:
    return [tool_record_authorization, tool_preflight_action, tool_execute_action]


__all__ = [
    "ALL_CREATE_TOOLS",
    "ALL_CREATE_TOOL_FUNCTIONS",
    "auth_toolset",
    "create_equity_grant",
    "create_material_contract",
    "create_related_party_transaction",
    "create_security_issuance",
    "create_subsidiary_financing",
    "create_token_treasury_transaction",
    "make_tools_for_action",
    "search_read_toolset",
    "tool_execute_action",
    "tool_inspect_register",
    "tool_preflight_action",
    "tool_read_record",
    "tool_record_authorization",
    "tool_search_records",
]
