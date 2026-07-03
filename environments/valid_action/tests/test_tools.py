"""Tool contract tests (spec test matrix section 9)."""

from __future__ import annotations

import json
from datetime import date

import pytest

from valid_action.core.fixtures import GOLDEN_FIXTURES
from valid_action.core.render import render_world
from valid_action.core.tools import (
    create_equity_grant,
    create_material_contract,
    create_security_issuance,
    create_related_party_transaction,
    create_subsidiary_financing,
    create_token_treasury_transaction,
    tool_execute_action,
    tool_inspect_register,
    tool_preflight_action,
    tool_read_record,
    tool_record_authorization,
    tool_search_records,
)
from valid_action.core.models import (
    ActionType,
    ApprovalMethod,
    AuthorizationAttempt,
    CreatedAction,
    DefectCode,
    EquityGrantRequest,
    EquityPlanCapacity,
    ExecutionAttempt,
    MaterialContractRequest,
    Person,
    RecordType,
    ValidActionWorld,
)


def _build_state(world: ValidActionWorld) -> dict:
    return {
        "world": world,
        "created_action": None,
        "authorization_attempts": [],
        "execution_attempt": None,
        "event_log": [],
        "preflight_remaining": 1,
        "preflight_used": 0,
        "documents_read": [],
        "search_queries": [],
        "register_inspections": [],
        "tool_errors": [],
        "score_breakdown": {},
        "validation_result": None,
        "turn_count": 0,
        "max_search_results": 5,
        "runtime": {"max_turns": 14},
    }


def _parse(response: str) -> dict:
    return json.loads(response)


def test_search_records_envelope():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    state = _build_state(world)
    response = tool_search_records(
        query="committee",
        entity_id=world.action_request.entity_id,
        record_type=None,
        task={"info": {"world": world.model_dump(mode="json")}},
        state=state,
    )
    payload = _parse(response)
    assert payload["tool"] == "search_records"
    assert "remaining_turns" in payload
    assert "preflight_checks_remaining" in payload
    assert "action_status" in payload
    assert "terminal" in payload


def test_search_records_empty_query_errors():
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    response = tool_search_records(query="", entity_id=None, record_type=None, task={}, state=state)
    payload = _parse(response)
    assert "error" in payload


def test_read_record_unknown_id_returns_suggestions():
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    response = tool_read_record(record_id="does_not_exist", task={}, state=state)
    payload = _parse(response)
    assert "error" in payload
    assert "suggestions" in payload


def test_inspect_register_unknown_register_type():
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    response = tool_inspect_register(
        entity_id=world.action_request.entity_id,
        register_type="bogus",
        as_of_date="2028-09-15",
        task={},
        state=state,
    )
    payload = _parse(response)
    assert "error" in payload


def test_create_equity_grant_matches_request():
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    request = world.action_request  # EquityGrantRequest
    response = create_equity_grant(
        entity_id=request.entity_id,
        recipient_person_id=request.recipient_person_id,
        award_type=request.award_type,
        units=request.units,
        vesting_months=request.vesting_months,
        cliff_months=request.cliff_months,
        effective_date=request.action_date.isoformat(),
        state=state,
        task={},
    )
    payload = _parse(response)
    assert payload["action_status"] == "draft"
    assert state["created_action"] is not None


def test_create_equity_grant_terms_mismatch():
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    request = world.action_request
    response = create_equity_grant(
        entity_id=request.entity_id,
        recipient_person_id=request.recipient_person_id,
        award_type=request.award_type,
        units=request.units + 1,  # wrong units
        vesting_months=request.vesting_months,
        cliff_months=request.cliff_months,
        effective_date=request.action_date.isoformat(),
        state=state,
        task={},
    )
    payload = _parse(response)
    assert "error" in payload
    assert state["created_action"] is None


def test_second_create_action_rejected():
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    request = world.action_request
    create_equity_grant(
        entity_id=request.entity_id,
        recipient_person_id=request.recipient_person_id,
        award_type=request.award_type,
        units=request.units,
        vesting_months=request.vesting_months,
        cliff_months=request.cliff_months,
        effective_date=request.action_date.isoformat(),
        state=state,
        task={},
    )
    response = create_equity_grant(
        entity_id=request.entity_id,
        recipient_person_id=request.recipient_person_id,
        award_type=request.award_type,
        units=request.units,
        vesting_months=request.vesting_months,
        cliff_months=request.cliff_months,
        effective_date=request.action_date.isoformat(),
        state=state,
        task={},
    )
    payload = _parse(response)
    assert "error" in payload


def test_record_authorization_before_create_action_rejected():
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    response = tool_record_authorization(
        action_id="act_x",
        authorizer_id="body_comp_g3",
        method="meeting",
        participant_ids=["role_comp_a_g3"],
        recused_ids=[],
        disclosure_record_ids=[],
        task={},
        state=state,
    )
    payload = _parse(response)
    assert "error" in payload


def test_preflight_exhausted_returns_error():
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    state["preflight_remaining"] = 0
    response = tool_preflight_action(action_id="anything", task={}, state=state)
    payload = _parse(response)
    assert "error" in payload


def test_execute_without_action_errors():
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    response = tool_execute_action(action_id="act_x", signatory_person_id="x", task={}, state=state)
    payload = _parse(response)
    assert "error" in payload


def test_execute_terminates_rollout():
    """A successful execute should set state['done'] = True."""
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    state = _build_state(world)
    request = world.action_request
    create_equity_grant(
        entity_id=request.entity_id,
        recipient_person_id=request.recipient_person_id,
        award_type=request.award_type,
        units=request.units,
        vesting_months=request.vesting_months,
        cliff_months=request.cliff_months,
        effective_date=request.action_date.isoformat(),
        state=state,
        task={},
    )
    # Add the committee approval
    tool_record_authorization(
        action_id=state["created_action"].action_id,
        authorizer_id="body_comp_g3",
        method="meeting",
        participant_ids=["role_comp_a_g3", "role_comp_b_g3"],
        recused_ids=[],
        disclosure_record_ids=[],
        task={},
        state=state,
    )
    response = tool_execute_action(
        action_id=state["created_action"].action_id,
        signatory_person_id="prs_ceo_g3",
        task={},
        state=state,
    )
    payload = _parse(response)
    assert payload["terminal"] is True
    assert state.get("done") is True
    assert state["execution_attempt"] is not None


def test_create_action_for_other_family_blocked():
    """Task-specific tool visibility: only one create tool exposed per task.
    This is enforced via per-task toolset filtering; here we just verify that
    calling a non-matching create tool against a different request errors."""
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    # G3 is an equity grant; trying to create a material contract should fail
    response = create_material_contract(
        entity_id="ent_unknown",
        counterparty_id="cp_x",
        contract_category="vendor",
        total_commitment="1000000",
        term_months=12,
        effective_date="2028-09-15",
        state=state,
        task={},
    )
    payload = _parse(response)
    assert "error" in payload


def test_create_token_treasury_matches():
    world = GOLDEN_FIXTURES["G6"]()
    render_world(world, seed=world.seed)
    state = _build_state(world)
    request = world.action_request
    response = create_token_treasury_transaction(
        entity_id=request.entity_id,
        counterparty_id=request.counterparty_id,
        token_units=str(request.token_units),
        price_per_token=str(request.price_per_token),
        lockup_months=request.lockup_months,
        effective_date=request.action_date.isoformat(),
        related_person_id=request.related_person_id,
        state=state,
        task={},
    )
    payload = _parse(response)
    assert payload["action_status"] == "draft"


def test_create_subsidiary_financing_matches():
    world = GOLDEN_FIXTURES["G7"]()
    render_world(world, seed=world.seed)
    state = _build_state(world)
    request = world.action_request
    response = create_subsidiary_financing(
        entity_id=request.entity_id,
        subsidiary_id=request.subsidiary_id,
        lender_id=request.lender_id,
        principal=str(request.principal),
        maturity_months=request.maturity_months,
        effective_date=request.action_date.isoformat(),
        parent_guarantee=request.parent_guarantee,
        state=state,
        task={},
    )
    payload = _parse(response)
    assert payload["action_status"] == "draft"


def test_create_security_issuance_matches():
    world = GOLDEN_FIXTURES["G5"]()
    render_world(world, seed=world.seed)
    state = _build_state(world)
    request = world.action_request
    response = create_security_issuance(
        entity_id=request.entity_id,
        purchaser_id=request.purchaser_id,
        class_id=request.class_id,
        units=request.units,
        price_per_unit=str(request.price_per_unit),
        financing_round=request.financing_round,
        effective_date=request.action_date.isoformat(),
        state=state,
        task={},
    )
    payload = _parse(response)
    assert payload["action_status"] == "draft"


def test_create_related_party_matches():
    world = GOLDEN_FIXTURES["G9"]()
    render_world(world, seed=world.seed)
    state = _build_state(world)
    request = world.action_request
    response = create_related_party_transaction(
        entity_id=request.entity_id,
        counterparty_id=request.counterparty_id,
        related_person_id=request.related_person_id,
        transaction_category=request.transaction_category,
        total_value=str(request.total_value),
        effective_date=request.action_date.isoformat(),
        state=state,
        task={},
    )
    payload = _parse(response)
    assert payload["action_status"] == "draft"
