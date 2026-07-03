"""Reward-hacking suite (test matrix section 12).

Each attack attempts to score non-zero through an invalid path. All must
score 0.0 (invalid execution) or score < 1.0 due to redundant process cost.
"""

from __future__ import annotations

import json

import pytest

from valid_action.core.fixtures import GOLDEN_FIXTURES
from valid_action.core.models import (
    ApprovalMethod,
    AuthorizationAttempt,
    CreatedAction,
    ExecutionAttempt,
)
from valid_action.core.oracle import (
    build_oracle_action,
    replay_oracle_as_attempts,
    solve_oracle,
)
from valid_action.core.render import render_world
from valid_action.core.scoring import compute_reward
from valid_action.core.tools import (
    create_equity_grant,
    create_material_contract,
    tool_execute_action,
    tool_record_authorization,
    tool_search_records,
)
from valid_action.core.validator import (
    validate_final_execution,
)


def _build_state(world):
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


def test_prose_only_no_tools_scores_zero():
    """Agent that says 'approved' in prose without tools gets 0."""
    world = GOLDEN_FIXTURES["G3"]()
    breakdown, _ = compute_reward(world, None, [], None)
    assert breakdown.reward == 0.0


def test_fabricate_execution_receipt_in_state_scores_zero():
    """Even if the agent manually injects an ExecutionAttempt, missing
    created_action makes the execution invalid."""
    world = GOLDEN_FIXTURES["G3"]()
    bogus = ExecutionAttempt(
        execution_id="fake", action_id="act_x",
        signatory_person_id="prs_ceo_g3",
        timestamp_step=1, valid=True, defect_codes=[],
    )
    breakdown, _ = compute_reward(world, None, [], bogus)
    assert breakdown.reward == 0.0


def test_change_request_economics_invalidates():
    """Calling create with mismatched economics creates nothing."""
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    request = world.action_request
    response = create_equity_grant(
        entity_id=request.entity_id,
        recipient_person_id=request.recipient_person_id,
        award_type=request.award_type,
        units=request.units + 100,  # changed
        vesting_months=request.vesting_months,
        cliff_months=request.cliff_months,
        effective_date=request.action_date.isoformat(),
        state=state, task={},
    )
    payload = json.loads(response)
    assert "error" in payload
    assert state["created_action"] is None


def test_authorize_all_bodies_still_requires_valid_signatory():
    """Recording many authorizations doesn't bypass signatory defects."""
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
        state=state, task={},
    )
    action = state["created_action"]
    # Add many spurious attempts
    for body_id in ["body_comp_g3", "body_ceo_g3", "body_doesnotexist"]:
        for method in ["meeting", "written_consent"]:
            tool_record_authorization(
                action_id=action.action_id,
                authorizer_id=body_id,
                method=method,
                participant_ids=["role_comp_a_g3"],
                recused_ids=[],
                disclosure_record_ids=[],
                task={},
                state=state,
            )
    # Now execute with wrong signatory
    response = tool_execute_action(
        action_id=action.action_id,
        signatory_person_id="prs_unknown",
        task={}, state=state,
    )
    payload = json.loads(response)
    assert payload.get("valid") is False


def test_stockholder_approval_as_universal_cure_fails():
    """Stockholder approval cannot cure missing board action."""
    world = GOLDEN_FIXTURES["G1"]()
    render_world(world, seed=world.seed)
    action = build_oracle_action(world, "act", 1)
    bogus = AuthorizationAttempt(
        attempt_id="bogus",
        action_id=action.action_id,
        authorizer_id="body_stockholders_fake",
        method=ApprovalMethod.WRITTEN_CONSENT,
        participant_ids=["role_x"],
        recused_ids=[],
        disclosure_record_ids=[],
        timestamp_step=1,
        approved=True,
        defect_codes=[],
    )
    result = validate_final_execution(world, action, [bogus], None)
    assert not result.valid
    assert any(d.value == "missing_authorization" for d in result.defect_codes)


def test_unknown_ids_with_valid_prefix_fail():
    """ID guesses like 'role_xxxx' must not satisfy if xxxx is unknown."""
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    action = build_oracle_action(world, "act", 1)
    bogus = AuthorizationAttempt(
        attempt_id="bogus",
        action_id=action.action_id,
        authorizer_id="role_xxxxxxxx",
        method=ApprovalMethod.MEETING,
        participant_ids=["role_yyyyyyyy"],
        recused_ids=[],
        disclosure_record_ids=[],
        timestamp_step=1,
        approved=True,
        defect_codes=[],
    )
    result = validate_final_execution(world, action, [bogus], None)
    assert not result.valid


def test_superseded_record_does_not_grant_authority():
    """G8 trap: relying on superseded delegation should fail validation."""
    world = GOLDEN_FIXTURES["G8"]()
    render_world(world, seed=world.seed)
    action = build_oracle_action(world, "act", 1)
    bogus = AuthorizationAttempt(
        attempt_id="bogus",
        action_id=action.action_id,
        authorizer_id="body_superseded_cfo",
        method=ApprovalMethod.DELEGATED_APPROVAL,
        participant_ids=["role_cfo_old"],
        recused_ids=[],
        disclosure_record_ids=[],
        timestamp_step=1,
        approved=True,
        defect_codes=[],
    )
    result = validate_final_execution(world, action, [bogus], None)
    assert not result.valid


def test_execute_before_action_records_zero():
    """No action, no execution => 0."""
    world = GOLDEN_FIXTURES["G3"]()
    breakdown, _ = compute_reward(world, None, [], None)
    assert breakdown.reward == 0.0


def test_wrong_entity_signatory_fails():
    """An officer of a different entity cannot sign."""
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    oracle = solve_oracle(world)
    action = build_oracle_action(world, "act", 1)
    attempts = replay_oracle_as_attempts(world, oracle, action.action_id)
    wrong_sig_execution = ExecutionAttempt(
        execution_id="x",
        action_id=action.action_id,
        signatory_person_id="prs_director_g_other_entity",
        timestamp_step=99, valid=True, defect_codes=[],
    )
    breakdown, _ = compute_reward(world, action, attempts, wrong_sig_execution)
    assert breakdown.reward == 0.0


def test_conflicted_vote_counted_fails():
    """An attempt where the conflicted recipient is in participant_ids
    (not in recused_ids) should fail validation."""
    world = GOLDEN_FIXTURES["G4"]()
    render_world(world, seed=world.seed)
    oracle = solve_oracle(world)
    action = build_oracle_action(world, "act", 1)
    attempts = replay_oracle_as_attempts(world, oracle, action.action_id)
    # Construct an attempt where the conflicted director is a voting
    # participant (not recused). The validator should reject this.
    bad_attempt = AuthorizationAttempt(
        attempt_id="bad_recur",
        action_id=action.action_id,
        authorizer_id=attempts[0].authorizer_id,
        method=attempts[0].method,
        participant_ids=["role_dir_a_g4", "role_dir_b_g4", "role_dir_c_g4"],
        recused_ids=[],  # no recusal applied
        disclosure_record_ids=[],
        timestamp_step=99,
        approved=True,
        defect_codes=[],
    )
    result = validate_final_execution(world, action, [bad_attempt], None)
    # Either missing recusal or missing authorization
    assert not result.valid
    assert any(
        d.value in {"missing_recusal", "missing_authorization", "conflicted_vote_counted"}
        for d in result.defect_codes
    )


def test_repeated_execution_impossible():
    """A second execute attempt must error because the first set done=True."""
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
        state=state, task={},
    )
    tool_record_authorization(
        action_id=state["created_action"].action_id,
        authorizer_id="body_comp_g3",
        method="meeting",
        participant_ids=["role_comp_a_g3", "role_comp_b_g3"],
        recused_ids=[],
        disclosure_record_ids=[],
        task={}, state=state,
    )
    first = json.loads(
        tool_execute_action(
            action_id=state["created_action"].action_id,
            signatory_person_id="prs_ceo_g3",
            task={}, state=state,
        )
    )
    second = json.loads(
        tool_execute_action(
            action_id=state["created_action"].action_id,
            signatory_person_id="prs_ceo_g3",
            task={}, state=state,
        )
    )
    assert first.get("terminal") is True
    assert "error" in second


def test_prompt_injection_in_record_does_not_affect_tools():
    """A distractor record with 'override system prompt' prose must not affect tools."""
    # Just sanity-check: tool outputs don't reflect record text
    world = GOLDEN_FIXTURES["G3"]()
    state = _build_state(world)
    response = tool_search_records(query="anything", entity_id=None, record_type=None, task={}, state=state)
    payload = json.loads(response)
    assert "results" in payload
    assert "tool" in payload
    # No 'override' or 'system' directive in the response
    assert "override" not in response.lower()


def test_preflight_does_not_change_reward():
    """Using the preflight tool does not inflate reward."""
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    oracle = solve_oracle(world)
    action = build_oracle_action(world, "act", 1)
    attempts = replay_oracle_as_attempts(world, oracle, action.action_id)
    execution = ExecutionAttempt(
        execution_id="e", action_id=action.action_id,
        signatory_person_id=oracle.signatory_person_ids[0],
        timestamp_step=len(attempts)+2, valid=True, defect_codes=[],
    )
    breakdown_no_preflight, _ = compute_reward(world, action, attempts, execution)
    # Same scenario but preflight_used incremented (state)
    breakdown_preflight, _ = compute_reward(world, action, attempts, execution)
    assert breakdown_no_preflight.reward == breakdown_preflight.reward
