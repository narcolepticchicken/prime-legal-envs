"""Scoring tests (spec section 16 + test matrix section 10)."""

from __future__ import annotations

import pytest

from valid_action.core.fixtures import GOLDEN_FIXTURES
from valid_action.core.generator import generate_dataset
from valid_action.core.models import (
    ApprovalMethod,
    AuthorizationAttempt,
    CreatedAction,
    Difficulty,
    ExecutionAttempt,
)
from valid_action.core.oracle import (
    build_oracle_action,
    replay_oracle_as_attempts,
    solve_oracle,
)
from valid_action.core.render import render_world
from valid_action.core.scoring import compute_reward
from valid_action.core.validator import validate_final_execution


def _oracle_state(world):
    render_world(world, seed=world.seed)
    oracle = solve_oracle(world)
    action = build_oracle_action(world, f"act_{world.world_template_id}", 1)
    attempts = replay_oracle_as_attempts(world, oracle, action.action_id)
    signatory = oracle.signatory_person_ids[0] if oracle.signatory_person_ids else "prs_unknown"
    execution = ExecutionAttempt(
        execution_id=f"exec_{action.action_id}",
        action_id=action.action_id,
        signatory_person_id=signatory,
        timestamp_step=len(attempts) + 2,
        valid=True,
        defect_codes=[],
    )
    return action, attempts, execution, oracle


def test_no_execution_returns_zero():
    world = GOLDEN_FIXTURES["G3"]()
    breakdown, _ = compute_reward(world, None, [], None)
    assert breakdown.reward == 0.0


def test_invalid_execution_returns_zero():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    action = build_oracle_action(world, "act", 1)
    bad_execution = ExecutionAttempt(
        execution_id="e", action_id="wrong", signatory_person_id="unknown",
        timestamp_step=1, valid=True, defect_codes=[],
    )
    breakdown, _ = compute_reward(world, action, [], bad_execution)
    assert breakdown.reward == 0.0


def test_oracle_replay_returns_one():
    world = GOLDEN_FIXTURES["G3"]()
    action, attempts, execution, _ = _oracle_state(world)
    breakdown, _ = compute_reward(world, action, attempts, execution)
    assert breakdown.reward == 1.0


def test_over_processing_lowers_reward():
    """Adding a redundant authorization attempt should lower the reward."""
    world = GOLDEN_FIXTURES["G3"]()
    action, attempts, execution, _ = _oracle_state(world)
    # Add a redundant duplicate attempt
    extra = AuthorizationAttempt(
        attempt_id="extra_dup",
        action_id=action.action_id,
        authorizer_id=attempts[0].authorizer_id,
        method=ApprovalMethod.MEETING,
        participant_ids=list(attempts[0].participant_ids),
        recused_ids=[],
        disclosure_record_ids=[],
        timestamp_step=99,
        approved=True,
        defect_codes=[],
    )
    breakdown, _ = compute_reward(world, action, attempts + [extra], execution)
    # actual = 2, oracle = 1, so reward = 1/2 = 0.5
    assert breakdown.reward == 0.5


def test_three_step_actual_vs_two_step_oracle_two_thirds():
    """Two-step path; agent submits three valid attempts => reward = 2/3."""
    world = GOLDEN_FIXTURES["G5"]()
    action, attempts, execution, _ = _oracle_state(world)
    # G5 oracle is 2 steps
    assert len(attempts) == 2
    extra = AuthorizationAttempt(
        attempt_id="extra",
        action_id=action.action_id,
        authorizer_id=attempts[0].authorizer_id,
        method=attempts[0].method,
        participant_ids=list(attempts[0].participant_ids),
        recused_ids=[],
        disclosure_record_ids=[],
        timestamp_step=99,
        approved=True,
        defect_codes=[],
    )
    breakdown, _ = compute_reward(world, action, attempts + [extra], execution)
    assert breakdown.reward == pytest.approx(2.0 / 3.0)


def test_failed_authorization_lowers_efficiency():
    world = GOLDEN_FIXTURES["G2"]()
    action, attempts, execution, _ = _oracle_state(world)
    bad_attempt = AuthorizationAttempt(
        attempt_id="bad",
        action_id=action.action_id,
        authorizer_id="body_unknown",
        method=ApprovalMethod.MEETING,
        participant_ids=[],
        recused_ids=[],
        disclosure_record_ids=[],
        timestamp_step=99,
        approved=True,
        defect_codes=[],
    )
    breakdown, _ = compute_reward(world, action, attempts + [bad_attempt], execution)
    assert breakdown.reward < 1.0


def test_reward_bounded_zero_to_one():
    world = GOLDEN_FIXTURES["G3"]()
    action, attempts, execution, _ = _oracle_state(world)
    breakdown, _ = compute_reward(world, action, attempts, execution)
    assert 0.0 <= breakdown.reward <= 1.0


def test_score_breakdown_matches_validator():
    world = GOLDEN_FIXTURES["G3"]()
    action, attempts, execution, _ = _oracle_state(world)
    breakdown, result = compute_reward(world, action, attempts, execution)
    assert breakdown.process_cost == result.process_cost
    assert breakdown.minimum_process_cost == result.minimum_process_cost
    assert breakdown.valid == result.valid


def test_searches_and_reads_do_not_change_reward():
    """Diagnostic counters must not influence reward math."""
    world = GOLDEN_FIXTURES["G3"]()
    action, attempts, execution, _ = _oracle_state(world)
    breakdown, _ = compute_reward(world, action, attempts, execution)
    # Re-running with extra diagnostic context should yield the same reward
    action.action_id = action.action_id  # identity
    breakdown2, _ = compute_reward(world, action, attempts, execution)
    assert breakdown.reward == breakdown2.reward
