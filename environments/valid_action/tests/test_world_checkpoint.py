"""World-checkpoint tests (spec section 22, checkpoint A).

Pass conditions:
  - all fixed fixtures validate as expected
  - oracle plans replay successfully
  - generator is deterministic
  - 1,000 generated worlds are solvable (this file uses small N for CI)
  - renderer contains every operative requirement
  - split fingerprints do not overlap
"""

from __future__ import annotations

import hashlib

import pytest

from valid_action.core.fixtures import (
    GOLDEN_FIXTURES,
    all_golden_worlds,
)
from valid_action.core.generator import generate_dataset
from valid_action.core.models import (
    AuthorizationAttempt,
    CreatedAction,
    Difficulty,
    ExecutionAttempt,
    ValidActionWorld,
)
from valid_action.core.oracle import (
    build_oracle_action,
    replay_oracle_as_attempts,
    solve_oracle,
)
from valid_action.core.render import render_world
from valid_action.core.serialization import compute_fingerprint
from valid_action.core.validator import validate_final_execution


def _oracle_replay(world: ValidActionWorld) -> tuple[CreatedAction, list[AuthorizationAttempt], ExecutionAttempt]:
    oracle = solve_oracle(world)
    assert oracle.feasible, f"oracle infeasible for {world.world_template_id}"
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
    return action, attempts, execution


@pytest.mark.parametrize("name,builder", list(GOLDEN_FIXTURES.items()))
def test_golden_worlds_build(name: str, builder):
    world = builder()
    assert world.world_template_id == f"{name}_" or world.world_template_id.startswith(name)
    assert world.action_request is not None
    assert world.oracle_solution.feasible


@pytest.mark.parametrize("name,builder", list(GOLDEN_FIXTURES.items()))
def test_golden_oracle_replay_valid(name: str, builder):
    world = builder()
    render_world(world, seed=world.seed)
    action, attempts, execution = _oracle_replay(world)
    result = validate_final_execution(world, action, attempts, execution)
    assert result.valid, f"{name} oracle replay defects: {[d.value for d in result.defect_codes]}"
    assert result.process_cost == world.oracle_solution.minimum_process_cost


@pytest.mark.parametrize("name,builder", list(GOLDEN_FIXTURES.items()))
def test_golden_minimum_cost_matches(name: str, builder):
    world = builder()
    render_world(world, seed=world.seed)
    oracle = solve_oracle(world)
    assert oracle.feasible
    assert oracle.minimum_process_cost == world.oracle_solution.minimum_process_cost


def test_g1_minimum_cost_one():
    world = GOLDEN_FIXTURES["G1"]()
    render_world(world, seed=world.seed)
    oracle = solve_oracle(world)
    assert oracle.minimum_process_cost == 1


def test_g5_minimum_cost_two():
    world = GOLDEN_FIXTURES["G5"]()
    render_world(world, seed=world.seed)
    oracle = solve_oracle(world)
    assert oracle.minimum_process_cost == 2


def test_g7_minimum_cost_three():
    world = GOLDEN_FIXTURES["G7"]()
    render_world(world, seed=world.seed)
    oracle = solve_oracle(world)
    assert oracle.minimum_process_cost == 3


def test_g8_only_uses_active_records():
    """The oracle must not rely on superseded delegation."""
    world = GOLDEN_FIXTURES["G8"]()
    render_world(world, seed=world.seed)
    superseded = [r for r in world.records if r.status == "superseded"]
    assert superseded, "fixture should have a superseded record"
    oracle = solve_oracle(world)
    assert oracle.feasible
    assert oracle.minimum_process_cost == 1


def test_generator_is_deterministic():
    a = generate_dataset(split="train", difficulty=Difficulty.MEDIUM, seed=17, num_examples=8)
    b = generate_dataset(split="train", difficulty=Difficulty.MEDIUM, seed=17, num_examples=8)
    fps_a = [compute_fingerprint(w) for w in a]
    fps_b = [compute_fingerprint(w) for w in b]
    assert fps_a == fps_b


def test_train_eval_fingerprints_disjoint():
    train = generate_dataset(split="train", difficulty=Difficulty.MEDIUM, seed=17, num_examples=16)
    eval_ = generate_dataset(split="eval", difficulty=Difficulty.MEDIUM, seed=17, num_examples=16)
    train_fps = {compute_fingerprint(w) for w in train}
    eval_fps = {compute_fingerprint(w) for w in eval_}
    assert not (train_fps & eval_fps), "train/eval fingerprints overlap"


def test_generator_small_batch_solvable():
    """Smoke check: 16 worlds per difficulty are solvable."""
    for diff in [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]:
        worlds = generate_dataset(split="train", difficulty=diff, seed=17, num_examples=16)
        for w in worlds:
            oracle = solve_oracle(w)
            assert oracle.feasible, f"{w.world_template_id} infeasible"
            assert oracle.minimum_process_cost == w.oracle_solution.minimum_process_cost


def test_renderer_deterministic():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=42)
    text1 = world.records[0].sections[0].text
    render_world(world, seed=42)
    text2 = world.records[0].sections[0].text
    assert text1 == text2


def test_renderer_hides_source_rule_ids_in_visible_records():
    from valid_action.core.serialization import visible_records

    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=42)
    visible = visible_records(world)
    for record in visible:
        for section in record["sections"]:
            assert "source_rule_ids" not in section


def test_canonical_json_stable():
    from valid_action.core.serialization import canonical_json, world_to_canonical

    world = GOLDEN_FIXTURES["G3"]()
    a = world_to_canonical(world)
    b = world_to_canonical(world)
    assert canonical_json(a) == canonical_json(b)


def test_all_six_action_families_appear():
    worlds = generate_dataset(split="train", difficulty=Difficulty.MEDIUM, seed=17, num_examples=128)
    families = {w.action_request.action_type.value for w in worlds}
    # MVP fixtures cover 5 of 6 families; the 6th is covered by expansion tasks.
    expected_families = {
        "material_contract",
        "equity_grant",
        "security_issuance",
        "related_party_transaction",
        "token_treasury_transaction",
        "subsidiary_financing",
    }
    missing = expected_families - families
    assert not missing, f"missing action families: {missing}"
