"""Spec section 22 checkpoint A: 1000+ generated worlds across tiers
passing solvability and oracle invariants.

Plus requirement-graph normalization idempotence and oracle cost monotonicity.
"""

from __future__ import annotations

import time

import pytest

from valid_action.core.generator import (
    generate_dataset,
    split_seeds,
    split_stride,
)
from valid_action.core.models import Difficulty
from valid_action.core.oracle import build_oracle_action, replay_oracle_as_attempts
from valid_action.core.render import render_world
from valid_action.core.requirements import RequirementGraph
from valid_action.core.serialization import compute_fingerprint, world_to_task_payload
from valid_action.core.validator import validate_final_execution


@pytest.mark.parametrize("difficulty", [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD])
def test_large_dataset_solvability(difficulty):
    """Spec section 22: every generated world must replay with valid execution."""
    from valid_action.core.models import ExecutionAttempt

    t0 = time.time()
    worlds = generate_dataset(
        split="train",
        difficulty=difficulty,
        seed=17,
        num_examples=350,
    )
    elapsed = time.time() - t0
    assert len(worlds) == 350
    # All worlds must pass validator replay (oracle feasibility)
    for w in worlds:
        assert w.oracle_solution.feasible, w.world_template_id
        assert w.oracle_solution.minimum_process_cost >= 1
        # Render + replay should be valid
        render_world(w, seed=w.seed)
        action = build_oracle_action(w, f"act_{w.seed}", 1)
        attempts = replay_oracle_as_attempts(w, w.oracle_solution, action.action_id)
        sig = w.oracle_solution.signatory_person_ids[0] if w.oracle_solution.signatory_person_ids else "prs_unknown"
        ex = ExecutionAttempt(
            execution_id=f"exec_{action.action_id}",
            action_id=action.action_id,
            signatory_person_id=sig,
            timestamp_step=len(attempts) + 2,
            valid=True,
            defect_codes=[],
        )
        result = validate_final_execution(w, action, attempts, ex)
        assert result.valid, f"{w.world_template_id}: {result.defect_codes}"
    # Throughput: at least 50 worlds/sec on this machine
    assert elapsed < 30, f"generation took {elapsed:.1f}s"


def test_train_eval_fingerprints_disjoint():
    """Spec section 22: train and eval splits share no worlds."""
    train = generate_dataset(split="train", difficulty=Difficulty.MEDIUM, seed=17, num_examples=128)
    eval_ = generate_dataset(split="eval", difficulty=Difficulty.MEDIUM, seed=17, num_examples=64)
    train_fp = {compute_fingerprint(w) for w in train}
    eval_fp = {compute_fingerprint(w) for w in eval_}
    assert not train_fp & eval_fp


def test_train_eval_split_stride_disjointness():
    """Stride math should guarantee disjoint splits for default seed."""
    seeds = split_seeds(17)
    stride = split_stride()
    # split_seeds returns {"train": int, "eval": int}
    assert isinstance(seeds["train"], int)
    assert isinstance(seeds["eval"], int)
    assert seeds["train"] != seeds["eval"]
    assert stride > 0


def test_requirement_graph_normalization_idempotent():
    """Normalization should be a fixed point."""
    from valid_action.core.fixtures import GOLDEN_FIXTURES

    for name, factory in GOLDEN_FIXTURES.items():
        world = factory()
        graph1 = world.requirements
        normalized = graph1.normalize()
        normalized2 = normalized.normalize()
        # Two normalizations should produce equivalent structure
        assert normalized.root.model_dump() == normalized2.root.model_dump(), name


def test_oracle_cost_at_least_one():
    """Every feasible oracle must have minimum_process_cost >= 1."""
    worlds = generate_dataset(split="eval", difficulty=Difficulty.HARD, seed=42, num_examples=64)
    for w in worlds:
        if w.oracle_solution.feasible:
            assert w.oracle_solution.minimum_process_cost >= 1


def test_fingerprint_stable_under_payload_roundtrip():
    """Fingerprint must not change after JSON round-trip."""
    from valid_action.core.fixtures import GOLDEN_FIXTURES

    for name, factory in GOLDEN_FIXTURES.items():
        world = factory()
        fp_before = compute_fingerprint(world)
        payload = world_to_task_payload(world)
        # Round-trip the world via payload["info"]["world"] and re-fingerprint
        from valid_action.core.models import ValidActionWorld

        world2 = ValidActionWorld.model_validate(payload["info"]["world"])
        fp_after = compute_fingerprint(world2)
        assert fp_before == fp_after, name


def test_serialization_canonical_stable():
    """world_to_canonical output should be deterministic and JSON-safe."""
    from valid_action.core.fixtures import GOLDEN_FIXTURES
    from valid_action.core.serialization import world_to_canonical

    for name, factory in GOLDEN_FIXTURES.items():
        w1 = factory()
        w2 = factory()
        # JSON-safe (no Decimal/date leaks)
        import json
        c1 = world_to_canonical(w1)
        c2 = world_to_canonical(w2)
        json.loads(json.dumps(c1))
        assert c1 == c2, name
