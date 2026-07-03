"""Baseline policies for valid-action (spec section 23).

Five baselines, each takes a (world, state) and returns a sequence of tool calls:
  1. random              — uniformly random tool calls until execute_action
  2. naive_max_approval  — always uses the largest body and meeting method
  3. keyword_heuristic   — picks the body that best matches the request family
  4. oracle_replay       — replays the computed oracle path (deterministic)
  5. search_only         — searches records but never executes (should score 0)

Each baseline is deterministic given a seed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from valid_action.core.generator import generate_dataset  # noqa: E402
from valid_action.core.models import (  # noqa: E402
    ApprovalMethod,
    AuthorizationAttempt,
    Difficulty,
    ExecutionAttempt,
)
from valid_action.core.oracle import (  # noqa: E402
    build_oracle_action,
    replay_oracle_as_attempts,
)
from valid_action.core.render import render_world  # noqa: E402
from valid_action.core.scoring import compute_reward  # noqa: E402
from valid_action.core.serialization import compute_fingerprint  # noqa: E402
from valid_action.core.tools import (  # noqa: E402
    create_equity_grant,
    create_material_contract,
    create_related_party_transaction,
    create_security_issuance,
    create_subsidiary_financing,
    create_token_treasury_transaction,
    tool_execute_action,
    tool_record_authorization,
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


_CREATE_DISPATCH = {
    "equity_grant": create_equity_grant,
    "material_contract": create_material_contract,
    "related_party_transaction": create_related_party_transaction,
    "security_issuance": create_security_issuance,
    "subsidiary_financing": create_subsidiary_financing,
    "token_treasury_transaction": create_token_treasury_transaction,
}


def _create_call(world, state):
    fn = _CREATE_DISPATCH[world.action_request.action_type.value]
    req = world.action_request
    common = dict(entity_id=req.entity_id, state=state, task={})
    if world.action_request.action_type.value == "equity_grant":
        return fn(
            recipient_person_id=req.recipient_person_id,
            award_type=req.award_type,
            units=req.units,
            vesting_months=req.vesting_months,
            cliff_months=req.cliff_months,
            effective_date=req.action_date.isoformat(),
            **common,
        )
    if world.action_request.action_type.value == "material_contract":
        return fn(
            counterparty_id=req.counterparty_id,
            contract_category=req.contract_category,
            total_commitment=str(req.total_commitment),
            term_months=req.term_months,
            includes_exclusivity=req.includes_exclusivity,
            includes_data_processing=req.includes_data_processing,
            effective_date=req.action_date.isoformat(),
            **common,
        )
    if world.action_request.action_type.value == "related_party_transaction":
        return fn(
            counterparty_id=req.counterparty_id,
            related_person_id=req.related_person_id,
            transaction_category=req.transaction_category,
            total_value=str(req.total_value),
            effective_date=req.action_date.isoformat(),
            **common,
        )
    if world.action_request.action_type.value == "security_issuance":
        return fn(
            purchaser_id=req.purchaser_id,
            class_id=req.class_id,
            units=req.units,
            price_per_unit=str(req.price_per_unit),
            financing_round=req.financing_round,
            effective_date=req.action_date.isoformat(),
            **common,
        )
    if world.action_request.action_type.value == "subsidiary_financing":
        return fn(
            subsidiary_id=req.subsidiary_id,
            lender_id=req.lender_id,
            principal=str(req.principal),
            maturity_months=req.maturity_months,
            effective_date=req.action_date.isoformat(),
            parent_guarantee=req.parent_guarantee,
            **common,
        )
    if world.action_request.action_type.value == "token_treasury_transaction":
        return fn(
            counterparty_id=req.counterparty_id,
            token_units=str(req.token_units),
            price_per_token=str(req.price_per_token),
            lockup_months=req.lockup_months,
            effective_date=req.action_date.isoformat(),
            related_person_id=req.related_person_id,
            **common,
        )
    raise ValueError(f"unknown action family: {world.action_request.action_type}")


def oracle_replay(world, state, seed=0):
    """Deterministic baseline: replays oracle's authorization path."""
    render_world(world, seed=world.seed)
    _create_call(world, state)
    if state.get("created_action") is None:
        return  # create failed
    oracle = world.oracle_solution
    action = state["created_action"]
    for step in oracle.authorization_steps:
        tool_record_authorization(
            action_id=action.action_id,
            authorizer_id=step.authorizer_id,
            method=step.method.value,
            participant_ids=list(step.participant_ids),
            recused_ids=list(step.recused_ids),
            disclosure_record_ids=list(step.disclosure_record_ids),
            task={},
            state=state,
        )
    if oracle.signatory_person_ids:
        tool_execute_action(
            action_id=action.action_id,
            signatory_person_id=oracle.signatory_person_ids[0],
            task={},
            state=state,
        )


def naive_max_approval(world, state, seed=0):
    """Always uses the largest body + meeting method."""
    render_world(world, seed=world.seed)
    _create_call(world, state)
    if state.get("created_action") is None:
        return
    # Pick the body with the most roles
    body = max(world.bodies, key=lambda b: sum(1 for r in world.roles if r.body_id == b.body_id))
    eligible = [r.role_id for r in world.roles if r.body_id == body.body_id]
    action = state["created_action"]
    tool_record_authorization(
        action_id=action.action_id,
        authorizer_id=body.body_id,
        method=ApprovalMethod.MEETING.value,
        participant_ids=eligible,
        recused_ids=[],
        disclosure_record_ids=[],
        task={},
        state=state,
    )
    # Sign with the first officer
    officer = next((r for r in world.roles if r.role_type == "officer"), None)
    if officer:
        tool_execute_action(
            action_id=action.action_id,
            signatory_person_id=officer.person_id,
            task={},
            state=state,
        )


def keyword_heuristic(world, state, seed=0):
    """Picks the body whose name contains a keyword from the request family."""
    render_world(world, seed=world.seed)
    _create_call(world, state)
    if state.get("created_action") is None:
        return
    family = world.action_request.action_type.value
    keywords = {
        "material_contract": ["board", "committee"],
        "equity_grant": ["comp", "committee"],
        "security_issuance": ["board", "stockholder"],
        "related_party_transaction": ["board"],
        "subsidiary_financing": ["board"],
        "token_treasury_transaction": ["board"],
    }.get(family, ["board"])
    body = None
    for kw in keywords:
        for b in world.bodies:
            if kw in b.body_id:
                body = b
                break
        if body:
            break
    body = body or world.bodies[0]
    eligible = [r.role_id for r in world.roles if r.body_id == body.body_id]
    action = state["created_action"]
    tool_record_authorization(
        action_id=action.action_id,
        authorizer_id=body.body_id,
        method=ApprovalMethod.MEETING.value,
        participant_ids=eligible,
        recused_ids=[],
        disclosure_record_ids=[],
        task={},
        state=state,
    )
    officer = next((r for r in world.roles if r.role_type == "officer"), None)
    if officer:
        tool_execute_action(
            action_id=action.action_id,
            signatory_person_id=officer.person_id,
            task={},
            state=state,
        )


def random_policy(world, state, seed=0):
    """Random tool selection."""
    rng = random.Random(seed + world.seed)
    render_world(world, seed=world.seed)
    _create_call(world, state)
    if state.get("created_action") is None:
        return
    action = state["created_action"]
    bodies = list(world.bodies)
    for _ in range(rng.randint(0, 3)):
        body = rng.choice(bodies)
        eligible = [r.role_id for r in world.roles if r.body_id == body.body_id]
        tool_record_authorization(
            action_id=action.action_id,
            authorizer_id=body.body_id,
            method=rng.choice([m.value for m in ApprovalMethod]),
            participant_ids=eligible,
            recused_ids=[],
            disclosure_record_ids=[],
            task={},
            state=state,
        )
    officer = next((r for r in world.roles if r.role_type == "officer"), None)
    if officer and rng.random() < 0.7:
        tool_execute_action(
            action_id=action.action_id,
            signatory_person_id=officer.person_id,
            task={},
            state=state,
        )


def search_only(world, state, seed=0):
    """Searches but never creates or executes; should score 0."""
    from valid_action.core.tools import tool_search_records, tool_inspect_register
    render_world(world, seed=world.seed)
    for q in ["board", "committee", "quorum", "approval", "vote", "conflict"]:
        tool_search_records(query=q, entity_id=None, record_type=None, task={}, state=state)
    tool_inspect_register(
        entity_id=world.action_request.entity_id,
        register_type="directors",
        as_of_date=world.action_request.action_date.isoformat(),
        task={},
        state=state,
    )


BASELINES = {
    "random": random_policy,
    "naive_max_approval": naive_max_approval,
    "keyword_heuristic": keyword_heuristic,
    "oracle_replay": oracle_replay,
    "search_only": search_only,
}


def run_baseline(baseline_name: str, num_examples: int, seed: int, difficulty: Difficulty):
    worlds = generate_dataset(split="eval", difficulty=difficulty, seed=seed, num_examples=num_examples)
    fn = BASELINES[baseline_name]
    rewards = []
    valid_count = 0
    for world in worlds:
        state = _build_state(world)
        fn(world, state, seed=seed)
        ex = state.get("execution_attempt") or ExecutionAttempt(
            execution_id="none",
            action_id="none",
            signatory_person_id="unknown",
            timestamp_step=0,
            valid=False,
            defect_codes=[],
        )
        breakdown, _ = compute_reward(
            world,
            state.get("created_action"),
            state.get("authorization_attempts", []),
            ex,
        )
        rewards.append(breakdown.reward)
        if breakdown.valid:
            valid_count += 1
    return {
        "baseline": baseline_name,
        "difficulty": difficulty.value,
        "num_examples": num_examples,
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "valid_rate": valid_count / max(1, len(rewards)),
        "max_reward": max(rewards) if rewards else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    parser.add_argument("--baselines", nargs="+", default=list(BASELINES.keys()))
    args = parser.parse_args()
    difficulty = Difficulty(args.difficulty)
    results = []
    for name in args.baselines:
        r = run_baseline(name, args.examples, args.seed, difficulty)
        results.append(r)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
