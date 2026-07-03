"""Prime Intellect Taskset wiring for valid-action.

This module is the thin Prime-facing layer per spec / AGENTS.md:
  - config_type subclass with typed fields
  - load_taskset + load_environment
  - per-rollout setup that rebuilds the typed world on state
  - search/read/register + auth + per-action create tools
  - one reward, several metrics, one stop

All rule logic lives in the core package.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

import verifiers as vf

from .core.generator import (
    GenerationError,
    generate_dataset,
    split_seeds,
    split_stride,
)
from .core.models import (
    ActionType,
    AuthorizationAttempt,
    CreatedAction,
    Difficulty,
    ExecutionAttempt,
    ValidActionWorld,
)
from .core.scoring import compute_reward
from .core.serialization import compute_fingerprint, world_to_task_payload
from .core.tools import (
    ALL_CREATE_TOOL_FUNCTIONS,
    ALL_CREATE_TOOLS,
    auth_toolset,
    search_read_toolset,
    tool_execute_action,
    tool_inspect_register,
    tool_preflight_action,
    tool_read_record,
    tool_record_authorization,
    tool_search_records,
)


SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "config" / "system_prompt.txt"


def _load_system_prompt() -> list[dict[str, str]]:
    if SYSTEM_PROMPT_PATH.exists():
        text = SYSTEM_PROMPT_PATH.read_text().strip()
        return [{"role": "system", "content": text}]
    return [
        {
            "role": "system",
            "content": (
                "You are corporate counsel operating inside a fictional "
                "corporate-governance environment. Complete the business request by "
                "creating, authorizing, and executing one corporate action through "
                "the available tools. Explanations do not change the world and "
                "receive no credit. Execution is irreversible and ends the rollout."
            ),
        }
    ]


# ---------- Config ----------


class ValidActionTasksetConfig(vf.TasksetConfig):
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    seed: int = 17
    num_train: int = 64
    num_eval: int = 32
    preflight_budget: int = 1
    max_search_results: int = 5
    include_distractors: bool = True
    system_prompt: object | None = None
    max_turns_override: int | None = None


# ---------- Source loader ----------


def _make_source_loader(
    *,
    split: str,
    difficulty: Difficulty,
    seed: int,
    num_examples: int,
    preflight_budget: int,
    max_search_results: int,
    max_turns_override: int | None,
):
    def source() -> list[dict[str, Any]]:
        worlds = generate_dataset(
            split=split,
            difficulty=difficulty,
            seed=seed,
            num_examples=num_examples,
        )
        rows: list[dict[str, Any]] = []
        for idx, world in enumerate(worlds):
            row = _world_to_row(
                world=world,
                split=split,
                index=idx,
                preflight_budget=preflight_budget,
                max_search_results=max_search_results,
                max_turns_override=max_turns_override,
            )
            rows.append(row)
        return rows

    return source


def _world_to_row(
    *,
    world: ValidActionWorld,
    split: str,
    index: int,
    preflight_budget: int,
    max_search_results: int,
    max_turns_override: int | None,
) -> dict[str, Any]:
    payload = world_to_task_payload(world)
    action_family = world.action_request.action_type.value
    create_tool_name = f"create_{action_family}"
    prompt_messages = _build_prompt(world)
    max_turns = max_turns_override if max_turns_override is not None else _max_turns_for(world.difficulty)
    return {
        "example_id": index,
        "prompt": prompt_messages,
        "info": {
            **payload["info"],
            "split": split,
            "index": index,
            "preflight_budget": preflight_budget,
            "max_search_results": max_search_results,
            "world_template_id": world.world_template_id,
            "world_seed": world.seed,
            "fingerprint": compute_fingerprint(world),
            "oracle_minimum_process_cost": world.oracle_solution.minimum_process_cost,
        },
        "max_turns": max_turns,
        "toolsets": {
            "action_create": {"show": [create_tool_name]},
        },
    }


def _build_prompt(world: ValidActionWorld) -> list[dict[str, Any]]:
    request = world.action_request
    summary = _request_summary(request)
    legal_name = world.entities[0].legal_name.rstrip(".")
    prompt_text = (
        f"You are corporate counsel to {legal_name}. "
        f"The company asks you to {summary} effective {request.action_date.isoformat()}. "
        f"Complete the corporate action without changing the economics. "
        f"The action counts only if executed through `execute_action`."
    )
    return [{"role": "user", "content": prompt_text}]


def _request_summary(request) -> str:
    if isinstance(request, type(request).__class__) and False:  # pragma: no cover
        pass
    if hasattr(request, "units") and hasattr(request, "award_type"):
        return f"grant {request.units} {request.award_type} units to {request.recipient_person_id}"
    if hasattr(request, "units") and hasattr(request, "class_id"):
        return (
            f"issue {request.units} units of {request.class_id} to {request.purchaser_id}"
        )
    if hasattr(request, "total_commitment"):
        return (
            f"execute a {request.contract_category} contract with counterparty "
            f"{request.counterparty_id} totaling {request.total_commitment}"
        )
    if hasattr(request, "total_value"):
        return (
            f"execute a related-party {request.transaction_category} of {request.total_value}"
        )
    if hasattr(request, "token_units"):
        return (
            f"sell {request.token_units} treasury tokens at {request.price_per_token} "
            f"with a {request.lockup_months}-month lockup"
        )
    if hasattr(request, "principal"):
        return (
            f"borrow {request.principal} for subsidiary {request.subsidiary_id} "
            f"with maturity {request.maturity_months} months"
        )
    return request.business_purpose


def _max_turns_for(difficulty: Difficulty) -> int:
    return {"easy": 10, "medium": 14, "hard": 18}[difficulty.value]


# ---------- Setup: rebuild typed world on rollout state ----------


async def _setup_rollout(
    task: Mapping[str, Any],
    state: vf.State,
) -> None:
    info = task.get("info", {})
    world_payload = info.get("world")
    if not isinstance(world_payload, Mapping):
        raise ValueError("task.info.world missing; cannot reconstruct world")
    world = ValidActionWorld.model_validate(world_payload)
    # Store the canonical dict so state stays JSON-serializable for transport
    state["world"] = world.model_dump(mode="json")
    state["created_action"] = None
    state["authorization_attempts"] = []
    state["execution_attempt"] = None
    state["event_log"] = []
    state["preflight_remaining"] = int(info.get("preflight_budget", 1))
    state["preflight_used"] = 0
    state["documents_read"] = []
    state["search_queries"] = []
    state["register_inspections"] = []
    state["tool_errors"] = []
    state["score_breakdown"] = {}
    state["validation_result"] = None
    state["turn_count"] = 0
    state["max_search_results"] = int(info.get("max_search_results", 5))
    runtime = state.setdefault("runtime", {})
    runtime["max_turns"] = int(task.get("max_turns", 14))


# ---------- Reward + metrics ----------


@vf.reward(weight=1.0)
async def final_validity_efficiency(task: Mapping[str, Any], state: vf.State) -> float:
    world = state.get("world")
    if world is None:
        return 0.0
    execution: ExecutionAttempt | None = state.get("execution_attempt")
    if execution is None or not execution.valid:
        return 0.0
    breakdown, _ = compute_reward(
        world=world,
        created_action=state.get("created_action"),
        authorization_attempts=list(state.get("authorization_attempts", [])),
        execution_attempt=execution,
    )
    return float(breakdown.reward)


@vf.metric
async def execution_attempted(task: Mapping[str, Any], state: vf.State) -> float:
    return float(state.get("execution_attempt") is not None)


@vf.metric
async def execution_valid(task: Mapping[str, Any], state: vf.State) -> float:
    execution: ExecutionAttempt | None = state.get("execution_attempt")
    return float(execution is not None and execution.valid)


@vf.metric
async def process_cost(task: Mapping[str, Any], state: vf.State) -> float:
    breakdown = state.get("score_breakdown") or {}
    return float(breakdown.get("process_cost", 0))


@vf.metric
async def process_efficiency_ratio(task: Mapping[str, Any], state: vf.State) -> float:
    breakdown = state.get("score_breakdown") or {}
    cost = breakdown.get("process_cost", 0)
    oracle = breakdown.get("minimum_process_cost", 0)
    if not cost or not oracle:
        return 0.0
    return float(min(1.0, oracle / cost))


@vf.metric
async def turns_used(task: Mapping[str, Any], state: vf.State) -> float:
    return float(state.get("turn_count", 0))


@vf.metric
async def tool_error_count(task: Mapping[str, Any], state: vf.State) -> float:
    return float(len(state.get("tool_errors", [])))


@vf.metric
async def preflight_used(task: Mapping[str, Any], state: vf.State) -> float:
    return float(state.get("preflight_used", 0))


@vf.metric
async def validation_defects(task: Mapping[str, Any], state: vf.State) -> float:
    result = state.get("validation_result") or {}
    return float(len(result.get("defect_codes", [])))


@vf.stop
async def executed_or_truncated(task: Mapping[str, Any], state: vf.State) -> bool:
    if state.get("execution_attempt") is not None:
        return True
    turn = int(state.get("turn_count", 0))
    max_turns = int((state.get("runtime") or {}).get("max_turns", 14))
    return turn >= max_turns


# ---------- Loader entrypoints ----------


def _build_toolsets() -> list[vf.Toolset]:
    """Create the tool surface. Search/read/register + auth are always visible.
    Per-task visibility gates the matching create_<family> tool through
    task['toolsets']['action_create']['show']."""
    search_tools = search_read_toolset()
    auth_tools = auth_toolset()
    create_tools = list(ALL_CREATE_TOOL_FUNCTIONS.values())
    return [
        vf.Toolset(tools=search_tools),
        vf.Toolset(tools=auth_tools),
        vf.Toolset(
            tools=create_tools,
            hide=list(ALL_CREATE_TOOLS),
        ),
    ]


def load_taskset(config: vf.TasksetConfig | Mapping[str, object] | None = None) -> vf.Taskset:
    cfg = ValidActionTasksetConfig(config)
    seeds = split_seeds(cfg.seed)
    train_seed = seeds["train"]
    eval_seed = seeds["eval"]
    difficulty = Difficulty(cfg.difficulty)
    system_prompt = cfg.system_prompt or _load_system_prompt()
    train_source = _make_source_loader(
        split="train",
        difficulty=difficulty,
        seed=train_seed,
        num_examples=cfg.num_train,
        preflight_budget=cfg.preflight_budget,
        max_search_results=cfg.max_search_results,
        max_turns_override=cfg.max_turns_override,
    )
    eval_source = _make_source_loader(
        split="eval",
        difficulty=difficulty,
        seed=eval_seed,
        num_examples=cfg.num_eval,
        preflight_budget=cfg.preflight_budget,
        max_search_results=cfg.max_search_results,
        max_turns_override=cfg.max_turns_override,
    )
    return vf.Taskset(
        source=train_source,
        eval_source=eval_source,
        taskset_id="valid-action",
        system_prompt=system_prompt,
        toolsets=_build_toolsets(),
        setups=[_setup_rollout],
        rewards=[final_validity_efficiency],
        metrics=[
            execution_attempted,
            execution_valid,
            process_cost,
            process_efficiency_ratio,
            turns_used,
            tool_error_count,
            preflight_used,
            validation_defects,
        ],
        stops=[executed_or_truncated],
        config=cfg,
    )


def load_environment(config: vf.EnvConfig | Mapping[str, object] | None = None) -> vf.Env:
    env_cfg = vf.EnvConfig(config)
    taskset = load_taskset(config=env_cfg.taskset)
    return vf.Env(taskset=taskset)


__all__ = [
    "ValidActionTasksetConfig",
    "load_environment",
    "load_taskset",
]
