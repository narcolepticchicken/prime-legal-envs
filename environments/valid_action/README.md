# valid-action

A Prime Intellect `verifiers.v1` environment that measures whether a model can complete one corporate-governance action end-to-end through state-changing tool calls. The agent must read fictional corporate records, identify the operative approval rules, and execute the action through the available tools. The reward is deterministic: `min(1.0, oracle_minimum_process_cost / actual_process_cost)`. There is no LLM judge.

### Overview
- **Environment ID**: `valid-action`
- **Short description**: Tool-using corporate-governance environment. Agent terminates a synthetic corporate action (contract, equity grant, security issuance, related-party transaction, token sale, or subsidiary financing) by orchestrating create + authorize + execute calls under a typed rule graph.
- **Tags**: `legal`, `tool-use`, `multi-turn`, `corporate-governance`, `deterministic`, `synthetic`

### Datasets
- **Primary dataset(s)**: `valid-action-train`, `valid-action-eval`
- **Source**: procedurally generated from 9 hand-authored golden templates (G1-G9) covering all six action families
- **Split sizes**: default 64 train / 32 eval; configurable via `num_train` and `num_eval`

### Task
- **Type**: multi-turn tool use (14-18 turns depending on difficulty)
- **Output format**: tool calls only — prose does not change the world and receives no credit
- **Rubric**: single reward `final_validity_efficiency = min(1.0, oracle / max(1, actual))` where `oracle` is the deterministic DP-computed minimum number of authorization steps and `actual` is the agent's attempt count (only valid steps counted, redundant attempts penalized via `actual > oracle`)

### Quickstart

```bash
# Run an evaluation against any model tier
prime eval run valid-action -m openai/gpt-4.1-mini -n 4 -r 1

# Or evaluate the environment offline via oracle-replay
cd environments/valid_action
python scripts/baselines.py --examples 32 --difficulty medium
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulty` | `easy` \| `medium` \| `hard` | `medium` | Affects max_turns (10/14/18) and number of distractor records (0/2/4) |
| `seed` | int | `17` | Master seed; train uses `seed`, eval uses `seed + split_stride()` |
| `num_train` | int | `64` | Number of train rows |
| `num_eval` | int | `32` | Number of eval rows |
| `preflight_budget` | int | `1` | How many preflight_action calls the agent may make per rollout |
| `max_search_results` | int | `5` | Per-call cap on search_records results |
| `include_distractors` | bool | `true` | Whether to inject distractor records into the world |
| `system_prompt` | object \| null | `null` | Override the default agent prompt |
| `max_turns_override` | int \| null | `null` | Force a specific max_turns value |

### Tools

The agent has access to eight tools organized into three toolsets. The `action_create` toolset is per-task filtered so only the create tool matching the action family is visible.

**Search / read toolset** (always visible):
- `search_records(query, entity_id?, record_type?)` — deterministic BM25 over rendered prose
- `read_record(record_id)` — returns sections with `source_rule_ids` stripped
- `inspect_register(entity_id, register_type, as_of_date)` — returns directors / officers / committee members / cap table / security capacity / equity plan capacity / subsidiaries

**Authorization toolset** (always visible):
- `record_authorization(action_id, authorizer_id, method, participant_ids, recused_ids?, disclosure_record_ids?)` — records one vote or consent step
- `preflight_action(action_id)` — deterministic dry-run of the full requirement graph; consumes one preflight credit

**Per-task create toolset** (one of six, filtered by action family):
- `create_material_contract(...)`, `create_equity_grant(...)`, `create_security_issuance(...)`, `create_related_party_transaction(...)`, `create_token_treasury_transaction(...)`, `create_subsidiary_financing(...)`

**Terminal tool**:
- `execute_action(action_id, signatory_person_id)` — irreversible; sets `state.done=True` and stops the rollout

### State model

The hidden state is a `ValidActionWorld`: an entity (e.g. `ent_lumen`), people, role appointments, governance bodies, security classes, holder positions, equity plan capacity, legal records (bylaws, board minutes, contracts, conflict disclosures, etc.), a typed requirement graph (AuthorizationRequirement, ConsentRequirement, CapacityRequirement, ConflictRequirement, SequenceRequirement, SignatoryRequirement, TermsRequirement, ProhibitionRequirement, composed via AllOf/AnyOf), and one action request. The agent-visible projection strips `source_rule_ids` and never reveals the oracle solution or the requirement graph structure.

### Reward

Single reward `final_validity_efficiency`:
- 0.0 if execution is missing, the action_id doesn't match, or any required step is missing
- `min(1.0, oracle_minimum_process_cost / max(1, actual_process_cost))` otherwise

Eight metrics are also emitted: `execution_attempted`, `execution_valid`, `process_cost`, `process_efficiency_ratio`, `turns_used`, `tool_error_count`, `preflight_used`, `validation_defects` (per-defect-category counts).

### Baselines

Five deterministic baselines are provided in `scripts/baselines.py`:

| Baseline | mean reward (medium, n=16) | valid_rate |
| -------- | -------------------------- | ---------- |
| `random` | 0.000 | 0.0 |
| `naive_max_approval` | 0.188 | 0.188 |
| `keyword_heuristic` | 0.188 | 0.188 |
| `oracle_replay` | 1.000 | 1.0 |
| `search_only` | 0.000 | 0.0 |

The oracle_replay baseline demonstrates the upper bound (100% solvable with reward 1.0). Naive baselines show the gap: ~19% of worlds are simple enough that any single-body meeting approval is sufficient. Random and search-only baselines score 0, validating that prose-only and tool-error-only paths do not accidentally score reward.

### Example trajectory

For a single G3 (employee RSU) world on seed 17:

```
turn 1: search_records(query="committee", entity_id="ent_lumen")
turn 2: read_record(record_id="rec_bylaws_g3")
turn 3: inspect_register(entity_id="ent_lumen", register_type="directors", as_of_date="2028-09-15")
turn 4: create_equity_grant(entity_id="ent_lumen", recipient_person_id="prs_maya_g3",
          award_type="rsu", units=42000, vesting_months=48, cliff_months=12,
          effective_date="2028-09-15")
turn 5: record_authorization(action_id="act_...", authorizer_id="body_comp_g3",
          method="meeting", participant_ids=["role_comp_a_g3", "role_comp_b_g3"])
turn 6: execute_action(action_id="act_...", signatory_person_id="prs_ceo_g3")
→ score: 1.0 (oracle=1, actual=1)
```

### Reproducibility

All randomness is seeded. The generator seeds train and eval splits from a single `seed` plus `split_stride()` and uses a salted SHA256 retag so seeded worlds share a structural fingerprint but have unique IDs. To reproduce the eval set used by CI:

```bash
cd environments/valid_action
python -m pytest tests/ -q   # 103 tests, all offline
python scripts/baselines.py --examples 32 --difficulty medium --seed 17
```

### Limitations

- Synthetic record prose is template-driven (16 wording variants) and may not reflect real-world document variation.
- The rule graph covers six action families common to US-style private-company governance; public-company rules (SEC filings, 14A disclosures, NYSE/Nasdaq listing standards) are out of scope.
- The oracle is a deterministic DP over a typed AND/OR graph; it does not simulate negotiation, amendment, or post-execution events.
- All six action families are covered, but the world is single-action-per-rollout by design; multi-action sequencing across roles is not in scope.

### Version history

- `v0.1.0` — initial scaffold via `prime env init valid-action --multi-file`
- `v0.2.0` — world checkpoint: 12 core modules, G1-G9 fixtures, generator, validator, oracle
- `v0.3.0` — environment checkpoint: Taskset wiring, toolsets, reward/metrics, 8 tool functions
- `v0.4.0` — release checkpoint: 5 baselines, 103 tests, public README, trace renderer stub
