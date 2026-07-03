"""Final reward and score breakdown (spec section 16)."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ExecutionAttempt, ValidActionWorld
from .validator import ValidationResult, validate_final_execution


@dataclass
class ScoreBreakdown:
    valid: bool
    defect_codes: list[str]
    process_cost: int
    minimum_process_cost: int
    reward: float

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "defect_codes": list(self.defect_codes),
            "process_cost": self.process_cost,
            "minimum_process_cost": self.minimum_process_cost,
            "reward": self.reward,
        }


def compute_reward(
    world: ValidActionWorld,
    created_action,
    authorization_attempts: list,
    execution_attempt: ExecutionAttempt | None,
) -> tuple[ScoreBreakdown, ValidationResult]:
    result = validate_final_execution(
        world=world,
        created_action=created_action,
        authorization_attempts=authorization_attempts,
        execution=execution_attempt,
    )
    reward = _calculate_reward(result, execution_attempt)
    breakdown = ScoreBreakdown(
        valid=result.valid,
        defect_codes=[d.value for d in result.defect_codes],
        process_cost=result.process_cost,
        minimum_process_cost=result.minimum_process_cost,
        reward=reward,
    )
    return breakdown, result


def _calculate_reward(result: ValidationResult, execution_attempt: ExecutionAttempt | None) -> float:
    if execution_attempt is None or not execution_attempt.valid:
        return 0.0
    if not result.valid:
        return 0.0
    actual = max(1, int(result.process_cost))
    oracle = max(1, int(result.minimum_process_cost))
    return min(1.0, oracle / actual)


__all__ = ["ScoreBreakdown", "compute_reward"]
