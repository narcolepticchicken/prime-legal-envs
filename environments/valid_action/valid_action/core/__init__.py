"""Core domain logic for the valid-action environment.

Modules:
    models: Pydantic domain models and enums.
    requirements: Typed requirement graph and AND/OR normalization.
    validator: Authorization attempt and final execution validation.
    oracle: Minimum valid process-cost solver.
    generator: Seeded world and task generation.
    render: Deterministic legal-record renderer.
    search: Deterministic lexical search index.
    serialization: Canonical JSON, fingerprinting, task row conversion.
    scoring: Final reward and score breakdown.
    fixtures: Golden scenarios (G1–G8).
    tools: Tool implementations for the Taskset.
"""

from .models import (
    ActionType,
    ApprovalMethod,
    BodyType,
    DefectCode,
    Difficulty,
    RecordType,
)
from .serialization import compute_fingerprint, world_to_task_payload

__all__ = [
    "ActionType",
    "ApprovalMethod",
    "BodyType",
    "DefectCode",
    "Difficulty",
    "RecordType",
    "compute_fingerprint",
    "world_to_task_payload",
]
