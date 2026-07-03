"""Typed requirement graph (spec section 7).

A RequirementGraph is an AND/OR tree of typed nodes. Atomic nodes
(Authorization, Capacity, Conflict, Sequence, Signatory, Terms, Prohibition,
Consent) hold rule parameters; AllOf and AnyOf compose them.

The validator checks atomic nodes against the world. The oracle computes
minimum process cost over the AND/OR tree using dynamic programming.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import ActionType, ApprovalMethod, BodyType, DefectCode


# ---------- Base ----------


class _NodeBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    source_record_id: str | None = None
    source_section_id: str | None = None
    applies_to: ActionType | None = None
    effective_from: date | None = None
    effective_until: date | None = None
    kind: str = Field(init=False)

    def is_operative(self, as_of: date) -> bool:
        if self.effective_from is not None and as_of < self.effective_from:
            return False
        if self.effective_until is not None and as_of > self.effective_until:
            return False
        return True


# ---------- Composites ----------


class AllOf(_NodeBase):
    kind: Literal["all_of"] = "all_of"
    children: list["RequirementNode"]


class AnyOf(_NodeBase):
    kind: Literal["any_of"] = "any_of"
    children: list["RequirementNode"]


# ---------- Atomic requirement nodes ----------


class AuthorizationRequirement(_NodeBase):
    kind: Literal["authorization"] = "authorization"
    authorizer_id: str
    permitted_methods: list[ApprovalMethod]
    eligible_voter_role_ids: list[str] = Field(default_factory=list)
    eligible_voter_person_ids: list[str] = Field(default_factory=list)
    quorum_formula: Literal[
        "majority_of_seated",
        "majority_of_eligible",
        "fixed_minimum",
        "voting_power_pct",
    ] = "majority_of_seated"
    quorum_value: int | Decimal = 1
    vote_threshold: Literal[
        "majority_present",
        "majority_eligible",
        "supermajority_pct",
        "unanimous_consent",
        "voting_power_threshold",
    ] = "majority_present"
    vote_threshold_value: Decimal | int | None = None
    conflicted_members_count_for_quorum: bool = False
    required_recommendation: bool = False
    cumulative_with: list[str] = Field(default_factory=list)


class ConsentRequirement(_NodeBase):
    kind: Literal["consent"] = "consent"
    consent_holder_id: str  # counterparty role id, or class body id
    consent_method: ApprovalMethod = ApprovalMethod.CONTRACTUAL_CONSENT
    participant_id: str | None = None
    does_not_replace_corporate_authority: bool = True


class CapacityRequirement(_NodeBase):
    kind: Literal["capacity"] = "capacity"
    capacity_kind: Literal[
        "authorized_shares",
        "plan_pool",
        "treasury_tokens",
        "borrowing_headroom",
    ]
    target_id: str  # class_id, plan_id, treasury_token_id, or debt_id
    min_available: int
    unit: Literal["units", "tokens", "decimal"] = "units"


class ConflictRequirement(_NodeBase):
    kind: Literal["conflict"] = "conflict"
    trigger_relationship: str  # e.g. "director", "officer", "5_percent_holder"
    related_person_id: str | None = None
    requires_disclosure_record: bool = True
    requires_recusal: bool = True
    required_approval_body_id: str | None = None
    required_approval_quorum_formula: str | None = None
    fairness_threshold: Decimal | None = None  # objective fairness route


class SequenceRequirement(_NodeBase):
    kind: Literal["sequence"] = "sequence"
    predecessor_requirement_id: str
    successor_requirement_id: str


class SignatoryRequirement(_NodeBase):
    kind: Literal["signatory"] = "signatory"
    eligible_role_ids: list[str] = Field(default_factory=list)
    eligible_titles: list[str] = Field(default_factory=list)
    eligible_person_ids: list[str] = Field(default_factory=list)
    max_commitment: Decimal | None = None
    entity_id: str | None = None
    active_on_date: date | None = None


class TermsRequirement(_NodeBase):
    kind: Literal["terms"] = "terms"
    expected_payload: dict[str, Any]


class ProhibitionRequirement(_NodeBase):
    kind: Literal["prohibition"] = "prohibition"
    predicate: str  # e.g. "treasury_floor_breach", "category_blocked"
    description: str = ""


# ---------- Discriminated union ----------


RequirementNode = Annotated[
    Union[
        AllOf,
        AnyOf,
        AuthorizationRequirement,
        ConsentRequirement,
        CapacityRequirement,
        ConflictRequirement,
        SequenceRequirement,
        SignatoryRequirement,
        TermsRequirement,
        ProhibitionRequirement,
    ],
    Field(discriminator="kind"),
]


# Rebuild forward references
AllOf.model_rebuild()
AnyOf.model_rebuild()


# ---------- Graph root ----------


class RequirementGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: RequirementNode

    def collect_ids(self) -> set[str]:
        ids: set[str] = set()
        stack = [self.root]
        while stack:
            node = stack.pop()
            ids.add(node.requirement_id)
            if isinstance(node, AllOf | AnyOf):
                stack.extend(node.children)
        return ids

    def normalize(self) -> "RequirementGraph":
        """Idempotent: flatten nested AllOf/AnyOf, dedupe trivial children."""
        return RequirementGraph(root=_normalize(self.root))


def _normalize(node: RequirementNode) -> RequirementNode:
    if isinstance(node, AllOf):
        flat: list[RequirementNode] = []
        for child in node.children:
            normalized = _normalize(child)
            if isinstance(normalized, AllOf):
                flat.extend(normalized.children)
            else:
                flat.append(normalized)
        return node.model_copy(update={"children": flat})
    if isinstance(node, AnyOf):
        flat = [_normalize(c) for c in node.children]
        return node.model_copy(update={"children": flat})
    return node


# ---------- Helpers used by validator/oracle ----------


def find_node(graph: RequirementGraph, requirement_id: str) -> RequirementNode | None:
    stack = [graph.root]
    while stack:
        node = stack.pop()
        if node.requirement_id == requirement_id:
            return node
        if isinstance(node, AllOf | AnyOf):
            stack.extend(node.children)
    return None


def collect_authorization_requirements(graph: RequirementGraph) -> list[AuthorizationRequirement]:
    found: list[AuthorizationRequirement] = []
    stack = [graph.root]
    while stack:
        node = stack.pop()
        if isinstance(node, AuthorizationRequirement):
            found.append(node)
        if isinstance(node, AllOf | AnyOf):
            stack.extend(node.children)
    return found


def collect_sequence_requirements(graph: RequirementGraph) -> list[SequenceRequirement]:
    found: list[SequenceRequirement] = []
    stack = [graph.root]
    while stack:
        node = stack.pop()
        if isinstance(node, SequenceRequirement):
            found.append(node)
        if isinstance(node, AllOf | AnyOf):
            stack.extend(node.children)
    return found


def collect_signatory_requirements(graph: RequirementGraph) -> list[SignatoryRequirement]:
    found: list[SignatoryRequirement] = []
    stack = [graph.root]
    while stack:
        node = stack.pop()
        if isinstance(node, SignatoryRequirement):
            found.append(node)
        if isinstance(node, AllOf | AnyOf):
            stack.extend(node.children)
    return found


def collect_conflict_requirements(graph: RequirementGraph) -> list[ConflictRequirement]:
    found: list[ConflictRequirement] = []
    stack = [graph.root]
    while stack:
        node = stack.pop()
        if isinstance(node, ConflictRequirement):
            found.append(node)
        if isinstance(node, AllOf | AnyOf):
            stack.extend(node.children)
    return found


def collect_terms_requirements(graph: RequirementGraph) -> list[TermsRequirement]:
    found: list[TermsRequirement] = []
    stack = [graph.root]
    while stack:
        node = stack.pop()
        if isinstance(node, TermsRequirement):
            found.append(node)
        if isinstance(node, AllOf | AnyOf):
            stack.extend(node.children)
    return found


def collect_capacity_requirements(graph: RequirementGraph) -> list[CapacityRequirement]:
    found: list[CapacityRequirement] = []
    stack = [graph.root]
    while stack:
        node = stack.pop()
        if isinstance(node, CapacityRequirement):
            found.append(node)
        if isinstance(node, AllOf | AnyOf):
            stack.extend(node.children)
    return found


def collect_prohibition_requirements(graph: RequirementGraph) -> list[ProhibitionRequirement]:
    found: list[ProhibitionRequirement] = []
    stack = [graph.root]
    while stack:
        node = stack.pop()
        if isinstance(node, ProhibitionRequirement):
            found.append(node)
        if isinstance(node, AllOf | AnyOf):
            stack.extend(node.children)
    return found


def collect_consent_requirements(graph: RequirementGraph) -> list[ConsentRequirement]:
    found: list[ConsentRequirement] = []
    stack = [graph.root]
    while stack:
        node = stack.pop()
        if isinstance(node, ConsentRequirement):
            found.append(node)
        if isinstance(node, AllOf | AnyOf):
            stack.extend(node.children)
    return found


def applies_to_action(node: RequirementNode, action_type: ActionType) -> bool:
    if node.applies_to is None:
        return True
    return node.applies_to == action_type


__all__ = [
    "AllOf",
    "AnyOf",
    "AuthorizationRequirement",
    "CapacityRequirement",
    "ConflictRequirement",
    "ConsentRequirement",
    "ProhibitionRequirement",
    "RequirementGraph",
    "RequirementNode",
    "SequenceRequirement",
    "SignatoryRequirement",
    "TermsRequirement",
    "applies_to_action",
    "collect_authorization_requirements",
    "collect_capacity_requirements",
    "collect_conflict_requirements",
    "collect_consent_requirements",
    "collect_prohibition_requirements",
    "collect_sequence_requirements",
    "collect_signatory_requirements",
    "collect_terms_requirements",
    "find_node",
]
