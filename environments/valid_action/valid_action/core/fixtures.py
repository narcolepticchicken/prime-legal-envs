"""Eight hand-authored golden scenarios (spec section 18).

Each fixture is a fully deterministic world with:
  - manually specified valid path;
  - at least two invalid paths;
  - expected defect codes;
  - exact oracle process cost.

Helpers at the top of the file build entity / person / role / body lists in
the correct order so the ValidActionWorld validator (which requires entities
to be declared before records) succeeds on first construction.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from .models import (
    ActionType,
    ApprovalMethod,
    BodyType,
    Difficulty,
    Entity,
    EquityGrantRequest,
    EquityPlanCapacity,
    GovernanceBody,
    HolderPosition,
    LegalRecord,
    MaterialContractRequest,
    OracleSolution,
    Person,
    RecordSection,
    RecordType,
    RelatedPartyTransactionRequest,
    RoleAppointment,
    SecurityClass,
    SecurityIssuanceRequest,
    SubsidiaryFinancingRequest,
    TokenTreasuryTransactionRequest,
    ValidActionWorld,
)
from .requirements import (
    AllOf,
    AnyOf,
    AuthorizationRequirement,
    CapacityRequirement,
    ConflictRequirement,
    ConsentRequirement,
    RequirementGraph,
    SequenceRequirement,
    SignatoryRequirement,
    TermsRequirement,
)


SCHEMA_VERSION = "valid-action.world.v1"


def _d(year: int, month: int, day: int) -> date:
    return date(year, month, day)


def _entities(
    ids: list[str],
    *,
    names: list[str] | None = None,
    parent_map: dict[str, str] | None = None,
) -> list[Entity]:
    out: list[Entity] = []
    for i, eid in enumerate(ids):
        out.append(
            Entity(
                entity_id=eid,
                legal_name=(names or [eid.replace("_", " ").title() + " Inc." for _ in ids])[i],
                entity_type="subsidiary" if parent_map and eid in parent_map else "corporation",
                parent_entity_id=parent_map.get(eid) if parent_map else None,
                jurisdiction="Northstar",
                formation_date=_d(2024, 1, 1),
            )
        )
    return out


def _people(ids: list[str], *, names: list[str] | None = None) -> list[Person]:
    out: list[Person] = []
    for i, pid in enumerate(ids):
        out.append(
            Person(
                person_id=pid,
                display_name=(names or [pid.replace("_", " ").title() for _ in ids])[i],
                active_from=_d(2024, 1, 1),
            )
        )
    return out


def _director_roles(
    *, entity_id: str, person_ids: list[str], body_id: str, suffix: str
) -> list[RoleAppointment]:
    return [
        RoleAppointment(
            role_id=f"role_dir_{chr(97 + i)}_{suffix}",
            entity_id=entity_id,
            person_id=pid,
            role_type="director",
            title="Director",
            body_id=body_id,
            effective_from=_d(2024, 1, 1),
        )
        for i, pid in enumerate(person_ids)
    ]


def _officer_roles(
    *,
    entity_id: str,
    person_ids: list[str],
    title: str = "Chief Executive Officer",
    body_id: str | None = None,
    suffix: str,
) -> list[RoleAppointment]:
    out: list[RoleAppointment] = []
    for i, pid in enumerate(person_ids):
        out.append(
            RoleAppointment(
                role_id=f"role_officer_{i}_{suffix}",
                entity_id=entity_id,
                person_id=pid,
                role_type="officer",
                title=title,
                body_id=body_id or f"body_officer_{suffix}",
                effective_from=_d(2024, 1, 1),
            )
        )
    return out


def _committee_member_roles(
    *, entity_id: str, person_ids: list[str], body_id: str, suffix: str
) -> list[RoleAppointment]:
    return [
        RoleAppointment(
            role_id=f"role_comp_{chr(97 + i)}_{suffix}",
            entity_id=entity_id,
            person_id=pid,
            role_type="committee_member",
            title="Committee Member",
            body_id=body_id,
            effective_from=_d(2024, 1, 1),
        )
        for i, pid in enumerate(person_ids)
    ]


def _board_body(body_id: str, entity_id: str, member_role_ids: list[str]) -> GovernanceBody:
    return GovernanceBody(
        body_id=body_id,
        entity_id=entity_id,
        body_type=BodyType.BOARD,
        display_name="Board of Directors",
        member_role_ids=member_role_ids,
    )


def _committee_body(body_id: str, entity_id: str, member_role_ids: list[str]) -> GovernanceBody:
    return GovernanceBody(
        body_id=body_id,
        entity_id=entity_id,
        body_type=BodyType.COMMITTEE,
        display_name="Committee",
        member_role_ids=member_role_ids,
    )


def _officer_body(body_id: str, entity_id: str, member_role_ids: list[str]) -> GovernanceBody:
    return GovernanceBody(
        body_id=body_id,
        entity_id=entity_id,
        body_type=BodyType.OFFICER,
        display_name="Officer Body",
        member_role_ids=member_role_ids,
    )


def _third_party_body(body_id: str, entity_id: str) -> GovernanceBody:
    return GovernanceBody(
        body_id=body_id,
        entity_id=entity_id,
        body_type=BodyType.THIRD_PARTY,
        display_name="Third Party",
        member_role_ids=[],
    )


# ---------- G1: Delegated ordinary contract ----------


def make_g1_delegated_contract() -> ValidActionWorld:
    action_date = _d(2028, 6, 15)
    entity_id = "ent_northstar"
    suffix = "g1"
    body_id = f"body_ceo_{suffix}"
    ceo_pid = f"prs_ceo_{suffix}"
    ceo_rid = f"role_officer_0_{suffix}"
    request = MaterialContractRequest(
        request_id="req_g1",
        entity_id=entity_id,
        action_date=action_date,
        business_purpose="Vendor agreement with Westwind Logistics.",
        counterparty_id="cp_westwind",
        contract_category="vendor_services",
        total_commitment=Decimal("1250000"),
        term_months=24,
    )
    graph = RequirementGraph(
        root=AllOf(
            requirement_id="root_g1",
            children=[
                TermsRequirement(
                    requirement_id="terms_g1",
                    expected_payload=request.model_dump(
                        mode="json",
                        exclude={"request_id", "business_purpose"},
                    ),
                ),
                AuthorizationRequirement(
                    requirement_id="auth_officer_g1",
                    authorizer_id=body_id,
                    permitted_methods=[ApprovalMethod.DELEGATED_APPROVAL],
                    eligible_voter_role_ids=[ceo_rid],
                    quorum_formula="fixed_minimum",
                    quorum_value=1,
                    vote_threshold="majority_eligible",
                ),
                SignatoryRequirement(
                    requirement_id="sig_g1",
                    eligible_role_ids=[ceo_rid],
                    max_commitment=Decimal("2000000"),
                    entity_id=entity_id,
                ),
            ],
        )
    )
    return ValidActionWorld(
        schema_version=SCHEMA_VERSION,
        world_template_id="G1_delegated_contract",
        seed=1001,
        difficulty=Difficulty.EASY,
        action_date=action_date,
        entities=_entities([entity_id]),
        people=_people([ceo_pid]),
        roles=_officer_roles(
            entity_id=entity_id, person_ids=[ceo_pid], body_id=body_id, suffix=suffix
        ),
        bodies=[_officer_body(body_id, entity_id, [f"role_officer_0_{suffix}"])],
        records=[
            LegalRecord(
                record_id="rec_delegation_g1",
                entity_id=entity_id,
                record_type=RecordType.DELEGATION_MATRIX,
                title="Delegation of Authority",
                effective_date=_d(2027, 1, 1),
                sections=[
                    RecordSection(
                        section_id="s_officer_limit",
                        heading="Officer contract authority",
                        text="",
                        source_rule_ids=["auth_officer_g1"],
                    )
                ],
            )
        ],
        requirements=graph,
        action_request=request,
        oracle_solution=OracleSolution(feasible=True, minimum_process_cost=1),
    )


# ---------- G2: Above-limit contract ----------


def make_g2_above_limit_contract() -> ValidActionWorld:
    action_date = _d(2028, 7, 1)
    entity_id = "ent_orion"
    suffix = "g2"
    body_id = f"body_board_{suffix}"
    ceo_pid = f"prs_ceo_{suffix}"
    ceo_rid = f"role_officer_0_{suffix}"
    dir_pids = [f"prs_d{i+1}_{suffix}" for i in range(3)]
    dir_rids = [f"role_dir_{chr(97 + i)}_{suffix}" for i in range(3)]
    request = MaterialContractRequest(
        request_id="req_g2",
        entity_id=entity_id,
        action_date=action_date,
        business_purpose="Acquisition of Helios Networks.",
        counterparty_id="cp_helios",
        contract_category="acquisition",
        total_commitment=Decimal("3500000"),
        term_months=36,
        includes_exclusivity=True,
    )
    graph = RequirementGraph(
        root=AllOf(
            requirement_id="root_g2",
            children=[
                TermsRequirement(
                    requirement_id="terms_g2",
                    expected_payload=request.model_dump(
                        mode="json",
                        exclude={"request_id", "business_purpose"},
                    ),
                ),
                AnyOf(
                    requirement_id="any_auth_g2",
                    children=[
                        AuthorizationRequirement(
                            requirement_id="auth_board_meeting_g2",
                            authorizer_id=body_id,
                            permitted_methods=[ApprovalMethod.MEETING],
                            eligible_voter_role_ids=dir_rids,
                            quorum_formula="majority_of_seated",
                            quorum_value=1,
                            vote_threshold="majority_present",
                        ),
                        AuthorizationRequirement(
                            requirement_id="auth_board_consent_g2",
                            authorizer_id=body_id,
                            permitted_methods=[ApprovalMethod.WRITTEN_CONSENT],
                            eligible_voter_role_ids=dir_rids,
                            quorum_formula="fixed_minimum",
                            quorum_value=3,
                            vote_threshold="unanimous_consent",
                        ),
                    ],
                ),
                SignatoryRequirement(
                    requirement_id="sig_g2",
                    eligible_role_ids=[ceo_rid],
                    entity_id=entity_id,
                ),
                SequenceRequirement(
                    requirement_id="seq_g2",
                    predecessor_requirement_id="any_auth_g2",
                    successor_requirement_id="sig_g2",
                ),
            ],
        )
    )
    return ValidActionWorld(
        schema_version=SCHEMA_VERSION,
        world_template_id="G2_above_limit_contract",
        seed=1002,
        difficulty=Difficulty.MEDIUM,
        action_date=action_date,
        entities=_entities([entity_id]),
        people=_people([*dir_pids, ceo_pid]),
        roles=[
            *_director_roles(
                entity_id=entity_id, person_ids=dir_pids, body_id=body_id, suffix=suffix
            ),
            *_officer_roles(
                entity_id=entity_id, person_ids=[ceo_pid], body_id=body_id, suffix=suffix
            ),
        ],
        bodies=[
            _board_body(body_id, entity_id, dir_rids),
        ],
        records=[
            LegalRecord(
                record_id="rec_bylaws_g2",
                entity_id=entity_id,
                record_type=RecordType.BYLAWS,
                title="Bylaws",
                effective_date=_d(2026, 5, 1),
                sections=[
                    RecordSection(
                        section_id="s_quorum",
                        heading="Quorum and voting",
                        text="",
                        source_rule_ids=["auth_board_meeting_g2"],
                    )
                ],
            )
        ],
        requirements=graph,
        action_request=request,
        oracle_solution=OracleSolution(feasible=True, minimum_process_cost=1),
    )


# ---------- G3: Employee RSU grant ----------


def make_g3_employee_rsu() -> ValidActionWorld:
    action_date = _d(2028, 9, 15)
    entity_id = "ent_lumen"
    suffix = "g3"
    comp_body = f"body_comp_{suffix}"
    officer_body = f"body_officer_{suffix}"
    comp_pids = [f"prs_comp_a_{suffix}", f"prs_comp_b_{suffix}"]
    comp_rids = [f"role_comp_{chr(97 + i)}_{suffix}" for i in range(2)]
    ceo_pid = f"prs_ceo_{suffix}"
    ceo_rid = f"role_officer_0_{suffix}"
    maya_pid = "prs_maya_g3"
    request = EquityGrantRequest(
        request_id="req_g3",
        entity_id=entity_id,
        action_date=action_date,
        business_purpose="Initial RSU grant for new engineering hire.",
        recipient_person_id=maya_pid,
        award_type="rsu",
        units=42000,
        vesting_months=48,
        cliff_months=12,
    )
    graph = RequirementGraph(
        root=AllOf(
            requirement_id="root_g3",
            children=[
                TermsRequirement(
                    requirement_id="terms_g3",
                    expected_payload=request.model_dump(
                        mode="json",
                        exclude={"request_id", "business_purpose"},
                    ),
                ),
                CapacityRequirement(
                    requirement_id="cap_plan_g3",
                    capacity_kind="plan_pool",
                    target_id=f"plan_main_{suffix}",
                    min_available=42000,
                ),
                AuthorizationRequirement(
                    requirement_id="auth_comp_g3",
                    authorizer_id=comp_body,
                    permitted_methods=[ApprovalMethod.MEETING],
                    eligible_voter_role_ids=comp_rids,
                    quorum_formula="majority_of_seated",
                    quorum_value=1,
                    vote_threshold="majority_present",
                ),
                SignatoryRequirement(
                    requirement_id="sig_g3",
                    eligible_role_ids=[ceo_rid],
                    entity_id=entity_id,
                ),
            ],
        )
    )
    return ValidActionWorld(
        schema_version=SCHEMA_VERSION,
        world_template_id="G3_employee_rsu",
        seed=1003,
        difficulty=Difficulty.EASY,
        action_date=action_date,
        entities=_entities([entity_id]),
        people=_people([*comp_pids, ceo_pid, maya_pid]),
        roles=[
            *_committee_member_roles(
                entity_id=entity_id, person_ids=comp_pids, body_id=comp_body, suffix=suffix
            ),
            *_officer_roles(
                entity_id=entity_id,
                person_ids=[ceo_pid],
                body_id=officer_body,
                suffix=suffix,
            ),
        ],
        bodies=[
            _committee_body(comp_body, entity_id, comp_rids),
            _officer_body(officer_body, entity_id, [ceo_rid]),
        ],
        plan_capacities=[
            EquityPlanCapacity(
                plan_id=f"plan_main_{suffix}",
                entity_id=entity_id,
                total_reserved=500000,
                granted=100000,
                cancelled_returned=0,
            )
        ],
        records=[
            LegalRecord(
                record_id="rec_plan_g3",
                entity_id=entity_id,
                record_type=RecordType.EQUITY_PLAN,
                title="2027 Equity Incentive Plan",
                effective_date=_d(2027, 2, 1),
                sections=[
                    RecordSection(
                        section_id="s_pool",
                        heading="Share reserve",
                        text="",
                        source_rule_ids=["cap_plan_g3"],
                    )
                ],
            ),
            LegalRecord(
                record_id="rec_comp_g3",
                entity_id=entity_id,
                record_type=RecordType.COMMITTEE_CHARTER,
                title="Compensation Committee Charter",
                effective_date=_d(2027, 3, 1),
                sections=[
                    RecordSection(
                        section_id="s_auth",
                        heading="Committee authority",
                        text="",
                        source_rule_ids=["auth_comp_g3"],
                    )
                ],
            ),
        ],
        requirements=graph,
        action_request=request,
        oracle_solution=OracleSolution(feasible=True, minimum_process_cost=1),
    )


# ---------- G4: Director equity grant with conflict ----------


def make_g4_director_grant_conflict() -> ValidActionWorld:
    action_date = _d(2028, 10, 3)
    entity_id = "ent_aster"
    suffix = "g4"
    board_body = f"body_board_{suffix}"
    comp_body = f"body_comp_{suffix}"
    officer_body = f"body_officer_{suffix}"
    priya_pid = f"prs_priya_{suffix}"
    jonah_pid = f"prs_jonah_{suffix}"
    elena_pid = f"prs_elena_{suffix}"
    ceo_pid = f"prs_ceo_{suffix}"
    request = EquityGrantRequest(
        request_id="req_g4",
        entity_id=entity_id,
        action_date=action_date,
        business_purpose="Annual director RSU grant.",
        recipient_person_id=priya_pid,
        award_type="rsu",
        units=60000,
        vesting_months=48,
        cliff_months=12,
    )
    priya_dir_rid = f"role_dir_a_{suffix}"
    jonah_dir_rid = f"role_dir_b_{suffix}"
    elena_dir_rid = f"role_dir_c_{suffix}"
    priya_comp_rid = f"role_comp_a_{suffix}"
    jonah_comp_rid = f"role_comp_b_{suffix}"
    ceo_rid = f"role_officer_0_{suffix}"
    graph = RequirementGraph(
        root=AllOf(
            requirement_id="root_g4",
            children=[
                TermsRequirement(
                    requirement_id="terms_g4",
                    expected_payload=request.model_dump(
                        mode="json",
                        exclude={"request_id", "business_purpose"},
                    ),
                ),
                CapacityRequirement(
                    requirement_id="cap_plan_g4",
                    capacity_kind="plan_pool",
                    target_id=f"plan_main_{suffix}",
                    min_available=60000,
                ),
                AnyOf(
                    requirement_id="any_auth_g4",
                    children=[
                        AuthorizationRequirement(
                            requirement_id="auth_comp_g4",
                            authorizer_id=comp_body,
                            permitted_methods=[ApprovalMethod.MEETING],
                            eligible_voter_role_ids=[jonah_comp_rid, priya_comp_rid],
                            quorum_formula="majority_of_eligible",
                            quorum_value=1,
                            vote_threshold="majority_eligible",
                            conflicted_members_count_for_quorum=False,
                        ),
                        AuthorizationRequirement(
                            requirement_id="auth_board_g4",
                            authorizer_id=board_body,
                            permitted_methods=[ApprovalMethod.MEETING],
                            eligible_voter_role_ids=[
                                priya_dir_rid,
                                jonah_dir_rid,
                                elena_dir_rid,
                            ],
                            quorum_formula="majority_of_eligible",
                            quorum_value=1,
                            vote_threshold="majority_eligible",
                            conflicted_members_count_for_quorum=False,
                        ),
                    ],
                ),
                ConflictRequirement(
                    requirement_id="conflict_g4",
                    trigger_relationship="director",
                    related_person_id=priya_pid,
                    requires_disclosure_record=True,
                    requires_recusal=True,
                    required_approval_body_id=board_body,
                ),
                SignatoryRequirement(
                    requirement_id="sig_g4",
                    eligible_role_ids=[ceo_rid],
                    entity_id=entity_id,
                ),
            ],
        )
    )
    return ValidActionWorld(
        schema_version=SCHEMA_VERSION,
        world_template_id="G4_director_grant_conflict",
        seed=1004,
        difficulty=Difficulty.HARD,
        action_date=action_date,
        entities=_entities([entity_id]),
        people=[
            Person(
                person_id=priya_pid,
                display_name="Priya Rao",
                active_from=_d(2024, 4, 1),
                relationships=["director_recipient"],
            ),
            Person(
                person_id=jonah_pid,
                display_name="Jonah Lee",
                active_from=_d(2024, 4, 1),
            ),
            Person(
                person_id=elena_pid,
                display_name="Elena Morales",
                active_from=_d(2024, 4, 1),
            ),
            Person(
                person_id=ceo_pid,
                display_name="CEO",
                active_from=_d(2024, 4, 1),
            ),
        ],
        roles=[
            RoleAppointment(
                role_id=priya_dir_rid,
                entity_id=entity_id,
                person_id=priya_pid,
                role_type="director",
                title="Director",
                body_id=board_body,
                effective_from=_d(2024, 4, 1),
            ),
            RoleAppointment(
                role_id=jonah_dir_rid,
                entity_id=entity_id,
                person_id=jonah_pid,
                role_type="director",
                title="Director",
                body_id=board_body,
                effective_from=_d(2024, 4, 1),
            ),
            RoleAppointment(
                role_id=elena_dir_rid,
                entity_id=entity_id,
                person_id=elena_pid,
                role_type="director",
                title="Director",
                body_id=board_body,
                effective_from=_d(2024, 4, 1),
            ),
            RoleAppointment(
                role_id=priya_comp_rid,
                entity_id=entity_id,
                person_id=priya_pid,
                role_type="committee_member",
                title="Compensation Committee Member",
                body_id=comp_body,
                effective_from=_d(2024, 4, 1),
            ),
            RoleAppointment(
                role_id=jonah_comp_rid,
                entity_id=entity_id,
                person_id=jonah_pid,
                role_type="committee_member",
                title="Compensation Committee Member",
                body_id=comp_body,
                effective_from=_d(2024, 4, 1),
            ),
            *_officer_roles(
                entity_id=entity_id,
                person_ids=[ceo_pid],
                body_id=officer_body,
                suffix=suffix,
            ),
        ],
        bodies=[
            _board_body(board_body, entity_id, [priya_dir_rid, jonah_dir_rid, elena_dir_rid]),
            _committee_body(comp_body, entity_id, [priya_comp_rid, jonah_comp_rid]),
            _officer_body(officer_body, entity_id, [ceo_rid]),
        ],
        plan_capacities=[
            EquityPlanCapacity(
                plan_id=f"plan_main_{suffix}",
                entity_id=entity_id,
                total_reserved=400000,
                granted=100000,
                cancelled_returned=0,
            )
        ],
        records=[
            LegalRecord(
                record_id="rec_comp_g4",
                entity_id=entity_id,
                record_type=RecordType.COMMITTEE_CHARTER,
                title="Compensation Committee Charter",
                effective_date=_d(2027, 4, 1),
                sections=[
                    RecordSection(
                        section_id="s_excl",
                        heading="Director exclusion",
                        text="",
                        source_rule_ids=["auth_comp_g4"],
                    )
                ],
            ),
            LegalRecord(
                record_id="rec_policy_g4",
                entity_id=entity_id,
                record_type=RecordType.CONFLICT_DISCLOSURE,
                title="Director Conflict Policy",
                effective_date=_d(2028, 1, 15),
                sections=[
                    RecordSection(
                        section_id="s_recusal",
                        heading="Recusal requirements",
                        text="",
                        source_rule_ids=["conflict_g4"],
                    )
                ],
            ),
            LegalRecord(
                record_id="rec_disclosure_g4",
                entity_id=entity_id,
                record_type=RecordType.CONFLICT_DISCLOSURE,
                title="Director Disclosure (Priya Rao)",
                effective_date=_d(2028, 9, 20),
                sections=[
                    RecordSection(
                        section_id="s_facts",
                        heading="Facts disclosed",
                        text="",
                        source_rule_ids=["conflict_g4"],
                    )
                ],
            ),
        ],
        requirements=graph,
        action_request=request,
        oracle_solution=OracleSolution(feasible=True, minimum_process_cost=1),
    )


# ---------- G5: Preferred financing ----------


def make_g5_preferred_financing() -> ValidActionWorld:
    action_date = _d(2028, 8, 10)
    entity_id = "ent_nimbus"
    suffix = "g5"
    board_body = f"body_board_{suffix}"
    preferred_body = f"body_preferred_{suffix}"
    officer_body = f"body_officer_{suffix}"
    dir_pids = [f"prs_dir_a_{suffix}", f"prs_dir_b_{suffix}", f"prs_dir_c_{suffix}"]
    dir_rids = [f"role_dir_{chr(97 + i)}_{suffix}" for i in range(3)]
    ceo_pid = f"prs_ceo_{suffix}"
    ceo_rid = f"role_officer_0_{suffix}"
    holder_pid = f"prs_holder_{suffix}"
    request = SecurityIssuanceRequest(
        request_id="req_g5",
        entity_id=entity_id,
        action_date=action_date,
        business_purpose="Series B preferred financing.",
        purchaser_id="cp_lead_g5",
        class_id="cls_preferred_g5",
        units=2000000,
        price_per_unit=Decimal("1.50"),
        financing_round="Series B",
    )
    graph = RequirementGraph(
        root=AllOf(
            requirement_id="root_g5",
            children=[
                TermsRequirement(
                    requirement_id="terms_g5",
                    expected_payload=request.model_dump(
                        mode="json",
                        exclude={"request_id", "business_purpose"},
                    ),
                ),
                CapacityRequirement(
                    requirement_id="cap_auth_g5",
                    capacity_kind="authorized_shares",
                    target_id="cls_preferred_g5",
                    min_available=2000000,
                ),
                AuthorizationRequirement(
                    requirement_id="auth_board_g5",
                    authorizer_id=board_body,
                    permitted_methods=[ApprovalMethod.WRITTEN_CONSENT],
                    eligible_voter_role_ids=dir_rids,
                    quorum_formula="fixed_minimum",
                    quorum_value=3,
                    vote_threshold="unanimous_consent",
                ),
                ConsentRequirement(
                    requirement_id="consent_class_g5",
                    consent_holder_id=preferred_body,
                    consent_method=ApprovalMethod.CONTRACTUAL_CONSENT,
                ),
                SignatoryRequirement(
                    requirement_id="sig_g5",
                    eligible_role_ids=[ceo_rid],
                    entity_id=entity_id,
                ),
                SequenceRequirement(
                    requirement_id="seq_g5",
                    predecessor_requirement_id="auth_board_g5",
                    successor_requirement_id="consent_class_g5",
                ),
            ],
        )
    )
    return ValidActionWorld(
        schema_version=SCHEMA_VERSION,
        world_template_id="G5_preferred_financing",
        seed=1005,
        difficulty=Difficulty.MEDIUM,
        action_date=action_date,
        entities=_entities([entity_id]),
        people=_people([*dir_pids, ceo_pid, holder_pid]),
        roles=[
            *_director_roles(
                entity_id=entity_id, person_ids=dir_pids, body_id=board_body, suffix=suffix
            ),
            *_officer_roles(
                entity_id=entity_id,
                person_ids=[ceo_pid],
                body_id=officer_body,
                suffix=suffix,
            ),
        ],
        bodies=[
            _board_body(board_body, entity_id, dir_rids),
            GovernanceBody(
                body_id=preferred_body,
                entity_id=entity_id,
                body_type=BodyType.SECURITY_CLASS,
                display_name="Preferred Class",
                member_role_ids=[],
            ),
            _officer_body(officer_body, entity_id, [ceo_rid]),
        ],
        security_classes=[
            SecurityClass(
                class_id="cls_common_g5",
                entity_id=entity_id,
                name="Common Stock",
                authorized=10000000,
                issued=5000000,
                reserved=0,
                votes_per_unit=Decimal("1"),
            ),
            SecurityClass(
                class_id="cls_preferred_g5",
                entity_id=entity_id,
                name="Preferred Stock",
                authorized=8000000,
                issued=3000000,
                reserved=0,
                votes_per_unit=Decimal("1"),
            ),
        ],
        holder_positions=[
            HolderPosition(
                holder_id=holder_pid,
                class_id="cls_preferred_g5",
                units=3000000,
                effective_from=_d(2024, 1, 1),
            )
        ],
        records=[
            LegalRecord(
                record_id="rec_charter_g5",
                entity_id=entity_id,
                record_type=RecordType.CHARTER,
                title="Restated Charter",
                effective_date=_d(2025, 1, 1),
                sections=[
                    RecordSection(
                        section_id="s_authorized",
                        heading="Authorized preferred",
                        text="",
                        source_rule_ids=["cap_auth_g5"],
                    )
                ],
            ),
            LegalRecord(
                record_id="rec_investor_g5",
                entity_id=entity_id,
                record_type=RecordType.INVESTOR_AGREEMENT,
                title="Investor Rights Agreement",
                effective_date=_d(2025, 1, 1),
                sections=[
                    RecordSection(
                        section_id="s_class_consent",
                        heading="Class consent for issuance",
                        text="",
                        source_rule_ids=["consent_class_g5"],
                    )
                ],
            ),
        ],
        requirements=graph,
        action_request=request,
        oracle_solution=OracleSolution(feasible=True, minimum_process_cost=2),
    )


# ---------- G6: Related-party token sale ----------


def make_g6_related_party_token_sale() -> ValidActionWorld:
    action_date = _d(2028, 5, 20)
    entity_id = "ent_chainlink"
    suffix = "g6"
    indep_body = f"body_independent_{suffix}"
    investor_body = f"body_investor_{suffix}"
    officer_body = f"body_officer_{suffix}"
    ind_pids = [f"prs_ind_a_{suffix}", f"prs_ind_b_{suffix}"]
    ind_rids = [f"role_dir_{chr(97 + i)}_{suffix}" for i in range(2)]
    ceo_pid = f"prs_ceo_{suffix}"
    ceo_rid = f"role_officer_0_{suffix}"
    request = TokenTreasuryTransactionRequest(
        request_id="req_g6",
        entity_id=entity_id,
        action_date=action_date,
        business_purpose="Treasury token sale to strategic partner.",
        counterparty_id="cp_partner_g6",
        token_units=Decimal("500000"),
        price_per_token=Decimal("2.10"),
        lockup_months=12,
        related_person_id=f"prs_affiliate_{suffix}",
    )
    graph = RequirementGraph(
        root=AllOf(
            requirement_id="root_g6",
            children=[
                TermsRequirement(
                    requirement_id="terms_g6",
                    expected_payload=request.model_dump(
                        mode="json",
                        exclude={"request_id", "business_purpose"},
                    ),
                ),
                TermsRequirement(
                    requirement_id="floor_g6",
                    expected_payload={"price_per_token": {"op": "ge", "value": "2.00"}},
                ),
                AuthorizationRequirement(
                    requirement_id="auth_independent_g6",
                    authorizer_id=indep_body,
                    permitted_methods=[ApprovalMethod.WRITTEN_CONSENT],
                    eligible_voter_role_ids=ind_rids,
                    quorum_formula="fixed_minimum",
                    quorum_value=2,
                    vote_threshold="unanimous_consent",
                ),
                ConsentRequirement(
                    requirement_id="consent_investor_g6",
                    consent_holder_id=investor_body,
                    consent_method=ApprovalMethod.CONTRACTUAL_CONSENT,
                    participant_id="cp_partner_g6",
                ),
                SignatoryRequirement(
                    requirement_id="sig_g6",
                    eligible_role_ids=[ceo_rid],
                    entity_id=entity_id,
                ),
            ],
        )
    )
    return ValidActionWorld(
        schema_version=SCHEMA_VERSION,
        world_template_id="G6_related_party_token_sale",
        seed=1006,
        difficulty=Difficulty.HARD,
        action_date=action_date,
        entities=_entities([entity_id]),
        people=_people([*ind_pids, ceo_pid, f"prs_affiliate_{suffix}"]),
        roles=[
            RoleAppointment(
                role_id=ind_rids[0],
                entity_id=entity_id,
                person_id=ind_pids[0],
                role_type="director",
                title="Independent Director",
                body_id=indep_body,
                effective_from=_d(2024, 3, 1),
            ),
            RoleAppointment(
                role_id=ind_rids[1],
                entity_id=entity_id,
                person_id=ind_pids[1],
                role_type="director",
                title="Independent Director",
                body_id=indep_body,
                effective_from=_d(2024, 3, 1),
            ),
            *_officer_roles(
                entity_id=entity_id,
                person_ids=[ceo_pid],
                body_id=officer_body,
                suffix=suffix,
            ),
        ],
        bodies=[
            _committee_body(indep_body, entity_id, ind_rids),
            _third_party_body(investor_body, entity_id),
            _officer_body(officer_body, entity_id, [ceo_rid]),
        ],
        records=[
            LegalRecord(
                record_id="rec_floor_g6",
                entity_id=entity_id,
                record_type=RecordType.DELEGATION_MATRIX,
                title="Treasury Sale Floor",
                effective_date=_d(2027, 11, 1),
                sections=[
                    RecordSection(
                        section_id="s_floor",
                        heading="Floor price",
                        text="",
                        source_rule_ids=["floor_g6"],
                    )
                ],
            ),
            LegalRecord(
                record_id="rec_investor_g6",
                entity_id=entity_id,
                record_type=RecordType.INVESTOR_AGREEMENT,
                title="Investor Agreement",
                effective_date=_d(2026, 6, 1),
                sections=[
                    RecordSection(
                        section_id="s_investor_consent",
                        heading="Investor consent",
                        text="",
                        source_rule_ids=["consent_investor_g6"],
                    )
                ],
            ),
        ],
        requirements=graph,
        action_request=request,
        oracle_solution=OracleSolution(feasible=True, minimum_process_cost=2),
    )


# ---------- G7: Subsidiary loan with parent guarantee ----------


def make_g7_subsidiary_loan_guarantee() -> ValidActionWorld:
    action_date = _d(2028, 4, 12)
    entity_id = "ent_holdings_g7"
    sub_id = "ent_sub_g7"
    suffix = "g7"
    sub_board = f"body_sub_board_{suffix}"
    parent_board = f"body_parent_board_{suffix}"
    lender_body = f"body_lender_{suffix}"
    officer_body = f"body_officer_{suffix}"
    sub_pids = [f"prs_sub_a_{suffix}", f"prs_sub_b_{suffix}"]
    sub_rids = [f"role_dir_{chr(97 + i)}_{suffix}" for i in range(2)]
    parent_pids = [f"prs_par_a_{suffix}", f"prs_par_b_{suffix}"]
    parent_rids = [f"role_par_dir_{chr(97 + i)}_{suffix}" for i in range(2)]
    ceo_pid = f"prs_ceo_{suffix}"
    ceo_rid = f"role_officer_0_{suffix}"
    request = SubsidiaryFinancingRequest(
        request_id="req_g7",
        entity_id=entity_id,
        action_date=action_date,
        business_purpose="Working capital facility for the operating subsidiary.",
        subsidiary_id=sub_id,
        lender_id="cp_lender_g7",
        principal=Decimal("4000000"),
        maturity_months=60,
        parent_guarantee=True,
    )
    graph = RequirementGraph(
        root=AllOf(
            requirement_id="root_g7",
            children=[
                TermsRequirement(
                    requirement_id="terms_g7",
                    expected_payload=request.model_dump(
                        mode="json",
                        exclude={"request_id", "business_purpose"},
                    ),
                ),
                AuthorizationRequirement(
                    requirement_id="auth_sub_board_g7",
                    authorizer_id=sub_board,
                    permitted_methods=[ApprovalMethod.WRITTEN_CONSENT],
                    eligible_voter_role_ids=sub_rids,
                    quorum_formula="fixed_minimum",
                    quorum_value=2,
                    vote_threshold="unanimous_consent",
                ),
                AuthorizationRequirement(
                    requirement_id="auth_parent_board_g7",
                    authorizer_id=parent_board,
                    permitted_methods=[ApprovalMethod.WRITTEN_CONSENT],
                    eligible_voter_role_ids=parent_rids,
                    quorum_formula="fixed_minimum",
                    quorum_value=2,
                    vote_threshold="unanimous_consent",
                ),
                ConsentRequirement(
                    requirement_id="consent_lender_g7",
                    consent_holder_id=lender_body,
                    consent_method=ApprovalMethod.CONTRACTUAL_CONSENT,
                    participant_id="cp_lender_g7",
                ),
                SignatoryRequirement(
                    requirement_id="sig_parent_g7",
                    eligible_role_ids=[ceo_rid],
                    entity_id=entity_id,
                ),
                SequenceRequirement(
                    requirement_id="seq_sub_first_g7",
                    predecessor_requirement_id="auth_sub_board_g7",
                    successor_requirement_id="auth_parent_board_g7",
                ),
                SequenceRequirement(
                    requirement_id="seq_consent_after_g7",
                    predecessor_requirement_id="auth_parent_board_g7",
                    successor_requirement_id="consent_lender_g7",
                ),
            ],
        )
    )
    return ValidActionWorld(
        schema_version=SCHEMA_VERSION,
        world_template_id="G7_subsidiary_loan_guarantee",
        seed=1007,
        difficulty=Difficulty.HARD,
        action_date=action_date,
        entities=_entities(
            [entity_id, sub_id],
            parent_map={sub_id: entity_id},
        ),
        people=_people([*sub_pids, *parent_pids, ceo_pid]),
        roles=[
            RoleAppointment(
                role_id=sub_rids[0],
                entity_id=sub_id,
                person_id=sub_pids[0],
                role_type="director",
                title="Subsidiary Director",
                body_id=sub_board,
                effective_from=_d(2024, 7, 1),
            ),
            RoleAppointment(
                role_id=sub_rids[1],
                entity_id=sub_id,
                person_id=sub_pids[1],
                role_type="director",
                title="Subsidiary Director",
                body_id=sub_board,
                effective_from=_d(2024, 7, 1),
            ),
            RoleAppointment(
                role_id=parent_rids[0],
                entity_id=entity_id,
                person_id=parent_pids[0],
                role_type="director",
                title="Parent Director",
                body_id=parent_board,
                effective_from=_d(2022, 1, 1),
            ),
            RoleAppointment(
                role_id=parent_rids[1],
                entity_id=entity_id,
                person_id=parent_pids[1],
                role_type="director",
                title="Parent Director",
                body_id=parent_board,
                effective_from=_d(2022, 1, 1),
            ),
            *_officer_roles(
                entity_id=entity_id,
                person_ids=[ceo_pid],
                body_id=officer_body,
                suffix=suffix,
            ),
        ],
        bodies=[
            _board_body(sub_board, sub_id, sub_rids),
            _board_body(parent_board, entity_id, parent_rids),
            _third_party_body(lender_body, entity_id),
            _officer_body(officer_body, entity_id, [ceo_rid]),
        ],
        records=[
            LegalRecord(
                record_id="rec_sub_charter_g7",
                entity_id=sub_id,
                record_type=RecordType.CHARTER,
                title="Subsidiary Charter",
                effective_date=_d(2024, 7, 1),
                sections=[
                    RecordSection(
                        section_id="s_authority",
                        heading="Subsidiary board authority",
                        text="",
                        source_rule_ids=["auth_sub_board_g7"],
                    )
                ],
            ),
            LegalRecord(
                record_id="rec_credit_g7",
                entity_id=entity_id,
                record_type=RecordType.DEBT_INSTRUMENT,
                title="Existing Credit Agreement",
                effective_date=_d(2026, 2, 1),
                sections=[
                    RecordSection(
                        section_id="s_consent",
                        heading="Lender consent threshold",
                        text="",
                        source_rule_ids=["consent_lender_g7"],
                    )
                ],
            ),
        ],
        requirements=graph,
        action_request=request,
        oracle_solution=OracleSolution(feasible=True, minimum_process_cost=3),
    )


# ---------- G9: Related-party transaction ----------


def make_g9_related_party_contract() -> ValidActionWorld:
    action_date = _d(2028, 3, 15)
    entity_id = "ent_vega"
    suffix = "g9"
    board_body = f"body_board_{suffix}"
    officer_body = f"body_officer_{suffix}"
    dir_pids = [f"prs_dir_a_{suffix}", f"prs_dir_b_{suffix}"]
    dir_rids = [f"role_dir_{chr(97 + i)}_{suffix}" for i in range(2)]
    ceo_pid = f"prs_ceo_{suffix}"
    ceo_rid = f"role_officer_0_{suffix}"
    related_pid = f"prs_related_{suffix}"
    request = RelatedPartyTransactionRequest(
        request_id="req_g9",
        entity_id=entity_id,
        action_date=action_date,
        business_purpose="Marketing services agreement with director-affiliated vendor.",
        counterparty_id="cp_related_g9",
        related_person_id=related_pid,
        transaction_category="marketing_services",
        total_value=Decimal("850000"),
    )
    graph = RequirementGraph(
        root=AllOf(
            requirement_id="root_g9",
            children=[
                TermsRequirement(
                    requirement_id="terms_g9",
                    expected_payload=request.model_dump(
                        mode="json",
                        exclude={"request_id", "business_purpose"},
                    ),
                ),
                AuthorizationRequirement(
                    requirement_id="auth_board_g9",
                    authorizer_id=board_body,
                    permitted_methods=[ApprovalMethod.WRITTEN_CONSENT],
                    eligible_voter_role_ids=dir_rids,
                    quorum_formula="majority_of_eligible",
                    quorum_value=1,
                    vote_threshold="majority_eligible",
                    conflicted_members_count_for_quorum=False,
                ),
                ConflictRequirement(
                    requirement_id="conflict_g9",
                    trigger_relationship="related_party",
                    related_person_id=related_pid,
                    requires_disclosure_record=True,
                    requires_recusal=True,
                    required_approval_body_id=board_body,
                ),
                SignatoryRequirement(
                    requirement_id="sig_g9",
                    eligible_role_ids=[ceo_rid],
                    entity_id=entity_id,
                ),
            ],
        )
    )
    return ValidActionWorld(
        schema_version=SCHEMA_VERSION,
        world_template_id="G9_related_party_contract",
        seed=1009,
        difficulty=Difficulty.HARD,
        action_date=action_date,
        entities=_entities([entity_id]),
        people=[
            Person(
                person_id=related_pid,
                display_name="Director Affiliate",
                active_from=_d(2024, 1, 1),
                relationships=["director_affiliate"],
            ),
            *_people([*dir_pids, ceo_pid]),
        ],
        roles=[
            RoleAppointment(
                role_id=dir_rids[0],
                entity_id=entity_id,
                person_id=dir_pids[0],
                role_type="director",
                title="Director",
                body_id=board_body,
                effective_from=_d(2024, 1, 1),
            ),
            RoleAppointment(
                role_id=dir_rids[1],
                entity_id=entity_id,
                person_id=related_pid,
                role_type="director",
                title="Director (Affiliate)",
                body_id=board_body,
                effective_from=_d(2024, 1, 1),
            ),
            *_officer_roles(
                entity_id=entity_id,
                person_ids=[ceo_pid],
                body_id=officer_body,
                suffix=suffix,
            ),
        ],
        bodies=[
            _board_body(board_body, entity_id, dir_rids),
            _officer_body(officer_body, entity_id, [ceo_rid]),
        ],
        records=[
            LegalRecord(
                record_id="rec_disclosure_g9",
                entity_id=entity_id,
                record_type=RecordType.CONFLICT_DISCLOSURE,
                title="Related Party Disclosure",
                effective_date=_d(2028, 2, 1),
                sections=[
                    RecordSection(
                        section_id="s_facts",
                        heading="Disclosure of related-party interest",
                        text="",
                        source_rule_ids=["conflict_g9"],
                    )
                ],
            ),
        ],
        requirements=graph,
        action_request=request,
        oracle_solution=OracleSolution(feasible=True, minimum_process_cost=1),
    )


# ---------- G8: Superseded delegation trap ----------


def make_g8_superseded_delegation() -> ValidActionWorld:
    action_date = _d(2028, 11, 1)
    entity_id = "ent_kestrel"
    suffix = "g8"
    board_body = f"body_board_{suffix}"
    officer_body = f"body_officer_{suffix}"
    dir_pids = [f"prs_dir_a_{suffix}", f"prs_dir_b_{suffix}", f"prs_dir_c_{suffix}"]
    dir_rids = [f"role_dir_{chr(97 + i)}_{suffix}" for i in range(3)]
    ceo_pid = f"prs_ceo_{suffix}"
    ceo_rid = f"role_officer_0_{suffix}"
    request = MaterialContractRequest(
        request_id="req_g8",
        entity_id=entity_id,
        action_date=action_date,
        business_purpose="Cloud services agreement.",
        counterparty_id="cp_cloud_g8",
        contract_category="vendor_services",
        total_commitment=Decimal("2500000"),
        term_months=24,
    )
    graph = RequirementGraph(
        root=AllOf(
            requirement_id="root_g8",
            children=[
                TermsRequirement(
                    requirement_id="terms_g8",
                    expected_payload=request.model_dump(
                        mode="json",
                        exclude={"request_id", "business_purpose"},
                    ),
                ),
                AuthorizationRequirement(
                    requirement_id="auth_board_g8",
                    authorizer_id=board_body,
                    permitted_methods=[ApprovalMethod.MEETING],
                    eligible_voter_role_ids=dir_rids,
                    quorum_formula="majority_of_seated",
                    quorum_value=1,
                    vote_threshold="majority_present",
                ),
                SignatoryRequirement(
                    requirement_id="sig_g8",
                    eligible_role_ids=[ceo_rid],
                    entity_id=entity_id,
                ),
            ],
        )
    )
    return ValidActionWorld(
        schema_version=SCHEMA_VERSION,
        world_template_id="G8_superseded_delegation",
        seed=1008,
        difficulty=Difficulty.MEDIUM,
        action_date=action_date,
        entities=_entities([entity_id]),
        people=_people([*dir_pids, ceo_pid]),
        roles=[
            *_director_roles(
                entity_id=entity_id, person_ids=dir_pids, body_id=board_body, suffix=suffix
            ),
            *_officer_roles(
                entity_id=entity_id,
                person_ids=[ceo_pid],
                body_id=officer_body,
                suffix=suffix,
            ),
        ],
        bodies=[
            _board_body(board_body, entity_id, dir_rids),
            _officer_body(officer_body, entity_id, [ceo_rid]),
        ],
        records=[
            LegalRecord(
                record_id="rec_old_delegation_g8",
                entity_id=entity_id,
                record_type=RecordType.DELEGATION_MATRIX,
                title="Delegation of Authority (2025)",
                effective_date=_d(2025, 1, 1),
                status="superseded",
                sections=[
                    RecordSection(
                        section_id="s_cfo_old",
                        heading="CFO authority",
                        text="",
                        source_rule_ids=[],
                    )
                ],
            ),
            LegalRecord(
                record_id="rec_new_delegation_g8",
                entity_id=entity_id,
                record_type=RecordType.DELEGATION_MATRIX,
                title="Delegation of Authority (2028)",
                effective_date=_d(2028, 3, 1),
                supersedes_record_ids=["rec_old_delegation_g8"],
                sections=[
                    RecordSection(
                        section_id="s_cfo_new",
                        heading="CFO authority",
                        text="",
                        source_rule_ids=[],
                    )
                ],
            ),
        ],
        requirements=graph,
        action_request=request,
        oracle_solution=OracleSolution(feasible=True, minimum_process_cost=1),
    )


GOLDEN_FIXTURES = {
    "G1": make_g1_delegated_contract,
    "G2": make_g2_above_limit_contract,
    "G3": make_g3_employee_rsu,
    "G4": make_g4_director_grant_conflict,
    "G5": make_g5_preferred_financing,
    "G6": make_g6_related_party_token_sale,
    "G7": make_g7_subsidiary_loan_guarantee,
    "G8": make_g8_superseded_delegation,
    "G9": make_g9_related_party_contract,
}


def all_golden_worlds() -> list[tuple[str, ValidActionWorld]]:
    return [(name, fn()) for name, fn in GOLDEN_FIXTURES.items()]


__all__ = [
    "GOLDEN_FIXTURES",
    "SCHEMA_VERSION",
    "all_golden_worlds",
    "make_g1_delegated_contract",
    "make_g2_above_limit_contract",
    "make_g3_employee_rsu",
    "make_g4_director_grant_conflict",
    "make_g5_preferred_financing",
    "make_g6_related_party_token_sale",
    "make_g7_subsidiary_loan_guarantee",
    "make_g8_superseded_delegation",
    "make_g9_related_party_contract",
]
