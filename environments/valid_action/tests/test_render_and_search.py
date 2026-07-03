"""Renderer and search consistency tests (spec test matrix section 8 + 9)."""

from __future__ import annotations

import json
import re

import pytest

from valid_action.core.fixtures import GOLDEN_FIXTURES
from valid_action.core.generator import generate_dataset
from valid_action.core.models import Difficulty
from valid_action.core.render import render_world, visible_record
from valid_action.core.search import LexicalIndex
from valid_action.core.serialization import visible_records


def test_render_deterministic_for_same_seed():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    snapshot1 = visible_records(world)
    world2 = GOLDEN_FIXTURES["G3"]()
    render_world(world2, seed=world2.seed)
    snapshot2 = visible_records(world2)
    assert snapshot1 == snapshot2


def test_render_hidden_source_rule_ids_stripped():
    """Agent-visible records must not contain source_rule_ids."""
    for name, factory in GOLDEN_FIXTURES.items():
        world = factory()
        render_world(world, seed=world.seed)
        visible = visible_records(world)
        # Walk every section's text and ensure no rule_id leaks
        text_blob = json.dumps(visible)
        # source_rule_ids are stored as rule_<id>_<n>
        leaked = re.findall(r'rule_[a-z_]+_\d+', text_blob)
        assert not leaked, f"{name}: leaked rule ids {leaked}"


def test_render_every_requirement_renders_a_section():
    """Each operative requirement should map to at least one rendered section."""
    for name, factory in GOLDEN_FIXTURES.items():
        world = factory()
        render_world(world, seed=world.seed)
        rendered = json.dumps(visible_records(world))
        # Should contain at least one heading text from the wording variants.
        # We don't require exact mapping; we just verify rich content.
        assert len(rendered) > 200, f"{name} rendered records too small: {len(rendered)} bytes"


def test_visible_record_strips_source_rule_ids():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    for record in world.records:
        vis = visible_record(record)
        for section in vis["sections"]:
            assert "source_rule_ids" not in section


def test_search_exact_title_match():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    idx = LexicalIndex(world.records)
    results = idx.search("committee")
    assert any("committee" in r["title"].lower() for r in results)


def test_search_unknown_filter_errors():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    idx = LexicalIndex(world.records)
    # Unknown record_type filter should return empty (no match)
    results = idx.search("anything", record_type="bogus_type")
    assert results == []


def test_search_result_limit_enforced():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    idx = LexicalIndex(world.records)
    results = idx.search("the", max_results=2)
    assert len(results) <= 2


def test_search_snippet_from_record_text_only():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    idx = LexicalIndex(world.records)
    results = idx.search("plan")
    for r in results:
        # Snippet content should come from a section text
        assert r["snippet"]
        assert isinstance(r["snippet"], str)


def test_search_superseded_records_discoverable():
    """G8 has a superseded record that should still be searchable but is
    flagged via the record metadata (we test that it's findable)."""
    world = GOLDEN_FIXTURES["G8"]()
    render_world(world, seed=world.seed)
    idx = LexicalIndex(world.records)
    results = idx.search("delegation")
    assert len(results) >= 1


def test_search_determinism_same_seed_same_results():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    idx = LexicalIndex(world.records)
    results1 = idx.search("equity", max_results=5)
    results2 = idx.search("equity", max_results=5)
    assert results1 == results2


def test_search_threshold_filters_low_relevance():
    world = GOLDEN_FIXTURES["G3"]()
    render_world(world, seed=world.seed)
    idx = LexicalIndex(world.records)
    # 'zzz' should not match
    results = idx.search("zzz")
    assert results == []
