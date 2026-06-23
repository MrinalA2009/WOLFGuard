"""Tests for benign control bank."""

import pytest

from deception_guardrail.data.controls import get_all_raw_controls
from deception_guardrail.data.schema import ALLOWED_CONTROL_TYPES


def test_at_least_1000_controls():
    controls = get_all_raw_controls()
    assert len(controls) >= 1000, f"Only {len(controls)} controls; need at least 1000"


def test_all_control_types_valid():
    controls = get_all_raw_controls()
    for ctrl in controls:
        assert ctrl.control_type in ALLOWED_CONTROL_TYPES, (
            f"Invalid control_type '{ctrl.control_type}'"
        )


def test_no_empty_prompts():
    controls = get_all_raw_controls()
    for ctrl in controls:
        assert ctrl.prompt.strip(), f"Empty prompt for control_type '{ctrl.control_type}'"


def test_all_control_types_represented():
    controls = get_all_raw_controls()
    found_types = {c.control_type for c in controls}
    for ct in ALLOWED_CONTROL_TYPES:
        assert ct in found_types, f"Control type '{ct}' has no examples"


def test_each_control_type_has_at_least_10():
    controls = get_all_raw_controls()
    counts: dict[str, int] = {}
    for c in controls:
        counts[c.control_type] = counts.get(c.control_type, 0) + 1
    for ct in ALLOWED_CONTROL_TYPES:
        assert counts.get(ct, 0) >= 10, (
            f"Control type '{ct}' has only {counts.get(ct, 0)} examples (need >= 10)"
        )


def test_no_duplicate_prompts():
    controls = get_all_raw_controls()
    prompts = [c.prompt.strip().lower() for c in controls]
    assert len(set(prompts)) == len(prompts), "Duplicate prompts found in control bank"
