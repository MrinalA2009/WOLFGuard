"""
Tests for dataset split integrity using the actual fact bank and build pipeline.
No model download required.
"""

import pytest

from deception_guardrail.data.build_dataset import build_pairs, pairs_to_probe_samples
from deception_guardrail.data.facts import get_all_raw_facts


class MockExpConfig:
    def __init__(self):
        self.seed = 42
        self.split_sizes = {"train": 700, "validation": 150, "test": 150}
        self.allowed_domains = [
            "geography", "science", "history", "math",
            "literature", "technology", "common_knowledge",
        ]
        self.control_types = []
        self.c_grid = [0.1, 1.0]
        self.metrics_output_paths = {}
        self.plot_output_paths = {}


@pytest.fixture
def cfg():
    return MockExpConfig()


@pytest.fixture
def pairs(cfg):
    return build_pairs(cfg)


@pytest.fixture
def samples(pairs):
    return pairs_to_probe_samples(pairs)


def test_fact_bank_has_at_least_1000_facts():
    facts = get_all_raw_facts()
    assert len(facts) >= 1000, f"Fact bank has only {len(facts)} unique facts"


def test_fact_bank_truth_ne_lie_target():
    facts = get_all_raw_facts()
    for rf in facts:
        assert rf.truth.strip().lower() != rf.lie_target.strip().lower(), (
            f"truth == lie_target for fact: '{rf.fact}'"
        )


def test_exactly_1000_pairs(pairs):
    assert len(pairs) == 1000


def test_exactly_2000_probe_samples(samples):
    assert len(samples) == 2000


def test_split_counts(pairs):
    split_counts = {}
    for p in pairs:
        split_counts[p.split] = split_counts.get(p.split, 0) + 1
    assert split_counts["train"] == 700
    assert split_counts["validation"] == 150
    assert split_counts["test"] == 150


def test_no_duplicate_pair_ids(pairs):
    pair_ids = [p.pair_id for p in pairs]
    assert len(set(pair_ids)) == len(pair_ids)


def test_no_pair_leakage(samples):
    """Both variants of a pair must be in the same split."""
    pair_splits: dict[str, str] = {}
    for s in samples:
        if s.pair_id in pair_splits:
            assert pair_splits[s.pair_id] == s.split, (
                f"PAIR LEAKAGE: pair {s.pair_id} appears in "
                f"'{pair_splits[s.pair_id]}' and '{s.split}'"
            )
        pair_splits[s.pair_id] = s.split


def test_each_pair_has_exactly_one_honest_and_one_deceptive(samples):
    from collections import defaultdict
    pair_conditions: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        pair_conditions[s.pair_id].append(s.condition)
    for pair_id, conds in pair_conditions.items():
        assert sorted(conds) == ["deceptive", "honest"], (
            f"Pair {pair_id} has wrong conditions: {conds}"
        )


def test_probe_samples_have_correct_labels(samples):
    for s in samples:
        if s.condition == "honest":
            assert s.label == 0
        else:
            assert s.label == 1


def test_sample_splits_match_pair_splits(pairs, samples):
    pair_to_split = {p.pair_id: p.split for p in pairs}
    for s in samples:
        assert s.split == pair_to_split[s.pair_id], (
            f"Sample {s.sample_id} has split '{s.split}' but its pair "
            f"{s.pair_id} has split '{pair_to_split[s.pair_id]}'"
        )


def test_all_domains_valid(pairs):
    allowed = {"geography", "science", "history", "math", "literature", "technology", "common_knowledge"}
    for p in pairs:
        assert p.domain in allowed, f"Invalid domain '{p.domain}' in pair {p.pair_id}"


def test_truth_ne_lie_target_in_pairs(pairs):
    for p in pairs:
        assert p.truth.strip().lower() != p.lie_target.strip().lower(), (
            f"truth == lie_target in pair {p.pair_id}"
        )


def test_fact_in_prompts(pairs):
    for p in pairs:
        assert p.fact in p.honest_prompt, f"Fact missing from honest_prompt in pair {p.pair_id}"
        assert p.fact in p.deceptive_prompt, f"Fact missing from deceptive_prompt in pair {p.pair_id}"


def test_lie_target_in_deceptive_prompt(pairs):
    for p in pairs:
        assert p.lie_target in p.deceptive_prompt, (
            f"lie_target '{p.lie_target}' missing from deceptive_prompt in pair {p.pair_id}"
        )


def test_deterministic_splits():
    """Same seed must always produce the same split assignment."""
    cfg = MockExpConfig()
    pairs1 = build_pairs(cfg)
    pairs2 = build_pairs(cfg)
    for p1, p2 in zip(pairs1, pairs2):
        assert p1.pair_id == p2.pair_id
        assert p1.split == p2.split
