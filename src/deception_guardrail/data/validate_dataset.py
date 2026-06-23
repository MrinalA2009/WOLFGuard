"""
Dataset validation: checks structural integrity and split hygiene before
any model is loaded. Exits non-zero on any failure.
"""

from pathlib import Path

from deception_guardrail.data.schema import (
    ALLOWED_CONTROL_TYPES,
    ALLOWED_DOMAINS,
    BenignControl,
    FactualLiePair,
    ProbeSample,
)
from deception_guardrail.utils.io import read_jsonl
from deception_guardrail.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    pass


def _load_pairs(path: Path) -> list[FactualLiePair]:
    raw = read_jsonl(path)
    return [FactualLiePair(**r) for r in raw]


def _load_samples(path: Path) -> list[ProbeSample]:
    raw = read_jsonl(path)
    return [ProbeSample(**r) for r in raw]


def _load_controls(path: Path) -> list[BenignControl]:
    raw = read_jsonl(path)
    return [BenignControl(**r) for r in raw]


def validate_pairs(pairs: list[FactualLiePair], split_sizes: dict) -> None:
    n_total = sum(split_sizes.values())

    if len(pairs) != n_total:
        raise ValidationError(
            f"Expected {n_total} pairs, got {len(pairs)}"
        )

    pair_ids = [p.pair_id for p in pairs]
    if len(set(pair_ids)) != len(pair_ids):
        raise ValidationError("Duplicate pair_ids found")

    split_counts: dict[str, int] = {}
    for p in pairs:
        if not p.pair_id:
            raise ValidationError(f"Empty pair_id in pair {p}")
        if not p.fact:
            raise ValidationError(f"Empty fact in pair {p.pair_id}")
        if not p.question:
            raise ValidationError(f"Empty question in pair {p.pair_id}")
        if not p.truth:
            raise ValidationError(f"Empty truth in pair {p.pair_id}")
        if not p.lie_target:
            raise ValidationError(f"Empty lie_target in pair {p.pair_id}")
        if not p.honest_prompt:
            raise ValidationError(f"Empty honest_prompt in pair {p.pair_id}")
        if not p.deceptive_prompt:
            raise ValidationError(f"Empty deceptive_prompt in pair {p.pair_id}")
        if p.domain not in ALLOWED_DOMAINS:
            raise ValidationError(
                f"Invalid domain '{p.domain}' in pair {p.pair_id}"
            )
        if p.truth.strip().lower() == p.lie_target.strip().lower():
            raise ValidationError(
                f"truth == lie_target for pair {p.pair_id}: '{p.truth}'"
            )
        if p.fact not in p.honest_prompt:
            raise ValidationError(
                f"Fact not found in honest_prompt for pair {p.pair_id}"
            )
        if p.fact not in p.deceptive_prompt:
            raise ValidationError(
                f"Fact not found in deceptive_prompt for pair {p.pair_id}"
            )
        if p.lie_target not in p.deceptive_prompt:
            raise ValidationError(
                f"lie_target not found in deceptive_prompt for pair {p.pair_id}"
            )
        split_counts[p.split] = split_counts.get(p.split, 0) + 1

    for split, expected_count in split_sizes.items():
        actual = split_counts.get(split, 0)
        if actual != expected_count:
            raise ValidationError(
                f"Expected {expected_count} pairs in '{split}', got {actual}"
            )

    logger.info(f"Pairs validation passed: {len(pairs)} pairs, splits {split_counts}")


def validate_probe_samples(
    samples: list[ProbeSample],
    pairs: list[FactualLiePair],
) -> None:
    expected_n = len(pairs) * 2
    if len(samples) != expected_n:
        raise ValidationError(
            f"Expected {expected_n} probe samples, got {len(samples)}"
        )

    sample_ids = [s.sample_id for s in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValidationError("Duplicate sample_ids found")

    pair_to_split = {p.pair_id: p.split for p in pairs}
    pair_samples: dict[str, list[ProbeSample]] = {}
    for s in samples:
        if not s.sample_id:
            raise ValidationError("Empty sample_id")
        if not s.prompt:
            raise ValidationError(f"Empty prompt in sample {s.sample_id}")
        if s.label not in (0, 1):
            raise ValidationError(f"Invalid label {s.label} in sample {s.sample_id}")

        expected_split = pair_to_split.get(s.pair_id)
        if expected_split is None:
            raise ValidationError(
                f"sample {s.sample_id} references unknown pair_id {s.pair_id}"
            )
        if s.split != expected_split:
            raise ValidationError(
                f"PAIR LEAKAGE: sample {s.sample_id} in split '{s.split}' "
                f"but its pair {s.pair_id} is in split '{expected_split}'"
            )

        pair_samples.setdefault(s.pair_id, []).append(s)

    # Each pair must have exactly one honest and one deceptive sample
    for pair_id, samps in pair_samples.items():
        conditions = sorted(s.condition for s in samps)
        if conditions != ["deceptive", "honest"]:
            raise ValidationError(
                f"Pair {pair_id} does not have exactly one honest and one deceptive sample: {conditions}"
            )

    logger.info(f"Probe samples validation passed: {len(samples)} samples")


def validate_no_pair_leakage(pairs: list[FactualLiePair]) -> None:
    pair_splits: dict[str, str] = {}
    for p in pairs:
        if p.pair_id in pair_splits:
            if pair_splits[p.pair_id] != p.split:
                raise ValidationError(
                    f"Pair {p.pair_id} appears in multiple splits: "
                    f"'{pair_splits[p.pair_id]}' and '{p.split}'"
                )
        pair_splits[p.pair_id] = p.split
    logger.info("No pair leakage detected")


def validate_controls(controls: list[BenignControl], min_count: int = 1000) -> None:
    if len(controls) < min_count:
        raise ValidationError(
            f"Expected at least {min_count} controls, got {len(controls)}"
        )

    ctrl_ids = [c.control_id for c in controls]
    if len(set(ctrl_ids)) != len(ctrl_ids):
        raise ValidationError("Duplicate control_ids found")

    for c in controls:
        if not c.control_id:
            raise ValidationError("Empty control_id")
        if not c.prompt:
            raise ValidationError(f"Empty prompt in control {c.control_id}")
        if c.control_type not in ALLOWED_CONTROL_TYPES:
            raise ValidationError(
                f"Invalid control_type '{c.control_type}' in control {c.control_id}"
            )

    logger.info(f"Controls validation passed: {len(controls)} controls")


def run_validation(
    processed_dir: Path,
    split_sizes: dict,
) -> None:
    pairs_path = processed_dir / "factual_lie_pairs.jsonl"
    samples_path = processed_dir / "probe_samples.jsonl"
    controls_path = processed_dir / "benign_controls.jsonl"

    for path in [pairs_path, samples_path, controls_path]:
        if not path.exists():
            raise ValidationError(f"Missing file: {path}")

    logger.info("Loading dataset files...")
    pairs = _load_pairs(pairs_path)
    samples = _load_samples(samples_path)
    controls = _load_controls(controls_path)

    validate_pairs(pairs, split_sizes)
    validate_probe_samples(samples, pairs)
    validate_no_pair_leakage(pairs)
    validate_controls(controls)

    logger.info("All validation checks passed.")
