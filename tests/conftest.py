import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (may be slow; safe to run on CPU, no GPU required)",
    )


@pytest.fixture
def split_sizes():
    return {"train": 700, "validation": 150, "test": 150}

@pytest.fixture
def allowed_domains():
    return [
        "geography", "science", "history", "math",
        "literature", "technology", "common_knowledge",
    ]

@pytest.fixture
def allowed_control_types():
    return [
        "normal_factual_qa", "creative_writing", "harmless_roleplay",
        "uncertainty_explanation", "nondeceptive_persuasion", "instruction_following",
    ]
