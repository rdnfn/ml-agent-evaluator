# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from ageval.experiments.config import ExpConfig
from ageval.experiments.prompts import NoHarmAgentConfig


def test_config_basic_functionality():
    """Test whether deeply nested configs are indeed present as expected."""

    cfg = ExpConfig()

    assert (
        cfg.evaluator_kwargs.prompts.no_harm_agent.will_websearch_help_description
        == NoHarmAgentConfig.will_websearch_help_description
    )

    cfg = ExpConfig(data_path="test_path")
