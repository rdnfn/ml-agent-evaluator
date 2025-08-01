# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pytest

from ageval.evaluators.agent.no_harm import NoHarmAgentEvaluator


def test_basic_agent_annotation_with_dummy_model():
    """Test that the agent annotator `evaluate` method runs without errors using dummy model."""

    # Note: The dummy model does not use any API calls.
    agent = NoHarmAgentEvaluator(
        model="dummy/dummy",
        baseline_kwargs={
            "model_kwargs": {"constant_return_str": "a"},
        },
        agent_kwargs={"model": "dummy/dummy"},
    )

    # with prompt
    result = agent.evaluate(text_a="test1", text_b="test2", prompt="test3")
    assert result["preferred_text"] == "text_a"


@pytest.mark.api
def test_basic_agent_annotation_with_api_model_baseline():
    """Test that the agent annotator `evaluate` method runs without errors using API model."""

    # Note: The dummy model does not use any API calls.
    agent = NoHarmAgentEvaluator()

    # with prompt
    result = agent.evaluate(text_a="test1", text_b="test2", prompt="test3")
    assert result["preferred_text"] == "text_b"


@pytest.mark.api
def test_basic_agent_annotation_with_api_model_agent():
    """Test that the agent annotator `evaluate` method runs without errors using API model."""

    # Note: The dummy model does not use any API calls.
    agent = NoHarmAgentEvaluator()

    # with prompt
    result = agent.evaluate(
        text_a="The golden gate bridge is blue",
        text_b="The golden gate bridge is red",
        prompt="What color is the golden gate bridge?",
    )
    assert result["preferred_text"] == "text_b"
