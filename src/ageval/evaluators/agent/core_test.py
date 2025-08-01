# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pytest

from ageval.evaluators.agent.core import PairwiseEvaluatorAgent
from ageval.evaluators.agent.tools.fact_checking import ToolFact
from ageval.evaluators.agent.tools.code_interpreter import ToolCodeInterpreter
from unittest.mock import patch, MagicMock


def _get_model_mock_fn(model_mock_obj):
    def _get_model_mock(*_, **__):
        return model_mock_obj

    return _get_model_mock


def test_initial_assessment():
    """Check that the correct structured output is set by the initial assessment method."""
    tools = {"fact_check": ToolFact, "code_interpreter": ToolCodeInterpreter}
    model_mock = MagicMock()
    with patch("ageval.evaluators.core.get_model", _get_model_mock_fn(model_mock)):
        agent = PairwiseEvaluatorAgent(model="dummy/dummy")
        initial_assessment = agent._initial_assessment("text", tools=tools)

    assert initial_assessment == {}
    model_mock.langchain_model.with_structured_output.assert_called_once_with(
        {
            "title": "TextAssessment",
            "description": "Assessment of a text.",
            "type": "object",
            "properties": {
                "contains_facts": {
                    "title": "Contains Facts",
                    "description": "Whether the text contains any facts that may be checked using a web search.",
                    "type": "boolean",
                },
                "is_like_wiki": {
                    "title": "Is Like Wiki",
                    "description": "Whether the response text could be from a wiki page.",
                    "type": "boolean",
                },
                "is_maths": {
                    "title": "Is Maths",
                    "description": "Whether the text is a solution to any kind of maths problem.",
                    "type": "boolean",
                },
                "is_wordcount": {
                    "title": "Is Wordcount",
                    "description": "Whether the text is providing a word count.",
                    "type": "boolean",
                },
                "confidence_websearch_will_help": {
                    "title": "Confidence Websearch Will Help",
                    "description": "Confidence that a websearch will help correctly select the better response. Integer between 0 (won't help) and 5 (will with absolute certainty help), 3 would mean 'may help'.Consider whether there are facts present in either response, and if (!) consider whether these facts can be checked in a websearch. For example a word count task can't be checked with a websearch, but the birthday of a celebrity may be checked. Remember that websearches do not help on maths problems.",
                    "type": "integer",
                },
                "code_useful": {
                    "title": "Code Useful",
                    "description": "Whether text might benefit from running code.",
                    "type": "boolean",
                },
            },
            "required": [
                "contains_facts",
                "is_like_wiki",
                "is_maths",
                "is_wordcount",
                "confidence_websearch_will_help",
                "code_useful",
            ],
        }
    )


def test_basic_agent_annotation_with_dummy_model():
    """Test that the agent annotator `evaluate` method runs without errors using dummy model."""

    # Note: The dummy model does not use any API calls.
    agent = PairwiseEvaluatorAgent(
        model="dummy/dummy",
        baseline_kwargs={"model_kwargs": {"constant_return_str": "a"}},
    )

    # with prompt
    result = agent.evaluate(text_a="test1", text_b="test2", prompt="test3")
    assert result["preferred_text"] == "text_a"

    # no prompt
    result2 = agent.evaluate(text_a="test1", text_b="test2")
    assert result2["preferred_text"] == "text_a"


@pytest.mark.api
def test_reverting_to_basemodel():
    """Test if agent correctly uses baseline if tools not relevant."""
    agent = PairwiseEvaluatorAgent(model="openai/gpt-3.5-turbo")

    # simple task that should not require any tool use
    result = agent.evaluate(prompt="Hi!", text_a="Hello!", text_b="Don't talk to me.")
    assert result["reverted_to_baseline"]
    assert result["preferred_text"] == "text_a"


@pytest.mark.api
def test_not_reverting_to_basemodel():
    """Test if agent uses tools when useful?"""
    agent = PairwiseEvaluatorAgent(model="openai/gpt-3.5-turbo")

    # simple task that uses a tool
    result = agent.evaluate(
        prompt="Where are the Olympics in 2024?",
        text_a="The Olympics 2024 are based in London.",
        text_b="The Olympics 2024 are based in Paris.",
    )
    assert not result["reverted_to_baseline"]
    assert result["preferred_text"] == "text_b"
