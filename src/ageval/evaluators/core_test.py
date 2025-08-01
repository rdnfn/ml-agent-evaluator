# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pytest

from ageval.evaluators.basic import BasicPairwiseEvaluator
from ageval.evaluators.dummy import OnlyFirst
from ageval.evaluators.baseline_rewardbench import RewardbenchPairwiseEvaluator
from ageval.evaluators.baseline_arenahard import ArenaHardPairwiseEvaluator
from ageval.evaluators.baseline_alpacaeval import AlpacaEvalPairwiseEvaluator


@pytest.mark.parametrize(
    "evaluator_cls,dummy_return_str,expected_pref",
    [
        (BasicPairwiseEvaluator, "a", "text_a"),
        (BasicPairwiseEvaluator, "b", "text_b"),
        (BasicPairwiseEvaluator, "c", None),
        (RewardbenchPairwiseEvaluator, "[[A]]", "text_a"),
        (RewardbenchPairwiseEvaluator, "[[B]]", "text_b"),
        (RewardbenchPairwiseEvaluator, "[[A]] and [[B]]", None),
        (RewardbenchPairwiseEvaluator, "Completely off-topic response", None),
        (ArenaHardPairwiseEvaluator, "[[A>>B]]", "text_a"),
        (ArenaHardPairwiseEvaluator, "[[A<B]]", "text_b"),
        (ArenaHardPairwiseEvaluator, "Blob", None),
        (OnlyFirst, None, "text_a"),
    ],
)
def test_evaluator_with_dummy(evaluator_cls, dummy_return_str, expected_pref):
    "Test various basic evaluator using dummy models without API calls."
    evaluator = evaluator_cls(
        model="dummy/dummy", model_kwargs={"constant_return_str": dummy_return_str}
    )
    pref = evaluator.evaluate("test_text_a", "text_text_b", "some_prompt")
    assert pref["preferred_text"] == expected_pref


@pytest.mark.api
@pytest.mark.parametrize(
    "evaluator_cls",
    [
        BasicPairwiseEvaluator,
        RewardbenchPairwiseEvaluator,
        ArenaHardPairwiseEvaluator,
        AlpacaEvalPairwiseEvaluator,
    ],
)
def test_evaluator_with_api(evaluator_cls):
    "Test various basic evaluator with API calls."
    evaluator = evaluator_cls(
        model="openai/gpt-3.5-turbo",
    )
    pref = evaluator.evaluate(
        "Cupertino",
        "Somewhere in NYC I think",
        "Where is the headquarter of Apple Inc.?",
    )
    assert pref["preferred_text"] == "text_a"

    pref_2 = evaluator.evaluate(
        "Cupertino",
        "NYC",
        "Where is a headquarter of the UN?",
    )
    assert pref_2["preferred_text"] == "text_b"


def test_evaluator_prompt_config():
    """Test whether prompt config correctly present."""
    evaluator = BasicPairwiseEvaluator(model="dummy/dummy")

    assert isinstance(
        evaluator.prompts.no_harm_agent.will_websearch_help_description, str
    )
