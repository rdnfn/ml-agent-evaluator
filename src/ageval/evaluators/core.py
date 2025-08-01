"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Core classes of evaluators.
"""

from loguru import logger
from ageval.constants import DEFAULT_OPENAI_MODEL
import ageval.utils
import ageval.evaluators.utils
from ageval.experiments.prompts import PromptConfig

import openai.version

if openai.version.VERSION != "0.27.2":
    from ageval.models import get_model
else:
    logger.warning(
        "Running in compatability mode for "
        "original SAFE implementation. "
        "Some LLM functionality may be limited."
    )
    get_model = lambda model_name, max_tokens: ""


class Evaluator:

    supports_additional_instructions: bool = False

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 256,
        additional_instructions: str = None,
        model_kwargs: dict | None = None,
        prompts: PromptConfig | dict | None = None,
        **kwargs,
    ) -> None:

        if additional_instructions is not None:
            if not self.supports_additional_instructions:
                logger.error(
                    "Additional instructions given but not supported by evaluator! "
                    "They will be ignored."
                    f"(Instructions: {additional_instructions})"
                )
            self.additional_instructions = additional_instructions
        else:
            self.additional_instructions = ""

        if prompts is None:
            prompts = PromptConfig()
        elif isinstance(prompts, dict):
            prompts = ageval.evaluators.utils.get_dataclass_from_dict(
                PromptConfig, prompts
            )

        self.prompts = prompts

        self.model = get_model(
            model,
            temp=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
        )

        if kwargs:
            logger.debug(f"Evaluator {self} ignored the kwargs: {kwargs}")


class SingleEvaluator(Evaluator):
    def evaluate(self, response: str, prompt: str | None = None) -> dict:
        raise NotImplementedError


class PairwiseEvaluator(Evaluator):
    def evaluate(self, text_a: str, text_b: str, prompt: str | None = None) -> dict:
        raise NotImplementedError


class PairwiseFromSingleEvaluator(PairwiseEvaluator):

    def __init__(
        self,
        evaluator: SingleEvaluator,
        larger_is_better: bool = True,
        additional_instructions: str = None,
        **kwargs,
    ) -> None:
        assert (
            additional_instructions is None
        ), "Additional instructions given but not supported for PairswiseFromSingleEvaluator."
        self.single_evaluator = evaluator
        self.larger_is_better = larger_is_better
        super().__init__(**kwargs)

    def evaluate(self, text_a: str, text_b: str, prompt: str | None = None) -> dict:
        text_a_dict = self.single_evaluator.evaluate(response=text_a, prompt=prompt)
        text_b_dict = self.single_evaluator.evaluate(response=text_b, prompt=prompt)

        text_a_score = text_a_dict["score"]
        text_b_score = text_b_dict["score"]

        if text_a_score is None and text_b_score is None:
            pref_text = "Tie (neither text evaluated)"
        elif text_a_score is None:
            pref_text = "text_b"
        elif text_b_score is None:
            pref_text = "text_a"
        elif text_a_score == text_b_score:
            pref_text = "Tie (both same score)"
        elif self.larger_is_better:
            if text_a_score > text_b_score:
                pref_text = "text_a"
            else:
                pref_text = "text_b"
        else:
            if text_a_score < text_b_score:
                pref_text = "text_a"
            else:
                pref_text = "text_b"

        return {
            "preferred_text": pref_text,
            "text_a": text_a_dict,
            "text_b": text_b_dict,
        }
