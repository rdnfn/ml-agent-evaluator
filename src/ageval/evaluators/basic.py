"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Basic evaluator.
"""

from string import punctuation
from loguru import logger

from ageval.constants import DEFAULT_OPENAI_MODEL
from ageval.evaluators.core import SingleEvaluator, PairwiseEvaluator
from ageval.models import get_model

BASIC_SINGLE_PROMPT = """"We have the following text: "{text}" {additional_instructions}

Give it a score between {min_score} and {max_score}. Only reply with the score value, nothing else.
"""

BASIC_PAIRWISE_PROMPT = """### Instruction
Please select the better of the two texts below.{additional_instructions}

### Text A:
{text_a}

### Text B:
{text_b}

### Request
Please reply simply with the letter of the better text ('A' or 'B'), nothing else.
"""


class BasicSingleEvaluator(SingleEvaluator):
    def __init__(
        self, min_score: int = 0, max_score: int = 10, model: str = DEFAULT_OPENAI_MODEL
    ) -> None:
        assert min_score < max_score, "`min_score` must be smaller than `max_score`."
        self.min_score = min_score
        self.max_score = max_score
        super().__init__(model)

    def evaluate(self, response: str, prompt: str | None = None) -> float:
        if prompt is not None:
            additional_instructions = f"\nThe text is written in response to the following instruction: '{prompt}'"
        else:
            additional_instructions = ""
        prompt_text = BASIC_SINGLE_PROMPT.format(
            text=response,
            min_score=self.min_score,
            max_score=self.max_score,
            additional_instructions=additional_instructions,
        )
        model_return = self.model.invoke(prompt_text)
        parsed_model_return = self.parse_str_value(model_return)

        return parsed_model_return

    def parse_str_value(self, str_value: str) -> float:
        msg = None
        try:
            parsed_value = float(str_value)
            assert (
                parsed_value <= self.max_score
            ), f"Value {parsed_value} greater than max score {self.max_score}"
            assert (
                parsed_value >= self.min_score
            ), f"Value {parsed_value} less than max score {self.min_score}"

            parsed_value = (parsed_value - self.min_score) / (
                self.max_score - self.min_score
            )
        except ValueError:
            parsed_value = None
            msg = f"Parsing of text '{str_value}' failed."
            logger.warning(msg)

        return {"score": parsed_value, "msg": [msg]}


class BasicPairwiseEvaluator(PairwiseEvaluator):

    supports_additional_instructions: bool = True

    def get_prompt(self, text_a: str, text_b: str, prompt: str | None):
        additional_instructions = "\n"
        additional_instructions += self.additional_instructions
        if prompt is not None:
            additional_instructions += f"\nThe texts were both written in response to the following instruction:\n'{prompt}'\n\n"

        prompt_text = BASIC_PAIRWISE_PROMPT.format(
            text_a=text_a,
            text_b=text_b,
            additional_instructions=additional_instructions,
        )
        return prompt_text

    def get_dummy_prompt(self):
        return self.get_prompt("<text_a>", "<text_b>", "<prompt>")

    def evaluate(self, text_a: str, text_b: str, prompt: str | None) -> int:
        prompt_text = self.get_prompt(
            text_a=text_a,
            text_b=text_b,
            prompt=prompt,
        )
        model_return = self.model.invoke(prompt_text)
        parsed_model_return = self.parse_str_value(model_return)

        return parsed_model_return

    def parse_str_value(self, str_value: str) -> int:
        str_value_cleaned = str_value.strip().strip(punctuation).lower()
        msg = None
        if str_value_cleaned == "a":
            pref_text = "text_a"
        elif str_value_cleaned == "b":
            pref_text = "text_b"
        else:
            msg = f"Evaluator returned value '{str_value}' could not be parsed."
            pref_text = None
            logger.warning(msg)
        return {"preferred_text": pref_text, "msg": [msg]}
