"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Evaluators from long-form factuality work.

https://github.com/google-deepmind/long-form-factuality
https://arxiv.org/abs/2403.18802 
"""

from ageval.evaluators.core import SingleEvaluator


class FactEvaluator(SingleEvaluator):

    def evaluate(self, text: str) -> float:
        return super().evaluate(text)