"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Dummy evaluators for testing purposes.
"""

from ageval.evaluators.core import SingleEvaluator, PairwiseEvaluator


class OnlyZero(SingleEvaluator):
    def evaluate(self, text) -> float:
        return 0


class OnlyFirst(PairwiseEvaluator):
    def evaluate(self, text_a, text_b, prompt=None) -> str:
        return {"preferred_text": "text_a", "msg": []}
