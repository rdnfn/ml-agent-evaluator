"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Baselines based on AlpacaEval auto-annotator implementation.

By default running AlpacaEval 2.0 default configuration:
https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo
"""

import random
from loguru import logger

from ageval.evaluators.core import PairwiseEvaluator
import alpaca_eval.annotators
import alpaca_eval.constants


class AlpacaEvalPairwiseEvaluator(PairwiseEvaluator):

    def __init__(
        self,
        alpacaeval_config: str = alpaca_eval.constants.DEFAULT_ANNOTATOR_CONFIG,
        **kwargs,
    ) -> None:

        self.alpacaeval_annotator = alpaca_eval.annotators.PairwiseAnnotator(
            annotators_config=alpacaeval_config,
            is_avoid_reannotations=False,  # we want to sample multiple rounds
            seed=random.randint(
                0, 999999999
            ),  # again attempting to ensure no prior annotations used
            # Note: this will produce a lot of files
        )
        super().__init__(
            **kwargs,
        )

    def evaluate(self, text_a: str, text_b: str, prompt: str | None) -> dict:
        ae_result = self.alpacaeval_annotator.annotate_pairs(
            [{"instruction": prompt, "output_1": text_a, "output_2": text_b}],
            num_procs=1,  # avoid other parallelization
        )[0]

        # AlpacaEval output here has the form:
        # [{[...] 'annotator': 'weighted_alpaca_eval_gpt4_turbo',
        # 'preference': 1.0116192901897887,[...]]
        # Where 'preference' close to 1 indicates preference for output 1
        # and close to 2 indicates preference for output 2

        pref_num = ae_result["preference"]

        if pref_num < 1.5:
            pref_text = "text_a"
        elif pref_num > 1.5:
            pref_text = "text_b"
        else:
            pref_text = None

        return {"preferred_text": pref_text, "alpacaeval_result": ae_result}
