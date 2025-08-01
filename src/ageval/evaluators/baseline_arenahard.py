"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Baselines based on Arena hard auto-annotator implementation.
"""

from loguru import logger

from ageval.evaluators.core import PairwiseEvaluator

# https://github.com/lmarena/arena-hard-auto/blob/main/config/judge_config.yaml.
# Adapted to not allow ties
# Original prompt: "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n3. Tie, relatively the same: [[A=B]]\n4. Assistant B is slightly better: [[B>A]]\n5. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is tie: [[A=B]]\"."

SYSTEM_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.\n\nBegin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.\n\nWhen evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.\n\nThen consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.\n\nThen consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.\n\nAfter providing your explanation, you must output only one of the following choices as your final verdict with a label:\n\n1. Assistant A is significantly better: [[A>>B]]\n2. Assistant A is slightly better: [[A>B]]\n\n3. Assistant B is slightly better: [[B>A]]\n4. Assistant B is significantly better: [[B>>A]]\n\nExample output: \"My final verdict is Assistant B is slightly better: [[B>A]]\". Note that ties are not allowed, always make a decision."

USER_PROMPT = "<|User Prompt|>\n{question_1}\n\n<|The Start of Assistant A's Answer|>\n{answer_1}\n<|The End of Assistant A's Answer|>\n\n<|The Start of Assistant B's Answer|>\n{answer_2}\n<|The End of Assistant B's Answer|>"


class ArenaHardPairwiseEvaluator(PairwiseEvaluator):

    def evaluate(self, text_a: str, text_b: str, prompt: str | None) -> int:
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT.format(
            question_1=prompt,
            answer_1=text_a,
            answer_2=text_b,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        model_return = self.model.invoke_with_messages(messages)
        parsed_model_return = self.parse_str_value(model_return)

        return parsed_model_return

    def parse_str_value(self, str_value: str) -> int:
        print(f"{str_value=}")
        msg = None
        pref_a_vals = ["[[A>>B]]", "[[A>B]]", "[[B<A]]", "[[B<<A]]"]
        pref_b_vals = ["[[B>>A]]", "[[B>A]]", "[[A<B]]", "[[A<<B]]"]
        pref_a = any([val in str_value for val in pref_a_vals])
        pref_b = any([val in str_value for val in pref_b_vals])
        if not (pref_a and pref_b):
            if pref_a:
                pref_text = "text_a"
            elif pref_b:
                pref_text = "text_b"
            else:
                msg = f"Value '{str_value}' returned by model in evaluator could not be parsed."
                pref_text = None
        else:
            msg = f"Value '{str_value}' could not be parsed, it contains both [[A]] and [[B]]."
            pref_text = None

        if msg is not None:
            logger.warning(msg)
        return {
            "preferred_text": pref_text,
            "msg": [msg],
            "full_return_value": str_value,
        }
