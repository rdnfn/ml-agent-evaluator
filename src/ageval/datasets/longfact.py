"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

LongFact dataset access tools.
"""

import random
import os
import pathlib
import json
import hashlib
from loguru import logger

# DEFAULT_LONGFACT_PATH = "../third_party/long_form_factuality/longfact/longfact-concepts_gpt4_01-10-2024_noduplicates"
DEFAULT_LONGFACT_PATH = "../third_party/long_form_factuality/longfact/longfact-objects_gpt4_01-12-2024_noduplicates"
# (From paper) "In Section 6, we benchmarked many large language models on LongFact-Objects.
# We chose to exclude LongFact-Concepts during this round of benchmarking
# because we found that LongFact- Concepts is an easier prompt set than LongFact-Objects."


def _load_longfact_data(path: str = None):
    if path is None:
        path = pathlib.Path(DEFAULT_LONGFACT_PATH)

    data = {}
    for longfact_file in os.listdir(path):
        with open(path / longfact_file, "r") as f:
            data[longfact_file.split(".")[-2]] = [json.loads(line) for line in f]

    logger.debug(f"Loaded {len(data)} topics with prompts from longfact.")

    return data


def load_original_row_from_hashed_prompt(hashed_prompts: list[str], *, path: str = None) -> list[dict[str, str]]:
    longfact_data = _load_longfact_data(path=path)
    all_prompts_hashed: dict[str, dict[str, str]] = {}
    for rows in longfact_data.values():
        prompts_hashed = {hashlib.sha256(row["prompt"].strip().encode('utf-8')).hexdigest(): row for row in rows}
        all_prompts_hashed.update(prompts_hashed)

    return [all_prompts_hashed[hashed_prompt] for hashed_prompt in hashed_prompts]


def sample_longfact_data(num_samples: int = 10, seed: int = 42, path: str = None):
    data = _load_longfact_data(path)
    random.seed(42)
    if num_samples < len(data):
        # sample num_samples keys and then sample one prompt from each concept
        sampled_concept_keys = random.sample(list(data.keys()), num_samples)
    else:
        sampled_concept_keys = list(data.keys())

    print(f"Sampling from {len(sampled_concept_keys)} concepts.")

    min_num_samples_per_key = num_samples // len(sampled_concept_keys)
    remainder = num_samples % len(sampled_concept_keys)
    sampled_prompts = []
    for key in sampled_concept_keys:
        num_samples_per_key = min_num_samples_per_key
        if remainder > 0:
            num_samples_per_key += 1
            remainder -= 1

        sampled_prompts += random.sample(data[key], num_samples_per_key)

    return sampled_prompts
