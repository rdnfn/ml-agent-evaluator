"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Tools for loading APPS coding dataset.

From https://huggingface.co/datasets/codeparrot/apps
"""

from datasets import load_dataset
import pandas as pd
import os
import json
from typing import Literal


def load(
    split: str = "test",
    subset: str = "competition",
    negative_solutions_type: Literal["manual", "gpt4_wrong", "manual_diff"] = "manual",
) -> pd.DataFrame:
    """Load APPS coding dataset.

    Args:
        split (str, optional): Whether "train" or "test"
            dataset should be loaded. Defaults to "test".
        subset (str, optional): Whether "introductory", "interview" or "competition"
            dataset should be loaded (easy to hard). Defaults to "competition".
        negative_solutions_type (str, optional): Whether to load `manual` incorrect
            solutions, or use solutions from GPT4 that were wrong.
            "manual": Take correct solution, break it.
            "gpt4_wrong": Take wrong GPT4 result.
            "manual_diff": Take significantly different correct solution, break it.

    Returns:
        pd.DataFrame: dataset as pandas dataframe.
    """

    hf_dataset = load_dataset(path="codeparrot/apps", name=subset, split=split)
    hf_dataset.set_format(type="pandas")
    df: pd.DataFrame = hf_dataset[:]

    # Take all items that have a solution: results in 310 solutions.
    df = df[df["solutions"].map(len) > 0]

    # Load the negative items too.
    negative_solutions_path = "../../../data/external/code_apps/"
    if negative_solutions_type == "manual":
        negative_solutions_path = os.path.join(
            negative_solutions_path, "negative_solutions_manual.json"
        )
    elif negative_solutions_type == "gpt4_wrong":
        negative_solutions_path = os.path.join(
            negative_solutions_path, "negative_solutions_gpt4.json"
        )
    elif negative_solutions_type == "manual_diff":
        negative_solutions_path = os.path.join(
            negative_solutions_path, "negative_solutions_gt_manual_subset.json"
        )
    else:
        raise ValueError(f"Unknown negative solutions type: {negative_solutions_type}!")
    negative_solutions_path = os.path.join(
        os.path.dirname(__file__),
        negative_solutions_path,
    )
    with open(negative_solutions_path) as in_file:
        negative_solutions_json = json.load(in_file)
    negative_solutions = pd.DataFrame.from_dict(
        negative_solutions_json,
        orient="index",
        columns=["problem_id", "negative_solution", "solution"],
    )
    negative_solutions.index.name = "problem_id"
    # Json is loaded as string.
    negative_solutions.index = negative_solutions.index.astype("int64")

    df = df.set_index("problem_id")
    # Join on `problem_id`.
    df = negative_solutions.join(df)

    result = pd.DataFrame()
    result[["text_a", "text_b", "prompt"]] = df[
        ["negative_solution", "solution", "question"]
    ]
    result["preferred_text"] = "text_b"

    return result
