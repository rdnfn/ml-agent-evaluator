"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Rewardbench dataset

From https://huggingface.co/datasets/allenai/reward-bench
"""

import pandas as pd


def load(
    split: str = "filtered",
    subset: str = "hep-python",
) -> pd.DataFrame:
    """Load Rewardbench dataset.

    Args:
        split (str, optional): Whether "raw" (unfiltered) or filtered
            dataset should be loaded. Defaults to "filtered".
        subset (str, optional): Data subset to be loaded. Defaults to "hep-python".
            Other options are: 'math-prm',  'xstest-should-respond',
            'hep-python',  'hep-java',  'hep-js',  'hep-cpp',  'hep-go',
            'hep-rust',  'xstest-should-refuse',  'donotanswer',
            'llmbar-adver-neighbor',  'refusals-dangerous',
            'llmbar-natural',  'alpacaeval-easy',  'refusals-offensive',
            'alpacaeval-hard',  'alpacaeval-length',  'llmbar-adver-GPTInst',
            'llmbar-adver-GPTOut',  'llmbar-adver-manual',  'mt-bench-med',
            'mt-bench-hard',  'mt-bench-easy', or "ALL".

    Returns:
        pd.DataFrame: dataset as pandas dataframe.
    """

    splits = {
        "raw": "data/raw-00000-of-00001.parquet",
        "filtered": "data/filtered-00000-of-00001.parquet",
    }
    df = pd.read_parquet("hf://datasets/allenai/reward-bench/" + splits[split])
    avail_subsets = list(df["subset"].unique())

    if subset != "ALL":
        assert (
            subset in avail_subsets
        ), f"Subset {subset} not found in avail subsets: {avail_subsets}"
        df = df[df["subset"] == subset]

    df[["text_a", "text_b"]] = df[["chosen", "rejected"]]
    df["preferred_text"] = "text_a"

    return df
