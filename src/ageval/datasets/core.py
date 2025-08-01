# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pathlib
import pandas as pd
import numpy as np

from loguru import logger

from ageval.experiments.config import ExpConfig

EXAMPLE_DATASETS = {"example": "./data/examples/example-truthfulqa.csv"}

FACTUAL_DATASETS = {
    "truthful-qa-100": "./data/external/truthful_qa/truthful_qa_100.csv",
    "truthful-qa-v2": "./data/external/truthful_qa/truthful_qa_400_v2.csv",
    "longfact-v7-100": "./data/generated/longfact/individual_generations_v7_100/longfact_v7_100.csv",
    "longfact-v8": "./data/generated/longfact/individual_generations_v8_fixflips/longfact_v8.csv",
}

REWARDBENCH_DATASETS = {
    "xstest-should-refuse": "./data/external/rewardbench/xstest-should-refuse.csv",
    "hep-python": "./data/external/rewardbench/hep-python.csv",
    "llmbar-adver-GPTOut": "./data/external/rewardbench/llmbar-adver-GPTOut.csv",
    "hep-java": "./data/external/rewardbench/hep-java.csv",
    "hep-js": "./data/external/rewardbench/hep-js.csv",
    "math-prm": "./data/external/rewardbench/math-prm.csv",
    "xstest-should-respond": "./data/external/rewardbench/xstest-should-respond.csv",
    "llmbar-adver-neighbor": "./data/external/rewardbench/llmbar-adver-neighbor.csv",
    "donotanswer": "./data/external/rewardbench/donotanswer.csv",
    "hep-go": "./data/external/rewardbench/hep-go.csv",
    "hep-cpp": "./data/external/rewardbench/hep-cpp.csv",
    "refusals-dangerous": "./data/external/rewardbench/refusals-dangerous.csv",
    "refusals-offensive": "./data/external/rewardbench/refusals-offensive.csv",
    "llmbar-adver-manual": "./data/external/rewardbench/llmbar-adver-manual.csv",
    "mt-bench-easy": "./data/external/rewardbench/mt-bench-easy.csv",
    "hep-rust": "./data/external/rewardbench/hep-rust.csv",
    "llmbar-natural": "./data/external/rewardbench/llmbar-natural.csv",
    "alpacaeval-easy": "./data/external/rewardbench/alpacaeval-easy.csv",
    "alpacaeval-hard": "./data/external/rewardbench/alpacaeval-hard.csv",
    "mt-bench-med": "./data/external/rewardbench/mt-bench-med.csv",
    "llmbar-adver-GPTInst": "./data/external/rewardbench/llmbar-adver-GPTInst.csv",
    "alpacaeval-length": "./data/external/rewardbench/alpacaeval-length.csv",
    "mt-bench-hard": "./data/external/rewardbench/mt-bench-hard.csv",
}

AVAILABLE_DATASETS = {**EXAMPLE_DATASETS, **FACTUAL_DATASETS, **REWARDBENCH_DATASETS}


def load(cfg: ExpConfig) -> pd.DataFrame:
    assert (
        cfg.data_path is None or cfg.data_name is None
    ), f"Both data_name ('{cfg.data_name}') and data_path ('{cfg.data_name}') set to non-default values. Only set one."

    assert not (
        cfg.data_name is None and cfg.data_path is None
    ), "No data_name or data_path set."

    if cfg.data_name is not None:
        path = get_path_from_name(cfg.data_name)
        logger.info(f"Using dataset from name: {cfg.data_name} (path: {path})")
    else:
        path = cfg.data_path
        logger.info(f"Loading dataset from path: {path}")

    return load_from_path(path=path)


def get_path_from_name(name: str) -> str:
    assert (
        name in AVAILABLE_DATASETS
    ), f"Data name '{name}' not found in available datasets ({AVAILABLE_DATASETS.keys()})"

    return AVAILABLE_DATASETS[name]


def load_from_path(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="index")
    df["preferred_text_idx"] = np.where(df["preferred_text"] == "text_a", 0, 1)
    return df
