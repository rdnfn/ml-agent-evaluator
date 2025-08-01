# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import dataclasses
import pytest
import pandas as pd

import ageval.evaluators
import ageval.evaluators.agent
import ageval.evaluators.basic
import ageval.experiments.core
from ageval.experiments.config import ExpConfig, EvaluatorConfig


@pytest.mark.api
def test_run_dummy_evaluator_on_pairwise_data():
    """Test running dummy pairwise evaluator on 5 data pairs."""

    cfg = ExpConfig(
        data_path="data/external/truthful_qa/truthful_qa_5.csv",
        evaluator_cls="ageval.evaluators.dummy.OnlyFirst",
    )
    df = ageval.datasets.core.load(cfg)
    evaluator_cls = ageval.experiments.core.get_evaluator_cls_from_str(
        cfg.evaluator_cls
    )
    results = ageval.experiments.core.run_evaluator_on_pairwise_data(
        df=df,
        evaluator_cls=evaluator_cls,
        evaluator_kwargs=dataclasses.asdict(cfg.evaluator_kwargs),
        randomize_position=False,
        parallelize=cfg.parallelize,
        num_workers=cfg.num_workers,
        show_progress_bar=cfg.show_progress_bar,
        verbose=cfg.verbose,
        save_dir=None,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
    )

    assert results["metrics"]["No evaluator annotation available"] == 0
    assert results["metrics"]["Agreed"] == 2
    assert results["metrics"]["Not agreed"] == 3

    assert not results["annotated_df"]["evaluator_result"][0]["text_position_flipped"]


@pytest.mark.api
@pytest.mark.parametrize(
    "evaluator_cls,agreed_expected,non_agreed_expected",
    [
        ("ageval.evaluators.basic.BasicPairwiseEvaluator", 5, 0),
        ("ageval.evaluators.agent.PairwiseEvaluatorAgent", 5, 0),
    ],
)
def test_run_api_evaluator_on_pairwise_data(
    evaluator_cls, agreed_expected, non_agreed_expected
):
    """Test running API-based pairwise evaluator on 5 data pairs."""

    cfg = ExpConfig(
        data_path="data/external/truthful_qa/truthful_qa_5.csv",
        evaluator_cls=evaluator_cls,
        evaluator_kwargs=EvaluatorConfig(model="openai/gpt-4o"),
    )
    df = ageval.datasets.core.load(cfg)
    evaluator_cls = ageval.experiments.core.get_evaluator_cls_from_str(
        cfg.evaluator_cls
    )
    results = ageval.experiments.core.run_evaluator_on_pairwise_data(
        df=df,
        evaluator_cls=evaluator_cls,
        evaluator_kwargs=dataclasses.asdict(cfg.evaluator_kwargs),
        randomize_position=True,
        parallelize=cfg.parallelize,
        num_workers=cfg.num_workers,
        show_progress_bar=cfg.show_progress_bar,
        verbose=cfg.verbose,
        save_dir=None,
        wandb_project=None,
        wandb_entity=None,
        wandb_group=None,
    )

    annotated_df: pd.DataFrame = results["annotated_df"]

    assert (
        results["metrics"]["No evaluator annotation available"] == 0
    ), "Non-expected result. Note this is probabilistic API-based test."
    assert (
        results["metrics"]["Agreed"] == agreed_expected
    ), "Non-expected result. Note this is probabilistic API-based test."
    assert (
        results["metrics"]["Not agreed"] == non_agreed_expected
    ), "Non-expected result. Note this is probabilistic API-based test."

    # check that some values were flipped

    positions_flipped = [
        annotated_df["evaluator_result"][i]["text_position_flipped"]
        for i in range(len(annotated_df))
    ]
    assert (
        True in positions_flipped
    ), "No position flipped, non-deterministic test - try again."
