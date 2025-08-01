# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from ageval.analysis.post_processing import generate_single_synth_agent_annotations

import ageval.analysis.data_loader


def test_generating_synth_agent_annotations():
    """Basic test of generating synthetic annotations."""

    multirun_path = "./data/testing/result_0211_truthfulqa_test"

    annotation_dfs, metric_dfs, results_dict = (
        ageval.analysis.data_loader.load_experiments_from_multirun(multirun_path)
    )

    df_ann = list(annotation_dfs.values())[0]

    agent_col = "agent_gpt-4o-2024-05-13_fact_check_code_interpreter_base-basic_n0"

    # test where the synthetic result should change over original agent
    baseline_col = "basic_gpt-3.5-turbo-0125_n0_log"
    annotation_df, _ = generate_single_synth_agent_annotations(
        annotation_df=df_ann,
        baseline_col=baseline_col,
        agent_col=agent_col,
        original_data_path="test",
    )
    assert not (df_ann[agent_col] == annotation_df["evaluator_pref_text"]).all()

    # test where the synthetic result should (most likely) not change anything
    # (note this is probabilistic, for some alternative baseline runs this would
    # not be identical)
    should_not_change_base_col = "basic_gpt-4o-2024-05-13_n0"
    annotation_df, _ = generate_single_synth_agent_annotations(
        annotation_df=df_ann,
        baseline_col=should_not_change_base_col,
        agent_col=agent_col,
        original_data_path="test",
    )
    assert (df_ann[agent_col] == annotation_df["evaluator_pref_text"]).all()
