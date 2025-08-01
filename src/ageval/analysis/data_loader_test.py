# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import ageval.analysis.data_loader


def test_loading_multirun_data():
    TEST_MULTIRUN_RESULT = "./data/testing/result_0211_truthfulqa_test"
    annotation_dfs, metric_dfs, results = (
        ageval.analysis.data_loader.load_experiments_from_multirun(
            path=TEST_MULTIRUN_RESULT
        )
    )

    # check data path in keys
    DATA_PATH = "./data/external/truthful_qa/truthful_qa_5.csv"
    assert DATA_PATH in annotation_dfs.keys()
    assert DATA_PATH in metric_dfs.keys()

    # check annotation df correct len
    assert len(annotation_dfs[DATA_PATH]) == 5

    # check certain annotation loaded correctly
    assert annotation_dfs[DATA_PATH]["basic_gpt-3.5-turbo-0125_n0"][3] == "text_b"

    # check certain metrics loaded correctly
    assert metric_dfs[DATA_PATH]["Agreed"][0] == 4

    expected_cols = set(
        [
            "index",
            "prompt",
            "text_a",
            "text_b",
            "preferred_text",
            "agent_gpt-4o-2024-05-13_fact_check_code_interpreter_base-basic_n0",
            "agent_gpt-4o-2024-05-13_fact_check_code_interpreter_base-basic_n0_log",
            "basic_gpt-4o-2024-05-13_n0",
            "basic_gpt-4o-2024-05-13_n0_log",
            "agent_gpt-3.5-turbo-0125_fact_check_code_interpreter_base-basic_n0",
            "agent_gpt-3.5-turbo-0125_fact_check_code_interpreter_base-basic_n0_log",
            "basic_gpt-3.5-turbo-0125_n0",
            "basic_gpt-3.5-turbo-0125_n0_log",
        ]
    )

    for annotation_df in annotation_dfs.values():
        # Ignore order of columns.
        assert set(annotation_df.columns) == expected_cols

    assert (
        annotation_dfs[DATA_PATH]["basic_gpt-3.5-turbo-0125_n0"][0]
        == annotation_dfs[DATA_PATH]["basic_gpt-4o-2024-05-13_n0"][0]
    )
