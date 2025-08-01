# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pandas as pd
import pytest

from ageval.experiments.metrics import compute_agreement, get_standard_metrics


@pytest.mark.parametrize(
    "agreement,pref_text,pred_metrics",
    [
        [
            ["Agreed"],
            ["text_a"],
            {
                "Agreement rate": 1.0,
                "Agreement rate (incl NAs)": 1.0,
                "Agreed": 1,
                "Num text_a evaluator pref": 1,
                "Num text_b evaluator pref": 0,
                "Not agreed": 0,
                "No evaluator annotation available": 0,
                "Not avail.": 0,
                "Sum": 1,
                "Agreed (%)": 100.0,
                "Not agreed (%)": 0.0,
                "Not avail. (%)": 0.0,
            },
        ],
        [
            ["Agreed", "Not agreed", "Agreed", "Agreed"],
            ["text_a", "text_b", "text_a", "text_a"],
            {
                "Agreement rate": 0.75,
                "Agreement rate (incl NAs)": 0.75,
                "Agreed": 3,
                "Num text_a evaluator pref": 3,
                "Num text_b evaluator pref": 1,
                "Not agreed": 1,
                "No evaluator annotation available": 0,
                "Not avail.": 0,
                "Sum": 4,
                "Agreed (%)": 75.0,
                "Not agreed (%)": 25.0,
                "Not avail. (%)": 0.0,
            },
        ],
        [
            [
                "Agreed",
                "Not agreed",
                "No evaluator annotation available",
                "Some other val",
            ],
            ["text_a", "text_b", "No choice", "No choice"],
            {
                "Agreement rate": 0.50,
                "Agreement rate (incl NAs)": 0.25,
                "Agreed": 1,
                "Num text_a evaluator pref": 1,
                "Num text_b evaluator pref": 1,
                "Not agreed": 1,
                "No evaluator annotation available": 1,
                "Some other val": 1,
                "Not avail.": 1,
                "Sum": 4,
                "Agreed (%)": 25.0,
                "Not agreed (%)": 25.0,
                "Not avail. (%)": 25.0,
            },
        ],
    ],
)
def test_evaluator_with_dummy(agreement, pref_text, pred_metrics):
    results_df = pd.DataFrame(
        {"evaluator_agreement": agreement, "evaluator_pref_text": pref_text}
    )

    assert pred_metrics == get_standard_metrics(results_df=results_df)
