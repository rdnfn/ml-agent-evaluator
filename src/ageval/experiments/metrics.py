# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pandas as pd
from loguru import logger


def compute_agreement(
    results_df: pd.DataFrame, include_na_as_not_agreed: bool = False
) -> float:
    value_counts = results_df["evaluator_agreement"].value_counts()
    agreed = value_counts.get("Agreed", 0)
    if not include_na_as_not_agreed:
        not_agreed = value_counts.get("Not agreed", 0)
    else:
        not_agreed = value_counts.sum() - agreed
    if not agreed + not_agreed == 0:
        agreement_rate = agreed / (agreed + not_agreed)
    else:
        logger.error(
            "Could not compute agreement. Both 'Agreed' and 'Not agreed' not in results."
        )
        agreement_rate = None
    return agreement_rate


def get_standard_metrics(results_df: pd.DataFrame) -> dict:
    """Compute standard metrics base on results df.

    Args:
        results_df (pd.DataFrame): dataframe with annotations. Must have columns
            "evaluator_agreement" and "evaluator_pref_text". The column "evaluator_agreement"
            should take values in ["Agreed", "Not agreed"].

    Returns:
        dict: dictionary with detailed results.
    """
    agreement_value_counts = results_df["evaluator_agreement"].value_counts().to_dict()
    eval_pref_counts = results_df["evaluator_pref_text"].value_counts().to_dict()
    metrics = {
        "Agreement rate": compute_agreement(results_df=results_df),
        "Agreement rate (incl NAs)": compute_agreement(
            results_df=results_df, include_na_as_not_agreed=True
        ),
        **agreement_value_counts,
        "Num text_a evaluator pref": eval_pref_counts.get("text_a", 0),
        "Num text_b evaluator pref": eval_pref_counts.get("text_b", 0),
    }

    # addtional metrics
    for col in ["Agreed", "Not agreed", "No evaluator annotation available"]:
        if col not in metrics:
            metrics[col] = 0
    metrics["Not avail."] = metrics["No evaluator annotation available"]

    metrics["Sum"] = len(results_df)
    metrics["Agreed (%)"] = (metrics["Agreed"] / metrics["Sum"]) * 100
    metrics["Not agreed (%)"] = (metrics["Not agreed"] / metrics["Sum"]) * 100
    metrics["Not avail. (%)"] = (metrics["Not avail."] / metrics["Sum"]) * 100

    return metrics
