# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import ast
import pandas as pd
import json
import pathlib
import yaml

from ageval.experiments.metrics import get_standard_metrics
from ageval.experiments.core import check_agreement


def get_synthetic_agent_column(
    df: pd.DataFrame, baseline_col: str, agent_col: str
) -> pd.DataFrame:
    """Create synthetic agent column based on an agent with a different baseline.

    Allows to check swapping out the baseline in an agent run.
    """

    df = df.copy()

    def combine_baseline_agent(row: pd.Series):
        # get the pref texts of agent and baselines
        agent_pref_text = row[agent_col]
        baseline_pref_text = row[baseline_col]

        # get agent log (is a dictionary saved as str)
        log_str = row[agent_col + "_log"]
        log_str = log_str.replace(
            "'text': nan", "'text': 'nan'"
        )  # fix rare issues with nan text values
        agent_log = ast.literal_eval(log_str)

        if agent_log.get("reverted_to_baseline", True):
            # revert to new baseline
            return baseline_pref_text
        else:
            # use agent pref if used in original case
            return agent_pref_text

    return df.apply(combine_baseline_agent, axis=1)


def generate_single_synth_agent_annotations(
    annotation_df: pd.DataFrame,
    baseline_col: str,
    agent_col: str,
    original_data_path: str,
    save_dir: str | None = None,
) -> tuple[pd.DataFrame, dict]:

    # columns of output csv: prompt, text_a, text_b, preferred_text, evaluator_pref_text, evaluator_result
    new_csv = annotation_df[["prompt", "text_a", "text_b", "preferred_text"]].copy()

    # add actual preference column of synthetic agent
    new_csv["evaluator_pref_text"] = get_synthetic_agent_column(
        df=annotation_df,
        baseline_col=baseline_col,
        agent_col=agent_col,
    )

    # add log (note that the baseline result here is NOT updated,
    # thus use with care)
    new_csv["evaluator_result"] = annotation_df[agent_col + "_log"]

    # add evaluator agreement column
    new_csv["evaluator_agreement"] = new_csv.apply(
        lambda row: check_agreement(row["preferred_text"], row["evaluator_pref_text"]),
        axis=1,
    )

    filename_base = f"{agent_col}_newbase-{baseline_col}"

    if save_dir:
        main_result_path = save_dir / f"synth_{filename_base}_01_full_results.csv"
        new_csv.to_csv(main_result_path, index_label="index")

    # computing metrics
    metrics = get_standard_metrics(new_csv)

    if save_dir:
        metrics_path = save_dir / f"synth_{filename_base}_02_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

    # add snythetic hydra configuration
    config = {
        "data_path": original_data_path,
        "evaluator_cls": "agent",
        "evaluator_kwargs": {
            "model": agent_col.split("_")[1],
            "tools": ["synthetic"],
            "baseline": baseline_col.split("_n")[0],
        },
        "synthetic": {
            "agent_origin_col": agent_col,
            "baseline_origin_col": baseline_col,
        },
    }

    if save_dir:
        yaml_dir: pathlib.Path = save_dir / ".hydra"
        yaml_dir.mkdir(exist_ok=True)
        yaml_path = yaml_dir / "config.yaml"
        with open(yaml_path, mode="w") as f:
            yaml.dump(config, f)

    return new_csv, metrics


def generate_synth_agent_results(
    agent_name: str,
    annotation_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    original_data_path: str,
    save_dir: str,
):
    """Create synthetic agent results based on existing agent and baseline results.

    This will create a similar structure as regular experiments in the save_path.

    Args:
        agent_name (str): name of agent to use
        annotation_df (pd.DataFrame): dafaframe with annotations
        metric_df (pd.DataFrame): dataframe with metrics
        save_path (str): path to save synthetic runs to
    """
    save_dir = pathlib.Path(save_dir)

    all_run_config_names = list(metric_df["model"])
    unique_run_config_names = sorted(list(set(all_run_config_names)))
    run_config_counts = {
        name: all_run_config_names.count(name) for name in unique_run_config_names
    }
    baseline_run_config_names = [
        name for name in unique_run_config_names if not "agent" in name
    ]

    num_run = 0
    for base_name in baseline_run_config_names:
        for i in range(run_config_counts[agent_name]):

            # generate individual run column names
            # corresponding to a single seed
            agent_run_col = f"{agent_name}_n{i}"
            baseline_run_col = f"{base_name}_n{i}"

            # if we have no more baseline runs to use, abort
            if i > run_config_counts[base_name]:
                break

            # small sanity check
            assert agent_run_col in annotation_df.columns
            assert baseline_run_col in annotation_df.columns

            # set up dir for individual (synthetic) run
            run_save_dir: pathlib.Path = save_dir / str(num_run)
            run_save_dir.mkdir(exist_ok=True, parents=True)

            generate_single_synth_agent_annotations(
                annotation_df=annotation_df,
                agent_col=agent_run_col,
                baseline_col=baseline_run_col,
                original_data_path=original_data_path,
                save_dir=run_save_dir,
            )

            num_run += 1


def generate_synth_agent_from_multirun(
    multirun_path: str | pathlib.Path,
):
    pass
