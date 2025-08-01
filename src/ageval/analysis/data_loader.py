"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Tools for loading experiment data.
"""

import pathlib
import re
import yaml
import pandas as pd

import ageval.experiments.metrics
import ageval.datasets.core


def load_experiments_from_multirun(
    path: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    """Load result csv's as one combined dataframe.

    Args:
        path (str): Path to logged results of experiment,
            or comma-separate string with such paths.
            Usually a subdir of './multirun/' dir.

    Returns:
        tuple consisting of:
            - dict[str, pd.DataFrame]: a dictionary with a pandas dataframe as items per dataset
                with datapoint pair as a single row and annotator annotations as columns.
                Keys are the dataset path.
            - dict[str, pd.DataFrame]: a dictionary with per dataset a dataframe that
                has each model as a row with overall metrics as columns.
                Keys are the dataset path.
            - dict: a dict per run experimental results dfs, configs, etc.
    """

    # get all run dirs
    run_dirs = []
    for single_path in str(path).split(","):
        run_dirs += list(pathlib.Path(single_path.strip()).iterdir())

    # load all (valid) run dirs
    # (sorted by data path and config name)
    results_by_path = {}
    for run_dir in run_dirs:
        # check if it is really a run dir with csv results
        # (has to be a directory with numeric name)
        if check_dir_is_valid_run(run_dir=run_dir):

            cfg = get_config(run_dir=run_dir)
            name = get_run_name(cfg=cfg)  # unique name for configuration
            df = get_results_df(run_dir=run_dir)  # df with annotations
            metrics = ageval.experiments.metrics.get_standard_metrics(df)  # metrics

            data_path = cfg["data_path"]

            # if the experiment uses data_name instead of path
            if data_path is None:
                data_path = ageval.datasets.core.get_path_from_name(cfg["data_name"])

            result_dict = {
                "data_path": data_path,
                "cfg": cfg,
                "df": df,
                "metrics": metrics,
            }

            # add result to collected data
            if data_path not in results_by_path:
                results_by_path[data_path] = {name: [result_dict]}
            else:
                if name not in results_by_path[data_path]:
                    results_by_path[data_path][name] = [result_dict]
                else:
                    results_by_path[data_path][name].append(result_dict)

    # per dataset, generate combined annotation and metrics dfs
    annotation_dfs = {}
    metric_dfs = {}
    for data_path, single_path_results in results_by_path.items():

        metric_dicts_single_path = []
        for name, result_dicts in single_path_results.items():
            for i, result_dict in enumerate(result_dicts):
                ann_df_name = name + f"_n{i}"
                single_df = result_dict["df"]

                # if no df created yet add basic columns
                if data_path not in annotation_dfs:
                    cols_in_annotation_dfs = [
                        "index",
                        "prompt",
                        "text_a",
                        "text_b",
                        "preferred_text",
                    ]
                    annotation_dfs[data_path] = single_df[cols_in_annotation_dfs]

                # add annotations of evaluator run
                annotation_dfs[data_path].loc[:, [ann_df_name]] = single_df[
                    "evaluator_pref_text"
                ]
                annotation_dfs[data_path].loc[:, [ann_df_name + "_log"]] = single_df[
                    "evaluator_result"
                ]

                # add metrics to metric dicts
                metric_dicts_single_path.append(
                    {"model": name, **result_dict["metrics"]}
                )
        metric_dfs[data_path] = pd.DataFrame(metric_dicts_single_path)

    return annotation_dfs, metric_dfs, results_by_path


def get_config(run_dir: pathlib.Path) -> dict:
    """Get yaml run config as dictionary."""
    yaml_path = run_dir / ".hydra" / "config.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_results_df(run_dir: pathlib.Path) -> pd.DataFrame:
    """Load results from csv file."""
    csv_files = list(run_dir.glob("*.csv"))
    assert (
        len(csv_files) <= 1
    ), f"Too many csv files ({len(csv_files)}): are there multiple results in a single run path? See {run_dir}"
    results_path = csv_files[0]
    df = pd.read_csv(results_path)
    return df


def check_dir_is_valid_run(run_dir: pathlib.Path) -> bool:
    """Check dir is from a valid and completed experiment run."""
    return (
        run_dir.is_dir()  # is directory
        and str(run_dir.name).isnumeric()  # has numeric run number as name
        and len(list(run_dir.glob("*.csv"))) > 0  # contains csv results file
    )


def get_run_name(cfg: dict) -> str:
    """Create a short run name to identify run

    Args:
        cfg (dict): configuration to use to construct run name

    Returns:
        str: short run name
    """

    def _get_short_evaluator_cls(evaluator_cls: str) -> str:
        return evaluator_cls.split(".")[-1].replace("PairwiseEvaluator", "").lower()

    # get short version of evaluator class string
    evaluator_cls = _get_short_evaluator_cls(cfg["evaluator_cls"])
    evaluator_kwargs = cfg["evaluator_kwargs"]

    # get model name (without provider)
    model = evaluator_kwargs["model"].split("/")[-1]

    # create standard name
    name = f"{evaluator_cls}_{model}"

    # if agent add tools used
    if "agent" in evaluator_cls:
        name += "_" + "_".join(evaluator_kwargs["tools"])

        # add baseline used
        name += "_base-" + _get_short_evaluator_cls(
            evaluator_kwargs.get("baseline", "basic")
        )

        if evaluator_kwargs.get("baseline_kwargs", None) is not None:
            # add entire baseline kwargs dict to str, removing all non-alphanumeric chars
            name += "-" + re.sub(r"\W+", "", str(cfg["baseline_kwargs"]))

    return name
