"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Functions to do data processing inside app.
"""

from typing import Any
import ast
import pathlib
import pandas as pd
import gradio as gr

import ageval.analysis.data_loader
from ageval.analysis.constants import (
    ALLOWED_ANNOTATION_VALUES,
    DEFAULT_GROUNDTRUTH_COL,
    DEFAULT_ANNOTATOR_1_COL,
    COMPUTE_FULL_METRICS,
)


def generate_load_dataset_fn(inp, out, state, load_subset):

    def load_dataset(path: str, chosen_dataset: str) -> dict[Any, Any]:
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
                chosen_dataset = path.split("/")[-1].replace(".csv", "")
                avail_datasets = [chosen_dataset]
                has_multiple_datasets = False
                gr.Info(f"Data successfully loaded from csv file '{path}'.", duration=3)
            else:
                dfs, _, _ = ageval.analysis.data_loader.load_experiments_from_multirun(
                    path
                )

                # set dataset to use (if single dataset, that one will be chosen)
                avail_datasets = list(dfs.keys())
                if not chosen_dataset in avail_datasets:
                    gr.Error(
                        f"Dataset {chosen_dataset} not in avail_dataset {avail_datasets}."
                    )
                    chosen_dataset = avail_datasets[0]

                if len(avail_datasets) > 1:
                    has_multiple_datasets = True
                else:
                    has_multiple_datasets = False

                df = dfs[chosen_dataset]

                gr.Info(
                    f"Data successfully loaded from multirun experiment results on dataset '{chosen_dataset}' from logging path '{path}'.",
                    duration=3,
                )
            groundtruth_col_str = (
                DEFAULT_GROUNDTRUTH_COL
                if DEFAULT_GROUNDTRUTH_COL in list(df.columns)
                else "(None)"
            )
            annotator_1_col_str = (
                DEFAULT_ANNOTATOR_1_COL
                if DEFAULT_ANNOTATOR_1_COL in list(df.columns)
                else "(None)"
            )
            return {
                state["data"]: df,
                **load_subset(
                    df,
                    0,
                    "All",
                    groundtruth_col_str,
                    annotator_1_col_str=annotator_1_col_str,
                    annotator_2_col_str="(None)",
                    other_cols_list=[],
                ),
                inp["dataset"]: gr.Dropdown(
                    choices=avail_datasets,
                    interactive=has_multiple_datasets,
                    value=chosen_dataset,
                    visible=True,
                ),
                inp["cols_group"]: gr.Group(visible=True),
                inp["data_nav_group"]: gr.Group(visible=True),
                out["container"]: gr.Column(visible=True),
                inp["truth_col"]: gr.Dropdown(
                    choices=["(None)"] + list(df.columns),
                    interactive=True,
                    value=groundtruth_col_str,
                ),
                inp["ann1_col"]: gr.Dropdown(
                    choices=["(None)"] + list(df.columns),
                    interactive=True,
                    value=annotator_1_col_str,
                ),
                inp["ann2_col"]: gr.Dropdown(
                    choices=["(None)"] + list(df.columns),
                    interactive=True,
                    value="(None)",
                ),
                inp["other_cols"]: gr.Dropdown(
                    choices=list(df.columns),
                    interactive=True,
                    value=[],
                ),
            }
        except Exception:
            raise gr.Error("Loading data failed.", duration=3)

    return load_dataset


def generate_change_subset_options_fn(inp, update_shown_datapoint):

    def change_subset_options(
        current_index,
        data_df,
        subset_df,
        groundtruth_col_str,
        annotator_1_col_str,
        annotator_2_col_str,
        other_cols_list,
    ):
        available_options = []

        if groundtruth_col_str != "(None)":
            available_options += ["Invalid (truth)"]

        if annotator_1_col_str != "(None)":
            if groundtruth_col_str != "(None)":
                available_options += [
                    "Agreed (truth, annotator 1)",
                    "Disagreed (truth, annotator 1)",
                    "Invalid (annotator 1)",
                ]

        if annotator_2_col_str != "(None)":
            if groundtruth_col_str != "(None)":
                available_options += [
                    "Agreed (truth, annotator 2)",
                    "Disagreed (truth, annotator 2)",
                    "Invalid (annotator 2)",
                ]
            if annotator_1_col_str != "(None)":
                available_options += [
                    "Agreed (annotator 1 & 2)",
                    "Disagreed (annotator 1 & 2)",
                    "Annotator 1 better than 2",
                    "Annotator 2 better than 1",
                ]

        available_options = ["All"] + sorted(available_options)

        if COMPUTE_FULL_METRICS:
            for i, option in enumerate(available_options):
                df = get_subset_from_df(
                    subset_str=option,
                    df=data_df,
                    truth=groundtruth_col_str,
                    ann1=annotator_1_col_str,
                    ann2=annotator_2_col_str,
                )
                available_options[i] = (
                    f"{option}  | {len(df)} ({100*len(df)/len(data_df):.1f}%)"
                )

        return {
            inp["subset"]: gr.Dropdown(
                choices=available_options, value=available_options[0]
            ),
            **update_shown_datapoint(
                index=current_index,
                subset_df=subset_df,
                groundtruth_col_str=groundtruth_col_str,
                annotator_1_col_str=annotator_1_col_str,
                annotator_2_col_str=annotator_2_col_str,
                other_cols_list=other_cols_list,
            ),
        }

    return change_subset_options


def generate_load_subset_fn(inp, state, update_shown_datapoint):

    def load_subset(
        df,
        index,
        subset_str,
        groundtruth_col_str,
        annotator_1_col_str,
        annotator_2_col_str,
        other_cols_list,
    ):
        try:
            df = get_subset_from_df(
                subset_str=subset_str,
                df=df,
                truth=groundtruth_col_str,
                ann1=annotator_1_col_str,
                ann2=annotator_2_col_str,
            )

            if len(df) == 0:
                raise gr.Error(
                    "Attempted to load empty subset. Pick different subset to see data points."
                )

            current_index = min(index, len(df) - 1)
            new_index = gr.Slider(
                minimum=0,
                maximum=len(df) - 1,
                step=1,
                value=current_index,
                label="Selected datapoint",
                info=f"within subset of size {len(df)}",
            )

            return {
                state["subset"]: df,
                inp["index"]: new_index,
                **update_shown_datapoint(
                    index=current_index,
                    subset_df=df,
                    groundtruth_col_str=groundtruth_col_str,
                    annotator_1_col_str=annotator_1_col_str,
                    annotator_2_col_str=annotator_2_col_str,
                    other_cols_list=other_cols_list,
                ),
            }
        except gr.exceptions.Error as e:
            raise e
        except Exception:
            raise gr.Error("Error when loading subset. Loading data set may help.")

    return load_subset


def convert_annotation_to_num(text: str) -> dict:
    """Convert annotation column entry to float value."""

    return_dict = {
        "mean": None,  # numerical annotation mean
        "invalid": None,  # num invalid annotation
        "original": None,  # original annotation string
    }
    mean = None
    valid = 0
    invalid = 0

    if text.startswith("[") and text.endswith("]"):
        # multiple values in list
        annotation_list = ast.literal_eval(text)
        num_list = []
        for ann in annotation_list:
            if ann in ALLOWED_ANNOTATION_VALUES:
                num_list.append(0 if ann == "text_a" else 1)
                valid += 1
            else:
                invalid += 1
        if num_list:
            mean = sum(num_list) / len(num_list)
            assert valid == len(num_list)
    else:
        if text in ALLOWED_ANNOTATION_VALUES:
            mean = 0 if text == "text_a" else 1
            valid += 1
        else:
            invalid += 1

    return {"mean": mean, "valid": valid, "invalid": invalid, "original": text}


def get_match(df, col1, col2, agreed=True, only_allowed_values=True):
    if only_allowed_values:
        df = df.loc[df[col1].isin(ALLOWED_ANNOTATION_VALUES)]
        df = df.loc[df[col2].isin(ALLOWED_ANNOTATION_VALUES)]
    if agreed:
        return df[df[col1] == df[col2]]
    else:
        return df[df[col1] != df[col2]]


def get_non_allowed_col_values(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.loc[~df[col].isin(ALLOWED_ANNOTATION_VALUES)]


def get_subset_from_df(
    subset_str: str, df: pd.DataFrame, truth: str, ann1: str, ann2: str
):
    subset_str = subset_str.split("|")[0].strip()

    if subset_str == "All":
        pass
    elif subset_str == "Agreed (truth, annotator 1)":
        df = get_match(df, truth, ann1)
    elif subset_str == "Agreed (truth, annotator 2)":
        df = get_match(df, truth, ann2)
    elif subset_str == "Disagreed (truth, annotator 1)":
        df = get_match(df, truth, ann1, agreed=False)
    elif subset_str == "Disagreed (truth, annotator 2)":
        df = get_match(df, truth, ann2, agreed=False)
    elif subset_str == "Invalid (truth)":
        df = get_non_allowed_col_values(df, truth)
    elif subset_str == "Invalid (annotator 1)":
        df = get_non_allowed_col_values(df, ann1)
    elif subset_str == "Invalid (annotator 2)":
        df = get_non_allowed_col_values(df, ann2)
    elif subset_str == "Agreed (annotator 1 & 2)":
        df = get_match(df, ann1, ann2)
    elif subset_str == "Disagreed (annotator 1 & 2)":
        df = get_match(df, ann1, ann2, agreed=False)
    elif subset_str == "Annotator 1 better than 2":
        df = get_match(df, truth, ann1)
        if len(df) > 0:
            df = get_match(df, ann1, ann2, agreed=False, only_allowed_values=False)
    elif subset_str == "Annotator 2 better than 1":
        df = get_match(df, truth, ann2)
        if len(df) > 0:
            df = get_match(df, ann1, ann2, agreed=False, only_allowed_values=False)
    else:
        gr.Warning(f"Subset '{subset_str}' could not be loaded.")

    return df
