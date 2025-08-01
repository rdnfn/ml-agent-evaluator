"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Functions for generating a view of data.
"""

import ast
import json
import gradio as gr
from gradio.components.base import Component
import pandas as pd


def get_update_shown_datapoint_fn(out: dict) -> callable:
    """Create a function that can update the components to a new datapoint.

    Args:
        out (dict): dictionary of output components

    Returns:
        callable: function that updates datapoint components
            (to be used as callback during change event of inputs)
    """

    def update_shown_datapoint(
        index: int,
        subset_df: pd.DataFrame,
        groundtruth_col_str: str,
        annotator_1_col_str: str,
        annotator_2_col_str: str,
        other_cols_list: list[str],
    ):
        try:
            if index > len(subset_df) - 1 or -index > len(subset_df):
                raise gr.Error(
                    (
                        "You have hit the limit of your data. "
                        f"Datapoint index ({index}) out-of-bounds for subset of size {len(subset_df)}. "
                        f"Change index to valid value between (incl.) -{len(subset_df)} and {len(subset_df)-1}."
                    )
                )

            row: pd.Series = subset_df.iloc[index]
            annotations = {}

            # getting values and labels for each annotator
            if groundtruth_col_str != "(None)":
                annotations[out["truth_col"]] = (
                    row[groundtruth_col_str],
                    f"Ground-truth annotation ({groundtruth_col_str})",
                )
            else:
                annotations[out["truth_col"]] = (None, None)

            if annotator_1_col_str != "(None)":
                annotations[out["ann1_col"]] = (
                    row[annotator_1_col_str],
                    f"Annotator 1 ({annotator_1_col_str})",
                )
            else:
                annotations[out["ann1_col"]] = (None, None)

            if annotator_2_col_str != "(None)":
                annotations[out["ann2_col"]] = (
                    row[annotator_2_col_str],
                    f"Annotator 2 ({annotator_2_col_str})",
                )
            else:
                annotations[out["ann2_col"]] = (None, None)

            # setting textboxes with values and labels
            annotation_components = {}
            for annotation, (annotation_str, name) in annotations.items():
                if annotation_str == "text_b":
                    annotation_replacement = gr.Textbox(
                        value="🅱️", text_align="right", visible=True, label=name
                    )
                elif annotation_str == "text_a":
                    annotation_replacement = gr.Textbox(
                        value="🅰️", text_align="left", visible=True, label=name
                    )
                elif annotation_str == None:
                    annotation_replacement = gr.Textbox(
                        value="",
                        text_align="left",
                        visible=False,
                        label=name,
                    )
                else:
                    annotation_replacement = gr.Textbox(
                        value=annotation_str,
                        text_align="left",
                        visible=True,
                        label=name,
                    )
                annotation_components[annotation] = annotation_replacement

            # adding additional columns (beyond the standard annotators)
            other_cols_dict = {}
            other_cols_value_dict = {}
            if other_cols_list:
                for i, col in enumerate(other_cols_list):
                    if col in row.keys():
                        value: str = row[col]
                        if (
                            isinstance(value, str)
                            and value.startswith("{")
                            and value.endswith("}")
                        ):
                            value = json.dumps(ast.literal_eval(value), indent=4)

                        other_cols_dict[out["other_cols_list"][i]] = gr.Textbox(
                            label=col, value=value, visible=True
                        )
                        other_cols_value_dict[col] = value
                    else:
                        gr.Warning(
                            f"Row with name {col} not found in columns {row.keys()}. Other columns list: {other_cols_list}."
                        )
            for col in out["other_cols_list"]:
                if col not in other_cols_dict:
                    other_cols_dict[col] = gr.Textbox(
                        label=None, value=None, visible=False
                    )

            other_cols_visible = bool(
                bool(other_cols_list) and len(other_cols_list) > 0
            )

            # define a string representation of all components
            data_box_dict = {
                "index": row["index"],
                "prompt": row["prompt"],
                "text_a": row["text_a"],
                "text_b": row["text_b"],
                annotations[out["truth_col"]][1]: annotations[out["truth_col"]][0],
                annotations[out["ann1_col"]][1]: annotations[out["ann1_col"]][0],
                annotations[out["ann2_col"]][1]: annotations[out["ann2_col"]][0],
                **other_cols_value_dict,
            }

            data_box_str = "\n---\n".join(
                [f"{key}: {value}" for key, value in data_box_dict.items()]
            )
            data_box_str = "```\n" + data_box_str + "\n```"

            # define the dict that is returned to update component
            return_dict = {
                out["prompt"]: row["prompt"],
                out["text_a"]: row["text_a"],
                out["text_b"]: row["text_b"],
                out["other_cols_container"]: gr.Accordion(
                    visible=other_cols_visible,
                ),
                **annotation_components,
                **other_cols_dict,
                out["plaintext"]: data_box_str,
            }

            return return_dict
        except gr.exceptions.Error as e:
            raise e
        except Exception as e:
            if isinstance(subset_df, pd.DataFrame) and len(subset_df) == 0:
                gr.Warning("Empty subset, no data to show.")
            else:
                gr.Warning(
                    (
                        f"Data attempted to be shown:\nIndex: {index}; cols: {groundtruth_col_str}, "
                        f"{annotator_1_col_str}, {annotator_2_col_str}; from df: {str(subset_df)[:100]}..."
                    )
                )
                gr.Warning(
                    f"No data can be shown. Is the data loaded yet? (Error: {e})"
                )
                return {key: "-" for key in out["per_datapoint_members"]}

    return update_shown_datapoint
