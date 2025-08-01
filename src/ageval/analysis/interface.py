# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import gradio as gr

from ageval.analysis.constants import (
    DEFAULT_PATH,
    MAX_NUM_ADDITIONAL_COLS,
    TITLE,
    TITLE_EMOJI,
)


def generate():
    gr.Markdown(f"# {TITLE_EMOJI} {TITLE}")
    with gr.Row():
        input_column = gr.Column(scale=1, variant="panel")
        output_column = gr.Column(visible=False, scale=2, variant="panel")

    inp = _create_input_column_parts(input_column)
    out = _create_output_column_parts(output_column)

    return inp, out


def _create_input_column_parts(input_column) -> dict:
    """Create input column parts.

    Returns a dict with all parts."""

    # User input column
    inp = {"container": input_column}
    with input_column:
        inp["data_load_group"] = gr.Tab(label="Data loading")
        with inp["data_load_group"]:
            with gr.Group():
                inp["path"] = gr.Textbox(
                    label="Data path",
                    placeholder="Path",
                    value=DEFAULT_PATH,
                    info=(
                        "Path to a csv file with annotator results. "
                        "The csv must have columns named 'text_a', "
                        "'text_b' and 'prompt'. Each annotator's preferences "
                        "should be in a separate column, with values 'text_a' "
                        "or 'text_b'."
                    ),
                )
                inp["load_data"] = gr.Button("Load")
            inp["dataset"] = gr.Dropdown(label="Dataset selection", visible=False)
            inp["data_load_group_members"] = [
                inp["path"],
                inp["load_data"],
                inp["dataset"],
            ]

        inp["data_nav_group"] = gr.Tab(visible=False, label="Navigation")
        with inp["data_nav_group"]:
            with gr.Group():
                inp["index"] = gr.Slider(
                    label="Selected datapoint",
                    minimum=0,
                    maximum=50,
                    step=1,
                    value=0,
                )
                with gr.Row():
                    inp["prev"] = gr.Button("←")
                    inp["next"] = gr.Button("→")

            inp["subset"] = gr.Dropdown(
                label="Subset of annotations to show",
                info=(
                    "Numbers on right are number of datapoints and percentage "
                    "of total number of datapoints (in brackets). "
                    "'Agreed' and 'Disagreed' cases are reported only "
                    "where both columns have valid preferences ('text_a'"
                    " or 'text_b' string). 'A better than B' cases are datapoints"
                    " where A agrees with truth and B disagrees or is invalid."
                ),
            )

            # add all to list for convenience
            inp["data_nav_group_members"] = [
                inp["subset"],
                inp["index"],
                inp["prev"],
                inp["next"],
            ]
            inp["cols_group"] = gr.Group(visible=False)
            with inp["cols_group"]:
                inp["truth_col"] = gr.Dropdown(label="Ground-truth labels")
                inp["ann1_col"] = gr.Dropdown(label="Annotator 1")
                inp["ann2_col"] = gr.Dropdown(label="Annotator 2")

                inp["other_cols"] = gr.Dropdown(
                    label="Other columns to show",
                    multiselect=True,
                    max_choices=MAX_NUM_ADDITIONAL_COLS,
                    visible=True,
                )
                inp["cols_group_members"] = [
                    inp["truth_col"],
                    inp["ann1_col"],
                    inp["ann2_col"],
                    inp["other_cols"],
                ]

    return inp


def _create_output_column_parts(output_column) -> dict:
    # Data output column
    out = {"container": output_column}
    with output_column:
        with gr.Group():
            out["prompt"] = gr.Textbox(label="Prompt")
            with gr.Row():
                out["text_a"] = gr.Textbox(label="🅰️ Response")
                out["text_b"] = gr.Textbox(label="🅱️ Response")
        out["truth_col"] = gr.Textbox(label="Ground-truth annotation")
        out["ann1_col"] = gr.Textbox(label="Annotator 1")
        out["ann2_col"] = gr.Textbox(label="Annotator 2")
        out["other_cols_container"] = gr.Accordion("Additional columns", open=True)
        with out["other_cols_container"]:
            out["other_cols_list"] = [
                gr.Textbox(visible=False) for i in range(MAX_NUM_ADDITIONAL_COLS)
            ]
        with gr.Accordion("Plain-text view (for copying)", open=False):
            out["plaintext"] = gr.Code(label="Data", value="None")

        out["per_datapoint_members"] = [
            out["prompt"],
            out["text_a"],
            out["text_b"],
            out["truth_col"],
            out["ann1_col"],
            out["ann2_col"],
            out["other_cols_container"],
            *out["other_cols_list"],
            out["plaintext"],
        ]

    return out
