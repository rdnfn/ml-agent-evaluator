# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import gradio as gr

import ageval.analysis.dataview
import ageval.analysis.data_processing


def generate(inp, out, state):
    update_shown_datapoint = ageval.analysis.dataview.get_update_shown_datapoint_fn(
        out=out
    )
    change_subset_options = (
        ageval.analysis.data_processing.generate_change_subset_options_fn(
            inp=inp, update_shown_datapoint=update_shown_datapoint
        )
    )
    load_subset = ageval.analysis.data_processing.generate_load_subset_fn(
        inp=inp, state=state, update_shown_datapoint=update_shown_datapoint
    )
    load_dataset = ageval.analysis.data_processing.generate_load_dataset_fn(
        inp=inp, out=out, state=state, load_subset=load_subset
    )

    return {
        "update_shown_datapoint": update_shown_datapoint,
        "change_subset_options": change_subset_options,
        "load_subset": load_subset,
        "load_dataset": load_dataset,
    }


def apply_to_components(inp, out, state, callbacks):

    possible_outputs = [*state.values(), *out["per_datapoint_members"]]

    load_data_args = dict(
        fn=callbacks["load_dataset"],
        inputs=[inp["path"], inp["dataset"]],
        outputs=possible_outputs
        + [
            inp["dataset"],
            inp["cols_group"],
            *inp["cols_group_members"],
            inp["data_nav_group"],
            *inp["data_nav_group_members"],
            out["container"],
        ],
    )

    inp["load_data"].click(**load_data_args)
    inp["dataset"].select(**load_data_args)

    inp["prev"].click(
        fn=lambda index_inp_int: {inp["index"]: gr.Slider(value=index_inp_int - 1)},
        inputs=inp["index"],
        outputs=inp["index"],
    )
    inp["next"].click(
        fn=lambda index_inp_int: {inp["index"]: gr.Slider(value=index_inp_int + 1)},
        inputs=inp["index"],
        outputs=inp["index"],
    )

    # components that can trigger a change in the available subsets
    change_subsets_triggers = [inp["truth_col"], inp["ann1_col"], inp["ann2_col"]]
    for component in change_subsets_triggers:
        component.change(
            fn=callbacks["change_subset_options"],
            inputs=[
                inp["index"],
                state["data"],
                state["subset"],
                *inp["cols_group_members"],
            ],
            outputs=possible_outputs + [inp["subset"], inp["index"]],
        )

    inp["subset"].change(  # automatically triggered by change_subset_options
        fn=callbacks["load_subset"],
        inputs=[
            state["data"],
            inp["index"],
            inp["subset"],
            *inp["cols_group_members"],
        ],
        outputs=possible_outputs + [inp["index"]],
    )

    components_just_changing_datapoint = [inp["index"], inp["other_cols"]]
    for comp in components_just_changing_datapoint:
        comp.change(
            fn=callbacks["update_shown_datapoint"],
            inputs=[
                inp["index"],
                state["subset"],
                *inp["cols_group_members"],
            ],
            outputs=possible_outputs,
        )
