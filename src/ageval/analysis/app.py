"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Module with App for inspecting annotations.

Use `gradio <path-to-module>` to run.
"""

import argparse
import gradio as gr

import ageval.analysis.interface
import ageval.analysis.callbacks
from ageval.analysis.constants import TITLE, DEFAULT_PATH


with gr.Blocks(analytics_enabled=False, title=TITLE, theme="default") as demo:

    state = {
        "data": gr.State(None),
        "subset": gr.State(None),
    }

    # generate the main interface (user inputs and data outputs)
    inp, out = ageval.analysis.interface.generate()

    # generate callback functions triggered by interface
    callbacks = ageval.analysis.callbacks.generate(inp=inp, out=out, state=state)

    # add callbacks to components so that they can be triggered
    ageval.analysis.callbacks.apply_to_components(
        inp=inp, out=out, state=state, callbacks=callbacks
    )


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        "-p",
        help="Path to experiment data (either csv or multirun dir)",
        default=DEFAULT_PATH,
    )
    args = parser.parse_args()
    inp["path"].value = args.path
    demo.launch()


if __name__ == "__main__":
    run()
