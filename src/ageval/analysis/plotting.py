# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import random
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
import scienceplots

# This setting below prevent a warning regarding the ticks from showing:
# /project-agent-evaluator/env/lib/python3.11/site-packages/matplotlib/ticker.py:2918:
# RuntimeWarning: invalid value encountered in log10 majorstep_no_exponent = 10 ** (np.log10(majorstep) % 1)
np.seterr(invalid="ignore")

models = [
    "anthropic/claude-3-5-sonnet-20240620",
    "anthropic/claude-3-haiku-20240307",
    "openai/gpt-3.5-turbo-0125",
    "openai/gpt-4-0613",
    "openai/gpt-4o-2024-05-13",
]
colors = [colormaps["tab20b"](idx / len(models)) for idx in range(len(models))]
model_color_mapping = {model: color for model, color in zip(models, colors)}

BASELINE_COLOR = "#40C69F"
AGENT_COLOR = "#2E765E"
HUMAN_COLOR = "#8ab1ce"

MODEL_PRETTY_NAMES = {
    # baselines
    "basic_gpt-4o-2024-05-13": "Simplest pick-best baseline (GPT-4o)",
    "basic_gpt-3.5-turbo-0125": "Simplest pick-best baseline (GPT-3.5-Turbo)",
    "rewardbench_gpt-4o-2024-05-13": "RewardBench baseline (GPT-4o)",
    "arenahard_gpt-4o-2024-05-13": "ArenaHard baseline (GPT-4o)",
    "alpacaeval_gpt-4o-2024-05-13": "AlpacaEval 2.0 baseline (GPT-4-Turbo)",
    "alpacaeval_gpt-3.5-turbo-0125": "AlpacaEval 2.0 baseline (GPT-4-Turbo)",
    # agents
    "agent_gpt-4o-2024-05-13_fact_check_code_interpreter_base-basic": "Agent (GPT-4o, base: pick-best)",
    "agent_gpt-4o-2024-05-13_math_checker_base-basic": "Agent (GPT-4o, tools: math check, base: pick-best)",
    "agent_gpt-3.5-turbo-0125_fact_check_code_interpreter_base-basic": "Agent (GPT-3.5-Turbo, base: pick-best)",
    "agent_gpt-4o-2024-05-13_synthetic_base-alpacaeval_gpt-4o-2024-05-13": "Agent (GPT-4o, base: AE 2.0)",
    "agent_gpt-4o-2024-05-13_synthetic_base-arenahard_gpt-4o-2024-05-13": "Agent (GPT-4o, base: ArenaHard)",
    "agent_gpt-4o-2024-05-13_synthetic_base-basic_gpt-4o-2024-05-13": "Agent (GPT-4o, base: pick-best)",
    "agent_gpt-4o-2024-05-13_synthetic_base-rewardbench_gpt-4o-2024-05-13": "Agent (GPT-4o, base: RewardBench)",
}


def get_color(model: str, color_separate_baseline_agent: bool = True) -> str:
    if color_separate_baseline_agent:
        if "agent" in model:
            return AGENT_COLOR
        elif "human" in model.lower():
            return HUMAN_COLOR
        else:
            return BASELINE_COLOR

    color = model_color_mapping.get(model, None)
    if color is None:
        return colormaps["tab20b"](random.random())


def get_pretty_name(model: str) -> str:
    """Get pretty readible name for model"""
    return MODEL_PRETTY_NAMES.get(model, model)


def get_related_runs(models) -> tuple[list[list[str]], list[list]]:
    """Get related runs and group them.

    Make agent and corresponding baseline into a group.
    """

    annotator_names: list = copy.deepcopy(models)
    agents = [name for name in annotator_names if "agent" in name]
    groups = []
    order = []

    # for each agent, find baseline results and group together
    for agent in sorted(agents):
        annotator_names.remove(agent)
        base_description = agent.split("_base-")[-1]
        llm_model = agent.split("_")[1]
        if llm_model not in base_description:
            # llm_model not always present for standard agents names,
            # thus adding
            baseline_name = base_description + "_" + llm_model
        else:
            # for synthetic agents this is already in description
            baseline_name = base_description

        if baseline_name in annotator_names:
            # if baseline exists, group together with agent
            annotator_names.remove(baseline_name)
            group = [baseline_name, agent]
        else:
            # otherwise group agent alone
            group = [agent]

        groups.append(group)
        order += group

    # add prev ungrouped names at end
    for name in annotator_names:
        groups.append([name])
        order.append(name)

    return groups, order


def get_grouped_model_locations(
    model_groups: list[list[str]],
    group_offset: float,
    height: float,
) -> list[float]:
    """Get the vertical location of each model bar."""
    locations = []
    group_index = 0
    model_index = 0
    for group in model_groups:
        for model in group:
            locations.append(group_index * group_offset + model_index * height)
            model_index -= 1
        group_index -= 1

    # make all location values positive
    min_loc = min(locations)
    locations = [loc - min_loc + 10 for loc in locations]

    return locations


def plot_runs(
    df: pd.DataFrame,
    title: str = "Plot",
    file_name: str | None = None,
    save_dir: str = "figures",
    x_label="Agreement with ground-truth annotations (%)",
    fig_height=3,
    fig_width=6,
    metrics=["Agreed (%)"],
    colors_per_metric: list | None = None,
    sort_metric=None,
    add_legend=False,
    legend_kwargs=None,
    group_baseline_agent=False,
    color_separate_baseline_agent=True,
    num_seeds=None,
    models_to_hide=None,
    close_figures=True,
    dark_mode=False,
):

    if sort_metric is None:
        sort_metric = metrics[0]
    if legend_kwargs is None:
        legend_kwargs = {}

    df = df.copy()
    df = df[["model"] + metrics]

    if models_to_hide is not None:
        df = df[~df.model.isin(models_to_hide)]

    counts = {}
    df_mean = df.groupby(["model"]).mean().sort_values(sort_metric, ascending=False)
    df_std = df.groupby(["model"]).std()
    seed_count = df["model"].value_counts()  # count the number of seeds per model
    if (seed_count.iloc[0] == seed_count).all():
        overall_count = seed_count.iloc[0]
    else:
        overall_count = None

    # overwrite if num seeds is set
    if num_seeds is not None:
        overall_count = num_seeds

    models = list(df_mean.index)

    # check grouping (if requested)
    if group_baseline_agent:
        model_groups, ordered_models = get_related_runs(models)

        # change order according to groups
        df_mean = df_mean.reindex(index=ordered_models)
        models = ordered_models
    else:
        # each model is its own group
        model_groups = [[model] for model in models]

    colors = [
        get_color(
            str(model).split(" ")[0],
            color_separate_baseline_agent=color_separate_baseline_agent,
        )
        for model in models
    ]

    height = 0.6  # the height of the bars: can also be len(x) sequence

    # get location of each model
    locations = get_grouped_model_locations(
        model_groups=model_groups, group_offset=height, height=height
    )

    # add std to each metric considered
    for metric in metrics:
        agreement_std = df_std[metric]
        agreement_std.name = f"{metric} std"
        df_plot = df_mean.merge(agreement_std, on="model")
        mean = list(df_plot[metric])
        x_err = list(df_plot[f"{metric} std"])
        counts[metric] = [mean, x_err]

    shown_models = models
    shown_models = [get_pretty_name(model).replace("_", "-") for model in shown_models]

    if overall_count is None:
        # if no overall seed count, add one per model
        updated_model_names = []
        for pretty_name, original_name in zip(shown_models, models):
            if (
                not "human" in pretty_name.lower()
                and not "human" in original_name.lower()
            ):
                updated_model_names.append(
                    pretty_name + f"\n({seed_count.loc[original_name]} seeds)"
                )
            else:
                updated_model_names.append(pretty_name)

        shown_models = updated_model_names
    else:
        # append seed count to title if not None
        title = title.replace(")", f", {overall_count} seeds)")

    plot_figure(
        fig_height=fig_height,
        fig_width=fig_width,
        models=models,
        counts=counts,
        metrics=metrics,
        colors=colors,
        colors_per_metric=colors_per_metric,
        locations=locations,
        height=height,
        shown_models=shown_models,
        add_legend=add_legend,
        title=title,
        legend_kwargs=legend_kwargs,
        x_label=x_label,
        save_dir=save_dir,
        file_name=file_name,
        close=close_figures,
        dark_mode=dark_mode,
    )


def plot_figure(
    fig_height: float,
    fig_width: float,
    models: list[str],
    counts: dict,
    metrics: list[str],
    colors: list[str],
    colors_per_metric: list[str],
    locations: list[float],
    height: float,
    shown_models: list[str],
    add_legend: bool,
    title: str,
    legend_kwargs: dict,
    x_label: str,
    save_dir: str,
    file_name: str,
    close: bool,
    dark_mode: bool,
):
    plt.style.use(["science", "no-latex", "nature"])

    if dark_mode:
        plt.style.use("dark_background")

    fig, ax = plt.subplots()
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    bottom = np.zeros(len(models))

    for i, (count_key, count) in enumerate(counts.items()):

        mean, std = count
        if len(metrics) == 1:
            color = colors
        elif colors_per_metric is not None:
            color = colors_per_metric[i]
        else:
            color = None

        assert len(locations) == len(mean)
        assert all([loc > 0 for loc in locations])

        if dark_mode:
            error_color = "white"
        else:
            error_color = "black"

        p = ax.barh(
            y=locations,
            width=mean,
            height=height,
            color=color,
            # hatch=hatches[i],
            error_kw=dict(ecolor=error_color, lw=1.5),
            xerr=std,
            label=count_key,
            left=bottom,
            align="center",
        )
        bottom += mean

        # add numbers to bars
        ax.bar_label(p, label_type="center", fmt="%.2f", color="black")

    # add model names
    # (note that this creates the warning addressed in line 13)
    ax.set_yticks(locations)
    ax.set_yticklabels(shown_models)
    # remove top and right spine (i.e. white line)
    ax.spines[["right", "top"]].set_visible(False)

    ax.set_title(title)

    if add_legend:
        ax.legend(**legend_kwargs)

    ax.set_xlabel(x_label)
    ax.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        right=False,
        left=True,
        labelbottom=True,
    )

    if file_name is not None:
        plt.savefig(f"{save_dir}/{file_name}.png", dpi=300)

    if close:
        plt.close()
