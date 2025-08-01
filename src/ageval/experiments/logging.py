"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Tools to help track and log experiment results.
"""

import pathlib
import json
import pandas as pd
from PIL import Image
from loguru import logger

import ageval.experiments.metrics
import ageval.analysis.data_loader
import ageval.analysis.plotting


def save_results(
    df: pd.DataFrame,
    metrics: dict,
    save_dir: str | pathlib.Path = None,
    filename_base: str = "results_unclassified",
):
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    main_result_path = save_dir / f"{filename_base}_01_full_results.csv"
    df.to_csv(main_result_path, index_label="index")

    metrics_path = save_dir / f"{filename_base}_02_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)


def analyze_combined_results(
    multirun_path: str | pathlib.Path | None,
    save_path: str | pathlib.Path | None = None,
    additional_results: pd.DataFrame | None = None,
    num_seeds: int | None = None,
    dataset_name: str | None = None,
    dataset_len: int | None = None,
    models_to_hide: list[str] | None = None,
) -> None:
    """Analyze all saved results in multirun path and save the overall metrics.

    Results are saved to `multirun_path/overall_results`.
    This function saves tables and plots with all runs combined.

    Args:
        multirun_path (str | pathlib.Path | None): path or comma-separate path list (as string).
        save_path (str | pathlib.Path | None, optional): path to save new plots and tables to. Defaults to None,
            which means the multirun_path is used.
        additional_results (pd.DataFrame | None, optional): dataframe with additional results, outside of the multirunpath.
            Defaults to None. Useful for example to add human results
        num_seeds (int | None, optional): Number of seeds that were used in the exeriment. Defaults to None. If None
            the number of seeds will be automatically detected.
        dataset_name (str | None, optional): Name of the dataset, put at the top of the plots. Defaults to None.
            If None, will be automatically detected.
        dataset_len (int | None, optional): Manually add dataset length. Defaults to None, meaning automatic detection.
        models_to_hide (list | None, optional): Any model name to hide. Defaults to None. See generated CSV
            files to determine the possible names (and naming conventions).
    """
    if save_path is None:
        save_path = pathlib.Path(multirun_path)
    else:
        save_path = pathlib.Path(save_path)
    results_path: pathlib.Path = save_path / "overall_results"
    results_path.mkdir(exist_ok=True, parents=True)

    if multirun_path is not None:
        annotation_dfs, metric_dfs, results_dict = (
            ageval.analysis.data_loader.load_experiments_from_multirun(multirun_path)
        )

        try:
            annotation_dfs["Average"], metric_dfs["Average"] = add_average_dfs(
                annotation_dfs=annotation_dfs, metric_dfs=metric_dfs
            )
        except AssertionError as e:
            # if there for example are not the same number of seeds across datasets
            # skip the average plotting
            logger.info("Could not compute average results, skipping this computation.")
    else:
        annotation_dfs = {dataset_name: None}
        metric_dfs = {dataset_name: additional_results}

    for dataset_path, ann_df in annotation_dfs.items():
        dataset_name = "_".join(dataset_path.split(".csv")[0].split("/")[-1:])

        metric_df = metric_dfs[dataset_path]

        if additional_results is not None:
            metric_df = pd.concat([metric_df, additional_results])

        if ann_df is not None:
            ann_df.to_csv(results_path / f"{dataset_name}-combined_annotations.csv")
            num_datapairs = len(ann_df)
        else:
            num_datapairs = dataset_len

        metric_df.to_csv(results_path / f"{dataset_name}-metrics.csv")

        ageval.analysis.plotting.plot_runs(
            df=metric_df,
            title=f"{dataset_name} ({num_datapairs} datapoints)",
            file_name=f"{dataset_name}-agreement_plot",
            save_dir=results_path,
            group_baseline_agent=True,
            num_seeds=num_seeds,
            models_to_hide=models_to_hide,
        )
        ageval.analysis.plotting.plot_runs(
            df=metric_df,
            title=f"{dataset_name} ({num_datapairs} datapoints)",
            file_name=f"{dataset_name}-agreement_plot_incl_invalid",
            save_dir=results_path,
            metrics=["Agreed (%)", "Not agreed (%)", "Not avail. (%)"],
            colors_per_metric=["#ACE6AB", "#E5A298", "#8EB2B8"],
            add_legend=True,
            legend_kwargs=dict(bbox_to_anchor=(1.0, 1.05)),
            num_seeds=num_seeds,
            models_to_hide=models_to_hide,
        )

    generate_combined_plots(
        save_path=results_path,
        glob_pattern="*invalid.png",
        file_name="all_results_combined_plot.png",
    )


def generate_combined_plots(
    save_path: str | pathlib.Path,
    glob_pattern="*.png",
    file_name="combined_plot.png",
) -> None:
    """Generate and save combined png, combining existing pngs vertically."""

    save_path = pathlib.Path(save_path)
    image_paths = sorted(list(save_path.glob(glob_pattern)))

    assert len(image_paths) > 0, "Not a single image found."

    images = [Image.open(x) for x in image_paths]
    widths, heights = zip(*(image.size for image in images))
    max_width, overall_height = max(widths), sum(heights)

    combined_image = Image.new("RGB", (max_width, overall_height))

    y_offset = 0
    for image in images:
        combined_image.paste(image, (0, y_offset))
        y_offset += image.size[1]

    combined_image.save(save_path / file_name)


def add_average_dfs(
    annotation_dfs: dict[pd.DataFrame],
    metric_dfs: dict[pd.DataFrame],
) -> None:

    # for annotations, we simply concatenate all (as one row is simply one pairwise comparison)
    average_annotation_df = pd.concat(annotation_dfs.values())

    # for metrics, we do a weighted average (weighting each dataset equal)
    # over each seed (as if we had run same number of seeds over entire dataset
    # with weighted results)
    average_metrics_df = get_average_metric_df(metric_dfs=metric_dfs)

    return average_annotation_df, average_metrics_df


def get_average_metric_df(metric_dfs: dict[pd.DataFrame]) -> pd.DataFrame:
    initial_df = list(metric_dfs.values())[0].sort_values(["model"]).copy()
    models = initial_df.model.unique()
    num_datasets = len(metric_dfs)
    ncols = list(initial_df.select_dtypes("number").columns)

    average_df = pd.DataFrame(columns=initial_df.columns)
    for model in models:
        model_average_df = None
        for i, (data_name, df) in enumerate(metric_dfs.items()):
            df = df[df["model"] == model][["model", *ncols]].reset_index()

            # check same number of trials for each dataset
            assert len(df) == len(
                initial_df[initial_df["model"] == model]
            ), f"Problem: {len(df)} vs {len(initial_df[initial_df['model'] == model])} ({data_name})"

            if model_average_df is None:
                model_average_df = df.copy()
            else:
                model_average_df[ncols] += df[ncols]
        average_df = pd.concat([average_df, model_average_df])

    average_df[ncols] = average_df[ncols] / num_datasets

    assert (average_df["Agreement rate"] < 1).all()

    return average_df
