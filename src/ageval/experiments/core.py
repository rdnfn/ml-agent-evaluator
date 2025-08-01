# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from typing import Type
import uuid
import os
import random
from loguru import logger
import pandas as pd
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from tqdm.contrib.concurrent import process_map
from omegaconf import DictConfig

import ageval.experiments.logging
import ageval.experiments.metrics
import ageval.analysis.plotting
import ageval.datasets.core
import ageval.utils
import ageval.experiments.config
from ageval.experiments.wandb_utils import (
    init_wandb_for_row,
    finish_wandb_for_row,
    init_main_wandb_run,
)
from ageval.evaluators.utils import get_evaluator_cls_from_str


def get_evaluator_annotation_allowing_failure(
    row: pd.Series,
) -> tuple[str, dict[str, any], str]:

    ageval.utils.setup_logger(level=row["log_level"])
    ageval.utils.setup_experiment_log_files(dir=row["save_dir"])

    try:
        init_wandb_for_row(row=row)
        preferred_text, eval_result, agreement = get_evaluator_annotation(row=row)
        finish_wandb_for_row(row=row, eval_result=eval_result, agreement=agreement)
        return preferred_text, eval_result, agreement
    except Exception as e:
        error_msg = f"Error: fact checking failed completely:\n{e}"
        logger.exception(error_msg)
        return (
            "Error generating evaluator annotation.",
            "Error",
            "No evaluator annotation available",
        )


def get_evaluator_annotation(
    row: pd.Series,
) -> tuple[str, dict[str, any], str]:
    # if called from iterrows iterable, split up tuple input
    if isinstance(row, tuple):
        _, row = row

    randomize_position = row["randomize_position"]

    if randomize_position:
        do_flip = random.choice([True, False])
    else:
        do_flip = False

    if do_flip:
        text_a, text_b = row["text_b"], row["text_a"]
    else:
        text_a, text_b = row["text_a"], row["text_b"]

    evaluator_cls = row["evaluator_cls"]
    evaluator_kwargs = row["evaluator_kwargs"]

    evaluator = evaluator_cls(**evaluator_kwargs)
    prompt = row.get("prompt", None)
    kwargs = {"text_a": text_a, "text_b": text_b, "prompt": prompt}

    eval_result = evaluator.evaluate(**kwargs)

    pref_text = eval_result["preferred_text"]

    if not do_flip:
        eval_result["text_position_flipped"] = False
    else:
        # flip back
        eval_result["text_position_flipped"] = True
        if pref_text == "text_a":
            pref_text = "text_b"
        elif pref_text == "text_b":
            pref_text = "text_a"

    agreement = check_agreement(
        pref_original=row["preferred_text"],
        pref_evaluator=pref_text,
    )

    return pref_text, eval_result, agreement


def check_agreement(pref_original: str, pref_evaluator: str) -> str:
    assert isinstance(pref_original, str)
    assert pref_original in ["text_a", "text_b"]

    if pref_evaluator in ["text_a", "text_b"]:
        if pref_original == pref_evaluator:
            return "Agreed"
        else:
            return "Not agreed"
    else:
        return "No evaluator annotation available"


def run_evaluator_on_pairwise_data(
    df: pd.DataFrame,
    evaluator_cls: Type | str,
    evaluator_kwargs: dict,
    randomize_position: bool,
    parallelize: bool = True,
    num_workers: int = None,
    show_progress_bar: bool = True,
    verbose: int = 1,  # See verbose in https://nalepae.github.io/pandarallel/user_guide/
    save_dir: str | None = "./ageval_results",
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    log_level: str | None = "INFO",
) -> dict:

    if save_dir is not None and "multirun" in save_dir:
        run_num = save_dir.split("/")[-1]
    else:
        run_num = 0

    logger.info(f"Test starting: running evaluator on pairwise data (run #{run_num}).")
    logger.info(f"Saving config and results to: {save_dir}")

    df = df.copy()

    if evaluator_kwargs is None:
        evaluator_kwargs = {}
    if num_workers is None:
        num_workers = min(32, os.cpu_count() + 4)

    df["wandb_project"] = wandb_project
    df["wandb_entity"] = wandb_entity
    df["wandb_group"] = wandb_group
    df["log_level"] = log_level
    df["evaluator_cls"] = evaluator_cls
    df["randomize_position"] = randomize_position
    df["save_dir"] = str(save_dir)
    df["evaluator_kwargs"] = [evaluator_kwargs] * len(df)

    if not parallelize:
        logger.info("Evaluating sequentially (without parallel processing).")
        df[["evaluator_pref_text", "evaluator_result", "evaluator_agreement"]] = (
            df.apply(
                get_evaluator_annotation_allowing_failure, axis=1, result_type="expand"
            )
        )
    else:
        logger.info(f"Evaluating in parallel (with {num_workers} workers).")
        results = process_map(
            get_evaluator_annotation_allowing_failure,
            [row for _, row in df.iterrows()],
            max_workers=num_workers,
        )
        df[["evaluator_pref_text", "evaluator_result", "evaluator_agreement"]] = results
    metrics = ageval.experiments.metrics.get_standard_metrics(results_df=df)

    if save_dir is not None:
        filename_base = f"{evaluator_kwargs['model'].replace('/','-')}"
        if "safe" in str(evaluator_cls):
            filename_base += "-safe"

        # save results for this single run
        ageval.experiments.logging.save_results(
            df=df,
            metrics=metrics,
            save_dir=save_dir,
            filename_base=filename_base,
        )

        # get path of enclosing dir
        multirun_path = "/".join(save_dir.split("/")[:-1])
        # analyze all results (including other runs) and save them
        ageval.experiments.logging.analyze_combined_results(multirun_path)

        logger.info(f"Results saved to: {save_dir}")
        logger.info(
            f"To view and analyze results in app, run:\nageval-app -p {multirun_path}"
        )

    logger.info("End: completed applying evaluator to pairwise data.")

    return {"metrics": metrics, "annotated_df": df}


@hydra.main(config_name="config", config_path=None, version_base="1.3")
def run_experiment(cfg: DictConfig) -> None:

    experiment_path = HydraConfig.get().runtime.output_dir

    ageval.utils.setup_logger(level=cfg.log_level)
    ageval.utils.setup_experiment_log_files(experiment_path)
    ageval.utils.setup_api_keys()

    wandb_group = str(uuid.uuid4())[:8]
    wandb_init_successful = init_main_wandb_run(cfg=cfg, group=wandb_group)

    if wandb_init_successful:
        df = ageval.datasets.core.load(cfg)
        wandb.log({"data_set_len": len(df)})

        evaluator_cls = get_evaluator_cls_from_str(cfg.evaluator_cls)

        results = run_evaluator_on_pairwise_data(
            df=df,
            evaluator_cls=evaluator_cls,
            evaluator_kwargs=dict(cfg.evaluator_kwargs),
            randomize_position=cfg.randomize_position,
            parallelize=cfg.parallelize,
            num_workers=cfg.num_workers,
            show_progress_bar=cfg.show_progress_bar,
            verbose=cfg.verbose,
            save_dir=experiment_path,
            wandb_project=cfg.wandb.project,
            wandb_entity=cfg.wandb.entity,
            wandb_group=wandb_group,
            log_level=cfg.log_level,
        )

        wandb.log(results["metrics"])
        wandb.finish()


if __name__ == "__main__":
    run_experiment()
