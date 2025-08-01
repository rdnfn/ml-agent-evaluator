# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import os
import copy
from dataclasses import asdict
import pandas as pd
from pandas.io.json._normalize import nested_to_record
import wandb
from loguru import logger
from omegaconf import OmegaConf

from ageval.experiments.config import ExpConfig


def init_wandb_for_row(row: pd.Series) -> None:
    text_a = row["text_a"]
    text_b = row["text_b"]
    wandb_project = row["wandb_project"]
    wandb_entity = row["wandb_entity"]
    wandb_group = row["wandb_group"]
    index = row.name
    prompt = row.get("prompt", None)
    kwargs = {"text_a": text_a, "text_b": text_b, "prompt": prompt}

    if wandb_project is not None:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            group=wandb_group,
        )
        wandb.log(
            {
                "data": {
                    "index": index,
                    "human_pref_text": row["preferred_text"],
                    **kwargs,
                },
            }
        )


def finish_wandb_for_row(row: pd.Series, eval_result: dict, agreement: str) -> None:
    wandb_project = row["wandb_project"]
    if wandb_project is not None:
        wandb.log({"eval_result": eval_result, "agreement": agreement})
        wandb.finish()


def check_if_run_exists(project: str, entity: str, config: dict):
    config = copy.deepcopy(config)
    # remove allow rerun config
    config["wandb"].pop("allow_rerun")

    checkable_config = nested_to_record({"config": config}, sep=".")
    filter_list = {
        "$and": [
            {key: value} for key, value in checkable_config.items() if value is not None
        ]
    }
    path = f"{entity}/{project}"
    logger.info(f"Checking for runs with identical config on wandb: '{path}'.")
    run_list = list(wandb.Api().runs(path=path, filters=filter_list))
    if run_list:
        logger.warning(f"Found previous run with same config on wandb:\n{config}")
        return True
    else:
        logger.info(
            f"No previous run with same config found on wandb.\n{config}\n{filter_list}"
        )
        return False


def get_loggable_config(cfg: ExpConfig) -> dict:
    logged_config = asdict(OmegaConf.to_object(cfg))
    logged_config["added"] = {}
    model = cfg.evaluator_kwargs.model
    logged_config["added"]["model_name_without_task_id"] = model.split("@")[0]
    logged_config["added"]["model_name_only"] = model.split("/")[1].split("@")[0]
    logged_config["added"]["model_provider"] = model.split("/")[0]

    if model.startswith("apple"):
        if "@" in model:
            task_id = model.split("@")[1]
        else:
            task_id = os.environ["TORI_TASK_ID"]
        logged_config["added"]["tori_task_id"] = task_id

    return logged_config


def init_main_wandb_run(cfg: ExpConfig, group: str) -> bool:

    if cfg.wandb.disabled:
        # note that due to subprocesses need to
        # set environment variable.
        os.environ["WANDB_MODE"] = "disabled"

    config = get_loggable_config(cfg)

    run_exists = (
        check_if_run_exists(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=config,
        )
        if not cfg.wandb.disabled
        else False
    )

    if run_exists:
        if not cfg.wandb.allow_rerun:
            logger.error(
                "Run already exists on wandb, skipping experiment. "
                "Delete (failed) run on wandb if exists or "
                "change wandb.allow_rerun parameter to forces rerun."
            )
            return False
        else:
            logger.warning(
                "Rerunning experiment that already exists. "
                "Change wandb.allow_rerun parameter to prevent rerun."
            )

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=config,
        group=group,
    )
    return True
