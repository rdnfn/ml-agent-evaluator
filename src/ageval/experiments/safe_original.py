"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Experimental tools for reproducing SAFE experiments.
"""

import json
import pathlib
import multiprocessing
from dataclasses import dataclass, field
import hydra
from hydra.core.config_store import ConfigStore
from loguru import logger

import ageval.datasets.longfact
import ageval.evaluators.safe.original
import ageval.utils

ageval.utils.setup_logger()

ageval.evaluators.safe.original.apply_all_patches()

DATA_DIR = pathlib.Path("notebooks/data/")


def load_model_data(model_name: str, num_tokens: int, path: pathlib.Path | str):
    model_path = (
        path / f"longfact_responses_{num_tokens}tok_{model_name.replace('/','-')}.json"
    )
    with open(model_path, "r") as f:
        data = json.load(f)

    logger.info(
        f"Loaded LongFact responses (model: {model_name}, max tokens: {num_tokens})"
    )

    return data


def run_safe_on_responses(
    model_names: list[str],
    num_responses: int,
    num_tokens: int,
    path: pathlib.Path | str,
    num_workers: int,
):
    logger.info("Starting SAFE algorithm")
    if num_workers > 1:
        pool: multiprocessing.Pool = multiprocessing.Pool(num_workers)

    for model_name in model_names:
        data = load_model_data(
            model_name=model_name,
            num_tokens=num_tokens,
            path=path,
        )[:num_responses]
        for i, model_response_dict in enumerate(data):
            save_path = (
                DATA_DIR
                / f"results_{num_tokens}tok/result_n{i:03d}_{model_name.replace('/','-')}.json"
            )

            prompt = model_response_dict["prompt"]
            response = model_response_dict["response"]

            if num_workers > 1:
                logger.info(f"Launching worker for response {i} for model {model_name}")
                p = pool.apply_async(
                    run_safe_on_single_response,
                    (prompt, response, save_path),
                    # callback=lambda value: logger.info(f"Success ({value})"),
                    error_callback=lambda value: logger.error(f"Error ({value})"),
                )
            else:
                run_safe_on_single_response(prompt, response, save_path)

    if num_workers > 1:
        pool.close()
        logger.info("Waiting for all processes to finish.")
        pool.join()
        logger.info("All processes and experiments finished.")


def run_safe_on_single_response(prompt, response, save_path):
    if save_path.exists():
        logger.info(f"Skipping processing of response, file exist already: {save_path}")
    else:
        result_dict = ageval.evaluators.safe.original.algorithm(
            prompt=prompt,
            response=response,
            algorithm_model="openai/gpt-3.5-turbo",
        )
        with open(save_path, "w") as f:
            json.dump(result_dict, f, indent=4)
        logger.info(f"SAFE experiment run complete. Saved to: {save_path}")


@dataclass
class ExpConfig:
    models: list[str] = field(
        default_factory=lambda: [
            "openai/gpt-3.5-turbo-0125",
            "openai/gpt-4o-2024-05-13",
            "openai/gpt-4-0125-preview",
        ]
    )  # models to evaluate
    num_responses: int = 10  # number of responses to process (apply SAFE to)
    path: pathlib.Path | str = field(
        default_factory=lambda: DATA_DIR
    )  # path to directory with responses
    num_tokens: int = (
        1024  # number of max token length of response (used for finding response files)
    )
    num_workers: int = 1


cs = ConfigStore.instance()
cs.store(name="config", node=ExpConfig)


@hydra.main(version_base=None, config_name="config")
def run_experiment(cfg: ExpConfig) -> None:
    run_safe_on_responses(
        model_names=cfg.models,
        num_responses=cfg.num_responses,
        num_tokens=cfg.num_tokens,
        path=cfg.path,
        num_workers=cfg.num_workers,
    )


if __name__ == "__main__":
    run_experiment()
