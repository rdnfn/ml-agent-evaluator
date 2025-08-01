"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Main experiment configuration dataclasses

Used by Hydra to generate the arguments in the ageval experiment CLI.
"""

from dataclasses import dataclass, field
from typing import Any
from hydra.core.config_store import ConfigStore

from ageval.experiments.prompts import PromptConfig

HYDRA_SETTINGS = {
    "run": {"dir": "exp/logs/singlerun/${now:%Y-%m-%d}_${now:%H-%M-%S}"},
    "sweep": {"dir": "exp/logs/multirun/${now:%Y-%m-%d}_${now:%H-%M-%S}"},
}

# Updated according to
# https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/automatic_schema_matching/
CONFIG_NAME = "base_config"


@dataclass
class WandbConfig:
    disabled: bool = True
    project: str = "default_project"
    entity: str = "arduin"
    allow_rerun: bool = (
        False  # whether to allow rerunning the same experiment with same config
    )


@dataclass
class EvaluatorConfig:
    model: str = "openai/gpt-3.5-turbo-0125"
    model_kwargs: dict | None = field(default_factory=dict)
    max_tokens: int = 512
    additional_instructions: str | None = None
    max_facts: int | None = None
    tools: list[str] | None = field(
        default_factory=lambda: ["fact_check", "code_interpreter", "math_checker"]
    )
    tools_to_always_run: list[str] | None = None
    prompt_type: str | None = None
    baseline: str | None = "ageval.evaluators.basic.BasicPairwiseEvaluator"
    baseline_kwargs: dict | None = None
    agent_act_threshold: int = 5  # no harm config
    prompts: PromptConfig = field(default_factory=PromptConfig)


@dataclass
class ExpConfig:
    # data
    data_path: str | None = None
    data_name: str | None = None

    # evaluator
    evaluator_cls: str = "ageval.evaluators.basic.BasicPairwiseEvaluator"
    evaluator_kwargs: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    # experiment details
    randomize_position: bool = (
        True  # whether to automatically randomly switch text a/b position
        # (helps avoid position bias, but introduces noise)
    )

    # parallelization
    parallelize: bool = True
    num_workers: int | None = None

    # logging
    show_progress_bar: bool = False
    verbose: int = 1  # See verbose in https://nalepae.github.io/pandarallel/user_guide/
    save_dir: str | None = "./ageval_results"
    wandb: WandbConfig = field(default_factory=WandbConfig)
    log_level: str = "INFO"  # altenatively set "DEBUG"

    # other configs
    dummy: int | None = None  # used to run multiple seeds


cs = ConfigStore.instance()
cs.store(name=CONFIG_NAME, node=ExpConfig)
