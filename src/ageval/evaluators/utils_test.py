# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from dataclasses import asdict

from ageval.experiments.config import PromptConfig, ExpConfig
from ageval.evaluators.utils import get_dataclass_from_dict


def test_getting_dataclass_from_dict() -> None:
    """Test basic functionality of getting dataclass from dict."""

    for data_cls in [PromptConfig, ExpConfig]:
        assert data_cls() == get_dataclass_from_dict(data_cls, asdict(data_cls()))
