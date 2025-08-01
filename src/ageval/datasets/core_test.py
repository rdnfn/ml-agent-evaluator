# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pytest

from ageval.datasets.core import load
from ageval.experiments.config import ExpConfig


def test_data_loading():
    """Test data loader using simple data."""

    with pytest.raises(AssertionError):
        load(cfg=ExpConfig())
    with pytest.raises(AssertionError):
        load(cfg=ExpConfig(data_path="Something", data_name="Something else"))

    load(cfg=ExpConfig(data_path="data/examples/example-truthfulqa.csv"))
    load(cfg=ExpConfig(data_name="example"))
