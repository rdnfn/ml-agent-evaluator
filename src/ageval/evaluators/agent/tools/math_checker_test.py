# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from ageval.evaluators.agent.tools.math_checker import ToolMathChecker
import pytest


@pytest.mark.api
def test_run_online():

    from ageval.utils import setup_api_keys

    setup_api_keys()

    tool = ToolMathChecker
    tool_result = tool.run(
        text="2 ^ 128 / 4 ^ 128 + 2",
        prompt="Create a math equation that looks hard, but will end up being equal to 2.",
        model_name="",
    )
    assert tool_result
