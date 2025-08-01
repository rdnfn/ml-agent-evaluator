# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from ageval.evaluators.agent.tools.code_interpreter import ToolCodeInterpreter
import pytest


@pytest.mark.api
def test_run_online():

    from ageval.utils import setup_api_keys

    setup_api_keys()

    tool = ToolCodeInterpreter
    tool_result = tool.run(
        text="x = 'Hello world'; print(x)",
        prompt="Write some python that will print 'Hello world'.",
        model_name="",
    )
    assert tool_result
