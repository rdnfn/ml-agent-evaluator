# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import ageval.models.openai
import ageval.utils
import pytest


@pytest.mark.api
def test_assistant_model():
    """Test OpenAI assistant model.

    Requires OpenAI API access."""
    ageval.utils.setup_api_keys()

    assistant = ageval.models.openai.AssistantModelOpenAI()
    answer = assistant.invoke(
        "What is 23454 * 2342. Use code interpreter! Only answer with number, nothing else"
    )
    assert int(answer.content) == 54929268
