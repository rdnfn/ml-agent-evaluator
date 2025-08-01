# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pytest

import ageval.evaluators.safe.original


@pytest.fixture
def response_data():
    return "The eiffel tower was built in 1923. It was widely considered as the tallest mountain."


def test_core_safe_functionality_with_llm(response_data):
    ageval.evaluators.safe.original.algorithm(
        prompt="", response=response_data, algorithm_model="openai/gpt-3.5-turbo"
    )
