# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import pytest

from ageval.analysis.data_processing import convert_annotation_to_num

testdata = [
    (
        '["text_a","text_b"]',
        {"mean": 0.5, "valid": 2, "invalid": 0, "original": '["text_a","text_b"]'},
    ),
    (
        '["text_a","text_a"]',
        {"mean": 0, "valid": 2, "invalid": 0, "original": '["text_a","text_a"]'},
    ),
    (
        '["text_b","text_b"]',
        {"mean": 1, "valid": 2, "invalid": 0, "original": '["text_b","text_b"]'},
    ),
    (
        '["text_c","text_b"]',
        {"mean": 1, "valid": 1, "invalid": 1, "original": '["text_c","text_b"]'},
    ),
    (
        "text_b",
        {"mean": 1, "valid": 1, "invalid": 0, "original": "text_b"},
    ),
    (
        "text_c",
        {"mean": None, "valid": 0, "invalid": 1, "original": "text_c"},
    ),
]


@pytest.mark.parametrize("input,expected", testdata)
def test_convert_annotation_to_num(input, expected):
    assert convert_annotation_to_num(input) == expected
