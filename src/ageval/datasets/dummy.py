"""
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

Core dataset module.
"""

from dataclasses import dataclass


def get_test_pairwise_data():
    return [
        {
            "text_a": "hello",
            "text_b": "world",
            "preference": "text_a",  # one of text_a, text_b, tie
        },
    ]


def get_test_single_data():
    return [
        {
            "text": "hello world",
            "score": 0.1,
        },
    ]
