"""
For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.

Utils for evaluators.
"""

from typing import Type, TypeVar
import importlib
import dataclasses


def get_evaluator_cls_from_str(evaluator_cls: str) -> Type:
    module_str = ".".join(evaluator_cls.split(".")[:-1])
    class_name = evaluator_cls.split(".")[-1]

    module = importlib.import_module(module_str)
    return getattr(module, class_name)


T = TypeVar("T")


def get_dataclass_from_dict(data_cls: Type[T], data_dict: dict) -> T:
    try:
        field_types = {field.name: field.type for field in dataclasses.fields(data_cls)}
        return data_cls(
            **{
                key: get_dataclass_from_dict(field_types[key], data_dict[key])
                for key in data_dict
            }
        )
    except:
        return data_dict
