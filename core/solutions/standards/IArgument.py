# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for arguments.
"""
from core.solutions.standards.IIdentifiable import IIdentifiable

from dataclasses import dataclass
from typing import Any


@dataclass
class IArgument(IIdentifiable):
    """Class for providing arguments."""

    # The type of the value of the argument.
    value_type: type = None

    # Whether the argument is optional.
    optional: bool = False

    # Whether the value property can be edited.
    readonly: bool = False

    # The current value of the argument.
    value: Any = None

    # The default value of the argument.
    default_value: Any = None

    # Possible allowed values.
    possible_values: list = None
