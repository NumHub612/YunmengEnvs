# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for qualitative data.
"""
from yunmeng.solutions.standards.IValueDefinition import IValueDefinition
from yunmeng.solutions.standards.ICategory import ICategory

from dataclasses import dataclass


@dataclass
class IQuality(IValueDefinition):
    """
    Class describes qualitative data, where value is specified as one category
    within a number of predefined (possible) categories.

    Qualitative data described items in terms of some quality or
    categorization that may be 'informal' or may use relatively ill-defined
    characteristics such as warmth and flavour. However,
    qualitative data can include well-defined aspects such as gender,
    nationality or commodity type.
    """

    # List of possible categories for this quality.
    categories: list[ICategory] = None

    # Flag indicating if this quality is defined by an ordered set or not.
    ordered: bool = False
