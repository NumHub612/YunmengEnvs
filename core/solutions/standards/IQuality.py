# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for qualitative data.
"""
from core.solutions.standards.IValueDefinition import IValueDefinition
from core.solutions.standards.ICategory import ICategory

from abc import abstractmethod
from typing import List


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

    @abstractmethod
    def get_categories(self) -> List[ICategory]:
        """
        Gets a list of the possible `ICategory` allowed.
        """
        pass

    @abstractmethod
    def is_ordered(self) -> bool:
        """
        Checks if this `IQuality` is defined by an ordered set or not.
        """
        pass
