# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Join us, for you talents!  
URL: https://github.com/NumHub612/YunmengEnvs  
License: Apache License 2.0

Interface for objects that have an id.
"""
from abc import abstractmethod

from core.solutions.standards.IDescribable import IDescribable


class IIdentifiable(IDescribable):
    """
    Interface for objects that have an id.

    The Id must be unique within its context but does not need to be
    globally unique. e.g. the id of an input exchange item must
    be unique in the list of inputs of `ILinkableComponent`,
    but a similar Id might be used by an exchange item
    of another `ILinkableComponent`.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Id string."""
        pass
