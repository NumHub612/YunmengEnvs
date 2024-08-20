# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunMengEnvs Project Contributors. Join us !
License: Apache License 2.0

Interfaces for objects that have an id.
"""
from abc import abstractmethod

from .IDescribable import IDescribable


class IIdentifiable(IDescribable):
    @abstractmethod
    def get_id(self) -> str:
        """
        Gets id string.

        The Id must be unique within its context but does not need to be globally
        unique. e.g. the id of an input exchange item must be unique in the list
        of inputs of `ILinkableComponent`, but a similar Id might be used by an exchange
        item of another `ILinkableComponent`.
        """
        pass
