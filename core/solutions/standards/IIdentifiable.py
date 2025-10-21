# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

To provide an id for an entity.
"""
from core.solutions.standards.IDescribable import IDescribable

from dataclasses import dataclass


@dataclass
class IIdentifiable(IDescribable):
    """
    To provide an id for an entity.

    The Id must be unique within its context but does not need to be
    globally unique. e.g. the id of an input exchange item must
    be unique in the list of inputs of `ILinkableComponent`,
    but a similar Id might be used by an exchange item
    of another `ILinkableComponent`.
    """

    id: str = ""
