# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface provides the args for an ExchangeItemValueChanged event.
"""
from yunmeng.solutions.standards.IBaseExchangeItem import IBaseExchangeItem

from dataclasses import dataclass


@dataclass
class ExchangeItemChangeEventArgs:
    """
    To provides the information that will be passed when
    firing an `ExchangeItemValueChanged` event.
    """

    # The exchange item that has been changed.
    exchange_item: IBaseExchangeItem = None

    # The message description of the change.
    message: str = ""
