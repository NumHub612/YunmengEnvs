# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for arguments of linkable component status change event.
"""
from yunmeng.solutions.standards.LinkableComponentStatus import LinkableComponentStatus
from yunmeng.solutions.standards.ILinkableComponent import ILinkableComponent

from dataclasses import dataclass


@dataclass
class LinkableComponentStatusChangeEventArgs:
    """Class for arguments of linkable component status change event."""

    linkable_component: ILinkableComponent
    message: str
    old_status: LinkableComponentStatus
    new_status: LinkableComponentStatus
