from enum import Enum
from typing import Optional


class LinkableComponentStatusChangeEventArgs:
    def __init__(self):
        self._linkable_component: Optional[ILinkableComponent] = None
        self._old_status: Optional[LinkableComponentStatus] = None
        self._new_status: Optional[LinkableComponentStatus] = None
        self._messages: str = ""

    def get_linkable_component(self) -> Optional[ILinkableComponent]:
        return self._linkable_component

    def set_linkable_component(self, obj: ILinkableComponent) -> None:
        self._linkable_component = obj

    def get_old_status(self) -> Optional[LinkableComponentStatus]:
        return self._old_status

    def set_old_status(self, value: LinkableComponentStatus) -> None:
        self._old_status = value

    def get_new_status(self) -> Optional[LinkableComponentStatus]:
        return self._new_status

    def set_new_status(self, value: LinkableComponentStatus) -> None:
        self._new_status = value

    def get_messages(self) -> str:
        return self._messages

    def set_messages(self, msg: str) -> None:
        self._messages = msg
