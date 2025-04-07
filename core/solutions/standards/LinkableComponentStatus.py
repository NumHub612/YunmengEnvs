# -*- encoding: utf-8 -*-
"""
Copyright (C) 2024, The YunmengEnvs Contributors. Welcome aboard YunmengEnvs!

Interface for standardized linkable component status.
"""
from enum import Enum


class LinkableComponentStatus(Enum):
    """
    Enumerates the possible statuses that a linkable component can be in.
    """

    CREATED = 1
    """
    The linkable component instance has just been created.
    This status must and will be followed by `INITIALIZING`.
    """

    INITIALIZING = 2
    """
    The linkable component is initializing itself.
    This status will end in a status change to `INITIALIZED` or `FAILED`.
    """

    INITIALIZED = 3
    """
    The linkable component has successfully initialized itself.
    The connections between its inputs/outputs and 
    those of other components can be established.
    """

    VALIDATING = 4
    """
    This status will end in a status change to `VALID` or `INVALID`.
    After links between the component's inputs/outputs and those of other components
    have been established, the component is validating whether its required
    input will be available when it updates itself, whether indeed it will be
    able to provide the required output during this update.
    """

    VALID = 5
    """
    The component is in a valid state.
    When updating itself its required input will be available, and it will be able
    to provide the required output.
    """

    WAITING_FOR_DATA = 6
    """
    The component wants to update itself, but is not yet able to perform the
    actual computation, because it is still waiting for input data
    from other components.
    """

    INVALID = 7
    """
    The component is in an invalid state.
    When updating itself not all required input will be available, and/or it will
    not be able to provide the required output. After the user has modified
    the connections between the component's inputs/outputs and those of other
    components, the `VALIDATING` state can be entered again.
    """

    PREPARING = 8
    """
    The component is preparing itself for the first `GetValues()` call.
    This state will end in a status change to `UPDATED` or `FAILED`.
    """

    UPDATING = 9
    """
    The component is updating itself. It receives all required input data
    from other components, and is now performing the actual computation.
    This state will end in a status change to `UPDATED`, `DONE` or `FAILED`.
    """

    UPDATED = 10
    """
    The component has successfully updated itself.
    """

    DONE = 11
    """
    The last update process that component performed was the final one.
    A next call to the `Update()` method will leave 
    the component's internal state unchanged.
    """

    FINISHING = 12
    """
    The ILinkableComponent was requested to perform the actions to be
    performed before it will either be disposed or re-initialized again.
    Typical actions would be writing the final result files, close all open files,
    free memory, etc. When all required actions have been performed,
    the status switches to `CREATED` when re-initialization is possible.
    The status switches to `FINISHED` when component is to be disposed.
    """

    FINISHED = 13
    """
    The ILinkableComponent has successfully performed its finalization actions.
    Re-initialization of the component instance isn't possible and shouldn't be
    attempted. Instead the instance should be disposed, e.g.
    through the garbage collection mechanism.
    """

    FAILED = 14
    """
    The component was requested to perform the actions to be performed before
    it will either be disposed or re-initialized again.
    The linkable component has failed initialize itself, failed to prepare
    itself for computation, or failed to complete its update process.
    Typical actions would be writing the final result files, close all open files,
    free memory, etc. When all required actions have been performed,
    the status switches back to `CREATED` if the component supports being
    re-initialized. If it can't be re-initialized, it can be released from memory.
    """
