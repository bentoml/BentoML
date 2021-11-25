

import logging
from typing import Optional

_lock = ...
_default_handler: Optional[logging.Handler] = ...
log_levels = ...
_default_log_level = ...
def get_log_levels_dict():
    ...

def get_logger(name: Optional[str] = ...) -> logging.Logger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom transformers module.
    """
    ...

def get_verbosity() -> int:
    """
    Return the current level for the 🤗 Transformers's root logger as an int.

    Returns:
        :obj:`int`: The logging level.

    .. note::

        🤗 Transformers has following logging levels:

        - 50: ``transformers.logging.CRITICAL`` or ``transformers.logging.FATAL``
        - 40: ``transformers.logging.ERROR``
        - 30: ``transformers.logging.WARNING`` or ``transformers.logging.WARN``
        - 20: ``transformers.logging.INFO``
        - 10: ``transformers.logging.DEBUG``
    """
    ...

def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the 🤗 Transformers's root logger.

    Args:
        verbosity (:obj:`int`):
            Logging level, e.g., one of:

            - ``transformers.logging.CRITICAL`` or ``transformers.logging.FATAL``
            - ``transformers.logging.ERROR``
            - ``transformers.logging.WARNING`` or ``transformers.logging.WARN``
            - ``transformers.logging.INFO``
            - ``transformers.logging.DEBUG``
    """
    ...

def set_verbosity_info():
    """Set the verbosity to the :obj:`INFO` level."""
    ...

def set_verbosity_warning():
    """Set the verbosity to the :obj:`WARNING` level."""
    ...

def set_verbosity_debug():
    """Set the verbosity to the :obj:`DEBUG` level."""
    ...

def set_verbosity_error():
    """Set the verbosity to the :obj:`ERROR` level."""
    ...

def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""
    ...

def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""
    ...

def add_handler(handler: logging.Handler) -> None:
    """adds a handler to the HuggingFace Transformers's root logger."""
    ...

def remove_handler(handler: logging.Handler) -> None:
    """removes given handler from the HuggingFace Transformers's root logger."""
    ...

def disable_propagation() -> None:
    """
    Disable propagation of the library log outputs. Note that log propagation is disabled by default.
    """
    ...

def enable_propagation() -> None:
    """
    Enable propagation of the library log outputs. Please disable the HuggingFace Transformers's default handler to
    prevent double logging if the root logger has been configured.
    """
    ...

def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:

    ::

        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE

    All handlers currently bound to the root logger are affected by this method.
    """
    ...

def reset_format() -> None:
    """
    Resets the formatting for HuggingFace Transformers's loggers.

    All handlers currently bound to the root logger are affected by this method.
    """
    ...

