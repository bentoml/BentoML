

import logging
from typing import Optional

""" Logging utilities. """
log_levels = ...
_default_log_level = ...
def get_logger(name: Optional[str] = ...) -> logging.Logger:
    """Return a logger with the specified name.
    This function is not supposed to be directly accessed by library users.
    """
    ...

def get_verbosity() -> int:
    """Return the current level for the HuggingFace Hub's root logger.
    Returns:
        Logging level, e.g., ``huggingface_hub.logging.DEBUG`` and ``huggingface_hub.logging.INFO``.
    .. note::
        HuggingFace Hub has following logging levels:
        - ``huggingface_hub.logging.CRITICAL``, ``huggingface_hub.logging.FATAL``
        - ``huggingface_hub.logging.ERROR``
        - ``huggingface_hub.logging.WARNING``, ``huggingface_hub.logging.WARN``
        - ``huggingface_hub.logging.INFO``
        - ``huggingface_hub.logging.DEBUG``
    """
    ...

def set_verbosity(verbosity: int) -> None:
    """Set the level for the HuggingFace Hub's root logger.
    Args:
        verbosity:
            Logging level, e.g., ``huggingface_hub.logging.DEBUG`` and ``huggingface_hub.logging.INFO``.
    """
    ...

def set_verbosity_info(): # -> None:
    ...

def set_verbosity_warning(): # -> None:
    ...

def set_verbosity_debug(): # -> None:
    ...

def set_verbosity_error(): # -> None:
    ...

def disable_propagation() -> None:
    """Disable propagation of the library log outputs.
    Note that log propagation is disabled by default.
    """
    ...

def enable_propagation() -> None:
    """Enable propagation of the library log outputs.
    Please disable the HuggingFace Hub's default handler to prevent double logging if the root logger has
    been configured.
    """
    ...

