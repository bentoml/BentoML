from contextlib import contextmanager

"""Global configuration for XGBoost"""

def config_doc(
    *, header=..., extra_note=..., parameters=..., returns=..., see_also=...
):  # -> (func: Unknown) -> (*args: Unknown, **kwargs: Unknown) -> Unknown:
    """Decorator to format docstring for config functions.

    Parameters
    ----------
    header: str
        An introducion to the function
    extra_note: str
        Additional notes
    parameters: str
        Parameters of the function
    returns: str
        Return value
    see_also: str
        Related functions
    """
    ...

@config_doc(
    header="""
    Set global configuration.
    """,
    parameters="""
    Parameters
    ----------
    new_config: Dict[str, Any]
        Keyword arguments representing the parameters and their values
            """,
)
def set_config(**new_config): ...
@config_doc(
    header="""
    Get current values of the global configuration.
    """,
    returns="""
    Returns
    -------
    args: Dict[str, Any]
        The list of global parameters and their values
            """,
)
def get_config(): ...
@contextmanager
@config_doc(
    header="""
    Context manager for global XGBoost configuration.
    """,
    parameters="""
    Parameters
    ----------
    new_config: Dict[str, Any]
        Keyword arguments representing the parameters and their values
            """,
    extra_note="""
    .. note::

        All settings, not just those presently modified, will be returned to their
        previous values when the context manager is exited. This is not thread-safe.
            """,
    see_also="""
    See Also
    --------
    set_config: Set global XGBoost configuration
    get_config: Get current values of the global configuration
            """,
)
def config_context(**new_config): ...
