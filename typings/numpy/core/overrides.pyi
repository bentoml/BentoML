"""Implementation of __array_function__ overrides from NEP-18."""
ARRAY_FUNCTION_ENABLED = ...
array_function_like_doc = ...

def set_array_function_like_doc(public_api): ...

ArgSpec = ...

def verify_matching_signatures(implementation, dispatcher):
    """Verify that a dispatcher function has the right signature."""
    ...

def set_module(module):
    """Decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module('numpy')
        def example():
            pass

        assert example.__module__ == 'numpy'
    """
    ...

_wrapped_func_source = ...

def array_function_dispatch(dispatcher, module=..., verify=..., docs_from_dispatcher=...):
    """Decorator for adding dispatch with the __array_function__ protocol.

    See NEP-18 for example usage.

    Parameters
    ----------
    dispatcher : callable
        Function that when called like ``dispatcher(*args, **kwargs)`` with
        arguments from the NumPy function call returns an iterable of
        array-like arguments to check for ``__array_function__``.
    module : str, optional
        __module__ attribute to set on new function, e.g., ``module='numpy'``.
        By default, module is copied from the decorated function.
    verify : bool, optional
        If True, verify the that the signature of the dispatcher and decorated
        function signatures match exactly: all required and optional arguments
        should appear in order with the same names, but the default values for
        all optional arguments should be ``None``. Only disable verification
        if the dispatcher's signature needs to deviate for some particular
        reason, e.g., because the function has a signature like
        ``func(*args, **kwargs)``.
    docs_from_dispatcher : bool, optional
        If True, copy docs from the dispatcher function onto the dispatched
        function, rather than from the implementation. This is useful for
        functions defined in C, which otherwise don't have docstrings.

    Returns
    -------
    Function suitable for decorating the implementation of a NumPy function.
    """
    ...

def array_function_from_dispatcher(
    implementation, module=..., verify=..., docs_from_dispatcher=...
):
    """Like array_function_dispatcher, but with function arguments flipped."""
    ...
