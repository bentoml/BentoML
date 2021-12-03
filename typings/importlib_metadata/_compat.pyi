"""
This type stub file was generated by pyright.
"""

__all__ = ["install", "NullFinder", "Protocol"]

def install(cls):
    """
    Class decorator for installation on sys.meta_path.

    Adds the backport DistributionFinder to sys.meta_path and
    attempts to disable the finder functionality of the stdlib
    DistributionFinder.
    """
    ...

def disable_stdlib_finder():  # -> None:
    """
    Give the backport primacy for discovering path-based distributions
    by monkey-patching the stdlib O_O.

    See #91 for more background for rationale on this sketchy
    behavior.
    """
    ...

class NullFinder:
    """
    A "Finder" (aka "MetaClassFinder") that never finds any modules,
    but may find distributions.
    """

    @staticmethod
    def find_spec(*args, **kwargs): ...
    find_module = ...

def pypy_partial(val):
    """
    Adjust for variable stacklevel on partial under PyPy.

    Workaround for #327.
    """
    ...
