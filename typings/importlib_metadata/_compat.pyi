

__all__ = ['install', 'NullFinder', 'PyPy_repr', 'Protocol']
def install(cls):
    """
    Class decorator for installation on sys.meta_path.

    Adds the backport DistributionFinder to sys.meta_path and
    attempts to disable the finder functionality of the stdlib
    DistributionFinder.
    """
    ...

def disable_stdlib_finder(): # -> None:
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
    def find_spec(*args, **kwargs): # -> None:
        ...
    
    find_module = ...


class PyPy_repr:
    """
    Override repr for EntryPoint objects on PyPy to avoid __iter__ access.
    Ref #97, #102.
    """
    affected = ...
    def __compat_repr__(self): # -> str:
        ...
    
    if affected:
        __repr__ = ...


