

import types

"""For compatibility and optional dependencies."""
STRING_TYPES = ...
def py_str(x):
    """convert c string back to python string"""
    ...

def lazy_isinstance(instance, module, name):
    '''Use string representation to identify a type.'''
    ...

class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.
    """
    def __init__(self, local_name, parent_module_globals, name, warning=...) -> None:
        ...
    
    def __getattr__(self, item): # -> Any:
        ...
    
    def __dir__(self): # -> list[str]:
        ...
    


