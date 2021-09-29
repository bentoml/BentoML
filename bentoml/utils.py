import inspect
import typing as t

from ._internal.environment.docker_image import builtin_docker_image
from ._internal.environment.local_py_modules import find_local_py_modules_used
from ._internal.environment.pip_pkg import (
    find_required_pypi_packages,
    lock_pypi_versions,
    with_pip_install_options,
)
from .exceptions import BentoMLException

__all__ = [
    "builtin_docker_image",
    "find_required_pypi_packages",
    "with_pip_install_options",
    "lock_pypi_versions",
    "find_local_py_modules_used",
]

_T = t.TypeVar("_T")

_REQUIRED_DOC_FIELD = ["Args", "Examples", "Returns"]


def docstrings(docs: str):
    def decorator(func: t.Callable[..., _T]):
        func.__doc__ = inspect.cleandoc(docs)
        if not all(i in func.__doc__ for i in _REQUIRED_DOC_FIELD):
            raise BentoMLException(
                f"BentoML docstring requires {', '.join(_REQUIRED_DOC_FIELD)} sections per frameworks module."
            )
        return func

    return decorator
