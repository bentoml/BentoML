import os
import typing as t

import cloudpickle

from ..types import GenericDictType, PathType
from ..utils import init_docstrings, returns_docstrings
from . import (
    _MT,
    LOAD_INIT_DOCS,
    SAVE_INIT_DOCS,
    SAVE_RETURNS_DOCS,
    check_load,
    check_save,
)

LOAD_RETURNS_DOCS = """\
Returns:
	model instance of a given frameworks after unpickle
"""


@init_docstrings(LOAD_INIT_DOCS)
@returns_docstrings(LOAD_RETURNS_DOCS)
@check_load
def load(name: str) -> t.Any:
    with open(os.path.join(path, f"{SAVE_NAMESPACE}{PICKLE_EXTENSION}"), "rb") as inf:
        return cloudpickle.load(inf)


@init_docstrings(SAVE_INIT_DOCS)
@returns_docstrings(SAVE_RETURNS_DOCS)
@check_save
def save(self, name: str, model: "_MT") -> None:
    with open(os.path.join(path, f"{SAVE_NAMESPACE}{PICKLE_EXTENSION}"), "wb") as inf:
        cloudpickle.dump(self._model, inf)
