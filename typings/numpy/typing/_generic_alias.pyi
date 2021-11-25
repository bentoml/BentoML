import sys
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    FrozenSet,
    Iterable,
    List,
    NoReturn,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np

__all__ = ["_GenericAlias", "NDArray"]
_T = ...

class _GenericAlias:
    """A python-based backport of the `types.GenericAlias` class.

    E.g. for ``t = list[int]``, ``t.__origin__`` is ``list`` and
    ``t.__args__`` is ``(int,)``.

    See Also
    --------
    :pep:`585`
        The PEP responsible for introducing `types.GenericAlias`.

    """

    __slots__ = ...
    @property
    def __origin__(self) -> type: ...
    @property
    def __args__(self) -> Tuple[Any, ...]: ...
    @property
    def __parameters__(self) -> Tuple[TypeVar, ...]:
        """Type variables in the ``GenericAlias``."""
        ...
    def __init__(self, origin: type, args: Any) -> None: ...
    @property
    def __call__(self) -> type: ...
    def __reduce__(self: _T) -> Tuple[Type[_T], Tuple[type, Tuple[Any, ...]]]: ...
    def __mro_entries__(self, bases: Iterable[object]) -> Tuple[type]: ...
    def __dir__(self) -> List[str]:
        """Implement ``dir(self)``."""
        ...
    def __hash__(self) -> int:
        """Return ``hash(self)``."""
        ...
    def __instancecheck__(self, obj: object) -> NoReturn:
        """Check if an `obj` is an instance."""
        ...
    def __subclasscheck__(self, cls: type) -> NoReturn:
        """Check if a `cls` is a subclass."""
        ...
    def __repr__(self) -> str:
        """Return ``repr(self)``."""
        ...
    def __getitem__(self: _T, key: Any) -> _T:
        """Return ``self[key]``."""
        ...
    def __eq__(self, value: object) -> bool:
        """Return ``self == value``."""
        ...
    _ATTR_EXCEPTIONS: ClassVar[FrozenSet[str]] = ...
    def __getattribute__(self, name: str) -> Any:
        """Return ``getattr(self, name)``."""
        ...

if sys.version_info >= (3, 9): ...
else:
    _GENERIC_ALIAS_TYPE = ...
ScalarType = ...
if TYPE_CHECKING:
    NDArray = np.ndarray[Any, np.dtype[ScalarType]]
else: ...
