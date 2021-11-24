
from contextlib import contextmanager
from typing import Generator, MutableMapping, Optional, Tuple, Union, overload

from _pytest.compat import final
from _pytest.fixtures import fixture

"""Monkeypatching and mocking functionality."""
RE_IMPORT_ERROR_NAME = ...
K = ...
V = ...
@fixture
def monkeypatch() -> Generator[MonkeyPatch, None, None]:
    """A convenient fixture for monkey-patching.

    The fixture provides these methods to modify objects, dictionaries or
    os.environ::

        monkeypatch.setattr(obj, name, value, raising=True)
        monkeypatch.delattr(obj, name, raising=True)
        monkeypatch.setitem(mapping, name, value)
        monkeypatch.delitem(obj, name, raising=True)
        monkeypatch.setenv(name, value, prepend=False)
        monkeypatch.delenv(name, raising=True)
        monkeypatch.syspath_prepend(path)
        monkeypatch.chdir(path)

    All modifications will be undone after the requesting test function or
    fixture has finished. The ``raising`` parameter determines if a KeyError
    or AttributeError will be raised if the set/deletion operation has no target.
    """
    ...

def resolve(name: str) -> object:
    ...

def annotated_getattr(obj: object, name: str, ann: str) -> object:
    ...

def derive_importpath(import_path: str, raising: bool) -> Tuple[str, object]:
    ...

class Notset:
    def __repr__(self) -> str:
        ...
    


notset = ...
@final
class MonkeyPatch:
    """Helper to conveniently monkeypatch attributes/items/environment
    variables/syspath.

    Returned by the :fixture:`monkeypatch` fixture.

    :versionchanged:: 6.2
        Can now also be used directly as `pytest.MonkeyPatch()`, for when
        the fixture is not available. In this case, use
        :meth:`with MonkeyPatch.context() as mp: <context>` or remember to call
        :meth:`undo` explicitly.
    """
    def __init__(self) -> None:
        ...
    
    @classmethod
    @contextmanager
    def context(cls) -> Generator[MonkeyPatch, None, None]:
        """Context manager that returns a new :class:`MonkeyPatch` object
        which undoes any patching done inside the ``with`` block upon exit.

        Example:

        .. code-block:: python

            import functools


            def test_partial(monkeypatch):
                with monkeypatch.context() as m:
                    m.setattr(functools, "partial", 3)

        Useful in situations where it is desired to undo some patches before the test ends,
        such as mocking ``stdlib`` functions that might break pytest itself if mocked (for examples
        of this see `#3290 <https://github.com/pytest-dev/pytest/issues/3290>`_.
        """
        ...
    
    @overload
    def setattr(self, target: str, name: object, value: Notset = ..., raising: bool = ...) -> None:
        ...
    
    @overload
    def setattr(self, target: object, name: str, value: object, raising: bool = ...) -> None:
        ...
    
    def setattr(self, target: Union[str, object], name: Union[object, str], value: object = ..., raising: bool = ...) -> None:
        """Set attribute value on target, memorizing the old value.

        For convenience you can specify a string as ``target`` which
        will be interpreted as a dotted import path, with the last part
        being the attribute name. For example,
        ``monkeypatch.setattr("os.getcwd", lambda: "/")``
        would set the ``getcwd`` function of the ``os`` module.

        Raises AttributeError if the attribute does not exist, unless
        ``raising`` is set to False.
        """
        ...
    
    def delattr(self, target: Union[object, str], name: Union[str, Notset] = ..., raising: bool = ...) -> None:
        """Delete attribute ``name`` from ``target``.

        If no ``name`` is specified and ``target`` is a string
        it will be interpreted as a dotted import path with the
        last part being the attribute name.

        Raises AttributeError it the attribute does not exist, unless
        ``raising`` is set to False.
        """
        ...
    
    def setitem(self, dic: MutableMapping[K, V], name: K, value: V) -> None:
        """Set dictionary entry ``name`` to value."""
        ...
    
    def delitem(self, dic: MutableMapping[K, V], name: K, raising: bool = ...) -> None:
        """Delete ``name`` from dict.

        Raises ``KeyError`` if it doesn't exist, unless ``raising`` is set to
        False.
        """
        ...
    
    def setenv(self, name: str, value: str, prepend: Optional[str] = ...) -> None:
        """Set environment variable ``name`` to ``value``.

        If ``prepend`` is a character, read the current environment variable
        value and prepend the ``value`` adjoined with the ``prepend``
        character.
        """
        ...
    
    def delenv(self, name: str, raising: bool = ...) -> None:
        """Delete ``name`` from the environment.

        Raises ``KeyError`` if it does not exist, unless ``raising`` is set to
        False.
        """
        ...
    
    def syspath_prepend(self, path) -> None:
        """Prepend ``path`` to ``sys.path`` list of import locations."""
        ...
    
    def chdir(self, path) -> None:
        """Change the current working directory to the specified path.

        Path can be a string or a py.path.local object.
        """
        ...
    
    def undo(self) -> None:
        """Undo previous changes.

        This call consumes the undo stack. Calling it a second time has no
        effect unless you do more monkeypatching after the undo call.

        There is generally no need to call `undo()`, since it is
        called automatically during tear-down.

        Note that the same `monkeypatch` fixture is used across a
        single test function invocation. If `monkeypatch` is used both by
        the test function itself and one of the test fixtures,
        calling `undo()` will undo all of the changes made in
        both functions.
        """
        ...
    


