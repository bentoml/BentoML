
from pathlib import Path
from typing import Optional

import attr
import py
from _pytest.compat import final
from _pytest.config import Config
from _pytest.fixtures import FixtureRequest, fixture

"""Support for providing temporary directories to test functions."""
@final
@attr.s(init=False)
class TempPathFactory:
    """Factory for temporary directories under the common base temp directory.

    The base directory can be configured using the ``--basetemp`` option.
    """
    _given_basetemp = ...
    _trace = ...
    _basetemp = ...
    def __init__(self, given_basetemp: Optional[Path], trace, basetemp: Optional[Path] = ..., *, _ispytest: bool = ...) -> None:
        ...
    
    @classmethod
    def from_config(cls, config: Config, *, _ispytest: bool = ...) -> TempPathFactory:
        """Create a factory according to pytest configuration.

        :meta private:
        """
        ...
    
    def mktemp(self, basename: str, numbered: bool = ...) -> Path:
        """Create a new temporary directory managed by the factory.

        :param basename:
            Directory base name, must be a relative path.

        :param numbered:
            If ``True``, ensure the directory is unique by adding a numbered
            suffix greater than any existing one: ``basename="foo-"`` and ``numbered=True``
            means that this function will create directories named ``"foo-0"``,
            ``"foo-1"``, ``"foo-2"`` and so on.

        :returns:
            The path to the new directory.
        """
        ...
    
    def getbasetemp(self) -> Path:
        """Return the base temporary directory, creating it if needed."""
        ...
    


@final
@attr.s(init=False)
class TempdirFactory:
    """Backward comptibility wrapper that implements :class:``py.path.local``
    for :class:``TempPathFactory``."""
    _tmppath_factory = ...
    def __init__(self, tmppath_factory: TempPathFactory, *, _ispytest: bool = ...) -> None:
        ...
    
    def mktemp(self, basename: str, numbered: bool = ...) -> py.path.local:
        """Same as :meth:`TempPathFactory.mktemp`, but returns a ``py.path.local`` object."""
        ...
    
    def getbasetemp(self) -> py.path.local:
        """Backward compat wrapper for ``_tmppath_factory.getbasetemp``."""
        ...
    


def get_user() -> Optional[str]:
    """Return the current user name, or None if getuser() does not work
    in the current environment (see #1010)."""
    ...

def pytest_configure(config: Config) -> None:
    """Create a TempdirFactory and attach it to the config object.

    This is to comply with existing plugins which expect the handler to be
    available at pytest_configure time, but ideally should be moved entirely
    to the tmpdir_factory session fixture.
    """
    ...

@fixture(scope="session")
def tmpdir_factory(request: FixtureRequest) -> TempdirFactory:
    """Return a :class:`_pytest.tmpdir.TempdirFactory` instance for the test session."""
    ...

@fixture(scope="session")
def tmp_path_factory(request: FixtureRequest) -> TempPathFactory:
    """Return a :class:`_pytest.tmpdir.TempPathFactory` instance for the test session."""
    ...

@fixture
def tmpdir(tmp_path: Path) -> py.path.local:
    """Return a temporary directory path object which is unique to each test
    function invocation, created as a sub directory of the base temporary
    directory.

    By default, a new base temporary directory is created each test session,
    and old bases are removed after 3 sessions, to aid in debugging. If
    ``--basetemp`` is used then it is cleared each session. See :ref:`base
    temporary directory`.

    The returned object is a `py.path.local`_ path object.

    .. _`py.path.local`: https://py.readthedocs.io/en/latest/path.html
    """
    ...

@fixture
def tmp_path(request: FixtureRequest, tmp_path_factory: TempPathFactory) -> Path:
    """Return a temporary directory path object which is unique to each test
    function invocation, created as a sub directory of the base temporary
    directory.

    By default, a new base temporary directory is created each test session,
    and old bases are removed after 3 sessions, to aid in debugging. If
    ``--basetemp`` is used then it is cleared each session. See :ref:`base
    temporary directory`.

    The returned object is a :class:`pathlib.Path` object.
    """
    ...

