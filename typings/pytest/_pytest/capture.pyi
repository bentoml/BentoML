
import contextlib
import functools
import io
from typing import (
    TYPE_CHECKING,
    Any,
    AnyStr,
    Generator,
    Generic,
    Iterator,
    TextIO,
    Tuple,
    Union,
)

from _pytest.compat import final
from _pytest.config import Config, hookimpl
from _pytest.config.argparsing import Parser
from _pytest.fixtures import SubRequest, fixture
from _pytest.nodes import Collector, Item
from typing_extensions import Literal

"""Per-test stdout/stderr capturing mechanism."""
if TYPE_CHECKING:
    _CaptureMethod = Literal["fd", "sys", "no", "tee-sys"]
def pytest_addoption(parser: Parser) -> None:
    ...

@hookimpl(hookwrapper=True)
def pytest_load_initial_conftests(early_config: Config): # -> Generator[None, None, None]:
    ...

class EncodedFile(io.TextIOWrapper):
    __slots__ = ...
    @property
    def name(self) -> str:
        ...
    
    @property
    def mode(self) -> str:
        ...
    


class CaptureIO(io.TextIOWrapper):
    def __init__(self) -> None:
        ...
    
    def getvalue(self) -> str:
        ...
    


class TeeCaptureIO(CaptureIO):
    def __init__(self, other: TextIO) -> None:
        ...
    
    def write(self, s: str) -> int:
        ...
    


class DontReadFromInput:
    encoding = ...
    def read(self, *args): # -> NoReturn:
        ...
    
    readline = ...
    readlines = ...
    __next__ = ...
    def __iter__(self): # -> Self@DontReadFromInput:
        ...
    
    def fileno(self) -> int:
        ...
    
    def isatty(self) -> bool:
        ...
    
    def close(self) -> None:
        ...
    
    @property
    def buffer(self): # -> Self@DontReadFromInput:
        ...
    


patchsysdict = ...
class NoCapture:
    EMPTY_BUFFER = ...
    __init__ = ...


class SysCaptureBinary:
    EMPTY_BUFFER = ...
    def __init__(self, fd: int, tmpfile=..., *, tee: bool = ...) -> None:
        ...
    
    def repr(self, class_name: str) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def start(self) -> None:
        ...
    
    def snap(self): # -> bytes:
        ...
    
    def done(self) -> None:
        ...
    
    def suspend(self) -> None:
        ...
    
    def resume(self) -> None:
        ...
    
    def writeorg(self, data) -> None:
        ...
    


class SysCapture(SysCaptureBinary):
    EMPTY_BUFFER = ...
    def snap(self): # -> str:
        ...
    
    def writeorg(self, data): # -> None:
        ...
    


class FDCaptureBinary:
    """Capture IO to/from a given OS-level file descriptor.

    snap() produces `bytes`.
    """
    EMPTY_BUFFER = ...
    def __init__(self, targetfd: int) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def start(self) -> None:
        """Start capturing on targetfd using memorized tmpfile."""
        ...
    
    def snap(self): # -> bytes:
        ...
    
    def done(self) -> None:
        """Stop capturing, restore streams, return original capture file,
        seeked to position zero."""
        ...
    
    def suspend(self) -> None:
        ...
    
    def resume(self) -> None:
        ...
    
    def writeorg(self, data): # -> None:
        """Write to original file descriptor."""
        ...
    


class FDCapture(FDCaptureBinary):
    """Capture IO to/from a given OS-level file descriptor.

    snap() produces text.
    """
    EMPTY_BUFFER = ...
    def snap(self): # -> str:
        ...
    
    def writeorg(self, data): # -> None:
        """Write to original file descriptor."""
        ...
    


@final
@functools.total_ordering
class CaptureResult(Generic[AnyStr]):
    """The result of :method:`CaptureFixture.readouterr`."""
    __slots__ = ...
    def __init__(self, out: AnyStr, err: AnyStr) -> None:
        ...
    
    def __len__(self) -> int:
        ...
    
    def __iter__(self) -> Iterator[AnyStr]:
        ...
    
    def __getitem__(self, item: int) -> AnyStr:
        ...
    
    def count(self, value: AnyStr) -> int:
        ...
    
    def index(self, value) -> int:
        ...
    
    def __eq__(self, other: object) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    
    def __lt__(self, other: object) -> bool:
        ...
    
    def __repr__(self) -> str:
        ...
    


class MultiCapture(Generic[AnyStr]):
    _state = ...
    _in_suspended = ...
    def __init__(self, in_, out, err) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def start_capturing(self) -> None:
        ...
    
    def pop_outerr_to_orig(self) -> Tuple[AnyStr, AnyStr]:
        """Pop current snapshot out/err capture and flush to orig streams."""
        ...
    
    def suspend_capturing(self, in_: bool = ...) -> None:
        ...
    
    def resume_capturing(self) -> None:
        ...
    
    def stop_capturing(self) -> None:
        """Stop capturing and reset capturing streams."""
        ...
    
    def is_started(self) -> bool:
        """Whether actively capturing -- not suspended or stopped."""
        ...
    
    def readouterr(self) -> CaptureResult[AnyStr]:
        ...
    


class CaptureManager:
    """The capture plugin.

    Manages that the appropriate capture method is enabled/disabled during
    collection and each test phase (setup, call, teardown). After each of
    those points, the captured output is obtained and attached to the
    collection/runtest report.

    There are two levels of capture:

    * global: enabled by default and can be suppressed by the ``-s``
      option. This is always enabled/disabled during collection and each test
      phase.

    * fixture: when a test function or one of its fixture depend on the
      ``capsys`` or ``capfd`` fixtures. In this case special handling is
      needed to ensure the fixtures take precedence over the global capture.
    """
    def __init__(self, method: _CaptureMethod) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def is_capturing(self) -> Union[str, bool]:
        ...
    
    def is_globally_capturing(self) -> bool:
        ...
    
    def start_global_capturing(self) -> None:
        ...
    
    def stop_global_capturing(self) -> None:
        ...
    
    def resume_global_capture(self) -> None:
        ...
    
    def suspend_global_capture(self, in_: bool = ...) -> None:
        ...
    
    def suspend(self, in_: bool = ...) -> None:
        ...
    
    def resume(self) -> None:
        ...
    
    def read_global_capture(self) -> CaptureResult[str]:
        ...
    
    def set_fixture(self, capture_fixture: CaptureFixture[Any]) -> None:
        ...
    
    def unset_fixture(self) -> None:
        ...
    
    def activate_fixture(self) -> None:
        """If the current item is using ``capsys`` or ``capfd``, activate
        them so they take precedence over the global capture."""
        ...
    
    def deactivate_fixture(self) -> None:
        """Deactivate the ``capsys`` or ``capfd`` fixture of this item, if any."""
        ...
    
    def suspend_fixture(self) -> None:
        ...
    
    def resume_fixture(self) -> None:
        ...
    
    @contextlib.contextmanager
    def global_and_fixture_disabled(self) -> Generator[None, None, None]:
        """Context manager to temporarily disable global and current fixture capturing."""
        ...
    
    @contextlib.contextmanager
    def item_capture(self, when: str, item: Item) -> Generator[None, None, None]:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector: Collector): # -> Generator[None, None, None]:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_runtest_setup(self, item: Item) -> Generator[None, None, None]:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item: Item) -> Generator[None, None, None]:
        ...
    
    @hookimpl(hookwrapper=True)
    def pytest_runtest_teardown(self, item: Item) -> Generator[None, None, None]:
        ...
    
    @hookimpl(tryfirst=True)
    def pytest_keyboard_interrupt(self) -> None:
        ...
    
    @hookimpl(tryfirst=True)
    def pytest_internalerror(self) -> None:
        ...
    


class CaptureFixture(Generic[AnyStr]):
    """Object returned by the :fixture:`capsys`, :fixture:`capsysbinary`,
    :fixture:`capfd` and :fixture:`capfdbinary` fixtures."""
    def __init__(self, captureclass, request: SubRequest, *, _ispytest: bool = ...) -> None:
        ...
    
    def close(self) -> None:
        ...
    
    def readouterr(self) -> CaptureResult[AnyStr]:
        """Read and return the captured output so far, resetting the internal
        buffer.

        :returns:
            The captured content as a namedtuple with ``out`` and ``err``
            string attributes.
        """
        ...
    
    @contextlib.contextmanager
    def disabled(self) -> Generator[None, None, None]:
        """Temporarily disable capturing while inside the ``with`` block."""
        ...
    


@fixture
def capsys(request: SubRequest) -> Generator[CaptureFixture[str], None, None]:
    """Enable text capturing of writes to ``sys.stdout`` and ``sys.stderr``.

    The captured output is made available via ``capsys.readouterr()`` method
    calls, which return a ``(out, err)`` namedtuple.
    ``out`` and ``err`` will be ``text`` objects.
    """
    ...

@fixture
def capsysbinary(request: SubRequest) -> Generator[CaptureFixture[bytes], None, None]:
    """Enable bytes capturing of writes to ``sys.stdout`` and ``sys.stderr``.

    The captured output is made available via ``capsysbinary.readouterr()``
    method calls, which return a ``(out, err)`` namedtuple.
    ``out`` and ``err`` will be ``bytes`` objects.
    """
    ...

@fixture
def capfd(request: SubRequest) -> Generator[CaptureFixture[str], None, None]:
    """Enable text capturing of writes to file descriptors ``1`` and ``2``.

    The captured output is made available via ``capfd.readouterr()`` method
    calls, which return a ``(out, err)`` namedtuple.
    ``out`` and ``err`` will be ``text`` objects.
    """
    ...

@fixture
def capfdbinary(request: SubRequest) -> Generator[CaptureFixture[bytes], None, None]:
    """Enable bytes capturing of writes to file descriptors ``1`` and ``2``.

    The captured output is made available via ``capfd.readouterr()`` method
    calls, which return a ``(out, err)`` namedtuple.
    ``out`` and ``err`` will be ``byte`` objects.
    """
    ...

