from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager

import attrs
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskID
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn

from ..bento import Bento
from ..models import Model
from ..tag import Tag

if t.TYPE_CHECKING:
    from rich.console import Console
    from rich.console import ConsoleOptions
    from rich.console import RenderResult

FILE_CHUNK_SIZE = 100 * 1024 * 1024  # 100Mb


@attrs.define
class CallbackIOWrapper(t.IO[bytes]):
    file: t.IO[bytes]
    read_cb: t.Callable[[int], None] | None = None
    write_cb: t.Callable[[int], None] | None = None
    start: int | None = None
    end: int | None = None

    def __attrs_post_init__(self) -> None:
        self.reset()

    def reset(self) -> int:
        read = self.tell() - (self.start or 0)
        self.file.seek(self.start or 0, 0)
        return read

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 2 and self.end is not None:
            length = self.file.seek(self.end, 0)
        else:
            length = self.file.seek(offset, whence)
        return length - (self.start or 0)

    def tell(self) -> int:
        return self.file.tell()

    def fileno(self) -> int:
        # Raise OSError to prevent access to the underlying file descriptor
        raise OSError("fileno")

    def __getattr__(self, name: str) -> t.Any:
        return getattr(self.file, name)

    def read(self, size: int = -1) -> bytes:
        pos = self.tell()
        if self.end is not None:
            if size < 0 or size > self.end - pos:
                size = self.end - pos
        res = self.file.read(size)
        if self.read_cb is not None:
            self.read_cb(len(res))
        return res

    def write(self, data: bytes) -> t.Any:  # type: ignore  # python buffer types are too new and seem to not support something like Buffer+Sized as of now
        res = super().write(data)
        if self.write_cb is not None:
            if hasattr(data, "__len__"):
                self.write_cb(len(data))
        return res

    def __iter__(self) -> t.Iterator[bytes]:
        return iter(self.file)


class Spinner:
    """A UI component that renders as follows:
    ┌────────────────────────────────────┐
    │ This is log                        │log
    │ This is another log                │
    │                                    │
    │ 00:45 Loading ...                  │spin
    └────────────────────────────────────┘
    Processing -----                00:05 progress

    Use it as a context manager to start the live updating.
    """

    def __init__(self):
        self.transmission_progress = Progress(
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            DownloadColumn(),
            "•",
            TransferSpeedColumn(),
            "•",
            TimeRemainingColumn(),
        )

        self._logs: list[str] = []
        self._spinner_progress = Progress(
            TextColumn("  "),
            TimeElapsedColumn(),
            TextColumn("[bold purple]{task.description}"),
            SpinnerColumn("simpleDots"),
        )
        self._spinner_task_id: t.Optional[TaskID] = None
        self._live = Live(self)

    @property
    def console(self) -> "Console":
        return self._live.console

    @contextmanager
    def spin(self, text: str) -> t.Generator[TaskID, None, None]:
        """Create a spinner as a context manager."""
        try:
            task_id = self.update(text, new=True)
            yield task_id
        finally:
            self._spinner_task_id = None
            self._spinner_progress.stop_task(task_id)
            self._spinner_progress.update(task_id, visible=False)

    def update(self, text: str, new: bool = False) -> TaskID:
        """Update the spin text."""
        if self._spinner_task_id is None or new:
            task_id = self._spinner_progress.add_task(text)
            if self._spinner_task_id is None:
                self._spinner_task_id = task_id
        else:
            task_id = self._spinner_task_id
            self._spinner_progress.update(task_id, description=text)
        return task_id

    def __rich_console__(
        self, console: "Console", options: "ConsoleOptions"
    ) -> RenderResult:
        yield Panel(Group(*self._logs, self._spinner_progress))
        yield self.transmission_progress

    def start(self) -> None:
        """Start live updating."""
        self._live.start()

    def stop(self) -> None:
        """Stop live updating."""
        if self._spinner_task_id is not None:
            self._spinner_progress.stop_task(self._spinner_task_id)
            self._spinner_progress.update(self._spinner_task_id, visible=False)
            self._spinner_task_id = None
        self._live.stop()

    def log(self, message: str) -> None:
        """Add a log message line."""
        self._logs.append(message)

    def __enter__(self) -> Spinner:
        self.start()
        return self

    def __exit__(self, *_: t.Any) -> None:
        self.stop()


class CloudClient(ABC):
    # Moved atrributes to __init__ because otherwise it will keep all the log when running SDK.
    def __init__(self):
        self.spinner = Spinner()

    @abstractmethod
    def push_model(
        self,
        model: Model,
        *,
        force: bool = False,
        threads: int = 10,
    ):
        pass

    @abstractmethod
    def push_bento(
        self,
        bento: Bento,
        *,
        force: bool = False,
        threads: int = 10,
    ):
        pass

    @abstractmethod
    def pull_model(
        self,
        tag: str | Tag,
        *,
        force: bool = False,
    ) -> Model:
        pass

    @abstractmethod
    def pull_bento(
        self,
        tag: str | Tag,
        *,
        force: bool = False,
    ) -> Bento:
        pass
