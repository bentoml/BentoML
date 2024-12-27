from __future__ import annotations

import typing as t
from contextlib import contextmanager

import attrs
from rich import get_console
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

if t.TYPE_CHECKING:
    from rich.console import Console
    from rich.console import ConsoleOptions
    from rich.console import RenderResult

FILE_CHUNK_SIZE = 100 * 1024 * 1024  # 100Mb
UPLOAD_RETRY_COUNT = 3


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

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or get_console()
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
            console=self.console,
        )

        self._logs: list[str] = []
        self._spinner_progress = Progress(
            TextColumn("  "),
            TimeElapsedColumn(),
            TextColumn("[bold purple]{task.description}"),
            SpinnerColumn("simpleDots"),
            console=self.console,
        )
        self._spinner_task_id: t.Optional[TaskID] = None
        self._live = Live(self, console=self.console)
        self._start_count = 0

    @contextmanager
    def spin(self, text: str) -> t.Generator[TaskID, None, None]:
        """Create a spinner as a context manager."""
        task_id = self.update(text, new=True)
        try:
            yield task_id
        finally:
            self._spinner_progress.remove_task(task_id)
            if self._spinner_task_id == task_id:
                self._spinner_task_id = None

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
        self._start_count += 1
        self._live.start()

    def stop(self) -> None:
        """Stop live updating."""
        self._start_count -= 1
        if self._start_count > 0:
            return
        if self._spinner_task_id is not None:
            self._spinner_progress.remove_task(self._spinner_task_id)
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
