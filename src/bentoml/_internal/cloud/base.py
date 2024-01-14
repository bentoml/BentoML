from __future__ import annotations

import io
import typing as t
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from tempfile import SpooledTemporaryFile

from rich.console import Group
from rich.panel import Panel
from rich.progress import BarColumn
from rich.progress import DownloadColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn

from ...exceptions import BentoMLException
from ..bento import Bento
from ..bento import BentoStore
from ..models import Model
from ..models import ModelStore
from ..tag import Tag

FILE_CHUNK_SIZE = 100 * 1024 * 1024  # 100Mb


def io_wrapper(
    memory: int,
    *,
    read_cb: t.Callable[[int], None] | None = None,
    write_cb: t.Callable[[int], None] | None = None,
) -> CallbackIOWrapper | CallbackSpooledTemporaryFileIO:
    """
    io_wrapper is a wrapper for SpooledTemporaryFileIO and CallbackIOWrapper
    """
    if memory == -1:
        return CallbackIOWrapper(read_cb=read_cb, write_cb=write_cb)
    elif memory > 0:
        return CallbackSpooledTemporaryFileIO(
            memory * 1024**3, read_cb=read_cb, write_cb=write_cb
        )
    else:
        raise BentoMLException(f"Option max_memory must be -1 or > 0, got {memory}")


class CallbackSpooledTemporaryFileIO(SpooledTemporaryFile):
    """
    A SpooledTemporaryFile wrapper that calls
    a callback when read/write is called
    """

    read_cb: t.Callable[[int], None] | None
    write_cb: t.Callable[[int], None] | None

    def __init__(
        self,
        max_size: int = 0,
        *,
        read_cb: t.Callable[[int], None] | None = None,
        write_cb: t.Callable[[int], None] | None = None,
    ):
        self.read_cb = read_cb
        self.write_cb = write_cb
        super().__init__(max_size)

    def read(self, *args):
        res = super().read(*args)
        if self.read_cb is not None:
            self.read_cb(len(res))
        return res

    def write(self, s):
        res = super().write(s)
        if self.write_cb is not None:
            if hasattr(s, "__len__"):
                self.write_cb(len(s))
        return res

    def size(self) -> int:
        """
        get the size of the file
        """
        current_pos = self.tell()
        self.seek(0, 2)
        file_size = self.tell()
        self.seek(current_pos)
        return file_size

    def chunk(self, start: int, end: int) -> bytes:
        """
        chunk the file slice of [start, end)
        """
        self.seek(start)
        if end < 0 or start > end:
            return self.read()
        return self.read(end - start)


class CallbackIOWrapper(io.BytesIO):
    read_cb: t.Callable[[int], None] | None
    write_cb: t.Callable[[int], None] | None

    def __init__(
        self,
        buffer: t.Any = None,
        *,
        read_cb: t.Callable[[int], None] | None = None,
        write_cb: t.Callable[[int], None] | None = None,
    ):
        self.read_cb = read_cb
        self.write_cb = write_cb
        super().__init__(buffer)

    def read(self, size: int | None = None) -> bytes:
        if size is not None:
            res = super().read(size)
        else:
            res = super().read()
        if self.read_cb is not None:
            self.read_cb(len(res))
        return res

    def write(self, data: bytes) -> t.Any:  # type: ignore  # python buffer types are too new and seem to not support something like Buffer+Sized as of now
        res = super().write(data)
        if self.write_cb is not None:
            if hasattr(data, "__len__"):
                self.write_cb(len(data))
        return res

    def size(self) -> int:
        """
        get the size of the buffer
        """
        return super().getbuffer().nbytes

    def chunk(self, start: int, end: int) -> bytes:
        """
        chunk the buffer slice of [start, end)
        """
        return super().getbuffer()[start:end]


class CloudClient(ABC):
    log_progress = Progress(TextColumn("{task.description}"))

    spinner_progress = Progress(
        TextColumn("  "),
        TimeElapsedColumn(),
        TextColumn("[bold purple]{task.fields[action]}"),
        SpinnerColumn("simpleDots"),
    )

    transmission_progress = Progress(
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

    progress_group = Group(
        Panel(Group(log_progress, spinner_progress)), transmission_progress
    )

    @contextmanager
    def spin(self, *, text: str):
        task_id = self.spinner_progress.add_task("", action=text)
        try:
            yield
        finally:
            self.spinner_progress.stop_task(task_id)
            self.spinner_progress.update(task_id, visible=False)

    @abstractmethod
    def push_model(
        self,
        model: Model,
        *,
        force: bool = False,
        threads: int = 10,
        context: str | None = None,
    ):
        pass

    @abstractmethod
    def push_bento(
        self,
        bento: Bento,
        *,
        force: bool = False,
        threads: int = 10,
        context: str | None = None,
    ):
        pass

    @abstractmethod
    def pull_model(
        self,
        tag: str | Tag,
        *,
        force: bool = False,
        context: str | None = None,
        model_store: ModelStore,
    ) -> Model:
        pass

    @abstractmethod
    def pull_bento(
        self,
        tag: str | Tag,
        *,
        force: bool = False,
        context: str | None = None,
        bento_store: BentoStore,
    ) -> Bento:
        pass
