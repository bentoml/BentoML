from __future__ import annotations

import typing as t
from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager
from functools import wraps

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

from ..bento import Bento
from ..bento import BentoStore
from ..models import Model
from ..models import ModelStore
from ..tag import Tag

FILE_CHUNK_SIZE = 100 * 1024 * 1024  # 100Mb


class ObjectWrapper(object):
    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._wrapped, name)

    def __setattr__(self, name: str, value: t.Any) -> None:
        return setattr(self._wrapped, name, value)

    def wrapper_getattr(self, name: str):
        """Actual `self.getattr` rather than self._wrapped.getattr"""
        return getattr(self, name)

    def wrapper_setattr(self, name: str, value: t.Any) -> None:
        """Actual `self.setattr` rather than self._wrapped.setattr"""
        return object.__setattr__(self, name, value)

    def __init__(self, wrapped: t.Any):
        """
        Thin wrapper around a given object
        """
        self.wrapper_setattr("_wrapped", wrapped)


class CallbackIOWrapper(ObjectWrapper):
    def __init__(
        self,
        callback: t.Callable[[int], None],
        stream: t.BinaryIO,
        method: t.Literal["read", "write"] = "read",
    ):
        """
        Wrap a given `file`-like object's `read()` or `write()` to report
        lengths to the given `callback`
        """
        super().__init__(stream)
        func = getattr(stream, method)
        if method == "write":

            @wraps(func)
            def write(data: t.Union[bytes, bytearray], *args: t.Any, **kwargs: t.Any):
                res = func(data, *args, **kwargs)
                callback(len(data))
                return res

            self.wrapper_setattr("write", write)
        elif method == "read":

            @wraps(func)
            def read(*args: t.Any, **kwargs: t.Any):
                data = func(*args, **kwargs)
                callback(len(data))
                return data

            self.wrapper_setattr("read", read)
        else:
            raise KeyError("Can only wrap read/write methods")


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
