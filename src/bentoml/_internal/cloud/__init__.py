from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from contextlib import contextmanager

from simple_di import Provide
from rich.panel import Panel
from rich.console import Group
from rich.progress import Progress
from rich.progress import BarColumn
from rich.progress import TextColumn
from rich.progress import SpinnerColumn
from rich.progress import DownloadColumn
from rich.progress import TimeElapsedColumn
from rich.progress import TimeRemainingColumn
from rich.progress import TransferSpeedColumn

from ..tag import Tag
from ..bento import Bento
from ..bento import BentoStore
from ..models import Model
from ..models import ModelStore
from ..configuration.containers import BentoMLContainer


class BaseCloudClient(ABC):
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
        model_store: ModelStore = Provide[BentoMLContainer.model_store],
        context: str | None = None,
    ) -> Model:
        pass

    @abstractmethod
    def pull_bento(
        self,
        tag: str | Tag,
        *,
        force: bool = False,
        bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
        context: str | None = None,
    ) -> Bento:
        pass

    @abstractmethod
    def push(
        self,
        item: Bento | Model | str | Tag,
        *,
        force: bool = False,
        threads: int = 10,
        context: str | None = None,
    ):
        pass

    @abstractmethod
    def pull(
        self,
        item: Bento | Model | str | Tag,
        *,
        force: bool = False,
        threads: int = 10,
        context: str | None = None,
    ) -> Bento | Model:
        pass
