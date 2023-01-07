from __future__ import annotations

import typing as t

import attr

from ..store import Store
from ..store import StoreItem
from ..runner import Runner
from ..runner.runnable import Runnable
from ..runner.runner_handle import RunnerHandle

if t.TYPE_CHECKING:
    from ..types import PathType


def save_model(**kwargs: t.Any):
    ...


def create_runners_from_repository(path: PathType) -> TritonRunnerRepository:
    ...


@attr.define
class TritonRunnerRepository:
    ...


def create_runner(**kwargs: t.Any) -> TritonRunner:
    ...


class TritonRunnable(Runnable):
    ...


class TritonRunner(Runner):
    ...


class TritonRunnerHandle(RunnerHandle):
    ...


class TritonModel(StoreItem):
    ...


class TritonModelStore(Store[TritonModel]):
    ...
