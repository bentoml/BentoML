import logging
from timeit import default_timer
from typing import TYPE_CHECKING
from contextvars import ContextVar

if TYPE_CHECKING:
    from .. import external_typing as ext

import abc


class SchedulingStrategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_num_workers() -> int:
        pass
