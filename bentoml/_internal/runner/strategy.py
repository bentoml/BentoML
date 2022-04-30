import os
import abc
import math
import typing as t
import logging

import attr

from .runnable import Runnable

logger = logging.getLogger(__name__)


@attr.define(frozen=True)
class Resource:
    cpu: int = attr.field()
    nvidia_gpu: int = attr.field()
    custom_resources: t.Dict[str, t.Union[float, int]] = attr.field(factory=dict)


class Strategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_worker_count(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
    ) -> int:
        ...

    @classmethod
    @abc.abstractmethod
    def setup_worker(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
        worker_index: int,
    ) -> None:
        ...


THREAD_ENVS = [
    "OMP_NUM_THREADS",
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
]


class DefaultStrategy(Strategy):
    @classmethod
    @abc.abstractmethod
    def get_worker_count(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
    ) -> int:
        # use nvidia gpu
        if resource_request.nvidia_gpu > 0 and runnable_class.SUPPORT_NVIDIA_GPU:
            return math.ceil(resource_request.nvidia_gpu)

        # use CPU
        if runnable_class.SUPPORT_CPU_MULTI_THREADING:
            return 1

        return math.ceil(resource_request.cpu)

    @classmethod
    @abc.abstractmethod
    def setup_worker(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
        worker_index: int,
    ) -> None:
        # use nvidia gpu
        if resource_request.nvidia_gpu > 0 and runnable_class.SUPPORT_NVIDIA_GPU:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index)
            return

        # use CPU
        if runnable_class.SUPPORT_CPU_MULTI_THREADING:
            thread_count = math.ceil(resource_request.cpu)
            for thread_env in THREAD_ENVS:
                os.environ[thread_env] = str(thread_count)
            return

        for thread_env in THREAD_ENVS:
            os.environ[thread_env] = "1"
        return
