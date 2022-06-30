import os
import abc
import math
import typing as t
import logging

from .resource import Resource
from .runnable import Runnable

logger = logging.getLogger(__name__)


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
    "BENTOML_NUM_THREAD",
]  # TODO(jiang): make it configurable?


class DefaultStrategy(Strategy):
    @classmethod
    def get_worker_count(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
    ) -> int:
        # use nvidia gpu
        if (
            resource_request.nvidia_gpu is not None
            and resource_request.nvidia_gpu > 0
            and runnable_class.SUPPORT_NVIDIA_GPU
        ):
            return math.ceil(resource_request.nvidia_gpu)

        # use CPU
        if resource_request.cpu is not None and resource_request.cpu > 0:
            if runnable_class.SUPPORT_CPU_MULTI_THREADING:
                return 1

            return math.ceil(resource_request.cpu)

        # this would not be reached by user since we always read system resource as default
        logger.warning("No resource request found, always use single worker")
        return 1

    @classmethod
    def setup_worker(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: Resource,
        worker_index: int,
    ) -> None:
        # use nvidia gpu
        if (
            resource_request.nvidia_gpu is not None
            and resource_request.nvidia_gpu > 0
            and runnable_class.SUPPORT_NVIDIA_GPU
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index - 1)
            logger.info(
                "Setting up worker: set CUDA_VISIBLE_DEVICES to %s",
                worker_index - 1,
            )
            return

        # use CPU
        if resource_request.cpu is not None and resource_request.cpu > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu
            if runnable_class.SUPPORT_CPU_MULTI_THREADING:
                thread_count = math.ceil(resource_request.cpu)
                for thread_env in THREAD_ENVS:
                    os.environ[thread_env] = str(thread_count)
                logger.info(
                    "Setting up worker: set CPU thread count to %s", thread_count
                )
                return
            else:
                for thread_env in THREAD_ENVS:
                    os.environ[thread_env] = "1"
                return
