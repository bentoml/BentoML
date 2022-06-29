from __future__ import annotations

import os
import abc
import math
import typing as t
import logging

from .runnable import Runnable
from ..resource import get_resource
from ..resource import system_resources

logger = logging.getLogger(__name__)


class Strategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_worker_count(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: dict[str, t.Any],
    ) -> int:
        ...

    @classmethod
    @abc.abstractmethod
    def setup_worker(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: dict[str, t.Any],
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
        resource_request: dict[str, t.Any] | None,
    ) -> int:
        if resource_request is None:
            resource_request = system_resources()

        # use nvidia gpu
        nvidia_gpus = get_resource(resource_request, "nvidia.com/gpu")
        if (
            nvidia_gpus is not None
            and nvidia_gpus > 0
            and "nvidia.com/gpu" in runnable_class.SUPPORTED_RESOURCES
        ):
            return math.ceil(nvidia_gpus)

        # use CPU
        cpus = get_resource(resource_request, "cpu")
        if cpus is not None and cpus > 0:
            if "cpu" not in runnable_class.SUPPORTED_RESOURCES:
                logger.warning(
                    f"No known supported resource available for {runnable_class}, falling back to using CPU."
                )

            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
                return 1

            return math.ceil(cpus)

        # this should not be reached by user since we always read system resource as default
        logger.warning(
            "No resource request found, falling back to using a single worker"
        )
        return 1

    @classmethod
    def setup_worker(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: dict[str, t.Any] | None,
        worker_index: int,
    ) -> None:
        if resource_request is None:
            resource_request = system_resources()

        # use nvidia gpu
        nvidia_gpus = get_resource(resource_request, "nvidia.com/gpu")
        if (
            nvidia_gpus is not None
            and nvidia_gpus > 0
            and "nvidia.com/gpu" in runnable_class.SUPPORTED_RESOURCES
        ):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_index - 1)
            logger.info(
                "Setting up worker: set CUDA_VISIBLE_DEVICES to %s",
                worker_index - 1,
            )
            return

        # use CPU
        cpus = get_resource(resource_request, "cpu")
        if cpus is not None and cpus > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu
            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
                thread_count = math.ceil(cpus)
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
