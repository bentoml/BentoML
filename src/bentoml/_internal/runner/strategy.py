from __future__ import annotations

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
    def get_worker_env(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: dict[str, t.Any],
        worker_index: int,
    ) -> dict[str, t.Any]:
        """
        Parameters
        ----------
        runnable_class : type[Runnable]
            The runnable class to be run.
        resource_request : dict[str, Any]
            The resource request of the runnable.
        worker_index : int
            The index of the worker, start from 0.
        """
        ...


THREAD_ENVS = [
    "BENTOML_NUM_THREAD",  # For custom Runner code
    "OMP_NUM_THREADS",  # openmp
    "OPENBLAS_NUM_THREADS",  # openblas,
    "MKL_NUM_THREADS",  # mkl,
    "VECLIB_MAXIMUM_THREADS",  # accelerate,
    "NUMEXPR_NUM_THREADS",  # numexpr
    # For huggingface fast tokenizer
    "RAYON_RS_NUM_CPUS",
    # For Tensorflow
    "TF_NUM_INTEROP_THREADS",
    "TF_NUM_INTRAOP_THREADS",
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
            and len(nvidia_gpus) > 0
            and "nvidia.com/gpu" in runnable_class.SUPPORTED_RESOURCES
        ):
            return len(nvidia_gpus)

        # use CPU
        cpus = get_resource(resource_request, "cpu")
        if cpus is not None and cpus > 0:
            if "cpu" not in runnable_class.SUPPORTED_RESOURCES:
                logger.warning(
                    "No known supported resource available for %s, falling back to using CPU.",
                    runnable_class,
                )

            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
                return 1

            return math.ceil(cpus)

        # this should not be reached by user since we always read system resource as default
        raise ValueError(
            f"No known supported resource available for {runnable_class}. Please check your resource request. "
            "Leaving it blank will allow BentoML to use system resources."
        )

    @classmethod
    def get_worker_env(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: dict[str, t.Any] | None,
        worker_index: int,
    ) -> dict[str, t.Any]:
        """
        Parameters
        ----------
        runnable_class : type[Runnable]
            The runnable class to be run.
        resource_request : dict[str, Any]
            The resource request of the runnable.
        worker_index : int
            The index of the worker, start from 0.
        """
        environ: dict[str, t.Any] = {}
        if resource_request is None:
            resource_request = system_resources()

        # use nvidia gpu
        nvidia_gpus = get_resource(resource_request, "nvidia.com/gpu")
        if (
            nvidia_gpus is not None
            and len(nvidia_gpus) > 0
            and "nvidia.com/gpu" in runnable_class.SUPPORTED_RESOURCES
        ):
            dev = str(nvidia_gpus[worker_index])
            environ["CUDA_VISIBLE_DEVICES"] = dev
            logger.info(
                "Environ for worker %s: set CUDA_VISIBLE_DEVICES to %s",
                worker_index,
                dev,
            )
            return environ

        # use CPU
        cpus = get_resource(resource_request, "cpu")
        if cpus is not None and cpus > 0:
            environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu
            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
                thread_count = math.ceil(cpus)
                for thread_env in THREAD_ENVS:
                    environ[thread_env] = str(thread_count)
                logger.info(
                    "Environ for worker %d: set CPU thread count to %d",
                    worker_index,
                    thread_count,
                )
                return environ
            else:
                for thread_env in THREAD_ENVS:
                    environ[thread_env] = "1"
                return environ

        return environ
