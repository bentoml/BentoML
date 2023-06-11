from __future__ import annotations

import abc
import contextvars
import logging
import math
import typing as t

from ..resource import get_resource, system_resources
from .runnable import Runnable

logger = logging.getLogger(__name__)

GPU_UNAVAILABLE_VAR: contextvars.ContextVar[
    t.Optional[t.List[int]]
] = contextvars.ContextVar("GPU_UNAVAILABLE_VAR", default=None)


class Strategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def get_worker_count(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: dict[str, t.Any] | None,
        workers_per_resource: int | float,
    ) -> int:
        ...

    @classmethod
    @abc.abstractmethod
    def get_worker_env(
        cls,
        runnable_class: t.Type[Runnable],
        resource_request: dict[str, t.Any] | None,
        workers_per_resource: int | float,
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
        workers_per_resource: int | float,
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
            return int(len(nvidia_gpus) * workers_per_resource)

        # use CPU
        cpus = get_resource(resource_request, "cpu")
        if cpus is not None and cpus > 0:
            if "cpu" not in runnable_class.SUPPORTED_RESOURCES:
                logger.warning(
                    "No known supported resource available for %s, falling back to using CPU.",
                    runnable_class,
                )

            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
                if isinstance(workers_per_resource, float):
                    raise ValueError(
                        "Fractional CPU multi threading support is not yet supported."
                    )
                return workers_per_resource

            return math.ceil(cpus) * workers_per_resource

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
        workers_per_resource: int | float,
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
        nvidia_gpus: list[int] | None = get_resource(resource_request, "nvidia.com/gpu")
        if (
            nvidia_gpus is not None
            and len(nvidia_gpus) > 0
            and "nvidia.com/gpu" in runnable_class.SUPPORTED_RESOURCES
        ):
            if isinstance(workers_per_resource, float):
                # NOTE: We hit this branch when workers_per_resource is set to
                # float, for example 0.5 or 0.25
                assigned_resource_per_worker = int(1 / workers_per_resource)
                if len(nvidia_gpus) < assigned_resource_per_worker:
                    raise IndexError(
                        f"There aren't enough assigned GPU for given worker id {worker_index}."
                    )
                unavailable = GPU_UNAVAILABLE_VAR.get()
                if unavailable is not None:
                    nvidia_gpus = [i for i in nvidia_gpus if i not in unavailable]
                    assigned_gpu = nvidia_gpus[:assigned_resource_per_worker]
                else:
                    assigned_gpu = nvidia_gpus[:assigned_resource_per_worker]
                    GPU_UNAVAILABLE_VAR.set(assigned_gpu)
                dev = ",".join(map(str, assigned_gpu))
            else:
                dev = str(nvidia_gpus[worker_index // workers_per_resource])
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
