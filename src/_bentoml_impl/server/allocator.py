from __future__ import annotations

import os
import warnings
from typing import Any

from simple_di import Provide
from simple_di import inject

from _bentoml_sdk import Service
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.resource import system_resources
from bentoml.exceptions import BentoMLConfigException

NVIDIA_GPU = "nvidia.com/gpu"
DISABLE_GPU_ALLOCATION_ENV = "BENTOML_DISABLE_GPU_ALLOCATION"


class ResourceAllocator:
    def __init__(self) -> None:
        self.system_resources = system_resources()
        self.remaining_gpus = len(self.system_resources[NVIDIA_GPU])
        self._available_gpus: list[tuple[float, float]] = [
            (1.0, 1.0)  # each item is (remaining, unit)
            for _ in range(self.remaining_gpus)
        ]

    def assign_gpus(self, count: float) -> list[int]:
        if count > self.remaining_gpus:
            warnings.warn(
                f"Requested {count} GPUs, but only {self.remaining_gpus} are remaining. "
                f"Serving may fail due to inadequate GPUs. Set {DISABLE_GPU_ALLOCATION_ENV}=1 "
                "to disable automatic allocation and allocate GPUs manually.",
                ResourceWarning,
                stacklevel=3,
            )
        self.remaining_gpus = max(0, self.remaining_gpus - count)
        if count < 1:  # a fractional GPU
            try:
                # try to find the GPU used with the same fragment
                gpu = next(
                    i
                    for i, v in enumerate(self._available_gpus)
                    if v[0] > 0 and v[1] == count
                )
            except StopIteration:
                try:
                    gpu = next(
                        i for i, v in enumerate(self._available_gpus) if v[0] == 1.0
                    )
                except StopIteration:
                    gpu = len(self._available_gpus)
                    self._available_gpus.append((1.0, count))
            remaining, _ = self._available_gpus[gpu]
            if (remaining := remaining - count) < count:
                # can't assign to the next one, mark it as zero.
                self._available_gpus[gpu] = (0.0, count)
            else:
                self._available_gpus[gpu] = (remaining, count)
            return [gpu]
        else:  # allocate n GPUs, n is a positive integer
            if int(count) != count:
                raise BentoMLConfigException(
                    "Float GPUs larger than 1 is not supported"
                )
            count = int(count)
            unassigned = [
                gpu
                for gpu, value in enumerate(self._available_gpus)
                if value[0] > 0 and value[1] == 1.0
            ]
            if len(unassigned) < count:
                warnings.warn(
                    f"Not enough GPUs to be assigned, {count} is requested",
                    ResourceWarning,
                )
                for _ in range(count - len(unassigned)):
                    unassigned.append(len(self._available_gpus))
                    self._available_gpus.append((1.0, 1.0))
            for gpu in unassigned[:count]:
                self._available_gpus[gpu] = (0.0, 1.0)
            return unassigned[:count]

    @inject
    def get_worker_env(
        self,
        service: Service[Any],
        services: dict[str, Any] = Provide[BentoMLContainer.config.services],
    ) -> tuple[int, list[dict[str, str]]]:
        config = services[service.name]

        num_gpus = 0
        num_workers = 1
        worker_env: list[dict[str, str]] = []
        if "gpu" in (config.get("resources") or {}):
            num_gpus = config["resources"]["gpu"]  # type: ignore
        if config.get("workers"):
            if (workers := config["workers"]) == "cpu_count":
                num_workers = int(self.system_resources["cpu"])
                # don't assign gpus to workers
                return num_workers, worker_env
            else:  # workers is a number
                num_workers = workers
        if num_gpus and DISABLE_GPU_ALLOCATION_ENV not in os.environ:
            assigned = self.assign_gpus(num_gpus)
            # assign gpus to all workers
            worker_env = [
                {"CUDA_VISIBLE_DEVICES": ",".join(map(str, assigned))}
                for _ in range(num_workers)
            ]
        return num_workers, worker_env
