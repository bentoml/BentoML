from __future__ import annotations

from typing import Any

from _bentoml_sdk import Service
from simple_di import Provide
from simple_di import inject

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.resource import system_resources
from bentoml.exceptions import BentoMLConfigException


class ResourceUnavailable(BentoMLConfigException):
    pass


NVIDIA_GPU = "nvidia.com/gpu"


class ResourceAllocator:
    def __init__(self) -> None:
        self.system_resources = system_resources()
        self._available_gpus: dict[int, tuple[float, float]] = dict.fromkeys(
            # each value is a two-number pair, (remaining, unit)
            # when a GPU has no remaining, it will be removed from the map.
            self.system_resources[NVIDIA_GPU],
            (1.0, 1.0),
        )

    def assign_gpus(self, count: float) -> list[int]:
        if count > (
            remaining_total := sum(v[0] for v in self._available_gpus.values())
        ):
            raise ResourceUnavailable(
                f"Requested {count} GPUs, but only {remaining_total} are available."
            )
        if count < 1:  # a fragmental GPU
            try:
                # try to find the GPU used with the same fragment
                gpu = next(k for k, v in self._available_gpus.items() if v[1] == count)
            except StopIteration:
                try:
                    gpu = next(
                        k for k, v in self._available_gpus.items() if v[0] == 1.0
                    )
                except StopIteration:
                    raise ResourceUnavailable(
                        f"Can't find an available GPU for {count} resources"
                    ) from None
            remaining, _ = self._available_gpus[gpu]
            if remaining - count < count:
                # can't assign to the next one, remove from the available GPUs.
                del self._available_gpus[gpu]
            else:
                self._available_gpus[gpu] = (remaining - count, count)
            return [gpu]
        else:  # allocate n GPUs, n is a positive integer
            if int(count) != count:
                raise BentoMLConfigException(
                    "Float GPUs larger than 1 is not supported"
                )
            count = int(count)
            unassigned = [
                gpu for gpu, value in self._available_gpus.items() if value[1] == 1.0
            ]
            if len(unassigned) < count:
                raise ResourceUnavailable(f"Unable to allocate {count} GPUs")
            assigned = unassigned[:count]
            for gpu in assigned:
                del self._available_gpus[gpu]
            return assigned

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
            elif isinstance(workers, int):
                num_workers = workers
                if num_gpus:
                    assigned = self.assign_gpus(num_gpus)
                    # assign gpus to all workers
                    worker_env = [
                        {"CUDA_VISIBLE_DEVICES": ",".join(map(str, assigned))}
                        for _ in range(num_workers)
                    ]
            else:  # workers is a list
                num_workers = len(workers)
                if not workers:
                    raise BentoMLConfigException("workers list is empty.")
                for worker in workers:
                    if isinstance(requested := worker["gpus"], list):
                        if any(
                            gpu
                            not in (system_gpus := self.system_resources[NVIDIA_GPU])
                            for gpu in requested
                        ):
                            raise ResourceUnavailable(
                                f"Requested GPUs {requested}, but only {system_gpus} are available"
                            )
                        worker_env.append(
                            {"CUDA_VISIBLE_DEVICES": ",".join(map(str, requested))}
                        )
                    else:
                        worker_env.append(
                            {
                                "CUDA_VISIBLE_DEVICES": ",".join(
                                    map(str, self.assign_gpus(requested))
                                )
                            }
                        )
        elif num_gpus > 0:
            assigned = self.assign_gpus(num_gpus)
            num_workers = len(assigned)
            worker_env = [{"CUDA_VISIBLE_DEVICES": str(i)} for i in assigned]
        return num_workers, worker_env
