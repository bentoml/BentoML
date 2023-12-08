from __future__ import annotations

import math
from typing import Any

from bentoml._internal.resource import system_resources
from bentoml.exceptions import BentoMLConfigException

from ..factory import Service


class ResourceUnaiable(BentoMLConfigException):
    pass


class Scheduler:
    def __init__(self) -> None:
        self.system_resources = system_resources()
        self._available_gpus = self.system_resources["nvidia.com/gpu"]

    def assign_gpus(self, count: int) -> list[int]:
        if count > len(self._available_gpus):
            raise ResourceUnaiable(
                f"Requested {count} GPUs, but only {len(self._available_gpus)} are available."
            )
        gpus, self._available_gpus = (
            self._available_gpus[:count],
            self._available_gpus[count:],
        )
        return gpus

    def get_worker_env(self, service: Service[Any]) -> tuple[int, list[dict[str, str]]]:
        config = service.config

        num_gpus = 0
        num_workers = 1
        worker_env: list[dict[str, str]] = []
        if "gpu" in config.get("resources", {}):
            num_gpus = math.ceil(config["resources"]["gpu"])  # type: ignore
        if "workers" in config:
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
                requested_gpus: set[int] = set()
                for worker in workers:
                    if isinstance(requested := worker["gpus"], int):
                        worker_env.append(
                            {
                                "CUDA_VISIBLE_DEVICES": ",".join(
                                    map(str, self.assign_gpus(requested))
                                )
                            }
                        )
                    else:
                        if len(requested) > len(self._available_gpus):
                            raise ResourceUnaiable(
                                f"Requested {len(requested)} GPUs, but only {len(self._available_gpus)} are available."
                            )
                        if any(gpu not in self._available_gpus for gpu in requested):
                            raise ResourceUnaiable(
                                f"Requested GPUs {requested}, but only {self._available_gpus} are available."
                            )
                        worker_env.append(
                            {"CUDA_VISIBLE_DEVICES": ",".join(map(str, requested))}
                        )
                        requested_gpus.update(requested)
                self._available_gpus = [
                    gpu for gpu in self._available_gpus if gpu not in requested_gpus
                ]
        elif "gpu" in config.get("resources", {}):
            num_workers = (
                int(1 / gpu)
                if (gpu := config.get("resources", {}).get("gpu", 1.0)) < 1
                # Assign one worker per GPU
                else int(gpu)
            )
            assigned = self.assign_gpus(num_gpus)
            if len(assigned) < num_workers:
                # multiple workers share the same GPU
                assigned = assigned * num_workers
            worker_env = [{"CUDA_VISIBLE_DEVICES": str(i)} for i in assigned]
        return num_workers, worker_env
