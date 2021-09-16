from typing import Dict

from simple_di import Provide

from bentoml._internal.configuration.containers import (
    BentoMLContainer,  # RuntimeContainer
)


class RunnerClient:
    def __init__(self, fd):
        self.fd = fd

    async def run(self, *args, **kwargs):
        pass

    async def run_batch(self, *args, **kwargs):
        pass


def get_runner_client(
    runner_name: str, fd_mapping: Dict[str, int] = Provide[BentoMLContainer.fd_mapping]
) -> RunnerClient:
    fd = fd_mapping.get(runner_name)
    return RunnerClient(fd)
