import typing as t
import traceback

from .runner import Runner
from .runner import RunnerImpl
from .runner import RunnerState
from .runner import SimpleRunner
from ...exceptions import BentoMLException
from ..runner.utils import Params
from ..runner.container import AutoContainer


class LocalRunner(RunnerImpl):
    def setup(self) -> None:
        if self._state != RunnerState.INIT:
            return
        self._state = RunnerState.SETTING_UP
        self._runner._setup()  # type: ignore[reportPrivateUsage]
        self._state = RunnerState.READY

    def shutdown(self) -> None:
        if self._state in (
            RunnerState.INIT,
            RunnerState.SHUTIING_DOWN,
            RunnerState.SHUTDOWN,
        ):
            return
        if self._state in (RunnerState.READY, RunnerState.SETTING_UP):
            self._state = RunnerState.SHUTIING_DOWN
            self._runner._shutdown()  # type: ignore[reportPrivateUsage]
            self._state = RunnerState.SHUTDOWN
            return
        raise RuntimeError("Runner is in unknown state")

    async def async_run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self.run(*args, **kwargs)

    async def async_run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self.run_batch(*args, **kwargs)

    def run(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if self._state is RunnerState.INIT:
            self.setup()

        if isinstance(self._runner, Runner):
            params = Params(*args, **kwargs)
            params = params.map(
                lambda i: AutoContainer.singles_to_batch(
                    [i],
                    batch_axis=self._runner.batch_options.input_batch_axis,
                )
            )
            try:
                batch_result = self._runner._run_batch(  # type: ignore[reportPrivateUsage]
                    *params.args,
                    **params.kwargs,
                )
            except Exception as e:  # pylint: disable=broad-except
                if isinstance(e, BentoMLException):
                    raise e from None
                else:
                    raise BentoMLException(
                        f"Exception happened in the runner {self._runner.name}._run_batch: {str(e)}"
                    ) from None
            single_results = AutoContainer.batch_to_singles(
                batch_result,
                batch_axis=self._runner.batch_options.output_batch_axis,
            )
            return single_results[0]

        if isinstance(self._runner, SimpleRunner):
            try:
                return self._runner._run(*args, **kwargs)  # type: ignore[reportPrivateUsage]
            except Exception as e:  # pylint: disable=broad-except
                if isinstance(e, BentoMLException):
                    raise e from None
                else:
                    raise BentoMLException(
                        f"Exception happened in the runner {self._runner.name}._run_batch: {str(e)}"
                    ) from None

    def run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        if self._state is RunnerState.INIT:
            self.setup()

        if isinstance(self._runner, Runner):
            try:
                return self._runner._run_batch(*args, **kwargs)  # type: ignore[reportPrivateUsage]
            except Exception as e:  # pylint: disable=broad-except
                if isinstance(e, BentoMLException):
                    raise e from None
                else:
                    raise BentoMLException(
                        f"Exception happened in the runner {self._runner.name}._run_batch: {traceback.format_exc()}"
                    )

        if isinstance(self._runner, SimpleRunner):
            try:
                raise RuntimeError("shall not call run_batch on a" " simple runner")
            except Exception as e:  # pylint: disable=broad-except
                if isinstance(e, BentoMLException):
                    raise e from None
                else:
                    raise BentoMLException(
                        f"Exception happened in the runner {self._runner.name}._run_batch: {traceback.format_exc()}"
                    ) from None

    def __del__(self):
        self.shutdown()
