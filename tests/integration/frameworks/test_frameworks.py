from __future__ import annotations

import os
import types
import typing as t
import logging
from typing import TYPE_CHECKING

import pytest

import bentoml
from bentoml.exceptions import NotFound
from bentoml._internal.models.model import ModelContext
from bentoml._internal.models.model import ModelSignature
from bentoml._internal.runner.runner import Runner
from bentoml._internal.runner.strategy import DefaultStrategy
from bentoml._internal.runner.runner_handle.local import LocalRunnerRef

from .models import FrameworkTestModel

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


@pytest.fixture(name="saved_model")
def fixture_saved_model(
    framework: types.ModuleType, test_model: FrameworkTestModel
) -> bentoml.Model:
    return framework.save_model(
        test_model.name, test_model.model, **test_model.save_kwargs
    )


def test_backward_warnings(
    framework: types.ModuleType,
    test_model: FrameworkTestModel,
    caplog: LogCaptureFixture,
    saved_model: bentoml.Model,
):
    # We want to cover the cases where the user is using the old API
    # and we want to make sure that the warning is raised.
    if (
        not hasattr(framework, "__test_backward_compatible__")
        or framework.__test_backward_compatible__ is False
    ):
        pytest.skip(
            "Module '%s' does not have a backward compatible warning."
            % framework.__name__
        )
    if hasattr(framework, "save"):
        with caplog.at_level(logging.WARNING):
            framework.save(test_model.name, test_model.model)
        assert (
            f'The "{framework.__name__}.save" method is deprecated. Use "{framework.__name__}.save_model" instead.'
            in caplog.text
        )
        caplog.clear()
    if hasattr(framework, "load"):
        with caplog.at_level(logging.WARNING):
            framework.load(saved_model.tag)
        assert (
            f'The "{framework.__name__}.load" method is deprecated. Use "{framework.__name__}.load_model" instead.'
            in caplog.text
        )
        caplog.clear()
    if hasattr(framework, "load_runner"):
        with caplog.at_level(logging.ERROR):
            framework.load_runner(saved_model.tag, "asdf")
            assert '"load_runner" arguments will be ignored.' in caplog.text
            caplog.clear()

        with caplog.at_level(logging.WARNING):
            framework.load_runner(saved_model.tag)
            assert (
                f'The "{framework.__name__}.load_runner" method is deprecated. Use "{framework.__name__}.get("{saved_model.tag}").to_runner()" instead.'
                in caplog.text
            )
            caplog.clear()


def test_wrong_module_load_exc(framework: types.ModuleType):
    with bentoml.models.create(
        "wrong_module",
        module=__name__,
        context=ModelContext("wrong_module", {"wrong_module": "1.0.0"}),
        signatures={},
    ) as ctx:
        tag = ctx.tag

        model = ctx

    with pytest.raises(
        NotFound, match=f"Model {tag} was saved with module {__name__}, "
    ):
        framework.get(tag)

    with pytest.raises(
        NotFound, match=f"Model {tag} was saved with module {__name__}, "
    ):
        framework.load_model(tag)

    with pytest.raises(
        NotFound, match=f"Model {tag} was saved with module {__name__}, "
    ):
        framework.load_model(model)


def test_model_options_init(
    framework: types.ModuleType, test_model: FrameworkTestModel
):
    if not hasattr(framework, "ModelOptions"):
        pytest.skip(f"No ModelOptions for framework '{framework.__name__}'")

    ModelOptions = framework.ModelOptions

    for configuration in test_model.configurations:
        from_kwargs = ModelOptions(**configuration.load_kwargs)
        from_with_options = from_kwargs.with_options(**configuration.load_kwargs)
        assert from_kwargs == from_with_options
        assert from_kwargs.to_dict() == from_with_options.to_dict()

        from_dict = ModelOptions(**from_kwargs.to_dict())
        assert from_dict == from_kwargs


def test_generic_arguments(framework: types.ModuleType, test_model: FrameworkTestModel):
    # test that the generic save API works
    from sklearn.preprocessing import StandardScaler  # type: ignore (bad sklearn types)

    scaler: StandardScaler = StandardScaler().fit(  # type: ignore (bad sklearn types)
        [[4], [3], [7], [8], [4], [3], [9], [6]]
    )
    assert scaler.mean_[0] == 5.5  # type: ignore (bad sklearn types)
    assert scaler.var_[0] == 4.75  # type: ignore (bad sklearn types)

    kwargs = test_model.save_kwargs.copy()
    if test_model.model_signatures:
        kwargs["signatures"] = test_model.model_signatures
        meths = list(test_model.model_signatures.keys())
    else:
        default_meth = "pytest-signature-rjM5"
        kwargs["signatures"] = {default_meth: {"batchable": True, "batch_dim": (1, 0)}}
        meths = [default_meth]
    kwargs["labels"] = {
        "pytest-label-N4nr": "pytest-label-value-4mH7",
        "pytest-label-7q72": "pytest-label-value-3mDd",
    }
    kwargs["custom_objects"] = {"pytest-custom-object-r7BU": scaler}
    kwargs["metadata"] = {
        "pytest-metadata-vSW4": [0, 9, 2],
        "pytest-metadata-qJJ3": "Wy5M",
    }
    bento_model = framework.save_model(
        test_model.name,
        test_model.model,
        **kwargs,
    )

    for meth in meths:
        assert bento_model.info.signatures[meth] == ModelSignature.from_dict(kwargs["signatures"][meth])  # type: ignore

    assert bento_model.info.labels == kwargs["labels"]
    assert bento_model.custom_objects["pytest-custom-object-r7BU"].mean_[0] == 5.5
    assert bento_model.custom_objects["pytest-custom-object-r7BU"].var_[0] == 4.75
    assert bento_model.info.metadata == kwargs["metadata"]


def test_get(
    framework: types.ModuleType,
    test_model: FrameworkTestModel,
    saved_model: bentoml.Model,
):
    # test that the generic get API works
    bento_model = framework.get(saved_model.tag)

    assert bento_model == saved_model
    assert bento_model.info.name == test_model.name

    bento_model_from_str = framework.get(str(saved_model.tag))
    assert bento_model == bento_model_from_str


def test_get_runnable(
    framework: types.ModuleType,
    saved_model: bentoml.Model,
):
    runnable = framework.get_runnable(saved_model)

    assert isinstance(
        runnable, t.Type
    ), "get_runnable for {bento_model.info.name} does not return a type"
    assert issubclass(
        runnable, bentoml.Runnable
    ), "get_runnable for {bento_model.info.name} doesn't return a subclass of bentoml.Runnable"
    assert (
        len(runnable.bentoml_runnable_methods__) > 0
    ), "get_runnable for {bento_model.info.name} gives a runnable with no methods"


def test_load(
    framework: types.ModuleType,
    test_model: FrameworkTestModel,
    saved_model: bentoml.Model,
):
    for configuration in test_model.configurations:
        model = framework.load_model(saved_model)
        configuration.check_model(model, {})
        if test_model.model_method_caller:
            for meth, inp in [
                (m, _i) for m, i in configuration.test_inputs.items() for _i in i
            ]:
                inp.check_output(
                    test_model.model_method_caller(
                        test_model, meth, tuple(inp.input_args), inp.input_kwargs
                    )
                )


def test_runnable(
    test_model: FrameworkTestModel,
    saved_model: bentoml.Model,
):
    for config in test_model.configurations:
        runner = saved_model.with_options(**config.load_kwargs).to_runner()
        runner.init_local()
        runner_handle = t.cast(LocalRunnerRef, runner._runner_handle)
        runnable = runner_handle._runnable
        config.check_runnable(runnable, {})
        runner.destroy()


def test_runner_batching(
    test_model: FrameworkTestModel,
    saved_model: bentoml.Model,
):
    from bentoml._internal.runner.utils import Params
    from bentoml._internal.runner.utils import payload_paramss_to_batch_params
    from bentoml._internal.runner.container import AutoContainer

    ran_tests = False

    for config in test_model.configurations:
        runner = saved_model.with_options(**config.load_kwargs).to_runner()
        runner.init_local()
        for meth, inputs in config.test_inputs.items():
            if not getattr(runner, meth).config.batchable:
                continue

            if len(inputs) < 2:
                continue

            ran_tests = True

            batch_dim = getattr(runner, meth).config.batch_dim
            paramss = [
                Params(*inp.input_args, **inp.input_kwargs).map(
                    # pylint: disable=cell-var-from-loop # lambda used before loop continues
                    lambda arg: AutoContainer.to_payload(arg, batch_dim=batch_dim[0])
                )
                for inp in inputs
            ]

            params, indices = payload_paramss_to_batch_params(paramss, batch_dim[0])

            batch_res = getattr(runner, meth).run(*params.args, **params.kwargs)

            outps = AutoContainer.batch_to_payloads(batch_res, indices, batch_dim[1])

            for i, outp in enumerate(outps):
                inputs[i].check_output(AutoContainer.from_payload(outp))

        runner.destroy()

    if not ran_tests:
        pytest.skip(
            "skipping batching tests because no configuration had multiple test inputs"
        )


def test_runner_cpu_multi_threading(
    framework: types.ModuleType,
    test_model: FrameworkTestModel,
    saved_model: bentoml.Model,
):
    resource_cfg = {"cpu": 2.0}
    ran_tests = False
    for config in test_model.configurations:
        model_with_options = saved_model.with_options(**config.load_kwargs)

        runnable: t.Type[bentoml.Runnable] = framework.get_runnable(model_with_options)
        if "cpu" not in runnable.SUPPORTED_RESOURCES:
            continue

        ran_tests = True

        runner = Runner(runnable)

        for meth, inputs in config.test_inputs.items():
            strategy = DefaultStrategy()

            os.environ.update(strategy.get_worker_env(runnable, resource_cfg, 1, 0))

            runner.init_local()

            runner_handle = t.cast(LocalRunnerRef, runner._runner_handle)
            runnable = runner_handle._runnable
            config.check_runnable(runnable, resource_cfg)
            if (
                hasattr(runnable, "model") and runnable.model is not None
            ):  # TODO: add a get_model to test models
                config.check_model(runnable.model, resource_cfg)

            for inp in inputs:
                outp = getattr(runner, meth).run(*inp.input_args, **inp.input_kwargs)
                inp.check_output(outp)

            runner.destroy()

    if not ran_tests:
        pytest.skip(
            f"no configurations for model '{test_model.name}' supported multiple CPU threads"
        )


def test_runner_cpu(
    framework: types.ModuleType,
    test_model: FrameworkTestModel,
    saved_model: bentoml.Model,
):
    resource_cfg = {"cpu": 1.0}

    ran_tests = False
    for config in test_model.configurations:
        model_with_options = saved_model.with_options(**config.load_kwargs)

        runnable: t.Type[bentoml.Runnable] = framework.get_runnable(model_with_options)
        if not runnable.SUPPORTS_CPU_MULTI_THREADING:
            continue

        ran_tests = True

        runner = Runner(runnable)

        for meth, inputs in config.test_inputs.items():
            strategy = DefaultStrategy()

            os.environ.update(strategy.get_worker_env(runnable, resource_cfg, 1, 0))

            runner.init_local()

            runner_handle = t.cast(LocalRunnerRef, runner._runner_handle)
            runnable = runner_handle._runnable
            config.check_runnable(runnable, resource_cfg)

            if (
                hasattr(runnable, "model") and runnable.model is not None
            ):  # TODO: add a get_model to test models
                config.check_model(runnable.model, resource_cfg)

            for inp in inputs:
                outp = getattr(runner, meth).run(*inp.input_args, **inp.input_kwargs)
                inp.check_output(outp)

            runner.destroy()

    if not ran_tests:
        pytest.skip(
            f"no configurations for model '{test_model.name}' supported multiple CPU threads"
        )


@pytest.mark.requires_gpus
def test_runner_nvidia_gpu(
    framework: types.ModuleType,
    test_model: FrameworkTestModel,
    saved_model: bentoml.Model,
):
    resource_cfg = {"nvidia.com/gpu": 1}

    ran_tests = False
    for config in test_model.configurations:
        model_with_options = saved_model.with_options(**config.load_kwargs)

        runnable: t.Type[bentoml.Runnable] = framework.get_runnable(model_with_options)
        if "nvidia.com/gpu" not in runnable.SUPPORTED_RESOURCES:
            continue

        ran_tests = True

        runner = Runner(runnable)

        for meth, inputs in config.test_inputs.items():
            strategy = DefaultStrategy()

            os.environ.update(strategy.get_worker_env(runnable, resource_cfg, 1, 0))

            runner.init_local()

            runner_handle = t.cast(LocalRunnerRef, runner._runner_handle)
            runnable = runner_handle._runnable

            config.check_runnable(runnable, resource_cfg)
            if (
                hasattr(runnable, "model") and runnable.model is not None
            ):  # TODO: add a get_model to test models
                config.check_model(runnable.model, resource_cfg)

            for inp in inputs:
                outp = getattr(runner, meth).run(*inp.input_args, **inp.input_kwargs)
                inp.check_output(outp)

            runner.destroy()

    if not ran_tests:
        pytest.skip(
            f"no configurations for model '{test_model.name}' supported running on Nvidia GPU"
        )
