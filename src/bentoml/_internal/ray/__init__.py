from __future__ import annotations

import typing as t

import requests
from ray.serve._private.http_util import ASGIHTTPSender

import bentoml
from bentoml import Tag
from bentoml._internal.runner import Runner

from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..bento.bento import BentoRunnerInfo
from .runner_handle import RayRunnerHandle

try:
    import ray
    from ray import serve
    from ray.runtime_env import RuntimeEnv
    from ray.serve.deployment import Deployment
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'ray[serve]' is required in order to use module 'bentoml.ray', install with 'pip install -U \"ray[serve]\"'. For more information, refer to https://docs.ray.io/",
    )


def _get_runner_deployment(svc: bentoml.Service, runner_name: str) -> Deployment:
    class RunnerDeployment:
        def __init__(self):
            self._runner = next(
                (runner for runner in svc.runners if runner.name == runner_name),
                None,
            )
            self._runner.init_local(quiet=True)
            for method in self._runner.runner_methods:

                async def _func(self, *args, **kwargs):
                    return await getattr(self._runner, method.name).async_run(
                        *args, **kwargs
                    )

                setattr(RunnerDeployment, method.name, _func)

    # TODO: apply RunnerMethod's max batch size configs
    return serve.deployment(RunnerDeployment, name=f"bento-runner-{runner_name}")


def _get_service_deployment(svc: bentoml.Service, *args, **kwargs) -> Deployment:
    @serve.deployment(name=f"bento-svc-{svc.name}", *args, **kwargs)
    class BentoDeployment:
        def __init__(self, **runner_deployments: t.Dict[str, Deployment]):
            from ..server.http_app import HTTPAppFactory

            # Use Ray's metrics setup. Ray has already initialized a prometheus client
            # for multi-process and this conflicts with BentoML's prometheus setup
            self.app = HTTPAppFactory(svc, enable_metrics=False)()

            for runner in svc.runners:
                assert runner.name in runner_deployments
                runner._set_handle(RayRunnerHandle, runner_deployments[runner.name])

        async def __call__(self, request: requests.Request):
            sender = ASGIHTTPSender()
            await self.app(request.scope, receive=request.receive, send=sender)
            return sender.build_asgi_response()

    return BentoDeployment


def get_bento_runtime_env(bento_tag: str | Tag) -> RuntimeEnv:
    # TODO: create Ray RuntimeEnv from Bento env configs
    return RuntimeEnv(
        pip=[],
        conda=[],
        env_vars=[],
    )


def _deploy_bento_runners(svc: bentoml.Service) -> t.Dict[str, Deployment]:
    # Deploy BentoML Runners as Ray serve deployments
    runner_deployments = {}
    for runner in svc.runners:
        runner_deployment = _get_runner_deployment(svc, runner.name).bind()
        runner_deployments[runner.name] = runner_deployment

    return runner_deployments


def deploy(
    target: str | Tag | bentoml.Service, ray_deployment_kwargs: t.Dict | None = None
) -> None:
    svc = target if isinstance(target, bentoml.Service) else bentoml.load(target)

    runner_deployments = _deploy_bento_runners(svc)
    ray_deployment_kwargs = ray_deployment_kwargs if ray_deployment_kwargs else {}

    return _get_service_deployment(svc, **ray_deployment_kwargs).bind(
        **runner_deployments
    )
