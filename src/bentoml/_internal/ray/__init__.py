from __future__ import annotations

import typing as t

import requests
from ray.serve._private.http_util import ASGIHTTPSender

import bentoml
from bentoml import Tag

from ...exceptions import MissingDependencyException
from .runner_handle import RayRunnerHandle

try:
    from ray import serve
    from ray.runtime_env import RuntimeEnv
    from ray.serve.deployment import Deployment
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """'ray[serve]' is required in order to use module 'bentoml.ray', install with 'pip install -U "ray[serve]"'. See https://docs.ray.io/ for more information.""",
    )


def _get_runner_deployment(
    svc: bentoml.Service,
    runner_name: str,
    runner_deployment_config: dict[str | t.Any] | None = None,
) -> Deployment:
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
    return serve.deployment(
        RunnerDeployment, name=f"bento-runner-{runner_name}", **runner_deployment_config
    )


def _get_service_deployment(svc: bentoml.Service, **kwargs) -> Deployment:
    @serve.deployment(name=f"bento-svc-{svc.name}", **kwargs)
    class BentoDeployment:
        def __init__(self, **runner_deployments: dict[str, Deployment]):
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


def _deploy_bento_runners(
    svc: bentoml.Service, runners_deployment_config_map: dict | None = None
) -> dict[str, Deployment]:
    # Deploy BentoML Runners as Ray serve deployments
    runner_deployments = {}
    for runner in svc.runners:
        runner_deployment_config = (
            runners_deployment_config_map.get(runner.name)
            if runner.name in runners_deployment_config_map
            else {}
        )
        runner_deployment = _get_runner_deployment(
            svc, runner.name, runner_deployment_config
        ).bind()
        runner_deployments[runner.name] = runner_deployment

    return runner_deployments


def deployment(
    target: str | Tag | bentoml.Bento | bentoml.Service,
    service_deployment_config: dict[str | t.Any] | None = None,
    runners_deployment_config_map: dict[str | dict[str | t.Any]] | None = None,
) -> Deployment:
    """
    Deploy a Bento or bentoml.Service to Ray

    Args:
        target: A bentoml.Service instance, Bento tag string, or Bento object
        service_deployment_config: Ray deployment config for BentoML API server
        runners_deployment_config_map: Ray deployment config map for all Runners

    Returns:
        A bound ray.serve.Deployment instance

    Example:

    .. code-block:: python
        :caption: `ray_demo.py`
        import bentoml

        classifier = bentoml.ray.deployment('iris_classifier:latest')

    .. code-block:: bash

        serve run ray_demo:classifier

    Configure BentoML-on-Ray deployment:

    .. code-block:: python
        import bentoml

        classifier = bentoml.ray.deployment(
            'iris_classifier:latest',
            {"route_prefix": "/hello", "num_replicas": 3, "ray_actor_options": {"num_cpus": 1}},
            {"iris_clf": {"num_replicas": 1, "ray_actor_options": {"num_cpus": 5}}}
        )
    """
    # TODO: validate Ray deployment options
    service_deployment_config = service_deployment_config or {}
    runners_deployment_config_map = runners_deployment_config_map or {}

    svc = target if isinstance(target, bentoml.Service) else bentoml.load(target)

    runner_deployments = _deploy_bento_runners(svc, runners_deployment_config_map)

    return _get_service_deployment(svc, **service_deployment_config).bind(
        **runner_deployments
    )
