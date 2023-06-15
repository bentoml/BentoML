from __future__ import annotations

import click
import typing as t
from bentoml._internal.types import LazyType
from bentoml._internal.cloud.deployment import Resource
from bentoml._internal.cloud.schemas import DeploymentMode
from bentoml._internal.cloud.schemas import DeploymentTargetType
from bentoml._internal.cloud.schemas import DeploymentSchema
from bentoml._internal.utils import rich_console
if t.TYPE_CHECKING:
    TupleStrAny = tuple[str, ...]
else:
    TupleStrAny = tuple


class NargsOptions(click.Option):
    """An option that supports nargs=-1.
    Derived from https://stackoverflow.com/a/48394004/8643197

    We mk add_to_parser to handle multiple value that is passed into this specific
    options.
    """

    def __init__(self, *args: t.Any, **attrs: t.Any):
        nargs = attrs.pop("nargs", -1)
        if nargs != -1:
            raise (f"'nargs' is set, and must be -1 instead of {nargs}")
        super(NargsOptions, self).__init__(*args, **attrs)
        self._prev_parser_process: t.Callable[
            [t.Any, click.parser.ParsingState], None
        ] | None = None
        self._nargs_parser: click.parser.Option | None = None

    def add_to_parser(self, parser: click.OptionParser, ctx: click.Context) -> None:
        def _parser(value: t.Any, state: click.parser.ParsingState):
            # method to hook to the parser.process
            done = False
            value = [value]
            # grab everything up to the next option
            assert self._nargs_parser is not None
            while state.rargs and not done:
                for prefix in self._nargs_parser.prefixes:
                    if state.rargs[0].startswith(prefix):
                        done = True
                if not done:
                    value.append(state.rargs.pop(0))

            value = tuple(value)

            # call the actual process
            assert self._prev_parser_process is not None
            self._prev_parser_process(value, state)

        retval = super(NargsOptions, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._nargs_parser = our_parser
                self._prev_parser_process = our_parser.process
                our_parser.process = _parser
                break
        return retval


def parse_runners_callback(
    _: click.Context, params: click.Parameter, value: tuple[str, ...] | None
) -> t.Any:
    if value is None:
        return value
    if not LazyType(TupleStrAny).isinstance(value):
        raise RuntimeError(f"{params} only accept multiple values.")
    parsedLs: list[dict[str, t.Any]] = list()
    for v in value:
        parsed: tuple[str, ...] = tuple()
        v = v[1:-1]
        if v == ",":
            continue
        if "," in v:
            parsed += tuple(v.split(","))
        else:
            parsed += tuple(v.split())
        parsedLs.append(
            dict(map(lambda x: x.strip()[1:-1].split("="), filter(lambda x: x, parsed)))
        )
        # Remove trailing space then single quotation mark in front and end, split by = to make a key,val pair
        # --runner a b c d
        # ("'a'", "'b'")
    return parsedLs


def add_deployment_command(cli: click.Group) -> None:
    from bentoml_cli.utils import BentoMLCommandGroup
    from bentoml._internal.configuration.containers import BentoMLContainer

    client = BentoMLContainer.bentocloud_client.get()

    @cli.group(name="deployment", cls=BentoMLCommandGroup)
    def deployment_cli():
        """Deployment Subcommands Groups"""

    @deployment_cli.command()
    @click.option(
        "-f",
        "--file",
        type=click.Path(exists=True),
        default=None,
        help="Create deployment using json file",
    )
    @click.option(
        "--deployment-name",
        type=click.STRING,
        required=True,
        help="Name of the deployment.",
    )
    @click.option(
        "--bento", type=click.STRING, required=True, help="Bento tag or name."
    )
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    @click.option(
        "--description",
        type=click.STRING,
        default=None,
        help="Description of the deployment.",
    )
    @click.option(
        "--expose-endpoint",
        type=click.BOOL,
        default=None,
        help="Expose endpoint or not.",
    )
    @click.option(
        "--cluster-name", type=click.STRING, default=None, help="Name of the cluster."
    )
    @click.option(
        "--kube-namespace",
        type=click.STRING,
        default=None,
        help="Kubernetes namespace.",
    )
    @click.option(
        "--resource-instance",
        type=click.STRING,
        default=None,
        help="Resource instance.",
    )
    @click.option(
        "--min-replicas", type=click.INT, default=None, help="Minimum replicas."
    )
    @click.option(
        "--max-replicas", type=click.INT, default=None, help="Maximum replicas."
    )
    @click.option(
        "--mode",
        type=click.Choice(DeploymentMode),
        default=None,
        help="Deployment mode.",
    )
    @click.option(
        "--type",
        type=click.Choice(["stable", "canary"]),
        default="stable",
        help="Deployment target type.",
    )
    @click.option(
        "--runner",
        cls=NargsOptions,
        multiple=True,
        callback=parse_runners_callback,
        default=None,
        help="Runners.",
    )
    def create(  # type: ignore
        deployment_name: str,
        bento: str,
        file: str | None = None,
        context: str | None = None,
        description: str | None = None,
        expose_endpoint: bool | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
        resource_instance: str | None = None,
        min_replicas: int | None = None,
        max_replicas: int | None = None,
        type: DeploymentTargetType | None = None,
        mode: DeploymentMode | None = None,
        runner: dict[str, t.Any] | None = None,
    ) -> DeploymentSchema:
        """Create a deployment."""
        if file is not None:
            return client.deployment.create_from_file(
                path=file, context=context, cluster_name=cluster_name
            )
        hpa_conf = Resource.for_hpa_conf(
            min_replicas=min_replicas, max_replicas=max_replicas
        )
        runners_conf = {
                i["runner-name"]: Resource.for_runner(
                    hpa_conf={
                        "min-replicas": i.get("min-replicas"),
                        "max_replicas": i.get("max-replicas"),
                    },
                    resource_instance=i.get("resource-instance"),
                )
                for i in runner
            } if len(runner) > 0 else None

        res = client.deployment.create(
            deployment_name=deployment_name,
            bento=bento,
            expose_endpoint=expose_endpoint,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
            resource_instance=resource_instance,
            context=context,
            description=description,
            type=type,
            mode=mode,
            runners_config=runners_conf,
            hpa_conf=hpa_conf,
        )
        rich_console.print(res)
        return res

    @deployment_cli.command()
    @click.option(
        "-f",
        "--file",
        type=click.Path(exists=True),
        default=None,
        help="Create deployment using json file",
    )
    @click.option(
        "--deployment-name",
        type=click.STRING,
        required=True,
        help="Name of the deployment.",
    )
    @click.option(
        "--bento", type=click.STRING, default=None, help="Bento tag or name."
    )
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    @click.option(
        "--description",
        type=click.STRING,
        default=None,
        help="Description of the deployment.",
    )
    @click.option(
        "--expose-endpoint",
        type=click.BOOL,
        default=None,
        help="Expose endpoint or not.",
    )
    @click.option(
        "--cluster-name", type=click.STRING, default=None, help="Name of the cluster."
    )
    @click.option(
        "--kube-namespace",
        type=click.STRING,
        default=None,
        help="Kubernetes namespace.",
    )
    @click.option(
        "--resource-instance",
        type=click.STRING,
        default=None,
        help="Resource instance.",
    )
    @click.option(
        "--min-replicas", type=click.INT, default=None, help="Minimum replicas."
    )
    @click.option(
        "--max-replicas", type=click.INT, default=None, help="Maximum replicas."
    )
    @click.option(
        "--mode",
        type=click.Choice(DeploymentMode),
        default=None,
        help="Deployment mode.",
    )
    @click.option(
        "--type",
        type=click.Choice(["stable", "canary"]),
        default="stable",
        help="Deployment target type.",
    )
    @click.option(
        "--runner",
        cls=NargsOptions,
        multiple=True,
        callback=parse_runners_callback,
        default=None,
        help="Runners.",
    )
    def update(  # type: ignore
        deployment_name: str,
        bento: str | None = None,
        file: str | None = None,
        context: str | None = None,
        description: str | None = None,
        expose_endpoint: bool | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
        resource_instance: str | None = None,
        min_replicas: int | None = None,
        max_replicas: int | None = None,
        type: DeploymentTargetType | None = None,
        mode: DeploymentMode | None = None,
        runner: dict[str, t.Any] | None = None,
    ) -> DeploymentSchema:
        """Update a deployment"""
        if file is not None:
            return client.deployment.update_from_file(
                path=file, context=context, cluster_name=cluster_name
            )
        hpa_conf = Resource.for_hpa_conf(
            min_replicas=min_replicas, max_replicas=max_replicas
        )
        runners_conf = {
                i["runner-name"]: Resource.for_runner(
                    hpa_conf={
                        "min-replicas": i.get("min-replicas"),
                        "max_replicas": i.get("max-replicas"),
                    },
                    resource_instance=i.get("resource-instance"),
                )
                for i in runner
            } if len(runner) > 0 else None
        runners_conf = None
        # Updated schema
        res = client.deployment.update(
            deployment_name=deployment_name,
            bento=bento,
            expose_endpoint=expose_endpoint,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
            resource_instance=resource_instance,
            context=context,
            description=description,
            type=type,
            mode=mode,
            runners_config=runners_conf,
            hpa_conf=hpa_conf,
        )
        rich_console.print(res)
        return res

    @deployment_cli.command()
    @click.option(
        "--deployment-name",
        type=click.STRING,
        required=True,
        help="Name of the deployment.",
    )
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    @click.option(
        "--cluster-name", type=click.STRING, default=None, help="Name of the cluster."
    )
    @click.option(
        "--kube-namespace",
        type=click.STRING,
        default=None,
        help="Kubernetes namespace.",
    )
    def get(  # type: ignore
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> DeploymentSchema:
        res = client.deployment.get(deployment_name=deployment_name,
                              context=context,
                              cluster_name=cluster_name,
                              kube_namespace=kube_namespace)
        rich_console.print(res)

    @deployment_cli.command()
    @click.option(
        "--deployment-name",
        type=click.STRING,
        required=True,
        help="Name of the deployment.",
    )
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    @click.option(
        "--cluster-name", type=click.STRING, default=None, help="Name of the cluster."
    )
    @click.option(
        "--kube-namespace",
        type=click.STRING,
        default=None,
        help="Kubernetes namespace.",
    )
    def terminate(  # type: ignore
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> DeploymentSchema:
        res = client.deployment.terminate(deployment_name=deployment_name,
                              context=context,
                              cluster_name=cluster_name,
                              kube_namespace=kube_namespace)
        rich_console.print(res)
        return res

    @deployment_cli.command()
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    @click.option(
        "--cluster-name", type=click.STRING, default=None, help="Name of the cluster."
    )
    @click.option(
        "--kube-namespace",
        type=click.STRING,
        default=None,
        help="Kubernetes namespace.",
    )
    def delete(  # type: ignore
        deployment_name: str,
        context: str | None = None,
        cluster_name: str | None = None,
        kube_namespace: str | None = None,
    ) -> DeploymentSchema:
        res = client.deployment.delete(deployment_name=deployment_name,
                              context=context,
                              cluster_name=cluster_name,
                              kube_namespace=kube_namespace)
        rich_console.print(res)
        return res

    @deployment_cli.command()
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    @click.option(
        "--cluster-name", type=click.STRING, default=None, help="Name of the cluster."
    )
    @click.option(
        "--query", type=click.STRING, default=None, help="Query for list request."
    )
    @click.option(
        "--search", type=click.STRING, default=None, help="Search for list request."
    )
    @click.option(
        "--start", type=click.STRING, default=None, help="Start for list request."
    )
    @click.option(
        "--count", type=click.STRING, default=None, help="Count for list request."
    )
    def list(  # type: ignore
        context: str | None = None,
        cluster_name: str | None = None,
        query: str | None = None,
        search: str | None = None,
        count: str | None = None,
        start: str | None = None,
    ) -> DeploymentSchema:
        res = client.deployment.list(context=context,
                                     cluster_name=cluster_name,
                                     query=query,
                                     search=search,
                                     count=count,
                                     start=start)
        rich_console.print(res)
        return res
