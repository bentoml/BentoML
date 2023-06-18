from __future__ import annotations

import click
import typing as t
from bentoml._internal.types import LazyType
if t.TYPE_CHECKING:
    TupleStrAny = tuple[str, ...]
    from bentoml._internal.cloud.schemas import DeploymentTargetType
    from bentoml._internal.cloud.schemas import DeploymentSchema
    from bentoml._internal.cloud.schemas import DeploymentMode
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
        ]
        self._nargs_parser: click.parser.Option

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


class Mutex(click.Option):
    """An option that supports mutually exclusive required options.
    Derived from https://stackoverflow.com/a/51235564
    """
    def __init__(self, *args, **kwargs):
        self.must_exist_if_none: list[str] = kwargs.pop("must_exist_if_none", None)
        self.cannot_exist_together: list[str] = kwargs.pop("cannot_exist_together", None)
        assert self.must_exist_if_none, "'must_exist_if_none' parameter required"
        assert self.cannot_exist_together, "'cannot_exist_together' parameter required"
        kwargs["help"] = (kwargs.get("help", "") + "Option is mutually exclusive with " + ", ".join(self.cannot_exist_together) + ".").strip()
        super(Mutex, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        current_opt: bool = self.name in opts
        if current_opt:
            for mutex_opt in self.cannot_exist_together:
                if mutex_opt in opts:
                    raise click.UsageError("Illegal usage: '" + str(self.name) + "' is mutually exclusive with '" + str(mutex_opt) + "'.")
        else:
            for mutex_opt in self.must_exist_if_none:
                if mutex_opt not in opts:
                    raise click.UsageError(f"Either {['--'+i for i in self.must_exist_if_none]} or {['--' + self.name]} is required for this operation")
        self.prompt = None
        return super(Mutex, self).handle_parse_result(ctx, opts, args)

def add_deployment_command(cli: click.Group) -> None:
    import json
    from bentoml_cli.utils import BentoMLCommandGroup
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.cloud.deployment import Resource
    from bentoml._internal.utils import rich_console as console, bentoml_cattr
    from rich.table import Table

    bento_store = BentoMLContainer.bento_store.get()
    client = BentoMLContainer.bentocloud_client.get()
    output_option = click.option("-o", '--output', type=click.Choice(['json', 'default']), default='default')

    def shared_decorator(f: t.Callable[..., t.Any]):
        options = [click.argument(
        "deployment_name",
        type=click.STRING,
        required=True,
    ),
    click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    ),
    click.option(
        "--cluster-name", type=click.STRING, default=None, help="Name of the cluster."
    ),
    click.option(
        "--kube-namespace",
        type=click.STRING,
        default=None,
        help="Kubernetes namespace.",
    ),
    output_option]
        for opt in reversed(options):
            f = opt(f)
        return f


    @cli.group(name="deployment", cls=BentoMLCommandGroup)
    def deployment_cli():
        """Deployment Subcommands Groups"""

    @deployment_cli.command()
    @click.option(
        "-f",
        "--file",
        type=click.File(),
        default=None,
        help="Create deployment using json file. ",
        cls=Mutex,
        must_exist_if_none=["deployment_name", 'bento'],
        cannot_exist_together=[
            "deployment_name",
            "bento",
            "description",
            "expose_endpoint",
            "kube_namespace",
            "resource_instance",
            "min_replicas",
            "max_replicas",
            "type",
            "mode",
            "runner",
            "force",
            "threads",
            "push"
        ],
    )
    @click.option(
        "--deployment-name",
        type=click.STRING,
        help="Name of the deployment.",
    )
    @click.option(
        "--bento", type=click.STRING, help="Bento tag or name.",
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
        type=click.Choice("deployment", "function"),
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
    @click.option(
        "-p",
        "--push",
        is_flag=True,
        default=False,
        help="Push the bento for deployment if it doesn't exist in bento cloud",
    )
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Forced push to yatai even if it exists in bento cloud",
    )
    @click.option(
        "-t",
        "--threads",
        default=10,
        help="Number of threads to use for upload",
    )
    @output_option
    def create(  # type: ignore
        deployment_name: str,
        bento: str,
        file: str,
        context: str,
        description: str,
        expose_endpoint: bool,
        cluster_name: str,
        kube_namespace: str,
        resource_instance: str,
        min_replicas: int,
        max_replicas: int,
        type: DeploymentTargetType,
        mode: DeploymentMode,
        runner: dict[str, t.Any],
        output: t.Literal["json", "default"],
        force: bool, threads: int, push: bool
    ) -> DeploymentSchema:
        """Create a deployment."""
        if file is not None:
            return client.deployment.create_from_file(
                path_or_stream=file, context=context, cluster_name=cluster_name
            )
        hpa_conf = Resource.for_hpa_conf(
            min_replicas=min_replicas, max_replicas=max_replicas
        )
        runners_conf = {
                i["runner-name"]: Resource.for_runner(
                    hpa_conf={
                        "min_replicas": i.get("min-replicas"),
                        "max_replicas": i.get("max-replicas"),
                    },
                    resource_instance=i.get("resource-instance"),
                )
                for i in runner
            } if runner else None
        if push:
            bento_obj = bento_store.get(bento)
            if not bento_obj:
                raise click.ClickException(f"Bento {bento} not found in local store")
            client.push_bento(
            bento_obj, force=force, threads=threads, context=context
            )

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
        if output =="default":
            console.print(res)
        elif output =="json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res

    @deployment_cli.command()
    @click.option(
        "--deployment-name",
        type=click.STRING,
        help="Name of the deployment.",
    )
    @click.option(
        "--bento", type=click.STRING, help="Bento tag or name.",
    )
    @click.option(
        "-f",
        "--file",
        type=click.File(),
        default=None,
        help="Create deployment using json file. ",
        cls=Mutex,
        must_exist_if_none=["deployment_name"],
        cannot_exist_together=[
            "deployment_name",
            "bento",
            "description",
            "expose_endpoint",
            "kube_namespace",
            "resource_instance",
            "min_replicas",
            "max_replicas",
            "type",
            "mode",
            "runner",
        ],
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
        type=click.Choice("deployment", "function"),
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
    @output_option
    def update(  # type: ignore
        deployment_name: str,
        bento: str,
        file: str,
        context: str,
        description: str,
        expose_endpoint: bool,
        cluster_name: str,
        kube_namespace: str,
        resource_instance: str,
        min_replicas: int,
        max_replicas: int,
        type: DeploymentTargetType,
        mode: DeploymentMode,
        runner: dict[str, t.Any],
        output: t.Literal["json", "default"]
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
                        "min_replicas": i.get("min-replicas"),
                        "max_replicas": i.get("max-replicas"),
                    },
                    resource_instance=i.get("resource-instance"),
                )
                for i in runner
            } if runner else None
        # Updated schema
        breakpoint()
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
        if output=="default":
            console.print(res)
        elif output=="json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res

    @deployment_cli.command()
    @shared_decorator
    def get(  # type: ignore
        deployment_name: str,
        context: str,
        cluster_name: str,
        kube_namespace: str,
        output: t.Literal["json", "default"]
    ) -> DeploymentSchema:
        res = client.deployment.get(deployment_name=deployment_name,
                                context=context,
                                cluster_name=cluster_name,
                                kube_namespace=kube_namespace)
        if output=="default":
            console.print(res)
        elif output=="json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res

    @deployment_cli.command()
    @shared_decorator
    def terminate(  # type: ignore
        deployment_name: str,
        context: str,
        cluster_name: str,
        kube_namespace: str,
        output: t.Literal["json", "default"]
    ) -> DeploymentSchema:
        res = client.deployment.terminate(deployment_name=deployment_name,
                                context=context,
                                cluster_name=cluster_name,
                                kube_namespace=kube_namespace)
        if output=="default":
            console.print(res)
        elif output=="json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res

    @deployment_cli.command()
    @shared_decorator
    def delete(  # type: ignore
        deployment_name: str,
        context: str,
        cluster_name: str,
        kube_namespace: str,
        output: t.Literal["json", "default"]
    ) -> DeploymentSchema:
        res = client.deployment.delete(deployment_name=deployment_name,
                                context=context,
                                cluster_name=cluster_name,
                                kube_namespace=kube_namespace)
        if output=="default":
            console.print(res)
        elif output=="json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
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
    @click.option("-o", '--output', help='display the output of this command', type=click.Choice(['json', 'default', 'table']), default = "table")
    def list(  # type: ignore
        context: str,
        cluster_name: str,
        query: str,
        search: str,
        count: str,
        start: str,
        output: t.Literal['json', 'default', 'table'],
    ) -> DeploymentSchema:
        res = client.deployment.list(context=context,
                                cluster_name=cluster_name,
                                query=query,
                                search=search,
                                count=count,
                                start=start,)
        if output=="table":
            table = Table(box=None)
            table.add_column("Deployment")
            table.add_column("Bento")
            table.add_column("Status")
            table.add_column("Created At")
            for deployment in res.items:
                target = deployment.latest_revision.targets[0]
                table.add_row(
                    deployment.name,
                    f"{target.bento.repository.name}:{target.bento.name}",
                    deployment.status.value,
                    deployment.created_at.astimezone().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                )
            console.print(table)
        elif output=="default":
            console.print(res)
        elif output=="json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res
