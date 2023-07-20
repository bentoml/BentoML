from __future__ import annotations

import typing as t

import click

if t.TYPE_CHECKING:
    TupleStrAny = tuple[str, ...]
    from bentoml._internal.cloud.schemas import DeploymentListSchema
    from bentoml._internal.cloud.schemas import DeploymentSchema

    from .utils import SharedOptions
else:
    TupleStrAny = tuple


def add_deployment_command(cli: click.Group) -> None:
    import json

    from rich.table import Table

    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.utils import bentoml_cattr
    from bentoml._internal.utils import rich_console as console
    from bentoml_cli.utils import BentoMLCommandGroup

    client = BentoMLContainer.bentocloud_client.get()
    output_option = click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "default"]),
        default="default",
        help="Display the output of this command.",
    )

    def shared_decorator(f: t.Callable[..., t.Any]):
        options = [
            click.argument(
                "deployment-name",
                type=click.STRING,
                required=True,
            ),
            click.option(
                "--cluster-name",
                type=click.STRING,
                default=None,
                help="Name of the cluster.",
            ),
            click.option(
                "--kube-namespace",
                type=click.STRING,
                default=None,
                help="Kubernetes namespace.",
            ),
            output_option,
        ]
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
        help="JSON file path for the deployment configuration",
    )
    @output_option
    @click.pass_obj
    def create(  # type: ignore
        shared_options: SharedOptions,
        file: str,
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Create a deployment on BentoCloud.

        \b
        A deployment can be created using a json file with configurations.
        The json file has the exact format as the one on BentoCloud Deployment UI.
        """
        res = client.deployment.create_from_file(
            path_or_stream=file, context=shared_options.cloud_context
        )
        if output == "default":
            console.print(res)
        elif output == "json":
            click.echo(json.dumps(bentoml_cattr.unstructure(res), indent=2))
        return res

    @deployment_cli.command()
    @click.option(
        "-f",
        "--file",
        type=click.File(),
        help="JSON file path for the deployment configuration",
    )
    @click.option("-n", "--name", type=click.STRING, help="Deployment name")
    @click.option("--bento", type=click.STRING, help="Bento tag")
    @output_option
    @click.pass_obj
    def update(  # type: ignore
        shared_options: SharedOptions,
        file: str | None,
        name: str | None,
        bento: str | None,
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Update a deployment on BentoCloud.

        \b
        A deployment can be updated using a json file with needed configurations.
        The json file has the exact format as the one on BentoCloud Deployment UI.
        """
        if file is not None:
            if name is not None:
                click.echo("Reading from file, ignoring --name", err=True)
            res = client.deployment.update_from_file(
                path_or_stream=file, context=shared_options.cloud_context
            )
        elif name is not None:
            res = client.deployment.update(
                name,
                bento=bento,
                context=shared_options.cloud_context,
                latest_bento=True,
            )
        else:
            raise click.BadArgumentUsage(
                "Either --file or --name is required for update command"
            )
        if output == "default":
            console.print(res)
        elif output == "json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res

    @deployment_cli.command()
    @shared_decorator
    @click.pass_obj
    def get(  # type: ignore
        shared_options: SharedOptions,
        deployment_name: str,
        cluster_name: str,
        kube_namespace: str,
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Get a deployment on BentoCloud."""
        res = client.deployment.get(
            deployment_name=deployment_name,
            context=shared_options.cloud_context,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
        )
        if output == "default":
            console.print(res)
        elif output == "json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res

    @deployment_cli.command()
    @shared_decorator
    @click.pass_obj
    def terminate(  # type: ignore
        shared_options: SharedOptions,
        deployment_name: str,
        cluster_name: str,
        kube_namespace: str,
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Terminate a deployment on BentoCloud."""
        res = client.deployment.terminate(
            deployment_name=deployment_name,
            context=shared_options.cloud_context,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
        )
        if output == "default":
            console.print(res)
        elif output == "json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res

    @deployment_cli.command()
    @shared_decorator
    @click.pass_obj
    def delete(  # type: ignore
        shared_options: SharedOptions,
        deployment_name: str,
        cluster_name: str,
        kube_namespace: str,
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Delete a deployment on BentoCloud."""
        res = client.deployment.delete(
            deployment_name=deployment_name,
            context=shared_options.cloud_context,
            cluster_name=cluster_name,
            kube_namespace=kube_namespace,
        )
        if output == "default":
            console.print(res)
        elif output == "json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res

    @deployment_cli.command()
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
    @click.option(
        "-o",
        "--output",
        help="Display the output of this command.",
        type=click.Choice(["json", "default", "table"]),
        default="table",
    )
    @click.pass_obj
    def list(  # type: ignore
        shared_options: SharedOptions,
        cluster_name: str,
        query: str,
        search: str,
        count: int,
        start: int,
        output: t.Literal["json", "default", "table"],
    ) -> DeploymentListSchema:
        """List existing deployments on BentoCloud."""
        res = client.deployment.list(
            context=shared_options.cloud_context,
            cluster_name=cluster_name,
            query=query,
            search=search,
            count=count,
            start=start,
        )
        if output == "table":
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
                    deployment.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                )
            console.print(table)
        elif output == "default":
            console.print(res)
        elif output == "json":
            unstructured = bentoml_cattr.unstructure(res)
            click.echo(json.dumps(unstructured, indent=2))
        return res
