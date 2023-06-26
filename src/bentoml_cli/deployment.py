from __future__ import annotations

import click
import typing as t

if t.TYPE_CHECKING:
    TupleStrAny = tuple[str, ...]
    from bentoml._internal.cloud.schemas import DeploymentSchema
    from bentoml._internal.cloud.schemas import DeploymentListSchema
else:
    TupleStrAny = tuple


def add_deployment_command(cli: click.Group) -> None:
    import json
    from bentoml_cli.utils import BentoMLCommandGroup
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.utils import rich_console as console, bentoml_cattr
    from rich.table import Table

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
                "--context", type=click.STRING, default=None, help="Yatai context name."
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
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    @output_option
    def create(  # type: ignore
        file: str,
        context: str,
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Create a deployment on BentoCloud.

        \b
        A deployment can be created using a json file with configurations.
        The json file has the exact format as the one on BentoCloud Deployment UI.
        """
        res = client.deployment.create_from_file(path_or_stream=file, context=context)
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
    @click.option(
        "--context", type=click.STRING, default=None, help="Yatai context name."
    )
    @output_option
    def update(  # type: ignore
        file: str,
        context: str,
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Update a deployment on BentoCloud.

        \b
        A deployment can be updated using a json file with needed configurations.
        The json file has the exact format as the one on BentoCloud Deployment UI.
        """
        res = client.deployment.update_from_file(
            path_or_stream=file,
            context=context,
        )
        if output == "default":
            console.print(res)
        elif output == "json":
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
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Get a deployment on BentoCloud."""
        res = client.deployment.get(
            deployment_name=deployment_name,
            context=context,
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
    def terminate(  # type: ignore
        deployment_name: str,
        context: str,
        cluster_name: str,
        kube_namespace: str,
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Terminate a deployment on BentoCloud."""
        res = client.deployment.terminate(
            deployment_name=deployment_name,
            context=context,
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
    def delete(  # type: ignore
        deployment_name: str,
        context: str,
        cluster_name: str,
        kube_namespace: str,
        output: t.Literal["json", "default"],
    ) -> DeploymentSchema:
        """Delete a deployment on BentoCloud."""
        res = client.deployment.delete(
            deployment_name=deployment_name,
            context=context,
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
    @click.option(
        "-o",
        "--output",
        help="Display the output of this command.",
        type=click.Choice(["json", "default", "table"]),
        default="table",
    )
    def list(  # type: ignore
        context: str,
        cluster_name: str,
        query: str,
        search: str,
        count: int,
        start: int,
        output: t.Literal["json", "default", "table"],
    ) -> DeploymentListSchema:
        """List existing deployments on BentoCloud."""
        res = client.deployment.list(
            context=context,
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
