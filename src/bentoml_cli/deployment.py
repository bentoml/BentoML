from __future__ import annotations

import typing as t

import click
import yaml
from rich.syntax import Syntax

from bentoml._internal.cloud.deployment import get_args_from_config
from bentoml._internal.cloud.schemas.modelschemas import AccessControl
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStrategy

if t.TYPE_CHECKING:
    TupleStrAny = tuple[str, ...]

    from .utils import SharedOptions
else:
    TupleStrAny = tuple


def add_deployment_command(cli: click.Group) -> None:
    import json

    import click_option_group as cog
    from rich.table import Table

    from bentoml._internal.cloud.deployment import Deployment
    from bentoml._internal.cloud.deployment import get_real_bento_tag
    from bentoml._internal.utils import rich_console as console
    from bentoml_cli.utils import BentoMLCommandGroup

    @cli.command()
    @click.argument(
        "target",
        type=click.STRING,
        required=False,
    )
    @click.option(
        "-n",
        "--name",
        type=click.STRING,
        help="Deployment name",
    )
    @click.option(
        "--cluster",
        type=click.STRING,
        help="Name of the cluster",
    )
    @click.option(
        "--access-type",
        type=click.Choice(
            [access_ctrl_type.value for access_ctrl_type in AccessControl]
        ),
        help="Type of access",
    )
    @click.option(
        "--scaling-min",
        type=click.INT,
        help="Minimum scaling value",
    )
    @click.option(
        "--scaling-max",
        type=click.INT,
        help="Maximum scaling value",
    )
    @click.option(
        "--instance-type",
        type=click.STRING,
        help="Type of instance",
    )
    @click.option(
        "--strategy",
        type=click.Choice(
            [deployment_strategy.value for deployment_strategy in DeploymentStrategy]
        ),
        help="Deployment strategy",
    )
    @click.option(
        "--env",
        type=click.STRING,
        help="List of environment variables pass by --env key=value, --env ...",
        multiple=True,
    )
    @click.option(
        "-f",
        "--config-file",
        type=click.File(),
        help="Configuration file path",
        default=None,
    )
    @click.option(
        "--wait/--no-wait",
        type=click.BOOL,
        is_flag=True,
        help="Do not wait for deployment to be ready",
        default=True,
    )
    @click.option(
        "--timeout",
        type=click.INT,
        default=300,
        help="Timeout for deployment to be ready in seconds",
    )
    @click.pass_obj
    def deploy(
        shared_options: SharedOptions,
        target: str | None,
        name: str | None,
        cluster: str | None,
        access_type: str | None,
        scaling_min: int | None,
        scaling_max: int | None,
        instance_type: str | None,
        strategy: str | None,
        env: tuple[str] | None,
        config_file: str | t.TextIO | None,
        wait: bool,
        timeout: int,
    ) -> None:
        """Create a deployment on BentoCloud.

        \b
        Create a deployment using parameters (standalone mode only), or using config yaml file.
        """
        from os import path

        deploy_name, bento_name, cluster_name = get_args_from_config(
            name=name, bento=target, config_file=config_file, cluster=cluster
        )
        if bento_name is None:
            raise click.BadParameter(
                "please provide a target to deploy or a config file with bento name"
            )

        # determine if target is a path or a name
        if path.exists(bento_name):
            # target is a path
            click.echo(f"building bento from {target} ...")
            bento_tag = get_real_bento_tag(project_path=target)
        else:
            click.echo(f"using bento {target}...")
            bento_tag = get_real_bento_tag(bento=target)

        deployment = Deployment.create(
            bento=bento_tag,
            name=deploy_name,
            cluster=cluster_name,
            access_type=access_type,
            scaling_min=scaling_min,
            scaling_max=scaling_max,
            instance_type=instance_type,
            strategy=strategy,
            envs=[
                {"key": item.split("=")[0], "value": item.split("=")[1]} for item in env
            ]
            if env is not None
            else None,
            config_file=config_file,
            context=shared_options.cloud_context,
        )
        if wait:
            deployment.wait_until_ready(timeout=timeout)
            click.echo(
                f"Deployment '{deployment.name}' created successfully in cluster '{deployment.cluster}'."
            )
        click.echo(f"To check the deployment, go to: {deployment.admin_console}.")

    output_option = click.option(
        "-o",
        "--output",
        type=click.Choice(["yaml", "json"]),
        default="yaml",
        help="Display the output of this command.",
    )

    def shared_decorator(
        f: t.Callable[..., t.Any] | None = None
    ) -> t.Callable[..., t.Any]:
        def decorate(f: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
            options = [
                click.option(
                    "--cluster",
                    type=click.STRING,
                    default=None,
                    help="Name of the cluster.",
                ),
            ]
            for opt in reversed(options):
                f = opt(f)
            return f

        if f:
            return decorate(f)
        else:
            return decorate

    @cli.group(name="deployment", cls=BentoMLCommandGroup)
    def deployment_cli():
        """Deployment Subcommands Groups"""

    @deployment_cli.command()
    @shared_decorator()
    @cog.optgroup.group(cls=cog.MutuallyExclusiveOptionGroup, name="target options")
    @cog.optgroup.option(
        "--bento",
        type=click.STRING,
        help="Bento name",
    )
    @cog.optgroup.option(
        "--project-path",
        type=click.Path(exists=True),
        help="Path to the project",
    )
    @click.option(
        "--access-type",
        type=click.Choice(
            [access_ctrl_type.value for access_ctrl_type in AccessControl]
        ),
        help="Type of access",
    )
    @click.option(
        "--scaling-min",
        type=click.INT,
        help="Minimum scaling value",
    )
    @click.option(
        "--scaling-max",
        type=click.INT,
        help="Maximum scaling value",
    )
    @click.option(
        "--instance-type",
        type=click.STRING,
        help="Type of instance",
    )
    @click.option(
        "--strategy",
        type=click.Choice(
            [deployment_strategy.value for deployment_strategy in DeploymentStrategy]
        ),
        help="Deployment strategy",
    )
    @click.option(
        "--env",
        type=click.STRING,
        help="List of environment variables pass by --env key=value, --env ...",
        multiple=True,
    )
    @click.option(
        "-f",
        "--config-file",
        type=click.File(),
        help="Configuration file path, mututally exclusive with other config options",
        default=None,
    )
    @click.pass_obj
    def update(  # type: ignore
        shared_options: SharedOptions,
        name: str | None,
        cluster: str | None,
        project_path: str | None,
        bento: str | None,
        access_type: str | None,
        scaling_min: int | None,
        scaling_max: int | None,
        instance_type: str | None,
        strategy: str | None,
        env: tuple[str] | None,
        config_file: t.TextIO | None,
    ) -> None:
        """Update a deployment on BentoCloud.

        \b
        A deployment can be updated using parameters (standalone mode only), or using config yaml file.
        You can also update bento by providing a project path or existing bento.
        """
        deploy_name, bento_name, cluster_name = get_args_from_config(
            name=name, bento=bento, cluster=cluster, config_file=config_file
        )
        if bento_name is None and project_path is None:
            target = None
        else:
            target = get_real_bento_tag(
                project_path=project_path,
                bento=bento,
                context=shared_options.cloud_context,
            )

        Deployment.update(
            bento=target,
            access_type=access_type,
            name=deploy_name,
            cluster=cluster_name,
            scaling_min=scaling_min,
            scaling_max=scaling_max,
            instance_type=instance_type,
            strategy=strategy,
            envs=[
                {"key": item.split("=")[0], "value": item.split("=")[1]} for item in env
            ]
            if env is not None
            else None,
            config_file=config_file,
            context=shared_options.cloud_context,
        )

        click.echo(f"Deployment '{name}' updated successfully.")

    @deployment_cli.command()
    @shared_decorator()
    @cog.optgroup.group(cls=cog.MutuallyExclusiveOptionGroup, name="target options")
    @cog.optgroup.option(
        "--bento",
        type=click.STRING,
        help="Bento name",
    )
    @cog.optgroup.option(
        "--project-path",
        type=click.Path(exists=True),
        help="Path to the project",
    )
    @click.option(
        "-n",
        "--name",
        type=click.STRING,
        help="Deployment name",
    )
    @click.option(
        "-f",
        "--config-file",
        type=click.File(),
        help="Configuration file path, mututally exclusive with other config options",
        default=None,
    )
    @click.pass_obj
    def apply(  # type: ignore
        shared_options: SharedOptions,
        name: str | None,
        cluster: str | None,
        project_path: str | None,
        bento: str | None,
        config_file: str | t.TextIO | None,
    ) -> None:
        """Update a deployment on BentoCloud.

        \b
        A deployment can be updated using parameters (standalone mode only), or using config yaml file.
        You can also update bento by providing a project path or existing bento.
        """
        deploy_name, bento_name, cluster_name = get_args_from_config(
            name=name, bento=bento, cluster=cluster, config_file=config_file
        )
        if bento_name is None and project_path is None:
            target = None
        else:
            target = get_real_bento_tag(
                project_path=project_path,
                bento=bento,
                context=shared_options.cloud_context,
            )

        Deployment.apply(
            bento=target,
            name=deploy_name,
            cluster=cluster_name,
            config_file=config_file,
            context=shared_options.cloud_context,
        )

        click.echo(f"Deployment '{name}' updated successfully.")

    @deployment_cli.command()
    @shared_decorator
    @click.argument(
        "name",
        type=click.STRING,
        required=True,
    )
    @output_option
    @click.pass_obj
    def get(  # type: ignore
        shared_options: SharedOptions,
        name: str,
        cluster: str | None,
        output: t.Literal["json", "default"],
    ) -> None:
        """Get a deployment on BentoCloud."""
        d = Deployment.get(name, context=shared_options.cloud_context, cluster=cluster)
        if output == "json":
            info = json.dumps(d.to_dict(), indent=2, default=str)
            console.print_json(info)
        else:
            info = yaml.dump(d.to_dict(), indent=2, sort_keys=False)
            console.print(Syntax(info, "yaml", background_color="default"))

    @deployment_cli.command()
    @shared_decorator
    @click.argument(
        "name",
        type=click.STRING,
        required=True,
    )
    @click.pass_obj
    def terminate(  # type: ignore
        shared_options: SharedOptions,
        name: str,
        cluster: str | None,
    ) -> None:
        """Terminate a deployment on BentoCloud."""
        Deployment.terminate(
            name, context=shared_options.cloud_context, cluster=cluster
        )
        click.echo(f"Deployment '{name}' terminated successfully.")

    @deployment_cli.command()
    @shared_decorator
    @click.pass_obj
    def delete(  # type: ignore
        shared_options: SharedOptions,
        name: str,
        cluster: str | None,
    ) -> None:
        """Delete a deployment on BentoCloud."""
        Deployment.delete(name, context=shared_options.cloud_context, cluster=cluster)
        click.echo(f"Deployment '{name}' deleted successfully.")

    @deployment_cli.command()
    @click.option(
        "--cluster", type=click.STRING, default=None, help="Name of the cluster."
    )
    @click.option(
        "--search", type=click.STRING, default=None, help="Search for list request."
    )
    @click.option(
        "-o",
        "--output",
        help="Display the output of this command.",
        type=click.Choice(["json", "yaml", "table"]),
        default="table",
    )
    @click.pass_obj
    def list(  # type: ignore
        shared_options: SharedOptions,
        cluster: str | None,
        search: str | None,
        output: t.Literal["json", "yaml", "table"],
    ) -> None:
        """List existing deployments on BentoCloud."""
        d_list = Deployment.list(
            context=shared_options.cloud_context, cluster=cluster, search=search
        )
        res: list[dict[str, t.Any]] = [d.to_dict() for d in d_list]
        if output == "table":
            table = Table(box=None)
            table.add_column("Deployment")
            table.add_column("created_at")
            table.add_column("Bento")
            table.add_column("Status")
            for info in d_list:
                table.add_row(
                    info.name,
                    info.created_at,
                    info.get_bento(refetch=False),
                    info.get_status(refetch=False).status,
                )
            console.print(table)
        elif output == "json":
            info = json.dumps(res, indent=2, default=str)
            console.print_json(info)
        else:
            info = yaml.dump(res, indent=2, sort_keys=False)
            console.print(Syntax(info, "yaml", background_color="default"))
