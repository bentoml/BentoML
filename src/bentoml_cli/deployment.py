from __future__ import annotations

import json
import typing as t

import click
import yaml
from rich.live import Live
from rich.syntax import Syntax

from bentoml._internal.cloud.base import Spinner
from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStrategy
from bentoml.exceptions import BentoMLException

if t.TYPE_CHECKING:
    TupleStrAny = tuple[str, ...]

    from .utils import SharedOptions
else:
    TupleStrAny = tuple


def add_deployment_command(cli: click.Group) -> None:
    from rich.table import Table

    from bentoml._internal.utils import rich_console as console
    from bentoml_cli.utils import BentoMLCommandGroup

    @cli.command()
    @click.argument(
        "bento",
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
        "--access-authorization",
        type=click.BOOL,
        help="Enable access authorization",
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
        "--config-dict",
        type=click.STRING,
        help="Configuration json string",
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
        default=1800,
        help="Timeout for deployment to be ready in seconds",
    )
    @click.pass_obj
    def deploy(
        shared_options: SharedOptions,
        bento: str | None,
        name: str | None,
        cluster: str | None,
        access_authorization: bool | None,
        scaling_min: int | None,
        scaling_max: int | None,
        instance_type: str | None,
        strategy: str | None,
        env: tuple[str] | None,
        config_file: str | t.TextIO | None,
        config_dict: str | None,
        wait: bool,
        timeout: int,
    ) -> None:
        """Create a deployment on BentoCloud.

        \b
        Create a deployment using parameters, or using config yaml file.
        """
        create_deployment(
            context=shared_options.cloud_context,
            bento=bento,
            name=name,
            cluster=cluster,
            access_authorization=access_authorization,
            scaling_min=scaling_min,
            scaling_max=scaling_max,
            instance_type=instance_type,
            strategy=strategy,
            env=env,
            config_file=config_file,
            config_dict=config_dict,
            wait=wait,
            timeout=timeout,
        )

    output_option = click.option(
        "-o",
        "--output",
        type=click.Choice(["yaml", "json"]),
        default="yaml",
        help="Display the output of this command.",
    )

    def shared_decorator(
        f: t.Callable[..., t.Any] | None = None,
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
    @click.argument(
        "name",
        type=click.STRING,
        required=False,
    )
    @click.option(
        "--bento",
        type=click.STRING,
        help="Bento name or path to Bento project directory",
    )
    @click.option(
        "--access-authorization",
        type=click.BOOL,
        help="Enable access authorization",
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
    @click.option(
        "--config-dict",
        type=click.STRING,
        help="Configuration json string",
        default=None,
    )
    @click.pass_obj
    def update(  # type: ignore
        shared_options: SharedOptions,
        name: str | None,
        cluster: str | None,
        bento: str | None,
        access_authorization: bool | None,
        scaling_min: int | None,
        scaling_max: int | None,
        instance_type: str | None,
        strategy: str | None,
        env: tuple[str] | None,
        config_file: t.TextIO | None,
        config_dict: str | None,
    ) -> None:
        """Update a deployment on BentoCloud.

        \b
        A deployment can be updated using parameters, or using config yaml file.
        You can also update bento by providing a project path or existing bento.
        """
        cfg_dict = None
        if config_dict is not None and config_dict != "":
            cfg_dict = json.loads(config_dict)
        config_params = DeploymentConfigParameters(
            name=name,
            context=shared_options.cloud_context,
            bento=bento,
            cluster=cluster,
            access_authorization=access_authorization,
            scaling_max=scaling_max,
            scaling_min=scaling_min,
            instance_type=instance_type,
            strategy=strategy,
            envs=(
                [
                    {"name": item.split("=")[0], "value": item.split("=")[1]}
                    for item in env
                ]
                if env is not None
                else None
            ),
            config_file=config_file,
            config_dict=cfg_dict,
        )
        try:
            config_params.verify()
        except BentoMLException as e:
            raise BentoMLException(
                f"Failed to create deployment due to invalid configuration: {e}"
            )
        deployment_info = Deployment.update(
            deployment_config_params=config_params,
            context=shared_options.cloud_context,
        )

        click.echo(f"Deployment '{deployment_info.name}' updated successfully.")

    @deployment_cli.command()
    @click.argument(
        "bento",
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
        "--access-authorization",
        type=click.BOOL,
        help="Enable access authorization",
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
        "-f",
        "--config-file",
        help="Configuration file path, mututally exclusive with other config options",
        default=None,
    )
    @click.option(
        "--config-dict",
        type=click.STRING,
        help="Configuration json string",
        default=None,
    )
    @click.pass_obj
    def apply(  # type: ignore
        shared_options: SharedOptions,
        bento: str | None,
        name: str | None,
        cluster: str | None,
        access_authorization: bool | None,
        scaling_min: int | None,
        scaling_max: int | None,
        instance_type: str | None,
        strategy: str | None,
        env: tuple[str] | None,
        config_file: str | t.TextIO | None,
        config_dict: str | None,
    ) -> None:
        """Apply a deployment on BentoCloud.

        \b
        A deployment can be applied using config yaml file.
        """
        cfg_dict = None
        if config_dict is not None and config_dict != "":
            cfg_dict = json.loads(config_dict)
        config_params = DeploymentConfigParameters(
            name=name,
            context=shared_options.cloud_context,
            bento=bento,
            cluster=cluster,
            access_authorization=access_authorization,
            scaling_max=scaling_max,
            scaling_min=scaling_min,
            instance_type=instance_type,
            strategy=strategy,
            envs=(
                [
                    {"key": item.split("=")[0], "value": item.split("=")[1]}
                    for item in env
                ]
                if env is not None
                else None
            ),
            config_file=config_file,
            config_dict=cfg_dict,
        )
        try:
            config_params.verify()
        except BentoMLException as e:
            raise BentoMLException(
                f"Failed to create deployment due to invalid configuration: {e}"
            )
        deployment_info = Deployment.apply(
            deployment_config_params=config_params,
            context=shared_options.cloud_context,
        )

        click.echo(f"Deployment '{deployment_info.name}' applied successfully.")

    @deployment_cli.command()
    @click.argument(
        "bento",
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
        "--access-authorization",
        type=click.BOOL,
        help="Enable access authorization",
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
        "--config-dict",
        type=click.STRING,
        help="Configuration json string",
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
        default=1800,
        help="Timeout for deployment to be ready in seconds",
    )
    @click.pass_obj
    def create(
        shared_options: SharedOptions,
        bento: str | None,
        name: str | None,
        cluster: str | None,
        access_authorization: bool | None,
        scaling_min: int | None,
        scaling_max: int | None,
        instance_type: str | None,
        strategy: str | None,
        env: tuple[str] | None,
        config_file: str | t.TextIO | None,
        config_dict: str | None,
        wait: bool,
        timeout: int,
    ) -> None:
        """Create a deployment on BentoCloud.

        \b
        Create a deployment using parameters, or using config yaml file.
        """
        create_deployment(
            context=shared_options.cloud_context,
            bento=bento,
            name=name,
            cluster=cluster,
            access_authorization=access_authorization,
            scaling_min=scaling_min,
            scaling_max=scaling_max,
            instance_type=instance_type,
            strategy=strategy,
            env=env,
            config_file=config_file,
            config_dict=config_dict,
            wait=wait,
            timeout=timeout,
        )

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
    @click.argument(
        "name",
        type=click.STRING,
        required=True,
    )
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
            table.add_column("Region")
            for info in d_list:
                table.add_row(
                    info.name,
                    info.created_at,
                    info.get_bento(refetch=False),
                    info.get_status(refetch=False).status,
                    info.cluster,
                )
            console.print(table)
        elif output == "json":
            info = json.dumps(res, indent=2, default=str)
            console.print_json(info)
        else:
            info = yaml.dump(res, indent=2, sort_keys=False)
            console.print(Syntax(info, "yaml", background_color="default"))

    @deployment_cli.command()
    @click.option(
        "--cluster", type=click.STRING, default=None, help="Name of the cluster."
    )
    @click.option(
        "-o",
        "--output",
        help="Display the output of this command.",
        type=click.Choice(["json", "yaml", "table"]),
        default="table",
    )
    @click.pass_obj
    def list_instance_types(  # type: ignore
        shared_options: SharedOptions,
        cluster: str | None,
        output: t.Literal["json", "yaml", "table"],
    ) -> None:
        """List existing instance types in cluster on BentoCloud."""
        d_list = Deployment.list_instance_types(
            context=shared_options.cloud_context, cluster=cluster
        )
        res: list[dict[str, t.Any]] = [d.to_dict() for d in d_list]
        if output == "table":
            table = Table(box=None)
            table.add_column("Name")
            table.add_column("Price")
            table.add_column("CPU")
            table.add_column("Memory")
            table.add_column("GPU")
            table.add_column("GPU Type")
            for info in d_list:
                table.add_row(
                    info.name,
                    info.price,
                    info.cpu,
                    info.memory,
                    info.gpu,
                    info.gpu_type,
                )
            console.print(table)
        elif output == "json":
            info = json.dumps(res, indent=2, default=str)
            console.print_json(info)
        else:
            info = yaml.dump(res, indent=2, sort_keys=False)
            console.print(Syntax(info, "yaml", background_color="default"))


def create_deployment(
    context: str | None,
    bento: str | None,
    name: str | None,
    cluster: str | None,
    access_authorization: bool | None,
    scaling_min: int | None,
    scaling_max: int | None,
    instance_type: str | None,
    strategy: str | None,
    env: tuple[str] | None,
    config_file: str | t.TextIO | None,
    config_dict: str | None,
    wait: bool,
    timeout: int,
) -> None:
    cfg_dict = None
    if config_dict is not None and config_dict != "":
        cfg_dict = json.loads(config_dict)
    config_params = DeploymentConfigParameters(
        name=name,
        context=context,
        bento=bento,
        cluster=cluster,
        access_authorization=access_authorization,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=(
            [{"name": item.split("=")[0], "value": item.split("=")[1]} for item in env]
            if env is not None
            else None
        ),
        config_file=config_file,
        config_dict=cfg_dict,
    )
    try:
        config_params.verify()
    except BentoMLException as e:
        raise BentoMLException(
            f"Failed to create deployment due to invalid configuration: {e}"
        )
    spinner = Spinner()
    with Live(spinner.progress_group):
        task_id = spinner.spinner_progress.add_task(
            "deploy", action="Deploying to BentoCloud"
        )
        deployment = Deployment.create(
            deployment_config_params=config_params,
            context=context,
        )
        spinner.log_progress.add_task(
            f"[bold green]Successfully created deployment '{deployment.name}' in cluster '{deployment.cluster}'[/bold green]"
        )
        spinner.log_progress.add_task(
            f"[bold blue]To check the deployment details, go to:\n{deployment.admin_console}[/bold blue]"
        )
        if wait:
            spinner.spinner_progress.update(
                task_id,
                action="[bold blue]Waiting for deployment to be ready, you can use --no-wait to skip this process[/bold blue]",
            )
            deployment.wait_until_ready(
                timeout=timeout, spinner_task_id=task_id, spinner=spinner
            )
        spinner.spinner_progress.stop_task(task_id)
