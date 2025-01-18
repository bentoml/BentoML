from __future__ import annotations

import json
import logging
import os
import typing as t
from http import HTTPStatus

import click
import rich
import rich.style
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table
from simple_di import Provide
from simple_di import inject

import bentoml.deployment
from bentoml._internal.cloud.base import Spinner
from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStatus
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStrategy
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import CLIException
from bentoml_cli.utils import BentoMLCommandGroup

logger = logging.getLogger("bentoml.cli.deployment")

if t.TYPE_CHECKING:
    from bentoml._internal.cloud import BentoCloudClient

    TupleStrAny = tuple[str, ...]
else:
    TupleStrAny = tuple


def raise_deployment_config_error(err: BentoMLException, action: str) -> t.NoReturn:
    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* BentoCloud API token is required for authorization. Run `bentoml cloud login` command to login"
        ) from None
    raise BentoMLException(
        f"Failed to {action} deployment due to invalid configuration: {err}"
    ) from None


def convert_env_to_dict(env: tuple[str] | None) -> list[dict[str, str]] | None:
    if env is None:
        return None
    collected_envs: list[dict[str, str]] = []
    if env:
        for item in env:
            if "=" in item:
                name, value = item.split("=", 1)
            else:
                name = item
                if name not in os.environ:
                    raise CLIException(f"Environment variable {name} not found")
                value = os.environ[name]
            collected_envs.append({"name": name, "value": value})
    return collected_envs


@click.command(name="deploy")
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
    help="List of environment variables pass by --env key[=value] --env ...",
    multiple=True,
)
@click.option(
    "--secret",
    type=click.STRING,
    help="List of secret names pass by --secret name1, --secret name2, ...",
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
    default=3600,
    help="Timeout for deployment to be ready in seconds",
)
def deploy_command(
    bento: str | None,
    name: str | None,
    cluster: str | None,
    access_authorization: bool | None,
    scaling_min: int | None,
    scaling_max: int | None,
    instance_type: str | None,
    strategy: str | None,
    env: tuple[str] | None,
    secret: tuple[str] | None,
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
        bento=bento,
        name=name,
        cluster=cluster,
        access_authorization=access_authorization,
        scaling_min=scaling_min,
        scaling_max=scaling_max,
        instance_type=instance_type,
        strategy=strategy,
        env=env,
        secret=secret,
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


@click.command(name="code")
@click.argument(
    "bento_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True),
    default=".",
)
@click.option(
    "--attach", help="Attach to the given deployment instead of creating a new one."
)
@click.option(
    "--env",
    type=click.STRING,
    help="List of environment variables pass by --env key[=value] --env ...",
    multiple=True,
)
@click.option(
    "--secret",
    type=click.STRING,
    help="List of secret names pass by --secret name1, --secret name2, ...",
    multiple=True,
)
@shared_decorator
@inject
def develop_command(
    bento_dir: str,
    cluster: str | None,
    attach: str | None,
    env: tuple[str] | None,
    secret: tuple[str] | None,
    _rest_client: RestApiClient = Provide[BentoMLContainer.rest_api_client],
):
    """Create or attach to a codespace.

    Create a new codespace:

        $ bentoml code

    Attach to an existing codespace:

        $ bentoml code --attach <codespace-name>
    """
    import questionary

    if attach and (env or secret):
        raise CLIException("Cannot specify both --attach and --env or --secret")

    console = Console(highlight=False)
    if attach:
        deployment = bentoml.deployment.get(attach)
    else:
        with console.status("Fetching codespaces..."):
            current_user = _rest_client.v1.get_current_user()
            if current_user is None:
                raise CLIException("current user is not found")
            deployments = [
                d
                for d in bentoml.deployment.list(cluster=cluster, dev=True)
                if d.is_dev
                and d.created_by == current_user.name
                and d.get_status(False).status
                in [
                    DeploymentStatus.Deploying.value,
                    DeploymentStatus.Running.value,
                    DeploymentStatus.ScaledToZero.value,
                    DeploymentStatus.Failed.value,
                ]
            ]

        chosen = questionary.select(
            message="Select a codespace to attach to or create a new one",
            choices=[{"name": d.name, "value": d} for d in deployments]
            + [{"name": "Create a new codespace", "value": "new"}],
        ).ask()

        if chosen == "new":
            deployment = create_deployment(
                bento=bento_dir,
                cluster=cluster,
                dev=True,
                wait=False,
                env=env,
                secret=secret,
            )
        elif chosen is None:
            return
        else:
            if env or secret:
                rich.print(
                    "[yellow]Warning:[/] --env and --secret are ignored when attaching to an existing codespace"
                )
            deployment = t.cast(Deployment, chosen)
    deployment.watch(bento_dir)


@click.group(name="deployment", cls=BentoMLCommandGroup)
def deployment_command():
    """Deployment Subcommands Groups"""


@deployment_command.command()
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
    help="List of environment variables pass by --env key[=value] --env ...",
    multiple=True,
)
@click.option(
    "--secret",
    type=click.STRING,
    help="List of secret names pass by --secret name1, --secret name2, ...",
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
@inject
def update(  # type: ignore
    name: str | None,
    cluster: str | None,
    bento: str | None,
    access_authorization: bool | None,
    scaling_min: int | None,
    scaling_max: int | None,
    instance_type: str | None,
    strategy: str | None,
    env: tuple[str] | None,
    secret: tuple[str] | None,
    config_file: t.TextIO | None,
    config_dict: str | None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
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
        bento=bento,
        cluster=cluster,
        access_authorization=access_authorization,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=convert_env_to_dict(env),
        secrets=list(secret) if secret is not None else None,
        config_file=config_file,
        config_dict=cfg_dict,
        cli=True,
    )
    try:
        config_params.verify(create=False)
    except BentoMLException as e:
        raise_deployment_config_error(e, "update")
    deployment_info = _cloud_client.deployment.update(
        deployment_config_params=config_params
    )

    rich.print(f"Deployment [green]'{deployment_info.name}'[/] updated successfully.")


@deployment_command.command()
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
    help="List of environment variables pass by --env key[=value] --env ...",
    multiple=True,
)
@click.option(
    "--secret",
    type=click.STRING,
    help="List of secret names pass by --secret name1, --secret name2, ...",
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
@inject
def apply(  # type: ignore
    bento: str | None,
    name: str | None,
    cluster: str | None,
    access_authorization: bool | None,
    scaling_min: int | None,
    scaling_max: int | None,
    instance_type: str | None,
    strategy: str | None,
    env: tuple[str] | None,
    secret: tuple[str] | None,
    config_file: str | t.TextIO | None,
    config_dict: str | None,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
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
        bento=bento,
        cluster=cluster,
        access_authorization=access_authorization,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=convert_env_to_dict(env),
        secrets=list(secret) if secret is not None else None,
        config_file=config_file,
        config_dict=cfg_dict,
        cli=True,
    )
    try:
        config_params.verify(create=False)
    except BentoMLException as e:
        raise_deployment_config_error(e, "apply")
    deployment_info = _cloud_client.deployment.apply(
        deployment_config_params=config_params
    )

    rich.print(f"Deployment [green]'{deployment_info.name}'[/] applied successfully.")


@deployment_command.command()
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
    help="List of environment variables pass by --env key[=value] --env ...",
    multiple=True,
)
@click.option(
    "--secret",
    type=click.STRING,
    help="List of secret names pass by --secret name1, --secret name2, ...",
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
    default=3600,
    help="Timeout for deployment to be ready in seconds",
)
def create(
    bento: str | None,
    name: str | None,
    cluster: str | None,
    access_authorization: bool | None,
    scaling_min: int | None,
    scaling_max: int | None,
    instance_type: str | None,
    strategy: str | None,
    env: tuple[str] | None,
    secret: tuple[str] | None,
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
        bento=bento,
        name=name,
        cluster=cluster,
        access_authorization=access_authorization,
        scaling_min=scaling_min,
        scaling_max=scaling_max,
        instance_type=instance_type,
        strategy=strategy,
        env=env,
        secret=secret,
        config_file=config_file,
        config_dict=config_dict,
        wait=wait,
        timeout=timeout,
    )


@deployment_command.command()
@shared_decorator
@click.argument(
    "name",
    type=click.STRING,
    required=True,
)
@output_option
def get(  # type: ignore
    name: str,
    cluster: str | None,
    output: t.Literal["json", "default"],
) -> None:
    """Get a deployment on BentoCloud."""
    d = bentoml.deployment.get(name, cluster=cluster)
    if output == "json":
        info = json.dumps(d.to_dict(), indent=2, default=str)
        rich.print_json(info)
    else:
        info = yaml.dump(d.to_dict(), indent=2, sort_keys=False)
        rich.print(Syntax(info, "yaml", background_color="default"))


@deployment_command.command()
@shared_decorator
@click.argument(
    "name",
    type=click.STRING,
    required=True,
)
@click.option("--wait", is_flag=True, help="Wait for the deployment to be terminated")
def terminate(  # type: ignore
    name: str, cluster: str | None, wait: bool
) -> None:
    """Terminate a deployment on BentoCloud."""
    bentoml.deployment.terminate(name, cluster=cluster, wait=wait)
    rich.print(f"Deployment [green]'{name}'[/] terminated successfully.")


@deployment_command.command()
@click.argument(
    "name",
    type=click.STRING,
    required=True,
)
@shared_decorator
def delete(  # type: ignore
    name: str, cluster: str | None
) -> None:
    """Delete a deployment on BentoCloud."""
    bentoml.deployment.delete(name, cluster=cluster)
    rich.print(f"Deployment [green]'{name}'[/] deleted successfully.")


@deployment_command.command(name="list")
@click.option("--cluster", type=click.STRING, default=None, help="Name of the cluster.")
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
@click.option(
    "--label",
    "labels",
    type=click.STRING,
    multiple=True,
    default=None,
    help="Filter deployments by label(s).",
    metavar="KEY=VALUE",
)
def list_command(  # type: ignore
    cluster: str | None,
    search: str | None,
    labels: tuple[str, ...] | None,
    output: t.Literal["json", "yaml", "table"],
) -> None:
    """List existing deployments on BentoCloud."""
    if labels is not None:
        # For labels like ["env=prod", "team=infra"]
        # This will output: "label:env=prod label:team=infra"
        labels_query = " ".join(f"label:{label}" for label in labels)

    try:
        d_list = bentoml.deployment.list(cluster=cluster, search=search, q=labels_query)
    except BentoMLException as e:
        raise_deployment_config_error(e, "list")
    res: list[dict[str, t.Any]] = [d.to_dict() for d in d_list]
    if output == "table":
        table = Table(box=None, expand=True)
        table.add_column("Deployment", overflow="fold")
        table.add_column("created_at", overflow="fold")
        table.add_column("Bento", overflow="fold")
        table.add_column("Status", overflow="fold")
        table.add_column("Region", overflow="fold")
        for info in d_list:
            table.add_row(
                info.name,
                info.created_at,
                info.get_bento(refetch=False),
                info.get_status(refetch=False).status,
                info.cluster,
            )
        rich.print(table)
    elif output == "json":
        info = json.dumps(res, indent=2, default=str)
        rich.print_json(info)
    else:
        info = yaml.dump(res, indent=2, sort_keys=False)
        rich.print(Syntax(info, "yaml", background_color="default"))


@deployment_command.command()
@click.option("--cluster", type=click.STRING, default=None, help="Name of the cluster.")
@click.option(
    "-o",
    "--output",
    help="Display the output of this command.",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
)
@inject
def list_instance_types(  # type: ignore
    cluster: str | None,
    output: t.Literal["json", "yaml", "table"],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> None:
    """List existing instance types in cluster on BentoCloud."""
    try:
        d_list = _cloud_client.deployment.list_instance_types(cluster=cluster)
    except BentoMLException as e:
        raise_deployment_config_error(e, "list_instance_types")
    res: list[dict[str, t.Any]] = [d.to_dict() for d in d_list]
    if output == "table":
        table = Table(box=None, expand=True)
        table.add_column("Name", overflow="fold")
        table.add_column("Price", overflow="fold")
        table.add_column("CPU", overflow="fold")
        table.add_column("Memory", overflow="fold")
        table.add_column("GPU", overflow="fold")
        table.add_column("GPU Type", overflow="fold")
        for info in d_list:
            table.add_row(
                info.name,
                info.price,
                info.cpu,
                info.memory,
                info.gpu,
                info.gpu_type,
            )
        rich.print(table)
    elif output == "json":
        info = json.dumps(res, indent=2, default=str)
        rich.print_json(info)
    else:
        info = yaml.dump(res, indent=2, sort_keys=False)
        rich.print(Syntax(info, "yaml", background_color="default"))


@inject
def create_deployment(
    bento: str | None = None,
    name: str | None = None,
    cluster: str | None = None,
    access_authorization: bool | None = None,
    scaling_min: int | None = None,
    scaling_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    env: tuple[str] | None = None,
    secret: tuple[str] | None = None,
    config_file: str | t.TextIO | None = None,
    config_dict: str | None = None,
    wait: bool = True,
    timeout: int = 3600,
    dev: bool = False,
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
) -> Deployment:
    cfg_dict = None
    if config_dict is not None and config_dict != "":
        cfg_dict = json.loads(config_dict)

    config_params = DeploymentConfigParameters(
        name=name,
        bento=bento,
        cluster=cluster,
        access_authorization=access_authorization,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=convert_env_to_dict(env),
        secrets=list(secret) if secret is not None else None,
        config_file=config_file,
        config_dict=cfg_dict,
        cli=True,
        dev=dev,
    )

    try:
        config_params.verify()
    except BentoMLException as e:
        raise_deployment_config_error(e, "create")

    console = Console(highlight=False)
    with Spinner(console=console) as spinner:
        spinner.update("Creating deployment on BentoCloud")
        deployment = _cloud_client.deployment.create(
            deployment_config_params=config_params
        )
        spinner.log(
            f':white_check_mark: Created deployment "{deployment.name}" in cluster "{deployment.cluster}"'
        )
        spinner.log(f":laptop_computer: View Dashboard: {deployment.admin_console}")
        if wait:
            spinner.update(
                "[bold blue]Waiting for deployment to be ready, you can use --no-wait to skip this process[/]",
            )
            retcode = deployment.wait_until_ready(timeout=timeout, spinner=spinner)
            if retcode != 0:
                raise SystemExit(retcode)
        return deployment
