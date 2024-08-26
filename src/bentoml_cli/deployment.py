from __future__ import annotations

import hashlib
import json
import logging
import os
import typing as t
from functools import partial
from http import HTTPStatus

import click
import rich
import yaml
from rich.syntax import Syntax
from rich.table import Table

from bentoml._internal.bento.build_config import BentoBuildConfig
from bentoml._internal.cloud.base import Spinner
from bentoml._internal.cloud.deployment import Deployment
from bentoml._internal.cloud.deployment import DeploymentConfigParameters
from bentoml._internal.cloud.deployment import DeploymentInfo
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStatus
from bentoml._internal.cloud.schemas.modelschemas import DeploymentStrategy
from bentoml._internal.utils import rich_console as console
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import InvalidArgument
from bentoml_cli.utils import BentoMLCommandGroup

logger = logging.getLogger("bentoml.cli.deployment")

if t.TYPE_CHECKING:
    TupleStrAny = tuple[str, ...]
else:
    TupleStrAny = tuple


def raise_deployment_config_error(err: BentoMLException, action: str) -> t.NoReturn:
    if err.error_code == HTTPStatus.UNAUTHORIZED:
        raise BentoMLException(
            f"{err}\n* BentoCloud sign up: https://cloud.bentoml.com/\n"
            "* Login with your API token: "
            "https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html"
        ) from None
    raise BentoMLException(
        f"Failed to {action} deployment due to invalid configuration: {err}"
    ) from None


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
    help="List of environment variables pass by --env key=value, --env ...",
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
@click.option(
    "--dev",
    is_flag=True,
    help="Create a development deployment and watch for changes",
    default=False,
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
    dev: bool,
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
        dev=dev,
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
        cli=True,
    )
    try:
        config_params.verify()
    except BentoMLException as e:
        raise_deployment_config_error(e, "update")
    deployment_info = Deployment.update(deployment_config_params=config_params)

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
        bento=bento,
        cluster=cluster,
        access_authorization=access_authorization,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=(
            [{"key": item.split("=")[0], "value": item.split("=")[1]} for item in env]
            if env is not None
            else None
        ),
        config_file=config_file,
        config_dict=cfg_dict,
        cli=True,
    )
    try:
        config_params.verify()
    except BentoMLException as e:
        raise_deployment_config_error(e, "apply")
    deployment_info = Deployment.apply(deployment_config_params=config_params)

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
    help="List of environment variables pass by --env key=value, --env ...",
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
    d = Deployment.get(name, cluster=cluster)
    if output == "json":
        info = json.dumps(d.to_dict(), indent=2, default=str)
        console.print_json(info)
    else:
        info = yaml.dump(d.to_dict(), indent=2, sort_keys=False)
        console.print(Syntax(info, "yaml", background_color="default"))


@deployment_command.command()
@shared_decorator
@click.argument(
    "name",
    type=click.STRING,
    required=True,
)
def terminate(  # type: ignore
    name: str, cluster: str | None
) -> None:
    """Terminate a deployment on BentoCloud."""
    Deployment.terminate(name, cluster=cluster)
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
    Deployment.delete(name, cluster=cluster)
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
def list_command(  # type: ignore
    cluster: str | None,
    search: str | None,
    output: t.Literal["json", "yaml", "table"],
) -> None:
    """List existing deployments on BentoCloud."""
    d_list = Deployment.list(cluster=cluster, search=search)
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
        console.print(table)
    elif output == "json":
        info = json.dumps(res, indent=2, default=str)
        console.print_json(info)
    else:
        info = yaml.dump(res, indent=2, sort_keys=False)
        console.print(Syntax(info, "yaml", background_color="default"))


@deployment_command.command()
@click.option("--cluster", type=click.STRING, default=None, help="Name of the cluster.")
@click.option(
    "-o",
    "--output",
    help="Display the output of this command.",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
)
def list_instance_types(  # type: ignore
    cluster: str | None,
    output: t.Literal["json", "yaml", "table"],
) -> None:
    """List existing instance types in cluster on BentoCloud."""
    d_list = Deployment.list_instance_types(cluster=cluster)
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
        console.print(table)
    elif output == "json":
        info = json.dumps(res, indent=2, default=str)
        console.print_json(info)
    else:
        info = yaml.dump(res, indent=2, sort_keys=False)
        console.print(Syntax(info, "yaml", background_color="default"))


def create_deployment(
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
    dev: bool = False,
) -> None:
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
        envs=(
            [{"name": item.split("=")[0], "value": item.split("=")[1]} for item in env]
            if env is not None
            else None
        ),
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

    with Spinner() as spinner:
        spinner.update("Creating deployment on BentoCloud")
        deployment = Deployment.create(deployment_config_params=config_params)
        spinner.log(
            f'✅ Created deployment "{deployment.name}" in cluster "{deployment.cluster}"'
        )
        spinner.log(f"💻 View Dashboard: {deployment.admin_console}")
        if wait:
            spinner.update(
                "[bold blue]Waiting for deployment to be ready, you can use --no-wait to skip this process[/]",
            )
            retcode = deployment.wait_until_ready(
                timeout=timeout,
                spinner=spinner,
                on_init=partial(_init_deployment_files, bento_dir=t.cast(str, bento))
                if dev
                else None,
            )
            if retcode != 0:
                raise SystemExit(retcode)
        elif dev:
            raise InvalidArgument(
                "Cannot use `--no-wait` flag when deploying using development mode"
            )
    if dev:
        _watch_dev_deployment(deployment, t.cast(str, bento))


REQUIREMENTS_TXT = "requirements.txt"


def _build_requirements_txt(bento_dir: str, config: BentoBuildConfig) -> bytes:
    from bentoml._internal.configuration import BENTOML_VERSION
    from bentoml._internal.configuration import clean_bentoml_version

    filename = config.python.requirements_txt
    content = b""
    if filename and os.path.exists(fullpath := os.path.join(bento_dir, filename)):
        with open(fullpath, "rb") as f:
            content = f.read()
    for package in config.python.packages or []:
        content += f"{package}\n".encode()
    bentoml_version = clean_bentoml_version(BENTOML_VERSION)
    content += f"bentoml=={bentoml_version}\n".encode()
    return content


def _get_bento_build_config(bento_dir: str) -> BentoBuildConfig:
    bentofile_path = os.path.join(bento_dir, "bentofile.yaml")
    if not os.path.exists(bentofile_path):
        return BentoBuildConfig(service="").with_defaults()
    else:
        # respect bentofile.yaml include and exclude
        with open(bentofile_path, "r") as f:
            return BentoBuildConfig.from_yaml(f).with_defaults()


def _init_deployment_files(deployment: DeploymentInfo, bento_dir: str) -> None:
    from bentoml._internal.bento.build_config import BentoPathSpec

    build_config = _get_bento_build_config(bento_dir)
    bento_spec = BentoPathSpec(build_config.include, build_config.exclude)
    upload_files: list[tuple[str, bytes]] = []
    requirements_content = _build_requirements_txt(bento_dir, build_config)
    ignore_patterns = bento_spec.from_path(bento_dir)
    for root, _, files in os.walk(bento_dir):
        for fn in files:
            full_path = os.path.join(root, fn)
            rel_path = os.path.relpath(full_path, bento_dir)
            if (
                not bento_spec.includes(full_path, recurse_exclude_spec=ignore_patterns)
                and rel_path != "bentofile.yaml"
            ):
                continue
            if rel_path == REQUIREMENTS_TXT:
                continue
            rich.print(f" [green]Uploading[/] {rel_path}")
            upload_files.append((rel_path, open(full_path, "rb").read()))
    rich.print(f" [green]Uploading[/] {REQUIREMENTS_TXT}")
    upload_files.append((REQUIREMENTS_TXT, requirements_content))
    deployment.upload_files(upload_files)


def _watch_dev_deployment(deployment: DeploymentInfo, bento_dir: str) -> None:
    import watchfiles

    from bentoml._internal.bento.build_config import BentoPathSpec

    build_config = _get_bento_build_config(bento_dir)
    bento_spec = BentoPathSpec(build_config.include, build_config.exclude)
    ignore_patterns = bento_spec.from_path(bento_dir)
    requirements_content = _build_requirements_txt(bento_dir, build_config)
    requirements_hash = hashlib.md5(requirements_content).hexdigest()
    _init_deployment_files(deployment, bento_dir)

    default_filter = watchfiles.filters.DefaultFilter()

    def watch_filter(change: watchfiles.Change, path: str) -> bool:
        if not default_filter(change, path):
            return False
        if path == "bentofile.yaml":
            return True
        return bento_spec.includes(path, recurse_exclude_spec=ignore_patterns)

    with Spinner() as spinner:
        spinner.update(
            f"Watching file changes in {bento_dir} for deployment {deployment.name}"
        )
        spinner.log(f"💻 View Dashboard: {deployment.admin_console}")

        for changes in watchfiles.watch(bento_dir, watch_filter=watch_filter):
            build_config = _get_bento_build_config(bento_dir)
            upload_files: list[tuple[str, bytes]] = []
            delete_files: list[str] = []

            for change, path in changes:
                rel_path = os.path.relpath(path, bento_dir)
                if rel_path == REQUIREMENTS_TXT:
                    continue
                if change == watchfiles.Change.deleted:
                    rich.print(f" [red]Deleting[/] {path}")
                    delete_files.append(rel_path)
                else:
                    rich.print(f" [green]Uploading[/] {path}")
                    upload_files.append((rel_path, open(path, "rb").read()))

            requirements_content = _build_requirements_txt(bento_dir, build_config)
            if (
                new_hash := hashlib.md5(requirements_content).hexdigest()
                != requirements_hash
            ):
                requirements_hash = new_hash
                rich.print(f" [green]Uploading[/] {REQUIREMENTS_TXT}")
                upload_files.append((REQUIREMENTS_TXT, requirements_content))
            if upload_files:
                deployment.upload_files(upload_files)
            if delete_files:
                deployment.delete_files(delete_files)
            if (status := deployment.get_status().status) in [
                DeploymentStatus.Failed.value,
                DeploymentStatus.ImageBuildFailed.value,
                DeploymentStatus.Terminated.value,
                DeploymentStatus.Terminating.value,
                DeploymentStatus.Unhealthy.value,
            ]:
                rich.print(
                    f'🚨 [bold red]Deployment "{deployment.name}" is not ready. Current status: "{status}"[/]'
                )
                return


if __name__ == "__main__":
    import sys
    # Testing code

    deployment = Deployment.get(sys.argv[1])
    _watch_dev_deployment(deployment, ".")
