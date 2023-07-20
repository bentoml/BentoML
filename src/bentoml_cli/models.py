from __future__ import annotations

import json
import typing as t

import click
import yaml
from rich.syntax import Syntax
from rich.table import Table

from bentoml_cli.utils import BentoMLCommandGroup
from bentoml_cli.utils import is_valid_bento_name
from bentoml_cli.utils import is_valid_bento_tag

if t.TYPE_CHECKING:
    from click import Context
    from click import Group
    from click import Parameter

    from .utils import SharedOptions


def parse_delete_targets_argument_callback(
    ctx: Context, params: Parameter, value: t.Any  # pylint: disable=unused-argument
) -> t.Any:
    if value is None:
        return value
    value = " ".join(value)
    if "," in value:
        delete_targets = value.split(",")
    else:
        delete_targets = value.split()
    delete_targets = list(map(str.strip, delete_targets))
    for delete_target in delete_targets:
        if not (
            is_valid_bento_tag(delete_target) or is_valid_bento_name(delete_target)
        ):
            raise click.BadParameter(
                'Bad formatting. Please present a valid bento bundle name or "name:version" tag. For list of bento bundles, separate delete targets by ",", for example: "my_service:v1,my_service:v2,classifier"'
            )
    return delete_targets


def add_model_management_commands(cli: Group) -> None:
    from bentoml import Tag
    from bentoml._internal.bento.bento import DEFAULT_BENTO_BUILD_FILE
    from bentoml._internal.bento.build_config import BentoBuildConfig
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.utils import calc_dir_size
    from bentoml._internal.utils import human_readable_size
    from bentoml._internal.utils import resolve_user_filepath
    from bentoml._internal.utils import rich_console as console
    from bentoml.exceptions import InvalidArgument
    from bentoml.models import import_model

    model_store = BentoMLContainer.model_store.get()
    cloud_client = BentoMLContainer.bentocloud_client.get()
    bento_store = BentoMLContainer.bento_store.get()

    @cli.group(name="models", cls=BentoMLCommandGroup)
    def model_cli():
        """Model Subcommands Groups"""

    @model_cli.command()
    @click.argument("model_tag", type=click.STRING)
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "yaml", "path"]),
        default="yaml",
    )
    def get(model_tag: str, output: str) -> None:  # type: ignore (not accessed)
        """Print Model details by providing the model_tag

        \b
        bentoml get iris_clf:qojf5xauugwqtgxi
        bentoml get iris_clf:qojf5xauugwqtgxi --output=json
        """
        model = model_store.get(model_tag)

        if output == "path":
            console.print(model.path)
        elif output == "json":
            info = json.dumps(model.info.to_dict(), indent=2, default=str)
            console.print_json(info)
        else:
            console.print(
                Syntax(str(model.info.dump()), "yaml", background_color="default")
            )

    @model_cli.command(name="list")
    @click.argument("model_name", type=click.STRING, required=False)
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "yaml", "table"]),
        default="table",
    )
    def list_models(model_name: str, output: str) -> None:  # type: ignore (not accessed)
        """List Models in local store

        \b
        # show all models saved
        $ bentoml models list

        \b
        # show all verions of bento with the name FraudDetector
        $ bentoml models list FraudDetector
        """

        models = model_store.list(model_name)
        res = [
            {
                "tag": str(model.tag),
                "module": model.info.module,
                "size": human_readable_size(calc_dir_size(model.path)),
                "creation_time": model.info.creation_time.astimezone().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
            for model in sorted(
                models, key=lambda x: x.info.creation_time, reverse=True
            )
        ]
        if output == "json":
            info = json.dumps(res, indent=2)
            console.print_json(info)
        elif output == "yaml":
            info = yaml.safe_dump(res, indent=2)
            console.print(Syntax(info, "yaml"), background_color="default")
        else:
            table = Table(box=None)
            table.add_column("Tag")
            table.add_column("Module")
            table.add_column("Size")
            table.add_column("Creation Time")
            for model in res:
                table.add_row(
                    model["tag"],
                    model["module"],
                    model["size"],
                    model["creation_time"],
                )
            console.print(table)

    @model_cli.command()
    @click.argument(
        "delete_targets",
        nargs=-1,
        callback=parse_delete_targets_argument_callback,
        required=True,
    )
    @click.option(
        "-y",
        "--yes",
        "--assume-yes",
        is_flag=True,
        help="Skip confirmation when deleting a specific model",
    )
    def delete(delete_targets: str, yes: bool) -> None:  # type: ignore (not accessed)
        """Delete Model in local model store.

        \b
        Examples:
            * Delete single model by "name:version", e.g: `bentoml models delete iris_clf:v1`
            * Bulk delete all models with a specific name, e.g.: `bentoml models delete iris_clf`
            * Bulk delete multiple models by name and version, separated by ",", e.g.: `bentoml models delete iris_clf:v1,iris_clf:v2`
            * Bulk delete multiple models by name and version, separated by " ", e.g.: `bentoml models delete iris_clf:v1 iris_clf:v2`
            * Bulk delete without confirmation, e.g.: `bentoml models delete IrisClassifier --yes`
        """  # noqa
        from bentoml.exceptions import BentoMLException

        def check_model_is_used(tag: Tag) -> None:
            in_use: list[Tag] = []
            for bento in bento_store.list():
                if bento._model_store is not None:
                    continue
                if any(model.tag == tag for model in bento.info.models):
                    in_use.append(bento.tag)
            if in_use:
                raise BentoMLException(
                    f"Model {tag} is being used by the following bentos and can't be deleted:\n  "
                    + "\n  ".join(map(str, in_use))
                )

        def delete_target(target: str) -> None:
            tag = Tag.from_str(target)

            if tag.version is None:
                to_delete_models = model_store.list(target)
            else:
                to_delete_models = [model_store.get(tag)]

            for model in to_delete_models:
                if yes:
                    delete_confirmed = True
                else:
                    delete_confirmed = click.confirm(f"delete model {model.tag}?")

                if delete_confirmed:
                    check_model_is_used(model.tag)
                    model_store.delete(model.tag)
                    click.echo(f"{model} deleted.")

        for target in delete_targets:
            delete_target(target)

    @model_cli.command()
    @click.argument("model_tag", type=click.STRING)
    @click.argument("out_path", type=click.STRING, default="", required=False)
    def export(model_tag: str, out_path: str) -> None:  # type: ignore (not accessed)
        """Export a Model to an external archive file

        arguments:

        \b
        MODEL_TAG: model identifier
        OUT_PATH: output path of exported model.
          If this argument is not provided, model is exported to name-version.bentomodel in the current directory.
          Besides native .bentomodel format, we also support formats like tar('tar'), tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip.

        examples:

        \b
        bentoml models export FraudDetector:latest
        bentoml models export FraudDetector:latest ./my_model.bentomodel
        bentoml models export FraudDetector:20210709_DE14C9 ./my_model.bentomodel
        bentoml models export FraudDetector:20210709_DE14C9 s3://mybucket/models/my_model.bentomodel
        """
        bentomodel = model_store.get(model_tag)
        out_path = bentomodel.export(out_path)
        click.echo(f"{bentomodel} exported to {out_path}.")

    @model_cli.command(name="import")
    @click.argument("model_path", type=click.STRING)
    def import_from(model_path: str) -> None:  # type: ignore (not accessed)
        """Import a previously exported Model archive file

        bentoml models import ./my_model.bentomodel
        bentoml models import s3://mybucket/models/my_model.bentomodel
        """
        bentomodel = import_model(model_path)
        click.echo(f"{bentomodel} imported.")

    @model_cli.command()
    @click.argument("model_tag", type=click.STRING, required=False)
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Force pull from yatai to local and overwrite even if it already exists in local",
    )
    @click.option(
        "-F",
        "--bentofile",
        type=click.STRING,
        default=DEFAULT_BENTO_BUILD_FILE,
        help="Path to bentofile. Default to 'bentofile.yaml'",
    )
    @click.pass_context
    def pull(ctx: click.Context, model_tag: str | None, force: bool, bentofile: str):  # type: ignore (not accessed)
        """Pull Model from a yatai server. If model_tag is not provided,
        it will pull models defined in bentofile.yaml.
        """
        from click.core import ParameterSource

        if model_tag is not None:
            if ctx.get_parameter_source("bentofile") != ParameterSource.DEFAULT:
                click.echo("-f bentofile is ignored when model_tag is provided")
            cloud_client.pull_model(model_tag, force=force, context=ctx.obj.context)
            return

        try:
            bentofile = resolve_user_filepath(bentofile, None)
        except FileNotFoundError:
            raise InvalidArgument(f'bentofile "{bentofile}" not found')

        with open(bentofile, "r", encoding="utf-8") as f:
            build_config = BentoBuildConfig.from_yaml(f)

        if not build_config.models:
            raise InvalidArgument(
                "No model to pull, please provide a model tag or define models in bentofile.yaml"
            )
        for model_spec in build_config.models:
            cloud_client.pull_model(
                model_spec.tag,
                force=force,
                context=t.cast("SharedOptions", ctx.obj).cloud_context,
                query=model_spec.filter,
            )

    @model_cli.command()
    @click.argument("model_tag", type=click.STRING)
    @click.option(
        "-f",
        "--force",
        is_flag=True,
        default=False,
        help="Forced push to yatai even if it exists in yatai",
    )
    @click.option(
        "-t",
        "--threads",
        default=10,
        help="Number of threads to use for upload",
    )
    @click.pass_obj
    def push(shared_options: SharedOptions, model_tag: str, force: bool, threads: int):  # type: ignore (not accessed)
        """Push Model to a yatai server."""
        model_obj = model_store.get(model_tag)
        if not model_obj:
            raise click.ClickException(f"Model {model_tag} not found in local store")
        cloud_client.push_model(
            model_obj,
            force=force,
            threads=threads,
            context=shared_options.cloud_context,
        )
