"""
User facing python APIs for managing local bentos and build new bentos
"""

from __future__ import annotations

import os
import typing as t
import logging
import subprocess
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml.exceptions import InvalidArgument

from ._internal.tag import Tag
from ._internal.bento import Bento
from ._internal.utils import resolve_user_filepath
from ._internal.bento.build_config import BentoBuildConfig
from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ._internal.bento import BentoStore
    from ._internal.types import PathType
    from ._internal.models import ModelStore

logger = logging.getLogger(__name__)

BENTOML_FIGLET = """
██████╗░███████╗███╗░░██╗████████╗░█████╗░███╗░░░███╗██╗░░░░░
██╔══██╗██╔════╝████╗░██║╚══██╔══╝██╔══██╗████╗░████║██║░░░░░
██████╦╝█████╗░░██╔██╗██║░░░██║░░░██║░░██║██╔████╔██║██║░░░░░
██╔══██╗██╔══╝░░██║╚████║░░░██║░░░██║░░██║██║╚██╔╝██║██║░░░░░
██████╦╝███████╗██║░╚███║░░░██║░░░╚█████╔╝██║░╚═╝░██║███████╗
╚═════╝░╚══════╝╚═╝░░╚══╝░░░╚═╝░░░░╚════╝░╚═╝░░░░░╚═╝╚══════╝
"""


@inject
def list(  # pylint: disable=redefined-builtin
    tag: t.Optional[t.Union[Tag, str]] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> "t.List[Bento]":
    return _bento_store.list(tag)


@inject
def get(
    tag: t.Union[Tag, str],
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> Bento:
    return _bento_store.get(tag)


@inject
def delete(
    tag: t.Union[Tag, str],
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    _bento_store.delete(tag)


@inject
def import_bento(
    path: str,
    input_format: t.Optional[str] = None,
    *,
    protocol: t.Optional[str] = None,
    user: t.Optional[str] = None,
    passwd: t.Optional[str] = None,
    params: t.Optional[t.Dict[str, str]] = None,
    subpath: t.Optional[str] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> Bento:
    """
    Import a bento.

    Examples:

    .. code-block:: python

        # imports 'my_bento' from '/path/to/folder/my_bento.bento'
        bentoml.import_bento('/path/to/folder/my_bento.bento')

        # imports 'my_bento' from '/path/to/folder/my_bento.tar.gz'
        # currently supported formats are tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip
        bentoml.import_bento('/path/to/folder/my_bento.tar.gz')
        # treats 'my_bento.ext' as a gzipped tarfile
        bentoml.import_bento('/path/to/folder/my_bento.ext', 'gz')

        # imports 'my_bento', which is stored as an uncompressed folder, from '/path/to/folder/my_bento/'
        bentoml.import_bento('/path/to/folder/my_bento', 'folder')

        # imports 'my_bento' from the S3 bucket 'my_bucket', path 'folder/my_bento.bento'
        # requires `fs-s3fs <https://pypi.org/project/fs-s3fs/>`_ ('pip install fs-s3fs')
        bentoml.import_bento('s3://my_bucket/folder/my_bento.bento')
        bentoml.import_bento('my_bucket/folder/my_bento.bento', protocol='s3')
        bentoml.import_bento('my_bucket', protocol='s3', subpath='folder/my_bento.bento')
        bentoml.import_bento('my_bucket', protocol='s3', subpath='folder/my_bento.bento',
                             user='<AWS access key>', passwd='<AWS secret key>',
                             params={'acl': 'public-read', 'cache-control': 'max-age=2592000,public'})

    For a more comprehensive description of what each of the keyword arguments (:code:`protocol`,
    :code:`user`, :code:`passwd`, :code:`params`, and :code:`subpath`) mean, see the
    `FS URL documentation <https://docs.pyfilesystem.org/en/latest/openers.html>`_.

    Args:
        tag: the tag of the bento to export
        path: can be one of two things:
            * a folder on the local filesystem
            * an `FS URL <https://docs.pyfilesystem.org/en/latest/openers.html>`_, for example
                :code:`'s3://my_bucket/folder/my_bento.bento'`
        protocol: (expert) The FS protocol to use when exporting. Some example protocols are :code:`'ftp'`,
            :code:`'s3'`, and :code:`'userdata'`
        user: (expert) the username used for authentication if required, e.g. for FTP
        passwd: (expert) the username used for authentication if required, e.g. for FTP
        params: (expert) a map of parameters to be passed to the FS used for export, e.g. :code:`{'proxy': 'myproxy.net'}`
            for setting a proxy for FTP
        subpath: (expert) the path inside the FS that the bento should be exported to
        _bento_store: the bento store to save the bento to

    Returns:
        Bento: the imported bento
    """
    return Bento.import_from(
        path,
        input_format,
        protocol=protocol,
        user=user,
        passwd=passwd,
        params=params,
        subpath=subpath,
    ).save(_bento_store)


@inject
def export_bento(
    tag: t.Union[Tag, str],
    path: str,
    output_format: t.Optional[str] = None,
    *,
    protocol: t.Optional[str] = None,
    user: t.Optional[str] = None,
    passwd: t.Optional[str] = None,
    params: t.Optional[t.Dict[str, str]] = None,
    subpath: t.Optional[str] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> str:
    """
    Export a bento.

    Examples:

    .. code-block:: python

        # exports 'my_bento' to '/path/to/folder/my_bento-version.bento' in BentoML's default format
        bentoml.export_bento('my_bento:latest', '/path/to/folder')
        # note that folders can only be passed if exporting to the local filesystem; otherwise the
        # full path, including the desired filename, must be passed

        # exports 'my_bento' to '/path/to/folder/my_bento.bento' in BentoML's default format
        bentoml.export_bento('my_bento:latest', '/path/to/folder/my_bento')
        bentoml.export_bento('my_bento:latest', '/path/to/folder/my_bento.bento')

        # exports 'my_bento' to '/path/to/folder/my_bento.tar.gz' in gzip format
        # currently supported formats are tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip
        bentoml.export_bento('my_bento:latest', '/path/to/folder/my_bento.tar.gz')
        # outputs a gzipped tarfile as 'my_bento.ext'
        bentoml.export_bento('my_bento:latest', '/path/to/folder/my_bento.ext', 'gz')

        # exports 'my_bento' to '/path/to/folder/my_bento/' as a folder
        bentoml.export_bento('my_bento:latest', '/path/to/folder/my_bento', 'folder')

        # exports 'my_bento' to the S3 bucket 'my_bucket' as 'folder/my_bento-version.bento'
        bentoml.export_bento('my_bento:latest', 's3://my_bucket/folder')
        bentoml.export_bento('my_bento:latest', 'my_bucket/folder', protocol='s3')
        bentoml.export_bento('my_bento:latest', 'my_bucket', protocol='s3', subpath='folder')
        bentoml.export_bento('my_bento:latest', 'my_bucket', protocol='s3', subpath='folder',
                             user='<AWS access key>', passwd='<AWS secret key>',
                             params={'acl': 'public-read', 'cache-control': 'max-age=2592000,public'})

    For a more comprehensive description of what each of the keyword arguments (:code:`protocol`,
    :code:`user`, :code:`passwd`, :code:`params`, and :code:`subpath`) mean, see the
    `FS URL documentation <https://docs.pyfilesystem.org/en/latest/openers.html>`_.

    Args:
        tag: the tag of the Bento to export
        path: can be one of two things:
            * a folder on the local filesystem
            * an `FS URL <https://docs.pyfilesystem.org/en/latest/openers.html>`_
                * for example, :code:`'s3://my_bucket/folder/my_bento.bento'`
        protocol: (expert) The FS protocol to use when exporting. Some example protocols are :code:`'ftp'`,
            :code:`'s3'`, and :code:`'userdata'`
        user: (expert) the username used for authentication if required, e.g. for FTP
        passwd: (expert) the username used for authentication if required, e.g. for FTP
        params: (expert) a map of parameters to be passed to the FS used for export, e.g. :code:`{'proxy': 'myproxy.net'}`
            for setting a proxy for FTP
        subpath: (expert) the path inside the FS that the bento should be exported to
        _bento_store: save Bento created to this BentoStore

    Returns:
        str: A representation of the path that the Bento was exported to. If it was exported to the local filesystem,
            this will be the OS path to the exported Bento. Otherwise, it will be an FS URL.
    """
    bento = get(tag, _bento_store=_bento_store)
    return bento.export(
        path,
        output_format,
        protocol=protocol,
        user=user,
        passwd=passwd,
        params=params,
        subpath=subpath,
    )


@inject
def push(
    tag: t.Union[Tag, str],
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    raise NotImplementedError


@inject
def pull(
    tag: t.Union[Tag, str],
    *,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
):
    raise NotImplementedError


@inject
def build(
    service: str,
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    description: t.Optional[str] = None,
    include: t.Optional[t.List[str]] = None,
    exclude: t.Optional[t.List[str]] = None,
    docker: t.Optional[t.Dict[str, t.Any]] = None,
    python: t.Optional[t.Dict[str, t.Any]] = None,
    conda: t.Optional[t.Dict[str, t.Any]] = None,
    version: t.Optional[str] = None,
    build_ctx: t.Optional[str] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "Bento":
    """
    User-facing API for building a Bento. The available build options are identical to the keys of a
    valid 'bentofile.yaml' file.

    This API will not respect any 'bentofile.yaml' files. Build options should instead be provided
    via function call parameters.

    Args:
        service: import str for finding the bentoml.Service instance build target
        labels: optional immutable labels for carrying contextual info
        description: optional description string in markdown format
        include: list of file paths and patterns specifying files to include in Bento,
            default is all files under build_ctx, beside the ones excluded from the
            exclude parameter or a :code:`.bentoignore` file for a given directory
        exclude: list of file paths and patterns to exclude from the final Bento archive
        docker: dictionary for configuring Bento's containerization process, see details
            in :class:`bentoml._internal.bento.build_config.DockerOptions`
        python: dictionary for configuring Bento's python dependencies, see details in
            :class:`bentoml._internal.bento.build_config.PythonOptions`
        conda: dictionary for configuring Bento's conda dependencies, see details in
            :class:`bentoml._internal.bento.build_config.CondaOptions`
        version: Override the default auto generated version str
        build_ctx: Build context directory, when used as
        _bento_store: save Bento created to this BentoStore
        _model_store: pull Models required from this ModelStore

    Returns:
        Bento: a Bento instance representing the materialized Bento saved in BentoStore

    Example:

        .. code-block::
            import bentoml

            bentoml.build(
                service="fraud_detector.py:svc",
                version="any_version_label",  # override default version generator
                description=open("README.md").read(),
                include=['*'],
                exclude=[], # files to exclude can also be specified with a .bentoignore file
                labels={
                    "foo": "bar",
                    "team": "abc"
                },
                python=dict(
                    packages=["tensorflow", "numpy"],
                    # requirements_txt="./requirements.txt",
                    index_url="http://<api token>:@mycompany.com/pypi/simple",
                    trusted_host=["mycompany.com"],
                    find_links=['thirdparty..'],
                    extra_index_url=["..."],
                    pip_args="ANY ADDITIONAL PIP INSTALL ARGS",
                    wheels=["./wheels/*"],
                    lock_packages=True,
                ),
                docker=dict(
                    distro="amazonlinux2",
                    setup_script="setup_docker_container.sh",
                    python_version="3.8",
                ),
            )

    """  # noqa: LN001
    build_config = BentoBuildConfig(
        service=service,
        description=description,
        labels=labels,
        include=include,
        exclude=exclude,
        docker=docker,  # type: ignore
        python=python,  # type: ignore
        conda=conda,  # type: ignore
    )

    bento = Bento.create(
        build_config=build_config,
        version=version,
        build_ctx=build_ctx,
    ).save(_bento_store)
    logger.info(BENTOML_FIGLET)
    logger.info('Successfully built %s at "%s"', bento, bento.path)
    return bento


@inject
def build_bentofile(
    bentofile: str = "bentofile.yaml",
    *,
    version: t.Optional[str] = None,
    build_ctx: t.Optional[str] = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "Bento":
    """
    Build a Bento base on options specified in a bentofile.yaml file.

    By default, this function will look for a `bentofile.yaml` file in current working
    directory.

    Args:
        bentofile: The file path to build config yaml file
        version: Override the default auto generated version str
        build_ctx: Build context directory, when used as
        _bento_store: save Bento created to this BentoStore
        _model_store: pull Models required from this ModelStore
    """
    try:
        bentofile = resolve_user_filepath(bentofile, build_ctx)
    except FileNotFoundError:
        raise InvalidArgument(f'bentofile "{bentofile}" not found')

    with open(bentofile, "r", encoding="utf-8") as f:
        build_config = BentoBuildConfig.from_yaml(f)

    bento = Bento.create(
        build_config=build_config,
        version=version,
        build_ctx=build_ctx,
    ).save(_bento_store)
    logger.info(BENTOML_FIGLET)
    logger.info('Successfully built %s at "%s"', bento, bento.path)
    return bento


@inject
def containerize(
    tag: Tag | str,
    docker_image_tag: str | None = None,
    *,
    add_host: dict[str, str] | None = None,
    allow: t.List[str] | None = None,
    build_args: dict[str, str] | None = None,
    build_context: dict[str, str] | None = None,
    builder: str | None = None,
    cache_from: str | t.List[str] | dict[str, str] | None = None,
    cache_to: str | t.List[str] | dict[str, str] | None = None,
    cgroup_parent: str | None = None,
    iidfile: PathType | None = None,
    labels: dict[str, str] | None = None,
    load: bool = True,
    metadata_file: PathType | None = None,
    network: str | None = None,
    no_cache: bool = False,
    no_cache_filter: t.List[str] | None = None,
    output: str | dict[str, str] | None = None,
    platform: str | t.List[str] | None = None,
    progress: t.Literal["auto", "tty", "plain"] = "auto",
    pull: bool = False,
    push: bool = False,
    quiet: bool = False,
    secrets: str | t.List[str] | None = None,
    shm_size: str | int | None = None,
    rm: bool = False,
    ssh: str | None = None,
    target: str | None = None,
    ulimit: str | None = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> bool:

    from bentoml._internal.utils import buildx

    env = {"DOCKER_BUILDKIT": "1", "DOCKER_SCAN_SUGGEST": "false"}

    # run health check whether buildx is install locally
    buildx.health()

    bento = _bento_store.get(tag)
    if docker_image_tag is None:
        docker_image_tag = str(bento.tag)

    dockerfile_path = os.path.join("env", "docker", "Dockerfile")

    logger.info(f"Building docker image for {bento}...")
    try:
        buildx.build(
            subprocess_env=env,
            cwd=bento.path,
            file=dockerfile_path,
            tags=docker_image_tag,
            add_host=add_host,
            allow=allow,
            build_args=build_args,
            build_context=build_context,
            builder=builder,
            cache_from=cache_from,
            cache_to=cache_to,
            cgroup_parent=cgroup_parent,
            iidfile=iidfile,
            labels=labels,
            load=load,
            metadata_file=metadata_file,
            network=network,
            no_cache=no_cache,
            no_cache_filter=no_cache_filter,
            output=output,
            platform=platform,
            progress=progress,
            pull=pull,
            push=push,
            quiet=quiet,
            secrets=secrets,
            shm_size=shm_size,
            rm=rm,
            ssh=ssh,
            target=target,
            ulimit=ulimit,
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed building docker image: {e}")
        if platform != "linux/amd64":
            logger.debug(
                f"""If you run into the following error: "failed to solve: pull access denied, repository does not exist or may require authorization: server message: insufficient_scope: authorization failed". This means Docker doesn't have context of your build platform {platform}. By default BentoML will set target build platform to the current machine platform via `uname -m`. Try again by specifying to build x86_64 (amd64) platform: bentoml containerize {str(bento.tag)} --platform linux/amd64"""
            )
        return False
    else:
        logger.info(f'Successfully built docker image "{docker_image_tag}"')
        return True


__all__ = [
    "list",
    "get",
    "delete",
    "import_bento",
    "export_bento",
    "push",
    "pull",
    "build",
    "build_bentofile",
    "containerize",
]
