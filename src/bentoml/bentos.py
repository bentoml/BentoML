"""
User facing python APIs for managing local bentos and build new bentos.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import typing as t

from simple_di import Provide
from simple_di import inject

from ._internal.bento import Bento
from ._internal.bento.build_config import BentoBuildConfig
from ._internal.configuration.containers import BentoMLContainer
from ._internal.tag import Tag
from ._internal.utils import resolve_user_filepath
from .exceptions import BadInput
from .exceptions import BentoMLException
from .exceptions import InvalidArgument

if t.TYPE_CHECKING:
    from ._internal.bento import BentoStore
    from ._internal.bento.build_config import CondaOptions
    from ._internal.bento.build_config import DockerOptions
    from ._internal.bento.build_config import ModelSpec
    from ._internal.bento.build_config import PythonOptions
    from ._internal.cloud import BentoCloudClient
    from .server import Server


logger = logging.getLogger(__name__)

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


@inject
def list(
    tag: Tag | str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> t.List[Bento]:
    return _bento_store.list(tag)


@inject
def get(
    tag: Tag | str,
    *,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> Bento:
    return _bento_store.get(tag)


@inject
def delete(
    tag: Tag | str,
    *,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
):
    _bento_store.delete(tag)


@inject
def import_bento(
    path: str,
    input_format: str | None = None,
    *,
    protocol: str | None = None,
    user: str | None = None,
    passwd: str | None = None,
    params: t.Optional[t.Dict[str, str]] = None,
    subpath: str | None = None,
    _bento_store: "BentoStore" = Provide[BentoMLContainer.bento_store],
) -> Bento:
    """
    Import a bento.

    Examples:

    .. code-block:: python

        # imports 'my_bento' from '/path/to/folder/my_bento.bento'
        bentoml.import_bento('/path/to/folder/my_bento.bento')

        # imports 'my_bento' from '/path/to/folder/my_bento.tar.gz'
        # currently supported formats are tar.gz ('gz'),
        # tar.xz ('xz'), tar.bz2 ('bz2'), and zip
        bentoml.import_bento('/path/to/folder/my_bento.tar.gz')
        # treats 'my_bento.ext' as a gzipped tarfile
        bentoml.import_bento('/path/to/folder/my_bento.ext', 'gz')

        # imports 'my_bento', which is stored as an
        # uncompressed folder, from '/path/to/folder/my_bento/'
        bentoml.import_bento('/path/to/folder/my_bento', 'folder')

        # imports 'my_bento' from the S3 bucket 'my_bucket',
        # path 'folder/my_bento.bento'
        # requires `fs-s3fs <https://pypi.org/project/fs-s3fs/>`_
        bentoml.import_bento('s3://my_bucket/folder/my_bento.bento')
        bentoml.import_bento('my_bucket/folder/my_bento.bento', protocol='s3')
        bentoml.import_bento('my_bucket', protocol='s3',
                             subpath='folder/my_bento.bento')
        bentoml.import_bento('my_bucket', protocol='s3',
                             subpath='folder/my_bento.bento',
                             user='<AWS access key>', passwd='<AWS secret key>',
                             params={'acl': 'public-read',
                                     'cache-control': 'max-age=2592000,public'})

    For a more comprehensive description of what each of the keyword arguments
    (:code:`protocol`, :code:`user`, :code:`passwd`,
     :code:`params`, and :code:`subpath`) mean, see the
    `FS URL documentation <https://docs.pyfilesystem.org/en/latest/openers.html>`_.

    Args:
        tag: the tag of the bento to export
        path: can be one of two things:
              * a folder on the local filesystem
              * an `FS URL <https://docs.pyfilesystem.org/en/latest/openers.html>`_,
                for example :code:`'s3://my_bucket/folder/my_bento.bento'`
        protocol: (expert) The FS protocol to use when exporting. Some example protocols
                  are :code:`'ftp'`, :code:`'s3'`, and :code:`'userdata'`
        user: (expert) the username used for authentication if required, e.g. for FTP
        passwd: (expert) the username used for authentication if required, e.g. for FTP
        params: (expert) a map of parameters to be passed to the FS used for
                export, e.g. :code:`{'proxy': 'myproxy.net'}` for setting a
                proxy for FTP
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
    tag: Tag | str,
    path: str,
    output_format: str | None = None,
    *,
    protocol: str | None = None,
    user: str | None = None,
    passwd: str | None = None,
    params: dict[str, str] | None = None,
    subpath: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> str:
    """
    Export a bento.

    To export a bento to S3, you must install BentoML with extras ``aws``:

    .. code-block:: bash

       » pip install bentoml[aws]

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
        path: can be either:
              * a folder on the local filesystem
              * an `FS URL <https://docs.pyfilesystem.org/en/latest/openers.html>`_. For example, :code:`'s3://my_bucket/folder/my_bento.bento'`
        protocol: (expert) The FS protocol to use when exporting. Some example protocols are :code:`'ftp'`, :code:`'s3'`, and :code:`'userdata'`
        user: (expert) the username used for authentication if required, e.g. for FTP
        passwd: (expert) the username used for authentication if required, e.g. for FTP
        params: (expert) a map of parameters to be passed to the FS used for export, e.g. :code:`{'proxy': 'myproxy.net'}` for setting a proxy for FTP
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
    tag: Tag | str,
    *,
    force: bool = False,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
):
    """Push Bento to a yatai server."""
    bento = _bento_store.get(tag)
    if not bento:
        raise BentoMLException(f"Bento {tag} not found in local store")
    _cloud_client.push_bento(bento, force=force)


@inject
def pull(
    tag: Tag | str,
    *,
    force: bool = False,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
    _cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
):
    _cloud_client.pull_bento(tag, force=force, bento_store=_bento_store)


@inject
def build(
    service: str,
    *,
    name: str | None = None,
    labels: dict[str, str] | None = None,
    description: str | None = None,
    include: t.List[str] | None = None,
    exclude: t.List[str] | None = None,
    docker: DockerOptions | dict[str, t.Any] | None = None,
    python: PythonOptions | dict[str, t.Any] | None = None,
    conda: CondaOptions | dict[str, t.Any] | None = None,
    models: t.List[ModelSpec | str | dict[str, t.Any]] | None = None,
    version: str | None = None,
    build_ctx: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> Bento:
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

    """
    build_config = BentoBuildConfig(
        service=service,
        name=name,
        description=description,
        labels=labels,
        include=include,
        exclude=exclude,
        docker=docker,
        python=python,
        conda=conda,
        models=models or [],
    )

    build_args = [sys.executable, "-m", "bentoml", "build"]

    if build_ctx is None:
        build_ctx = "."
    build_args.append(build_ctx)

    if version is not None:
        build_args.extend(["--version", version])
    build_args.extend(["--output", "tag"])

    copied = os.environ.copy()
    copied.setdefault("BENTOML_HOME", BentoMLContainer.bentoml_home.get())

    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", prefix="bentoml-build-", suffix=".yaml"
    ) as f:
        build_config.to_yaml(f)
        bentofile_path = os.path.join(os.path.dirname(f.name), f.name)
        build_args.extend(["--bentofile", bentofile_path])
        try:
            return get(
                _parse_tag_from_outputs(
                    subprocess.check_output(build_args, env=copied)
                ),
                _bento_store=_bento_store,
            )
        except subprocess.CalledProcessError as e:
            raise BentoMLException(
                f"Failed to build BentoService bundle (Lookup for traceback):\n{e}"
            ) from e


def _parse_tag_from_outputs(output: bytes) -> str:
    matched = re.search(
        r"^__tag__:([^:\n]+:[^:\n]+)$",
        output.decode("utf-8").strip(),
        flags=re.MULTILINE,
    )
    if matched is None:
        raise BentoMLException(
            f"Failed to find tag from output: {output}\nNote: Output from 'bentoml build' might not be correct. Please open an issue on GitHub."
        )
    return matched.group(1)


@inject
def build_bentofile(
    bentofile: str = "bentofile.yaml",
    *,
    version: str | None = None,
    build_ctx: str | None = None,
    _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> Bento:
    """
    Build a Bento base on options specified in a bentofile.yaml file.

    By default, this function will look for a `bentofile.yaml` file in current working
    directory.

    Args:
        bentofile: The file path to build config yaml file
        version: Override the default auto generated version str
        build_ctx: Build context directory, when used as
        _bento_store: save Bento created to this BentoStore
    """
    try:
        bentofile = resolve_user_filepath(bentofile, build_ctx)
    except FileNotFoundError:
        raise InvalidArgument(f'bentofile "{bentofile}" not found')

    build_args = [sys.executable, "-m", "bentoml", "build"]
    if build_ctx is None:
        build_ctx = "."
    build_args.append(build_ctx)
    if version is not None:
        build_args.extend(["--version", version])
    build_args.extend(["--bentofile", bentofile, "--output", "tag"])

    copied = os.environ.copy()
    copied.setdefault("BENTOML_HOME", BentoMLContainer.bentoml_home.get())
    try:
        return get(
            _parse_tag_from_outputs(subprocess.check_output(build_args, env=copied)),
            _bento_store=_bento_store,
        )
    except subprocess.CalledProcessError as e:
        raise BentoMLException(
            f"Failed to build BentoService bundle (Lookup for traceback):\n{e}"
        ) from e


def containerize(bento_tag: Tag | str, **kwargs: t.Any) -> bool:
    """
    DEPRECATED: Use :meth:`bentoml.container.build` instead.
    """
    from .container import build

    # Add backward compatibility for bentoml.bentos.containerize
    logger.warning(
        "'%s.containerize' is deprecated, use '%s.build' instead.",
        __name__,
        "bentoml.container",
    )
    if "docker_image_tag" in kwargs:
        kwargs["image_tag"] = kwargs.pop("docker_image_tag", None)
    if "labels" in kwargs:
        kwargs["label"] = kwargs.pop("labels", None)
    if "tags" in kwargs:
        kwargs["tag"] = kwargs.pop("tags", None)
    try:
        build(bento_tag, **kwargs)
        return True
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Failed to containerize %s: %s", bento_tag, e)
        return False


@inject
def serve(
    bento: str | Tag | Bento,
    server_type: str = "http",
    reload: bool = False,
    production: bool = False,
    env: t.Literal["conda"] | None = None,
    host: str | None = None,
    port: int | None = None,
    working_dir: str | None = None,
    api_workers: int | None = Provide[BentoMLContainer.api_server_workers],
    backlog: int = Provide[BentoMLContainer.api_server_config.backlog],
    ssl_certfile: str | None = Provide[BentoMLContainer.ssl.certfile],
    ssl_keyfile: str | None = Provide[BentoMLContainer.ssl.keyfile],
    ssl_keyfile_password: str | None = Provide[BentoMLContainer.ssl.keyfile_password],
    ssl_version: int | None = Provide[BentoMLContainer.ssl.version],
    ssl_cert_reqs: int | None = Provide[BentoMLContainer.ssl.cert_reqs],
    ssl_ca_certs: str | None = Provide[BentoMLContainer.ssl.ca_certs],
    ssl_ciphers: str | None = Provide[BentoMLContainer.ssl.ciphers],
    enable_reflection: bool = Provide[BentoMLContainer.grpc.reflection.enabled],
    enable_channelz: bool = Provide[BentoMLContainer.grpc.channelz.enabled],
    max_concurrent_streams: int
    | None = Provide[BentoMLContainer.grpc.max_concurrent_streams],
    grpc_protocol_version: str | None = None,
) -> Server[t.Any]:
    logger.warning(
        "bentoml.serve and bentoml.bentos.serve are deprecated; use bentoml.Server instead."
    )

    if server_type == "http":
        from .server import HTTPServer

        if host is None:
            host = t.cast(str, BentoMLContainer.http.host.get())
        if port is None:
            port = t.cast(int, BentoMLContainer.http.port.get())

        res = HTTPServer(
            bento=bento,
            reload=reload,
            production=production,
            env=env,
            host=host,
            port=port,
            working_dir=working_dir,
            api_workers=api_workers,
            backlog=backlog,
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
            ssl_keyfile_password=ssl_keyfile_password,
            ssl_version=ssl_version,
            ssl_cert_reqs=ssl_cert_reqs,
            ssl_ca_certs=ssl_ca_certs,
            ssl_ciphers=ssl_ciphers,
        )
    elif server_type == "grpc":
        from .server import GrpcServer

        if host is None:
            host = t.cast(str, BentoMLContainer.grpc.host.get())
        if port is None:
            port = t.cast(int, BentoMLContainer.grpc.port.get())

        res = GrpcServer(
            bento=bento,
            reload=reload,
            production=production,
            env=env,
            host=host,
            port=port,
            working_dir=working_dir,
            api_workers=api_workers,
            backlog=backlog,
            enable_reflection=enable_reflection,
            enable_channelz=enable_channelz,
            max_concurrent_streams=max_concurrent_streams,
            grpc_protocol_version=grpc_protocol_version,
        )
    else:
        raise BadInput(f"Unknown server type: '{server_type}'")

    res.start()
    return res
