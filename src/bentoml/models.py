from __future__ import annotations

import typing as t
from types import ModuleType
from typing import TYPE_CHECKING
from contextlib import contextmanager

from simple_di import inject
from simple_di import Provide

from .exceptions import BentoMLException
from ._internal.tag import Tag
from ._internal.utils import calc_dir_size
from ._internal.models import Model
from ._internal.models import ModelContext
from ._internal.models import ModelOptions
from ._internal.utils.analytics import track
from ._internal.utils.analytics import ModelSaveEvent
from ._internal.configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ._internal.models import ModelStore
    from ._internal.models.model import ModelSignaturesType
    from ._internal.yatai_client import YataiClient


@inject
def list(  # pylint: disable=redefined-builtin
    tag: t.Optional[t.Union[Tag, str]] = None,
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.List["Model"]:
    return _model_store.list(tag)


@inject
def get(
    tag: t.Union[Tag, str],
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "Model":
    return _model_store.get(tag)


@inject
def delete(
    tag: t.Union[Tag, str],
    *,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
):
    _model_store.delete(tag)


@inject
def import_model(
    path: str,
    input_format: t.Optional[str] = None,
    *,
    protocol: t.Optional[str] = None,
    user: t.Optional[str] = None,
    passwd: t.Optional[str] = None,
    params: t.Optional[t.Dict[str, str]] = None,
    subpath: t.Optional[str] = None,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Model:
    """
    Import a bento model exported with :code:`bentoml.models.export_model`. To import a model saved
    with a framework, see the :code:`save` function under the relevant framework, e.g.
    :code:`bentoml.sklearn.save`.

    Examples:

    .. code-block:: python

        # imports 'my_model' from '/path/to/folder/my_model.bentomodel'
        bentoml.models.import_model('/path/to/folder/my_model.bentomodel')

        # imports 'my_model' from '/path/to/folder/my_model.tar.gz'
        # currently supported formats are tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip
        bentoml.models.import_model('/path/to/folder/my_model.tar.gz')
        # treats 'my_model.ext' as a gzipped tarfile
        bentoml.models.import_model('/path/to/folder/my_model.ext', 'gz')

        # imports 'my_model', which is stored as an uncompressed folder, from '/path/to/folder/my_model/'
        bentoml.models.import_model('/path/to/folder/my_model', 'folder')

        # imports 'my_model' from the S3 bucket 'my_bucket', path 'folder/my_model.bentomodel'
        # requires `fs-s3fs <https://pypi.org/project/fs-s3fs/>`_ ('pip install fs-s3fs')
        bentoml.models.import_model('s3://my_bucket/folder/my_model.bentomodel')
        bentoml.models.import_model('my_bucket/folder/my_model.bentomodel', protocol='s3')
        bentoml.models.import_model('my_bucket', protocol='s3', subpath='folder/my_model.bentomodel')
        bentoml.models.import_model('my_bucket', protocol='s3', subpath='folder/my_model.bentomodel',
                                    user='<AWS access key>', passwd='<AWS secret key>',
                                    params={'acl': 'public-read', 'cache-control': 'max-age=2592000,public'})

    For a more comprehensive description of what each of the keyword arguments (:code:`protocol`,
    :code:`user`, :code:`passwd`, :code:`params`, and :code:`subpath`) mean, see the
    `FS URL documentation <https://docs.pyfilesystem.org/en/latest/openers.html>`_.

    Args:
        tag: the tag of the model to export
        path: can be one of two things:
              * a folder on the local filesystem
              * an `FS URL <https://docs.pyfilesystem.org/en/latest/openers.html>`_, for example :code:`'s3://my_bucket/folder/my_model.bentomodel'`
        protocol: (expert) The FS protocol to use when exporting. Some example protocols are :code:`'ftp'`, :code:`'s3'`, and :code:`'userdata'`
        user: (expert) the username used for authentication if required, e.g. for FTP
        passwd: (expert) the username used for authentication if required, e.g. for FTP
        params: (expert) a map of parameters to be passed to the FS used for export, e.g. :code:`{'proxy': 'myproxy.net'}` for setting a proxy for FTP
        subpath: (expert) the path inside the FS that the model should be exported to
        _model_store: the model store to save the model to

    Returns:
        Model: the imported model
    """
    return Model.import_from(
        path,
        input_format,
        protocol=protocol,
        user=user,
        passwd=passwd,
        params=params,
        subpath=subpath,
    ).save(_model_store)


@inject
def export_model(
    tag: t.Union[Tag, str],
    path: str,
    output_format: t.Optional[str] = None,
    *,
    protocol: t.Optional[str] = None,
    user: t.Optional[str] = None,
    passwd: t.Optional[str] = None,
    params: t.Optional[t.Dict[str, str]] = None,
    subpath: t.Optional[str] = None,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Export a BentoML model.

    Examples:

    .. code-block:: python

        # exports 'my_model' to '/path/to/folder/my_model-version.bentomodel' in BentoML's default format
        bentoml.models.export_model('my_model:latest', '/path/to/folder')
        # note that folders can only be passed if exporting to the local filesystem; otherwise the
        # full path, including the desired filename, must be passed

        # exports 'my_model' to '/path/to/folder/my_model.bentomodel' in BentoML's default format
        bentoml.models.export_model('my_model:latest', '/path/to/folder/my_model')
        bentoml.models.export_model('my_model:latest', '/path/to/folder/my_model.bentomodel')

        # exports 'my_model' to '/path/to/folder/my_model.tar.gz in gzip format
        # currently supported formats are tar.gz ('gz'), tar.xz ('xz'), tar.bz2 ('bz2'), and zip
        bentoml.models.export_model('my_model:latest', '/path/to/folder/my_model.tar.gz')
        bentoml.models.export_model('my_model:latest', '/path/to/folder/my_model.tar.gz', 'gz')

        # exports 'my_model' to '/path/to/folder/my_model/ as a folder
        bentoml.models.export_model('my_model:latest', '/path/to/folder/my_model', 'folder')

        # exports 'my_model' to the S3 bucket 'my_bucket' as 'folder/my_model-version.bentomodel'
        bentoml.models.export_model('my_model:latest', 's3://my_bucket/folder')
        bentoml.models.export_model('my_model:latest', 'my_bucket/folder', protocol='s3')
        bentoml.models.export_model('my_model:latest', 'my_bucket', protocol='s3', subpath='folder')
        bentoml.models.export_model('my_model:latest', 'my_bucket', protocol='s3', subpath='folder',
                                    user='<AWS access key>', passwd='<AWS secret key>',
                                    params={'acl': 'public-read', 'cache-control': 'max-age=2592000,public'})

    For a more comprehensive description of what each of the keyword arguments (:code:`protocol`,
    :code:`user`, :code:`passwd`, :code:`params`, and :code:`subpath`) mean, see the
    `FS URL documentation <https://docs.pyfilesystem.org/en/latest/openers.html>`_.

    Args:
        tag: the tag of the model to export
        path: can be one of two things:
              * a folder on the local filesystem
              * an `FS URL <https://docs.pyfilesystem.org/en/latest/openers.html>`_, for example, :code:`'s3://my_bucket/folder/my_model.bentomodel'`
        protocol: (expert) The FS protocol to use when exporting. Some example protocols are :code:`'ftp'`, :code:`'s3'`, and :code:`'userdata'`.
        user: (expert) the username used for authentication if required, e.g. for FTP
        passwd: (expert) the username used for authentication if required, e.g. for FTP
        params: (expert) a map of parameters to be passed to the FS used for export, e.g. :code:`{'proxy': 'myproxy.net'}` for setting a proxy for FTP
        subpath: (expert) the path inside the FS that the model should be exported to
        _model_store: the model store to get the model to save from

    Returns:
        str: A representation of the path that the model was exported to. If it was exported to the local filesystem,
            this will be the OS path to the exported model. Otherwise, it will be an FS URL.
    """
    model = get(tag, _model_store=_model_store)
    return model.export(
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
    force: bool = False,
    _model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    _yatai_client: YataiClient = Provide[BentoMLContainer.yatai_client],
):
    model_obj = _model_store.get(tag)
    if not model_obj:
        raise BentoMLException(f"Model {tag} not found in local store")
    _yatai_client.push_model(model_obj, force=force)


@inject
def pull(
    tag: t.Union[Tag, str],
    *,
    force: bool = False,
    _yatai_client: YataiClient = Provide[BentoMLContainer.yatai_client],
) -> Model:
    return _yatai_client.pull_model(tag, force=force)


@inject
@contextmanager
def create(
    name: str,
    *,
    module: str = "",
    api_version: str | None = None,
    signatures: ModelSignaturesType,
    labels: dict[str, t.Any] | None = None,
    options: ModelOptions | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
    context: ModelContext,
    _model_store: ModelStore = Provide[BentoMLContainer.model_store],
) -> t.Generator[Model, None, None]:
    options = ModelOptions() if options is None else options
    api_version = "v1" if api_version is None else api_version
    res = Model.create(
        name,
        module=module,
        api_version=api_version,
        labels=labels,
        signatures=signatures,
        options=options,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
    )
    external_modules = [] if external_modules is None else external_modules
    imported_modules = []
    try:
        res.enter_cloudpickle_context(external_modules, imported_modules)
        yield res
    except Exception as e:
        raise e
    else:
        res.flush()
        res.save(_model_store)
        track(
            ModelSaveEvent(
                module=res.info.module,
                model_size_in_kb=calc_dir_size(res.path_of("/")) / 1024,
            ),
        )
    finally:
        res.exit_cloudpickle_context(imported_modules)


__all__ = [
    "list",
    "get",
    "delete",
    "import_model",
    "export_model",
    "push",
    "pull",
    "ModelContext",
    "ModelOptions",
]
