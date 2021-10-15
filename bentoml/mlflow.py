import functools
import os
import typing as t
from hashlib import sha256
from pathlib import Path

from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from .exceptions import InvalidArgument, MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    from _internal.models.store import ModelStore
    from mlflow.pyfunc import PyFuncModel

try:
    import mlflow
    import mlflow.models
    from mlflow.exceptions import MlflowException
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
    from mlflow.tracking.artifact_utils import _download_artifact_from_uri
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        `mlflow` is required to use with `bentoml.mlflow`.
        Instruction: `pip install -U mlflow`
        """
    )

_MLFLOW_PROJECT_FILENAME = "MLproject"


def _uri_to_filename(uri: str) -> str:
    return f"mlflow{sha256(uri.encode('utf-8')).hexdigest()}"


def _load(
    tag: str,
    model_store: "ModelStore",
    load_as_projects: bool = False,
    **mlflow_project_run_kwargs: str,
) -> t.Union[t.Tuple[str, t.Callable[..., t.Any]], "PyFuncModel"]:
    model_info = model_store.get(tag)

    if "import_from_uri" in model_info.context:
        if load_as_projects:
            assert _MLFLOW_PROJECT_FILENAME.lower() in model_info.options

            uri = model_info.options["artifacts_path"]
            func: t.Callable[[str], t.Callable[..., t.Any]] = functools.partial(
                mlflow.projects.run, **mlflow_project_run_kwargs
            )
            return uri, func
        else:
            mlmodel_fpath = Path(
                model_info.options["artifacts_path"], MLMODEL_FILE_NAME
            )
            if not mlmodel_fpath.exists():
                raise MlflowException(
                    f"Could not find an '{MLMODEL_FILE_NAME}'"
                    f" configuration file at '{str(model_info.path)}'",
                    RESOURCE_DOES_NOT_EXIST,
                )
            model = mlflow.models.Model.load(mlmodel_fpath)

            # flavors will usually consists of two keys,
            #  one is the frameworks impl, the other is
            #  mlflow's python function. We will try to
            #  use the python function dictionary to load
            #  the model instance
            pyfunc_config = list(map(lambda x: x[1], model.flavors.items()))[1]  # type: ignore[no-any-return] # noqa # pylint: disable
            loader_module = getattr(mlflow, pyfunc_config["loader_module"])
    else:
        loader_module = mlflow.pyfunc
    return loader_module.load_model(os.path.join(model_info.path, SAVE_NAMESPACE))


@inject
def load_projects(
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **mlflow_project_run_kwargs: str,
) -> t.Tuple[str, t.Callable[..., t.Any]]:
    return _load(
        tag=tag,
        model_store=model_store,
        load_as_projects=True,
        **mlflow_project_run_kwargs,
    )


@inject
def load(
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "PyFuncModel":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `mlflow.pyfunc.PyFuncModel` from BentoML modelstore.

    Examples::
    """  # noqa
    return _load(tag=tag, model_store=model_store)


def _save_to_modelstore(
    name: str,
    identifier: t.Union["mlflow.models.Model", str],
    loader_module: t.Type["mlflow.pyfunc"],
    metadata: t.Optional[t.Dict[str, t.Any]],
    model_store: "ModelStore",
) -> str:
    context = {"mlflow": mlflow.__version__, "import_from_uri": False}
    if isinstance(identifier, str):
        context["import_from_uri"] = True
    else:
        assert loader_module is not None, (
            "`loader_module` cannot be None" " when saving model."
        )
        if "mlflow" not in loader_module.__name__:
            raise InvalidArgument("given `loader_module` is not omitted by mlflow.")
    with model_store.register(
        name,
        module=__name__,
        options=None,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        if isinstance(identifier, str):
            ctx.options = {"uri": identifier, _MLFLOW_PROJECT_FILENAME.lower(): False}
            project_or_artifact_path = _download_artifact_from_uri(
                identifier, output_path=ctx.path
            )
            _g = Path(project_or_artifact_path).rglob(_MLFLOW_PROJECT_FILENAME)
            if len(list(_g)) != 0:
                ctx.options[_MLFLOW_PROJECT_FILENAME.lower()] = True
            ctx.options["artifacts_path"] = project_or_artifact_path
        else:
            assert loader_module is not None, (
                "`loader_module` is required"
                " in order to save given model"
                " from MLflow."
            )
            loader_module.save_model(identifier, os.path.join(ctx.path, SAVE_NAMESPACE))
        return ctx.tag  # type: ignore


@inject
def save(
    name: str,
    model: "mlflow.models.Model",
    loader_module: t.Type["mlflow.pyfunc"],
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`mlflow.models.Model`):
            All mlflow models are of type :obj:`mlflow.models.Model`
        loader_module (`types.ModuleType`):
            flavors supported by :obj:`mlflow`
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
    """  # noqa
    return _save_to_modelstore(
        name=name,
        identifier=model,
        loader_module=loader_module,
        metadata=metadata,
        model_store=model_store,
    )


@inject
def import_from_uri(
    uri: str,
    *,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    return _save_to_modelstore(
        name=_uri_to_filename(uri),
        identifier=uri,
        loader_module=mlflow.pyfunc,
        metadata=metadata,
        model_store=model_store,
    )


class _MLflowRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        self._model_store = model_store

    @property
    def required_models(self) -> t.List[str]:
        return [self._model_store.get(self.name).tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return 1

    @property
    def num_replica(self) -> int:
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        ...

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: t.Any) -> t.Any:  # type: ignore[override]
        ...


@inject
def load_runner(
    tag: str,
    *,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_MLflowRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.mlflow.load_runner` implements a Runner class that
    has to options to wrap around a PyFuncModel, or infer from given MLflow flavor to
    load BentoML internal Runner implementation, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances loaded from `bentoml.mlflow`. This can be either a `mlflow.pyfunc.PyFuncModel`
        Runner or BentoML runner instance that maps to MLflow flavors

    Examples::
    """  # noqa
    return _MLflowRunner(
        tag=tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
