import functools
import importlib
import logging
import os
import re
import typing as t
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlparse

import yaml
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from .exceptions import BentoMLException, InvalidArgument, MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import

    from _internal.models.store import ModelStore, StoreCtx
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

logger = logging.getLogger(__name__)

_RunnerType = t.Type[Runner]

_MLFLOW_PROJECT_FILENAME = "MLproject"

_S3_ERR_MSG = """\
In order to import from S3 bucket, make sure to setup your credentials
 either via environment variables `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
 and `AWS_DEFAULT_REGION`. Or one can choose to setup credentials under `$HOME/.aws/credentials`.
 Refers to https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-envvars.html
"""  # noqa

_NOT_MLFLOW_MODEL_ERR_MSG = """\
{path} directory is not a valid MLflow Model format. Possible problem is that your S3 bucket contains
 incorrect format of MLflow Model.
"""  # noqa

_PYFUNCRUNNER_WARNING = """\
Loading {name} as a PyFuncRunner. This runner is a light wrapper around `mlflow.models.Models`
 and provide a easy way to run prediction. However, this is NOT OPTIMIZED FOR PERFORMANCE and
 it is RECOMMENDED to use BentoML internal frameworks runner implementation via `load_runner()`
 for the best performance.
"""  # noqa


def _clean_name(name: str) -> str:
    return re.sub(r"\W|^(?=\d)-", "_", name)


def _is_s3_url(uri: str) -> bool:
    return urlparse(uri).scheme.startswith("s3")


def _uri_to_filename(uri: str) -> str:
    return f"b{sha256(uri.encode('utf-8')).hexdigest()[:23]}"


def _load(
    tag: str,
    model_store: "ModelStore",
    load_as_projects: bool = False,
    **mlflow_project_run_kwargs: str,
) -> t.Union[t.Tuple[t.Callable[..., t.Any], str], "PyFuncModel"]:
    model_info = model_store.get(tag)
    if model_info.context["import_from_uri"]:
        if load_as_projects:
            assert "mlproject_path" in model_info.options
            func: t.Callable[[str], t.Callable[..., t.Any]] = functools.partial(
                mlflow.projects.run, **mlflow_project_run_kwargs
            )
            return func, model_info.options["mlproject_path"]
        else:
            mlmodel_fpath = Path(model_info.path, MLMODEL_FILE_NAME)
            if not mlmodel_fpath.exists():
                raise MlflowException(
                    f"Could not find '{MLMODEL_FILE_NAME}'"
                    f" configuration file at '{str(model_info.path)}'",
                    RESOURCE_DOES_NOT_EXIST,
                )
            model = mlflow.models.Model.load(mlmodel_fpath)

            # Flavors will usually consists of two keys,
            #  one is the frameworks impl, the other is
            #  mlflow's python function. We will try to
            #  use the python function dictionary to load
            #  the model instance

            # fmt: off
            pyfunc_config = list(map(lambda x: x[1], model.flavors.items()))[1]  # type: ignore[no-any-return] # noqa # pylint: disable
            loader_module = getattr(mlflow, pyfunc_config["loader_module"])
            # fmt: on
    else:
        loader_module = mlflow.pyfunc
    return loader_module.load_model(os.path.join(model_info.path, SAVE_NAMESPACE))


# TODO: supports different options for different type of tracking uri runs
@inject
def load_project(
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **mlflow_project_run_kwargs: str,
) -> t.Tuple[t.Callable[..., t.Any], str]:
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


def _check_for_mlmodel_path(path: str, ctx: "StoreCtx") -> bool:
    _mlmodel_path = None
    for f in Path(path).iterdir():
        if f.is_dir():
            try:
                _mlmodel_path = [i for i in f.rglob("MLmodel")][0]
            except IndexError:
                pass
        elif f.resolve().name == "MLmodel":
            _mlmodel_path = f
        else:
            continue
    if _mlmodel_path is not None:
        with _mlmodel_path.open("r") as mf:
            conf = yaml.safe_load(mf.read())
            # fmt: off
            ctx.options["flavor"] = conf["flavors"]["python_function"]["loader_module"]  # noqa # pylint: disable
            # fmt: on
        return True
    return False


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
            ctx.options = {"uri": identifier}
            mlflow_obj_path = _download_artifact_from_uri(
                identifier, output_path=ctx.path
            )
            p_iter = Path(mlflow_obj_path).rglob(f"**/{_MLFLOW_PROJECT_FILENAME}")
            try:
                ctx.options["mlproject_path"] = [str(i.parent) for i in p_iter][0]
                mlproject_exists = True
            except IndexError:
                mlproject_exists = False
            try:
                mlmodel_exists = _check_for_mlmodel_path(mlflow_obj_path, ctx)
            except (NotADirectoryError, NameError):
                mlmodel_exists = False
            if not mlmodel_exists and not mlproject_exists:
                logger.warning(
                    f"`{identifier.split('/')[-1]}` is neither MLflow Projects"
                    " nor MLflow Models. This will result in errors when loading"
                    f" {ctx.tag}. If you to import your saved weight from S3 refers"
                    f" to correct framework of choice for its S3 related-functions."
                )
        else:
            assert loader_module is not None, (
                "`loader_module` is required"
                " in order to save given model"
                " from MLflow."
            )
            ctx.options["flavor"] = loader_module.__name__
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
        loader_module (`t.Type["mlflow.pyfunc"]`):
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
    s3_default_region: t.Optional[str] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    if _is_s3_url(uri):
        assert all(
            i in os.environ for i in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        ), _S3_ERR_MSG
        if "AWS_DEFAULT_REGION" not in os.environ:
            assert s3_default_region is not None, (
                "Since you don't have `AWS_DEFAULT_REGION` set"
                " in your envars, make sure to provide it via"
                " `s3_default_region`."
            )
            os.environ["AWS_DEFAULT_REGION"] = s3_default_region
    return _save_to_modelstore(
        name=_uri_to_filename(uri),
        identifier=uri,
        loader_module=mlflow.pyfunc,
        metadata=metadata,
        model_store=model_store,
    )


class _PyFuncRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        logger.warning(_PYFUNCRUNNER_WARNING)
        self._model_store = model_store
        self._model_info = self._model_store.get(tag)

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
        try:
            self._model = mlflow.pyfunc.load_model(
                os.path.join(self._model_info.path, SAVE_NAMESPACE)
            )
        except FileNotFoundError:
            for f in Path(self._model_info.path).iterdir():
                if f.is_dir():
                    assert "MLmodel" in [
                        _.name for _ in f.iterdir()
                    ], _NOT_MLFLOW_MODEL_ERR_MSG.format(path=f)
                    self._model = mlflow.pyfunc.load_model(str(f))
                file = f.resolve()  # file could be symlink
                if file.name == "MLmodel":
                    self._model = mlflow.pyfunc.load_model(str(f.parent))
                    break
            raise BentoMLException(
                "Unable to load a valid MLmodel during `_setup()`. Make sure"
                f" {self.name} follows MLflow Models format."
            )
        finally:
            self._predict_fn = getattr(self._model, "predict")

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: t.Any) -> t.Any:  # type: ignore[override]
        return self._predict_fn(input_data)


class _MLflowRunner:
    def __init__(self, *args: str, **kwargs: str):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be initialized to"
            f" its framework implementation via"
            f" `{self.__class__.__name__}.to_runner_impl(tag,resource_quota,batch_options, **kwargs)`"  # noqa # pylint: disable
        )

    @classmethod
    @inject
    def to_runner_impl(
        cls,
        tag: str,
        *runners_args: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
        **runners_kwargs: str,
    ) -> "Runner":
        model_info = model_store.get(tag)
        try:
            _, *flavors = model_info.options["flavor"].split(".")
            flavor = flavors[0]
            if flavor == "pyfunc":
                flavor = "mlflow"
            return getattr(importlib.import_module(f"bentoml.{flavor}"), "load_runner")(  # type: ignore # noqa # pylint: disable
                tag,
                *runners_args,
                resource_quota=resource_quota,
                batch_options=batch_options,
                model_store=model_store,
                **runners_kwargs,
            )
        except KeyError:
            if "mlproject_path" in model_info.options:
                raise BentoMLException(
                    """\
                Runner is not designed to run MLflow Projects.
                If you wish to serve the results of MLflow Projects, it is recommended
                 to run the projects first,then save and load runner from the
                 trained weight to BentoML:
                  
                  import bentoml.mlflow
                  
                  run, uri = bentoml.mlflow.load_project(tag)
                  # run the projects
                  # access the model name via UI with `mlflow ui`
                  submitted = run(uri)
                  model_uri = f'runs:/{submitted}/model'
                  
                  # then use BentoML frameworks to save and load the runner
                  tag = bentoml.mlflow.import_from_uri(model_uri)
                  runner = bentoml.mlflow.load_runner(tag)
                """  # noqa
                )
            raise BentoMLException("Invalid MLflow model/projects")


@inject
def load_runner(
    tag: str,
    *,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **runners_kwargs: str,
) -> "Runner":
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
    return _MLflowRunner.to_runner_impl(
        tag=tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
        **runners_kwargs,
    )
