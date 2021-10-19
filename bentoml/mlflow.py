import importlib
import importlib.util
import logging
import os
import typing as t
from distutils.dir_util import copy_tree
from pathlib import Path, PurePath
from urllib.parse import urlparse

import yaml
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import SAVE_NAMESPACE
from ._internal.runner import Runner
from .exceptions import BentoMLException, InvalidArgument, MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import

    from _internal.models.store import ModelStore
    from mlflow.pyfunc import PyFuncModel

try:
    import mlflow
    from mlflow.models import Model
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.tracking.artifact_utils import _download_artifact_from_uri
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        `mlflow` is required to use with `bentoml.mlflow`.
        Instruction: `pip install -U mlflow`
        """
    )


logger = logging.getLogger(__name__)

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


def _is_s3_url(uri: str) -> bool:
    return urlparse(uri).scheme.startswith("s3")


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
    model_info = model_store.get(tag)
    if model_info.context["import_from_uri"]:
        model_folder = model_info.options["uri"].split("/")[-1]
        if "mlproject_path" in model_info.options:
            raise BentoMLException(
                """\
            BentoML doesn't support loading MLflow Projects.
            """
            )
        mlmodel_fpath = Path(model_info.path, model_folder, MLMODEL_FILE_NAME)
        if not mlmodel_fpath.exists():
            raise BentoMLException(f"{MLMODEL_FILE_NAME} cannot be found.")
        model = Model.load(mlmodel_fpath)  # pragma: no cover
        # fmt: off
        pyfunc_config = list(map(lambda x: x[1], model.flavors.items()))[0]  # type: ignore[no-any-return] # noqa # pylint: disable
        loader_module = importlib.import_module(str(pyfunc_config['loader_module']))
        # fmt: on
    else:
        model_folder = SAVE_NAMESPACE
        loader_module = mlflow.pyfunc
    return loader_module.load_model(os.path.join(model_info.path, model_folder))


def _save(
    name: str,
    identifier: t.Union["mlflow.models.Model", str],
    loader_module: t.Optional[t.Type["mlflow.pyfunc"]],
    metadata: t.Optional[t.Dict[str, t.Any]],
    model_store: "ModelStore",
) -> str:
    context = {"mlflow": mlflow.__version__, "import_from_uri": False}
    if isinstance(identifier, str):
        context["import_from_uri"] = True
    else:
        assert loader_module is not None, (
            "`loader_module` is required"
            " in order to save given model"
            " from MLflow."
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
            ctx.options = {"uri": Path(identifier).as_uri(), "flavor": "na"}
            mlflow_obj_path = _download_artifact_from_uri(
                identifier, output_path=ctx.path
            )
            try:
                for f in Path(mlflow_obj_path).glob(f"**/{_MLFLOW_PROJECT_FILENAME}"):
                    ctx.options["mlproject_path"] = str(f.parent)
                if "mlproject_path" not in ctx.options:
                    raise FileNotFoundError
            except FileNotFoundError:
                try:
                    _mlmodel_path = None
                    for f in Path(mlflow_obj_path).iterdir():
                        if f.is_dir():
                            if any(f for f in f.glob("MLmodel")):
                                _mlmodel_path = [i for i in f.rglob("MLmodel")][0]
                                break
                        elif f.resolve().name == "MLmodel":
                            _mlmodel_path = f
                            break
                    if _mlmodel_path is not None:
                        with _mlmodel_path.open("r") as mf:
                            conf = yaml.safe_load(mf.read())
                            # fmt: off
                            ctx.options["flavor"] = conf["flavors"]["python_function"]["loader_module"]  # noqa # pylint: disable
                            # fmt: on
                except (NotADirectoryError, NameError):
                    pass
            if (
                ctx.options["flavor"] == "na" and "mlproject_path" not in ctx.options
            ):  # pragma: no cover
                logger.warning(
                    f"`{identifier.split('/')[-1]}` is neither MLflow Projects"
                    " nor MLflow Models. This will result in errors when loading"
                    f" {ctx.tag}. If you to import your saved weight from S3 refers"
                    f" to correct framework of choice for its S3 related-functions."
                )
        else:
            assert loader_module is not None, "`loader_module` cannot be None."
            ctx.options["flavor"] = loader_module.__name__
            loader_module.save_model(identifier, os.path.join(ctx.path, SAVE_NAMESPACE))
        return ctx.tag  # type: ignore


@inject
def save(
    name: str,
    model: "mlflow.models.Model",
    loader_module: t.Optional[t.Type["mlflow.pyfunc"]] = None,
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
    return _save(
        name=name,
        identifier=model,
        loader_module=loader_module,
        metadata=metadata,
        model_store=model_store,
    )


@inject
def import_from_uri(
    name: str,
    uri: str,
    loader_module: t.Optional[t.Type["mlflow.pyfunc"]] = None,
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
    return _save(
        name=name,
        identifier=uri,
        loader_module=loader_module,
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
        logger.warning(_PYFUNCRUNNER_WARNING.format(name=tag))
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
            self._predict_fn = getattr(self._model, "predict")
        except FileNotFoundError:  # pragma: no cover
            # this is included in tests, but didn't get covered.
            # TODO(aarnphm): investigate
            for f in Path(self._model_info.path).iterdir():
                if f.is_dir():
                    assert "MLmodel" in [
                        _.name for _ in f.iterdir()
                    ], _NOT_MLFLOW_MODEL_ERR_MSG.format(path=f)
                    self._model = mlflow.pyfunc.load_model(str(f))
                file = f.resolve()  # file could be symlink
                if file.name == "MLmodel":
                    self._model = mlflow.pyfunc.load_model(str(f.parent))
                self._predict_fn = getattr(self._model, "predict")
            raise BentoMLException(
                "Unable to load a valid MLmodel during `_setup()`. Make sure"
                f" {self.name} follows MLflow Models format."
            )

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: t.Any) -> t.Any:  # type: ignore[override]
        return self._predict_fn(input_data)


class _MLflowRunner:
    def __init__(self, *args: str, **kwargs: str):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to initialized"
            " BentoML internal runners implementation correspondingly via"
            f" `{self.__class__.__name__}.to_runner_impl(tag, resource_quota, batch_options, **kwargs)`"  # noqa # pylint: disable
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
        **runner_kwargs: str,
    ) -> "Runner":
        model_info = model_store.get(tag)
        try:
            _, *flavors = model_info.options["flavor"].split(".")
            flavor = flavors[0]
            if flavor == "pyfunc":
                return _PyFuncRunner(
                    tag=tag,
                    resource_quota=resource_quota,
                    batch_options=batch_options,
                    model_store=model_store,
                )
            runner = getattr(importlib.import_module(f"bentoml.{flavor}"), "load_runner")(  # type: ignore # noqa # pylint: disable
                tag,
                *runners_args,
                resource_quota=resource_quota,
                batch_options=batch_options,
                model_store=model_store,
                **runner_kwargs,
            )
            setattr(runner, "_from_mlflow", True)
            return runner
        except (IndexError, ImportError):
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
            raise BentoMLException(
                "Invalid MLflow Models. This is likely to be"
                " BentoML internal errors when BentoML failed"
                " to detect MLflow flavors."
            )


@inject
def load_runner(
    tag: str,
    *runner_args: str,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **runner_kwargs: str,
) -> "Runner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.mlflow.load_runner` implements a Runner class that
    has to options to wrap around a PyFuncModel, or infer from given MLflow flavor to
    load BentoML internal Runner implementation, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        runner_args (sequence of positional arguments, `optional`):
            All remaining positional arguments applied for different runners. Refers to
             the Runner for the framework you are currently using for more information.
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        runner_kwargs (remaining dictionary of keyword arguments, `optional`):
            Can be used for the remaining of the runner kwargs options. Refers to BentoML
             frameworks runners for more information.

    Returns:
        Runner instances loaded from `bentoml.mlflow`. This can be either a `mlflow.pyfunc.PyFuncModel`
        Runner or BentoML runner instance that maps to MLflow flavors

    Examples::
    """  # noqa
    return _MLflowRunner.to_runner_impl(
        tag,
        *runner_args,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
        **runner_kwargs,
    )
