import os
import typing as t
import importlib
import importlib.util
from typing import TYPE_CHECKING
from pathlib import Path

import yaml
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import Model as BentoModel
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:

    import mlflow.pyfunc
    from mlflow.pyfunc import PyFuncModel

    from ..models import ModelStore


try:
    import mlflow
    from mlflow.models import Model
    from mlflow.models.model import MLMODEL_FILE_NAME
    from mlflow.tracking.artifact_utils import _download_artifact_from_uri
except ImportError:
    raise MissingDependencyException(
        """\
        `mlflow` is required to use with `bentoml.mlflow`.
        Instruction: `pip install -U mlflow`
        """
    )

_mlflow_version = get_pkg_version("mlflow")


def _strike(text: str) -> str:
    return "".join(["\u0336{}".format(c) for c in text])


def _validate_file_exists(fname: str, parent: str) -> t.Tuple[bool, str]:
    for path in Path(parent).iterdir():
        if path.is_dir():
            return _validate_file_exists(fname, str(path))
        if path.name == fname:
            return True, str(path)
    return False, ""


@inject
def load(
    tag: t.Union[str, Tag],
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
    model = model_store.get(tag)
    mlflow_folder = model.path_of(model.info.options["mlflow_folder"])
    mlmodel_fpath = Path(mlflow_folder, MLMODEL_FILE_NAME)
    if not mlmodel_fpath.exists():
        raise BentoMLException(f"{MLMODEL_FILE_NAME} cannot be found.")
    flavors = Model.load(mlmodel_fpath).flavors  # pragma: no cover
    module = list(flavors.values())[0]["loader_module"]
    loader_module = importlib.import_module(module)
    return loader_module.load_model(mlflow_folder)  # noqa


def save(*args: str, **kwargs: str) -> None:  # noqa # pylint: disable
    raise EnvironmentError(
        f"""\n
BentoML won't provide a `save` API for MLflow.
If you currently working with `mlflow.<flavor>.save_model`, we kindly suggest you
 to replace `mlflow.<flavor>.save_model` with BentoML's `save` API as we also supports
 the majority of ML frameworks that MLflow supports. An example below shows how you can
 migrate your code for PyTorch from MLflow to BentoML::

    - {_strike('import mlflow.pytorch')}
    + import bentoml.pytorch

    # PyTorch model logics
    ...
    model = EmbeddingBag()
    - {_strike('mlflow.pytorch.save_model(model, "embed_bag")')}
    + bentoml.pytorch.save("embed_bag", model)

If you want to import MLflow models from local directory or a given file path, you can
 utilize `bentoml.mlflow.import_from_uri`::

    import mlflow.pytorch
    + import bentoml.mlflow

    path = "./my_pytorch_model"
    mlflow.pytorch.save_model(model, path)
    + tag = bentoml.mlflow.import_from_uri("mlflow_pytorch_model", path)

If your current workflow with MLflow involve `log_model` as well as importing models from MLflow Registry,
 you can import those directly to BentoML modelstore using `bentoml.mlflow.import_from_uri`. We also accept
 MLflow runs syntax, as well as models registry uri.
An example showing how to integrate your current `log_model` with `mlflow.sklearn` to BentoML::

    import mlflow.sklearn
    + import mlflow
    + import bentoml.mlflow

    # Log sklearn model `sk_learn_rfr` and register as version 1
    ...
    reg_name = "sk-learn-random-forest-reg-model"
    artifact_path = "sklearn_model"
    mlflow.sklearn.log_model(
        sk_model=sk_learn_rfr,
        artifact_path=artifact_path,
        registered_model_name=reg_name
    )

    # refers to https://www.mlflow.org/docs/latest/tracking.html#logging-functions
    + current_run = mlflow.active_run().info.run_id
    + uri = "runs:/%s/%s" % (current_run, artifact_path)
    + tag = bentoml.mlflow.import_from_uri("runs_mlflow_sklearn", uri)

An example showing how to import from MLflow models registry to BentoML modelstore. With this usecase, we
 recommend you to load the model into memory first with `mlflow.<flavor>.load_model` then save the model using
 BentoML `save` API::

    import mlflow.sklearn
    + import bentoml.sklearn

    reg_model_name = "sk-learn-random-forest-reg-model"
    model_uri = "models:/%s/1" % reg_model_name
    + loaded = mlflow.sklearn.load_model(model_uri, *args, **kwargs)
    + tag = bentoml.sklearn.save("my_model", loaded, *args, **kwargs)

    # you can also use `bentoml.mlflow.import_from_uri` to import the model directly after
    #  defining `model_uri`
    + import bentoml.mlflow
    + tag = bentoml.mlflow.import_from_uri("my_model", model_uri)
    """  # noqa
    )


@inject
def import_from_uri(
    name: str,
    uri: str,
    *,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    context: t.Dict[str, t.Any] = {
        "framework_name": "mlflow",
        "pip_dependencies": [f"mlflow=={_mlflow_version}"],
    }

    _model = BentoModel.create(
        name,
        module=__name__,
        options=None,
        context=context,
        metadata=metadata,
    )

    _model.info.options = {"uri": uri}
    mlflow_obj_path = _download_artifact_from_uri(uri, output_path=_model.path)
    _model.info.options["mlflow_folder"] = os.path.relpath(mlflow_obj_path, _model.path)
    exists, _ = _validate_file_exists("MLproject", mlflow_obj_path)
    if exists:
        raise BentoMLException("BentoML doesn't accept MLflow Projects.")
    exists, fpath = _validate_file_exists("MLmodel", mlflow_obj_path)
    if not exists:
        raise BentoMLException("Downloaded path is not a valid MLflow Model.")
    with Path(fpath).open("r") as mf:
        conf = yaml.safe_load(mf.read())
        flavor = conf["flavors"]["python_function"]
        _model.info.options["flavor"] = flavor["loader_module"]

    _model.save(model_store)

    return _model.tag


class _PyFuncRunner(Runner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(str(tag), resource_quota, batch_options)
        self._model_store = model_store
        self._model_info = self._model_store.get(tag)

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_info.tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        path = self._model_info.info.options["mlflow_folder"]
        artifact_path = self._model_info.path_of(path)
        self._model = mlflow.pyfunc.load_model(artifact_path, suppress_warnings=False)

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: t.Any) -> t.Any:  # type: ignore[override]
        return self._model.predict(input_data)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_PyFuncRunner":
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
    return _PyFuncRunner(
        tag,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
