import os
import typing as t
from typing import TYPE_CHECKING
from pathlib import Path

import yaml
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model as BentoModel
from bentoml import Runner
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException
from bentoml._internal.models import SAVE_NAMESPACE

from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from mlflow.pyfunc import PyFuncModel

    from ..models import ModelStore


try:
    import mlflow.pyfunc
    from mlflow.tracking.artifact_utils import _download_artifact_from_uri
except ImportError:
    raise MissingDependencyException(
        """\
        `mlflow` is required to use with `bentoml.mlflow`.
        Instruction: `pip install -U mlflow`
        """
    )

MODULE_NAME = "bentoml.mlflow"


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
    tag: Tag,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "PyFuncModel":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `mlflow.pyfunc.PyFuncModel` from BentoML modelstore.

    Examples:

    .. code-block:: python

    """  # noqa
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    mlflow_folder = model.path_of(model.info.options["mlflow_folder"])
    return mlflow.pyfunc.load_model(mlflow_folder, suppress_warnings=False)


SAVE_WARNING = f"""\
BentoML won't provide a :func:`save` API for MLflow.
If you currently working with :code:`mlflow.<flavor>.save_model`, we kindly suggest you
to replace :code:`mlflow.<flavor>.save_model` with BentoML's `save` API as we also supports
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
utilize :code:`bentoml.mlflow.import_from_uri`::

    import mlflow.pytorch
    + import bentoml.mlflow

    path = "./my_pytorch_model"
    mlflow.pytorch.save_model(model, path)
    + tag = bentoml.mlflow.import_from_uri("mlflow_pytorch_model", path)

If your current workflow with MLflow involve :func:`log_model` as well as importing models from MLflow Registry,
you can import those directly to BentoML modelstore using :code:`bentoml.mlflow.import_from_uri`. We also accept
MLflow runs syntax, as well as models registry uri.
An example showing how to integrate your current :func:`log_model` with :code:`mlflow.sklearn` to BentoML::

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
recommend you to load the model into memory first with :code:`mlflow.<flavor>.load_model` then save the model using
BentoML :func:`save` API::

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


def save(*args: str, **kwargs: str) -> None:  # noqa # pylint: disable
    raise EnvironmentError(SAVE_WARNING)


save.__doc__ = SAVE_WARNING


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
        "pip_dependencies": [f"mlflow=={get_pkg_version('mlflow')}"],
    }

    _model = BentoModel.create(
        name,
        module=MODULE_NAME,
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
        tag: Tag,
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(name, resource_quota, batch_options)
        self._model_store = model_store
        self._model_tag = tag

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._model_tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override]
        self._model = load(self._model_tag, model_store=self._model_store)

    # pylint: disable=arguments-differ
    def _run_batch(self, input_data: t.Any) -> t.Any:  # type: ignore[override]
        return self._model.predict(input_data)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    name: t.Optional[str] = None,
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
        tag (`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        resource_quota (`Dict[str, Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`Dict[str, Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances loaded from `bentoml.mlflow`. This can be either a `mlflow.pyfunc.PyFuncModel`
        Runner or BentoML runner instance that maps to MLflow flavors

    Examples:

    .. code-block:: python

    """  # noqa
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _PyFuncRunner(
        tag,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
