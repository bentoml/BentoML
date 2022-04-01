import os
import typing as t
from typing import TYPE_CHECKING
from pathlib import Path

import yaml
from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..utils.pkg import get_pkg_version
from .common.model_runner import BaseModelRunner
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
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "PyFuncModel":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`mlflow.pyfunc.PyFuncModel`: an instance of `mlflow.pyfunc.PyFuncModel` from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        model = bentoml.mlflow.load("mlflow_sklearn_model")

    """  # noqa
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    mlflow_folder = model.path_of(model.info.options["mlflow_folder"])
    return mlflow.pyfunc.load_model(mlflow_folder, suppress_warnings=False)  # type: ignore


SAVE_WARNING = f"""\
BentoML won't provide a :func:`save` API for MLflow. If one uses :code:`bentoml.mlflow.save`, it will
raises :obj:`BentoMLException`:

    - If you currently working with :code:`mlflow.<flavor>.save_model`, we kindly suggest you
      to replace :code:`mlflow.<flavor>.save_model` with BentoML's `save` API as we also supports
      the majority of ML frameworks that MLflow supports. An example below shows how you can
      migrate your code for PyTorch from MLflow to BentoML:

    .. code-block:: diff

        - {_strike('import mlflow.pytorch')}
        + import bentoml

        # PyTorch model logics
        ...
        model = EmbeddingBag()
        - {_strike('mlflow.pytorch.save_model(model, "embed_bag")')}
        + bentoml.pytorch.save("embed_bag", model)

    - If you want to import MLflow models from local directory or a given file path, you can
      utilize :code:`bentoml.mlflow.import_from_uri`:

    .. code-block:: diff

        import mlflow.pytorch
        + import bentoml

        path = "./my_pytorch_model"
        mlflow.pytorch.save_model(model, path)
        + tag = bentoml.mlflow.import_from_uri("mlflow_pytorch_model", path)

    - If your current workflow with MLflow involve :func:`log_model` as well as importing models from MLflow Registry,
      you can import those directly to BentoML modelstore using :code:`bentoml.mlflow.import_from_uri`. We also accept
      MLflow runs syntax, as well as models registry uri.
      An example showing how to integrate your current :func:`log_model` with :code:`mlflow.sklearn` to BentoML:

    .. code-block:: diff

        import mlflow.sklearn
        + import mlflow
        + import bentoml

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
BentoML :func:`save` API:

.. code-block:: diff

    import mlflow.sklearn
    + import bentoml

    reg_model_name = "sk-learn-random-forest-reg-model"
    model_uri = "models:/%s/1" % reg_model_name
    + loaded = mlflow.sklearn.load_model(model_uri, *args, **kwargs)
    + tag = bentoml.sklearn.save("my_model", loaded, *args, **kwargs)

    # you can also use `bentoml.mlflow.import_from_uri` to import the model directly after
    #  defining `model_uri`
    + import bentoml
    + tag = bentoml.mlflow.import_from_uri("my_model", model_uri)

"""  # noqa


def save(*args: str, **kwargs: str) -> None:  # noqa # pylint: disable
    raise BentoMLException(SAVE_WARNING)


save.__doc__ = SAVE_WARNING


def import_from_uri(
    name: str,
    uri: str,
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Imports a MLFlow model format to BentoML modelstore via given URI.

    Args:
        name (:code:`str`):
            Name for your MLFlow model to be saved under BentoML modelstore.
        uri (:code:`str`): URI accepts all MLflow defined APIs in terms of referencing artifacts.
            All available accepted URI (extracted from `MLFlow Concept <https://mlflow.org/docs/latest/concepts.html>`_):
                - :code:`/Users/me/path/to/local/model`
                - :code:`relative/path/to/local/model`
                - :code:`s3://my_bucket/path/to/model`
                - :code:`hdfs://<host>:<port>/<path>`
                - :code:`runs:/<mlflow_run_id>/run-relative/path/to/model`
                - :code:`models:/<model_name>/<model_version>`
                - :code:`models:/<model_name>/<stage>`
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`~bentoml.Tag` object that can be used to retrieve the model with :func:`bentoml.tensorflow.load`:

    Example:

    .. code-block:: python

        from sklearn import svm, datasets

        import mlflow
        import bentoml

        # Load training data
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        # Model Training
        clf = svm.SVC()
        clf.fit(X, y)

        # Wrap up as a custom pyfunc model
        class ModelPyfunc(mlflow.pyfunc.PythonModel):

            def load_context(self, context):
                self.model = clf

            def predict(self, context, model_input):
                return self.model.predict(model_input)

        # Log model
        with mlflow.start_run() as run:
            model = ModelPyfunc()
            mlflow.pyfunc.log_model("model", python_model=model)
            print("run_id: {}".format(run.info.run_id))

        model_uri = f"runs:/{run.info.run_id}/model"

        # import given model from local uri to BentoML modelstore:
        tag = bentoml.mlflow.import_from_uri("model", model_uri)

        # from MLFlow Models API
        model_tag = bentoml.mlflow.import_from_uri("mymodel", "models:/mymodel/1")
    """  # noqa
    context: t.Dict[str, t.Any] = {
        "framework_name": "mlflow",
        "pip_dependencies": [f"mlflow=={get_pkg_version('mlflow')}"],
    }

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        custom_objects=custom_objects,
        options=None,
        context=context,
        metadata=metadata,
    ) as _model:

        _model.info.options = {"uri": uri}
        mlflow_obj_path = _download_artifact_from_uri(uri, output_path=_model.path)
        _model.info.options["mlflow_folder"] = os.path.relpath(
            mlflow_obj_path, _model.path
        )
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

        return _model.tag


class _PyFuncRunner(BaseModelRunner):
    @property
    def num_replica(self) -> int:
        return 1

    def _setup(self) -> None:
        self._model = load(self._tag, self.model_store)

    def _run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._model.predict(*args, **kwargs)  # type: ignore


@inject
def load_runner(
    tag: t.Union[str, Tag],
    name: t.Optional[str] = None,
) -> "_PyFuncRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.mlflow.load_runner` implements a Runner class that
    has to options to wrap around a PyFuncModel, or infer from given MLflow flavor to
    load BentoML internal Runner implementation, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.

    Returns:
        :obj:`bentoml._internal.runner.Runner`: Runner instances loaded from `bentoml.mlflow`.

    .. note::


       Currently this is an instance of :obj:`~bentoml._internal.frameworks.mlflow._PyFuncRunner` which is the base
       runner. We recommend users to use the correspond frameworks' Runner implementation provided by BentoML for the best performance.

    On our roadmap, the intention for this API is to load the coresponding framework runner automatically. For example:

    .. code-block:: python

        import bentoml

        # this tag `mlflow_pytorch` is imported from mlflow
        # when user use `bentoml.mlflow.load_runner`, it should returns
        # bentoml._internal.frameworks.pytorch.PyTorchRunner` instead
        # and kwargs can be passed directly to `PyTorchRunner`
        runner = bentoml.mlflow.load_runner("mlflow_pytorch", **kwargs)

    Examples:

    .. code-block:: python

        import bentoml

        runner = bentoml.mlflow.load_runner(tag)
        runner.run_batch([[1,2,3,4,5]])

    """
    return _PyFuncRunner(
        tag,
        name=name,
    )
