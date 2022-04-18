import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import PKL_EXT
from ..models import SAVE_NAMESPACE
from ..utils.pkg import get_pkg_version
from .common.model_runner import BaseModelRunner
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline

    from .. import external_typing as ext
    from ..models import ModelStore

try:
    import joblib
    from joblib import parallel_backend
except ImportError:  # pragma: no cover
    try:
        from sklearn.utils._joblib import joblib
        from sklearn.utils._joblib import parallel_backend
    except ImportError:
        raise MissingDependencyException(
            """sklearn is required in order to use the module `bentoml.sklearn`, install
             sklearn with `pip install sklearn`. For more information, refer to
             https://scikit-learn.org/stable/install.html
            """
        )

MODULE_NAME = "bentoml.sklearn"


@inject
def load(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Union["BaseEstimator", "Pipeline"]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`Union[BaseEstimator, Pipeline]`: an instance of :obj:`sklearn` model from BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml

        sklearn = bentoml.sklearn.load('my_model:latest')
    """
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    model_file = model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}")

    return joblib.load(model_file)


def save(
    name: str,
    model: t.Union["BaseEstimator", "Pipeline"],
    *,
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[BaseEstimator, Pipeline]`):
            Instance of model to be saved.
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        import bentoml

        from sklearn.datasets import load_iris
        from sklearn.neighbors import KNeighborsClassifier

        model = KNeighborsClassifier()
        iris = load_iris()
        X = iris.data[:, :4]
        Y = iris.target
        model.fit(X, Y)

        tag = bentoml.sklearn.save('kneighbors', model)

        # load the model back:
        loaded = bentoml.sklearn.load("kneighbors:latest")
        # or:
        loaded = bentoml.sklearn.load(tag)
    """  # noqa
    context = {
        "framework_name": "sklearn",
        "pip_dependencies": [f"scikit-learn=={get_pkg_version('scikit-learn')}"],
    }

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
    ) as _model:

        joblib.dump(model, _model.path_of(f"{SAVE_NAMESPACE}{PKL_EXT}"))

        return _model.tag


class _SklearnRunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        function_name: str,
        name: t.Optional[str] = None,
    ):
        super().__init__(tag, name=name)
        self._backend = "loky"
        self._function_name = function_name

    @property
    def _num_threads(self) -> int:
        return max(round(self.resource_quota.cpu), 1)

    @property
    def num_replica(self) -> int:
        return 1

    def _setup(self) -> None:
        self._model = load(self._tag, model_store=self.model_store)
        self._infer_func = getattr(self._model, self._function_name)

    def _run_batch(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        inputs: t.Union["ext.NpNDArray", "ext.PdDataFrame"],
    ) -> "ext.NpNDArray":
        with parallel_backend(self._backend, n_jobs=self._num_threads):
            return self._infer_func(inputs)


def load_runner(
    tag: t.Union[str, Tag],
    function_name: str = "predict",
    *,
    name: t.Optional[str] = None,
) -> "_SklearnRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`bentoml.sklearn.load_runner` implements a Runner class that
    wrap around a Sklearn joblib model, which optimize it for the BentoML runtime.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore..
        function_name (:code:`str`, `optional`, default to :code:`predict`):
            Predict function used by a given sklearn model.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for the target :mod:`bentoml.sklearn` model

    Examples:

    .. code-block:: python

        import bentoml

        runner = bentoml.sklearn.load_runner("my_model:latest")
        runner.run([[1,2,3,4]])
    """
    return _SklearnRunner(
        tag=tag,
        function_name=function_name,
        name=name,
    )
