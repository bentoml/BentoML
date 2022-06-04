from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import bentoml
from bentoml import Tag
from bentoml.models import Model
from bentoml.models import ModelContext
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..utils.pkg import get_pkg_version

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline

    from bentoml.types import ModelSignature
    from bentoml._internal.models.model import ModelSignaturesType

    from .. import external_typing as ext

    SklearnModel: t.TypeAlias = BaseEstimator | Pipeline


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
MODEL_FILENAME = "saved_model.pkl"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> Model:
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    return model


def load_model(
    bento_model: str | Tag | Model,
) -> SklearnModel:
    """
    Load the scikit-learn model with the given tag from the local BentoML model store.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.
        ...
    Returns:
        ``BaseEstimator`` ``|`` ``Pipeline``:
            The scikit-learn model loaded from the model store or BentoML :obj:`~bentoml.Model`.
    Example:
    .. code-block:: python
        import bentoml
        sklearn = bentoml.sklearn.load_model('my_model:latest')
    """  # noqa
    if not isinstance(bento_model, Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, failed loading with {MODULE_NAME}."
        )
    model_file = bento_model.path_of(MODEL_FILENAME)

    return joblib.load(model_file)


def save_model(
    name: str,
    model: SklearnModel,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: t.Dict[str, str] | None = None,
    custom_objects: t.Dict[str, t.Any] | None = None,
    metadata: t.Dict[str, t.Any] | None = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[BaseEstimator, Pipeline]`):
            Instance of model to be saved.
        signatures (:code: `Dict[str, ModelSignatureDict]`)
            Methods to expose for running inference on the target model. Signatures are
             used for creating Runner instances when serving model with bentoml.Service
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
             e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is
        the user-defined model's name, and a generated `version`.

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

        tag = bentoml.sklearn.save_model('kneighbors', model)

        # load the model back:
        loaded = bentoml.sklearn.load_model("kneighbors:latest")
        # or:
        loaded = bentoml.sklearn.load_model(tag)
    """  # noqa
    context = ModelContext(
        framework_name="sklearn",
        framework_versions={"scikit-learn": get_pkg_version("scikit-learn")},
    )

    if signatures is None:
        signatures = {"predict": {"batchable": False}}
        logger.info(
            f"Using the default model signature for sklearn ({signatures}) for model {name}."
        )

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
        signatures=signatures,
    ) as bento_model:

        joblib.dump(model, bento_model.path_of(MODEL_FILENAME))

        return bento_model.tag


def get_runnable(bento_model: Model):
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class SklearnRunnable(bentoml.Runnable):
        SUPPORT_NVIDIA_GPU = False  # type: ignore
        SUPPORT_CPU_MULTI_THREADING = True  # type: ignore

        def __init__(self):
            super().__init__()
            self.model = load_model(bento_model)

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: SklearnRunnable, input_data: ext.NpNDArray | ext.PdDataFrame
        ) -> ext.NpNDArray:
            # TODO: set inner_max_num_threads and n_jobs param here base on strategy env vars
            with parallel_backend(backend="loky"):
                return getattr(self.model, method_name)(input_data)

        SklearnRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return SklearnRunnable
