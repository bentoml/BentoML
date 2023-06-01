from __future__ import annotations

import os
import typing as t
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import attr
import numpy as np

import bentoml
from bentoml import Tag
from bentoml.models import ModelOptions
from bentoml.exceptions import NotFound
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..utils.pkg import get_pkg_version
from ..models.model import ModelContext

if TYPE_CHECKING:
    from bentoml.types import ModelSignature
    from bentoml.types import ModelSignatureDict

    from .. import external_typing as ext

try:
    import catboost as cb
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'catboost' is required in order to use module 'bentoml.catboost', install catboost with 'pip install catboost'. For more information, refer to https://catboost.ai/en/docs/concepts/installation."
    )

MODULE_NAME = "bentoml.catboost"
MODEL_FILENAME = "saved_model.cbm"
DEFAULT_MODEL_TRAINING_CLASS_NAME = "CatBoost"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> bentoml.Model:
    """
    Get the BentoML model with the given tag.

    Args:
        tag_like (``str`` ``|`` :obj:`~bentoml.Tag`):
            The tag of the model to retrieve from the model store.

    Returns:
        :obj:`~bentoml.Model`: A BentoML :obj:`~bentoml.Model` with the matching tag.

    Example:

    .. code-block:: python

        import bentoml
        # target model must be from the BentoML model store
        model = bentoml.catboost.get("my_catboost_model")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(bento_model: str | Tag | bentoml.Model) -> cb.CatBoost:
    """
    Load the CatBoost model with the given tag from the local BentoML model store.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.

    Returns:
        :obj:`~catboost.CatBoost`: The CatBoost model loaded from the model store or BentoML :obj:`~bentoml.Model`.

    Example:

    .. code-block:: python

        import bentoml
        # target model must be from the BentoML model store
        booster = bentoml.catboost.load_model("my_catboost_model")
    """  # noqa: LN001
    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    model_file = bento_model.path_of(MODEL_FILENAME)
    cb_class_name: str = bento_model.info.options.training_class_name  # type: ignore
    cb_class: t.Type[cb.CatBoost] = getattr(cb, cb_class_name)
    if not issubclass(cb_class, cb.CatBoost):
        raise BentoMLException(f"{cb_class_name} is not a valid CatBoost class.")
    cb_instance = cb_class()
    booster: cb.CatBoost = cb_instance.load_model(fname=model_file)
    return booster


@attr.define
class CatBoostOptions(ModelOptions):
    training_class_name: str = attr.field(factory=str)


def save_model(
    name: Tag | str,
    model: cb.CatBoost,
    *,
    signatures: dict[str, ModelSignatureDict] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save an CatBoost model instance to the BentoML model store.

    Args:
        name:
            The name to give to the model in the BentoML store. This must be a valid
            :obj:`~bentoml.Tag` name.
        model:
            The CatBoost model to be saved.
        signatures:
            Signatures of predict methods to be used. If not provided, the signatures default to
            ``{"predict": {"batchable": False}}``. See :obj:`~bentoml.types.ModelSignature` for more
            details.
        labels:
            A default set of management labels to be associated with the model. An example is
            ``{"training-set": "data-1"}``.
        custom_objects:
            Custom objects to be saved with the model. An example is
            ``{"my-normalizer": normalizer}``.

            Custom objects are currently serialized with cloudpickle, but this implementation is
            subject to change.
        external_modules (:code:`List[ModuleType]`, `optional`, default to :code:`None`):
            user-defined additional python modules to be saved alongside the model or custom objects,
            e.g. a tokenizer module, preprocessor module, model configuration module
        metadata:
            Metadata to be associated with the model. An example is ``{"max_depth": 2}``.

            Metadata is intended for display in model management UI and therefore must be a default
            Python type, such as ``str`` or ``int``.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the
        user-defined model's name, and a generated `version` by BentoML.

    Example:

    .. code-block:: python

        import bentoml
        import numpy as np

        from catboost import CatBoostClassifier, Pool

        # initialize data
        train_data = np.random.randint(0, 100, size=(100, 10))

        train_labels = np.random.randint(0, 2, size=(100))

        test_data = catboost_pool = Pool(train_data, train_labels)

        model = CatBoostClassifier(iterations=2,
                                   depth=2,
                                   learning_rate=1,
                                   loss_function='Logloss',
                                   verbose=True)
        # train the model
        model.fit(train_data, train_labels)

        # save the model to the BentoML model store
        bento_model = bentoml.catboost.save_model("my_catboost_model", model)
    """
    if not isinstance(model, cb.CatBoost):
        raise TypeError(f"Given model ({model}) is not a catboost.CatBoost.")

    context: ModelContext = ModelContext(
        framework_name="catboost",
        framework_versions={"catboost": get_pkg_version("catboost")},
    )

    if signatures is None:
        signatures = {
            "predict": {"batchable": False},
        }
        logger.info(
            'Using the default model signature for CatBoost (%s) for model "%s".',
            signatures,
            name,
        )

    options = CatBoostOptions(
        training_class_name=model.__class__.__name__,
    )

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
        options=options,
    ) as bento_model:
        model.save_model(bento_model.path_of(MODEL_FILENAME))  # type: ignore (incomplete CatBoost types)

        return bento_model


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class CatBoostRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        predict_params: t.Dict[str, t.Any]

        def __init__(self):
            super().__init__()
            self.model = load_model(bento_model)

            self.predict_params = {
                "task_type": "CPU",
            }

            # check for resources
            available_gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if available_gpus not in ("", "-1"):
                self.predict_params["task_type"] = "GPU"
            else:
                nthreads = os.getenv("OMP_NUM_THREADS")
                if nthreads is not None and nthreads != "":
                    nthreads = max(int(nthreads), 1)
                else:
                    nthreads = -1
                self.predict_params["thread_count"] = nthreads

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                try:
                    self.predict_fns[method_name] = getattr(self.model, method_name)
                except AttributeError:
                    raise InvalidArgument(
                        f"No method with name {method_name} found for CatBoost model of type {self.model.__class__}"
                    )

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: CatBoostRunnable,
            input_data: ext.NpNDArray | ext.CbPool | ext.PdDataFrame,
        ) -> ext.NpNDArray:
            if not isinstance(input_data, cb.Pool):
                input_data = cb.Pool(input_data)
            res = self.predict_fns[method_name](input_data, **self.predict_params)
            return np.asarray(res)  # type: ignore (incomplete np types)

        CatBoostRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return CatBoostRunnable
