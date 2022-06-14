from __future__ import annotations

import typing as t
import logging
from typing import TYPE_CHECKING

import numpy as np

import bentoml
from bentoml import Tag
from bentoml.exceptions import NotFound
from bentoml.exceptions import InvalidArgument
from bentoml.exceptions import MissingDependencyException
from bentoml._internal.models.model import ModelContext

from ..utils.pkg import get_pkg_version

if TYPE_CHECKING:
    from bentoml.types import ModelSignature
    from bentoml.types import ModelSignatureDict

    from .. import external_typing as ext

try:
    import lightgbm as lgb  # type: ignore[reportMissingTypeStubs]
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """lightgbm is required in order to use module `bentoml.lightgbm`, install
        lightgbm with `pip install lightgbm`. For more information, refer to
        https://github.com/microsoft/LightGBM/tree/master/python-package
        """
    )

MODULE_NAME = "bentoml.lightgbm"
MODEL_FILENAME = "saved_model.ubj"
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
        model = bentoml.lightgbm.get("my_lightgbm_model")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    return model


def load_model(bento_model: str | Tag | bentoml.Model) -> lgb.basic.Booster:
    """
    Load the LightGNM model with the given tag from the local BentoML model store.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.
    Returns:
        :obj:`~lightgbm.basic.Booster`: The LightGBM model loaded from the model store or BentoML :obj:`~bentoml.Model`.

    Example:

    .. code-block:: python

        import bentoml
        gbm = bentoml.lightgbm.load("my_lightgbm_model:latest")
    """  # noqa
    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)
        assert isinstance(bento_model, bentoml.Model)

    model_file = bento_model.path_of(MODEL_FILENAME)
    booster = lgb.basic.Booster(model_file=model_file)
    return booster


def save_model(
    name: str,
    model: lgb.basic.Booster,
    *,
    signatures: dict[str, ModelSignatureDict] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> Tag:
    """
    Save a LightGBM model instance to the BentoML model store.

    Args:
        name (``str``):
            The name to give to the model in the BentoML store. This must be a valid
            :obj:`~bentoml.Tag` name.
        model (:obj:`~lgb.basic.Booster`):
            The LightGBM model (booster) to be saved.
        signatures (``dict[str, ModelSignatureDict]``, optional):
            Signatures of predict methods to be used. If not provided, the signatures default to
            ``{"predict": {"batchable": False}}``. See :obj:`~bentoml.types.ModelSignature` for more
            details.
        labels (``dict[str, str]``, optional):
            A default set of management labels to be associated with the model. An example is
            ``{"training-set": "data-1"}``.
        custom_objects (``dict[str, Any]``, optional):
            Custom objects to be saved with the model. An example is
            ``{"my-normalizer": normalizer}``.

            Custom objects are currently serialized with cloudpickle, but this implementation is
            subject to change.
        metadata (``dict[str, Any]``, optional):
            Metadata to be associated with the model. An example is ``{"max_depth": 2}``.

            Metadata is intended for display in model management UI and therefore must be a default
            Python type, such as ``str`` or ``int``.
    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the
        user-defined model's name, and a generated `version` by BentoML.

    Example:

    .. code-block:: python

        import bentoml

        import lightgbm as lgb
        import pandas as pd

        # load a dataset
        df_train = pd.read_csv("regression.train", header=None, sep="\t")
        df_test = pd.read_csv("regression.test", header=None, sep="\t")

        y_train = df_train[0]
        y_test = df_test[0]
        X_train = df_train.drop(0, axis=1)
        X_test = df_test.drop(0, axis=1)

        # create dataset for lightgbm
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        # specify your configurations as a dict
        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": {"l2", "l1"},
            "num_leaves": 31,
            "learning_rate": 0.05,
        }

        # train
        gbm = lgb.train(
            params, lgb_train, num_boost_round=20, valid_sets=lgb_eval
        )

        # `save` the booster to BentoML modelstore:
        tag = bentoml.lightgbm.save("my_lightgbm_model", gbm, booster_params=params)
    """  # noqa: LN001

    # Ensure that `model` is actually the Booster object, and not for example one of the scikit-learn wrapper objects.
    if not isinstance(model, lgb.basic.Booster):
        # Deal with a LightGBM issue (https://github.com/microsoft/LightGBM/issues/3014)
        if not hasattr(model, "fitted_"):
            model.fitted_ = True

        try:
            model = model.booster_
        except AttributeError as e:
            logger.error(
                "Unable to obtain a lightgbm.basic.Booster from the specified model."
            )
            raise e

    assert isinstance(
        model, lgb.basic.Booster
    ), "Unable to obtain a lightgbm.basic.Booster from the specified model."

    context: ModelContext = ModelContext(
        framework_name="lightgbm",
        framework_versions={"lightgbm": get_pkg_version("lightgbm")},
    )

    if signatures is None:
        logger.info(
            'Using default model signature `{"predict": {"batchable": False}}` for LightGBM model'
        )
        signatures = {
            "predict": {"batchable": False},
        }

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
    ) as bento_model:
        model.save_model(bento_model.path_of(MODEL_FILENAME))

        return bento_model.tag


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class LightGBMRunnable(bentoml.Runnable):
        SUPPORT_NVIDIA_GPU = False  # afaik LightGBM only supports GPU during training, not for inference.
        SUPPORT_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            self.model = load_model(bento_model)

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                try:
                    self.predict_fns[method_name] = getattr(self.model, method_name)
                except AttributeError:
                    raise InvalidArgument(
                        f"No method with name {method_name} found for LightGBM model of type {self.model.__class__}"
                    )

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: LightGBMRunnable,
            input_data: ext.NpNDArray | ext.PdDataFrame,
        ) -> ext.NpNDArray:
            res = self.predict_fns[method_name](input_data)
            return np.asarray(res)

        LightGBMRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return LightGBMRunnable
