from __future__ import annotations

import os
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
    import xgboost as xgb
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """xgboost is required in order to use module `bentoml.xgboost`, install
        xgboost with `pip install xgboost`. For more information, refers to
        https://xgboost.readthedocs.io/en/latest/install.html
        """
    )

MODULE_NAME = "bentoml.xgboost"
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
        model = bentoml.xgboost.get("my_xgboost_model")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )
    return model


def load_model(bento_model: str | Tag | bentoml.Model) -> xgb.core.Booster:
    """
    Load the XGBoost model with the given tag from the local BentoML model store.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.
    Returns:
        :obj:`~xgboost.core.Booster`: The XGBoost model loaded from the model store or BentoML :obj:`~bentoml.Model`.
    Example:

    .. code-block:: python

        import bentoml
        # target model must be from the BentoML model store
        booster = bentoml.xgboost.load_model("my_xgboost_model")
    """  # noqa: LN001
    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)
        assert isinstance(bento_model, bentoml.Model)

    model_file = bento_model.path_of(MODEL_FILENAME)
    booster = xgb.core.Booster(model_file=model_file)
    return booster


def save_model(
    name: str,
    model: xgb.core.Booster,
    *,
    signatures: dict[str, ModelSignatureDict] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> Tag:
    """
    Save an XGBoost model instance to the BentoML model store.

    Args:
        name (``str``):
            The name to give to the model in the BentoML store. This must be a valid
            :obj:`~bentoml.Tag` name.
        model (:obj:`~xgboost.core.Booster`):
            The XGBoost model to be saved.
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

        import xgboost as xgb
        import bentoml

        # read in data
        dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
        dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
        # specify parameters via map
        param = dict(max_depth=2, eta=1, objective='binary:logistic')
        num_round = 2
        bst = xgb.train(param, dtrain, num_round)
        ...

        # `save` the booster to BentoML modelstore:
        tag = bentoml.xgboost.save_model("my_xgboost_model", bst, booster_params=param)
    """  # noqa: LN001
    context: ModelContext = ModelContext(
        framework_name="xgboost",
        framework_versions={"xgboost": get_pkg_version("xgboost")},
    )

    if signatures is None:
        logger.info(
            'Using default model signature `{"predict": {"batchable": False}}` for XGBoost model'
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
        model.save_model(bento_model.path_of(MODEL_FILENAME))  # type: ignore (incomplete XGBoost types)

        return bento_model.tag


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class XGBoostRunnable(bentoml.Runnable):
        SUPPORT_NVIDIA_GPU = True
        SUPPORT_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            self.model = load_model(bento_model)

            # check for resources
            available_gpus = os.getenv("NVIDIA_VISIBLE_DEVICES")
            if available_gpus is not None and available_gpus != "":
                self.model.set_param({"predictor": "gpu_predictor", "gpu_id": 0})  # type: ignore (incomplete XGBoost types)
            else:
                nthreads = os.getenv("OMP_NUM_THREADS")
                if nthreads is not None and nthreads != "":
                    nthreads = max(int(nthreads), 1)
                else:
                    nthreads = 1
                self.model.set_param({"predictor": "cpu_predictor", "nthread": nthreads})  # type: ignore (incomplete XGBoost types)

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                try:
                    self.predict_fns[method_name] = getattr(self.model, method_name)
                except AttributeError:
                    raise InvalidArgument(
                        f"No method with name {method_name} found for XGBoost model of type {self.model.__class__}"
                    )

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(
            self: XGBoostRunnable,
            input_data: ext.NpNDArray
            | ext.PdDataFrame,  # TODO: add support for DMatrix
        ) -> ext.NpNDArray:
            dmatrix = xgb.DMatrix(input_data)

            res = self.predict_fns[method_name](dmatrix)
            return np.asarray(res)  # type: ignore (incomplete np types)

        XGBoostRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return XGBoostRunnable
