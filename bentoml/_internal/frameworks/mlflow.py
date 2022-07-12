from __future__ import annotations

import os
import shutil
import typing as t
import logging
from typing import TYPE_CHECKING

import bentoml
from bentoml import Tag
from bentoml.models import ModelContext
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException

from ..utils import LazyLoader

if TYPE_CHECKING:
    from bentoml.types import ModelSignature
    from bentoml.types import ModelSignatureDict


if TYPE_CHECKING:
    import mlflow
else:
    mlflow = LazyLoader(
        "mlflow",
        globals(),
        "mlflow",
        exc_msg="`mlflow` is required in order to use module `bentoml.mlflow`, install "
        "mlflow with `pip install mlflow`. For more information, refer to "
        "https://mlflow.org/",
    )


MODULE_NAME = "bentoml.mlflow"
MLFLOW_MODEL_FOLDER = "mlflow_model"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> bentoml.Model:
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def load_model(
    bento_model: str | Tag | bentoml.Model,
) -> mlflow.pyfunc.PyFuncModel:
    """
    Load the MLFlow model with the given tag from the local BentoML model store.

    Args:
        bento_model: Either the tag of the model to get from the store, or a BentoML
            ``~bentoml.Model`` instance to load the model from.

    Returns:
        The MLFlow model loaded as PyFuncModel from the BentoML model store.

    Example:

    .. code-block:: python

        import bentoml
        pyfunc_model = bentoml.mlflow.load_model('my_model:latest')
        pyfunc_model.predict( input_df )
    """  # noqa
    if not isinstance(bento_model, bentoml.Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    return mlflow.pyfunc.load_model(bento_model.path_of(MLFLOW_MODEL_FOLDER))


def import_model(
    name: str,
    model_uri: str,
    *,
    signatures: dict[str, ModelSignatureDict | ModelSignature] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    metadata: dict[str, t.Any] | None = None,
    # ...
) -> bentoml.Model:
    """
    import mlflow model from a URI to the BentoML model store.

    Args:
        name:
            The name to give to the model in the BentoML store. This must be a valid
            :obj:`~bentoml.Tag` name.
        model_uri:
            The <FRAMEWORK> model to be saved.
        signatures:
            Signatures of predict methods to be used. If not provided, the signatures
            default to {"predict": {"batchable": False}}. See
            :obj:`~bentoml.types.ModelSignature` for more details.
        labels:
            A default set of management labels to be associated with the model. For
            example: ``{"training-set": "data-v1"}``.
        custom_objects:
            Custom objects to be saved with the model. An example is
            ``{"my-normalizer": normalizer}``. Custom objects are serialized with
            cloudpickle.
        metadata:
            Metadata to be associated with the model. An example is ``{"param_a": .2}``.

            Metadata is intended for display in a model management UI and therefore all
            values in metadata dictionary must be a primitive Python type, such as
            ``str`` or ``int``.

    Returns:
        A :obj:`~bentoml.Model` instance referencing a saved model in the local BentoML
        model store.

    Example:

    .. code-block:: python

        import bentoml

        bentoml.mlflow.import_model(
            'my_mlflow_model',
            model_uri="runs:/<mlflow_run_id>/run-relative/path/to/model",
            signatures={
                "batchable": True
            }
        )
    """
    context = ModelContext(
        framework_name="mlflow",
        framework_versions={"mlflow": mlflow.__version__},
    )

    if signatures is None:
        signatures = {
            "predict": {"batchable": False},
        }
        logger.info(
            f"Using the default model signature for MLFlow pyfunc model ({signatures}) for model {name}."
        )
    if len(signatures.keys()) != 1 or "predict" not in signatures:
        raise BentoMLException(
            f"MLFlow pyfunc model support only the `predict` method, signatures={signatures} is not supported"
        )

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        options=None,
        custom_objects=custom_objects,
        metadata=metadata,
        context=context,
    ) as bento_model:
        from mlflow.models import Model as MLflowModel
        from mlflow.pyfunc import FLAVOR_NAME as PYFUNC_FLAVOR_NAME
        from mlflow.artifacts import download_artifacts
        from mlflow.models.model import MLMODEL_FILE_NAME

        local_path = download_artifacts(
            artifact_uri=model_uri, dst_path=bento_model.path
        )
        mlflow_model_path = bento_model.path_of(MLFLOW_MODEL_FOLDER)
        shutil.move(local_path, mlflow_model_path)
        mlflow_model_file = os.path.join(mlflow_model_path, MLMODEL_FILE_NAME)

        if not os.path.exists(mlflow_model_file):
            raise BentoMLException("artifact is not a mlflow model")

        model_meta = MLflowModel.load(mlflow_model_file)
        if PYFUNC_FLAVOR_NAME not in model_meta.flavors:
            raise BentoMLException(
                "Target MLFlow model does not support python_function flavor"
            )

        return bento_model


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """
    assert "predict" in bento_model.info.signatures.keys()
    predict_signature = bento_model.info.signatures["predict"]

    class MLflowPyfuncRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True  # type: ignore

        def __init__(self):
            super().__init__()
            self.model = load_model(bento_model)

        @bentoml.Runnable.method(
            batchable=predict_signature.batchable,
            batch_dim=predict_signature.batch_dim,
            input_spec=None,
            output_spec=None,
        )
        def predict(self, input_data):
            return self.model.predict(input_data)

    return MLflowPyfuncRunnable


def get_mlflow_model(tag_like: str | Tag) -> mlflow.models.Model:
    bento_model = get(tag_like)
    return mlflow.models.Model.load(bento_model.path_of(MLFLOW_MODEL_FOLDER))
