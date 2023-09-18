from __future__ import annotations

import os
import shutil
import typing as t
import logging
import tempfile
from typing import TYPE_CHECKING

import bentoml
from bentoml import Tag
from bentoml.models import ModelContext
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

if TYPE_CHECKING:
    from types import ModuleType

    from bentoml.types import ModelSignature
    from bentoml.types import ModelSignatureDict


try:
    import mlflow
    import mlflow.models
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'mlflow' is required in order to use module 'bentoml.mlflow', install mlflow with 'pip install mlflow'. For more information, refer to https://mlflow.org/",
    )


MODULE_NAME = "bentoml.mlflow"
MLFLOW_MODEL_FOLDER = "mlflow_model"
API_VERSION = "v1"

logger = logging.getLogger(__name__)


def get(tag_like: str | Tag) -> bentoml.Model:
    """
    Get the BentoML model with the given tag.

    Args:
        tag_like: The tag of the model to retrieve from the model store.

    Returns:
        :obj:`~bentoml.Model`: A BentoML :obj:`~bentoml.Model` with the matching tag.

    Example:

    .. code-block:: python

       import bentoml
       # target model must be from the BentoML model store
       model = bentoml.mlflow.get("my_mlflow_model")
    """
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
    Load the MLflow `PyFunc <https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel>`_ model with the given tag from the local BentoML model store.

    Args:
        bento_model: Either the tag of the model to get from the store, or a BentoML
            ``~bentoml.Model`` instance to load the model from.

    Returns:
        The MLflow model loaded as PyFuncModel from the BentoML model store.

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
    name: Tag | str,
    model_uri: str,
    *,
    signatures: dict[str, ModelSignature] | dict[str, ModelSignatureDict] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
    # ...
) -> bentoml.Model:
    """
    Import MLflow model from a artifact URI to the BentoML model store.

    Args:
        name:
            The name to give to the model in the BentoML store. This must be a valid
            :obj:`~bentoml.Tag` name.
        model_uri:
            The MLflow model to be saved.
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
                "predict": {"batchable": True},
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
            'Using the default model signature for MLflow (%s) for model "%s".',
            signatures,
            name,
        )
    if len(signatures) != 1 or "predict" not in signatures:
        raise BentoMLException(
            f"MLflow pyfunc model support only the `predict` method, signatures={signatures} is not supported"
        )

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        signatures=signatures,
        labels=labels,
        options=None,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
        context=context,
    ) as bento_model:
        from mlflow.models import Model as MLflowModel
        from mlflow.pyfunc import FLAVOR_NAME as PYFUNC_FLAVOR_NAME
        from mlflow.models.model import MLMODEL_FILE_NAME

        # Explicitly provide a destination dir to mlflow so that we don't
        # accidentially download into the root of the bento model temp dir
        # (using a model:/ url can cause this)
        download_dir = tempfile.mkdtemp(dir=bento_model.path)

        try:
            # Prefer public API download_artifacts introduced in MLflow 1.25
            from mlflow.artifacts import download_artifacts

            local_path = download_artifacts(
                artifact_uri=model_uri, dst_path=download_dir
            )
        except (ModuleNotFoundError, ImportError):
            # For MLflow < 1.25
            from mlflow.tracking.artifact_utils import _download_artifact_from_uri

            local_path: str = _download_artifact_from_uri(
                artifact_uri=model_uri, output_path=download_dir
            )
        finally:
            mlflow_model_path = bento_model.path_of(MLFLOW_MODEL_FOLDER)
            # Rename model folder from original artifact name to fixed "mlflow_model"
            shutil.move(local_path, mlflow_model_path)  # type: ignore (local_path is bound)
            # Remove the tempdir if it still exists.
            # NOTE for models:/ uri downloads, the download_dir itself is actually renamed
            # in the previous line, not a subdir of download_dir like other methods.
            # Calling rmtree unchecked will lead to models:/ downloads failing
            if os.path.exists(download_dir):
                shutil.rmtree(download_dir)

        mlflow_model_file = os.path.join(mlflow_model_path, MLMODEL_FILE_NAME)

        if not os.path.exists(mlflow_model_file):
            raise BentoMLException(f'artifact "{model_uri}" is not a MLflow model')

        model_meta = MLflowModel.load(mlflow_model_file)
        if PYFUNC_FLAVOR_NAME not in model_meta.flavors:
            raise BentoMLException(
                f'MLflow model "{model_uri}" does not support the required python_function flavor'
            )

        return bento_model


def get_runnable(bento_model: bentoml.Model) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """
    assert "predict" in bento_model.info.signatures
    predict_signature = bento_model.info.signatures["predict"]

    class MLflowPyfuncRunnable(bentoml.Runnable):
        # The only case that multi-threading may not be supported is when user define a
        # custom python_function MLflow model with pure python code, but there's no way
        # of telling that from the MLflow model metadata. It should be a very rare case,
        # because most custom python_function models are likely numpy code or model
        # inference with pre/post-processing code.
        SUPPORTED_RESOURCES = ("cpu",)
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()
            self.model = load_model(bento_model)

        @bentoml.Runnable.method(
            batchable=predict_signature.batchable,
            batch_dim=predict_signature.batch_dim,
            input_spec=None,
            output_spec=None,
        )
        def predict(self, input_data: t.Any) -> t.Any:
            return self.model.predict(input_data)

    return MLflowPyfuncRunnable


def get_mlflow_model(tag_like: str | Tag) -> mlflow.models.Model:
    bento_model = get(tag_like)
    return mlflow.models.Model.load(bento_model.path_of(MLFLOW_MODEL_FOLDER))
