import functools
import typing as t
from pathlib import Path

import cloudpickle
import numpy as np
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.models import H5_EXT, HDF5_EXT, JSON_EXT, PKL_EXT, SAVE_NAMESPACE
from .exceptions import MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    from _internal.models.store import ModelStore
    from mypy.typeshed.stdlib.contextlib import _GeneratorContextManager  # noqa
    from tensorflow.python.client.session import BaseSession
    from tensorflow.python.framework.ops import Graph

try:
    import tensorflow as tf

    # TODO(aarnphm): separation of Keras and Tensorflow
    # https://twitter.com/fchollet/status/1404967230048149506?lang=en
    import tensorflow.keras as tfk
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        `tensorflow` is required to use `bentoml.keras`, since
        we will use Tensorflow as Keras backend.\n
        Instruction: Refers to https://www.tensorflow.org/install for
        more information of your use case.
        """
    )

from bentoml.tensorflow import _TensorflowRunner

TF2 = tf.__version__.startswith("2")

# Global instance of tf.Session()
_graph: "Graph" = tf.compat.v1.get_default_graph()
_sess: "BaseSession" = tf.compat.v1.Session(graph=_graph)

# make sess.as_default() type safe
_default_sess: t.Callable[
    [], "_GeneratorContextManager[BaseSession]"
] = functools.partial(_sess.as_default)

_load: t.Callable[[t.BinaryIO], "tfk.Model"] = functools.partial(cloudpickle.load)

_CUSTOM_OBJ_FNAME = f"{SAVE_NAMESPACE}_custom_objects{PKL_EXT}"
_SAVED_MODEL_FNAME = f"{SAVE_NAMESPACE}{H5_EXT}"
_MODEL_WEIGHT_FNAME = f"{SAVE_NAMESPACE}_weights{HDF5_EXT}"
_MODEL_JSON_FNAME = f"{SAVE_NAMESPACE}_json{JSON_EXT}"


@inject
def load(
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "tfk.Model":
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        an instance of `xgboost.core.Booster` from BentoML modelstore.

    Examples::
    """  # noqa

    model_info = model_store.get(tag)
    default_custom_objects = None
    if model_info.options["custom_objects"]:
        assert Path(model_info.path, _CUSTOM_OBJ_FNAME).is_file()
        with Path(model_info.path, _CUSTOM_OBJ_FNAME).open("rb") as dcof:
            default_custom_objects = _load(dcof)

    with _default_sess():
        if model_info.options["store_as_json"]:
            assert Path(model_info.path, _MODEL_JSON_FNAME).is_file()
            with Path(model_info.path, _MODEL_JSON_FNAME).open("r") as jsonf:
                model_json = jsonf.read()
            model = tfk.models.model_from_json(
                model_json, custom_objects=default_custom_objects
            )
        else:
            model = tfk.models.load_model(
                str(Path(model_info.path, _SAVED_MODEL_FNAME)),
                custom_objects=default_custom_objects,
            )
        try:
            # if model is a dictionary
            return model["model"]
        except TypeError:
            return model


@inject
def save(
    name: str,
    model: "tfk.Model",
    *,
    store_as_json: t.Optional[bool] = False,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`tensorflow.keras.Model`):
            Instance of model to be saved
        store_as_json (`bool`, `optional`, default to `False`):
            Whether to store Keras model as JSON and weights
        custom_objects (`GenericDictType`, `optional`, default to `None`):
            Dictionary of Keras custom objects for mod
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
    """  # noqa
    tf.compat.v1.keras.backend.get_session()
    context = {"tensorflow": tf.__version__}
    options = {
        "store_as_json": store_as_json,
        "custom_objects": True if custom_objects is not None else False,
    }
    with model_store.register(
        name,
        module=__name__,
        options=options,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        if custom_objects is not None:
            with Path(ctx.path, _CUSTOM_OBJ_FNAME).open("wb") as cof:
                cloudpickle.dump(custom_objects, cof)
        if store_as_json:
            with Path(ctx.path, _MODEL_JSON_FNAME).open("w") as jf:
                jf.write(model.to_json())
            model.save_weights(str(Path(ctx.path, _MODEL_WEIGHT_FNAME)))
        else:
            model.save(str(Path(ctx.path, _SAVED_MODEL_FNAME)))
        return ctx.tag  # type: ignore


class _KerasRunner(_TensorflowRunner):
    @inject
    def __init__(
        self,
        tag: str,
        predict_fn_name: str,
        device_id: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(
            tag=tag,
            predict_fn_name=predict_fn_name,
            device_id=device_id,
            resource_quota=resource_quota,
            batch_options=batch_options,
            model_store=model_store,
        )
        self._session = _sess

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:  # type: ignore[override] # noqa
        self._session.config = self._config_proto
        try:
            self._model = load(self.name, model_store=self._model_store)
        except FileNotFoundError:
            if self._from_mlflow:
                # a special flags to determine whether the runner is
                # loaded from mlflow
                import bentoml.mlflow

                self._model = bentoml.mlflow.load(
                    self.name, model_store=self._model_store
                )
        self._predict_fn = getattr(self._model, self._predict_fn_name)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[override]
        self, input_data: t.Union[t.List[t.Union[int, float]], np.ndarray, tf.Tensor]
    ) -> t.Union[np.ndarray, tf.Tensor]:
        if not isinstance(input_data, (np.ndarray, tf.Tensor)):
            input_data = np.array(input_data)
        with self._device:
            if TF2:
                tf.compat.v1.global_variables_initializer()
            else:
                self._session.run(tf.compat.v1.global_variables_initializer())
                with _default_sess():
                    self._predict_fn(input_data)
            return self._predict_fn(input_data)


@inject
def load_runner(
    tag: str,
    *,
    predict_fn_name: str = "predict",
    device_id: str = "CPU:0",
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_KerasRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.tensorflow.load_runner` implements a Runner class that
    wrap around a Keras model, with Tensorflow as backend, which optimize it for the BentoML runtime.

    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `predict`):
            inference function to be used.
        device_id (`t.Union[str, int, t.List[t.Union[str, int]]]`, `optional`, default to `CPU:0`):
            Optional devices to put the given model on. Refers to https://www.tensorflow.org/api_docs/python/tf/config
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.keras` model

    Examples::
    """  # noqa
    return _KerasRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        device_id=device_id,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
