# type: ignore[reportMissingTypeStubs]
import typing as t
import functools
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import cloudpickle
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml.exceptions import MissingDependencyException

from ..models import Model
from ..models import H5_EXT
from ..models import PKL_EXT
from ..models import HDF5_EXT
from ..models import JSON_EXT
from ..models import SAVE_NAMESPACE
from ..utils.tensorflow import get_tf_version
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:  # pragma: no cover
    from tensorflow.python.framework.ops import Graph
    from tensorflow.python.client.session import BaseSession

    from ..models import ModelStore

try:
    import tensorflow as tf

    # TODO: separation of Keras and Tensorflow
    # https://twitter.com/fchollet/status/1404967230048149506?lang=en
    import tensorflow.keras as keras
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        `tensorflow` is required to use `bentoml.keras`, since
        we will use Tensorflow as Keras backend.\n
        Instruction: Refers to https://www.tensorflow.org/install for
        more information of your use case.
        """
    )

from .tensorflow import _TensorflowRunner  # type: ignore[reportPrivateUsage]

_tf_version = get_tf_version()
TF2 = _tf_version.startswith("2")


# Global instance of tf.Session()
_graph: "Graph" = tf.compat.v1.get_default_graph()
_sess: "BaseSession" = tf.compat.v1.Session(graph=_graph)


_CUSTOM_OBJ_FNAME = f"{SAVE_NAMESPACE}_custom_objects{PKL_EXT}"
_SAVED_MODEL_FNAME = f"{SAVE_NAMESPACE}{H5_EXT}"
_MODEL_WEIGHT_FNAME = f"{SAVE_NAMESPACE}_weights{HDF5_EXT}"
_MODEL_JSON_FNAME = f"{SAVE_NAMESPACE}_json{JSON_EXT}"


def get_session() -> "BaseSession":
    """
    Return TF1 sessions for bentoml.keras. Users should
     use this for TF1 in order to run prediction.
    Example::
        session = bentoml.keras.get_session()
        # Initialize variables in the graph/model
        session.run(tf.global_variables_initializer())
        with session.as_default():
            loaded = bentoml.keras.load(tag, model_store=modelstore)
    """  # noqa: LN001
    return _sess


@inject
def load(
    tag: t.Union[str, Tag],
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "keras.Model":
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

    model = model_store.get(tag)
    default_custom_objects = None
    if model.info.options["custom_objects"]:
        assert Path(model.path_of(_CUSTOM_OBJ_FNAME)).is_file()
        with Path(model.path_of(_CUSTOM_OBJ_FNAME)).open("rb") as dcof:
            default_custom_objects = cloudpickle.load(dcof)

    with get_session().as_default():
        if model.info.options["store_as_json"]:
            assert Path(model.path_of(_MODEL_JSON_FNAME)).is_file()
            with Path(model.path_of(_MODEL_JSON_FNAME)).open("r") as jsonf:
                model_json = jsonf.read()
            model = keras.models.model_from_json(
                model_json, custom_objects=default_custom_objects
            )
        else:
            model = keras.models.load_model(
                model.path_of(_SAVED_MODEL_FNAME),
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
    model: "keras.Model",
    *,
    store_as_json: t.Optional[bool] = False,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
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
    context: t.Dict[str, t.Any] = {
        "framework_name": "keras",
        "pip_dependencies": [f"tensorflow=={_tf_version}"],
    }
    options = {
        "store_as_json": store_as_json,
        "custom_objects": True if custom_objects is not None else False,
    }
    _model = Model.create(
        name,
        module=__name__,
        options=options,
        context=context,
        metadata=metadata,
    )

    if custom_objects is not None:
        with Path(_model.path_of(_CUSTOM_OBJ_FNAME)).open("wb") as cof:
            cloudpickle.dump(custom_objects, cof)
    if store_as_json:
        with Path(_model.path_of(_MODEL_JSON_FNAME)).open("w") as jf:
            jf.write(model.to_json())
        model.save_weights(_model.path_of(_MODEL_WEIGHT_FNAME))
    else:
        model.save(_model.path_of(_SAVED_MODEL_FNAME))

    _model.save(model_store)

    return _model.tag


class _KerasRunner(_TensorflowRunner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        device_id: str,
        predict_kwargs: t.Optional[t.Dict[str, t.Any]],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(
            tag=tag,
            predict_fn_name=predict_fn_name,
            device_id=device_id,
            partial_kwargs=predict_kwargs,
            resource_quota=resource_quota,
            batch_options=batch_options,
            model_store=model_store,
        )
        self._session = get_session()

    @property
    def session(self) -> "BaseSession":
        return self._session

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        self._session.config = self._config_proto
        self._model = load(self._tag, model_store=self._model_store)
        raw_predict_fn = getattr(self._model, self._predict_fn_name)
        self._predict_fn = functools.partial(raw_predict_fn, **self._partial_kwargs)

    # pylint: disable=arguments-differ
    def _run_batch(  # type: ignore[override]
        self,
        input_data: t.Union[
            t.List[t.Union[int, float]], "np.ndarray[t.Any, np.dtype[t.Any]]", tf.Tensor
        ],
    ) -> t.Union["np.ndarray[t.Any, np.dtype[t.Any]]", tf.Tensor]:
        if not isinstance(input_data, (np.ndarray, tf.Tensor)):
            input_data = np.array(input_data)
        with tf.device(self._device_id):
            if TF2:
                tf.compat.v1.global_variables_initializer()
            else:
                self._session.run(tf.compat.v1.global_variables_initializer())
                with get_session().as_default():
                    self._predict_fn(input_data)
            return self._predict_fn(input_data)


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "predict",
    device_id: str = "CPU:0",
    predict_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
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
        predict_kwargs (`t.Dict[str, t.Any]`, `optional`, default to `None`):
            Dictionary of `predict()` kwargs that can be shared across different model.
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
    """  # noqa: LN001
    return _KerasRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        device_id=device_id,
        predict_kwargs=predict_kwargs,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
