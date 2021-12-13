import os
import re
import uuid
import typing as t
import logging
import pathlib
import functools
from typing import TYPE_CHECKING
from distutils.dir_util import copy_tree

from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Runner
from bentoml.exceptions import MissingDependencyException

from ..types import PathType
from ..models import Model
from ..runner.utils import Params
from ..utils.tensorflow import get_arg_names
from ..utils.tensorflow import get_tf_version
from ..utils.tensorflow import cast_tensor_by_spec
from ..utils.tensorflow import get_input_signatures
from ..utils.tensorflow import get_restored_functions
from ..utils.tensorflow import pretty_format_restored_model
from ..configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    import numpy as np
    import tensorflow.keras as keras
    from _internal.models import ModelStore

try:
    import tensorflow as tf
    from tensorflow.python.client import device_lib
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
    `tensorflow` is required in order to use `bentoml.tensorflow`.
    Instruction: `pip install tensorflow`
    """
    )

try:
    import tensorflow.python.training.tracking.tracking as tracking
except ImportError:  # pragma: no cover
    import tensorflow_core.python.training.tracking.tracking as tracking

try:
    import importlib.metadata as importlib_metadata
except ImportError:
    import importlib_metadata

_tf_version = get_tf_version()
logger = logging.getLogger(__name__)
TF2 = _tf_version.startswith("2")

try:
    import tensorflow_hub as hub
    from tensorflow_hub import resolve
    from tensorflow_hub import native_module
except ImportError:  # pragma: no cover
    logger.warning(
        """\
    If you want to use `bentoml.tensorflow.import_from_tensorflow_hub(),
     make sure to `pip install --upgrade tensorflow_hub` before using.
     """
    )
    hub = None
    resolve, native_module = None, None

try:
    _tfhub_version = importlib_metadata.version("tensorflow_hub")
except importlib_metadata.PackageNotFoundError:
    _tfhub_version = None


def _clean_name(name: str) -> str:  # pragma: no cover
    return re.sub(r"\W|^(?=\d)-", "_", name)


AUTOTRACKABLE_CALLABLE_WARNING: str = """
Importing SavedModels from TensorFlow 1.x. `outputs = imported(inputs)`
 will not be supported by BentoML due to `tensorflow` API.\n
See https://www.tensorflow.org/api_docs/python/tf/saved_model/load for
 more details.
"""

TF_FUNCTION_WARNING: str = """
Due to TensorFlow's internal mechanism, only methods
 wrapped under `@tf.function` decorator and the Keras default function
 `__call__(inputs, training=False)` can be restored after a save & load.\n
You can test the restored model object via `bentoml.tensorflow.load(path)`
"""

KERAS_MODEL_WARNING: str = """
BentoML detected that {name} is being used to pack a Keras API
 based model. In order to get optimal serving performance, we recommend
 to wrap your keras model `call()` methods with `@tf.function` decorator.
"""


def _is_gpu_available() -> bool:
    try:
        return len(tf.config.list_physical_devices("GPU")) > 0
    except AttributeError:
        return tf.test.is_gpu_available()  # type: ignore


class _tf_function_wrapper:  # pragma: no cover
    def __init__(
        self,
        origin_func: t.Callable[..., t.Any],
        arg_names: t.Optional[t.List[str]] = None,
        arg_specs: t.Optional[t.List[t.Any]] = None,
        kwarg_specs: t.Optional[t.Dict[str, t.Any]] = None,
    ) -> None:
        self.origin_func = origin_func
        self.arg_names = arg_names
        self.arg_specs = arg_specs
        self.kwarg_specs = {k: v for k, v in zip(arg_names or [], arg_specs or [])}
        self.kwarg_specs.update(kwarg_specs or {})

    def __call__(self, *args: str, **kwargs: str) -> t.Any:
        if self.arg_specs is None and self.kwarg_specs is None:
            return self.origin_func(*args, **kwargs)

        for k in kwargs:
            if k not in self.kwarg_specs:
                raise TypeError(f"Function got an unexpected keyword argument {k}")

        arg_keys = {k for k, _ in zip(self.arg_names, args)}  # type: ignore[arg-type]
        _ambiguous_keys = arg_keys & set(kwargs)  # type: t.Set[str]
        if _ambiguous_keys:
            raise TypeError(f"got two values for arguments '{_ambiguous_keys}'")

        # INFO:
        # how signature with kwargs works?
        # https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/eager/function.py#L1519
        transformed_args: t.Tuple[t.Any, ...] = tuple(
            cast_tensor_by_spec(arg, spec) for arg, spec in zip(args, self.arg_specs)  # type: ignore[arg-type] # noqa
        )

        transformed_kwargs = {
            k: cast_tensor_by_spec(arg, self.kwarg_specs[k])
            for k, arg in kwargs.items()
        }
        return self.origin_func(*transformed_args, **transformed_kwargs)

    def __getattr__(self, k: t.Any) -> t.Any:
        return getattr(self.origin_func, k)

    @classmethod
    def hook_loaded_model(cls, loaded_model: t.Any) -> None:
        funcs = get_restored_functions(loaded_model)
        for k, func in funcs.items():
            arg_names = get_arg_names(func)
            sigs = get_input_signatures(func)
            if not sigs:
                continue
            arg_specs, kwarg_specs = sigs[0]
            setattr(
                loaded_model,
                k,
                cls(
                    func,
                    arg_names=arg_names,
                    arg_specs=arg_specs,
                    kwarg_specs=kwarg_specs,
                ),
            )


def _load_tf_saved_model(path: str) -> t.Union["tracking.AutoTrackable", t.Any]:
    if TF2:
        return tf.saved_model.load(path)
    else:
        loaded = tf.compat.v2.saved_model.load(path)
        if isinstance(loaded, tracking.AutoTrackable) and not hasattr(
            loaded, "__call__"
        ):
            logger.warning(AUTOTRACKABLE_CALLABLE_WARNING)
        return loaded


@inject
def load(
    tag: t.Union[str, Tag],
    tfhub_tags: t.Optional[t.List[str]] = None,
    tfhub_options: t.Optional[t.Any] = None,
    load_as_wrapper: t.Optional[bool] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Any:  # returns tf.sessions or keras models
    """
    Load a model from BentoML local modelstore with given name.
    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        tfhub_tags (`str`, `optional`, defaults to `None`):
            A set of strings specifying the graph variant to use, if loading from a v1 module.
        tfhub_options (`tf.saved_model.SaveOptions`, `optional`, default to `None`):
            `tf.saved_model.LoadOptions` object that specifies options for loading. This
             argument can only be used from TensorFlow 2.3 onwards.
        load_as_wrapper (`bool`, `optional`, default to `True`):
            Load the given weight that is saved from tfhub as either `hub.KerasLayer` or `hub.Module`.
             The latter only applies for TF1.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
    Returns:
        an instance of `xgboost.core.Booster` from BentoML modelstore.
    Examples::
    """  # noqa: LN001
    model = model_store.get(tag)
    if model.info.context["import_from_tfhub"]:
        assert load_as_wrapper is not None, (
            "You have to specified `load_as_wrapper=True | False`"
            " to load a `tensorflow_hub` module. If True is chosen,"
            " then BentoML will return either an instance of `hub.KerasLayer`"
            " or `hub.Module` depending on your TF version. For most usecase,"
            " we recommend to keep `load_as_wrapper=True`. If you wish to extend"
            " the functionalities of the given model, set `load_as_wrapper=False`"
            " will return a tf.SavedModel object."
        )
        assert all(
            i is not None for i in [hub, resolve, native_module]
        ), MissingDependencyException(
            "`tensorflow_hub` is required to load a tfhub module."
        )
        module_path = model.path_of(model.info.options["local_path"])
        if load_as_wrapper:
            assert hub is not None
            wrapper_class = hub.KerasLayer if TF2 else hub.Module
            return wrapper_class(module_path)
        # In case users want to load as a SavedModel file object.
        # https://github.com/tensorflow/hub/blob/master/tensorflow_hub/module_v2.py#L93
        is_hub_module_v1 = tf.io.gfile.exists(
            native_module.get_module_proto_path(module_path)
        )
        if tfhub_tags is None and is_hub_module_v1:
            tfhub_tags = []

        if tfhub_options is not None:
            if not isinstance(tfhub_options, tf.saved_model.SaveOptions):
                raise EnvironmentError(
                    f"`tfhub_options` has to be of type `tf.saved_model.SaveOptions`, got {type(tfhub_options)} instead."
                )
            if not hasattr(getattr(tf, "saved_model", None), "LoadOptions"):
                raise NotImplementedError(
                    "options are not supported for TF < 2.3.x,"
                    f" Current version: {_tf_version}"
                )
            # tf.compat.v1.saved_model.load_v2() is TF2 tf.saved_model.load() before TF2
            obj = tf.compat.v1.saved_model.load_v2(
                module_path, tags=tfhub_tags, options=tfhub_options
            )
        else:
            obj = tf.compat.v1.saved_model.load_v2(module_path, tags=tfhub_tags)
        obj._is_hub_module_v1 = (
            is_hub_module_v1  # pylint: disable=protected-access # noqa
        )
        return obj
    else:
        tf_model = _load_tf_saved_model(model.path)
        _tf_function_wrapper.hook_loaded_model(tf_model)
        logger.warning(TF_FUNCTION_WARNING)
        # pretty format loaded model
        logger.info(pretty_format_restored_model(tf_model))
        if hasattr(tf_model, "keras_api"):
            logger.warning(KERAS_MODEL_WARNING.format(name=__name__))
        return tf_model


@inject
def import_from_tfhub(
    identifier: t.Union[str, "hub.Module", "hub.KerasLayer"],
    name: t.Optional[str] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    context: t.Dict[str, t.Any] = {
        "framework_name": "tensorflow",
        "pip_dependencies": [
            f"tensorflow=={_tf_version}",
            f"tensorflow_hub=={_tfhub_version}",
        ],
        "import_from_tfhub": True,
    }
    if name is None:
        if isinstance(identifier, str):
            if identifier.startswith(("http://", "https://")):
                name = _clean_name(identifier.split("/", maxsplit=3)[-1])
            else:
                name = _clean_name(identifier.split("/")[-1])
        else:
            name = f"{identifier.__class__.__name__}_{uuid.uuid4().hex[:5].upper()}"
    _model = Model.create(
        name,
        module=__name__,
        options=None,
        context=context,
        metadata=metadata,
    )
    if isinstance(identifier, str):
        os.environ["TFHUB_CACHE_DIR"] = _model.path
        fpath = resolve(identifier)
        folder = fpath.split("/")[-1]
        _model.info.options = {"model": identifier, "local_path": folder}
    else:
        if hasattr(identifier, "export"):
            # hub.Module.export()
            with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()) as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                identifier.export(_model.path, sess)
        else:
            tf.saved_model.save(identifier, _model.path)
        _model.info.options = {
            "model": identifier.__class__.__name__,
            "local_path": ".",
        }

    _model.save(model_store)
    return _model.tag


@inject
def save(
    name: str,
    model: t.Union["keras.Model", "tf.Module", PathType],
    *,
    signatures: t.Optional[t.Union[t.Callable[..., t.Any], t.Dict[str, t.Any]]] = None,
    options: t.Optional["tf.saved_model.SaveOptions"] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> Tag:
    """
    Save a model instance to BentoML modelstore.
    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`xgboost.core.Booster`):
            Instance of model to be saved
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        signatures (`Union[Callable[..., Any], dict]`, `optional`, default to `None`):
            Refers to `Signatures explanation <https://www.tensorflow.org/api_docs/python/tf/saved_model/save>`_
             from Tensorflow documentation for more information.
        options (`tf.saved_model.SaveOptions`, `optional`, default to `None`):
            :obj:`tf.saved_model.SaveOptions` object that specifies options for saving.
    Raises:
        ValueError: If `obj` is not trackable.
    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.
    Examples::
        import tensorflow as tf
        import bentoml.tensorflow
        tag = bentoml.transformers.save("my_tensorflow_model", model)
    """  # noqa

    context: t.Dict[str, t.Any] = {
        "framework_name": "tensorflow",
        "pip_dependencies": [f"tensorflow=={_tf_version}"],
        "import_from_tfhub": False,
    }
    _model = Model.create(
        name,
        module=__name__,
        options=None,
        context=context,
        metadata=metadata,
    )

    if isinstance(model, (str, bytes, os.PathLike, pathlib.Path)):  # type: ignore[reportUnknownMemberType] # noqa
        assert os.path.isdir(model)
        copy_tree(str(model), _model.path)
    else:
        if TF2:
            tf.saved_model.save(
                model, _model.path, signatures=signatures, options=options
            )
        else:
            if options:
                logger.warning(
                    f"Parameter 'options: {str(options)}' is ignored when "
                    f"using tensorflow {tf.__version__}"
                )
            tf.saved_model.save(model, _model.path, signatures=signatures)

    _model.save(model_store)
    return _model.tag


class _TensorflowRunner(Runner):
    @inject
    def __init__(
        self,
        tag: t.Union[str, Tag],
        predict_fn_name: str,
        device_id: str,
        partial_kwargs: t.Optional[t.Dict[str, t.Any]],
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        in_store_tag = model_store.get(tag).tag
        self._tag = in_store_tag
        super().__init__(str(in_store_tag), resource_quota, batch_options)

        self._device_id = device_id
        self._configure(device_id)
        self._predict_fn_name = predict_fn_name
        assert any(device_id in d.name for d in device_lib.list_local_devices())
        self._partial_kwargs: t.Dict[str, t.Any] = (
            partial_kwargs if partial_kwargs is not None else dict()
        )
        self._model_store = model_store

    def _configure(self, device_id: str) -> None:
        if TF2 and "GPU" in device_id:
            tf.config.set_visible_devices(device_id, "GPU")
        self._config_proto = dict(
            allow_soft_placement=True,
            log_device_placement=False,
            intra_op_parallelism_threads=self.num_concurrency_per_replica,
            inter_op_parallelism_threads=self.num_concurrency_per_replica,
        )

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]

    @property
    def num_concurrency_per_replica(self) -> int:
        if _is_gpu_available() and self.resource_quota.on_gpu:
            return 1
        return int(round(self.resource_quota.cpu))

    @property
    def num_replica(self) -> int:
        if _is_gpu_available() and self.resource_quota.on_gpu:
            return len(tf.config.list_physical_devices("GPU"))
        return 1

    # pylint: disable=attribute-defined-outside-init
    def _setup(self) -> None:
        # setup a global session for model runner
        self._session = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(**self._config_proto)
        )
        self._model = load(self._tag, model_store=self._model_store)
        if not TF2:
            raw_predict_fn = self._model.signatures["serving_default"]
        else:
            raw_predict_fn = getattr(self._model, self._predict_fn_name)
        self._predict_fn = functools.partial(raw_predict_fn, **self._partial_kwargs)

    def _run_batch(
        self,
        *args: t.Union[
            t.List[t.Union[int, float]], "np.ndarray[t.Any, np.dtype[t.Any]]", tf.Tensor
        ],
        **kwargs: t.Union[
            t.List[t.Union[int, float]], "np.ndarray[t.Any, np.dtype[t.Any]]", tf.Tensor
        ],
    ) -> "np.ndarray[t.Any, np.dtype[t.Any]]":
        params = Params[
            t.Union[
                t.List[t.Union[int, float]],
                "np.ndarray[t.Any, np.dtype[t.Any]]",
                tf.Tensor,
            ]
        ](*args, **kwargs)

        with tf.device(self._device_id):

            def _mapping(
                item: t.Union[
                    t.List[t.Union[int, float]],
                    "np.ndarray[t.Any, np.dtype[t.Any]]",
                    tf.Tensor,
                ]
            ) -> tf.Tensor:
                if not isinstance(item, tf.Tensor):
                    return tf.convert_to_tensor(item, dtype=tf.float32)
                else:
                    return item

            params = params.map(_mapping)

            if TF2:
                tf.compat.v1.global_variables_initializer()
                res = self._predict_fn(*params.args, **params.kwargs)
            else:
                self._session.run(tf.compat.v1.global_variables_initializer())
                res = self._session.run(self._predict_fn(*params.args, **params.kwargs))
            return t.cast(
                "np.ndarray[t.Any, np.dtype[t.Any]]",
                res.numpy() if TF2 else res["prediction"],
            )


@inject
def load_runner(
    tag: t.Union[str, Tag],
    *,
    predict_fn_name: str = "__call__",
    device_id: str = "CPU:0",
    partial_kwargs: t.Optional[t.Dict[str, t.Any]] = None,
    resource_quota: t.Union[None, t.Dict[str, t.Any]] = None,
    batch_options: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> "_TensorflowRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.tensorflow.load_runner` implements a Runner class that
    wrap around a Tensorflow model, which optimize it for the BentoML runtime.
    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        predict_fn_name (`str`, default to `__call__`):
            inference function to be used.
        partial_kwargs (`t.Dict[str, t.Any]`, `optional`, default to `None`):
            Dictionary of partial kwargs that can be shared across different model.
        device_id (`t.Union[str, int, t.List[t.Union[str, int]]]`, `optional`, default to `CPU:0`):
            Optional devices to put the given model on. Refers to https://www.tensorflow.org/api_docs/python/tf/config
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
    Returns:
        Runner instances for `bentoml.tensorflow` model
    Examples::
    """  # noqa
    return _TensorflowRunner(
        tag=tag,
        predict_fn_name=predict_fn_name,
        device_id=device_id,
        partial_kwargs=partial_kwargs,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
