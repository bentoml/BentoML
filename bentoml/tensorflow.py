import functools
import logging
import os
import pathlib
import typing as t
from distutils.dir_util import copy_tree

from simple_di import Provide, WrappedCallable
from simple_di import inject as _inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.runner import Runner
from ._internal.types import PathType
from ._internal.utils.tensorflow import (
    cast_tensor_by_spec,
    get_arg_names,
    get_input_signatures,
    get_restored_functions,
    pretty_format_restored_model,
)
from .exceptions import MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    from _internal.models.store import ModelStore

try:
    import tensorflow as tf
    import tensorflow.python.training.tracking.tracking as tracking
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
    `tensorflow` is required in order to use `bentoml.transformers`.
    Instruction: `pip install tensorflow`
    """
    )

TF2 = tf.__version__.startswith("2")

logger = logging.getLogger(__name__)

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

inject: t.Callable[[WrappedCallable], WrappedCallable] = functools.partial(
    _inject, squeeze_none=False
)


class _tf_function_wrapper:
    def __init__(
        self,
        origin_func: t.Callable[..., t.Any],
        arg_names: t.Optional[list] = None,
        arg_specs: t.Optional[list] = None,
        kwarg_specs: t.Optional[dict] = None,
    ) -> None:
        self.origin_func = origin_func
        self.arg_names = arg_names
        self.arg_specs = arg_specs
        self.kwarg_specs = {k: v for k, v in zip(arg_names or [], arg_specs or [])}
        self.kwarg_specs.update(kwarg_specs or {})

    def __call__(self, *args, **kwargs):
        if self.arg_specs is None and self.kwarg_specs is None:
            return self.origin_func(*args, **kwargs)

        for k in kwargs:
            if k not in self.kwarg_specs:
                raise TypeError(f"Function got an unexpected keyword argument {k}")

        arg_keys = {k for k, _ in zip(self.arg_names, args)}
        _ambiguous_keys = arg_keys & set(kwargs)
        if _ambiguous_keys:
            raise TypeError(f"got two values for arguments '{_ambiguous_keys}'")

        # INFO:
        # how signature with kwargs works?
        # https://github.com/tensorflow/tensorflow/blob/v2.0.0/tensorflow/python/eager/function.py#L1519

        transformed_args = tuple(
            cast_tensor_by_spec(arg, spec) for arg, spec in zip(args, self.arg_specs)
        )

        transformed_kwargs = {
            k: cast_tensor_by_spec(arg, self.kwarg_specs[k])
            for k, arg in kwargs.items()
        }
        return self.origin_func(*transformed_args, **transformed_kwargs)

    def __getattr__(self, k):
        return getattr(self.origin_func, k)

    @classmethod
    def hook_loaded_model(cls, loaded_model) -> None:  # noqa
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
    tag: str,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Any:  # returns tf.sessions or keras models
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
    model = _load_tf_saved_model(str(model_info.path))
    _tf_function_wrapper.hook_loaded_model(model)
    logger.warning(TF_FUNCTION_WARNING)
    # pretty format loaded model
    logger.info(pretty_format_restored_model(model))
    if hasattr(model, "keras_api"):
        logger.warning(KERAS_MODEL_WARNING.format(name=__name__))
    return model


@inject
def save(
    name: str,
    model: t.Union["tf.keras.Model", "tf.Module", PathType, pathlib.PurePath],
    *,
    signatures: t.Optional[
        t.Union[t.Callable[[t.Any], t.Any], t.Dict[str, t.Any]]
    ] = None,
    options: t.Optional["tf.saved_model.SaveOptions"] = None,
    metadata: t.Union[None, t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> str:
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
    """  # noqa
    context = {"tensorflow": tf.__version__}
    with model_store.register(
        name,
        module=__name__,
        options=None,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        if not isinstance(model, (str, bytes, pathlib.PurePath, os.PathLike)):
            if TF2:
                tf.saved_model.save(
                    model, str(ctx.path), signatures=signatures, options=options
                )
            else:
                if options:
                    logger.warning(
                        f"Parameter 'options: {str(options)}' is ignored when "
                        f"using tensorflow {tf.__version__}"
                    )
                tf.saved_model.save(model, str(ctx.path), signatures=signatures)
        else:
            assert os.path.isdir(model)
            copy_tree(str(model), str(ctx.path))


class _TensorflowRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        resource_quota: t.Dict[str, t.Any],
        batch_options: t.Dict[str, t.Any],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)

    @property
    def required_models(self) -> t.List[str]:
        ...

    @property
    def num_concurrency_per_replica(self) -> int:
        ...

    @property
    def num_replica(self) -> int:
        ...

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:
        ...

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _run_batch(self, input_data) -> t.Any:
        ...


@inject
def load_runner(
    tag: str,
    *,
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
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
    )
