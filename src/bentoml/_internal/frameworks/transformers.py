from __future__ import annotations

import os
import typing as t
import logging
from types import ModuleType
from functools import lru_cache

import attr

import bentoml

from ..tag import Tag
from ..types import LazyType
from ..utils import LazyLoader
from ..utils.pkg import find_spec
from ..utils.pkg import get_pkg_version
from ..utils.pkg import pkg_version_info
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..models.model import Model
from ..models.model import ModelContext
from ..models.model import ModelOptions
from ..configuration import DEBUG_ENV_VAR
from ..configuration import get_debug_mode

if t.TYPE_CHECKING:
    import cloudpickle

    from bentoml.types import ModelSignature

    from ..models.model import ModelSignaturesType
    from ..external_typing import transformers as ext

    class SimpleDefaultMapping(t.TypedDict, total=False):
        pt: tuple[str, ...]
        tf: tuple[str, ...]

    class ModelDefaultMapping(t.TypedDict):
        model: SimpleDefaultMapping

    DefaultMapping = (
        ModelDefaultMapping
        | SimpleDefaultMapping
        | dict[tuple[str, ...], ModelDefaultMapping]
    )

    TupleStr = tuple[str, ...]
    TupleAutoModel = tuple[ext.BaseAutoModelClass, ...]
else:
    TupleStr = TupleAutoModel = tuple
    DefaultMapping = SimpleDefaultMapping = ModelDefaultMapping = dict

    cloudpickle = LazyLoader(
        "cloudpickle",
        globals(),
        "cloudpickle",
        exc_msg="'cloudpickle' is required to save/load custom pipeline.",
    )


try:
    import transformers
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "'transformers' is required in order to use module 'bentoml.transformers'. Install transformers with 'pip install transformers'."
    )


__all__ = ["load_model", "save_model", "get_runnable", "get"]

_object_setattr = object.__setattr__

MODULE_NAME = "bentoml.transformers"
API_VERSION = "v1"
PIPELINE_PICKLE_NAME = f"pipeline.{API_VERSION}.pkl"


logger = logging.getLogger(__name__)

TRANSFORMERS_VERSION = pkg_version_info("transformers")

HAS_PIPELINE_REGISTRY = TRANSFORMERS_VERSION >= (4, 21, 0)


@lru_cache(maxsize=1)
def _check_flax_supported() -> None:  # pragma: no cover
    _supported = TRANSFORMERS_VERSION[0] >= 4

    if not _supported:
        logger.warning(
            "Detected transformers version: %s.%s.%s, which doesn't have supports for Flax. Update 'transformers' to 4.x and above to have Flax supported.",
            *TRANSFORMERS_VERSION,
        )
    else:
        _flax_available = find_spec("jax") is not None and find_spec("flax") is not None
        if _flax_available:
            _jax_version = get_pkg_version("jax")
            _flax_version = get_pkg_version("flax")
            logger.info(
                "Jax version %s, Flax version %s available.",
                _jax_version,
                _flax_version,
            )
        else:
            logger.warning(
                "No versions of Flax or Jax found on the current machine. In order to use Flax with transformers 4.x and above, refer to https://github.com/google/flax#quick-install"
            )


def _deep_convert_to_tuple(
    dct: dict[str, str | TupleStr | list[str] | dict[str, t.Any]]
) -> dict[str, str | TupleStr | list[str] | dict[str, t.Any]]:
    for k, v in dct.items():
        override = v
        if isinstance(v, list):
            override = tuple(v)
        elif isinstance(v, dict):
            override = _deep_convert_to_tuple(v)
        dct[k] = override
    return dct


def _validate_pipeline_type(
    _: t.Any, attribute: attr.Attribute[str | list[str]], value: str | list[str]
) -> None:
    """
    Validate the type of the given pipeline definition. The value is expected to be a `str`.
    `list` type is also allowed here to maintain compatibility with an earlier introduced bug.

    TODO: disallow list type in the next minor version release.
    """
    if not isinstance(value, str) and not isinstance(value, list):
        raise ValueError(f"{attribute.name} must be a string")


if t.TYPE_CHECKING:

    class TaskDefinition(t.TypedDict):
        impl: type[ext.TransformersPipeline]
        tf: TupleAutoModel | ext.BaseAutoModelClass | None
        pt: TupleAutoModel | ext.BaseAutoModelClass | None
        default: t.NotRequired[DefaultMapping]
        type: t.NotRequired[str]

else:
    TaskDefinition = dict


def _autoclass_converter(
    value: tuple[ext.BaseAutoModelClass | str, ...] | None
) -> TupleStr:
    if value is None:
        return TupleStr()
    elif isinstance(value, t.Iterable):
        value = tuple(value)
    elif not isinstance(value, tuple):
        value = (value,)
    return tuple(it if isinstance(it, str) else it.__qualname__ for it in value)


@attr.define
class TransformersOptions(ModelOptions):
    """Options for the Transformers model."""

    task: str = attr.field(factory=str, validator=attr.validators.instance_of(str))
    tf: TupleStr = attr.field(
        validator=attr.validators.deep_iterable(
            attr.validators.instance_of(str), attr.validators.instance_of(TupleStr)
        ),
        factory=TupleStr,
        converter=_autoclass_converter,
    )
    pt: TupleStr = attr.field(
        validator=attr.validators.deep_iterable(
            attr.validators.instance_of(str), attr.validators.instance_of(TupleStr)
        ),
        factory=TupleStr,
        converter=_autoclass_converter,
    )
    default: DefaultMapping = attr.field(factory=dict, converter=_deep_convert_to_tuple)
    type: str = attr.field(
        validator=attr.validators.optional(_validate_pipeline_type),
        default=None,
    )
    kwargs: t.Dict[str, t.Any] = attr.field(factory=dict)

    @staticmethod
    def process_task_mapping(
        impl: type[ext.TransformersPipeline],
        pt: tuple[ext.BaseAutoModelClass | str, ...]
        | TupleAutoModel
        | ext.BaseAutoModelClass
        | None = None,
        tf: tuple[ext.BaseAutoModelClass | str, ...]
        | TupleAutoModel
        | ext.BaseAutoModelClass
        | None = None,
        default: DefaultMapping | None = None,
        type: str | None = None,
    ) -> TaskDefinition:
        if pt is None:
            pt = TupleAutoModel()
        elif not isinstance(pt, tuple):
            pt = (pt,)
        pt = tuple(convert_to_autoclass(it) if isinstance(it, str) else it for it in pt)

        if tf is None:
            tf = TupleAutoModel()
        elif not isinstance(tf, tuple):
            tf = (tf,)
        tf = tuple(convert_to_autoclass(it) if isinstance(it, str) else it for it in tf)

        task_impl = TaskDefinition(impl=impl, pt=pt, tf=tf)

        if default is not None:
            if "model" not in default and ("pt" in default or "tf" in default):
                # case for SimpleDefaultMapping, needs then to convert to ModelDefaultMapping
                default = ModelDefaultMapping(
                    model=t.cast(SimpleDefaultMapping, default)
                )
            task_impl["default"] = default
        if type is not None:
            task_impl["type"] = type

        return task_impl

    @classmethod
    def from_task(cls, task: str, definition: TaskDefinition) -> TransformersOptions:
        return cls(
            task=task,
            tf=definition.get("tf", None),  # type: ignore (handle by cattrs converter)
            pt=definition.get("pt", None),  # type: ignore (handle by cattrs converter)
            default=definition.get("default", {}),
            type=definition.get("type", "text"),
        )

    def to_dict(self) -> dict[str, t.Any]:
        return {
            "task": self.task,
            "tf": tuple(self.tf),
            "pt": tuple(self.pt),
            "default": {
                "_".join(k) if isinstance(k, tuple) else k: v
                for k, v in self.default.items()
            },
            "type": self.type,
            "kwargs": self.kwargs,
        }


def convert_to_autoclass(cls_name: str) -> ext.BaseAutoModelClass:
    if not hasattr(transformers, cls_name):
        raise BentoMLException(
            f"Given {cls_name} is not a valid Transformers auto class. For more information, "
            "please see https://huggingface.co/docs/transformers/main/en/model_doc/auto"
        )
    return getattr(transformers, cls_name)


def get(tag_like: str | Tag) -> Model:
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
       model = bentoml.transformers.get("my_pipeline:latest")
    """
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, not loading with {MODULE_NAME}."
        )
    return model


def register_pipeline(
    task: str,
    impl: type[ext.TransformersPipeline],
    pt: tuple[ext.BaseAutoModelClass | str, ...]
    | TupleAutoModel
    | ext.BaseAutoModelClass
    | None = None,
    tf: tuple[ext.BaseAutoModelClass | str, ...]
    | TupleAutoModel
    | ext.BaseAutoModelClass
    | None = None,
    default: DefaultMapping | None = None,
    type: str | None = None,
):
    task_impl = TransformersOptions.process_task_mapping(impl, pt, tf, default, type)

    if HAS_PIPELINE_REGISTRY:
        from transformers.pipelines import PIPELINE_REGISTRY

        PIPELINE_REGISTRY.register_pipeline(
            task,
            pipeline_class=impl,
            pt_model=task_impl["pt"],
            tf_model=task_impl["tf"],
            default=task_impl.get("default", None),
            type=task_impl.get("type", None),
        )
    else:
        # For backward compatibility
        from transformers.pipelines import SUPPORTED_TASKS

        SUPPORTED_TASKS.setdefault(task, task_impl)

        _object_setattr(impl, "_registered_impl", {task: task_impl})


def delete_pipeline(task: str) -> None:
    """
    Remove pipelines from current registry by task name.
    """
    if HAS_PIPELINE_REGISTRY:
        from transformers.pipelines import PIPELINE_REGISTRY

        del PIPELINE_REGISTRY.supported_tasks[task]
    else:
        # For backward compatibility
        from transformers.pipelines import SUPPORTED_TASKS

        del SUPPORTED_TASKS[task]


def load_model(
    bento_model: str | Tag | Model,
    **kwargs: t.Any,
) -> ext.TransformersPipeline:
    """
    Load the Transformers model from BentoML local modelstore with given name.

    Args:
        bento_model: Either the tag of the model to get from the store,
                     or a BentoML :class:`~bentoml.Model` instance to load the
                     model from.
        attrs: Additional keyword arguments to pass into the pipeline.

    Returns:
        ``transformers.pipeline.base.Pipeline``: The Transformers pipeline loaded
        from the model store.

    Example:

    .. code-block:: python

        import bentoml
        pipeline = bentoml.transformers.load_model('my_model:latest')
    """
    _check_flax_supported()

    if not isinstance(bento_model, Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    from transformers.pipelines import get_supported_tasks

    options = t.cast(TransformersOptions, bento_model.info.options)

    task = options.task
    pipeline: ext.TransformersPipeline | None = None
    pipeline_class: type[ext.TransformersPipeline] | None = None

    # Set trust_remote_code to True to allow loading custom pipeline.
    kwargs.setdefault("trust_remote_code", False)

    if os.path.exists(bento_model.path_of(PIPELINE_PICKLE_NAME)):
        with open(bento_model.path_of(PIPELINE_PICKLE_NAME), "rb") as f:
            pipeline_class = cloudpickle.load(f)

    if task not in get_supported_tasks():
        logger.debug(
            "'%s' is not a supported task, trying to load custom pipeline.", task
        )

        with open(bento_model.path_of(PIPELINE_PICKLE_NAME), "rb") as f:
            pipeline_class = cloudpickle.load(f)

        register_pipeline(
            task,
            pipeline_class,
            tuple(convert_to_autoclass(auto_class) for auto_class in options.pt),
            tuple(convert_to_autoclass(auto_class) for auto_class in options.tf),
            options.default,
            options.type,
        )
        kwargs["trust_remote_code"] = True

    kwargs.setdefault("pipeline_class", pipeline_class)

    assert (
        task in get_supported_tasks()
    ), f"Task '{task}' failed to register into pipeline registry."

    kwargs.update(options.kwargs)
    if len(kwargs) > 0:
        logger.debug(
            "Loading '%s' pipeline (tag='%s') with kwargs %s.",
            task,
            bento_model.tag,
            kwargs,
        )
    try:
        return transformers.pipeline(task=task, model=bento_model.path, **kwargs)
    except Exception:
        # When loading a custom pipeline that is not available on huggingface hub,
        # the class registered in the pipeline registry will be a path to a Python file path.
        # Currently, it doesn't handle relative imports correctly, so users will need to use
        # external_modules when using 'save_model'.
        logger.debug(
            "If you are loading a custom pipeline, See https://huggingface.co/docs/transformers/main/en/add_new_pipeline#how-to-create-a-custom-pipeline for more information. We recommend to upload the custom pipeline to HuggingFace Hub to ensure consistency."
        )
        if pipeline is not None:
            logger.info(
                "Exception caught when trying to load pipeline for task '%s'. set '%s=True' to see the full exception. Return the pipeline pickle.",
                task,
                DEBUG_ENV_VAR,
            )
            logger.debug(
                "If the pipeline is a custom pipeline, Make sure to add the following to your saving code: 'import importlib; bentoml.transformers.save_model(..., external_modules=[importlib.import_module(%s.__module__)])'",
                pipeline,
            )
            if get_debug_mode():
                import traceback

                traceback.print_exc()

            # Only return pipeline if pipeline is not None.
            return pipeline

        # Otherwise, raise the exception.
        raise


@t.overload
def save_model(
    name: str,
    pipeline: ext.TransformersPipeline,
    task_name: t.LiteralString = ...,
    task_definition: dict[str, t.Any] = ...,
    *,
    signatures: ModelSignaturesType | None = ...,
    labels: dict[str, str] | None = ...,
    custom_objects: dict[str, t.Any] | None = ...,
    external_modules: t.List[ModuleType] | None = ...,
    metadata: dict[str, t.Any] | None = ...,
) -> bentoml.Model:
    ...


@t.overload
def save_model(
    name: str,
    pipeline: ext.TransformersPipeline,
    task_name: t.LiteralString | None = ...,
    task_definition: TaskDefinition | None = ...,
    *,
    signatures: ModelSignaturesType | None = ...,
    labels: dict[str, str] | None = ...,
    custom_objects: dict[str, t.Any] | None = ...,
    external_modules: t.List[ModuleType] | None = ...,
    metadata: dict[str, t.Any] | None = ...,
) -> bentoml.Model:
    ...


def save_model(
    name: str,
    pipeline: ext.TransformersPipeline,
    task_name: str | None = None,
    task_definition: dict[str, t.Any] | TaskDefinition | None = None,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name: Name for given model instance. This should pass Python identifier check.
        pipeline: Instance of the Transformers pipeline to be saved.

                  See module `src/transformers/pipelines/__init__.py <https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py#L129>`_ for more details.
        task_name: Name of pipeline task. If not provided, the task name will be derived from ``pipeline.task``.
        task_definition: Task definition for the Transformers custom pipeline. The definition is a dictionary
                         consisting of the following keys:

                        - ``impl`` (``type[transformers.Pipeline]``): The name of the pipeline implementation module. The name should be the same as the pipeline passed in the ``pipeline`` argument.
                        - ``tf`` (:code:`tuple[AnyType]`): The name of the Tensorflow auto model class. One of ``tf`` and ``pt`` auto model class argument is required.
                        - ``pt`` (:code:`tuple[AnyType]`): The name of the PyTorch auto model class. One of ``tf`` and ``pt`` auto model class argument is required.
                        - ``default`` (:code:`Dict[str, AnyType]`): The names of the default models, tokenizers, feature extractors, etc.
                        - ``type`` (:code:`str`): The type of the pipeline, e.g. ``text``, ``audio``, ``image``, ``multimodal``.

                        Example:

                        .. code-block:: python

                            task_definition = {
                                "impl": Text2TextGenerationPipeline,
                                "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
                                "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
                                "default": {"model": {"pt": "t5-base", "tf": "t5-base"}},
                                "type": "text",
                            }

                        See module `src/transformers/pipelines/__init__.py <https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py#L129>`_ for more details.
        signatures: Methods to expose for running inference on the target model. Signatures are used for creating :obj:`~bentoml.Runner` instances when serving model with :obj:`~bentoml.Service`
        labels: User-defined labels for managing models, e.g. ``team=nlp``, ``stage=dev``.
        custom_objects: Custom objects to be saved with the model. An example is ``{"my-normalizer": normalizer}``.

                        Custom objects are currently serialized with cloudpickle, but this implementation is subject to change.
        external_modules: user-defined additional python modules to be saved alongside the model or custom objects,
                          e.g. a tokenizer module, preprocessor module, model configuration module
        metadata: Custom metadata for given model.

    .. note::

        Both arguments ``task_name`` and ``task_definition`` must be provided to set save a custom pipeline.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format ``name:version`` where ``name`` is the user-defined model's name, and a generated ``version``.

    Examples:

    .. code-block:: python

        import bentoml

        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
        bento_model = bentoml.transformers.save_model("text-generation-pipeline", generator)
    """  # noqa
    _check_flax_supported()
    if not LazyType["ext.TransformersPipeline"]("transformers.Pipeline").isinstance(
        pipeline
    ):
        raise BentoMLException(
            "'pipeline' must be an instance of 'transformers.pipelines.base.Pipeline'. "
            "To save other Transformers types like models, tokenizers, configs, feature "
            "extractors, construct a pipeline with the model, tokenizer, config, or feature "
            "extractor specified as arguments, then call 'save_model' with the pipeline. "
            "Refer to https://huggingface.co/docs/transformers/main_classes/pipelines "
            "for more information on pipelines. If transformers doesn't provide a task you "
            "need, refer to the custom pipeline section to create your own pipeline.\n"
            """
            ```python
            import bentoml
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

            bentoml.transformers.save_model("text-generation-pipeline", generator)
            ```
            """
        )

    context = ModelContext(
        framework_name="transformers",
        framework_versions={"transformers": get_pkg_version("transformers")},
    )

    if signatures is None:
        signatures = {
            "__call__": {"batchable": False},
        }
        logger.info(
            'Using the default model signature for Transformers (%s) for model "%s".',
            signatures,
            name,
        )

    from transformers.pipelines import check_task
    from transformers.pipelines import get_supported_tasks

    # NOTE: safe casting to annotate task_definition types
    task_definition = (
        t.cast(TaskDefinition, task_definition)
        if task_definition is not None
        else task_definition
    )

    if metadata is None:
        metadata = {}
    metadata["_is_custom_pipeline"] = False

    if task_name is not None and task_definition is not None:
        logger.info(
            "Arguments 'task_name' and 'task_definition' are provided. Saving model with pipeline task name '%s' and task definition '%s'.",
            task_name,
            task_definition,
        )
        if pipeline.task != task_name:
            raise BentoMLException(
                f"Argument 'task_name' '{task_name}' does not match pipeline task name '{pipeline.task}'."
            )

        assert "impl" in task_definition, "'task_definition' requires 'impl' key."

        impl = task_definition["impl"]
        if type(pipeline) != impl:
            raise BentoMLException(
                f"Argument 'pipeline' is not an instance of {impl}. It is an instance of {type(pipeline)}."
            )

        # Should only use this for custom pipelines
        metadata["_is_custom_pipeline"] = True
        options_args = (task_name, task_definition)

        if task_name not in get_supported_tasks():
            logger.info(
                "Task '%s' is not available in the pipeline registry. Trying to register it.",
                task_name,
            )
            register_pipeline(task_name, **task_definition)

        assert (
            task_name in get_supported_tasks()
        ), f"Task '{task_name}' failed to register into pipeline registry."
    else:
        assert (
            task_definition is None
        ), "'task_definition' must be None if 'task_name' is not provided."

        # if task_name is None, then we derive the task from pipeline.task
        options_args = t.cast(
            "tuple[str, TaskDefinition]",
            check_task(pipeline.task if task_name is None else task_name)[:2],
        )

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        api_version=API_VERSION,
        labels=labels,
        context=context,
        options=TransformersOptions.from_task(*options_args),
        signatures=signatures,
        custom_objects=custom_objects,
        external_modules=external_modules,
        metadata=metadata,
    ) as bento_model:
        pipeline.save_pretrained(bento_model.path)

        # NOTE: we want to pickle the class so that tensorflow, flax pipeline will also work.
        # the weights is already save, so we only need to save the class.
        with open(bento_model.path_of(PIPELINE_PICKLE_NAME), "wb") as f:
            cloudpickle.dump(pipeline.__class__, f)
        return bento_model


def get_runnable(bento_model: bentoml.Model) -> type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    model_options = t.cast(TransformersOptions, bento_model.info.options)

    class TransformersRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            available_gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if available_gpus not in ("", "-1"):
                # assign GPU resources
                if not available_gpus.isdigit():
                    raise ValueError(
                        f"Expecting numeric value for CUDA_VISIBLE_DEVICES, got {available_gpus}."
                    )
                else:
                    kwargs = {"device": int(available_gpus)}
            else:
                # assign CPU resources
                kwargs = {}
            kwargs.update(model_options.kwargs)

            self.model = load_model(bento_model, **kwargs)

            # backward compatibility with previous BentoML versions.
            self.pipeline = self.model

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                self.predict_fns[method_name] = getattr(self.pipeline, method_name)

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(self: TransformersRunnable, *args: t.Any, **kwargs: t.Any) -> t.Any:
            return getattr(self.pipeline, method_name)(*args, **kwargs)

        TransformersRunnable.add_method(
            _run,
            name=method_name,
            batchable=options.batchable,
            batch_dim=options.batch_dim,
            input_spec=options.input_spec,
            output_spec=options.output_spec,
        )

    for method_name, options in bento_model.info.signatures.items():
        add_runnable_method(method_name, options)

    return TransformersRunnable
