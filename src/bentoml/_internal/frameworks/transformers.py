from __future__ import annotations

import os
import typing as t
import logging
import platform
import warnings
from types import ModuleType

import attr

import bentoml

from ..tag import Tag
from ..types import LazyType
from ..utils import LazyLoader
from ..utils.pkg import get_pkg_version
from ..utils.pkg import pkg_version_info
from ...exceptions import NotFound
from ...exceptions import BentoMLException
from ...exceptions import MissingDependencyException
from ..models.model import Model
from ..models.model import ModelContext
from ..models.model import ModelOptions
from ..models.model import ModelSignature

if t.TYPE_CHECKING:
    import torch
    import tensorflow as tf
    import cloudpickle
    from transformers.models.auto.auto_factory import (
        _BaseAutoModelClass as BaseAutoModelClass,
    )

    from ..models.model import ModelSignaturesType

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
    TupleAutoModel = tuple[BaseAutoModelClass, ...]

    class PreTrainedProtocol(t.Protocol):
        @property
        def framework(self) -> str:
            ...

        def save_pretrained(self, save_directory: str, **kwargs: t.Any) -> None:
            ...

        @classmethod
        def from_pretrained(
            cls, pretrained_model_name_or_path: str, *args: t.Any, **kwargs: t.Any
        ) -> PreTrainedProtocol:
            ...

    P = t.ParamSpec("P")

else:
    TupleStr = TupleAutoModel = tuple
    DefaultMapping = SimpleDefaultMapping = ModelDefaultMapping = dict

    torch = LazyLoader("torch", globals(), "torch")
    tf = LazyLoader("tf", globals(), "tensorflow")
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

if t.TYPE_CHECKING:
    TransformersPreTrained = (
        transformers.PreTrainedTokenizerBase
        | transformers.PreTrainedTokenizer
        | transformers.PreTrainedTokenizerFast
        | transformers.PreTrainedModel
        | transformers.TFPreTrainedModel
        | transformers.FlaxPreTrainedModel
        | transformers.image_processing_utils.BaseImageProcessor
        | transformers.SequenceFeatureExtractor
        | transformers.Pipeline
    )


__all__ = ["load_model", "save_model", "get_runnable", "get"]

_object_setattr = object.__setattr__

MODULE_NAME = "bentoml.transformers"
API_VERSION = "v2"
PIPELINE_PICKLE_NAME = f"pipeline.{API_VERSION}.pkl"
PRETRAINED_PROTOCOL_NAME = f"pretrained.{API_VERSION}.pkl"


logger = logging.getLogger(__name__)

TRANSFORMERS_VERSION = pkg_version_info("transformers")

HAS_PIPELINE_REGISTRY = TRANSFORMERS_VERSION >= (4, 21, 0)


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
        impl: type[transformers.Pipeline]
        tf: TupleAutoModel | BaseAutoModelClass | None
        pt: TupleAutoModel | BaseAutoModelClass | None
        default: t.NotRequired[DefaultMapping]
        type: t.NotRequired[str]

else:
    TaskDefinition = dict


def _autoclass_converter(
    value: tuple[BaseAutoModelClass | str, ...] | None
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
        impl: type[transformers.Pipeline],
        pt: tuple[BaseAutoModelClass | str, ...]
        | TupleAutoModel
        | BaseAutoModelClass
        | None = None,
        tf: tuple[BaseAutoModelClass | str, ...]
        | TupleAutoModel
        | BaseAutoModelClass
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


def convert_to_autoclass(cls_name: str) -> BaseAutoModelClass:
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

    .. note::


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
    impl: type[transformers.Pipeline],
    pt: tuple[BaseAutoModelClass | str, ...]
    | TupleAutoModel
    | BaseAutoModelClass
    | None = None,
    tf: tuple[BaseAutoModelClass | str, ...]
    | TupleAutoModel
    | BaseAutoModelClass
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


@t.overload
def load_model(
    bento_model: str | Tag | Model, **kwargs: t.Any
) -> transformers.Pipeline:
    ...


@t.overload
def load_model(
    bento_model: str | Tag | Model, *args: t.Any, **kwargs: t.Any
) -> TransformersPreTrained:
    ...


def load_model(bento_model: str | Tag | Model, *args: t.Any, **kwargs: t.Any) -> t.Any:
    """
    Load the Transformers model from BentoML local modelstore with given name.

    Args:
        bento_model: Either the tag of the model to get from the store,
                     or a BentoML :class:`~bentoml.Model` instance to load the
                     model from.
        args: Additional model args to be passed into the model if the object is a Transformers PreTrained protocol.
              This shouldn't be used when the bento_model is a pipeline.
        kwargs: Additional keyword arguments to pass into the pipeline.

    Returns:
        ``transformers.pipeline.base.Pipeline``: The Transformers pipeline loaded
        from the model store.

    Example:

    .. code-block:: python

        import bentoml
        pipeline = bentoml.transformers.load_model('my_model:latest')
    """
    if not isinstance(bento_model, Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    options = t.cast(TransformersOptions, bento_model.info.options)
    api_version = bento_model.info.api_version
    task = options.task
    pipeline: transformers.Pipeline | None = None

    if api_version == "v1":
        # NOTE: backward compatibility
        from transformers.pipelines import get_supported_tasks

        logger.warning(
            "Upgrading '%s' to with newer 'save_model' is recommended. Loading model with version 1.",
            bento_model,
        )

        # Set trust_remote_code to True to allow loading custom pipeline.
        kwargs.setdefault("trust_remote_code", False)
        kwargs.update(options.kwargs)
        if len(kwargs) > 0:
            logger.debug(
                "Loading '%s' pipeline (tag='%s') with kwargs %s.",
                task,
                bento_model.tag,
                kwargs,
            )

        # v1 pickled the pipeline with 'pipeline.v1.pkl'.
        fname = "pipeline.v1.pkl"
        if os.path.exists(bento_model.path_of(fname)):
            with open(bento_model.path_of(fname), "rb") as f:
                pipeline = cloudpickle.load(f)

        if task not in get_supported_tasks():
            logger.debug("'%s' is a custom task. Registering it to the registry.", task)
            kwargs["trust_remote_code"] = True
            # most likely a custom pipeline
            if pipeline is None:
                # if the pipeline is not loaded from the pickle, and it is custom, then we should be able to load
                # it with model=bento_model.path
                return transformers.pipeline(model=bento_model.path, **kwargs)
            else:
                register_pipeline(
                    task,
                    pipeline.__class__,
                    tuple(
                        convert_to_autoclass(auto_class) for auto_class in options.pt
                    ),
                    tuple(
                        convert_to_autoclass(auto_class) for auto_class in options.tf
                    ),
                    options.default,
                    options.type,
                )

        kwargs.setdefault("pipeline_class", pipeline.__class__ if pipeline else None)

        assert (
            task in get_supported_tasks()
        ), f"Task '{task}' is not a valid task for pipeline (available: {get_supported_tasks()})."

        return (
            transformers.pipeline(task=task, model=bento_model.path, **kwargs)
            if pipeline is None
            else pipeline
        )
    else:
        if api_version != "v2":
            logger.warning(
                "Got unknown API version '%s', unexpected errors may occur.",
                api_version,
            )

        if "_pretrained_class" in bento_model.info.metadata:
            with open(bento_model.path_of(PRETRAINED_PROTOCOL_NAME), "rb") as f:
                protocol: PreTrainedProtocol = cloudpickle.load(f)
            return protocol.from_pretrained(bento_model.path, *args, **kwargs)
        else:
            assert (
                len(args) == 0
            ), "Positional args are not supported for pipeline. Make sure to only use kwargs instead."
            with open(bento_model.path_of(PIPELINE_PICKLE_NAME), "rb") as f:
                pipeline_class: type[transformers.Pipeline] = cloudpickle.load(f)

            from transformers.pipelines import get_supported_tasks

            if task not in get_supported_tasks():
                logger.debug(
                    "'%s' is not a supported task, trying to load custom pipeline.",
                    task,
                )

                register_pipeline(
                    task,
                    pipeline_class,
                    tuple(
                        convert_to_autoclass(auto_class) for auto_class in options.pt
                    ),
                    tuple(
                        convert_to_autoclass(auto_class) for auto_class in options.tf
                    ),
                    options.default,
                    options.type,
                )
                kwargs["trust_remote_code"] = True

            kwargs.setdefault("pipeline_class", pipeline_class)

            assert (
                task in get_supported_tasks()
            ), f"Task '{task}' is not a valid task for pipeline (available: {get_supported_tasks()})."

            kwargs.update(options.kwargs)
            if len(kwargs) > 0:
                logger.debug(
                    "Loading '%s' pipeline (tag='%s') with kwargs %s.",
                    task,
                    bento_model.tag,
                    kwargs,
                )
            try:
                return transformers.pipeline(
                    task=task, model=bento_model.path, **kwargs
                )
            except Exception:
                # NOTE: When loading a custom pipeline that is not available on huggingface hub,
                # the class registered in the pipeline registry will be a path to a Python file path.
                # Currently, it doesn't handle relative imports correctly, so users will need to use
                # external_modules when using 'save_model'.
                logger.debug(
                    "If you are loading a custom pipeline, See https://huggingface.co/docs/transformers/main/en/add_new_pipeline#how-to-create-a-custom-pipeline for more information. We recommend to upload the custom pipeline to HuggingFace Hub to ensure consistency. You can also try adding the pipeline instance to 'external_modules': 'import importlib; bentoml.transformers.save_model(..., external_modules=[importlib.import_module(pipeline_instance.__module__)])'"
                )
                raise


def make_default_signatures(pretrained: t.Any) -> ModelSignaturesType:
    default_config = ModelSignature(batchable=False)
    infer_fn = ("__call__",)

    # NOTE: for all processor type recommend to use custom signatures since it is
    # a per case basis.
    if pkg_version_info("transformers")[:2] < (4, 17):
        logger.warning(
            "Given transformers version is less than 4.17.0, signatures inference will be disabled. Make sure to specify the signatures manually."
        )
        return {}

    if transformers.processing_utils.ProcessorMixin in pretrained.__class__.__bases__:
        logger.info(
            "Given '%s' extends the 'transformers.ProcessorMixin'. Make sure to specify the signatures manually if it has additional functions.",
            pretrained.__class__.__name__,
        )
        return {k: default_config for k in ("__call__", "batch_decode", "decode")}

    if isinstance(pretrained, transformers.PreTrainedTokenizerBase):
        infer_fn = (
            "__call__",
            "tokenize",
            "encode",
            "encode_plus",
            "batch_encode_plus",
            "pad",
            "create_token_type_ids_from_sequences",
            "build_inputs_with_special_tokens",
            "prepare_for_model",
            "truncate_sequences",
            "convert_tokens_to_string",
            "batch_decode",
            "decode",
            "get_special_tokens_mask",
            "clean_up_tokenization",
            "prepare_seq2seq_batch",
        )
    elif isinstance(pretrained, transformers.PreTrainedModel):
        infer_fn = (
            "__call__",
            "forward",
            "generate",
            "contrastive_search",
            "greedy_search",
            "sample",
            "beam_search",
            "beam_sample",
            "group_beam_search",
            "constrained_beam_search",
        )
    elif isinstance(pretrained, transformers.TFPreTrainedModel):
        infer_fn = (
            "__call__",
            "predict",
            "call",
            "generate",
            "compute_transition_scores",
            "greedy_search",
            "sample",
            "beam_search",
            "contrastive_search",
        )
    elif isinstance(pretrained, transformers.FlaxPreTrainedModel):
        infer_fn = ("__call__", "generate")
    elif isinstance(pretrained, transformers.image_processing_utils.BaseImageProcessor):
        infer_fn = ("__call__", "preprocess")
    elif isinstance(pretrained, transformers.SequenceFeatureExtractor):
        infer_fn = ("pad",)
    elif not isinstance(pretrained, transformers.Pipeline):
        logger.warning(
            "Unable to infer default signatures for '%s'. Make sure to specify it manually.",
            pretrained,
        )
        return {}

    return {k: default_config for k in infer_fn}


def save_model(
    name: Tag | str,
    pretrained_or_pipeline: TransformersPreTrained
    | transformers.Pipeline
    | PreTrainedProtocol
    | None = None,
    pipeline: transformers.Pipeline | None = None,
    task_name: str | None = None,
    task_definition: dict[str, t.Any] | TaskDefinition | None = None,
    *,
    signatures: ModelSignaturesType | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    external_modules: t.List[ModuleType] | None = None,
    metadata: dict[str, t.Any] | None = None,
    **save_kwargs: t.Any,
) -> bentoml.Model:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name: Name for given model instance. This should pass Python identifier check.
        pretrained_or_pipeline: Instance of the Transformers pipeline to be saved, or any instance that ``transformers`` supports.
                                The object instance should have ``save_pretrained`` and ``from_pretrained`` (follows the protocol that is defined by `transformers`.)
        task_name: Name of pipeline task. If not provided, the task name will be derived from ``pipeline.task``. Only needed when the target is a custom pipeline.
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
        save_kwargs: Additional keyword arguments to be pass to ````

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
    # backward compatibility
    if pipeline is not None:
        warnings.warn(
            f"The 'pipeline={pipeline}' argument is deprecated and will be removed in the future. Please use 'pretrained_or_pipeline' instead.",
            DeprecationWarning,
        )
        pretrained_or_pipeline = pipeline

    assert (
        pretrained_or_pipeline is not None
    ), "Please provide a pipeline or a pretrained object as a second argument."

    # The below API are introduced since 4.18
    if pkg_version_info("transformers")[:2] >= (4, 18):
        from transformers.utils import is_tf_available
        from transformers.utils import is_flax_available
        from transformers.utils import is_torch_available
    else:
        from .utils.transformers import is_tf_available
        from .utils.transformers import is_flax_available
        from .utils.transformers import is_torch_available

    framework_versions = {"transformers": get_pkg_version("transformers")}
    if is_torch_available():
        framework_versions["torch"] = get_pkg_version("torch")
    if is_tf_available():
        from .utils.tensorflow import get_tf_version

        framework_versions[
            "tensorflow-macos" if platform.system() == "Darwin" else "tensorflow"
        ] = get_tf_version()
    if is_flax_available():
        framework_versions.update(
            {
                "flax": get_pkg_version("flax"),
                "jax": get_pkg_version("jax"),
                "jaxlib": get_pkg_version("jaxlib"),
            }
        )
    context = ModelContext(
        framework_name="transformers", framework_versions=framework_versions
    )

    if signatures is None:
        signatures = make_default_signatures(pretrained_or_pipeline)
        # NOTE: ``make_default_signatures`` can return an empty dict, hence we will only
        # log when signatures are available.
        if signatures:
            logger.info(
                'Using the default model signature for Transformers (%s) for model "%s".',
                signatures,
                name,
            )

    if LazyType("transformers.Pipeline").isinstance(pretrained_or_pipeline):
        from transformers.pipelines import check_task
        from transformers.pipelines import get_supported_tasks

        # NOTE: safe casting to annotate task_definition types
        task_definition = (
            t.cast(TaskDefinition, task_definition)
            if task_definition is not None
            else task_definition
        )

        pipeline_ = t.cast("transformers.Pipeline", pretrained_or_pipeline)

        if task_name is not None and task_definition is not None:
            logger.info(
                "Arguments 'task_name' and 'task_definition' are provided. Saving model with pipeline task name '%s' and task definition '%s'.",
                task_name,
                task_definition,
            )
            if pipeline_.task != task_name:
                raise BentoMLException(
                    f"Argument 'task_name' '{task_name}' does not match pipeline task name '{pipeline_.task}'."
                )

            assert "impl" in task_definition, "'task_definition' requires 'impl' key."

            impl = task_definition["impl"]
            if type(pipeline_) != impl:
                raise BentoMLException(
                    f"Argument 'pipeline' is not an instance of {impl}. It is an instance of {type(pipeline_)}."
                )
            options_args = (task_name, task_definition)

            if task_name not in get_supported_tasks():
                register_pipeline(task_name, **task_definition)
                logger.info(
                    "Task '%s' is a custom task and has been registered to the pipeline registry.",
                    task_name,
                )

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
                check_task(pipeline_.task if task_name is None else task_name)[:2],
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
            pipeline_.save_pretrained(bento_model.path, **save_kwargs)

            # NOTE: we want to pickle the class so that tensorflow, flax pipeline will also work.
            # the weights is already save, so we only need to save the class.
            with open(bento_model.path_of(PIPELINE_PICKLE_NAME), "wb") as f:
                cloudpickle.dump(pipeline_.__class__, f)
            return bento_model
    else:
        pretrained = t.cast("PreTrainedProtocol", pretrained_or_pipeline)
        assert all(
            hasattr(pretrained, defn) for defn in ("save_pretrained", "from_pretrained")
        ), f"'pretrained={pretrained}' is not a valid Transformers object. It must have 'save_pretrained' and 'from_pretrained' methods."
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "_pretrained_class": pretrained.__class__.__name__,
            }
        )
        if hasattr(pretrained, "framework") and isinstance(
            pretrained,
            (
                transformers.PreTrainedModel,
                transformers.TFPreTrainedModel,
                transformers.FlaxPreTrainedModel,
            ),
        ):
            # NOTE: Only PreTrainedModel and variants has this, not tokenizer.
            metadata["_framework"] = pretrained.framework

        with bentoml.models.create(
            name,
            module=MODULE_NAME,
            api_version=API_VERSION,
            labels=labels,
            context=context,
            options=TransformersOptions(),
            signatures=signatures,
            custom_objects=custom_objects,
            external_modules=external_modules,
            metadata=metadata,
        ) as bento_model:
            pretrained.save_pretrained(bento_model.path, **save_kwargs)

            with open(bento_model.path_of(PRETRAINED_PROTOCOL_NAME), "wb") as f:
                cloudpickle.dump(pretrained.__class__, f)
            return bento_model


def get_runnable(bento_model: bentoml.Model) -> type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class TransformersRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            available_gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
            # assign CPU resources
            kwargs = {}

            if available_gpus not in ("", "-1"):
                # assign GPU resources
                if not available_gpus.isdigit():
                    raise ValueError(
                        f"Expecting numeric value for CUDA_VISIBLE_DEVICES, got {available_gpus}."
                    )
                if "_pretrained_class" not in bento_model.info.metadata:
                    # NOTE: then this is a pipeline. We then pass the device to it.
                    kwargs["device"] = int(available_gpus)
            if "_pretrained_class" not in bento_model.info.metadata:
                self.model = load_model(bento_model, **kwargs)
            else:
                if "_framework" in bento_model.info.metadata:
                    if "torch" == bento_model.info.metadata["_framework"]:
                        self.model = t.cast(
                            transformers.PreTrainedModel,
                            load_model(bento_model, **kwargs),
                        ).to(
                            torch.device(
                                "cuda" if available_gpus not in ("", "-1") else "cpu"
                            )
                        )
                        torch.set_default_tensor_type("torch.cuda.FloatTensor")
                    elif "tf" == bento_model.info.metadata["_framework"]:
                        with tf.device(
                            "/device:CPU:0"
                            if available_gpus in ("", "-1")
                            else f"/device:GPU:{available_gpus}"
                        ):
                            self.model = t.cast(
                                transformers.TFPreTrainedModel,
                                load_model(bento_model, **kwargs),
                            )
                    else:
                        # NOTE: we need to hide all GPU from TensorFlow, otherwise it will try to allocate
                        # memory on the GPU and make it unavailable for JAX.
                        tf.config.experimental.set_visible_devices([], "GPU")
                        self.model = t.cast(
                            transformers.FlaxPreTrainedModel,
                            load_model(bento_model, **kwargs),
                        )
                else:
                    logger.warning(
                        "Current %s is saved with an older version of BentoML. Setting GPUs on this won't work as expected. Make sure to save it with a newer version of BentoML.",
                        bento_model,
                    )
                    self.model = load_model(bento_model, **kwargs)

            # NOTE: backward compatibility with previous BentoML versions.
            self.pipeline = self.model

            self.predict_fns: dict[str, t.Callable[..., t.Any]] = {}
            for method_name in bento_model.info.signatures:
                self.predict_fns[method_name] = getattr(self.model, method_name)

    def add_runnable_method(method_name: str, options: ModelSignature):
        def _run(self: TransformersRunnable, *args: t.Any, **kwargs: t.Any) -> t.Any:
            return self.predict_fns[method_name](*args, **kwargs)

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
