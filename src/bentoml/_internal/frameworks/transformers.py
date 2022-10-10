from __future__ import annotations

import os
import typing as t
import logging
from types import ModuleType
from typing import TYPE_CHECKING

import attr

import bentoml
from bentoml import Tag
from bentoml.models import Model
from bentoml.models import ModelContext
from bentoml.models import ModelOptions
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import LazyType
from ..utils import LazyLoader
from ..utils.pkg import find_spec
from ..utils.pkg import get_pkg_version

if TYPE_CHECKING:
    import transformers

    from bentoml.types import ModelSignature

    from ..models.model import ModelSignaturesType
    from ..external_typing import transformers as ext
else:
    exc_msg = "transformers is required in order to use module `bentoml.transformers`. Install transformers with `pip install transformers`."
    transformers = LazyLoader(
        "transformers", globals(), "transformers", exc_msg=exc_msg
    )


__all__ = ["load_model", "save_model", "get_runnable", "get"]


MODULE_NAME = "bentoml.transformers"
API_VERSION = "v1"
PIPELINE_PICKLE_NAME = f"pipeline.{API_VERSION}.pkl"


logger = logging.getLogger(__name__)


def _check_flax_supported() -> None:  # pragma: no cover
    _supported: bool = get_pkg_version("transformers").startswith("4")

    if not _supported:
        logger.warning(
            "Detected transformers version: "
            f"{get_pkg_version('transformers')}, which "
            "doesn't have supports for Flax. "
            "Update `transformers` to 4.x and "
            "above to have Flax supported."
        )
    else:
        _flax_available = find_spec("jax") is not None and find_spec("flax") is not None
        if _flax_available:
            _jax_version = get_pkg_version("jax")
            _flax_version = get_pkg_version("flax")
            logger.info(
                f"Jax version {_jax_version}, "
                f"Flax version {_flax_version} available."
            )
        else:
            logger.warning(
                "No versions of Flax or Jax are found under "
                "the current machine. In order to use "
                "Flax with transformers 4.x and above, "
                "refers to https://github.com/google/flax#quick-install"
            )


def _deep_convert_to_tuple(dct: dict[str, t.Any]) -> dict[str, tuple[str, str | None]]:
    for k, v in dct.items():
        if isinstance(v, list):
            dct[k] = tuple(v)  # type: ignore
    return dct


def _validate_type(_: t.Any, attribute: attr.Attribute[t.Any], value: t.Any) -> None:
    """
    Validate the type of the given pipeline definition. The value is expected to be a `str`.
    `list` type is also allowed here to maintain compatibility with an earlier introduced bug.

    TODO: disallow list type in the next minor version release.
    """
    if not isinstance(value, str) and not isinstance(value, list):
        raise ValueError(f"{attribute.name} must be a string")


@attr.define
class TransformersOptions(ModelOptions):
    """Options for the Transformers model."""

    task: str = attr.field(validator=attr.validators.instance_of(str))
    tf: t.Tuple[str] = attr.field(
        validator=attr.validators.optional(
            attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(str)
            )
        ),  # type: ignore
        factory=tuple,
        converter=tuple,
    )
    pt: t.Tuple[str] = attr.field(
        validator=attr.validators.optional(
            attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(str)
            )
        ),  # type: ignore
        factory=tuple,
        converter=tuple,
    )
    default: t.Dict[str, t.Any] = attr.field(
        factory=dict, converter=_deep_convert_to_tuple
    )
    type: str = attr.field(
        validator=attr.validators.optional(_validate_type),
        default=None,
    )
    kwargs: t.Dict[str, t.Any] = attr.field(factory=dict)


def _convert_to_auto_class(cls_name: str) -> ext.BaseAutoModelClass:
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


def load_model(
    bento_model: str | Tag | Model,
    **kwargs: t.Any,
) -> ext.TransformersPipeline:
    """
    Load the Transformers model from BentoML local modelstore with given name.

    Args:
        bento_model (``str`` ``|`` :obj:`~bentoml.Tag` ``|`` :obj:`~bentoml.Model`):
            Either the tag of the model to get from the store, or a BentoML `~bentoml.Model`
            instance to load the model from.
        kwargs (:code:`Any`):
            Additional keyword arguments to pass to the model.

    Returns:
        ``Pipeline``:
            The Transformers pipeline loaded from the model store.

    Example:

    .. code-block:: python

        import bentoml
        pipeline = bentoml.transformers.load_model('my_model:latest')
    """  # noqa
    _check_flax_supported()

    if not isinstance(bento_model, Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    from transformers.pipelines import SUPPORTED_TASKS

    if TYPE_CHECKING:
        options = t.cast(TransformersOptions, bento_model.info.options)
    else:
        options = bento_model.info.options

    task: str = bento_model.info.options.task  # type: ignore
    if task not in SUPPORTED_TASKS:
        try:
            import cloudpickle  # type: ignore
        except ImportError:  # pragma: no cover
            raise MissingDependencyException(
                "Module `cloudpickle` is required in order to use to load custom pipelines."
            )

        with open(bento_model.path_of(PIPELINE_PICKLE_NAME), "rb") as f:
            pipeline = cloudpickle.load(f)

        SUPPORTED_TASKS[task] = {
            "impl": type(pipeline),
            "tf": tuple(
                _convert_to_auto_class(auto_class) for auto_class in options.tf
            ),
            "pt": tuple(
                _convert_to_auto_class(auto_class) for auto_class in options.pt
            ),
            "default": options.default,
            "type": options.type,
        }

    extra_kwargs: dict[str, t.Any] = options.kwargs
    extra_kwargs.update(kwargs)
    if len(extra_kwargs) > 0:
        logger.info(
            f"Loading '{task}' pipeline '{bento_model.tag}' with kwargs {extra_kwargs}."
        )
    return transformers.pipeline(task=task, model=bento_model.path, **extra_kwargs)


def save_model(
    name: str,
    pipeline: ext.TransformersPipeline,
    task_name: str | None = None,
    task_definition: dict[str, t.Any] | None = None,
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

                        - ``impl`` (:code:`str`): The name of the pipeline implementation module. The name should be the same as the pipeline passed in the ``pipeline`` argument.
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
        external_modules (:code:`List[ModuleType]`, `optional`, default to :code:`None`):
            user-defined additional python modules to be saved alongside the model or custom objects,
            e.g. a tokenizer module, preprocessor module, model configuration module
        metadata: Custom metadata for given model.

    .. note::

        Both arguments ``task_name`` and ``task_definition`` must be provided to set save a custom pipeline.

    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is
        the user-defined model's name, and a generated `version`.

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
    if not isinstance(
        pipeline,
        LazyType["ext.TransformersPipeline"]("transformers.pipelines.base.Pipeline"),  # type: ignore
    ):
        raise BentoMLException(
            "`pipeline` must be an instance of `transformers.pipelines.base.Pipeline`. "
            "To save other Transformers types like models, tokenizers, configs, feature "
            "extractors, construct a pipeline with the model, tokenizer, config, or feature "
            "extractor specified as arguments, then call save_model with the pipeline. "
            "Refer to https://huggingface.co/docs/transformers/main_classes/pipelines "
            "for more information on pipelines. If transformers doesn't provide a task you "
            "need, refers to the custom pipeline section to create your own pipelines."
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
            f"Using the default model signature for Transformers ({signatures}) for model {name}."
        )

    if task_name is not None and task_definition is not None:
        from transformers.pipelines import TASK_ALIASES
        from transformers.pipelines import SUPPORTED_TASKS

        try:
            import cloudpickle  # type: ignore
        except ImportError:  # pragma: no cover
            raise MissingDependencyException(
                "Module `cloudpickle` is required in order to use to save custom pipelines."
            )

        logger.info(
            f"Arguments `task_name` and `task_definition` are provided. Saving model with pipeline "
            f"task name '{task_name}' and task definition '{task_definition}'."
        )

        if pipeline.task is None or pipeline.task != task_name:
            raise BentoMLException(
                f"Argument `task_name` '{task_name}' does not match pipeline task name '{pipeline.task}'."
            )

        impl: type = task_definition["impl"]
        if type(pipeline) != impl:
            raise BentoMLException(
                f"Argument `pipeline` is not an instance of {impl}. It is an instance of {type(pipeline)}."
            )

        if task_name in TASK_ALIASES:
            task_name = TASK_ALIASES[task_name]

        if task_name in SUPPORTED_TASKS:
            if SUPPORTED_TASKS[task_name] != task_definition:
                raise BentoMLException(
                    f"Argument `task_definition` '{task_definition}' does not match pipeline task "
                    "definition '{SUPPORTED_TASKS[task_name]}'."
                )
        else:
            SUPPORTED_TASKS[task_name] = task_definition

        options = TransformersOptions(
            task=task_name,
            pt=tuple(
                auto_class.__qualname__ for auto_class in task_definition.get("pt", ())
            ),
            tf=tuple(
                auto_class.__qualname__ for auto_class in task_definition.get("tf", ())
            ),
            default=task_definition.get("default", {}),
            type=task_definition.get("type", "text"),
        )

        with bentoml.models.create(
            name,
            module=MODULE_NAME,
            api_version=API_VERSION,
            labels=labels,
            context=context,
            options=options,
            signatures=signatures,
            custom_objects=custom_objects,
            external_modules=external_modules,
            metadata=metadata,
        ) as bento_model:
            pipeline.save_pretrained(bento_model.path)

            with open(bento_model.path_of(PIPELINE_PICKLE_NAME), "wb") as f:
                cloudpickle.dump(pipeline, f)

            return bento_model

    else:
        options = TransformersOptions(task=pipeline.task)

        with bentoml.models.create(
            name,
            module=MODULE_NAME,
            api_version=API_VERSION,
            labels=labels,
            context=context,
            options=options,
            signatures=signatures,
            custom_objects=custom_objects,
            external_modules=external_modules,
            metadata=metadata,
        ) as bento_model:
            pipeline.save_pretrained(bento_model.path)

            return bento_model


def get_runnable(
    bento_model: bentoml.Model,
) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class TransformersRunnable(bentoml.Runnable):
        SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
        SUPPORTS_CPU_MULTI_THREADING = True

        def __init__(self):
            super().__init__()

            available_gpus: str = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if available_gpus is not None and available_gpus not in ("", "-1"):
                # assign GPU resources
                if not available_gpus.isdigit():
                    raise ValueError(
                        f"Expecting numeric value for CUDA_VISIBLE_DEVICES, got {available_gpus}."
                    )
                else:
                    kwargs = {
                        "device": int(available_gpus),
                    }
            else:
                # assign CPU resources
                kwargs = {}

            self.pipeline = load_model(bento_model, **kwargs)

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
