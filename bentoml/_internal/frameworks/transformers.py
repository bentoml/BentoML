from __future__ import annotations

import os
import typing as t
import logging
import importlib
import importlib.util
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
from ..utils.pkg import get_pkg_version

if TYPE_CHECKING:
    from bentoml.types import ModelSignature
    from bentoml.types import ModelSignatureDict

    from ..external_typing import transformers as ext

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
        _flax_available = (
            importlib.util.find_spec("jax") is not None
            and importlib.util.find_spec("flax") is not None
        )
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


try:
    import transformers

    _check_flax_supported()
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        "transformers is required in order to use module `bentoml.transformers`. "
        "Install transformers with `pip install transformers`."
    )


@attr.define
class TransformersOptions(ModelOptions):
    """Options for the Transformers model."""

    task: str = attr.field(validator=[attr.validators.instance_of(str)])
    tf: t.Tuple[str] = attr.field(
        validator=[
            attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(str)
            )
        ],  # type: ignore
        factory=(tuple),
    )
    pt: t.Tuple[str] = attr.field(
        validator=[
            attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(str)
            )
        ],  # type: ignore
        factory=(tuple),
    )
    default: t.Dict[str, t.Any] = attr.field(factory=dict)
    type: str = (attr.field(validator=[attr.validators.instance_of(str)], default=None),)  # type: ignore
    kwargs: t.Dict[str, t.Any] = attr.field(factory=dict)


def get(tag_like: str | Tag) -> Model:
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
    if not isinstance(bento_model, Model):
        bento_model = get(bento_model)

    if bento_model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, not loading with {MODULE_NAME}."
        )

    from transformers.pipelines import SUPPORTED_TASKS

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
                getattr(transformers, auto_class)  # type: ignore
                for auto_class in bento_model.info.options.tf  # type: ignore
            ),
            "pt": tuple(
                getattr(transformers, auto_class)  # type: ignore
                for auto_class in bento_model.info.options.pt  # type: ignore
            ),
            "default": bento_model.info.options.default,  # type: ignore
            "type": bento_model.info.options.type,  # type: ignore
        }

    kwargs: t.Dict[str, t.Any] = bento_model.info.options.kwargs  # type: ignore
    kwargs.update(kwargs)
    if len(kwargs) > 0:
        logger.info(
            f"Loading '{task}' pipeline '{bento_model.tag}' with kwargs {kwargs}."
        )
    return transformers.pipeline(task=task, model=bento_model.path, **kwargs)  # type: ignore


def save_model(
    name: str,
    pipeline: ext.TransformersPipeline,
    task_name: str | None = None,
    task_definition: t.Dict[str, t.Any] | None = None,
    *,
    signatures: dict[str, ModelSignatureDict | ModelSignature] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> bentoml.Model:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        pipeline (:code:`Pipeline`):
            Instance of the Transformers pipeline to be saved. See Transformers
            ``module src/transformers/pipelines/__init__.py`` for more details,
            https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/__init__.py#L129.
        task_name (:code:`str`):
            Name of pipeline task. If not provided, the task name will be derived from ``pipeline.task``.
            Both arguments ``task_name`` and ``task_definition`` must be provided to set save a custom pipeline.
        task_definition (:code:`dict`):
            Task definition for the Transformers custom pipeline. The definition is a dictionary
            consisting of the following keys:

            ``impl`` (:code:`str`): The name of the pipeline implementation module. The name should
            be the same as the pipeline passed in the ``pipeline`` argument.
            ``tf`` (:code:`tuple[AnyType]`): The name of the Tensorflow auto model class. One of ``tf`` and ``pt``
            auto model class argument is required.
            ``pt`` (:code:`tuple[AnyType]`): The name of the PyTorch auto model class. One of ``tf`` and ``pt``
            auto model class argument is required.
            ``default`` (:code:`Dict[str, AnyType]`): The names of the default models, tokenizers, feature extractors, etc.
            ``type`` (:code:`str`): The type of the pipeline, e.g. ``text``, ``audio``, ``image``, ``multimodal``.

            Example:

            .. code-block:: json

                {
                    "impl": Text2TextGenerationPipeline,
                    "tf": (TFAutoModelForSeq2SeqLM,) if is_tf_available() else (),
                    "pt": (AutoModelForSeq2SeqLM,) if is_torch_available() else (),
                    "default": {"model": {"pt": "t5-base",
                    "tf": "t5-base"}}, "type": "text",
                }

            See Transformers ``module src/transformers/pipelines/__init__.py`` for more details.
            ``task_name`` and ``task_definition`` must be both provided or both not provided.
        signatures (:code:`Dict[str, bool | BatchDimType | AnyType | tuple[AnyType]]`)
            Methods to expose for running inference on the target model. Signatures are
             used for creating Runner instances when serving model with bentoml.Service
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (``dict[str, Any]``, optional):
            Custom objects to be saved with the model. An example is
            ``{"my-normalizer": normalizer}``.

            Custom objects are currently serialized with cloudpickle, but this implementation is
            subject to change.
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.

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

            available_gpus: str = os.getenv("CUDA_VISIBLE_DEVICES")
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
