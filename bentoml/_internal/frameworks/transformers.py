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


MODULE_NAME = "bentoml.transformers"
API_VERSION = "v1"


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
        """\
        transformers is required in order to use module `bentoml.transformers`.
        Instruction: Install transformers with `pip install transformers`.
        """
    )


@attr.define
class TransformersOptions(ModelOptions):
    """Options for the Transformers model."""

    task: str = attr.field(
        validator=[
            attr.validators.instance_of(str),
            lambda instance, attribute, value: transformers.pipelines.check_task(value),  # type: ignore
        ]
    )

    pipeline: bool = attr.field(
        default=True, validator=attr.validators.instance_of(bool)
    )

    kwargs: t.Dict[str, t.Any] = attr.field(factory=dict)

    @classmethod
    def with_options(cls, **kwargs: t.Any) -> ModelOptions:
        return cls(**kwargs)

    @staticmethod
    def to_dict(options: ModelOptions) -> dict[str, t.Any]:
        return attr.asdict(options)


def get(tag_like: str | Tag) -> Model:
    model = bentoml.models.get(tag_like)
    if model.info.module not in (MODULE_NAME, __name__):
        raise NotFound(
            f"Model {model.tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
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
            f"Model {bento_model.tag} was saved with module {bento_model.info.module}, failed loading with {MODULE_NAME}."
        )

    bento_model.info.parse_options(TransformersOptions)

    pipeline_task: str = bento_model.info.options.task  # type: ignore
    pipeline_kwargs: t.Dict[str, t.Any] = bento_model.info.options.kwargs  # type: ignore
    pipeline_kwargs.update(kwargs)
    if len(pipeline_kwargs) > 0:
        logger.info(
            f"Loading '{pipeline_task}' pipeline '{bento_model.tag}' with kwargs {pipeline_kwargs}."
        )
    return transformers.pipeline(task=pipeline_task, model=bento_model.path, **pipeline_kwargs)  # type: ignore


def save_model(
    name: str,
    pipeline: ext.TransformersPipeline,
    *,
    signatures: dict[str, ModelSignatureDict | ModelSignature] | None = None,
    labels: dict[str, str] | None = None,
    custom_objects: dict[str, t.Any] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        pipeline (:code:`Pipeline`):
            Instance of the Transformers pipeline to be saved.
        signatures (:code: `Dict[str, bool | BatchDimType | AnyType | tuple[AnyType]]`)
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
        tag = bentoml.transformers.save_model("text-generation-pipeline", generator)

        # load the model back:
        loaded = bentoml.transformers.load_model("text-generation-pipeline:latest")
        # or:
        loaded = bentoml.transformers.load_model(tag)
    """  # noqa
    if not LazyType["ext.TransformersPipeline"](
        "transformers.pipelines.base.Pipeline"
    ).isinstance(pipeline):
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
    options = TransformersOptions(task=pipeline.task)

    if signatures is None:
        signatures = {
            "__call__": {"batchable": False},
        }
        logger.info(
            f"Using the default model signature for Transformers ({signatures}) for model {name}."
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

        return bento_model.tag


def get_runnable(
    bento_model: bentoml.Model,
) -> t.Type[bentoml.Runnable]:
    """
    Private API: use :obj:`~bentoml.Model.to_runnable` instead.
    """

    class TransformersRunnable(bentoml.Runnable):
        SUPPORT_NVIDIA_GPU = True  # type: ignore
        SUPPORT_CPU_MULTI_THREADING = True  # type: ignore

        def __init__(self):
            super().__init__()

            available_gpus = os.getenv("NVIDIA_VISIBLE_DEVICES")
            if available_gpus is not None and available_gpus != "":
                # assign GPU resources
                kwargs = {
                    "device": available_gpus,
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
