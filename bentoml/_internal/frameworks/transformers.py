import re
import typing as t
import logging
import importlib.util
from typing import TYPE_CHECKING
from importlib import import_module

from simple_di import inject
from simple_di import Provide

import bentoml
from bentoml import Tag
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import LazyType
from ..utils.pkg import get_pkg_version
from .common.model_runner import BaseModelRunner
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:

    from ..models import ModelStore
    from ..external_typing import transformers as ext

try:
    import transformers
    from transformers.models.auto.configuration_auto import AutoConfig
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        transformers is required in order to use module `bentoml.transformers`.
        Instruction: Install transformers with `pip install transformers`.
        """
    )


def check_model_type(identifier: t.Any) -> bool:
    return (
        LazyType["ext.TransformersModelType"](
            "transformers.modeling_utils.PreTrainedModel"
        ).isinstance(identifier)
        or LazyType["ext.TransformersModelType"](
            "transformers.modeling_flax_utils.FlaxPreTrainedModel"
        ).isinstance(identifier)
        or LazyType["ext.TransformersModelType"](
            "transformers.modeling_tf_utils.TFPreTrainedModel"
        ).isinstance(identifier)
    )


def check_tokenizer_type(tokenizer: t.Any) -> bool:
    return LazyType["ext.TransformersTokenizerType"](
        "transformers.tokenization_utils.PreTrainedTokenizer"
    ).isinstance(tokenizer) or LazyType["ext.TransformersTokenizerType"](
        "transformers.tokenization_utils_fast.PreTrainedTokenizerFast"
    ).isinstance(
        tokenizer
    )


def check_fe_type(fe: t.Any) -> bool:
    return LazyType["ext.PreTrainedFeatureExtractor"](
        "transformers.feature_extraction_sequence_utils.SequenceFeatureExtractor"
    ).isinstance(fe) or LazyType["ext.PreTrainedFeatureExtractor"](
        "transformers.feature_extraction_utils.FeatureExtractionMixin"
    ).isinstance(
        fe
    )


def clean_name(name: str) -> str:
    return re.sub(r"\W|^(?=\d)-", "_", name)


def check_flax_supported() -> None:  # pragma: no cover
    _supported: bool = get_pkg_version("transformers").startswith("4")
    _flax_available = (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec("flax") is not None
    )
    if not _supported:
        logger.warning(
            "Detected transformers version: "
            f"{get_pkg_version('transformers')}, which "
            "doesn't have supports for Flax. "
            "Update `transformers` to 4.x and "
            "above to have Flax supported."
        )
    else:
        if _flax_available:
            _jax_version = get_pkg_version("jax")
            _flax_version = get_pkg_version("flax")
            logger.info(
                f"JAX version {_jax_version}, "
                f"Flax version {_flax_version} available."
            )
        else:
            logger.warning(
                "No versions of Flax or Jax are found under "
                "the current machine. In order to use "
                "Flax with transformers 4.x and above, "
                "refers to https://github.com/google/flax#quick-install"
            )


MODULE_NAME = "bentoml.transformers"


@inject
def load(
    tag: t.Union[str, Tag],
    from_tf: bool = False,
    from_flax: bool = False,
    *,
    return_config: bool = False,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **kwargs: t.Any,
) -> t.Union[
    "ext.TransformersPipeline",
    t.Tuple[
        "ext.PretrainedConfig",
        "ext.TransformersModelType",
        t.Union["ext.TransformersTokenizerType", "ext.PreTrainedFeatureExtractor"],
    ],
    t.Tuple[
        None,
        "ext.TransformersModelType",
        t.Union["ext.TransformersTokenizerType", "ext.PreTrainedFeatureExtractor"],
    ],
]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        from_tf (:code:`bool`, `optional`, defaults to :code:`False`):
            Load the model weights from a TensorFlow checkpoint save file.
        from_flax (:code:`bool`, `optional`, defaults to :code:`False`):
            Load the model weights from a Flax checkpoint save file
        return_config (:code:`bool`, `optional`, default to :code:`False`):
            Whether or not to return configuration of the Transformers model.
        config_kwargs (:code:`Dict[str, Any]`, `optional`):
            Kwargs to pass into :code:`Config` object.
        model_kwargs (:code:`Dict[str, Any]`, `optional`):
            Kwargs to pass into :code:`Model` object.
        tokenizer_kwargs (:code:`Dict[str, Any]`, `optional`):
            Kwargs to pass into :code:`Tokenizer` object.
        feature_extractor_kwargs (:code:`Dict[str, Any]`, `optional`):
            Kwargs to pass into :code:`FeatureExtractor` object.
        kwargs (:code:`Dict[str, Any]`, `optional`):
            Other kwargs that can be parsed to transformers that is neither configs, model, tokenizer, and feature extractor.

        .. warnings::
            Make sure to add the corresponding kwargs for your Transformers :code:`Model`, :code:`Tokenizer`, :code:`Config`, :code:`FeatureExtractor` to the correct kwargs dict.

        .. warnings::
            Currently :code:`kwargs` accepts all kwargs for corresponding Pipeline.

    Returns:
        :obj:`Union[Pipeline, Tuple[Optional[PretrainedConfig], Union[PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel], Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast, PreTrainedFeatureExtractor]]]]`: either returning a
        pipeline or a tuple containing :obj:`PretrainedConfig`, :obj:`Model` class object defined by :obj:`transformers`, with an optional :obj:`Tokenizer` class, or :obj:`FeatureExtractor` class for the given model saved in BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
        model, tokenizer = bentoml.transformers.load('custom_gpt2')

    If you want to returns an config object:

    .. code-block:: python

        import bentoml
        config, model, tokenizer = bentoml.transformers.load('custom_gpt2', return_config=True, tokenizer_kwargs={"use_fast":True})

    If the pipeline is saved with :code:`bentoml.transformers.save()`, then :code:`load()` will return pipeline objects:

    .. code-block:: python

        import bentoml
        pipeline = bentoml.transformers.load("roberta_text_classification", return_all_scores=True)
    """  # noqa
    check_flax_supported()  # pragma: no cover
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )

    if model.info.context["pipeline"]:
        _tasks = model.info.context["task"]
        return transformers.pipeline(_tasks, model.path, **kwargs)
    else:
        config_kwargs = kwargs.pop("config_kwargs", {})
        config: "ext.PretrainedConfig" = AutoConfig.from_pretrained(
            model.path, **config_kwargs
        )

        model_kwargs = kwargs.pop("model_kwargs", {})

        _model, _tokenizer = (
            model.info.options["model"],
            model.info.options["tokenizer"],
        )
        _feature_extractor = model.info.options["feature_extractor"]

        if _tokenizer is False:
            tokenizer: t.Optional["ext.TransformersTokenizerType"] = None
        else:
            tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})
            tokenizer = getattr(
                import_module("transformers"), _tokenizer
            ).from_pretrained(
                model.path, from_tf=from_tf, from_flax=from_flax, **tokenizer_kwargs
            )
        if _feature_extractor is False:
            feature_extractor: t.Optional["ext.PreTrainedFeatureExtractor"] = None
        else:
            feature_extractor_kwargs = kwargs.pop("feature_extractor_kwargs ", {})
            feature_extractor = getattr(
                import_module("transformers"), _feature_extractor
            ).from_pretrained(model.path, **feature_extractor_kwargs)

        tfe = tokenizer if tokenizer is not None else feature_extractor

        tmodel: "ext.TransformersModelType" = getattr(import_module("transformers"), _model).from_pretrained(  # type: ignore[reportUnknownMemberType]
            model.path,
            config=config,
            **model_kwargs,
        )

        if return_config:
            return config, tmodel, tfe  # type: ignore
        return None, tmodel, tfe  # type: ignore


def save(
    name: str,
    obj: t.Union["ext.TransformersModelType", "ext.TransformersPipeline"],  # type: ignore
    *,
    tokenizer: t.Optional["ext.TransformersTokenizerType"] = None,  # type: ignore
    feature_extractor: t.Optional["ext.PreTrainedFeatureExtractor"] = None,  # type: ignore
    labels: t.Optional[t.Dict[str, str]] = None,
    custom_objects: t.Optional[t.Dict[str, t.Any]] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        obj (:code:`Union[transformers.PreTrainedModel, transformers.TFPreTrainedModel, transformers.FlaxPreTrainedModel]`):
            Model/Pipeline instance provided by :obj:`transformers`. This can be retrieved from their
            :code:`AutoModel` class. You can also use any type of models/automodel provided
            by :obj:`transformers`. Refers to `Models API <https://huggingface.co/transformers/main_classes/model.html>`_
            for more information.
        tokenizer (:code:`Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]`, `optional`, default to `None`):
            Tokenizer instance provided by :obj:`transformers`. This can be retrieved from their
            their :code:`AutoTokenizer` class. You can also use any type of Tokenizer
            accordingly to your use case provided by :obj:`transformers`. Refers to
            `Tokenizer API <https://huggingface.co/docs/transformers/main_classes/tokenizer>`_
            for more information
        feature_extractor (:code:`transformers.PreTrainedFeatureExtractor`, `optional`, default to `None`):
            Feature Extractor instance provided by :obj:`transformers`. This can be retrieved from their
            their :code:`AutoFeatureExtractor` class. You can also use any type of Feature Extractor
            accordingly to your use case provided by :obj:`transformers`. Refers to
            `Feature Extractor API <https://huggingface.co/docs/transformers/main_classes/feature_extractor>`_
            for more information
        labels (:code:`Dict[str, str]`, `optional`, default to :code:`None`):
            user-defined labels for managing models, e.g. team=nlp, stage=dev
        custom_objects (:code:`Dict[str, Any]]`, `optional`, default to :code:`None`):
            user-defined additional python objects to be saved alongside the model,
            e.g. a tokenizer instance, preprocessor function, model configuration json
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
    Returns:
        :obj:`~bentoml.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        import bentoml
        model = AutoModelForQuestionAnswering.from_pretrained("gpt2", from_flax=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2", from_flax=True)

        # transfer training and modification goes here
        ...

        tag = bentoml.transformers.save("flax_gpt2", model=model, tokenizer=tokenizer)
    """
    if not check_model_type(obj):
        if not LazyType["ext.TransformersPipeline"](
            "transformers.pipelines.base.Pipeline"
        ).isinstance(obj):
            raise BentoMLException(
                "`model` is neither a Transformers pipeline nor a Transformers model."
            )

    check_flax_supported()  # pragma: no cover
    context: t.Dict[str, t.Any] = {
        "framework_name": "transformers",
        "pip_dependencies": [f"transformers=={get_pkg_version('transformers')}"],
        "pipeline": False,
        "task": False,
    }
    options: t.Dict[str, t.Any] = {
        "model": "",
        "tokenizer": False,
        "feature_extractor": False,
    }

    with bentoml.models.create(
        name,
        module=MODULE_NAME,
        context=context,
        options=options,
        labels=labels,
        custom_objects=custom_objects,
        metadata=metadata,
    ) as _model:
        if LazyType["ext.TransformersPipeline"](
            "transformers.pipelines.base.Pipeline"
        ).isinstance(obj):
            if tokenizer is not None or feature_extractor is not None:
                logger.warning(
                    "Currently saving a Transformers pipeline. Given params `tokenizer` or `feature_extractor` is useless."
                )
            obj.save_pretrained(_model.path)
            _model.info.context["pipeline"] = True
            _model.info.context["task"] = getattr(obj, "task")
            _model.info.options["model"] = getattr(obj, "model").__class__.__name__
            _tokenizer, _fe = getattr(obj, "tokenizer"), getattr(
                obj, "feature_extractor"
            )
            if getattr(obj, "feature_extractor") is not None:
                _model.info.options["feature_extractor"] = _fe.__class__.__name__
            elif check_tokenizer_type(_tokenizer):
                _model.info.options["tokenizer"] = _tokenizer.__class__.__name__
        elif check_model_type(obj):
            _model.info.options["model"] = obj.__class__.__name__
            obj.save_pretrained(_model.path)
            if tokenizer is not None:
                if not check_tokenizer_type(tokenizer):
                    raise BentoMLException(
                        "`tokenizer` is neither type `PreTrainedTokenizer` nor `PreTrainedTokenizerFast`"
                    )
                _model.info.options["tokenizer"] = tokenizer.__class__.__name__
                tokenizer.save_pretrained(_model.path)
            elif feature_extractor is not None:
                if not check_fe_type(feature_extractor):
                    raise BentoMLException(
                        "`feature_extractor` is not of type `PreTrainedFeatureExtractor`"
                    )
                _model.info.options[
                    "feature_extractor"
                ] = feature_extractor.__class__.__name__
                feature_extractor.save_pretrained(_model.path)
            else:
                logger.warning(
                    "Saving a Transformer model usually includes either a `tokenizer` or `feature_extractor`. None received."
                )
        else:
            raise BentoMLException(
                "Unknown type for `model`."
                f" Got {type(obj)} while only accepted"
                " one of the following types:"
                " `Pipeline` and `Union[PreTrainedModel,TFPreTrainedModel,FlaxPreTrainedModel]`"
            )

        return _model.tag


# TODO: import_from_huggingface_hub


class _TransformersRunner(BaseModelRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        tasks: str,
        *,
        framework: str,
        device: int,
        name: t.Optional[str] = None,
        **pipeline_kwargs: t.Any,
    ):
        super().__init__(tag=tag, name=name)

        try:
            transformers.pipelines.check_task(tasks)  # type: ignore
        except KeyError as e:
            raise BentoMLException(
                f"{e}, as `{tasks}` is not recognized by transformers."
            )
        self._tasks = tasks
        self._framework = framework
        self._device = device
        self._pipeline_kwargs = pipeline_kwargs

        # pipeline arguments
        self._feature_extractor = pipeline_kwargs.pop("feature_extractor", None)
        self._revision = pipeline_kwargs.pop("revision", None)  # type: t.Optional[str]
        self._use_fast = pipeline_kwargs.pop("use_fast", True)  # type: bool
        self._use_auth_token = pipeline_kwargs.pop(
            "use_auth_token", None
        )  # type: t.Optional[t.Union[str, bool]]
        self._model_kwargs = pipeline_kwargs.pop(
            "model_kwargs", {}
        )  # type: t.Dict[str, t.Any]
        self._kwargs = pipeline_kwargs

        # tokenizer-related
        try:
            self._has_tokenizer = (
                self.model_info.info.options["feature_extractor"] is False
            )
        except Exception:
            self._has_tokenizer = False
        self._tokenizer = None
        self._pipeline = None
        self._config = None

    @property
    def num_replica(self) -> int:
        # TODO: supports multiple GPUS
        return 1

    def _setup(self) -> None:
        params = load(
            self._tag,
            from_flax="flax" in self._framework,
            from_tf="tf" in self._framework,
            return_config=True,
            revision=self._revision,
            use_fast=self._use_fast,
            use_auth_token=self._use_auth_token,
            model_store=self.model_store,
            **self._kwargs,
        )

        if self.model_info.info.context["pipeline"]:
            self._pipeline = params
        else:
            self._config, self._model, _tfe = params
            if not self._has_tokenizer:
                self._feature_extractor = _tfe
            else:
                self._tokenizer = _tfe
            self._pipeline = transformers.pipeline(
                self._tasks,
                config=self._config,
                model=self._model,
                tokenizer=self._tokenizer,  # type: ignore[reportGeneralTypeIssues]
                framework=self._framework,
                feature_extractor=self._feature_extractor,
                revision=self._revision,
                use_fast=self._use_fast,
                use_auth_token=self._use_auth_token,
                model_kwargs=self._model_kwargs,
                device=self._device,
                **self._kwargs,
            )

    def _run_batch(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self._pipeline(*args, **kwargs)  # type: ignore


def load_runner(
    tag: t.Union[str, Tag],
    *,
    tasks: str,
    framework: str = "pt",
    device: int = -1,
    name: t.Optional[str] = None,
    **pipeline_kwargs: t.Any,
) -> "_TransformersRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. :func:`~bentoml.transformers.load_runner` implements a Runner class
    that wrap around a transformers pipeline, which optimize it for the BentoML runtime.

    .. warning::
       :func:`load_runner` will try to load the model from given :obj:`tag`. If the model does not
       exists, then BentoML will fallback to initialize pipelines from transformers,
       thus files will be loaded from huggingface cache.

    Args:
        tag (:code:`Union[str, Tag]`):
            Tag of a saved model in BentoML local modelstore.
        tasks (:code:`str`):
            Given tasks for pipeline. Refers to `Task Summary <https://huggingface.co/transformers/task_summary.html>`_
            for more information.
        framework (:code:`str`, default to :code:`pt`):
            Given frameworks supported by transformers: PyTorch, Tensorflow
        device (`int`, `optional`, default to :code:`-1`):
            Default GPU devices to be used by runner.
        **pipeline_kwargs(`Any`):
            Refers to `Pipeline Docs <https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline>`_ for more information
            on :obj:`kwargs` that is applicable for your specific pipeline.

    Returns:
        :obj:`~bentoml._internal.runner.Runner`: Runner instances for :mod:`bentoml.transformers` model

    Examples:

    .. code-block:: python

        import transformers
        import bentoml
        runner = bentoml.transformers.load_runner("gpt2:latest", tasks='zero-shot-classification', framework=tf)
        runner.run_batch(["In today news, ...", "The stocks market seems ..."])
    """
    return _TransformersRunner(
        tag=tag,
        tasks=tasks,
        framework=framework,
        device=device,
        name=name,
        **pipeline_kwargs,
    )
