import re
import json
import typing as t
import logging
import importlib.util
from typing import TYPE_CHECKING
from importlib import import_module

from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Model
from bentoml import Runner
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..types import LazyType
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers.configuration_utils import PretrainedConfig
    from transformers.models.auto.auto_factory import (
        _BaseAutoModelClass,  # type: ignore[reportPrivateUsage]
    )

    from .. import ext_typing as ext
    from ..models import ModelStore

try:
    import transformers
    from transformers import AutoModel
    from transformers import AutoConfig
    from transformers import TFAutoModel
    from transformers import AutoTokenizer
    from transformers import FlaxAutoModel
    from transformers import AutoFeatureExtractor
    from transformers.file_utils import http_get
    from transformers.file_utils import CONFIG_NAME
    from transformers.file_utils import WEIGHTS_NAME
    from transformers.file_utils import hf_bucket_url
    from transformers.file_utils import http_user_agent
    from transformers.file_utils import TF2_WEIGHTS_NAME
    from transformers.file_utils import FLAX_WEIGHTS_NAME
except ImportError:  # pragma: no cover
    raise MissingDependencyException(
        """\
        transformers is required in order to use module `bentoml.transformers`.
        Instruction: Install transformers with `pip install transformers`.
        """
    )


_hfhub_exc = """\
`huggingface_hub` is required to use `bentoml.transformers.import_from_huggingface_hub()`.
Instruction: `pip install huggingface_hub`
"""


_FRAMEWORK_ALIASES: t.Dict[str, str] = {"pt": "pytorch", "tf": "tensorflow"}

_AUTOMODEL_PREFIX_MAPPING: t.Dict[str, str] = {
    "pytorch": "AutoModel",
    "tensorflow": "TFAutoModel",
    "flax": "FlaxAutoModel",
}

_AUTOMODEL_LM_HEAD_MAPPING: t.Dict[str, str] = {
    "causal": "ForCausalLM",
    "masked": "ForMaskedLM",
    "seq2seq": "ForSeq2SeqLM",
    "sequence-classification": "ForSequenceClassification",
    "question-answering": "ForQuestionAnswering",
    "token-classification": "ForTokenClassification",
    "multiple-choice": "ForMultipleChoice",
    "next-sentence-prediction": "ForNextSentencePrediction",
    "image-classification": "ForImageClassification",
    "audio-classification": "ForAudioClassification",
    "ctc": "ForCTC",
    "speech-seq2seq": "ForSpeechSeq2Seq",
    "object-detection": "ForObjectDetection",
}


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


def load_autoclass(framework: str, lm_head: str) -> "_BaseAutoModelClass":
    """Getting transformers Auto Classes from given frameworks and lm_head"""
    if (
        framework not in _AUTOMODEL_PREFIX_MAPPING
        and framework not in _FRAMEWORK_ALIASES
    ):
        raise AttributeError(
            f"{framework} is either invalid aliases "
            "or not supported by transformers. "
            "Accepted: pt(alias to pytorch), "
            "tf(alias to tensorflow), and flax."
        )
    if lm_head not in _AUTOMODEL_LM_HEAD_MAPPING:
        raise AttributeError(
            f"`{lm_head}` alias for lm_head is invalid."
            f" Accepted: {[*_AUTOMODEL_LM_HEAD_MAPPING.keys()]}."
            " If you need any other AutoModel type provided by transformers,"
            " feel free to open a PR at https://github.com/bentoml/BentoML."
        )
    framework_prefix = (
        _FRAMEWORK_ALIASES[framework] if framework in _FRAMEWORK_ALIASES else framework
    )
    class_inst = f"{_AUTOMODEL_PREFIX_MAPPING[framework_prefix]}{_AUTOMODEL_LM_HEAD_MAPPING[lm_head]}"  # noqa
    try:
        return getattr(import_module("transformers"), class_inst)
    except AttributeError as e:
        raise BentoMLException(
            f"{e}\n\nPlease refers "
            f"to https://huggingface.co/transformers/model_doc/auto.html."
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

_SAVE_CONFLICTS_ERR = """\
When `tokenizer={tokenizer}`, model should be of type
Union[`PreTrainedModel`, `TFPreTrainedModel`, `FlaxPreTrainedModel`]. Currently
`type(model)={model}`

If you want to save the weight directly from
`transformers` and save it to BentoML do:
    `bentoml.transformers.import_from_huggingface_hub('bert-base-uncased')`.

If you are training a model from scratch using transformers, to save into BentoML do:
    `bentoml.transformers.save('bert_model', model=my_bert_model, tokenizer=my_tokenizer)`

If you want to import directly from a `transformers.pipeline` then do:
    # pipeline = transformers.pipelines('sentiment-analysis')
    `bentoml.transformers.save("senta-pipe", pipeline)
"""


@inject
def load(
    tag: t.Union[str, Tag],
    from_tf: bool = False,
    from_flax: bool = False,
    framework: str = "pt",
    lm_head: str = "causal",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **kwargs: str,
) -> t.Tuple[
    "PretrainedConfig",
    "ext.TransformersModelType",
    t.Optional["ext.TransformersTokenizerType"],
    t.Optional["ext.PreTrainedFeatureExtractor"],
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
        framework (:code:`str`, default to :code:`pt`):
            Given frameworks supported by transformers: PyTorch, Tensorflow, Flax
        lm_head (:code:`str`, default to :code:`causal`):
            Language model head for your model. For most use cases causal are applied.
            Refers to `transformers <https://huggingface.co/docs/transformers/index>`_ for more details on which type of
            language model head is applied to your use case and model.
        kwargs (:code:`str`, `optional`):
            kwargs that can be parsed to transformers Models instance.

    Returns:
        :obj:`Tuple[PretrainedConfig, Union[PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel], Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]]]`: a tuple containing
        :obj:`PretrainedConfig`, :obj:`Model` class object defined by :obj:`transformers`, with an optional :obj:`Tokenizer` class, or :obj:`FeatureExtractor` class for the given model saved in BentoML modelstore.

    Examples:

    .. code-block:: python

        import bentoml
        config, model, tokenizer, _ = bentoml.transformers.load('custom_gpt2', framework="pt", lm_head="causal")
    """  # noqa
    check_flax_supported()  # pragma: no cover
    model = model_store.get(tag)
    if model.info.module not in (MODULE_NAME, __name__):
        raise BentoMLException(
            f"Model {tag} was saved with module {model.info.module}, failed loading with {MODULE_NAME}."
        )

    config, unused_kwargs = AutoConfig.from_pretrained(
        model.path, return_unused_kwargs=True, **kwargs
    )  # type: ignore[reportUnknownMemberType]

    _model, _tokenizer = model.info.context["model"], model.info.context["tokenizer"]
    _feature_extractor = model.info.context["feature_extractor"]

    if _tokenizer is False:
        tokenizer: t.Optional["ext.TransformersTokenizerType"] = None
    else:
        tokenizer = getattr(import_module("transformers"), _tokenizer).from_pretrained(
            model.path, from_tf=from_tf, from_flax=from_flax
        )
    if _feature_extractor is False:
        feature_extractor: t.Optional["ext.PreTrainedFeatureExtractor"] = None
    else:
        feature_extractor = getattr(
            import_module("transformers"), _feature_extractor
        ).from_pretrained(model.path)

    try:
        # Cover cases where some model repo doesn't include a model
        #  name under their config.json. An example is
        #  google/bert_uncased_L-2_H-128_A-2
        t_model = load_autoclass(framework, lm_head).from_pretrained(  # type: ignore[reportUnknownMemberType]
            model.path, config=config, **unused_kwargs
        )
    except Exception:  # noqa # pylint: disable=broad-except
        if feature_extractor is not None:
            if from_tf:
                framework = "tf"
            elif from_flax:
                framework = "flax"
            else:
                framework = "pt"
            loader = load_autoclass(framework, lm_head)
        else:
            loader = getattr(import_module("transformers"), _model)
        t_model: "ext.TransformersModelType" = loader.from_pretrained(  # type: ignore[reportUnknownMemberType]
            model.path,
            config=config,
            **unused_kwargs,
        )
    return config, t_model, tokenizer, feature_extractor


def _save(
    name: str,
    *,
    model_identifier: t.Union[
        str, "ext.TransformersModelType", "ext.TransformersPipeline"
    ],
    tokenizer: t.Optional["ext.TransformersTokenizerType"],
    metadata: t.Optional[t.Dict[str, t.Any]],
    keep_download_from_hub: bool,
    model_store: "ModelStore",
    **transformers_options_kwargs: str,
) -> Tag:

    # AutoConfig kwargs options
    cache_dir = transformers_options_kwargs.pop("cache_dir", None)
    force_download = transformers_options_kwargs.pop("force_download", False)
    resume_download = transformers_options_kwargs.pop("resume_download", True)
    proxies = transformers_options_kwargs.pop("proxies", None)
    revision = transformers_options_kwargs.pop("revision", "main")
    trust_remote_code = transformers_options_kwargs.pop("trust_remote_code", False)

    from_tf = transformers_options_kwargs.pop("from_tf", False)
    from_flax = transformers_options_kwargs.pop("from_flax", False)
    use_auth_token = transformers_options_kwargs.pop("use_auth_token", None)

    # AutoTokenizer kwargs options
    subfolder = transformers_options_kwargs.pop("subfolder", None)
    use_fast = transformers_options_kwargs.pop("use_fast", True)
    tokenizer_type = transformers_options_kwargs.pop("tokenizer_type", None)

    check_flax_supported()  # pragma: no cover
    context: t.Dict[str, t.Any] = {
        "framework_name": "transformers",
        "pip_dependencies": [f"transformers=={get_pkg_version('transformers')}"],
        "feature_extractor": False,
        "pipeline": False,
        "tokenizer": False,
    }
    options: t.Dict[str, t.Any] = {"revision": revision}

    _model = Model.create(
        name,
        module=MODULE_NAME,
        context=context,
        options=options,
        metadata=metadata,
    )

    if from_tf:
        automodel_cls = TFAutoModel
    elif from_flax:
        automodel_cls = FlaxAutoModel
    else:
        automodel_cls = AutoModel

    if isinstance(model_identifier, str):
        try:
            meta = model_store.get(name)
            rev = meta.info.options["revision"]
            if rev == "main" and not keep_download_from_hub:
                logger.warning(
                    f"{name} is found under BentoML modelstore.\nFor most use cases of using pretrained model,"
                    f" you don't have to re-download the model. returning {meta.tag} ...\nIf you"
                    " still insist on downloading, then specify `keep_download_from_hub=True` in"
                    " `import_from_huggingface_hub`."
                )
                return meta.tag
            else:
                pass
        except (NotFound, FileNotFoundError):
            pass

        # Load the model from pretrained
        # This is pretty slow approach
        model = automodel_cls.from_pretrained(
            model_identifier,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            revision=revision,
            **transformers_options_kwargs,
        )
        model.save_pretrained(_model.path)

        with open(_model.path_of(CONFIG_NAME), "r", encoding="utf-8") as of:
            _config = json.loads(of.read())
            try:
                arch = "".join(_config["architectures"])
                if from_tf:
                    _model.info.context["model"] = f"TF{arch}"
                elif from_flax:
                    _model.info.context["model"] = f"Flax{arch}"
                else:
                    _model.info.context["model"] = arch
            except KeyError:
                _model.info.context["model"] = ""

        # NOTE: With Tokenizer there are way too many files
        #  to be included, per frameworks. Thus we will load
        #  a Tokenizer instance, then save it to path.
        try:
            _tokenizer: "ext.TransformersTokenizerType" = AutoTokenizer.from_pretrained(
                # type: ignore[reportUnknownMemberType]
                model_identifier,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                revision=revision,
                subfolder=subfolder,
                use_fast=use_fast,
                tokenizer_type=tokenizer_type,
                trust_remote_code=trust_remote_code,
                **transformers_options_kwargs,
            )
            _ = _tokenizer.save_pretrained(_model.path)  # type: ignore[reportUnknownMemberType]
            _model.info.context["tokenizer"] = type(_tokenizer).__name__
        except (ValueError, KeyError):
            # For pretrained model that doesn't have a tokenizer,
            # it will have a feature extractor instead
            _feature_extractor: "ext.PreTrainedFeatureExtractor" = (
                AutoFeatureExtractor.from_pretrained(
                    model_identifier,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    revision=revision,
                    use_auth_token=use_auth_token,
                    **transformers_options_kwargs,
                )
            )
            _feature_extractor.save_pretrained(_model.path)  # type: ignore[reportUnknownMemberType]
            _model.info.context["feature_extractor"] = type(_feature_extractor).__name__
    elif LazyType["ext.TransformersPipeline"](
        "transformers.pipelines.base.Pipeline"
    ).isinstance(model_identifier):
        _model.info.context["model"] = model_identifier.model.__class__.__name__
        _model.info.context["tokenizer"] = model_identifier.tokenizer.__class__.__name__
        _model.info.context["pipeline"] = True
        model_identifier.save_pretrained(_model.path)
    elif check_model_type(model_identifier):
        assert tokenizer is not None and check_tokenizer_type(
            tokenizer
        ), _SAVE_CONFLICTS_ERR.format(tokenizer=tokenizer, model=type(model_identifier))
        _model.info.context["model"] = model_identifier.__class__.__name__
        _model.info.context["tokenizer"] = tokenizer.__class__.__name__
        model_identifier.save_pretrained(_model.path)
        tokenizer.save_pretrained(_model.path)
    else:
        raise BentoMLException(
            "Unknown type for `model_identifier`."
            f" Got {type(model_identifier)} while only accepted"
            " one of the following three type: `str`,"
            " `Pipeline` and `Union[PreTrainedModel,TFPreTrainedModel,FlaxPreTrainedModel]`"
        )

    _model.save(model_store)
    return _model.tag


@inject
def save(
    name: str,
    *,
    model: t.Union["ext.TransformersModelType", "ext.TransformersPipeline"],
    tokenizer: t.Optional["ext.TransformersTokenizerType"] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **transformers_options_kwargs: str,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (:code:`str`):
            Name for given model instance. This should pass Python identifier check.
        model (:code:`Union[transformers.PreTrainedModel, transformers.TFPreTrainedModel, transformers.FlaxPreTrainedModel]`):
            Model instance provided by :obj:`transformers`. This can be retrieved from their
            :code:`AutoModel` class. You can also use any type of models/automodel provided
            by :obj:`transformers`. Refers to `Models API <https://huggingface.co/transformers/main_classes/model.html>`_
            for more information.
        tokenizer (:code:`Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast]`):
            Toeknizer instance provided by :obj:`transformers`. This can be retrieved from their
            their :code:`AutoTokenizer` class. You can also use any type of Tokenizer
            accordingly to your use case provided by :obj:`transformers`. Refers to
            `Tokenizer API <https://huggingface.co/transformers/main_classes/tokenizer.html>`_
            for more information
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        from_tf (:code:`bool`, `optional`, defaults to :code:`False`):
            Load the model weights from a TensorFlow checkpoint save file
        from_flax (:code:`bool`, `optional`, defaults to :code:`False`):
            Load the model weights from a Flax checkpoint save file
        revision (:code:`str`, `optional`, defaults to :code:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a
            commit id, since we use a git-based system for storing models and other
            artifacts on *huggingface.co*, so ``revision`` can be any
            identifier allowed by git.
        mirror (:code:`str`, `optional`):
            Mirror source to accelerate downloads in China. If you are from China and
            have an accessibility problem, you can set this option to resolve it. Note
            that we do not guarantee the timeliness or safety. Please refer to the
            mirror site for more information.
        proxies (:code:`Dict[str, str]`, `optional`):
            A dictionary of proxy servers to use by protocol or endpoint, e.g.
            :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The
            proxies are used on each request.
        use_auth_token (:code:`str` or :code:`bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If
            :obj:`True`, will use the token generated when running
            :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
        force_download (:code:`bool`, `optional`, defaults to :code:`False`):
            Whether or not to force the (re-)download of the model weights and
            configuration files, overriding the cached versions if they exist.
        resume_download (:code:`bool`, `optional`, defaults to :code:`False`):
            Whether or not to delete incompletely received files. Will attempt to resume
            the download if such a file exists.

    .. note::


       Some parameters are direct port from :func:`from_pretrained` arguments. This is to
       ensure that when doing :func:`save` operations we don't actually load the model class into memory

    .. warning::


        :code:`save` should **ONLY BE USED** when training or working with a customized pretrained transformers model.
        Otherwise, use :code:`import_from_huggingface_hub` for most usecases.

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        import bentoml

        model = AutoModelForQuestionAnswering.from_pretrained("gpt2", from_flax=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2", from_flax=True)
        # custom training and modification goes here

        tag = bentoml.transformers.save("flax_gpt2", model=model, tokenizer=tokenizer)
    """
    if not check_model_type(model):
        if not LazyType["ext.TransformersPipeline"](
            "transformers.pipelines.base.Pipeline"
        ).isinstance(model):
            raise BentoMLException(
                "If you want to import model directly from huggingface hub"
                " please use `import_from_huggingface_hub` instead. `save` should"
                " only be used for saving custom pretrained model and tokenizer"
            )
    return _save(
        name=name,
        model_identifier=model,
        tokenizer=tokenizer,
        metadata=metadata,
        keep_download_from_hub=False,
        model_store=model_store,
        **transformers_options_kwargs,
    )


@inject
def import_from_huggingface_hub(
    name: str,
    *,
    save_namespace: t.Optional[str] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    keep_download_from_hub: bool = False,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **transformers_options_kwargs: str,
) -> Tag:
    """
    Import a model from hugging face hub and save it to bentoml modelstore.

    Args:
        name (:code:`str`):
            Model name retrieved from HuggingFace Model hub. This shouldn't be a model
            instance. If you would like to save a model instance refers to
            :func:`~bentoml.transformers.save` for more information.
        save_namespace (:code:`str`, default to given `name`):
            Name to save model to BentoML modelstore.
        metadata (:code:`Dict[str, Any]`, `optional`,  default to :code:`None`):
            Custom metadata for given model.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        keep_download_from_hub (`bool`, `optional`, default to :code:`False`):
            Whether to re-download pretrained model from hub.
        from_tf (`bool`, `optional`, defaults to `False`):
            Load the model weights from a TensorFlow checkpoint save file
        from_flax (`bool`, `optional`, defaults to `False`):
            Load the model weights from a Flax checkpoint save file
        revision(:code:`str`, `optional`, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a
            commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so ``revision`` can be any identifier allowed
            by git.
        mirror(:code:`str`, `optional`):
            Mirror source to accelerate downloads in China. If you are from China and
            have an accessibility problem, you can set this option to resolve it. Note
            that we do not guarantee the timeliness or safety. Please refer to the
            mirror site for more information.
        proxies (:code:`Dict[str, str], `optional`):
            A dictionary of proxy servers to use by protocol or endpoint, e.g.
            :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The
            proxies are used on each request.
        use_auth_token (:code:`str` or `bool`, `optional`):
            The token to use as HTTP bearer authorization for remote files. If
            :obj:`True`, will use the token generated when running
            :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
        force_download (`bool`, `optional`, defaults to `False`):
            Whether or not to force the (re-)download of the model weights and
            configuration files, overriding the cached versions if they exist.
        resume_download (`bool`, `optional`, defaults to `False`):
            Whether or not to delete incompletely received files. Will attempt to resume
            the download if such a file exists.
        subfolder (:code:`str`, *optional*):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co (e.g. for
            facebook/rag-token-base), specify it here.
        use_fast (:code:`bool`, *optional*, defaults to :code:`True`):
            Whether or not to try to load the fast version of the tokenizer.
        tokenizer_type (:code:`str`, *optional*):
            Tokenizer type to be loaded.
        trust_remote_code (:code:`bool`, *optional*, defaults to `False`):
            Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.

    .. note::


       Some parameters are direct port from :func:`from_pretrained()` arguments. This
       ensures that when doing :func:`save` operations we don't actually load the model
       class into memory

    Returns:
        :obj:`~bentoml._internal.types.Tag`: A :obj:`tag` with a format `name:version` where `name` is the user-defined model's name, and a generated `version` by BentoML.

    Examples:

    .. code-block:: python

        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        import bentoml

        tag = bentoml.transformers.import_from_huggingface_hub("gpt2", from_tf=True)
    """
    # TODO: pass down kwargs for AutoTokenizer
    save_namespace = clean_name(name) if save_namespace is None else save_namespace
    return _save(
        name=save_namespace,
        model_identifier=name,
        tokenizer=None,
        keep_download_from_hub=keep_download_from_hub,
        metadata=metadata,
        model_store=model_store,
        **transformers_options_kwargs,
    )


class _TransformersRunner(Runner):
    @inject
    def __init__(
        self,
        tag: Tag,
        tasks: str,
        *,
        framework: str,
        lm_head: str,
        device: int,
        name: str,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
        **pipeline_kwargs: t.Any,
    ):
        in_store_tag = model_store.get(tag).tag
        self._tag = in_store_tag
        super().__init__(name, resource_quota, batch_options)

        try:
            transformers.pipelines.check_task(tasks)
        except KeyError as e:
            raise BentoMLException(
                f"{e}, as `{tasks}` is not recognized by transformers."
            )
        self._tasks = tasks
        self._model_store = model_store
        self._framework = framework
        self._lm_head = lm_head
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

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]

    @property
    def num_concurrency(self) -> int:
        return self.num_replica

    @property
    def num_replica(self) -> int:
        # TODO: supports multiple GPUS
        return 1

    # pylint: disable=arguments-differ,attribute-defined-outside-init
    def _setup(self) -> None:
        try:
            _ = self._model_store.get(self._tag)
            self._config, self._model, self._tokenizer, self._feature_extractor = load(
                self._tag,
                model_store=self._model_store,
                from_flax=False,
                from_tf="tf" in self._framework,
                framework=self._framework,
                lm_head=self._lm_head,
            )
        except FileNotFoundError:
            self._config, self._model, self._tokenizer = None, None, None
        if self._tokenizer is None:
            self._pipeline: "ext.TransformersPipeline" = transformers.pipeline(
                self._tasks
            )
        else:
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

    # pylint: disable=arguments-differ
    def _run_batch(
        self, input_data: t.Union[t.Any, t.List[t.Any]]
    ) -> t.Union[t.Any, t.List[t.Any]]:  # type: ignore[override]  # noqa
        res: t.Any = self._pipeline(input_data)
        return res


def load_runner(
    tag: t.Union[str, Tag],
    *,
    tasks: str,
    framework: str = "pt",
    lm_head: str = "causal",
    device: int = -1,
    name: t.Optional[str] = None,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
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
        lm_head (:code:`str`, default to :code:`causal`):
            Language model attention head for your model. For most use case :obj:`causal` are applied.
            Refers to `HuggingFace Docs <https://huggingface.co/docs/transformers/main_classes/model>`_
            for more details on which type of language model head is applied to your use case.
        resource_quota (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Dictionary to configure resources allocation for runner.
        batch_options (:code:`Dict[str, Any]`, `optional`, default to :code:`None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (:mod:`~bentoml._internal.models.store.ModelStore`, default to :mod:`BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
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
    tag = Tag.from_taglike(tag)
    if name is None:
        name = tag.name
    return _TransformersRunner(
        tag=tag,
        tasks=tasks,
        framework=framework,
        lm_head=lm_head,
        device=device,
        name=name,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
        **pipeline_kwargs,
    )
