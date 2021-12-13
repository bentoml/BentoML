import os
import re
import json
import typing as t
import logging
import tempfile
import functools
import importlib.util
from typing import TYPE_CHECKING
from pathlib import Path
from importlib import import_module
from contextlib import contextmanager

import requests
from filelock import FileLock
from simple_di import inject
from simple_di import Provide

from bentoml import Tag
from bentoml import Runner
from bentoml.exceptions import NotFound
from bentoml.exceptions import BentoMLException
from bentoml.exceptions import MissingDependencyException

from ..models import Model
from ..models import JSON_EXT
from ..utils.pkg import get_pkg_version
from ..configuration.containers import BentoMLContainer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from transformers import PretrainedConfig
    from transformers import TFPreTrainedModel
    from transformers import FlaxPreTrainedModel
    from transformers import PreTrainedTokenizer
    from transformers import PreTrainedTokenizerFast
    from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    from ..models import ModelStore

try:
    import transformers
    from transformers import Pipeline
    from transformers import AutoConfig
    from transformers import AutoTokenizer
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

_transformers_version = get_pkg_version("transformers")

try:
    from huggingface_hub import HfFolder
except ImportError:
    HfFolder = None

_hfhub_exc = """\
`huggingface_hub` is required to use `bentoml.transformers.import_from_huggingface_hub()`.
Instruction: `pip install huggingface_hub`
"""

_T = t.TypeVar("_T")
_F = t.TypeVar("_F", bound=t.Callable[..., t.Any])
_T_co = t.TypeVar("_T_co", covariant=True)

_ModelType = t.Union["PreTrainedModel", "TFPreTrainedModel", "FlaxPreTrainedModel"]
_TokenizerType = t.Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]

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


class _GeneratorContextManager(t.ContextManager[_T_co]):
    def __call__(self, func: _F) -> _F:
        ...


def _load_autoclass(framework: str, lm_head: str) -> "_BaseAutoModelClass":
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


def _clean_name(name: str) -> str:
    return re.sub(r"\W|^(?=\d)-", "_", name)


def _check_flax_supported() -> None:  # pragma: no cover
    _supported: bool = _transformers_version.startswith("4")
    _flax_available = (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec("flax") is not None
    )
    if not _supported:
        logger.warning(
            "Detected transformers version: "
            f"{_transformers_version}, which "
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
    `bentoml.transoformers.save("senta-pipe", pipeline)
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
    t.Union["PreTrainedModel", "TFPreTrainedModel", "FlaxPreTrainedModel"],
    t.Optional[t.Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]],
]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        from_tf (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Load the model weights from a TensorFlow checkpoint save file
        from_flax (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Load the model weights from a Flax checkpoint save file
        framework (`str`, default to `pt`):
            Given frameworks supported by transformers: PyTorch, Tensorflow, Flax
        lm_head (`str`, default to `causal`):
            Language model head for your model. For most usecase causal are applied.
             Refers to huggingface.co/transformers for more details on which type of
             language model head is applied to your use case and model.
        kwargs (:obj:`str`, `Optional`):
            kwargs that can be parsed to transformers Models instance.

    Returns:
        a Tuple containing `model` and `tokenizer` for your given model saved at BentoML
         modelstore.

    Examples::
        import bentoml.transformers
        model, tokenizer = bentoml.transformers.load('custom_gpt2', framework="flax",
                                                     lm_head="masked")
    """
    _check_flax_supported()  # pragma: no cover
    model = model_store.get(tag)
    _model, _tokenizer = model.info.options["model"], model.info.options["tokenizer"]

    if _tokenizer != "na":
        tokenizer = getattr(import_module("transformers"), _tokenizer).from_pretrained(
            model.path, from_tf=from_tf, from_flax=from_flax
        )
    else:
        tokenizer = None
    config = AutoConfig.from_pretrained(model.path)

    try:
        # Cover cases where some model repo doesn't include a model
        #  name under their config.json. An example is
        #  google/bert_uncased-L-2-H-128-A-2
        t_model = _load_autoclass(framework, lm_head).from_pretrained(
            model.path,
            config=config,
            **kwargs,
        )
    except (AttributeError, BentoMLException):  # noqa
        t_model = getattr(import_module("transformers"), _model).from_pretrained(
            model.path,
            config=config,
            from_tf=from_tf,
            from_flax=from_flax,
            **kwargs,
        )
    return config, t_model, tokenizer


def _download_from_hub(
    hf_url: str,
    output_dir: str,
    force_download: bool = False,
    proxies: t.Union[str, None] = None,
    etag_timeout: int = 10,
    resume_download: bool = False,
    user_agent: t.Union[t.Dict[str, str], str, None] = None,
    use_auth_token: t.Union[bool, str, None] = None,
) -> None:
    """
    Modification of https://github.com/huggingface/transformers/blob/master/src/transformers/file_utils.py
    """
    if HfFolder is None:
        raise BentoMLException(_hfhub_exc)
    headers = {"user-agent": http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError(
                "You specified use_auth_token=True, "
                "but a huggingface token was not found."
            )
        headers["authorization"] = f"Bearer {token}"

    url_to_download = hf_url
    etag = None
    try:
        r = requests.head(
            hf_url,
            headers=headers,
            allow_redirects=False,
            proxies=proxies,
            timeout=etag_timeout,
        )
        r.raise_for_status()
        etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
        # We favor a custom header indicating the etag of the linked resource, and
        # we fallback to the regular etag header.
        # If we don't have any of those, raise an error.
        if etag is None:
            raise OSError(
                "Distant resource does not have an ETag, "
                "we won't be able to reliably ensure reproducibility."
            )
        # In case of a redirect,
        # save an extra redirect on the request.get call,
        # and ensure we download the exact atomic version even if it changed
        # between the HEAD and the GET (unlikely, but hey).
        if 300 <= r.status_code <= 399:
            url_to_download = r.headers["Location"]
    except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
        # Actually raise for those subclasses of ConnectionError
        raise
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Otherwise, our Internet connection is down.
        # etag is None
        pass
    fname = Path(output_dir, hf_url.split("/")[-1])
    logger.info(f"Downloading {hf_url} to {str(fname)}")
    fpath = Path(output_dir, fname)
    # From now on, etag is not None.
    if fpath.exists() and not force_download:
        return

    # Prevent parallel downloads of the same file with a lock.
    lock_path = fpath.with_suffix(".lock")
    with FileLock(str(lock_path)):

        # If the download just completed while the lock was activated.
        if os.path.exists(fpath) and not force_download:
            # Even if returning early like here, the lock will be released.
            return

        _tmp_file_manager: t.Union[
            t.Callable[[], _GeneratorContextManager[t.BinaryIO]],
            t.Callable[[t.Any, t.Any], t.IO[str]],
        ]

        if resume_download:
            incomplete_path = fpath.with_suffix(".incomplete")

            @contextmanager
            def _resume_file_manager() -> t.Generator[t.BinaryIO, None, None]:
                with open(incomplete_path, "ab") as f:
                    yield f

            _tmp_file_manager = _resume_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            _tmp_file_manager = functools.partial(
                tempfile.NamedTemporaryFile, mode="wb", dir=output_dir, delete=False
            )
            resume_size = 0
    with _tmp_file_manager() as temp_file:
        http_get(
            url_to_download,
            temp_file,
            proxies=proxies,
            resume_size=resume_size,
            headers=headers,
        )

    logger.info(f"storing {hf_url} at {output_dir}")
    os.replace(temp_file.name, str(fname))

    # NamedTemporaryFile creates a file with
    #  hardwired 0600 perms (ignoring umask), so fixing it.
    umask = os.umask(0o666)
    os.umask(umask)
    os.chmod(fname, 0o666 & ~umask)
    logger.info(f"creating metadata file for {fpath.name}")
    meta = {"url": hf_url, "etag": etag}
    meta_path = fpath.with_suffix(f".metadata{JSON_EXT}")
    with meta_path.open("w") as meta_file:
        json.dump(meta, meta_file)


def _save(
    name: str,
    *,
    model_identifier: t.Union[str, _ModelType, "Pipeline"],
    tokenizer: t.Optional[_TokenizerType],
    metadata: t.Optional[t.Dict[str, t.Any]],
    keep_download_from_hub: bool,
    model_store: "ModelStore",
    **transformers_options_kwargs: str,
) -> Tag:
    _check_flax_supported()  # pragma: no cover
    context: t.Dict[str, t.Any] = {
        "framework_name": "transformers",
        "pip_dependencies": [f"transformers=={_transformers_version}"],
    }

    if isinstance(model_identifier, str):
        try:
            info = model_store.get(name)
            if not keep_download_from_hub:
                logger.warning(
                    f"{name} is found under BentoML modelstore.\nFor most usecases of using pretrained model,"
                    f" you don't have to redownload the model. returning {info.tag}...\nIf you still insist on downloading,"
                    " then specify `keep_download_from_hub=True` in `import_from_huggingface_hub`"
                )
                return info.tag
            else:
                pass
        except (NotFound, FileNotFoundError):
            pass

    _model = Model.create(
        name,
        module=__name__,
        context=context,
        options=None,
        metadata=metadata,
    )

    if tokenizer is not None:
        assert not isinstance(model_identifier, str) or isinstance(
            model_identifier, Pipeline
        ), _SAVE_CONFLICTS_ERR.format(tokenizer=tokenizer, model=type(model_identifier))
        _model.info.options = {
            "model": model_identifier.__class__.__name__,
            "tokenizer": tokenizer.__class__.__name__,
            "pipeline": False,
        }
        model_identifier.save_pretrained(_model.path)
        tokenizer.save_pretrained(_model.path)
    else:
        if isinstance(model_identifier, Pipeline):
            # model_identifier is a Pipeline
            _model.info.options = {
                "model": model_identifier.model.__class__.__name__,
                "tokenizer": model_identifier.tokenizer.__class__.__name__,
                "pipeline": True,
            }
            model_identifier.save_pretrained(_model.path)
        else:
            from_tf = transformers_options_kwargs.pop("from_tf", False)
            from_flax = transformers_options_kwargs.pop("from_flax", False)
            revision = transformers_options_kwargs.pop("revision", None)
            mirror = transformers_options_kwargs.pop("mirror", None)
            proxies = transformers_options_kwargs.pop("proxies", None)
            use_auth_token = transformers_options_kwargs.pop("use_auth_token", None)
            force_download = bool(
                transformers_options_kwargs.pop("force_download", False)
            )
            resume_download = bool(
                transformers_options_kwargs.pop("resume_download", True)
            )

            # download config file
            config_file_url = hf_bucket_url(
                model_identifier,
                filename=CONFIG_NAME,
                revision=revision,
                mirror=mirror,
            )
            _download_from_hub(
                config_file_url,
                output_dir=_model.path,
                proxies=proxies,
                use_auth_token=use_auth_token,
                force_download=force_download,
                resume_download=resume_download,
            )

            # download weight file set correct filename
            if from_tf:
                weight_filename = TF2_WEIGHTS_NAME
            elif from_flax:
                weight_filename = FLAX_WEIGHTS_NAME
            else:
                weight_filename = WEIGHTS_NAME

            weight_file_url = hf_bucket_url(
                model_identifier,
                filename=weight_filename,
                revision=revision,
                mirror=mirror,
            )
            _download_from_hub(
                weight_file_url,
                output_dir=_model.path,
                proxies=proxies,
                use_auth_token=use_auth_token,
                force_download=force_download,
                resume_download=resume_download,
            )

            # _model.info.options will be None here
            with open(_model.path_of(CONFIG_NAME), "r", encoding="utf-8") as of:
                _config = json.loads(of.read())
                try:
                    arch = "".join(_config["architectures"])
                    if from_tf:
                        _model.info.options["model"] = f"TF{arch}"
                    elif from_flax:
                        _model.info.options["model"] = f"Flax{arch}"
                    else:
                        _model.info.options["model"] = arch
                except KeyError:
                    _model.info.options["model"] = ""
            _model.info.options["pipeline"] = False

            # NOTE: With Tokenizer there are way too many files
            #  to be included, per frameworks. Thus we will load
            #  a Tokenizer instance, then save it to path.
            try:
                _tokenizer_inst = AutoTokenizer.from_pretrained(model_identifier)
                _tokenizer_inst.save_pretrained(_model.path)
                _model.info.options["tokenizer"] = type(_tokenizer_inst).__name__
            except ValueError:
                _model.info.options["tokenizer"] = "na"

    _model.save(model_store)
    return _model.tag


@inject
def save(
    name: str,
    *,
    model: _ModelType,
    tokenizer: t.Optional[_TokenizerType] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **transformers_options_kwargs: str,
) -> Tag:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`t.Union["PreTrainedModel", "TFPreTrainedModel", "FlaxPreTrainedModel"]`,
               required):
            Model instance provided by transformers. This can be retrieved from their
             `AutoModel` class. You can also use any type of models/automodel provided
             by transformers. Refers to https://huggingface.co/transformers/main_classes/model.html
        tokenizer (`t.Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]`):
            Tokenizer instance provided by transformers. This can be retrieved from
             their `AutoTokenizer` class. You can also use any type of Tokenizer
             accordingly to your usecase provided by transformers. Refers to
             https://huggingface.co/transformers/main_classes/tokenizer.html
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        from_tf (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Load the model weights from a TensorFlow checkpoint save file
        from_flax (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Load the model weights from a Flax checkpoint save file
        revision(:obj:`str`, `Optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a
            commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so ``revision`` can be any
            identifier allowed by git.
        mirror(:obj:`str`, `Optional`):
            Mirror source to accelerate downloads in China. If you are from China and
            have an accessibility problem, you can set this option to resolve it. Note
            that we do not guarantee the timeliness or safety. Please refer to the
            mirror site for more information.
        proxies (:obj:`Dict[str, str], `Optional`):
            A dictionary of proxy servers to use by protocol or endpoint, e.g.
            :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The
            proxies are used on each request.
        use_auth_token (:obj:`str` or `bool`, `Optional`):
            The token to use as HTTP bearer authorization for remote files. If
            :obj:`True`, will use the token generated when running
            :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
        force_download (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Whether or not to force the (re-)download of the model weights and
            configuration files, overriding the cached versions if they exist.
        resume_download (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Whether or not to delete incompletely received files. Will attempt to resume
            the download if such a file exists.

        .. note::
            some parameters are direct port from `from_pretrained()` arguments. This
            ensures that when doing save operations we don't actually load the model
            class, which can take a while to do so.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        import bentoml.transformers

        model = AutoModelForQuestionAnswering.from_pretrained("gpt2", from_flax=True)
        tokenizer = AutoTokenizer.from_pretrained("gpt2", from_flax=True)
        # custom training and modification goes here

        tag = bentoml.transformers.save("flax_gpt2", model=model, tokenizer=tokenizer)
    """
    if isinstance(model, str) and not tokenizer:
        raise EnvironmentError(
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
        name (`str`):
            Model name retrieved from huggingface hub. This shouldn't be a model
             instance. If you would like to save a model instance refers to
             `~bentoml.transformers.save` for more information.
        save_namespace (`str`, default to given `name`):
            Name to save model to BentoML modelstore.
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
        keep_download_from_hub (`bool`, `optional`, default to `False`):
            Whether to re-download pretrained model from hub.
        from_tf (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Load the model weights from a TensorFlow checkpoint save file
        from_flax (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Load the model weights from a Flax checkpoint save file
        revision(:obj:`str`, `Optional`, defaults to :obj:`"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a
            commit id, since we use a git-based system for storing models and other
            artifacts on huggingface.co, so ``revision`` can be any identifier allowed
            by git.
        mirror(:obj:`str`, `Optional`):
            Mirror source to accelerate downloads in China. If you are from China and
            have an accessibility problem, you can set this option to resolve it. Note
            that we do not guarantee the timeliness or safety. Please refer to the
            mirror site for more information.
        proxies (:obj:`Dict[str, str], `Optional`):
            A dictionary of proxy servers to use by protocol or endpoint, e.g.
            :obj:`{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The
            proxies are used on each request.
        use_auth_token (:obj:`str` or `bool`, `Optional`):
            The token to use as HTTP bearer authorization for remote files. If
            :obj:`True`, will use the token generated when running
            :obj:`transformers-cli login` (stored in :obj:`~/.huggingface`).
        force_download (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Whether or not to force the (re-)download of the model weights and
            configuration files, overriding the cached versions if they exist.
        resume_download (:obj:`bool`, `Optional`, defaults to :obj:`False`):
            Whether or not to delete incompletely received files. Will attempt to resume
            the download if such a file exists.

        .. note::
            some parameters are direct port from `from_pretrained()` arguments. This
            ensures that when doing save operations we don't actually load the model
            class, which can take up a lot of time and resources.

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer
        import bentoml.transformers

        tag = bentoml.transformers.import_from_huggingface_hub("gpt2", from_tf=True)
    """
    save_namespace = _clean_name(name) if save_namespace is None else save_namespace
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
        tag: t.Union[str, Tag],
        tasks: str,
        *,
        framework: str,
        lm_head: str,
        device: int,
        resource_quota: t.Optional[t.Dict[str, t.Any]],
        batch_options: t.Optional[t.Dict[str, t.Any]],
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
        **pipeline_kwargs: t.Any,
    ):
        in_store_tag = model_store.get(tag).tag
        self._tag = in_store_tag
        super().__init__(str(in_store_tag), resource_quota, batch_options)

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
        self._feature_extractor = pipeline_kwargs.pop(
            "feature_extractor", None
        )  # type: t.Optional[t.Union[str, PreTrainedFeatureExtractor]]
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
    def _setup(self) -> None:  # type: ignore[override]
        try:
            _ = self._model_store.get(self._tag)
            self._config, self._model, self._tokenizer = load(
                self._tag,
                model_store=self._model_store,
                from_flax=False,
                from_tf="tf" in self._framework,
                framework=self._framework,
                lm_head=self._lm_head,
            )
        except FileNotFoundError:
            self._config, self._model, self._tokenizer = None, None, None
        self._pipeline: "Pipeline" = transformers.pipeline(
            self._tasks,
            config=self._config,
            model=self._model,
            tokenizer=self._tokenizer,
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
    def _run_batch(self, input_data: t.Union[_T, t.List[_T]]) -> t.Union[_T, t.List[_T]]:  # type: ignore[override]  # noqa
        res = self._pipeline(input_data)  # type: t.Union[_T, t.List[_T]]
        return res


def load_runner(
    tag: t.Union[str, Tag],
    *,
    tasks: str,
    framework: str = "pt",
    lm_head: str = "casual",
    device: int = -1,
    resource_quota: t.Optional[t.Dict[str, t.Any]] = None,
    batch_options: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **pipeline_kwargs: t.Any,
) -> "_TransformersRunner":
    """
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.transformers.load_runner` implements a Runner class
    that wrap around a transformers pipeline, which optimize it for the BentoML runtime.

    .. warning::
        `load_runner` will try to load the model from given `tag`. If the model does not
         exists, then BentoML will fallback to initialize pipelines from transformers,
         thus files will be loaded from huggingface cache.


    Args:
        tag (`str`):
            Model tag to retrieve model from modelstore
        tasks (`str`):
            Given tasks for pipeline. Refers to https://huggingface.co/transformers/task_summary.html
             for more information.
        framework (`str`, default to `pt`):
            Given frameworks supported by transformers: PyTorch, Tensorflow
        device (`int`, `optional`, default to `-1`):
            Default GPU devices to be used by runner.
        lm_head (`str`, default to `causal`):
            Language model head for your model. For most usecase causal are applied.
             Refers to huggingface.co/transformers for more details on which type of
             language model head is applied to your usecase.
        resource_quota (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure resources allocation for runner.
        batch_options (`t.Dict[str, t.Any]`, default to `None`):
            Dictionary to configure batch options for runner in a service context.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        Runner instances for `bentoml.transformers` model

    Examples::
        import transformers
        import bentoml.transformers

        runner = bentoml.transformers.load_runner("gpt2:latest", tasks='zero-shot-classification',
                                                  framework=tf)
        runner.run_batch(["In today news, ...", "The stocks market seems ..."])
    """
    return _TransformersRunner(
        tag=tag,
        tasks=tasks,
        framework=framework,
        lm_head=lm_head,
        device=device,
        resource_quota=resource_quota,
        batch_options=batch_options,
        model_store=model_store,
        **pipeline_kwargs,
    )
