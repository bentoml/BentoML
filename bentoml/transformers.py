import functools
import importlib.util
import json
import logging
import os
import tempfile
import typing as t
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path

import importlib_metadata
import requests
from filelock import FileLock
from simple_di import Provide, inject

from ._internal.configuration.containers import BentoMLContainer
from ._internal.runner import Runner
from ._internal.types import PathType
from .exceptions import BentoMLException, MissingDependencyException

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import io

    from transformers import (  # noqa
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        FlaxAutoModel,
        FlaxPreTrainedModel,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        TFAutoModel,
        TFPreTrainedModel,
    )
    from transformers.models.auto.auto_factory import _BaseAutoModelClass  # noqa

    from ._internal.models.store import ModelStore
try:
    import transformers
    from huggingface_hub import HfFolder
    from transformers.file_utils import (
        CONFIG_NAME,
        FLAX_WEIGHTS_NAME,
        TF2_WEIGHTS_NAME,
        WEIGHTS_NAME,
        hf_bucket_url,
        http_get,
        http_user_agent,
    )

except ImportError:
    raise MissingDependencyException(
        """transformers is required in order to use module `bentoml.transformers`, install transformers with 
        `pip install transformers`."""
    )

_V = t.TypeVar("_V")

_ModelType = t.TypeVar(
    "_ModelType",
    bound=t.Union["PreTrainedModel", "TFPreTrainedModel", "FlaxPreTrainedModel"],
)


def _check_flax_supported() -> None:
    _supported: bool = transformers.__version__.startswith("4")
    _flax_available = (
        importlib.util.find_spec("jax") is not None
        and importlib.util.find_spec("flax") is not None
    )
    if not _supported:
        logger.warning(
            f"Detected transformers version: {transformers.__version__}, which doesn't have "
            "supports for Flax. Update `transformers` to 4.x and above to have Flax supported."
        )
    else:
        try:
            _jax_version = importlib_metadata.version("jax")
            _flax_version = importlib_metadata.version("flax")
            logger.info(
                f"JAX version {_jax_version}, Flax version {_flax_version} available."
            )
        except importlib_metadata.PackageNotFoundError:
            logger.warning(
                "No versions of Flax is found under the current machine. In order to use Flax with "
                "transformers 4.x and above, refers to https://github.com/google/flax#quick-install"
            )


_SAVE_CONFLICTS_ERR = """\
`tokenizer=None` if `model` is type `str`, currently got `tokenizer={tokenizer}`

If you want to save the weight directly from
`transformers` and save it to BentoML do:
    `bentoml.transformers.save('bert_model', model='bert-uncased')`.

If you are training a model from scratch using transformers, to save into BentoML do:
    `bentoml.transformers.save('bert_model', model=my_bert_model, tokenizer=my_tokenizer)`
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


def _load_autoclass(framework: str, lm_head: str) -> "_BaseAutoModelClass":
    if (
        framework not in _AUTOMODEL_PREFIX_MAPPING
        and framework not in _FRAMEWORK_ALIASES
    ):
        raise AttributeError(
            f"{framework} is either invalid aliases or not supported by transformers."
            " Accepted: pt(alias to pytorch), tf(alias to tensorflow), and flax"
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
    class_inst = f"{_AUTOMODEL_PREFIX_MAPPING[framework_prefix]}{_AUTOMODEL_LM_HEAD_MAPPING[lm_head]}"
    try:
        return getattr(import_module("transformers"), class_inst)
    except AttributeError as e:
        raise BentoMLException(
            f"{e}\n\nPlease refers to https://huggingface.co/transformers/model_doc/auto.html."
        )


# model = bentoml.transformers.load("my_transformers_name", framework='pt')
@inject
def load(
    tag: str,
    *,
    framework: str = "pt",
    lm_head: str = "causal",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> t.Tuple[
    "_BaseAutoModelClass", t.Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]
]:
    """
    Load a model from BentoML local modelstore with given name.

    Args:
        tag (`str`):
            Tag of a saved model in BentoML local modelstore.
        framework (`str`, default to `pt`):
            Given frameworks supported by transformers: PyTorch, Tensorflow, Flax
        lm_head (`str`, default to `causal`):
            Language model head for your model. For most usecase causal are applied.
             Refers to huggingface.co/transformers for more details on which type of
             language model head is applied to your usecase.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.

    Returns:
        a Tuple containing `model` and `tokenizer` for your given model saved at BentoML modelstore.

    Examples::
        import bentoml.transformers
        model, tokenizer = bentoml.transformers.load('custom_gpt2', framework="flax", lm_head="masked")
    """
    _check_flax_supported()
    model_info = model_store.get(tag)
    _autoclass = _load_autoclass(framework, lm_head)

    model = _autoclass.from_pretrained(model_info.path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_info.path)
    return model, tokenizer


def _download_from_hub(
    hf_url: str,
    output_dir: str,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: t.Union[t.Dict[str, str], str, None] = None,
    use_auth_token: t.Union[bool, str, None] = None,
) -> PathType:
    headers = {"user-agent": http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError(
                "You specified use_auth_token=True, but a huggingface token was not found."
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
                "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
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
        return fpath

    # Prevent parallel downloads of the same file with a lock.
    lock_path = fpath.with_suffix(".lock")
    with FileLock(str(lock_path)):

        # If the download just completed while the lock was activated.
        if os.path.exists(fpath) and not force_download:
            # Even if returning early like here, the lock will be released.
            return fpath

        if resume_download:
            incomplete_path = fpath.with_suffix(".incomplete")

            @contextmanager
            def _resume_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resume_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = functools.partial(
                tempfile.NamedTemporaryFile, mode="wb", dir=output_dir, delete=False
            )
            resume_size = 0
    with temp_file_manager() as temp_file:
        http_get(
            url_to_download,
            temp_file,
            proxies=proxies,
            resume_size=resume_size,
            headers=headers,
        )

    logger.info(f"storing {hf_url} at {output_dir}")
    os.replace(temp_file.name, str(fname))

    # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
    umask = os.umask(0o666)
    os.umask(umask)
    os.chmod(fname, 0o666 & ~umask)
    logger.info(f"creating metadata file for {fpath.name}")
    meta = {"url": hf_url, "etag": etag}
    meta_path = fpath.with_suffix(".metadata.json")
    with meta_path.open("w") as meta_file:
        json.dump(meta, meta_file)


def _save(
    name: str,
    *,
    model: t.Union[str, _ModelType],
    tokenizer: t.Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
    metadata: t.Optional[t.Dict[str, t.Any]],
    model_store: "ModelStore",
    **transformers_options_kwargs,
):
    _check_flax_supported()
    context = {"transformers": transformers.__version__}

    if isinstance(model, str):
        assert not tokenizer, _SAVE_CONFLICTS_ERR
        options = None
    else:
        assert tokenizer, "`tokenizer` cannot be None or undefined."
        options = {
            "model": model.__class__.__name__,
            "tokenizer": tokenizer.__class__.__name__,
        }

    with model_store.register(
        name,
        module=__name__,
        framework_context=context,
        options=options,
        metadata=metadata,
    ) as ctx:
        if isinstance(model, str):
            revision = transformers_options_kwargs.pop("revision", None)
            mirror = transformers_options_kwargs.pop("mirror", None)
            proxies = transformers_options_kwargs.pop("proxies", None)
            use_auth_token = transformers_options_kwargs.pop("use_auth_token", None)

            from_tf = transformers_options_kwargs.pop("from_tf", False)
            from_flax = transformers_options_kwargs.pop("from_flax", False)

            force_download = transformers_options_kwargs.pop("force_download", False)
            resume_download = transformers_options_kwargs.pop("resume_download", True)

            # download config file
            config_file_url = hf_bucket_url(
                model, filename=CONFIG_NAME, revision=revision, mirror=mirror
            )
            _download_from_hub(
                config_file_url,
                output_dir=ctx.path,
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
                model,
                filename=weight_filename,
                revision=revision,
                mirror=mirror,
            )
            _download_from_hub(
                weight_file_url,
                output_dir=ctx.path,
                proxies=proxies,
                use_auth_token=use_auth_token,
                force_download=force_download,
                resume_download=resume_download,
            )

            # NOTE: With Tokenizer there are way too many files
            #  to be included, thus for now we will load a Tokenizer instance,
            #  then save it to path.
            # TODO(aarnphm):
            _tokenizer_inst = transformers.AutoTokenizer.from_pretrained(model)
            _tokenizer_inst.save_pretrained(ctx.path)

            # ctx.options will be None here
            with Path(ctx.path, CONFIG_NAME).open("r", encoding="utf-8") as of:
                _config = json.loads(of.read())
                ctx.options = {
                    "model": "".join(_config["architectures"]),
                    "tokenizer": type(_tokenizer_inst).__name__,
                }
        else:
            model.save_pretrained(ctx.path)
            tokenizer.save_pretrained(ctx.path)
        return ctx.tag


@inject
def save(
    name: str,
    *,
    model: _ModelType,
    tokenizer: t.Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **transformers_options_kwargs,
) -> str:
    """
    Save a model instance to BentoML modelstore.

    Args:
        name (`str`):
            Name for given model instance. This should pass Python identifier check.
        model (`t.Union["PreTrainedModel", "TFPreTrainedModel", "FlaxPreTrainedModel"]`, required):
            Model instance provided by transformers. This can be retrieved from their `AutoModel`
             class. You can also use any type of models/automodel provided by transformers. Refers to
             https://huggingface.co/transformers/main_classes/model.html
        tokenizer (`t.Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]`):
            Tokenizer instance provided by transformers. This can be retrieved from their `AutoTokenizer`
             class. You can also use any type of Tokenizer accordingly to your usecase provided by
             transformers. Refers to https://huggingface.co/transformers/main_classes/tokenizer.html
        metadata (`t.Optional[t.Dict[str, t.Any]]`, default to `None`):
            Custom metadata for given model.
        model_store (`~bentoml._internal.models.store.ModelStore`, default to `BentoMLContainer.model_store`):
            BentoML modelstore, provided by DI Container.
    **transformers_options_kwargs,

    Returns:
        tag (`str` with a format `name:version`) where `name` is the defined name user
        set for their models, and version will be generated by BentoML.

    Examples::
        import xgboost as xgb
        import bentoml.xgboost

        # read in data
        dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
        dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
        # specify parameters via map
        param = dict(max_depth=2, eta=1, objective='binary:logistic')
        num_round = 2
        bst = xgb.train(param, dtrain, num_round)
        ...

        tag = bentoml.xgboost.save("my_xgboost_model", bst, booster_params=param)
        # example tag: my_xgboost_model:20210929_153BC4

        # load the booster back:
        bst = bentoml.xgboost.load("my_xgboost_model:latest") # or
        bst = bentoml.xgboost.load(tag)

    """
    if isinstance(model, str):
        raise EnvironmentError(
            "If you want to import model directly from huggingface hub"
            " please use `import_from_huggingface_hub` instead. `save` should"
            " only be used for saving custom pretrained model and tokenizer"
        )
    return _save(
        name=name,
        model=model,
        tokenizer=tokenizer,
        metadata=metadata,
        model_store=model_store,
        **transformers_options_kwargs,
    )


@inject
def import_from_huggingface_hub(
    name: str,
    *,
    local_name: t.Union[str, None] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    **transformers_options_kwargs,
):
    local_name = name if local_name is None else local_name
    return _save(
        name=local_name,
        model=name,
        tokenizer=None,
        metadata=metadata,
        model_store=model_store,
        **transformers_options_kwargs,
    )


def load_runner(
    tag: str,
    *,
    tasks: str,
    resource_quota: t.Dict[str, t.Any] = None,
    batch_options: t.Dict[str, t.Any] = None,
) -> "_TransformersRunner":
    """\
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.transformers.load_runner` implements a Runner class that
    wrap around a Transformers model, which optimize it for the BentoML runtime.

    Returns:
        Runner instances for the target `bentoml.transformers` model
    """
    return _TransformersRunner(
        tag=tag, tasks=tasks, resource_quota=resource_quota, batch_options=batch_options
    )


class _TransformersRunner(Runner):
    @inject
    def __init__(
        self,
        tag: str,
        tasks: str,
        resource_quota: t.Dict[str, t.Any],
        batch_options: t.Dict[str, t.Any],
        *,
        framework: str = "pt",
        lm_head: str = "causal",
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        super().__init__(tag, resource_quota, batch_options)
        try:
            transformers.pipelines.check_task(tasks)
        except KeyError as e:
            raise BentoMLException(
                f"{e}, given tasks is not recognized by transformers."
            )
        self._tasks = tasks
        self._model_store = model_store
        self._framework = framework
        self._lm_head = lm_head

    @property
    def num_concurrency(self):
        return self.num_replica

    @property
    def num_replica(self):
        # TODO: supports multiple GPUS
        return 1

    def _setup(self):
        try:
            _ = self._model_store.get(self.name)
            model, tokenizer = load(
                self.name,
                framework=self._framework,
                lm_head=self._lm_head,
                model_store=self._model_store,
            )
        except FileNotFoundError:
            model, tokenizer = None, None
            pass
        self._pipeline = transformers.pipeline(
            self._tasks, model=model, tokenizer=tokenizer
        )

    def _run_batch(self, input_data: _V) -> _V:
        return self._pipeline(input_data)
