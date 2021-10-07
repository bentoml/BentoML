import os
import typing as t
from importlib import import_module
from pathlib import Path

from ._internal.models import store as _stores
from ._internal.runner import Runner
from .exceptions import BentoMLException, MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    from transformers import (  # noqa
        AutoConfig,
        AutoModel,
        AutoTokenizer,
        FlaxAutoModel,
        FlaxPreTrainedModel,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
        TFAutoModel,
        TFPreTrainedModel,
    )
    from transformers.models.auto.auto_factory import _BaseAutoModelClass  # noqa
try:
    import jax.numpy as jnp
    import transformers
    from transformers import AutoConfig, PretrainedConfig
    from transformers.file_utils import (
        CONFIG_NAME,
        FLAX_WEIGHTS_NAME,
        TF2_WEIGHTS_NAME,
        TF_WEIGHTS_NAME,
        WEIGHTS_NAME,
        cached_path,
        filename_to_url,
        get_from_cache,
        hf_bucket_url,
        is_offline_mode,
        is_remote_url,
    )
except ImportError:
    raise MissingDependencyException(
        """transformers is required in order to use module `bentoml.transformers`, install transformers with 
        `pip install transformers`."""
    )

_V = t.TypeVar("_V")

ModelType = t.TypeVar(
    "ModelType",
    bound=t.Union["PreTrainedModel", "TFPreTrainedModel", "FlaxPreTrainedModel"],
)


def _check_flax_supported() -> None:
    _supported: bool = transformers.__version__.startswith("4")
    if not _supported:
        raise BentoMLException(
            "BentoML will only support transformers 4.x forwards to support FlaxModel"
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


def _infer_autoclass(framework: str, lm_head: str) -> "_BaseAutoModelClass":
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
            f"{e}\n\nPlease refers to https://huggingface.co/transformers/model_doc/auto.html"
        )


# model = bentoml.transformers.load("my_transformers_name", framework='pt')
def load(
    tag: str,
    *,
    framework: str = "pt",
    lm_head: str = "causal",
) -> t.Tuple["AutoTokenizer", "_BaseAutoModelClass"]:
    _check_flax_supported()
    model_info = _stores.get(tag)
    _autoclass = _infer_autoclass(framework, lm_head)

    model = _autoclass.from_pretrained(model_info.path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_info.path)
    return (model, tokenizer)


# my_model = transformers.AutoModel.from_pretrained("bert-uncased")
# my_tokenizer = transformers.AutoTokenizer.from_pretrained("bert-uncased")
# ... training
# tag = bentoml.transformers.save("my_transformers_model", model=my_model, tokenizer=my_tokenizer)

# if you don't have your own tokenizer

# save model directly from transformers hub
# tag = bentoml.transformers.save("my_transformers_model", model_name="bert-uncased")


def _infer_model_tokenizer_class(model_name: str) -> t.Dict[str, str]:
    # config = AutoConfig.from_pretrained(model_name)
    # if type(config) in MODEL_FOR_PRETRAINING_MAPPING.values()
    return {}


# save logics
#  when model is a type string -> download from huggingface hub to bentoml modelstore
#  when model is a ModelType -> tokenizer should also be provided -> then just saved it directly to modelstore
# TODO: supports for serializing a pipeline
def save(
    name: str,
    *model_args,
    model: t.Union[str, ModelType],
    tokenizer: t.Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"] = None,
    config: "PretrainedConfig" = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
    **transformers_kwargs,
) -> str:
    _check_flax_supported()
    context = {"transformers": transformers.__version__}

    if isinstance(model, str):
        assert not tokenizer, _SAVE_CONFLICTS_ERR
        options = _infer_model_tokenizer_class(model)
    else:
        assert tokenizer, "`tokenizer` cannot be None or undefined."
        options = {
            "model": model.__class__.__name__,
            "tokenizer": tokenizer.__class__.__name__,
        }

    with _stores.register(
        name,
        module=__name__,
        options=options,
        framework_context=context,
        metadata=metadata,
    ) as ctx:
        if isinstance(model, str):
            # assuming users will want model from modelhub
            revision = transformers_kwargs.pop("revision", None)
            mirror = transformers_kwargs.pop("mirror", None)

            force_download = transformers_kwargs.pop("force_download", False)
            resume_download = transformers_kwargs.pop("resume_download", True)
            proxies = transformers_kwargs.pop("proxies", None)
            local_files_only = transformers_kwargs.pop("local_files_only", False)
            use_auth_token = transformers_kwargs.pop("use_auth_token", None)
            # below are the fields that we aren't accepting for save
            # from_tf = transformers_kwargs.pop("from_tf", False)
            # state_dict = transformers_kwargs.pop("state_dict", None)
            # output_loading_info = transformers_kwargs.pop("output_loading_info", False)

            config_file_url = hf_bucket_url(
                model, filename=CONFIG_NAME, revision=revision, mirror=mirror
            )
            resolved_config_path = cached_path(
                config_file_url,
                cache_dir=ctx.path,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
            )
            url, _ = filename_to_url(resolved_config_path, cache_dir=ctx.path)
            print(url)

            # if is_offline_mode() and not local_files_only:
            #     local_files_only = True
            # if not isinstance(config, PretrainedConfig):
            #     config_path = config if config is not None else model
            #     config = AutoConfig.from_pretrained(
            #         config_path,
            #         *model_args,
            #         cache_dir=ctx.path,
            #         force_download=force_download,
            #         resume_download=resume_download,
            #         proxies=proxies,
            #         local_files_only=local_files_only,
            #         use_auth_token=use_auth_token,
            #         revision=revision,
            #         **transformers_kwargs,
            #     )

            if os.path.isdir(model):
                # if (
                #     from_tf
                #     and Path(
                #         model, TF_WEIGHTS_NAME + ".index"
                #     ).is_file()
                # ):
                #     # Load from a TF 1.0 checkpoint in priority if from_tf
                #     archive_file = Path(
                #         model, TF_WEIGHTS_NAME + ".index"
                #     )
                # elif (
                #     from_tf
                #     and Path(
                #         model, TF2_WEIGHTS_NAME
                #     ).is_file()
                # ):
                #     # Load from a TF 2.0 checkpoint in priority if from_tf
                #     archive_file = Path(
                #         model, TF2_WEIGHTS_NAME
                #     )
                if Path(model, WEIGHTS_NAME).is_file():
                    # Load from a PyTorch checkpoint
                    archive_file = Path(model, WEIGHTS_NAME)
                elif Path(model, FLAX_WEIGHTS_NAME).is_file():
                    # Load from a Flax checkpoint
                    archive_file = Path(model, FLAX_WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        f"Error no file named {[WEIGHTS_NAME, FLAX_WEIGHTS_NAME]} found in directory {model}"
                    )
            elif Path(model).is_file() or is_remote_url(model):
                archive_file = model
            else:
                archive_file = hf_bucket_url(
                    model,
                    filename=WEIGHTS_NAME,
                    revision=revision,
                    mirror=mirror,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    use_auth_token=use_auth_token,
                )
                ctx.path = resolved_archive_file
            except EnvironmentError:
                msg = (
                    f"Can't load weights for '{model}'. Make sure that:\n\n"
                    f"- '{model}' is a correct model identifier listed on "
                    f"'https://huggingface.co/models'\n\n - or '{model}' "
                    "is the correct path to a directory containing a file named one "
                    f"of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)
        else:
            model.save_pretrained(ctx.path)
            tokenizer.save_pretrained(ctx.path)
        return ctx.tag


def import_from_huggingface_hub(name: str):
    ...


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
    def __init__(
        self,
        tag: str,
        tasks: str,
        resource_quota: t.Dict[str, t.Any],
        batch_options: t.Dict[str, t.Any],
    ):
        super().__init__(tag, resource_quota, batch_options)
        try:
            transformers.pipelines.check_task(tasks)
        except KeyError as e:
            raise BentoMLException(
                f"{e}, given tasks is not recognized by transformers."
            )
        self._tasks = tasks

    @property
    def num_concurrency(self):
        return self.num_replica

    @property
    def num_replica(self):
        # TODO: supports multiple GPUS
        return 1

    def _setup(self):
        self._pipeline = transformers.pipeline(self._tasks)

    def _run_batch(self, input_data: _V) -> _V:
        return self._pipeline(input_data)
