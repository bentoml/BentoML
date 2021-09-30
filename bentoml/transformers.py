import os
import pathlib
import typing as t
from importlib import import_module

from ._internal.models import JSON_EXT, SAVE_NAMESPACE
from ._internal.models import store as _stores
from ._internal.service import Runner
from .exceptions import BentoMLException, MissingDependencyException

if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import transformers
    from transformers import (  # noqa
        AutoModel,
        AutoTokenizer,
        FlaxAutoModel,
        FlaxPreTrainedModel,
        PreTrainedModel,
        TFAutoModel,
        TFPreTrainedModel,
    )
    from transformers.models.auto.auto_factory import _BaseAutoModelClass  # noqa
try:
    import transformers
except ImportError:
    raise MissingDependencyException(
        """transformers is required in order to use module `bentoml.transformers`, install transformers with 
        `pip install transformers`.""")

_T = t.TypeVar("_T")

TransformersInput = t.TypeVar(
    "TransformersInput",
    bound=t.Union[str, PreTrainedModel, TFPreTrainedModel, FlaxPreTrainedModel],
)
TransformersOutput = t.TypeVar(
    "TransformersOutput",
    bound=t.Dict[
        str,
        t.Union[AutoTokenizer, _BaseAutoModelClass],
    ],
)

FRAMEWORK_ALIASES: t.Dict[str, str] = {"pt": "pytorch", "tf": "tensorflow"}

FRAMEWORK_AUTOMODEL_PREFIX_MAPPING: t.Dict[str, str] = {
    "pytorch": "AutoModel",
    "tensorflow": "TFAutoModel",
    "flax": "FlaxAutoModel",
}

AUTOMODEL_LM_HEAD_MAPPING: t.Dict[str, str] = {
    "causal": "ForCausalLM",
    "masked": "ForMaskedLM",
    "seq-to-seq": "ForSeq2SeqLM",
}


def _check_flax_supported() -> None:
    _supported: bool = transformers.__version__.startswith("4")
    if not _supported:
        raise BentoMLException(
            "BentoML will only support transformers 4.x forwards to support FlaxModel"
        )


def _lm_head_module_name(framework: str, lm_head: str) -> str:
    if (
        framework not in FRAMEWORK_AUTOMODEL_PREFIX_MAPPING
        and framework not in FRAMEWORK_ALIASES
    ):
        raise AttributeError(
            f"{framework} is either invalid aliases or not supported by transformers."
            " Accepted: pt(alias to pytorch), tf(alias to tensorflow), and flax"
        )
    if lm_head not in AUTOMODEL_LM_HEAD_MAPPING:
        raise AttributeError(
            f"`{lm_head}` alias for lm_head is invalid."
            f" Accepted: {[*AUTOMODEL_LM_HEAD_MAPPING.keys()]}."
            " If you need any other AutoModel type provided by transformers,"
            " feel free to open a PR at https://github.com/bentoml/BentoML."
        )
    framework_prefix = (
        FRAMEWORK_ALIASES[framework] if framework in FRAMEWORK_ALIASES else framework
    )
    return (
        FRAMEWORK_AUTOMODEL_PREFIX_MAPPING[framework_prefix]
        + AUTOMODEL_LM_HEAD_MAPPING[lm_head]
    )


def _load_from_directory(
    path: PathType, model_type: str, tokenizer_type: str
) -> TransformersOutput:
    transformers_model = getattr(
        import_module("transformers"), model_type
    ).from_pretrained(str(path))
    tokenizer = getattr(import_module("transformers"), tokenizer_type).from_pretrained(
        str(path)
    )
    return {"model": transformers_model, "tokenizer": tokenizer}


def _load_from_string(model_name: str, lm_head: str) -> TransformersOutput:
    try:
        transformers_model = getattr(
            import_module("transformers"), lm_head
        ).from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        return {"model": transformers_model, "tokenizer": tokenizer}
    except EnvironmentError:
        raise NotFound(f"{model_name} is not provided by transformers")


def _validate_transformers_dict(
    transformers_dict: TransformersOutput,
) -> None:
    if not transformers_dict.get("model"):
        raise InvalidArgument(
            " 'model' key is not found in the dictionary."
            " Expecting a dictionary of with keys 'model' and 'tokenizer'"
        )
    if not transformers_dict.get("tokenizer"):
        raise InvalidArgument(
            "'tokenizer' key is not found in the dictionary. "
            "Expecting a dictionary of with keys 'model' and 'tokenizer'"
        )

    model_class = str(type(transformers_dict.get("model")).__module__)
    tokenizer_class = str(type(transformers_dict.get("tokenizer")).__module__)
    # if either model or tokenizer is not an object of transformers
    if not model_class.startswith("transformers"):
        raise InvalidArgument(
            "Expecting a transformers model object but object passed is {}".format(
                type(transformers_dict.get("model"))
            )
        )
    if not tokenizer_class.startswith("transformers"):
        raise InvalidArgument(
            "Expecting a transformers model object but object passed is {}".format(
                type(transformers_dict.get("tokenizer"))
            )
        )


def load(
    name_or_path_or_dict: t.Union[PathType, dict],
    framework: t.Optional[str] = "pt",
    lm_head: t.Optional[str] = "causal",
) -> TransformersOutput:
    _check_flax_supported()
    if isinstance(name_or_path_or_dict, (str, bytes, os.PathLike, pathlib.PurePath)):
        name_or_path = str(name_or_path_or_dict)
        if os.path.isdir(name_or_path):
            with open(os.path.join(name_or_path, "__model_class_type.txt"), "r") as f:
                _model_type = f.read().strip()
            with open(
                os.path.join(name_or_path, "__tokenizer_class_type.txt"), "r"
            ) as f:
                _tokenizer_type = f.read().strip()
            loaded_dict = _load_from_directory(
                name_or_path, _model_type, _tokenizer_type
            )
        else:
            _lm_head = _lm_head_module_name(framework, lm_head)
            loaded_dict = _load_from_string(name_or_path, _lm_head)
    else:
        _validate_transformers_dict(name_or_path_or_dict)
        loaded_dict = name_or_path_or_dict
    return loaded_dict


def _save_model_type(path: PathType, model_type: str, tokenizer_type: str) -> None:
    with open(os.path.join(path, "__model_class_type.txt"), "w") as f:
        f.write(model_type)
    with open(os.path.join(path, "__tokenizer_class_type.txt"), "w") as f:
        f.write(tokenizer_type)


def save(
    name: str,
    *,
    model: TransformersInput,
    tokenizer: t.Optional[transformers.AutoTokenizer] = None,
    metadata: t.Optional[t.Dict[str, t.Any]] = None,
) -> str:
    _check_flax_supported()
    context = {"transformers": transformers.__version__}
    options = {"model": model.__class__.__name__}
    with _stores.register(
        name, module=__name__, framework_context=context, metadata=metadata
    ) as ctx:
        _model_type = model.get("model").__class__.__name__
        _tokenizer_type = model.get("tokenizer").__class__.__name__
        model.get("model").save_pretrained(ctx.path)
        model.get("tokenizer").save_pretrained(ctx.path)
        _save_model_type(ctx.path, _model_type, _tokenizer_type)
    return f"{name}:{ctx.version}"


def import_from_huggingface_hub(name: str):
    ...


def load_runner(tag: str, *, tasks: str, resource_quota: t.Dict[str, t.Any] = None,
                batch_options: t.Dict[str, t.Any] = None) -> "_TransformersRunner":
    """\
    Runner represents a unit of serving logic that can be scaled horizontally to
    maximize throughput. `bentoml.transformers.load_runner` implements a Runner class that
    wrap around a Transformers model, which optimize it for the BentoML runtime.

    Returns:
        Runner instances for the target `bentoml.transformers` model
    """
    return _TransformersRunner(tag=tag, tasks=tasks, resource_quota=resource_quota, batch_options=batch_options)


class _TransformersRunner(Runner):
    def __init__(self, tag: str,
                tasks: str,
                resource_quota: t.Dict[str, t.Any],
                batch_options: t.Dict[str, t.Any]):
        super().__init__(tag, resource_quota, batch_options)
        try:
            transformers.pipelines.check_task(tasks)
        except KeyError as e:
            raise BentoMLException(f"{e}, givent tasks is not recognized by transformers.")
        self._tasks = tasks
        
        ...

    @property
    def num_concurrency(self):
        return self.num_replica

    @property
    def num_replica(self):
        # TODO: supports multiple GPUS
        return 1

    def _setup(self):
        pass

    def _run_batch(self, input_data: "_T") -> "_T":
        pass


# model = transformers.BertModel.from_pretrained("bert_uncased")

# torch.save(model, "bert.pt")
# torch.load('bert.pt')
