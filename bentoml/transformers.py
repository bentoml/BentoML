import os
import pathlib
import typing as t
from importlib import import_module

import bentoml._internal.constants as _const

from ._internal.models.base import Model
from ._internal.types import GenericDictType, PathType
from ._internal.utils import LazyLoader
from .exceptions import BentoMLException, InvalidArgument, NotFound

_exc = _const.IMPORT_ERROR_MSG.format(
    fwr="transformers",
    module=__name__,
    inst="`pip install transformers`",
)


if t.TYPE_CHECKING:  # pragma: no cover
    # pylint: disable=unused-import
    import transformers
    from transformers import AutoTokenizer, PreTrainedModel  # noqa
    from transformers.models.auto.auto_factory import _BaseAutoModelClass  # noqa
else:
    transformers = LazyLoader("transformers", globals(), "transformers", exc_msg=_exc)

TransformersModelInput = t.TypeVar(
    "TransformersModelInput",
    bound=t.Union[str, os.PathLike, "PreTrainedModel"],
)
TransformersModelOutput = t.TypeVar(
    "TransformersModelOutput",
    bound=t.Dict[
        str,
        t.Union["AutoTokenizer", "_BaseAutoModelClass"],
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


class TransformersModel(Model):
    """
    Model class for saving/loading :obj:`transformers` models.

    Args:
        model (`Union[str, os.PathLike, Dict[str, Union[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]]`):
            A dictionary `{'model':<model_obj>, 'tokenizer':<tokenizer_obj>}`
             to setup Transformers model
        metadata (`GenericDictType`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`transformers` is required by TransformersModel
        InvalidArgument:
            :obj:`model` must be either a dictionary
             or a path for saved transformers model or
             a pre-trained model string provided by transformers
        NotFound:
            if the provided model name or model path is not found

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento.py`::

        TODO:

    """  # noqa # pylint: enable=line-too-long

    def __init__(
        self,
        model: TransformersModelInput,
        metadata: t.Optional[GenericDictType] = None,
    ):
        _check_flax_supported()
        super(TransformersModel, self).__init__(model, metadata=metadata)

    @staticmethod
    def _load_from_directory(
        path: PathType, model_type: str, tokenizer_type: str
    ) -> TransformersModelOutput:
        transformers_model = getattr(
            import_module("transformers"), model_type
        ).from_pretrained(str(path))
        tokenizer = getattr(
            import_module("transformers"), tokenizer_type
        ).from_pretrained(str(path))
        return {"model": transformers_model, "tokenizer": tokenizer}

    @staticmethod
    def _load_from_string(model_name: str, lm_head: str) -> TransformersModelOutput:
        try:
            transformers_model = getattr(
                import_module("transformers"), lm_head
            ).from_pretrained(model_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            return {"model": transformers_model, "tokenizer": tokenizer}
        except EnvironmentError:
            raise NotFound(f"{model_name} is not provided by transformers")

    @staticmethod
    def _validate_transformers_dict(
        transformers_dict: TransformersModelOutput,
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

    @classmethod
    def load(  # pylint: disable=arguments-differ
        cls,
        name_or_path_or_dict: t.Union[PathType, dict],
        framework: t.Optional[str] = "pt",
        lm_head: t.Optional[str] = "causal",
    ) -> TransformersModelOutput:
        if isinstance(
            name_or_path_or_dict, (str, bytes, os.PathLike, pathlib.PurePath)
        ):
            name_or_path = str(name_or_path_or_dict)
            if os.path.isdir(name_or_path):
                with open(
                    os.path.join(name_or_path, "__model_class_type.txt"), "r"
                ) as f:
                    _model_type = f.read().strip()
                with open(
                    os.path.join(name_or_path, "__tokenizer_class_type.txt"), "r"
                ) as f:
                    _tokenizer_type = f.read().strip()
                loaded_dict = cls._load_from_directory(
                    name_or_path, _model_type, _tokenizer_type
                )
            else:
                _lm_head = _lm_head_module_name(framework, lm_head)
                loaded_dict = cls._load_from_string(name_or_path, _lm_head)
        else:
            cls._validate_transformers_dict(name_or_path_or_dict)
            loaded_dict = name_or_path_or_dict
        return loaded_dict

    @staticmethod
    def _save_model_type(path: PathType, model_type: str, tokenizer_type: str) -> None:
        with open(os.path.join(path, "__model_class_type.txt"), "w") as f:
            f.write(model_type)
        with open(os.path.join(path, "__tokenizer_class_type.txt"), "w") as f:
            f.write(tokenizer_type)

    def save(self, path: PathType) -> None:
        _model_type = self._model.get("model").__class__.__name__
        _tokenizer_type = self._model.get("tokenizer").__class__.__name__
        self._model.get("model").save_pretrained(path)
        self._model.get("tokenizer").save_pretrained(path)
        self._save_model_type(path, _model_type, _tokenizer_type)
