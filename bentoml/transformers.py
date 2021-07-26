import os
import pathlib
import typing as t
from importlib import import_module

from ._internal.artifacts import ModelArtifact
from ._internal.types import MetadataType, PathType
from .exceptions import InvalidArgument, MissingDependencyException, NotFound

try:
    import transformers
except ImportError:
    raise MissingDependencyException("transformers is required by TransformersModel")


class TransformersModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`transformers` models.

    Args:
        model (`Dict[str, Union[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]`):
            TODO:
        metadata (`Dict[str, Any]`,  `optional`, default to `None`):
            Class metadata

    Raises:
        MissingDependencyException:
            :obj:`transformers` is required by TransformersModel
        InvalidArgument:
            invalid argument type, model being packed
            must be either a dictionary of format
            {'model':transformers model object,
            'tokenizer':transformers tokenizer object}
            or a directory path where the model is saved
            or a pre-trained model provided by transformers
            which can be loaded using transformers.AutoModelWithLMHead
        NotFound:
            if the provided model name or model path is not found

    Example usage under :code:`train.py`::

        TODO:

    One then can define :code:`bento_service.py`::

        TODO:

    Pack bundle under :code:`bento_packer.py`::

        TODO:
    """  # noqa # pylint: enable=line-too-long

    _model_type: str = "AutoModelWithLMHead"

    def __init__(
        self, model, metadata: t.Optional[MetadataType] = None,
    ):
        super(TransformersModel, self).__init__(model, metadata=metadata)

    @classmethod
    def _load_from_directory(
        cls, path: PathType, model_type: str, tokenizer_type: str
    ) -> t.Dict[str, t.Any]:
        transformers_model = getattr(
            import_module("transformers"), model_type
        ).from_pretrained(path)
        tokenizer = getattr(
            import_module("transformers"), tokenizer_type
        ).from_pretrained(path)
        return {"model": transformers_model, "tokenizer": tokenizer}

    @staticmethod
    def _load_from_dict(transformers_dict: t.Dict[str, t.Any]) -> dict:
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
        return transformers_dict

    @classmethod
    def _load_from_string(cls, model_name: str) -> dict:
        try:
            transformers_model = getattr(
                import_module("transformers"), cls._model_type
            ).from_pretrained(model_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            return {"model": transformers_model, "tokenizer": tokenizer}
        except EnvironmentError:
            raise NotFound(f"{model_name} is not available within transformers")
        except AttributeError:
            raise NotFound(f"transformers has no model type called {cls._model_type}")

    @classmethod
    def load(cls, path: t.Union[PathType, dict]):
        if isinstance(path, (str, bytes, os.PathLike, pathlib.PurePath)):
            str_path = str(path)
            if os.path.isdir(str_path):
                with open(os.path.join(path, "__model__type.txt"), "r") as f:
                    _model_type = f.read().strip()
                with open(os.path.join(path, "tokenizer_type.txt"), "r") as f:
                    _tokenizer_type = f.read().strip()
                loaded_model = cls._load_from_directory(
                    str_path, _model_type, _tokenizer_type
                )
            else:
                loaded_model = cls._load_from_string(str_path)
        elif isinstance(path, dict):
            loaded_model = cls._load_from_dict(path)
        else:
            err_msg: str = """\
            Expected either model name or a dictionary only
            containing `model` and `tokenizer` as keys, but
            got {path} instead.
            """
            raise InvalidArgument(err_msg.format(path=type(path)))
        return loaded_model

    def _save_model_type(self, path: PathType, tokenizer_type: str) -> None:
        with open(os.path.join(path, "__model__type.txt"), "w") as f:
            f.write(self._model_type)
        with open(os.path.join(path, "tokenizer_type.txt"), "w") as f:
            f.write(tokenizer_type)

    def save(self, path: PathType) -> None:
        self._model_type = self._model.get("model").__class__.__name__
        tokenizer_type = self._model.get("tokenizer").__class__.__name__
        self._model.get("model").save_pretrained(path)
        self._model.get("tokenizer").save_pretrained(path)
        self._save_model_type(path, tokenizer_type)
