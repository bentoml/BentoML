# ==============================================================================
#     Copyright (c) 2021 Atalaya Tech. Inc
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
# ==============================================================================

import os
import typing as t
from importlib import import_module

from ._internal.artifacts import ModelArtifact
from ._internal.exceptions import InvalidArgument, MissingDependencyException, NotFound
from ._internal.types import MetadataType, PathType


class TransformersModel(ModelArtifact):
    """
    Model class for saving/loading :obj:`transformers` models.

    Args:
        model (`Dict[str, Union[transformers.PreTrainedModel,transformers.PreTrainedTokenizer]`):
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
    """

    try:
        import transformers
    except ImportError:
        raise MissingDependencyException(
            'transformers is required by TransformersModel'
        )

    _model_type: str = "AutoModelWithLMHead"

    def __init__(
        self,
        model,
        metadata: t.Optional[MetadataType] = None,
        model_type: t.Optional[str] = None,
    ):
        super(TransformersModel, self).__init__(model, metadata=metadata)
        if model_type is not None:
            self._model_type = model_type

    @classmethod
    def _load_from_directory(
        cls, path: PathType, tokenizer_type: str
    ) -> t.Dict[str, t.Any]:
        if tokenizer_type is None:
            raise NotFound(
                "Type of transformers tokenizer not found. "
                "This should be present in a file called 'tokenizer_type.txt' "
                "in the artifacts of the bundle."
            )
        transformers_model = getattr(
            import_module("transformers"), cls._model_type
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
            import transformers

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
    def load(cls, path: PathType):
        with open(os.path.join(path, "__model__type.txt"), "r") as f:
            cls._model_type = f.read().strip()
        with open(os.path.join(path, "tokenizer_type.txt"), "r") as f:
            cls._tokenizer_type = f.read().strip()
        if isinstance(path, str):
            if os.path.isdir(path):
                loaded_model = cls._load_from_directory(path)
            else:
                loaded_model = cls._load_from_string(path)
        elif isinstance(path, dict):
            loaded_model = cls._load_from_dict(path)
        else:
            raise InvalidArgument(
                "Expected either string representation of the model name or dict of format"
                "{'model':<transformers model object>,'tokenizer':<tokenizer object>}, "
                f"got {type(path)} instead"
            )
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
