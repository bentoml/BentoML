import os
from importlib import import_module

from bentoml.service.env import BentoServiceEnv

from bentoml.exceptions import (
    InvalidArgument,
    MissingDependencyException,
    NotFound,
)
from bentoml.service import BentoServiceArtifact

try:
    import transformers
except ImportError:
    transformers = None


class TransformersModelArtifact(BentoServiceArtifact):
    """
    Artifact class for saving/loading Transformers models

    Args:
        name (string): name of the artifact

    Raises:
        MissingDependencyException: transformers package
            is required for TransformersModelArtifact
        InvalidArgument: invalid argument type, model being packed
            must be either a dictionary of format
            {'model':transformers model object,
            'tokenizer':transformers tokenizer object}
            or a directory path where the model is saved
            or a pre-trained model provided by transformers
            which can be loaded using transformers.AutoModelWithLMHead
        NotFound: if the provided model name or model path is not found

    Example usage:

    >>> import bentoml
    >>> from transformers import AutoModelWithLMHead, AutoTokenizer
    >>> from bentoml.adapters import JsonInput
    >>>
    >>> @bentoml.env(pip_packages=["transformers==3.1.0", "torch==1.6.0"])
    >>> @bentoml.artifacts([TransformersModelArtifact("gptModel")])
    >>> class TransformerService(bentoml.BentoService):
    >>>     @bentoml.api(input=JsonInput(), batch=False)
    >>>     def predict(self, parsed_json):
    >>>         src_text = parsed_json.get("text")
    >>>         model = self.artifacts.gptModel.get("model")
    >>>         tokenizer = self.artifacts.gptModel.get("tokenizer")
    >>>         input_ids = tokenizer.encode(src_text, return_tensors="pt")
    >>>         output = model.generate(input_ids, max_length=50)
    >>>         output = tokenizer.decode(output[0], skip_special_tokens=True)
    >>>         return output
    >>>
    >>>
    >>> ts = TransformerService()
    >>>
    >>> model_name = "gpt2"
    >>> model = AutoModelWithLMHead.from_pretrained("gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> # Option 1: Pack using dictionary (recommended)
    >>> artifact = {"model": model, "tokenizer": tokenizer}
    >>> ts.pack("gptModel", artifact)
    >>> # Option 2: pack using the name of the model
    >>> # ts.pack("gptModel","gpt2")
    >>> # Note that while packing using the name of the model,
    >>> # ensure that the model can be loaded using
    >>> # transformers.AutoModelWithLMHead (eg GPT, Bert, Roberta etc.)
    >>> # If this is not the case (eg AutoModelForQuestionAnswering, BartModel etc)
    >>> # then pack the model by passing a dictionary
    >>> # with the model and tokenizer declared explicitly
    >>> saved_path = ts.save()
    """

    def __init__(self, name):
        super().__init__(name)
        self._model = None
        self._tokenizer_type = None
        self._model_type = "AutoModelWithLMHead"

        if transformers is None:
            raise MissingDependencyException(
                "transformers package is required to use TransformersModelArtifact"
            )

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def _load_from_directory(self, path):
        if self._model_type is None:
            raise NotFound(
                "Type of transformers model not found. "
                "This should be present in a file called "
                "'_model_type.txt' in the artifacts of the bundle."
            )
        if self._tokenizer_type is None:
            raise NotFound(
                "Type of transformers tokenizer not found. "
                "This should be present in a file called 'tokenizer_type.txt' "
                "in the artifacts of the bundle."
            )
        transformers_model = getattr(
            import_module("transformers"), self._model_type
        ).from_pretrained(path)
        tokenizer = getattr(
            import_module("transformers"), self._tokenizer_type
        ).from_pretrained(path)
        return {"model": transformers_model, "tokenizer": tokenizer}

    def _load_from_dict(self, model):
        if not model.get("model"):
            raise InvalidArgument(
                " 'model' key is not found in the dictionary."
                " Expecting a dictionary of with keys 'model' and 'tokenizer'"
            )
        if not model.get("tokenizer"):
            raise InvalidArgument(
                "'tokenizer' key is not found in the dictionary. "
                "Expecting a dictionary of with keys 'model' and 'tokenizer'"
            )

        model_class = str(type(model.get("model")).__module__)
        tokenizer_class = str(type(model.get("tokenizer")).__module__)
        # if either model or tokenizer is not an object of transformers
        if not model_class.startswith("transformers"):
            raise InvalidArgument(
                "Expecting a transformers model object but object passed is {}".format(
                    type(model.get("model"))
                )
            )
        if not tokenizer_class.startswith("transformers"):
            raise InvalidArgument(
                "Expecting a transformers model object but object passed is {}".format(
                    type(model.get("tokenizer"))
                )
            )
        # success
        return model

    def _load_from_string(self, model_name):
        try:
            transformers_model = getattr(
                import_module("transformers"), self._model_type
            ).from_pretrained(model_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            return {"model": transformers_model, "tokenizer": tokenizer}
        except EnvironmentError:
            raise NotFound(
                "model with the name {} is not present "
                "in the transformers library".format(model_name)
            )
        except AttributeError:
            raise NotFound(
                "transformers has no model type called {}".format(self._model_type)
            )

    def set_dependencies(self, env: BentoServiceEnv):
        if env._infer_pip_packages:
            env.add_pip_packages(['xgboost'])

    def pack(self, model, metadata=None):
        loaded_model = None
        if isinstance(model, str):
            if os.path.isdir(model):
                loaded_model = self._load_from_directory(model)
            else:
                loaded_model = self._load_from_string(model)
        elif isinstance(model, dict):
            loaded_model = self._load_from_dict(model)
        else:
            raise InvalidArgument(
                "Expecting a Dictionary of format "
                "{'model':<transformers model object>,'tokenizer':<tokenizer object> }"
            )
        self._model = loaded_model
        return self

    def load(self, path):
        path = self._file_path(path)
        with open(os.path.join(path, "_model_type.txt"), "r") as f:
            self._model_type = f.read().strip()
        with open(os.path.join(path, "tokenizer_type.txt"), "r") as f:
            self._tokenizer_type = f.read().strip()
        return self.pack(path)

    def _save_model_type(self, path):
        with open(os.path.join(path, "_model_type.txt"), "w") as f:
            f.write(self._model_type)
        with open(os.path.join(path, "tokenizer_type.txt"), "w") as f:
            f.write(self._tokenizer_type)

    def save(self, dst):
        path = self._file_path(dst)
        self._model_type = self._model.get("model").__class__.__name__
        self._tokenizer_type = self._model.get("tokenizer").__class__.__name__
        self._model.get("model").save_pretrained(path)
        self._model.get("tokenizer").save_pretrained(path)
        self._save_model_type(path)
        return path

    def get(self):
        return self._model
