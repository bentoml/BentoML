import os
import logging
from bentoml.artifact import BentoServiceArtifact
from bentoml.exceptions import MissingDependencyException, InvalidArgument, NotFound

logger = logging.getLogger(__name__)
try:
    import transformers
except ImportError:
    raise MissingDependencyException(
        "transformers package is required to use TransformersModelArtifact"
    )


class TransformersArtifact(BentoServiceArtifact):
    """[summary]
    Abstraction for saving/loading HuggingFace Transformers models

    Args:
        name (string): name of the artifact

    Raises:
        MissingDependencyException: transformers package is required for TransformersArtifact
        InvalidArgument: invalid argument type, model being packed must be a dictionary of format {'model':transformers model object,'tokenizer':transformers tokenizer object} or a directory where the model is saved or a pretrained model provied by transformers which can be loaded by transformers.AutoModel
        NotFound: if the provided model name or model path is not found

    Example usage:

    >>> import torch.nn as nn
    >>>
    >>> class Net(nn.Module):
    >>>     def __init__(self):
    >>>         super(Net, self).__init__()
    >>>         ...
    >>>
    >>>     def forward(self, x):
    >>>         ...
    >>>
    >>> net = Net()
    >>> # Train model with data
    >>>
    >>>
    >>> import bentoml
    >>> from bentoml.adapters import JsonInput
    >>> from bentoml.frameworks.transformers import TransformersArtifact
    >>> 
    >>> 
    >>> # Explicitly add either torch or tensorflow dependency which will be used by transformers
    >>> @bentoml.env(pip_dependencies=["torch==1.6.0", "transformers==3.1.0"])
    >>> @bentoml.artifacts([TransformersArtifact('gptModel')])
    >>> class TransformersService(bentoml.BentoService):
    >>>     @bentoml.api(input=JsonInput())
    >>>     def predict(self, parsed_json):
    >>>          src_text = parsed_json[0].get("text")
    >>>          model = self.artifacts.gptModel.get("model")
    >>>          tokenizer = self.artifacts.gptModel.get("tokenizer")
    >>>          input_ids = tokenizer.encode(src_text, return_tensors="pt")
    >>>          output = model.generate(input_ids, max_length=50)
    >>>          output = [tokenizer.decode(output[0], skip_special_tokens=True)]
    >>>          return output
    >>>
    >>>
    >>> svc = TransformersService()
    >>> ts = TransformersGPT2TextGenerator()
    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelWithLMHead.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)
    >>> ts.pack("gptModel", {"model": model, "tokenizer": tokenizer})
    >>> saved_path = ts.save()
    """

    def __init__(self, name):
        super(TransformersArtifact, self).__init__(name)
        self._model = None
        self._model_type = None
        self._tokenizer_type = None

    def _file_path(self, base_path):
        return os.path.join(base_path, self.name)

    def _load_from_directory(self, path):
        if self._model_type is None:
            raise NotFound(
                "Type of transformers model not found. This should be present in a file called 'model_type.txt' in the artifacts of the bundle."
            )
        if self._tokenizer_type is None:
            raise NotFound(
                "Type of transformers tokenizer not found. This should be present in a file called 'tokenizer_type.txt' in the artifacts of the bundle."
            )
        transformers_model = eval(self._model_type).from_pretrained(path)
        tokenizer = eval(self._tokenizer_type).from_pretrained(path)
        self._model = {"model": transformers_model, "tokenizer": tokenizer}

    def _load_from_dict(self, model):
        if not model.get("model"):
            raise InvalidArgument(
                " 'model' key is not found in the dictionary. Expecting a dictionary of with keys 'model' and 'tokenizer'"
            )
        if not model.get("tokenizer"):
            raise InvalidArgument(
                "'tokenizer' key is not found in the dictionary. Expecting a dictionary of with keys 'model' and 'tokenizer'"
            )

        model_class = str(type(model.get("model")).__module__)
        tokenizer_class = str(type(model.get("tokenizer")).__module__)
        # if either model or tokenizer is not an object of transformers
        if not model_class.startswith("transformers"):
            raise InvalidArgument(
                "Expecting a transformers model object but object passes is {}".format(
                    type(model.get("model"))
                )
            )
        if not tokenizer_class.startswith("transformers"):
            raise InvalidArgument(
                "Expecting a transformers model object but object passes is {}".format(
                    type(model.get("tokenizer"))
                )
            )
        # success
        self._model = model

    def _load_from_string(self, model_name):
        try:
            transformers_model = transformers.AutoModel.from_pretrained(model_name)
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self._model = {"model": transformers_model, "tokenizer": tokenizer}
        except Exception as error:
            raise NotFound(
                "model with the name {} is not present in the transformers library".format(
                    model_name
                )
            )

    def pack(self, model):
        if isinstance(model, str):
            if os.path.isdir(model):
                self._load_from_directory(model)
            else:
                self._load_from_string(model)
        elif isinstance(model, dict):
            self._load_from_dict(model)
        else:
            raise InvalidArgument(
                "Expecting a Dictionary of format {'model':<transformers model object>,'tokenizer':<tokenizer object> }"
            )

        return self

    def load(self, path):
        path = self._file_path(path)
        with open(os.path.join(path, "model_type.txt"), "r") as f:
            self._model_type = f.read().strip()
        with open(os.path.join(path, "tokenizer_type.txt"), "r") as f:
            self._tokenizer_type = f.read().strip()
        return self.pack(path)

    def _save_model_type(self, path):
        with open(os.path.join(path, "model_type.txt"), "w") as f:
            f.write(self._model_type)
        with open(os.path.join(path, "tokenizer_type.txt"), "w") as f:
            f.write(self._tokenizer_type)

    def save(self, dst):
        path = self._file_path(dst)
        self._model_type = "transformers." + self._model.get("model").__class__.__name__
        self._tokenizer_type = (
            "transformers." + self._model.get("tokenizer").__class__.__name__
        )
        self._model.get("model").save_pretrained(path)
        self._model.get("tokenizer").save_pretrained(path)
        self._save_model_type(path)
        return path

    def get(self):
        return self._model
