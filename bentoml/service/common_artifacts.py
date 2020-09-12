import os
import re
import logging

from bentoml.utils import cloudpickle
from bentoml.service.artifacts import BentoServiceArtifact


logger = logging.getLogger(__name__)


class PickleArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading python objects with pickle serialization

    Args:
        name (str): Name for the artifact
        pickle_module (module|str): The python module will be used for pickle
            and unpickle artifact, default pickle module in BentoML's fork of
            `cloudpickle`, which is identical to the Apache Spark fork
        pickle_extension (str): The extension format for pickled file.
    """

    def __init__(self, name, pickle_module=cloudpickle, pickle_extension=".pkl"):
        super(PickleArtifact, self).__init__(name)

        self._pickle_extension = pickle_extension
        self._obj = None

        if isinstance(pickle_module, str):
            self._pickle = __import__(pickle_module)
        else:
            self._pickle = pickle_module

    def _pkl_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._pickle_extension)

    def pack(self, obj):  # pylint:disable=arguments-differ
        self._obj = obj
        return self

    def load(self, path):
        with open(self._pkl_file_path(path), "rb") as pkl_file:
            obj = self._pickle.load(pkl_file)
        self.pack(obj)
        return self

    def get(self):
        return self._obj

    def save(self, dst):
        with open(self._pkl_file_path(dst), "wb") as pkl_file:
            self._pickle.dump(self._obj, pkl_file)


class JSONArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading objects to/from JSON files.

    Args:
        name (str): Name of the artifact
        file_extension (:obj:`str`, optional): The file extension used for the saved
            text file. Defaults to ".txt"
        encoding (:obj:`str`, optional): The encoding will be used for saving/loading
            text. Defaults to "utf8"
        json_module (module|object, optional): Namespace/object implementing `loads()`
            and `dumps()` methods for serializing/deserializing to/from JSON string.
            Defaults to stdlib's json module.
    """

    def __init__(
        self, name, file_extension=".json", encoding="utf-8", json_module=None
    ):
        super().__init__(name)
        self._content = None
        self._json_dumps_kwargs = None

        self._file_extension = file_extension
        self._encoding = encoding
        if json_module:
            self.json_module = json_module
        else:
            import json

            self.json_module = json

    def _file_path(self, base_path):
        return os.path.join(
            base_path,
            re.sub("[^-a-zA-Z0-9_.() ]+", "", self.name) + self._file_extension,
        )

    def load(self, path):
        with open(self._file_path(path), "rt", encoding=self._encoding) as fp:
            content = self.json_module.loads(fp.read())
        return self.pack(content)

    def pack(self, content, **json_dumps_kwargs):  # pylint:disable=arguments-differ
        self._content = content
        self._json_dumps_kwargs = json_dumps_kwargs
        return self

    def get(self):
        return self._content

    def save(self, dst):
        with open(self._file_path(dst), "wt", encoding=self._encoding) as fp:
            fp.write(self.json_module.dumps(self._content, **self._json_dumps_kwargs))


class TextFileArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading string to/from text files

    Args:
        name (str): Name of the artifact
        file_extension (:obj:`str`, optional): The file extension used for the saved
            text file. Defaults to ".txt"
        encoding (str): The encoding will be used for saving/loading text. Defaults
            to "utf8"

    """

    def __init__(self, name, file_extension=".txt", encoding="utf-8"):
        super(TextFileArtifact, self).__init__(name)
        self._file_extension = file_extension
        self._encoding = encoding
        self._content = None

    def _text_file_path(self, base_path):
        return os.path.join(
            base_path,
            re.sub('[^-a-zA-Z0-9_.() ]+', '', self.name) + self._file_extension,
        )

    def load(self, path):
        with open(self._text_file_path(path), "rb") as f:
            content = f.read().decode(self._encoding)
        return self.pack(content)

    def pack(self, content):  # pylint:disable=arguments-differ
        self._content = content
        return self

    def get(self):
        return self._content

    def save(self, dst):
        with open(self._text_file_path(dst), "wb") as f:
            f.write(self._content.encode(self._encoding))
