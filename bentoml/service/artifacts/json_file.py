import os
import re

from bentoml.service import BentoServiceArtifact


JSON_ARTIFACT_EXTENSION = ".json"


class JSONArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading objects to/from JSON files.

    Args:
        name (str): Name of the artifact
        encoding (:obj:`str`, optional): The encoding will be used for saving/loading
            text. Defaults to "utf8"
        json_module (module|object, optional): Namespace/object implementing `loads()`
            and `dumps()` methods for serializing/deserializing to/from JSON string.
            Defaults to stdlib's json module.
    """

    def __init__(self, name, encoding="utf-8", json_module=None):
        super().__init__(name)
        self._content = None
        self._json_dumps_kwargs = None

        self._encoding = encoding
        if json_module:
            self.json_module = json_module
        else:
            import json

            self.json_module = json

    def _file_path(self, base_path):
        return os.path.join(
            base_path,
            re.sub("[^-a-zA-Z0-9_.() ]+", "", self.name) + JSON_ARTIFACT_EXTENSION,
        )

    def load(self, path):
        with open(self._file_path(path), "rt", encoding=self._encoding) as fp:
            content = self.json_module.loads(fp.read())
        return self.pack(content)

    def pack(
        self, content, metadata=None, **json_dumps_kwargs
    ):  # pylint:disable=arguments-renamed
        self._content = content
        self._json_dumps_kwargs = json_dumps_kwargs
        return self

    def get(self):
        return self._content

    def save(self, dst):
        with open(self._file_path(dst), "wt", encoding=self._encoding) as fp:
            fp.write(self.json_module.dumps(self._content, **self._json_dumps_kwargs))
