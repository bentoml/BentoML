import os
import re

from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper


class JSONArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading objects to/from JSON files.

    Args:
        name (str): Name of the artifact
        file_extension (:obj:`str`, optional): The file extention used for the saved
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
        return _JSONArtifactWrapper(self, content, **json_dumps_kwargs)


class _JSONArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, content, **json_dumps_kwargs):
        """
        Args:
              **json_dumps_kwargs: keyword args forwarded to call to json.dumps().
        """
        super().__init__(spec)
        self._content = content
        self._json_dumps_kwargs = json_dumps_kwargs

    def get(self):
        return self._content

    def save(self, dst):
        with open(self.spec._file_path(dst), "wt", encoding=self.spec._encoding) as fp:
            fp.write(
                self.spec.json_module.dumps(self._content, **self._json_dumps_kwargs)
            )
