import os
import re

from bentoml.service import BentoServiceArtifact


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

    def pack(self, content, metadata=None):  # pylint:disable=arguments-renamed
        self._content = content
        return self

    def get(self):
        return self._content

    def save(self, dst):
        with open(self._text_file_path(dst), "wb") as f:
            f.write(self._content.encode(self._encoding))
