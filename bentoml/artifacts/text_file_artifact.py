import os
from bentoml.artifacts import Artifact


class TextFileArtifact(Artifact):
    """
    Abstraction for saving/loading string to/from text files
    """

    def __init__(self, name, file_extension='.txt', encoding='utf8'):
        self.content = None
        # TODO: validate file_extension (with a regex?)
        self._file_extension = file_extension
        self._encoding = encoding
        super(TextFileArtifact, self).__init__(name)

    def text_file_path(self, base_path):
        os.path.join(base_path, self.name + self._file_extension)

    def load(self, base_path):
        with open(self.text_file_path(base_path), "rb") as f:
            self.content = f.read().decode(self._encoding)

    def save(self, base_path):
        if not self.content:
            raise RuntimeError("Must 'pack' artifact before 'save'.")

        with open(self.text_file_path(base_path), "wb") as f:
            f.write(self.content.encode(self._encoding))

    def pack(self, content):  # pylint:disable=arguments-differ
        self.content = content

    def get(self):
        return self.content
