# Copyright 2019 Atalaya Tech, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from bentoml.artifact import ArtifactSpec, ArtifactInstance


class TextFileArtifact(ArtifactSpec):
    """Abstraction for saving/loading string to/from text files

    Args:
        name (string): Name of the artifact
        file_extension (string): The file format artifact will be saved as.
            Default is .txt
        encoding (string): The encoding will be used for saving/loading text.
            Default is utf8

    """

    def __init__(self, name, file_extension=".txt", encoding="utf8"):
        # TODO: validate file name and extension?
        super(TextFileArtifact, self).__init__(name)
        self._file_extension = file_extension
        self._encoding = encoding

    def _text_file_path(self, base_path):
        return os.path.join(base_path, self.name + self._file_extension)

    def load(self, path):
        with open(self._text_file_path(path), "rb") as f:
            content = f.read().decode(self._encoding)
        return self.pack(content)

    def pack(self, content):  # pylint:disable=arguments-differ
        return _TextFileArtifactInstance(self, content)


class _TextFileArtifactInstance(ArtifactInstance):
    def __init__(self, spec, content):
        super(_TextFileArtifactInstance, self).__init__(spec)

        self._content = content

    def get(self):
        return self._content

    def save(self, dst):
        with open(self.spec._text_file_path(dst), "wb") as f:
            f.write(self._content.encode(self.spec._encoding))
