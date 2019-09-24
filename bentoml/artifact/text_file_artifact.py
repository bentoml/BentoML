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
import re

from bentoml.artifact import BentoServiceArtifact, BentoServiceArtifactWrapper


class TextFileArtifact(BentoServiceArtifact):
    """Abstraction for saving/loading string to/from text files

    Args:
        name (str): Name of the artifact
        file_extension (:obj:`str`, optional): The file extention used for the saved
            text file. Defaults to ".txt"
        encoding (str): The encoding will be used for saving/loading text. Defaults
            to "utf8"

    """

    def __init__(self, name, file_extension=".txt", encoding="utf8"):
        super(TextFileArtifact, self).__init__(name)
        self._file_extension = file_extension
        self._encoding = encoding

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
        return _TextFileArtifactWrapper(self, content)


class _TextFileArtifactWrapper(BentoServiceArtifactWrapper):
    def __init__(self, spec, content):
        super(_TextFileArtifactWrapper, self).__init__(spec)

        self._content = content

    def get(self):
        return self._content

    def save(self, dst):
        with open(self.spec._text_file_path(dst), "wb") as f:
            f.write(self._content.encode(self.spec._encoding))
