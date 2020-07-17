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


import os
import re

from bentoml.artifact import BentoServiceArtifact


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
