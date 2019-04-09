# BentoML - Machine Learning Toolkit for packaging and deploying models
# Copyright (C) 2019 Atalaya Tech, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from bentoml.artifact import ArtifactSpec, ArtifactInstance


class TextFileArtifact(ArtifactSpec):
    """
    Abstraction for saving/loading string to/from text files
    """

    def __init__(self, name, file_extension='.txt', encoding='utf8'):
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
