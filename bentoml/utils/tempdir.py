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

import os.path
import logging
import tempfile
import shutil
from bentoml.configuration import get_debug_mode


logger = logging.getLogger(__name__)


class TempDirectory(object):
    """
    Helper class that creates and cleans up a temporary directory, can
    be used as a context manager.

    >>> with TempDirectory() as tempdir:
    >>>     print(os.path.isdir(tempdir))
    """

    def __init__(
        self, cleanup=True, prefix="temp",
    ):

        self._cleanup = cleanup
        self._prefix = prefix
        self.path = None

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.path)

    def __enter__(self):
        self.create()
        return self.path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if get_debug_mode():
            logger.debug(
                'BentoML in debug mode, keeping temp directory "%s"', self.path
            )
            return

        if self._cleanup:
            self.cleanup()

    def create(self):
        if self.path is not None:
            logger.debug("Skipped temp directory creation: %s", self.path)
            return self.path

        tempdir = tempfile.mkdtemp(prefix="bentoml-{}-".format(self._prefix))
        self.path = os.path.realpath(tempdir)
        logger.debug("Created temporary directory: %s", self.path)

    def cleanup(self, ignore_errors=False):
        """
        Remove the temporary directory created
        """
        if self.path is not None and os.path.exists(self.path):
            shutil.rmtree(self.path, ignore_errors=ignore_errors)
        self.path = None
