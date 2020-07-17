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

from bentoml.exceptions import InvalidArgument

ARTIFACTS_DIR_NAME = "artifacts"


class BentoServiceArtifact:
    """
    BentoServiceArtifact is the base abstraction for describing the trained model
    serialization and deserialization process.
    """

    def __init__(self, name):
        if not name.isidentifier():
            raise ValueError(
                "Artifact name must be a valid python identifier, a string \
                is considered a valid identifier if it only contains \
                alphanumeric letters (a-z) and (0-9), or underscores (_). \
                A valid identifier cannot start with a number, or contain any spaces."
            )
        self._name = name

    @property
    def pip_dependencies(self):
        return []

    @property
    def name(self):
        """
        The name of an artifact. It can be used to reference this artifact in a
        BentoService inference API callback function, via `self.artifacts[NAME]`
        """
        return self._name

    def pack(self, data):
        """
        Pack the in-memory trained model object to this BentoServiceArtifact

        Note: add "# pylint:disable=arguments-differ" to child class's pack method
        """

    def load(self, path):
        """
        Load artifact assuming it was 'self.save' on the same `path`
        """

    def save(self, dst):
        """
        Save artifact to given dst path
        """

    def get(self):
        """
        Get returns a reference to the artifact being packed or loaded from path
        """


class BentoServiceArtifactWrapper:
    """
    Deprecated: use only the BentoServiceArtifact to define artifact operations
    """


class ArtifactCollection(dict):
    """
    A dict of artifact instances (artifact.name -> artifact_instance)
    """

    def __setitem__(self, key: str, artifact: BentoServiceArtifact):
        if key != artifact.name:
            raise InvalidArgument(
                "Must use Artifact name as key, {} not equal to {}".format(
                    key, artifact.name
                )
            )
        self.add(artifact)

    def __getattr__(self, item: str):
        return self[item].get()

    def add(self, artifact: BentoServiceArtifact):
        super(ArtifactCollection, self).__setitem__(artifact.name, artifact)

    def save(self, dst):
        """
        bulk operation for saving all artifacts in self.values() to `dst` path
        """
        save_path = os.path.join(dst, ARTIFACTS_DIR_NAME)
        os.mkdir(save_path)
        for artifact in self.values():
            artifact.save(save_path)
