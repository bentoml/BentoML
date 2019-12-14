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

from bentoml.exceptions import InvalidArgument

ARTIFACTS_DIR_NAME = "artifacts"


class BentoServiceArtifact(object):
    """
    Artifact is a spec describing how to pack and load different types
    of model dependencies to/from file system.

    A call to pack and load should return an BentoServiceArtifactWrapper that can
    be saved to file system or retrieved for BentoService workload
    """

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """
        The name of an artifact is used to set parameters for #pack when used
        in BentoService, and as name of sub-directory for the artifact files
        when saving to file system
        """
        return self._name

    def pack(self, data):
        """
        Pack the in-memory representation of the target artifacts and make sure
        they are ready for 'save' and 'get'

        Note: add "# pylint:disable=arguments-differ" to child class's pack method
        """

    def load(self, path):
        """
        Load artifact assuming it was 'self.save' on the given base_path
        """


class BentoServiceArtifactWrapper(object):
    """
    BentoServiceArtifactWrapper is an object representing a materialized Artifact,
    either loaded from file system or packed with data in a python session
    """

    def __init__(self, spec):
        self._spec = spec

    @property
    def spec(self):
        """
        :return: reference to the BentoServiceArtifact that generated this
        BentoServiceArtifactWrapper
        """
        return self._spec

    def save(self, dst):
        """
        Save artifact to given dst path
        """

    def get(self):
        """
        Get returns a reference to the artifact being packed
        """


class ArtifactCollection(dict):
    """
    A dict of artifact instances (artifact.name -> artifact_instance)
    """

    def __setitem__(self, key, artifact):
        if key != artifact.spec.name:
            raise InvalidArgument(
                "Must use Artifact name as key, {} not equal to {}".format(
                    key, artifact.spec.name
                )
            )

        self.add(artifact)

    def __getattr__(self, item):
        return self[item].get()

    def add(self, artifact):
        if not isinstance(artifact, BentoServiceArtifactWrapper):
            raise InvalidArgument(
                "ArtifactCollection only accepts type BentoServiceArtifactWrapper,"
                "Must call BentoServiceArtifact#pack or BentoServiceArtifact#load "
                "before adding to an ArtifactCollection"
            )

        super(ArtifactCollection, self).__setitem__(artifact.spec.name, artifact)

    def save(self, dst):
        """
        bulk operation for saving all artifacts in self.values() to `dst` path
        """
        save_path = os.path.join(dst, ARTIFACTS_DIR_NAME)
        os.mkdir(save_path)
        for artifact in self.values():
            artifact.save(save_path)
