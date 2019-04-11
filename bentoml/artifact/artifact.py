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

ARTIFACTS_SUBPATH = "artifacts"


class ArtifactSpec(object):
    """
    Artifact is a spec describing how to pack and load different types
    of model dependencies to/from file system.

    A call to pack and load should return an ArtifactInstance that can
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


class ArtifactInstance(object):
    """
    ArtifactInstance is an object representing a materialized Artifact, either
    loaded from file system or packed with data in a python session
    """

    def __init__(self, spec):
        self._spec = spec

    @property
    def spec(self):
        """
        :return: reference to the ArtifactSpec that produced this ArtifactInstance
        """
        return self._spec

    def save(self, dst):
        """
        Save artifact to given dst path
        """

    def get(self):
        """
        Get returns an python object which provides all the functionality this
        artifact provides
        """


class ArtifactCollection(dict):
    """
    A dict of artifact instances (artifact.name -> artifact_instance)
    """

    def __setitem__(self, key, artifact):
        if key != artifact.spec.name:
            raise ValueError('Must use Artifact name as key, {} not equal to {}'.format(
                key, artifact.spec.name))

        self.add(artifact)

    def __getattr__(self, item):
        return self[item].get()

    def add(self, artifact):
        if not isinstance(artifact, ArtifactInstance):
            raise TypeError('ArtifactCollection only accepts type bentoml.ArtifactInstance,'
                            'Must call Artifact#pack or Artifact#load before adding to'
                            'an ArtifactCollection')

        super(ArtifactCollection, self).__setitem__(artifact.spec.name, artifact)

    def save(self, dst):
        """
        bulk operation for saving all artifacts in self.values() to `dst` path
        """
        save_path = os.path.join(dst, ARTIFACTS_SUBPATH)
        os.mkdir(save_path)
        for artifact in self.values():
            artifact.save(save_path)

    @classmethod
    def load(cls, path, artifacts_spec):
        """
        bulk operation for loading all artifacts from path based on a list of ArtifactSpec
        """
        load_path = os.path.join(path, ARTIFACTS_SUBPATH)
        artifacts = cls()

        for artifact_spec in artifacts_spec:
            artifact_instance = artifact_spec.load(load_path)
            artifacts.add(artifact_instance)

        return artifacts
