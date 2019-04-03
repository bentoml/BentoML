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

from abc import ABCMeta, abstractmethod
from six import add_metaclass, iteritems

ARTIFACTS_SUBPATH = "artifacts"


@add_metaclass(ABCMeta)
class Artifact(object):
    """
    Artifact is the base abstraction for saving and loading Model dependencies
    to/from file system.
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def save(self, base_path):
        """
        Save artifact to given base_path
        """

    @abstractmethod
    def load(self, base_path):
        """
        Load artifact assuming it was 'self.save' on the given base_path
        """

    @abstractmethod
    def pack(self, data):
        """
        Pack the in-memory representation of the target artifacts and make sure
        they are ready for 'save' and 'get'

        Note: add "# pylint:disable=arguments-differ" to child class's pack method
        """

    @abstractmethod
    def get(self):
        """
        Get returns an python object which provides all the functionality this
        artifact provides
        """


class ArtifactCollection(object):
    """
    A collection of artifacts
    """

    def __init__(self, collection_subpath=ARTIFACTS_SUBPATH):
        self._artifacts_dict = dict()
        self._collection_subpath = collection_subpath

    def add(self, artifact):
        if artifact.name in self._artifacts_dict.keys():
            # TODO: add warn about overwrite
            pass

        if not isinstance(artifact, Artifact):
            raise TypeError('ArtifactCollection only accepts type bentoml.Artifact')

        self._artifacts_dict[artifact.name] = artifact

    def save(self, path):
        save_path = os.path.join(path, self._collection_subpath)
        os.mkdir(save_path)
        for artifact in self._artifacts_dict.values():
            artifact.save(save_path)

    def load(self, path):
        load_path = os.path.join(path, self._collection_subpath)
        for artifact in self._artifacts_dict.values():
            artifact.load(load_path)

    def __getitem__(self, item):
        # TODO: Catch and handle invalid attribute error
        return self._artifacts_dict.__getitem__(item).get()

    def __getattr__(self, item):
        # TODO: Catch and handle invalid attribute error
        return self._artifacts_dict.__getitem__(item).get()

    def pack(self, *args, **kwargs):
        """
        ArtifactCollection.pack is a wrapper that allow packing a list
        of artifacts with one method call

        >>> from bentoml.artifacts import PickleArtifact
        >>> artifacts = ArtifactsCollection
        >>> artifacts.add(PickleArtifact('clf'))
        >>> artifacts.add(PickleArtifact('feature_columns'))
        >>> # Pack each artifact individually:
        >>> artifacts.pack('clf', my_clf)
        >>> # or
        >>> artifacts.pack(feature_columns=my_feature_columns)
        >>> # or pack multiple artifacts at one time
        >>> artifacts.pack(clf=my_clf, feature_columns=my_feature_columns)
        """
        if args and args[0] in self._artifacts_dict:
            self._artifacts_dict[args[0]].pack(*args[1:], **kwargs)
        elif not args and kwargs:
            for artifact_name, artifact_args in iteritems(kwargs):
                if isinstance(artifact_args, dict):
                    self._artifacts_dict[artifact_name].pack(**artifact_args)
                elif isinstance(artifact_args, list):
                    self._artifacts_dict[artifact_name].pack(*artifact_args)
                else:
                    self._artifacts_dict[artifact_name].pack(artifact_args)

                # TODO: catch error when artifact_name not in the dict
