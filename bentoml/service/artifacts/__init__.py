import os
import re
import logging
from typing import List
from pathlib import Path

from bentoml.exceptions import InvalidArgument, FailedPrecondition
from bentoml.service.env import BentoServiceEnv
from bentoml.utils.ruamel_yaml import YAML

logger = logging.getLogger(__name__)

ARTIFACTS_DIR_NAME = "artifacts"


class BentoServiceArtifact:
    """
    BentoServiceArtifact is the base abstraction for describing the trained model
    serialization and deserialization process.
    """

    def __new__(cls, *args, **kwargs):
        obj = super(BentoServiceArtifact, cls).__new__(cls)

        # store the args and kwargs used for creating this artifact instance - this is
        # used internally for the _copy method below

        # TODO: this warning may due to pylint's bug (issue 4668). We may
        # remove the following comment later
        # pylint: disable=unused-private-member
        obj.__args = args
        obj.__kwargs = kwargs

        return obj

    def _copy(self):
        """
        Create a new empty artifact instance with the same name and arguments, this
        is only used internally for BentoML to create new artifact instances when
        creating a new BentoService instance. BentoML stores a reference to the
        arguments provided for initializing this artifact class for the purpose of
        recreating a new instance with the same config
        """
        return self.__class__(*self.__args, **self.__kwargs)

    def __init__(self, name: str):
        if not name.isidentifier():
            raise ValueError(
                "Artifact name must be a valid python identifier, a string \
                is considered a valid identifier if it only contains \
                alphanumeric letters (a-z) and (0-9), or underscores (_). \
                A valid identifier cannot start with a number, or contain any spaces."
            )
        self._name = name
        self._packed = False
        self._loaded = False
        self._metadata = dict()

    @property
    def packed(self):
        return self._packed

    @property
    def loaded(self):
        return self._loaded

    @property
    def metadata(self):
        return self._metadata

    @property
    def is_ready(self):
        return self.packed or self.loaded

    @property
    def name(self):
        """
        The name of an artifact. It can be used to reference this artifact in a
        BentoService inference API callback function, via `self.artifacts[NAME]`
        """
        return self._name

    def pack(self, model, metadata: dict = None):  # pylint: disable=unused-argument
        """
        Pack the in-memory trained model object to this BentoServiceArtifact

        Note: add "# pylint:disable=arguments-differ" to child class's pack method
        """

    def load(self, path):
        """
        Load artifact assuming it was 'self.save' on the same `path`
        """

    def _metadata_path(self, base_path):
        return os.path.join(
            base_path, re.sub("[^-a-zA-Z0-9_.() ]+", "", self.name) + ".yml",
        )

    def save(self, dst):
        """
        Save artifact to given dst path

        In a BentoService saved bundle, all artifacts are being saved to the
        `{service.name}/artifact/` directory. Thus, `save` must use the artifact name
        as a way to differentiate its files from other artifacts.

        For example, a BentoServiceArtifact class' save implementation should not create
        a file named `config.json` in the `dst` directory. Because when a BentoService
        has multiple artifacts of this same artifact type, they will all try to create
        this `config.json` file and overwrite each other. The right way to do it here,
        is use the name as the prefix(or part of the file name), e.g.
         f"{artifact.name}-config.json"
        """

    def get(self):
        """
        Get returns a reference to the artifact being packed or loaded from path
        """

    def set_dependencies(self, env: BentoServiceEnv):
        """modify target BentoServiceEnv instance to ensure the required dependencies
        are listed in the BentoService environment spec

        :param env: target BentoServiceEnv instance to modify
        """

    def __getattribute__(self, item):
        if item == 'pack':
            original = object.__getattribute__(self, item)

            def wrapped_pack(*args, **kwargs):
                if self.packed:
                    logger.warning(
                        "`pack` an artifact multiple times may lead to unexpected "
                        "behaviors"
                    )
                if 'metadata' in kwargs:
                    if isinstance(kwargs['metadata'], dict):
                        self._metadata = kwargs['metadata']
                    else:
                        raise TypeError(
                            "Setting a non-dictionary metadata " "is not supported."
                        )
                ret = original(*args, **kwargs)
                # do not set `self._pack` if `pack` has failed with an exception raised
                self._packed = True
                return ret

            return wrapped_pack

        elif item == 'load':
            original = object.__getattribute__(self, item)

            def wrapped_load(*args, **kwargs):
                if self.packed:
                    logger.warning(
                        "`load` on a 'packed' artifact may lead to unexpected behaviors"
                    )
                if self.loaded:
                    logger.warning(
                        "`load` an artifact multiple times may lead to unexpected "
                        "behaviors"
                    )

                # load metadata if exists
                path = args[0]  # load(self, path)
                meta_path = self._metadata_path(path)
                if os.path.isfile(meta_path):
                    with open(meta_path) as file:
                        yaml = YAML()
                        yaml_content = file.read()
                        self._metadata = yaml.load(yaml_content)

                ret = original(*args, **kwargs)
                # do not set self._loaded if `load` has failed with an exception raised
                self._loaded = True
                return ret

            return wrapped_load

        elif item == 'save':

            def wrapped_save(*args, **kwargs):
                if not self.is_ready:
                    raise FailedPrecondition(
                        "Trying to save empty artifact. An artifact needs to be `pack` "
                        "with model instance or `load` from saved path before saving"
                    )

                # save metadata
                dst = args[0]  # save(self, dst)
                if self.metadata:
                    yaml = YAML()
                    yaml.dump(self.metadata, Path(self._metadata_path(dst)))

                original = object.__getattribute__(self, item)
                return original(*args, **kwargs)

            return wrapped_save

        elif item == 'get':

            def wrapped_get(*args, **kwargs):
                if not self.is_ready:
                    raise FailedPrecondition(
                        "Trying to access empty artifact. An artifact needs to be "
                        "`pack` with model instance or `load` from saved path before "
                        "it can be used for inference"
                    )
                original = object.__getattribute__(self, item)
                return original(*args, **kwargs)

            return wrapped_get

        return object.__getattribute__(self, item)


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
        """Proxy to `BentoServiceArtifact#get()` to allow easy access to the model
        artifact in its native form. E.g. for a BentoService class with an artifact
        `SklearnModelArtifact('model')`, when user code accesses `self.artifacts.model`
        in the BentoService API callback function, instead of returning an instance of
        BentoServiceArtifact, it returns a Sklearn model object
        """
        return self[item].get()

    def get(self, item: str) -> BentoServiceArtifact:
        """Access the BentoServiceArtifact instance by artifact name
        """
        return self[item]

    def get_artifact_list(self) -> List[BentoServiceArtifact]:
        """Access the list of BentoServiceArtifact instances
        """
        return super(ArtifactCollection, self).values()

    def add(self, artifact: BentoServiceArtifact):
        super(ArtifactCollection, self).__setitem__(artifact.name, artifact)

    def save(self, dst):
        """Save all BentoServiceArtifact instances in self.values() to `dst` path
        if they are `packed` or `loaded`
        """
        save_path = os.path.join(dst, ARTIFACTS_DIR_NAME)
        os.mkdir(save_path)
        for artifact in self.get_artifact_list():
            if artifact.is_ready:
                artifact.save(save_path)
            else:
                logger.warning(
                    "Skip saving empty artifact, %s('%s')",
                    artifact.__class__.__name__,
                    artifact.name,
                )

    @classmethod
    def from_declared_artifact_list(cls, artifacts_list: List[BentoServiceArtifact]):
        artifact_collection = cls()
        for artifact in artifacts_list:
            artifact_collection.add(artifact._copy())
        return artifact_collection

    def load_all(self, path):
        for artifact in self.values():
            artifact.load(path)
