from __future__ import annotations

import abc
import typing as t

import attrs
from fs.base import FS
from simple_di import Provide
from simple_di import inject

from bentoml._internal.bento.bento import BentoModelInfo
from bentoml._internal.cloud import BentoCloudClient
from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models import Model as StoredModel
from bentoml._internal.models import ModelStore
from bentoml._internal.models.model import copy_model
from bentoml._internal.tag import Tag
from bentoml._internal.types import PathType
from bentoml.exceptions import NotFound

if t.TYPE_CHECKING:
    from bentoml._internal.cloud.schemas.schemasv1 import ModelSchema

T = t.TypeVar("T")


class Model(abc.ABC, t.Generic[T]):
    """A model reference to a artifact in various registries."""

    @property
    @abc.abstractmethod
    def revision(self) -> str:
        """Get the revision of the model."""

    @abc.abstractmethod
    def to_info(self, alias: str | None = None) -> BentoModelInfo:
        """Return the model info object."""

    @abc.abstractmethod
    def resolve(self, base_path: t.Union[PathType, FS, None] = None) -> T:
        """Get the actual object of the model."""

    @t.overload
    def __get__(self, instance: None, owner: t.Type[t.Any]) -> t.Self: ...

    @t.overload
    def __get__(self, instance: t.Any, owner: t.Type[t.Any]) -> T: ...

    def __get__(self, instance: t.Any, owner: type) -> T | t.Self:
        if instance is None:
            return self
        if getattr(self, "__resolved", None) is None:
            self.__resolved = self.resolve()
        return self.__resolved


@attrs.frozen
class BentoModel(Model[StoredModel]):
    tag: Tag = attrs.field(converter=Tag.from_taglike)

    @property
    def revision(self) -> str:
        if (stored := self.stored) is not None:
            return stored.tag.version
        model = self._get_remote_model()
        if model is None:
            raise NotFound(f"Model {self.tag} not found either locally or remotely.")
        return model.version

    @inject
    def _get_remote_model(
        self,
        rest_client: RestApiClient = Provide[BentoMLContainer.rest_api_client],
    ) -> ModelSchema | None:
        if self.tag.version in (None, "latest"):
            return rest_client.v1.get_latest_model(self.tag.name)
        else:
            return rest_client.v1.get_model(self.tag.name, self.tag.version)

    @property
    @inject
    def stored(
        self, model_store: ModelStore = Provide[BentoMLContainer.model_store]
    ) -> StoredModel | None:
        try:
            return model_store.get(self.tag)
        except NotFound:
            return None

    def to_info(self, alias: str | None = None) -> BentoModelInfo:
        stored = self.stored
        if stored is not None:
            return BentoModelInfo.from_bento_model(stored, alias)
        model = self._get_remote_model()
        if model is None:
            raise NotFound(f"Model {self.tag} not found either locally or remotely.")
        tag = Tag(self.tag.name, model.version)
        return BentoModelInfo(
            tag=tag,
            alias=alias,
            module=model.manifest.module,
            creation_time=model.created_at,
        )

    @inject
    def resolve(
        self,
        base_path: t.Union[PathType, FS, None] = None,
        global_model_store: ModelStore = Provide[BentoMLContainer.model_store],
        cloud_client: BentoCloudClient = Provide[BentoMLContainer.bentocloud_client],
    ) -> StoredModel:
        stored = self.stored
        if base_path is not None:
            model_store = ModelStore(base_path)
        else:
            model_store = global_model_store
        if stored is not None:
            if base_path is not None:
                copy_model(
                    stored.tag,
                    src_model_store=global_model_store,
                    target_model_store=model_store,
                )
            return stored
        return cloud_client.pull_model(self.tag, model_store=model_store)
