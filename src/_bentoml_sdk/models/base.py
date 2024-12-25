from __future__ import annotations

import abc
import typing as t

import attrs
from fs.base import FS
from simple_di import Provide
from simple_di import inject

from bentoml._internal.bento.bento import BentoModelInfo
from bentoml._internal.cloud.client import RestApiClient
from bentoml._internal.cloud.schemas.modelschemas import LabelItemSchema
from bentoml._internal.cloud.schemas.modelschemas import ModelManifestSchema
from bentoml._internal.cloud.schemas.schemasv1 import CreateModelSchema
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models import Model as StoredModel
from bentoml._internal.models import ModelStore
from bentoml._internal.models.model import copy_model
from bentoml._internal.tag import Tag
from bentoml._internal.types import PathType
from bentoml._internal.utils.filesystem import calc_dir_size
from bentoml.exceptions import NotFound

if t.TYPE_CHECKING:
    from bentoml._internal.cloud.schemas.schemasv1 import ModelSchema

T = t.TypeVar("T")


class Model(abc.ABC, t.Generic[T]):
    """A model reference to a artifact in various registries."""

    @abc.abstractmethod
    def to_info(self, alias: str | None = None) -> BentoModelInfo:
        """Return the model info object."""

    @classmethod
    @abc.abstractmethod
    def from_info(cls, info: BentoModelInfo) -> Model[T]:
        """Return the model object from the model info."""

    @abc.abstractmethod
    def to_create_schema(self) -> CreateModelSchema:
        """Return the create model schema object."""

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
        if getattr(self, "_Model__resolved", None) is None:
            object.__setattr__(self, "_Model__resolved", self.resolve())
        return self.__resolved  # type: ignore[attr-defined]


@attrs.frozen
class BentoModel(Model[StoredModel]):
    """A model reference to a BentoML model.

    Args:
        tag (Tag): The model tag.

    Returns:
        Model: The bento model object.
    """

    tag: Tag = attrs.field(converter=Tag.from_taglike)

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

    @classmethod
    def from_info(cls, info: BentoModelInfo) -> BentoModel:
        return cls(tag=info.tag)

    @inject
    def resolve(
        self,
        base_path: t.Union[PathType, FS, None] = None,
        global_model_store: ModelStore = Provide[BentoMLContainer.model_store],
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
        cloud_client = BentoMLContainer.bentocloud_client.get()
        model = cloud_client.model.pull(self.tag, model_store=model_store)
        assert model is not None, "non-bentoml model"
        return model

    def to_create_schema(self) -> CreateModelSchema:
        """Return the create model schema object."""
        stored = self.stored
        if stored is None:
            raise ValueError("Not allowed to create a model without local store")
        info = stored.info
        labels = [
            LabelItemSchema(key=key, value=value) for key, value in info.labels.items()
        ]
        return CreateModelSchema(
            description="",
            version=stored.tag.version or "",
            build_at=info.creation_time,
            labels=labels,
            manifest=ModelManifestSchema(
                module=info.module,
                metadata=info.metadata,
                context=info.context.to_dict(),
                options=info.options.to_dict(),
                api_version=info.api_version,
                bentoml_version=info.context.bentoml_version,
                size_bytes=calc_dir_size(stored.path),
            ),
        )

    def __str__(self) -> str:
        return str(self.tag)
