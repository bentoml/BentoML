import typing as t
from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from bentoml import Runner
from bentoml import SimpleRunner

from ...types import Tag
from ...configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ....models import ModelStore


class BaseModelRunner(Runner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        name: t.Optional[str] = None,
    ):
        super().__init__(name)
        self._tag = Tag.from_taglike(tag)

    @property
    @inject
    def _model_info(
        self,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        return model_store.get(self._tag)

    @property
    def default_name(self) -> str:
        return self._tag.name

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]


class BaseModelSimpleRunner(SimpleRunner):
    def __init__(
        self,
        tag: t.Union[str, Tag],
        name: t.Optional[str] = None,
    ):
        super().__init__(name)
        self._tag = Tag.from_taglike(tag)

    @property
    @inject
    def _model_info(
        self,
        model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
    ):
        return model_store.get(self._tag)

    @property
    def default_name(self) -> str:
        return self._tag.name

    @property
    def required_models(self) -> t.List[Tag]:
        return [self._tag]
