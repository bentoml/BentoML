from typing import TYPE_CHECKING

from simple_di import inject
from simple_di import Provide

from .. import calc_dir_size
from .schemas import BentoBuildEvent
from ...configuration.containers import BentoMLContainer

if TYPE_CHECKING:
    from ...models import ModelStore
    from ...bento.bento import Bento


__all__ = ["get_event_cli_build"]


@inject
def get_event_cli_build(
    *,
    bento: "Bento",
    model_store: "ModelStore" = Provide[BentoMLContainer.model_store],
) -> BentoBuildEvent:  # pragma: no cover
    return BentoBuildEvent(
        bento_creation_timestamp=bento.info.creation_time,
        bento_size_in_kb=calc_dir_size(bento._fs.getsyspath("/")),
        model_size_in_kb=calc_dir_size(bento._fs.getsyspath("/models")),
        num_of_models=len(bento.info.models),
        num_of_runners=len(bento.info.runners),
        model_types=[model_store.get(i).info.module for i in bento.info.models],
        runner_types=[v for v in bento.info.runners.values()],
    )
