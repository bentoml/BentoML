from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from .schemas import BentoBuildEvent

if TYPE_CHECKING:
    from .schemas import EventMeta
    from ...bento.bento import Bento

    class AnalyticCliProtocol(t.Protocol):
        def __call__(
            self, cmd_group: str, cmd_name: str, return_value: t.Any
        ) -> EventMeta:
            ...


def _cli_bentoml_build_event(
    cmd_group: str,
    cmd_name: str,
    return_value: Bento | None,
) -> BentoBuildEvent:  # pragma: no cover
    from .. import calc_dir_size

    if return_value is not None:
        bento = return_value
        return BentoBuildEvent(
            cmd_group=cmd_group,
            cmd_name=cmd_name,
            bento_creation_timestamp=bento.info.creation_time,
            bento_size_in_kb=calc_dir_size(bento.path_of("/")) / 1024,
            model_size_in_kb=calc_dir_size(bento.path_of("/models")) / 1024,
            num_of_models=len(bento.info.models),
            num_of_runners=len(bento.info.runners),
            model_types=[m.module for m in bento.info.models],
            runnable_types=[r.runnable_type for r in bento.info.runners],
        )
    else:
        return BentoBuildEvent(
            cmd_group=cmd_group,
            cmd_name=cmd_name,
        )


cli_events_map: dict[str, dict[str, AnalyticCliProtocol]] = {
    "cli": {"build": _cli_bentoml_build_event}
}
