from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import BentoBuildEvent

if TYPE_CHECKING:
    from ...bento.bento import Bento


def _cli_bentoml_build_event(
    cmd_group: str,
    cmd_name: str,
    return_value: Bento | None,
) -> BentoBuildEvent:  # pragma: no cover
    from ...bento.bento import BentoInfo
    from ...bento.bento import BentoInfoV2

    if return_value is None:
        return BentoBuildEvent(cmd_group=cmd_group, cmd_name=cmd_name)
    bento = return_value
    total_size = bento.total_size()
    if isinstance(bento.info, BentoInfoV2):
        num_of_runners = len(bento.info.services) - 1
    elif isinstance(bento.info, BentoInfo):
        num_of_runners = len(bento.info.runners)
    else:
        num_of_runners = 0
    return BentoBuildEvent(
        cmd_group=cmd_group,
        cmd_name=cmd_name,
        bento_creation_timestamp=bento.info.creation_time,
        bento_size_in_kb=bento.file_size / 1024,
        model_size_in_kb=(total_size - bento.file_size) / 1024,
        num_of_models=len(bento.info.all_models),
        num_of_runners=num_of_runners,
        model_types=[m.module or "" for m in bento.info.all_models],
    )


cli_events_map = {"bentos": {"build": _cli_bentoml_build_event}}
