import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from bentoml._internal.service import Service


def load_bento(bento_path_or_tag: str) -> "Service":
    raise NotImplementedError()
