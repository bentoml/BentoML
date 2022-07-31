from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from ...utils import LazyLoader

if TYPE_CHECKING:
    import pydantic
else:
    pydantic = LazyLoader(
        "pydantic",
        globals(),
        "pydantic",
        exc_msg="Missing required dependency: 'pydantic'. Install with 'pip install pydantic'.",
    )
