from typing import NotRequired
from typing import TypedDict

Impl = TypedDict("Impl", {"class": str, "err": NotRequired[str]})

known_implementations: dict[str, Impl]
