import typing
from typing import Text, Tuple

if typing.TYPE_CHECKING: ...

def make_repr(
    class_name: Text, *args: object, **kwargs: Tuple[object, object]
) -> Text: ...
