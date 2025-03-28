from __future__ import annotations

import types
from typing import Any
from typing import TypeVar
from typing import overload

from pydantic import BaseModel
from pydantic import ValidationError
from simple_di import Provide
from simple_di import inject

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import InvalidArgument

Model = TypeVar("Model", bound=BaseModel)


@overload
def use_arguments(model: type[Model]) -> Model: ...


@overload
def use_arguments(model: None = None) -> types.SimpleNamespace: ...


@inject
def use_arguments(
    model: type[Model] | None = None,
    *,
    arguments: dict[str, Any] = Provide[BentoMLContainer.bento_arguments],
) -> Model | types.SimpleNamespace:
    """Declare and use bento arguments in the current module. Values should be provided by CLI.

    If a pydantic.BaseModel is passsed, it will used to parse and validate the arguments.
    Otherwise, a namespace object will be returned so that you can access the values by attribute.

    Example:

    ```python
    from pydantic import BaseModel

    class MyArgs(BaseModel):
        name: str
        kind: str = "animal"

    args = use_arguments(MyArgs)
    # $ bentoml serve --arg name=dog
    print(args.name) # dog
    print(args.kind) # animal
    ```
    """
    if model is None:
        return types.SimpleNamespace(**arguments)
    try:
        return model(**arguments)
    except ValidationError as e:
        raise InvalidArgument(
            f"Argument error. Please provide correct arguments via --arg option: {e}"
        ) from e


def set_arguments(**arguments: Any) -> None:
    """Set the arguments to be used by use_arguments."""
    BentoMLContainer.bento_arguments.set(arguments)
