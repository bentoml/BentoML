from __future__ import annotations

import inspect
import typing as t

from pydantic import BaseModel
from pydantic import Field
from pydantic import RootModel
from pydantic import create_model
from typing_extensions import get_origin


def get_input_spec(
    func: t.Callable[..., t.Any], *, skip_self: bool = False
) -> type[BaseModel] | None:
    signature = inspect.signature(func)

    fields: dict[str, tuple[str, t.Any]] = {}
    parameter_tuples = iter(signature.parameters.items())
    if skip_self:
        next(parameter_tuples)

    for name, param in parameter_tuples:
        if name in ("context", "ctx"):
            # Reserved name for context object passed in
            continue
        if param.kind in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            # FIXME: we are not able to create input model from var args
            return None
        annotation = param.annotation
        if annotation is param.empty:
            raise TypeError(
                f"Missing type annotation for parameter {name} in function {func}"
            )
        default = param.default
        if default is param.empty:
            default = Field()
        fields[name] = (annotation, default)

    try:
        return create_model("Input", __module__=func.__module__, **fields)
    except (ValueError, TypeError):
        return None


def get_output_spec(func: t.Callable[..., t.Any]) -> type[BaseModel] | None:
    signature = inspect.signature(func)
    return_annotation = signature.return_annotation
    if return_annotation is inspect.Signature.empty:
        raise TypeError(f"Missing return type annotation for function {func}")

    origin = get_origin(return_annotation)
    if origin in (t.Iterator, t.AsyncGenerator, t.Generator):
        return_annotation = return_annotation[0]
    return ensure_output_spec(return_annotation)


def ensure_output_spec(output_type: type) -> type[BaseModel] | None:
    if issubclass(output_type, BaseModel):
        return output_type

    try:
        return create_model("Output", __base__=RootModel[output_type])
    except (ValueError, TypeError):
        return None
