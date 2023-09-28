from __future__ import annotations

import inspect
import typing as t

from pydantic import BaseModel
from pydantic import Field
from pydantic import RootModel
from pydantic import create_model
from typing_extensions import get_origin

from .models import IODescriptor
from .models import IOMixin


def get_input_spec(
    func: t.Callable[..., t.Any], *, skip_self: bool = False
) -> type[IODescriptor] | None:
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
        return t.cast(
            t.Type[IODescriptor],
            create_model(
                "Input", __module__=func.__module__, __base__=IODescriptor, **fields
            ),
        )
    except (ValueError, TypeError):
        return None


def get_output_spec(func: t.Callable[..., t.Any]) -> type[IODescriptor] | None:
    signature = inspect.signature(func)
    return_annotation = signature.return_annotation
    if return_annotation is inspect.Signature.empty:
        raise TypeError(f"Missing return type annotation for function {func}")

    origin = get_origin(return_annotation)
    if origin in (t.Iterator, t.AsyncGenerator, t.Generator):
        return_annotation = return_annotation[0]
    try:
        return ensure_io_descriptor(return_annotation)
    except (ValueError, TypeError):
        return None


def ensure_io_descriptor(output_type: type) -> type[IODescriptor] | None:
    if issubclass(output_type, BaseModel):
        if not issubclass(output_type, IODescriptor):

            class Output(output_type, IOMixin):
                pass

            return t.cast(t.Type[IODescriptor], Output)

        return output_type
    return t.cast(
        t.Type[IODescriptor],
        create_model("Output", __base__=(IOMixin, RootModel[output_type])),
    )


def get_underlying_for_root(model: type[RootModel[t.Any]]) -> type:
    while RootModel not in model.__bases__:
        if RootModel in model.__bases__:
            break
        try:
            model = next(
                base for base in model.__bases__ if issubclass(base, RootModel)
            )
        except StopIteration:
            raise ValueError(f"Cannot find underlying model for {model}")
    return model.__pydantic_generic_metadata__["args"][0]
