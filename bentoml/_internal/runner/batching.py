from __future__ import annotations

import typing as t
from typing import TYPE_CHECKING

from bentoml.exceptions import InvalidArgument

from .utils import Params
from .container import AutoContainer

if TYPE_CHECKING:
    from .container import Payload


def batch_params(
    paramss: t.Sequence[Params[Payload]],
    batch_dim: int,
    # TODO: support mapping from arg to batch dimension
) -> tuple[Params[t.Any], list[int]]:
    if len(paramss) == 0:
        return (Params(), [])

    args: list[t.Any] = []
    kwargs: dict[str, t.Any] = {}

    if len(paramss[0].args) > 0:
        arg_idxs = iter(range(len(paramss[0].args)))

        idx = next(arg_idxs)
        batched, first_indices = AutoContainer.from_batch_payloads(
            [params.args[idx] for params in paramss],
            batch_dim=batch_dim,
        )
        args.append(batched)

        for idx in arg_idxs:
            batched, indices = AutoContainer.from_batch_payloads(
                [params.args[idx] for params in paramss],
                batch_dim=batch_dim,
            )
            args.append(batched)

            if first_indices != indices:
                raise InvalidArgument(
                    f"argument lengths for parameter {idx} do not match the arguments lengths for the first argument"
                )

        kwarg_keys = iter(paramss[0].kwargs.keys())
    else:
        kwarg_keys = iter(paramss[0].kwargs.keys())

        key = next(kwarg_keys)
        batched, first_indices = AutoContainer.from_batch_payloads(
            [params.kwargs[key] for params in paramss],
            batch_dim=batch_dim,
        )
        kwargs[key] = batched

    for key in kwarg_keys:
        batched, indices = AutoContainer.from_batch_payloads(
            [params.kwargs[key] for params in paramss],
            batch_dim=batch_dim,
        )
        kwargs[key] = batched

        if first_indices != indices:
            raise InvalidArgument(
                f"argument lengths for parameter '{key}' do not match the arguments lengths for the first argument"
            )

    return (Params(*args, **kwargs), first_indices)
