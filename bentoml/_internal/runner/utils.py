from __future__ import annotations

import typing as t
import logging
import itertools
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from aiohttp import MultipartWriter
    from starlette.requests import Request

    from ..runner.container import Payload


T = t.TypeVar("T")
To = t.TypeVar("To")


CUDA_SUCCESS = 0


class Params(t.Generic[T]):
    args: tuple[T, ...]
    kwargs: dict[str, T]

    def __init__(
        self,
        *args: T,
        **kwargs: T,
    ):
        self.args = args
        self.kwargs = kwargs

    def map(self, function: t.Callable[[T], To]) -> Params[To]:
        args = tuple(function(a) for a in self.args)
        kwargs = {k: function(v) for k, v in self.kwargs.items()}
        return Params[To](*args, **kwargs)

    def imap(self, function: t.Callable[[T], t.Iterable[To]]) -> t.Iterator[Params[To]]:
        args_iter = tuple(iter(function(a)) for a in self.args)
        kwargs_iter = {k: iter(function(v)) for k, v in self.kwargs.items()}

        try:
            while True:
                args = tuple(next(a) for a in args_iter)
                kwargs = {k: next(v) for k, v in kwargs_iter.items()}
                yield Params[To](*args, **kwargs)
        except StopIteration:
            pass

    def items(self) -> t.Iterator[t.Tuple[t.Union[int, str], T]]:
        return itertools.chain(enumerate(self.args), self.kwargs.items())

    @classmethod
    def agg(
        cls,
        params_list: t.Sequence[Params[T]],
        agg_func: t.Callable[[t.Sequence[T], int], To] = lambda b, _: b,
    ) -> Params[To]:
        if not params_list:
            return Params()

        args: t.List[To] = []
        kwargs: t.Dict[str, To] = {}

        for j, _ in enumerate(params_list[0].args):
            arg: t.List[T] = []
            for params in params_list:
                arg.append(params.args[j])
            args.append(agg_func(arg, j))
        for k in params_list[0].kwargs:
            kwarg: t.List[T] = []
            for params in params_list:
                kwarg.append(params.kwargs[k])
            kwargs[k] = agg_func(kwarg, -1)
        return Params(*tuple(args), **kwargs)

    @property
    def sample(self) -> T:
        if self.args:
            return self.args[0]
        return next(iter(self.kwargs.values()))


PAYLOAD_META_HEADER = "Bento-Payload-Meta"


def payload_params_to_multipart(params: Params[Payload]) -> MultipartWriter:
    import json

    from multidict import CIMultiDict
    from aiohttp.multipart import MultipartWriter

    multipart = MultipartWriter(subtype="form-data")
    for key, payload in params.items():
        multipart.append(
            payload.data,
            headers=CIMultiDict(
                (
                    (PAYLOAD_META_HEADER, json.dumps(payload.meta)),
                    ("Content-Type", f"application/vnd+bentoml.{payload.container}"),
                    ("Content-Disposition", f'form-data; name="{key}"'),
                )
            ),
        )
    return multipart


async def multipart_to_payload_params(request: Request) -> Params[Payload]:
    import json

    from bentoml._internal.runner.container import Payload
    from bentoml._internal.utils.formparser import populate_multipart_requests

    parts = await populate_multipart_requests(request)
    max_arg_index = -1
    kwargs: t.Dict[str, Payload] = {}
    args_map: t.Dict[int, Payload] = {}
    for field_name, req in parts.items():
        payload = Payload(
            data=await req.body(),
            meta=json.loads(req.headers[PAYLOAD_META_HEADER]),
            container=req.headers["Content-Type"].strip("application/vnd+bentoml."),
        )
        if field_name.isdigit():
            arg_index = int(field_name)
            args_map[arg_index] = payload
            max_arg_index = max(max_arg_index, arg_index)
        else:
            kwargs[field_name] = payload
    args = tuple(args_map[i] for i in range(max_arg_index + 1))
    return Params(*args, **kwargs)
