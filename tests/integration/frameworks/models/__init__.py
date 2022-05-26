from __future__ import annotations

import typing as t

import attr

from bentoml._internal.runner.resource import Resource


@attr.define
class FrameworkTestModel:
    name: str
    model: t.Any
    configurations: list[FrameworkTestModelConfiguration]

    save_kwargs: dict[str, t.Any] = attr.Factory(dict)


@attr.define
class FrameworkTestModelConfiguration:
    test_inputs: dict[str, list[FrameworkTestModelInput]]

    load_kwargs: dict[str, t.Any] = attr.Factory(dict)
    check_model: t.Callable[[t.Any, Resource], None] = lambda _, __: None


@attr.define
class FrameworkTestModelInput:
    input_args: list[t.Any]
    expected: t.Any | t.Callable[[t.Any], bool]
    input_kwargs: dict[str, t.Any] = attr.Factory(dict)

    preprocess: t.Callable[[t.Any], t.Any] = lambda v: v

    def check_output(self, outp: t.Any):
        if isinstance(self.expected, t.Callable):
            assert self.expected(
                outp
            ), f"Output from model call ({', '.join(map(str, self.input_args))}, **{self.input_kwargs}) is not as expected"
        else:
            assert (
                outp == self.expected
            ), f"Output from model call ({', '.join(map(str, self.input_args))}, **{self.input_kwargs}) is not as expected"
