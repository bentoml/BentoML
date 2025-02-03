from __future__ import annotations

import typing as t

import attr
import numpy as np


@attr.define
class FrameworkTestModel:
    name: str
    model: t.Any
    configurations: list[FrameworkTestModelConfiguration]

    save_kwargs: dict[str, t.Any] = attr.Factory(dict)
    # when raw model method call is not simply `method(*args,
    # **kwargs)` format or raw model method call does not simply
    # return the outputs, then use this to override default behavior
    # when testing raw model inputs with expected outputs
    model_method_caller: (
        t.Callable[
            [FrameworkTestModel, str, tuple[t.Any, ...], dict[str, t.Any]], t.Any
        ]
        | None
    ) = attr.field(default=None)
    # when framework has some special signatures requirements
    model_signatures: dict[str, t.Any] | None = attr.field(default=None)


@attr.define
class FrameworkTestModelConfiguration:
    test_inputs: dict[str, list[FrameworkTestModelInput]]
    load_kwargs: dict[str, t.Any] = attr.Factory(dict)
    check_model: t.Callable[[t.Any, dict[str, t.Any]], None] = (  # noqa: E731
        lambda _, __: None
    )
    check_runnable: t.Callable[[t.Any, dict[str, t.Any]], None] = (  # noqa: E731
        lambda _, __: None
    )


@attr.define
class FrameworkTestModelInput:
    input_args: list[t.Any]
    expected: t.Any | t.Callable[[t.Any], bool | None]
    input_kwargs: dict[str, t.Any] = attr.Factory(dict)

    def check_output(self, outp: t.Any):
        if isinstance(self.expected, t.Callable):
            result = self.expected(outp)
            if result is not None:
                assert result, (
                    f"Output from model call (args={', '.join(map(str, self.input_args))}, kwargs={self.input_kwargs}) is not expected (output={outp})"
                )
        else:
            check = outp == self.expected
            if isinstance(check, np.ndarray):
                check = check.all()
            assert check, (
                f"Output from model call (args={', '.join(map(str, self.input_args))}, kwargs={self.input_kwargs}) is not expected (output={outp}, expected={self.expected})"
            )
