import ast
import inspect
import typing as t
from pathlib import Path

from bentoml._internal.models import Model


def assert_have_file_extension(directory: str, ext: str):
    _dir = Path(directory)
    assert _dir.is_dir(), f"{directory} is not a directory"
    assert any(f.suffix == ext for f in _dir.iterdir())


def get_model_annotations(func: t.Callable[..., t.Any]) -> bool:
    source = inspect.getsource(func)
    mod = ast.parse(source)
    assert mod.body, "Invalid AST parse"
    body = getattr(mod, "body")[0]
    model_args = body.args.args[0]
    if model_args.annotation.s == Model.__name__:
        return True
    return False
