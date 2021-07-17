import json
from typing import Any

class DataclassJsonEncoder(json.JSONEncoder):
    def default(self, o): ...

class json_serializer:
    fields: Any
    compat: Any
    def __init__(self, fields: Any | None = ..., compat: bool = ...) -> None: ...
    def __call__(self, klass): ...
