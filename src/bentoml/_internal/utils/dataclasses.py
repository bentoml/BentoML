import json
import typing as t
from dataclasses import asdict
from dataclasses import fields as get_fields
from dataclasses import is_dataclass


class DataclassJsonEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, o: t.Any):  # pylint: disable=method-hidden
        if is_dataclass(o):
            if hasattr(o, "to_json"):
                return o.to_json()
            else:
                return asdict(o)
        return super().default(o)


class json_serializer:
    def __init__(self, fields: t.Optional[t.List[str]] = None, compat: bool = False):
        self.fields = fields
        self.compat = compat

    @staticmethod
    def _extract_nested(obj: t.Any):
        if hasattr(obj, "to_json"):
            return obj.to_json()
        return obj

    T = t.TypeVar(
        "T",
    )

    def __call__(self, klass: t.Type[T]) -> t.Type[T]:
        if not is_dataclass(klass):
            raise TypeError(
                f"{self.__class__.__name__} only accepts dataclasses, "
                f"got {klass.__name__}"
            )
        default_map = {
            f.name: f.default_factory() if callable(f.default_factory) else f.default
            for f in get_fields(klass)
        }

        if self.fields is None:
            _fields = tuple(k for k in default_map.keys() if not k.startswith("_"))
        else:
            _fields = self.fields

        if self.compat:

            def to_json_compat(data_obj: t.Any):
                return {
                    k: self._extract_nested(getattr(data_obj, k))
                    for k in _fields
                    if default_map[k] != getattr(data_obj, k)
                }

            klass.to_json = to_json_compat  # type: ignore

        else:

            def to_json(data_obj: t.Any):
                return {k: self._extract_nested(getattr(data_obj, k)) for k in _fields}

            klass.to_json = to_json  # type: ignore

        return klass
