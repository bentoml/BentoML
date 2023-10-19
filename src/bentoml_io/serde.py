import abc
import io
import json
import pickle
import typing as t

from pydantic import BaseModel
from pydantic import RootModel

T = t.TypeVar("T", bound=BaseModel)


class Serde(abc.ABC):
    media_type: str

    @abc.abstractmethod
    def serialize_model(self, model: BaseModel) -> bytes:
        ...

    @abc.abstractmethod
    def deserialize_model(self, model_bytes: bytes, cls: type[T]) -> T:
        ...

    @abc.abstractmethod
    def serialize(self, obj: t.Any) -> bytes:
        ...

    @abc.abstractmethod
    def deserialize(self, obj_bytes: bytes) -> t.Any:
        ...


class JSONSerde(Serde):
    media_type = "application/json"

    def serialize_model(self, model: BaseModel) -> bytes:
        return model.model_dump_json().encode("utf-8")

    def deserialize_model(self, model_bytes: bytes, cls: type[T]) -> T:
        return cls.model_validate_json(model_bytes)

    def serialize(self, obj: t.Any) -> bytes:
        return json.dumps(obj).encode("utf-8")

    def deserialize(self, obj_bytes: bytes) -> t.Any:
        return json.loads(obj_bytes)


class PickleSerde(Serde):
    media_type = "application/vnd.bentoml+pickle"

    def serialize_model(self, model: BaseModel) -> bytes:
        if isinstance(model, RootModel):
            model_data: t.Any = model.root
        else:
            model_data = {k: getattr(model, k) for k in model.model_fields}
        return pickle.dumps(model_data)

    def deserialize_model(self, model_bytes: bytes, cls: type[T]) -> T:
        obj = pickle.loads(model_bytes)
        if not isinstance(obj, cls):
            obj = cls.model_validate(obj)
        return obj

    def serialize(self, obj: t.Any) -> bytes:
        return pickle.dumps(obj)

    def deserialize(self, obj_bytes: bytes) -> t.Any:
        return pickle.loads(obj_bytes)


class ArrowSerde(Serde):
    media_type = "application/vnd.bentoml+arrow"

    def serialize_model(self, model: BaseModel) -> bytes:
        from .arrow import serialize_to_arrow

        buffer = io.BytesIO()
        serialize_to_arrow(model, buffer)
        return buffer.getvalue()

    def deserialize_model(self, model_bytes: bytes, cls: type[T]) -> T:
        from .arrow import deserialize_from_arrow

        buffer = io.BytesIO(model_bytes)
        return deserialize_from_arrow(cls, buffer)

    def serialize(self, obj: t.Any) -> bytes:
        raise NotImplementedError(
            "Serializing arbitrary object to Arrow is not supported"
        )

    def deserialize(self, obj_bytes: bytes) -> t.Any:
        raise NotImplementedError(
            "Deserializing arbitrary object from Arrow is not supported"
        )


ALL_SERDE: t.Mapping[str, type[Serde]] = {
    s.media_type: s for s in [JSONSerde, PickleSerde, ArrowSerde]
}
