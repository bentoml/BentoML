from __future__ import annotations

import typing as t
from abc import ABCMeta
from abc import abstractmethod
from http import HTTPStatus
from typing import TYPE_CHECKING
from functools import partial
from collections import namedtuple

from google.protobuf.message import Message

from bentoml.exceptions import BentoMLException

from ..lazy_loader import LazyLoader

if TYPE_CHECKING:
    from bentoml.grpc.v1 import struct_pb2

    CoderCallback = list[
        tuple[
            t.Callable[[t.Any], bool],
            t.Callable[[t.Any], bytes] | t.Callable[[bytes], t.Any],
        ]
    ]
else:
    struct_pb2 = LazyLoader("struct_pb2", globals(), "bentoml.grpc.v1.struct_pb2")


F = t.Callable[..., t.Any]
P = t.TypeVar("P")

__all__ = ["can_encode", "encode_structure", "decode_proto", "register_codec"]


def register_codec(cdc: t.Type[Codec[t.Any]]) -> None:
    # register a codec.
    _codecs.append(cdc())


def _get_encoders() -> CoderCallback:
    return [(c.can_encode, c.do_encode) for c in _codecs]


def _get_decoders() -> CoderCallback:
    return [(c.can_decode, c.do_decode) for c in _codecs]


def _map_structure(obj: t.Any, coders: CoderCallback) -> Message:
    for can, do in coders:
        if can(obj):
            fn = partial(_map_structure, coders=coders)
            return do(obj, fn)
    raise EncodeError(f"No codec found for {str(obj)} of type {type(obj)}")


def can_encode(nested: t.Any) -> bool:
    # Determines whether a nested structure can be encoded into a proto.
    try:
        encode_structure(nested)
    except EncodeError:
        return False
    return True


def encode_structure(nested: t.Any) -> Message:
    # encode given nested object to result protos.
    return _map_structure(nested, _get_encoders())


def decode_proto(proto: Message) -> t.Any:
    # decode given proto to nested object.
    return _map_structure(proto, _get_decoders())


class EncodeError(BentoMLException):
    """Error raised when a codec cannot encode an object or not implemented."""

    error_code = HTTPStatus.NOT_IMPLEMENTED


class Codec(t.Generic[P], metaclass=ABCMeta):
    @abstractmethod
    def can_encode(self, obj: t.Any) -> bool:
        """
        Check if the codec can encode the given object.
        """

    @abstractmethod
    def do_encode(self, obj: P, encode_fn: F) -> struct_pb2.ContentsProto:
        """
        Encode the given object.
        """

    def can_decode(self, value: Message) -> bool:
        """
        Check if the codec can decode the given bytes.
        """

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> P:
        """
        Decode the given bytes.
        """


class _NoneCodec(Codec[None]):
    def can_encode(self, obj: t.Any) -> bool:
        return obj is None

    def do_encode(self, obj: None, encode_fn: F) -> struct_pb2.ContentsProto:
        del obj, encode_fn
        value = struct_pb2.ContentsProto()
        value.none.CopyFrom(struct_pb2.NoneValue())
        return value

    def can_decode(self, value: Message) -> bool:
        return value.HasField("none")

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> None:
        del value, decode_fn
        return None


class _Float64Codec(Codec[float]):
    def can_encode(self, obj: t.Any) -> bool:
        return isinstance(obj, float)

    def do_encode(self, obj: float, encode_fn: F) -> struct_pb2.ContentsProto:
        del encode_fn
        val = struct_pb2.ContentsProto()
        val.float64 = obj
        return val

    def can_decode(self, value: Message) -> bool:
        return value.HasField("float64")

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> float:
        del decode_fn
        return float(value.float64)


class _Int64Codec(Codec[int]):
    def can_encode(self, obj: t.Any) -> bool:
        return not isinstance(obj, bool) and isinstance(obj, int)

    def do_encode(self, obj: int, encode_fn: F) -> struct_pb2.ContentsProto:
        del encode_fn
        value = struct_pb2.ContentsProto()
        value.int64 = obj
        return value

    def can_decode(self, value: Message) -> bool:
        return value.HasField("int64")

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> int:
        del decode_fn
        return int(value.int64)


class _StringCodec(Codec[str]):
    def can_encode(self, obj: t.Any) -> bool:
        return isinstance(obj, str)

    def do_encode(self, obj: str, encode_fn: F) -> struct_pb2.ContentsProto:
        del encode_fn
        val = struct_pb2.ContentsProto()
        val.string = obj
        return val

    def can_decode(self, value: Message) -> bool:
        return value.HasField("string")

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> str:
        del decode_fn
        return str(value.string)


class _BoolCodec(Codec[bool]):
    def can_encode(self, obj: t.Any) -> bool:
        return isinstance(obj, bool)

    def do_encode(self, obj: bool, encode_fn: F) -> struct_pb2.ContentsProto:
        del encode_fn
        val = struct_pb2.ContentsProto()
        val.bool = obj
        return val

    def can_decode(self, value: Message) -> bool:
        return value.HasField("bool")

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> bool:
        del decode_fn
        return value.bool


class _ShapeCodec(Codec[t.Tuple[t.List[t.Tuple[int, str]], bool]]):
    def can_encode(self, obj: t.Any) -> bool:
        return _is_tuple(obj)

    def do_encode(
        self, obj: t.Tuple[t.List[t.Tuple[int, str]], bool], encode_fn: F
    ) -> struct_pb2.ContentsProto:
        return struct_pb2.ContentsProto()

    def can_decode(self, value: Message) -> bool:
        return value.HasField("shape")

    def do_decode(
        self, value: struct_pb2.ContentsProto, decode_fn: F
    ) -> t.Tuple[t.List[t.Tuple[int, str]], bool]:
        return value


class _ListCodec(Codec[t.List[t.Any]]):
    def can_encode(self, obj: t.Any) -> bool:
        return isinstance(obj, list)

    def do_encode(self, obj: list[t.Any], encode_fn: F) -> struct_pb2.ContentsProto:
        encoded = struct_pb2.ContentsProto()
        encoded.list.CopyFrom(struct_pb2.ListValue())
        for element in obj:
            encoded.list.values.add().CopyFrom(encode_fn(element))
        return encoded

    def can_decode(self, value: Message) -> bool:
        return value.HasField("list")

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> list[t.Any]:
        return [decode_fn(element) for element in value.list.values]


class _TupleCodec(Codec[t.Tuple[t.Any, ...]]):
    def can_encode(self, obj: t.Any) -> bool:
        return _is_tuple(obj)

    def do_encode(
        self, obj: tuple[t.Any, ...], encode_fn: F
    ) -> struct_pb2.ContentsProto:
        encoded = struct_pb2.ContentsProto()
        encoded.tuple.CopyFrom(struct_pb2.TupleValue())
        for element in obj:
            encoded.tuple.values.add().CopyFrom(encode_fn(element))
        return encoded

    def can_decode(self, value: Message) -> bool:
        return value.HasField("tuple")

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> tuple[t.Any]:
        return tuple(decode_fn(element) for element in value.tuple.values)


def _is_tuple(obj: t.Any) -> bool:
    return not _is_named_tuple(obj) and isinstance(obj, tuple)


def _is_named_tuple(instance: type) -> bool:
    # Returns True iff `instance` is a `namedtuple`.
    if not isinstance(instance, tuple):
        return False
    return (
        hasattr(instance, "_fields")
        and isinstance(instance._fields, t.Sequence)
        and all(isinstance(f, str) for f in instance._fields)  # type: ignore
    )


class _DictCodec(Codec[t.Dict[str, t.Any]]):
    def can_encode(self, obj: t.Any) -> bool:
        return isinstance(obj, dict)

    def do_encode(
        self, obj: dict[str, t.Any], encode_fn: F
    ) -> struct_pb2.ContentsProto:
        encoded = struct_pb2.ContentsProto()
        encoded.dict.CopyFrom(struct_pb2.DictValue())
        for key, value in obj.items():
            encoded.dict.fields[key].CopyFrom(encode_fn(value))
        return encoded

    def can_decode(self, value: Message) -> bool:
        return value.HasField("dict")

    def do_decode(
        self, value: struct_pb2.ContentsProto, decode_fn: F
    ) -> dict[str, t.Any]:
        return {key: decode_fn(val) for key, val in value.dict.fields.items()}


class _NamedTupleCodec(Codec[t.NamedTuple]):
    def can_encode(self, obj: t.Any) -> bool:
        return _is_named_tuple(obj)

    def do_encode(self, obj: t.NamedTuple, encode_fn: F) -> struct_pb2.ContentsProto:
        encoded = struct_pb2.ContentsProto()
        encoded.namedtuple.CopyFrom(struct_pb2.NamedTupleValue())
        encoded.namedtuple.name = obj.__class__.__name__
        for key in obj._fields:
            pair = encoded.namedtuple.values.add()
            pair.key = key
            pair.value.CopyFrom(encode_fn(obj._asdict()[key]))
        return encoded

    def can_decode(self, value: Message) -> bool:
        return value.HasField("namedtuple")

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> t.NamedTuple:
        kv_pairs = value.namedtuple.values
        items = [(pair.key, decode_fn(pair.value)) for pair in kv_pairs]
        # TODO: makes this type safe by using t.NamedTuple
        namedtuple_klass = namedtuple(  # type: ignore (no type support yet)
            value.namedtuple.name, list(map(lambda x: x[0], items))
        )
        return namedtuple_klass(**dict(items))


class _DataFrameCodec(Codec[t.List[str]]):
    _df_type: t.Literal["dataframe_columns", "dataframe_indices"]

    def __new__(cls):
        assert cls._df_type in ["dataframe_columns", "dataframe_indices"]
        if cls._df_type == "dataframe_columns":
            res = object.__new__(_DataFrameColumnsCodec)
        elif cls._df_type == "dataframe_indices":
            res = object.__new__(_DataFrameIndicesCodec)
        else:
            raise EncodeError("Unsupported DataFrame type.")
        return res

    def can_encode(self, obj: t.Iterable[t.Any]) -> bool:
        return isinstance(obj, list) and all(isinstance(x, str) for x in obj)

    def do_encode(self, obj: list[t.Any], encode_fn: F) -> struct_pb2.ContentsProto:
        del encode_fn
        encoded = struct_pb2.ContentsProto()
        df = getattr(encoded, self._df_type)
        df.CopyFrom(struct_pb2.ListValue())
        for element in obj:
            df.values.add().CopyFrom(str(element))
        return encoded

    def can_decode(self, value: Message) -> bool:
        return value.HasField(self._df_type)

    def do_decode(self, value: struct_pb2.ContentsProto, decode_fn: F) -> list[str]:
        del decode_fn
        return [str(element) for element in getattr(value, self._df_type).values]


class _DataFrameColumnsCodec(_DataFrameCodec):
    _df_type = "dataframe_columns"


class _DataFrameIndicesCodec(_DataFrameCodec):
    _df_type = "dataframe_indices"


_codecs: list[Codec[t.Any]] = [
    _NoneCodec(),
    _Float64Codec(),
    _Int64Codec(),
    _StringCodec(),
    _BoolCodec(),
    _ListCodec(),
    _TupleCodec(),
    _DictCodec(),
    _NamedTupleCodec(),
    _DataFrameColumnsCodec(),
    _DataFrameIndicesCodec(),
]
