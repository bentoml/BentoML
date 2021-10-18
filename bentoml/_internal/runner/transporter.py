import abc
import typing as t
from typing import TYPE_CHECKING

import cloudpickle
from simple_di import Provide, inject

from bentoml._internal.configuration.containers import BentoMLContainer

from .utils import TypeRef

if TYPE_CHECKING:
    import numpy as np


DataType = t.TypeVar("DataType")
PayloadType = t.TypeVar("PayloadType")


class Transporter(t.Generic[DataType, PayloadType]):
    @abc.abstractmethod
    def to_payload(self, data: DataType) -> PayloadType:
        ...

    @abc.abstractmethod
    def from_payload(self, payload: PayloadType) -> DataType:
        ...


class NdarrayTransporter(Transporter["np.ndarray", bytes]):
    @inject
    def __init__(self, plasma_client=Provide[BentoMLContainer.plasma_db]):
        self.plasma_client = plasma_client

    def to_payload(self, data):
        if self.plasma_client:
            return self.plasma_client.put(data).binary()

        return cloudpickle.dumps(data)

    def from_payload(self, payload):
        if self.plasma_client:
            import pyarrow.plasma as plasma

            return self.plasma_client.get(plasma.ObjectID(payload))

        return cloudpickle.loads(payload)


class DefaultTransporter(Transporter[t.Any, bytes]):
    def to_payload(self, data):
        return cloudpickle.dumps(data)

    def from_payload(self, payload):
        return cloudpickle.loads(payload)


class TransporterRegistry:
    TRANSPORTER_TYPE_MAP: t.Dict[TypeRef, t.Type[Transporter]] = dict()

    @classmethod
    def register_transporter(
        cls,
        data_type: t.Union[TypeRef, type],
        transporter_cls: t.Type[Transporter],
    ):
        data_type = TypeRef.from_type(data_type)
        cls.TRANSPORTER_TYPE_MAP[data_type] = transporter_cls

    @classmethod
    def find_by_type(cls, type_: type) -> t.Type["Transporter"]:
        typeref = TypeRef.from_type(type_)
        matched = cls.TRANSPORTER_TYPE_MAP.get(typeref)
        if matched:
            return matched
        return DefaultTransporter


def register_builtin_transporters():
    TransporterRegistry.register_transporter(
        TypeRef("numpy", "ndarray"), NdarrayTransporter
    )


register_builtin_transporters()


def data_to_payload(data):
    transporter_cls = TransporterRegistry.find_by_type(type(data))
    transporter = transporter_cls()
    return transporter.to_payload(data)
