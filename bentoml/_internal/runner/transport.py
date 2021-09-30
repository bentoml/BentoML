from typing import Generic, TypeVar

from simple_di import Provide, inject

from bentoml._internal.configuration.containers import BentoMLContainer


SingleType = TypeVar('SingleType')
BatchType = TypeVar('BatchType')


class Transporter:
    def to_payload(self, input_):
        ...

    def from_payload(self, payload):
        ...


class Container:
    def to_payload(self, input_):
        ...

    def from_payload(self, payload):
        ...


class NdarrayContainer(Container):
    @inject
    def __init__(self, plasma=Provide[BentoMLContainer.plasma_db]):
        self._plasma = plasma

    def to_payload(self, input_):
        pid = self._plasma.put(input_)
        payload = dict(
            type=self.__class__.__name__,
        )
        return

    def from_payload(self, payload):
        ...


def get_transporter(type_: type) -> Transporter:
    import numpy

    if type_ is numpy.ndarray:
        return PlasmaNdarrayTransporter()

    raise TypeError("type must be numpy.ndarray")
