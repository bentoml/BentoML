import typing as t
from dataclasses import dataclass

import requests

from bentoml.io import IODescriptor


@dataclass
class Endpoint:
    name: str
    input_io_desc: IODescriptor[t.Any]
    output_io_desc: IODescriptor[t.Any]

    def serialize(self, obj: t.Any) -> str:
        return self.input_io_desc.serialize(obj)

    def deserialize(self, obj: bytes) -> t.Any:
        return self.output_io_desc.deserialize(obj)


class TestClient:
    def __init__(
        self,
        host: str = "localhost",
        port: t.Union[int, str] = 3000,
        endpoints: t.Optional[t.Sequence[Endpoint]] = None,
    ):
        self._host = host
        self._port = port
        self.endpoints = dict()
        if endpoints is not None:
            for endpoint in endpoints:
                self.endpoints[endpoint.name] = endpoint

    def add_endpoint(self, endpoint: Endpoint):
        self.endpoints[endpoint.name] = endpoint

    def remove_endpoint(self, endpoint: Endpoint):
        del self.endpoints[endpoint.name]

    @property
    def _url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def get_prediction(self, endpoint: str, data: t.Any) -> t.Any:
        if endpoint not in self.endpoints:
            raise ValueError(f"Endpoint {endpoint} not found")
        url = f"{self._url}/{endpoint}"
        serialized_data = self.endpoints[endpoint].serialize(data)
        res = requests.post(url, data=serialized_data)
        res.raise_for_status()
        deserialized_data = self.endpoints[endpoint].deserialize(res.content)
        return deserialized_data

    def _readyz(self) -> requests.Response:
        url = f"{self._url}/readyz"
        res = requests.get(url)
        return res

    def is_ready(self) -> bool:
        try:
            res = self._readyz()
            res.raise_for_status()
        except:
            return False
        return True
