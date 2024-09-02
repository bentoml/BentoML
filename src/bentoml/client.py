"""
Bento Client.
=============

See https://docs.bentoml.com/en/latest/guides/client.html for more information.

.. code-block:: python

    import bentoml

    client = bentoml.client.Client.from_url("localhost:3000")

    client.predict(np.array([[5.9, 3, 5.1, 1.8]]))
"""

from __future__ import annotations

from ._internal.client import AsyncClient
from ._internal.client import Client
from ._internal.client import SyncClient
from ._internal.client.grpc import AsyncGrpcClient
from ._internal.client.grpc import GrpcClient
from ._internal.client.grpc import SyncGrpcClient
from ._internal.client.http import AsyncHTTPClient
from ._internal.client.http import HTTPClient
from ._internal.client.http import SyncHTTPClient

__all__ = [
    "AsyncClient",
    "SyncClient",
    "Client",
    "AsyncHTTPClient",
    "SyncHTTPClient",
    "HTTPClient",
    "AsyncGrpcClient",
    "SyncGrpcClient",
    "GrpcClient",
]
