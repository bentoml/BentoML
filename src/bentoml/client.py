"""
Bento Client.
=============

See https://docs.bentoml.org/en/latest/guides/client.html for more information.

.. code-block:: python

    import bentoml

    client = bentoml.client.Client.from_url("localhost:3000")

    client.predict(np.array([[5.9, 3, 5.1, 1.8]]))
"""
from __future__ import annotations

from ._internal.client import Client
from ._internal.client.grpc import GrpcClient
from ._internal.client.http import HTTPClient

__all__ = ["Client", "HTTPClient", "GrpcClient"]
