import logging

import numpy as np

from bentoml import HTTPServer
from bentoml.client import Client

logging.basicConfig(level=logging.WARN)

if __name__ == "__main__":
    server = HTTPServer("iris_classifier:latest", production=True, port=3000)

    # Start the server in a separate process and connect to it using a client
    with server.start() as client:
        res = client.classify(np.array([[4.9, 3.0, 1.4, 0.2]]))
        print(f"Successfully received results, {res}")

        # Alternatively, you can use Client.from_url to connect to an already running server
        client = Client.from_url("http://localhost:3000")
        res = client.classify(np.array([[4.9, 3.0, 1.4, 0.2]]))
        print(f"Successfully received results, {res}")
