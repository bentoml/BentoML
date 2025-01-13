===========================
Define a WebSocket endpoint
===========================

BentoML allows you to set up WebSocket endpoints for real-time communication within your services. WebSocket support is ideal for use cases like real-time messaging, streaming data, or voice AI applications.

Usage
-----

BentoML Services can handle WebSocket connections through `FastAPI's WebSocket support <https://fastapi.tiangolo.com/advanced/websockets/>`_. To set up a WebSocket server:

1. Create a FastAPI application.
2. :doc:`Mount it to your BentoML Service </build-with-bentoml/asgi>`.

Once the BentoML server starts, the WebSocket server is also initialized and ready to accept connections.

Here is a simple example:

.. code-block:: python
    :caption: `service.py`

    import bentoml
    from fastapi import FastAPI, WebSocket

    # Initialize FastAPI app
    app = FastAPI()

    # Define BentoML Service and mount the app
    @bentoml.service(
        traffic={"timeout": 30}
    )
    @bentoml.asgi_app(app, path="/chat")
    class WebSocketService:
        def __init__(self):
            # Initialize your resources here (e.g., models, configurations)
            print("Service initialized")

        @app.websocket("/ws")
        async def websocket_endpoint(self, websocket: WebSocket):
            await websocket.accept()
            # Define your custom logic here
            print("WebSocket connection accepted")
            try:
                while True:
                    data = await websocket.receive_text()
                    print(f"Received: {data}")
                    await websocket.send_text(f"Echo: {data}")
            except Exception as e:
                print(f"Connection closed: {e}")

Send requests
-------------

The :doc:`BentoML Python client </build-with-bentoml/clients>` does not yet support WebSocket endpoints. You'll need to implement your own client. Here's an example using the ``websockets`` library to call the endpoint defined above:

.. code-block:: python

    import asyncio
    import websockets

    async def test_websocket():
        # /chat comes from the asgi_app path, /ws from the endpoint
        # Adjust URL as needed
        uri = "ws://localhost:3000/chat/ws"
        async with websockets.connect(uri) as websocket:
            # Send a test message
            await websocket.send("Hello BentoML")
            response = await websocket.recv()
            print(f"Response: {response}")

    # Run the test
    asyncio.run(test_websocket())

Learn more
----------

For a more practical example, see `how to build a voice AI application with an open-source model using Twilio ConversationRelay and BentoML <https://github.com/bentoml/BentoTwilioConversationRelay>`_.
