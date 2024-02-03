=======
Clients
=======

BentoML provides a client implementation that allows you to make synchronous and asynchronous requests to BentoML :doc:`/guides/services`.

This document explains how to use :doc:`BentoML clients </reference/client>`.

Client types
------------

Depending on your requirements, you can create a BentoML client object using the following classes.

- ``bentoml.SyncHTTPClient``: Defines a synchronous client, suitable for straightforward, blocking operations where your application waits for the response before proceeding.
- ``bentoml.AsyncHTTPClient``: Defines an asynchronous client, suitable for non-blocking operations, allowing your application to handle other tasks while waiting for responses.

Create a client
---------------

When creating a client, you need to specify the server address. In addition, to enhance resource management and reduces the risk of connection leaks, we recommend you create the client within a context manager.

Suppose your BentoML Service has an endpoint named ``summarize`` that takes a string ``text`` as input and returns a summarized version of the text as below.

.. code-block:: python

    class Summarization:
        def __init__(self) -> None:
            # Load model into pipeline
            self.pipeline = pipeline('summarization')

        @bentoml.api
        def summarize(self, text: str) -> str:
            result = self.pipeline(text)
            return result[0]['summary_text']

After you start the ``Summarization`` Service, you can create the following clients as needed to interact with it.

.. tab-set::

    .. tab-item:: Synchronous

        .. code-block:: python

            with bentoml.SyncHTTPClient('http://localhost:3000') as client:
                response = client.summarize(text="Your long text to summarize")
                print(response)

    .. tab-item:: Asynchronous

        .. code-block:: python

            async with bentoml.AsyncHTTPClient('http://localhost:3000') as client:
                response = await client.summarize(text="Your long text to summarize")
                print(response)

In the above synchronous and asynchronous clients, requests are sent to the ``summarize`` endpoint of the Service hosted at ``http://localhost:3000``. The BentoML client implementation supports methods corresponding to the Service APIs and they should be called with the same arguments (``text`` in this example) as defined in the Service. These methods are dynamically created based on the Service's endpoints, providing a direct mapping to the Service’s functionality.

In this example, the ``summarize`` method on the client is directly mapped to the ``summarize`` method in the ``Summarization`` Service. The data passed to the ``summarize`` method (``text="Your long text to summarize"``) conforms to the expected input of the Service.

Check Service readiness
-----------------------

Before making calls to specific Service methods, you can use the ``is_ready`` method of the client to check if the Service is ready to handle requests. This ensures that your API calls are made only when the Service is up and running. For example:

.. code-block:: python

    with bentoml.SyncHTTPClient('http://localhost:3000') as client:
        if client.is_ready():
            response = client.summarize(text="Your long text to summarize.")
            print(response)
        else:
            print("Service is not ready")

Input and output
----------------

BentoML clients support handling different input and output types.

JSON
^^^^

You can easily handle JSONable data input and JSON output with BentoML's HTTP clients, which are designed to seamlessly serialize and deserialize JSON data.

When you send data that can be serialized to JSON (for example, dictionaries, lists, strings, and numbers), you simply pass it as arguments to the client method corresponding to your Service API.

.. code-block:: python

    with bentoml.SyncHTTPClient('http://localhost:3000') as client:
        data_to_send = {'name': 'Alice', 'age': 30}
        response = client.predict(data=data_to_send)
        print(response)

When the BentoML Service returns JSON data, the client automatically deserializes this JSON into a Python data structure (like a dictionary or a list, depending on the JSON structure).

Files
^^^^^

BentoML clients support a variety of file types, such as images and generic binary files.

For file inputs, you pass a ``Path`` object pointing to the file. The client handles the file reading and sends it as part of the request.

.. code-block:: python

    from pathlib import Path

    with bentoml.SyncHTTPClient('http://localhost:3000') as client:
        file_path = Path('/path/to/your/file')
        response = client.generate(img=file_path)
        print(response)

You can also use URLs as the input as below:

.. code-block:: python

    with bentoml.SyncHTTPClient('http://localhost:3000') as client:
        image_url = 'https://example.org/1.png'
        response = client.generate(img=image_url)
        print(response)

If the endpoint returns a file, the client provides the output as a ``Path`` object. You can use this ``Path`` object to access, read, or process the file. For example, if the file is an image, you can save it to a path; if it's a CSV, you can read its contents.

Streaming
^^^^^^^^^

You can add streaming logic to a BentoML client, which is especially useful when dealing with large amounts of data or real-time data feeds. Streamed output is returned a generator or async generator, depending on the client type.

.. tab-set::

    .. tab-item:: Synchronous

        For synchronous streaming, ``SyncHTTPClient`` uses a Python generator to output data as it is received from the stream.

        .. code-block:: python

            with bentoml.SyncHTTPClient("http://localhost:3000") as client:
                for data_chunk in client.stream_data():
                    # Process each chunk of data as it arrives
                    process_data(data_chunk)

            def process_data(data_chunk):
                # Add processing logic
                print("Processing data chunk:", data_chunk)
                # Add more logic here to handle the data chunk

    .. tab-item:: Asynchronous

        For asynchronous streaming, ``AsyncHTTPClient`` uses an async generator. This allows for asynchronous iteration over the streaming data.

        .. code-block:: python

            async with bentoml.AsyncHTTPClient("http://localhost:3000") as client:
                async for data_chunk in client.stream_data():
                    # Process each chunk of data as it arrives
                    await process_data_async(data_chunk)

            async def process_data_async(data_chunk):
                # Add processing logic
                print("Processing data chunk asynchronously:", data_chunk)
                # Add more complex asynchronous processing here
                await some_async_operation(data_chunk)

Authorization
-------------

When working with BentoML Services that require authentication, you can authorize clients (``SyncHTTPClient`` and ``AsyncHTTPClient``) using a token. This token, typically a JWT (JSON Web Token) or some other form of API key, is used to ensure that the client is allowed to access the specified BentoML Service. The token is included in the HTTP headers of each request made by the client, allowing the server to validate the client's credentials.

To authorize a client, you pass the token as an argument during initialization.

.. code-block:: python

    with bentoml.SyncHTTPClient('http://localhost:3000', token='your_token_here') as client:
        response = client.summarize(text="Your long text to summarize.")
        print(response)
