====================
Call an API endpoint
====================

BentoML provides a client implementation that allows you to make synchronous and asynchronous requests to BentoML :doc:`Services </build-with-bentoml/services>`.

This document explains how to use :doc:`BentoML clients </reference/bentoml/client>`.

Client types
------------

Depending on your requirements, you can create a BentoML client object using the following classes.

- ``bentoml.SyncHTTPClient``: Defines a synchronous client, suitable for straightforward, blocking operations where your application waits for the response before proceeding.
- ``bentoml.AsyncHTTPClient``: Defines an asynchronous client, suitable for non-blocking operations, allowing your application to handle other tasks while waiting for responses.

Create a client
---------------

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

After you start the ``Summarization`` Service, you can create the following clients by specifying the server address.

.. tab-set::

    .. tab-item:: Synchronous

        .. code-block:: python

            import bentoml

            client = bentoml.SyncHTTPClient('http://localhost:3000')
            summarized_text: str = client.summarize(text="Your long text to summarize")
            print(summarized_text)

            # Close the client to release resources
            client.close()

    .. tab-item:: Asynchronous

        .. code-block:: python

            import asyncio
            import bentoml

            async def async_client_operation():
                client = bentoml.AsyncHTTPClient('http://localhost:3000')
                summarized_text: str = await client.summarize(text="Your long text to summarize")
                print(summarized_text)

                # Close the client to release resources
                await client.close()

            asyncio.run(async_client_operation())

In the above synchronous and asynchronous clients, requests are sent to the ``summarize`` endpoint of the Service hosted at ``http://localhost:3000``. The BentoML client implementation supports methods corresponding to the Service APIs and they should be called with the same arguments (``text`` in this example) as defined in the Service. These methods are dynamically created based on the Service's endpoints, providing a direct mapping to the Service’s functionality.

In this example, the ``summarize`` method on the client is directly mapped to the ``summarize`` method in the ``Summarization`` Service. The data passed to the ``summarize`` method (``text="Your long text to summarize"``) conforms to the expected input of the Service.

.. note::

    If you deploy your Service to BentoCloud, you can get the client of a Deployment by using ``get_client()`` or ``get_async_client()``. For more information, see :ref:`scale-with-bentocloud/deployment/call-deployment-endpoints:interact with the Deployment`.

Use a context manager
^^^^^^^^^^^^^^^^^^^^^

To enhance resource management and reduce the risk of connection leaks, we recommend you create a client within a context manager as below.

.. tab-set::

    .. tab-item:: Synchronous

        .. code-block:: python

            import bentoml

            with bentoml.SyncHTTPClient('http://localhost:3000') as client:
                summarized_text: str = client.summarize(text="Your long text to summarize")
                print(summarized_text)

    .. tab-item:: Asynchronous

        .. code-block:: python

            import bentoml

            async with bentoml.AsyncHTTPClient('http://localhost:3000') as client:
                summarized_text: str = await client.summarize(text="Your long text to summarize")
                print(summarized_text)

Check Service readiness
-----------------------

Before making calls to specific Service methods, you can use the ``is_ready`` method of the client to check if the Service is ready to handle requests. This ensures that your API calls are made only when the Service is up and running.

.. code-block:: python

    import bentoml

    client = bentoml.SyncHTTPClient('http://localhost:3000')
    if client.is_ready():
        summarized_text: str = client.summarize(text="Your long text to summarize.")
        print("Summarized text:", summarized_text)
    else:
        print("Service is not ready")

    client.close()

Alternatively, use the ``server_ready_timeout`` parameter to specify the maximum duration in seconds the client will wait for the BentoML Service to become ready before timing out. This is useful during the initial connection to a Service that might be starting up. If the Service does not become ready within the specified timeout, the client will raise a timeout exception.

.. code-block:: python

    import bentoml

    client = bentoml.SyncHTTPClient(
      'http://localhost:3000',
      server_ready_timeout=60  # Wait up to 60 seconds for the Service to be ready
    )
    summarized_text: str = client.summarize(text="Your long text to summarize")
    print(summarized_text)

    client.close()

.. _call-a-task-endpoint:

Call a task endpoint
--------------------

You can create clients to interact with Services defined with :ref:`task <bentoml-tasks>` endpoints by submitting inputs and then asynchronously checking for results at a later time. This is particularly useful for scenarios where the client does not need to actively wait for the task to complete. For more information, see :doc:`/get-started/async-task-queues`.

Input and output
----------------

BentoML clients support handling different input and output types.

JSON
^^^^

You can easily handle JSONable data input and JSON output with BentoML's HTTP clients, which are designed to seamlessly serialize and deserialize JSON data.

For input, when you send data that can be serialized to JSON (for example, dictionaries, lists, strings, and numbers), you simply pass it as arguments to the client method corresponding to your Service API.

The following code comes from the Service ``SentenceEmbedding`` of `this example project <https://github.com/bentoml/BentoSentenceTransformers>`_, which accepts JSONable input (lists in this case).

.. code-block:: python

    import typing as t

    @bentoml.service
    class SentenceEmbedding:
        ...

        @bentoml.api
        def encode(self, sentences: t.List[str] = SAMPLE_SENTENCES) -> np.ndarray:
        ...

To create a client to handle JSONable input for Services like ``SentenceEmbedding``:

.. code-block:: python

    import bentoml
    import typing as t

    client = bentoml.SyncHTTPClient("http://localhost:3000")

    # Specify the sentences for the request
    sentences_list: t.List[str] = [
        "The sun dips below the horizon, painting the sky orange.",
        "A gentle breeze whispers through the autumn leaves.",
        "The moon casts a silver glow on the tranquil lake.",
        # Add more if necessary
    ]

    # Make the request using the Service endpoint
    result = client.encode(sentences=sentences_list)

    # Print the result
    print(f"Encoded sentences result: {result}")

    client.close()

For output, when a BentoML Service returns JSON data, the client automatically deserializes this JSON into a Python data structure (like a dictionary or a list, depending on the JSON structure).

The following code comes from the Service ``WhisperX`` of this `example project <https://github.com/bentoml/BentoWhisperX>`_, which returns JSONable output (dictionaries in this case).

.. code-block:: python

    import typing as t
    from pathlib import Path

    @bentoml.service
    class WhisperX:
        ...

        @bentoml.api
        def transcribe(self, audio_file: Path) -> t.Dict:
        ...

To create a client to handle JSONable output for Services like ``WhisperX``:

.. code-block:: python

    import bentoml
    import typing as t

    client = bentoml.SyncHTTPClient('http://localhost:3000')

    # Set the audio URL
    audio_url = 'https://example.org/female.wav'

    # The response is expected to be a dictionary
    response: t.Dict = client.transcribe(audio_file=audio_url)

    print(response)

.. tip::

    You can print specific values of keys from the JSON response. For example, the Service ``WhisperX`` returns the following and you can output the text of the first segment:

    .. code-block:: python

        response = {
            "segments": [
                {
                    "start": 0.009,
                    "end": 2.813,
                    "text": " The Hispaniola was rolling scuppers under in the ocean swell.",
                    "words": [
                        {"word": "The", "start": 0.009, "end": 0.069, "score": 0.0},
                        {"word": "Hispaniola", "start": 0.109, "end": 0.81, "score": 0.917},
                        # Other words omitted...
                    ],
                },
                # Other segments omitted...
            ],
            "word_segments": [
                {"word": "The", "start": 0.009, "end": 0.069, "score": 0.0},
                {"word": "Hispaniola", "start": 0.109, "end": 0.81, "score": 0.917},
                # Other words omitted...
            ],
        }

        # Print the text of the first segment
        # Add the following line to your client code
        print("Segment text:", response["segments"][0]["text"])

Files
^^^^^

BentoML clients support a variety of file types, such as images and generic binary files.

For file input, you pass a ``Path`` object pointing to the file. The client handles the file reading and sends it as part of the request. For file output, the client provides the output as a ``Path`` object. You can use this ``Path`` object to access, read, or process the file.

The following code snippet comes from the :doc:`/examples/controlnet` example, which accepts and returns an image file.

.. code-block:: python

    import PIL
    from PIL.Image import Image as PIL_Image

    @bentoml.service
    class ControlNet:
        ...

        @bentoml.api
        async def generate(self, image: PIL_Image, params: Params) -> PIL_Image:
        ...

To create a client to handle file input and output for Services like ``ControlNet``:

.. code-block:: python

    import bentoml
    from pathlib import Path

    client = bentoml.SyncHTTPClient("http://localhost:3000")

    # Specify the image path and other parameters for the request
    image_path: Path = Path("/path/to/example-image.png")
    params = {
        "prompt": "A young man walking in a park, wearing jeans.",
        "negative_prompt": "ugly, disfigured, ill-structure, low resolution",
        "controlnet_conditioning_scale": 0.5,
        "num_inference_steps": 25
    }

    # Make the request using the Service endpoint
    result_path: Path = client.generate(
        image=image_path,
        params=params,
    )

    print(f"Generated file saved at: {result_path}")

    client.close()

You can also use URLs as the input as below:

.. code-block:: python

    import bentoml
    from pathlib import Path

    client = bentoml.SyncHTTPClient("http://localhost:3000")

    # Specify the image URL and other parameters for the request
    image_url = 'https://example.org/1.png'
    # The remaining code is the same
    ...

Streaming
^^^^^^^^^

You can add streaming logic to a BentoML client, which is especially useful when dealing with large amounts of data or real-time data feeds. Streamed output is returned a generator or async generator, depending on the client type.

.. tab-set::

    .. tab-item:: Synchronous

        For synchronous streaming, ``SyncHTTPClient`` uses a Python generator to output data as it is received from the stream.

        .. code-block:: python

            import bentoml

            client = bentoml.SyncHTTPClient("http://localhost:3000")
            for data_chunk in client.stream_data():
                # Process each chunk of data as it arrives
                process_data(data_chunk)

            client.close()

            def process_data(data_chunk):
                # Add processing logic
                print("Processing data chunk:", data_chunk)
                # Add more logic here to handle the data chunk

    .. tab-item:: Asynchronous

        For asynchronous streaming, ``AsyncHTTPClient`` uses an async generator. This allows for asynchronous iteration over the streaming data.

        .. code-block:: python

            import bentoml

            client = bentoml.AsyncHTTPClient("http://localhost:3000")
            async for data_chunk in client.stream_data():
                # Process each chunk of data as it arrives
                await process_data_async(data_chunk)

            await client.close()

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

    import bentoml

    client = bentoml.SyncHTTPClient('http://localhost:3000', token='your_token_here')
    summarized_text: str = client.summarize(text="Your long text to summarize")
    print(summarized_text)

    client.close()

Error handling
--------------

Handling errors, checking for error code and messages, and implementing retries are important for reliable client-server communication. Here are some strategies and examples on error handling and retries.

Basics
^^^^^^

When interacting with a BentoML Service, errors like network issues, Service downtime, or invalid input, may occur. Proper error handling allows your client to respond gracefully to these issues.

You can use ``try`` and ``except`` blocks to catch exceptions that may occur during the request:

.. code-block:: python

    import bentoml
    from bentoml.exceptions import BentoMLException

    client = bentoml.SyncHTTPClient('http://localhost:3000')

    try:
        summarized_text: str = client.summarize(text="Your long text to summarize.")
        print(summarized_text)
    except BentoMLException as e:
        print(f"An error occurred: {e}")
    finally:
        client.close()

When catching exceptions, it's useful to examine specific error code or messages to determine the cause of the failure. This can guide the retry logic or inform you about the issue more precisely.

Implement retry logic
^^^^^^^^^^^^^^^^^^^^^

Retrying failed requests can help overcome temporary issues like network disruptions or service unavailability. When implementing retries, consider exponential backoff to avoid overwhelming the server or the network.

Here's a simple example of implementing retries with exponential backoff.

.. code-block:: python

    import time
    from bentoml.exceptions import BentoMLException
    import bentoml

    def retry_request(client, max_retries=3, backoff_factor=2):
        for attempt in range(max_retries):
            try:
                summarized_text: str = client.summarize(text="Your long text to summarize.")
                return summarized_text
            except BentoMLException as e:
                print(f"Attempt {attempt+1}: An error occurred: {e}")
                time.sleep(backoff_factor ** attempt)
        print("Max retries reached. Giving up.")

    client = bentoml.SyncHTTPClient('http://localhost:3000')

    try:
        response = retry_request(client)
        if response:
            print(response)
    finally:
        client.close()
