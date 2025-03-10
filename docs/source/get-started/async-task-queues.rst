=================
Async task queues
=================

Async tasks enable you to handle certain inference tasks in a fire-and-forget style. Instead of waiting for a response, you can submit a request that runs in the background and immediately receive a unique identifier. This identifier can then be used at any point to check the status of the task or retrieve the results once the task is complete. This allows for more efficient use of resources and improved responsiveness in scenarios where immediate results are not required.

Async tasks are ideal for:

- Batch processing: Running inference on large volumes of data
- Asynchronous generation: Generating text, images, or other media that may take a long time to complete
- Time insensitive tasks: Tasks that do not need to be completed immediately that can be executed with a lower priority

.. image:: ../../_static/img/get-started/tasks/async_tasks.png
    :width: 800px
    :align: center
    :alt: BentoML async task architecture


Task results are stored for a configurable amount of time, default 24 hours, and are deleted after that period.

.. note::

    The diagram above illustrates the architecture of async tasks running on BentoCloud. When a BentoML service is deployed locally, the request queue and ephemeral storage are created in non-persistent memory, intended solely for development purposes.

Define a task endpoint
----------------------

You can define a task endpoint using the ``@bentoml.task`` decorator in the Service constructor. If you already have a ``@bentoml.api`` and would like to convert it to an async task, you can simply change the decorator only without modifying the function implementation. Here's an example:

.. code-block:: python

    import bentoml
    from PIL.Image import Image

    @bentoml.service
    class ImageGenerationService:

        @bentoml.task
        def long_running_image_generation(self, prompt: str) -> Image:
            # Process the prompt in a long-running process
            return image

Under the hood, BentoML automatically generates several endpoints for creating the task, getting the task status, and retrieving the task results.

- ``POST /submit``: Submit a task to the queue. A unique task identifier is returned immediately.
- ``GET /status``: Get the status of a task given the task identifier.
- ``GET /get``: Get the result of a task given the task identifier.
- ``POST /cancel``: Attempt to cancel a task given the task identifier, if the task hasn't started execution.
- ``PUT /retry``: Retry a task given the task identifier.

Call a task endpoint
--------------------

Async tasks can be submitted through the ``SyncHTTPClient`` or ``AsyncHTTPClient`` clients, using the ``submit()`` function on the endpoint name.

.. code-block:: python

    import bentoml

    prompt = "a scenic mountain view that ..."
    client = bentoml.SyncHTTPClient('http://localhost:3000')
    # The arguments are the same as the Service method, just call with `.submit()`
    task = client.long_running_image_generation.submit(prompt=prompt)
    print("Task submitted, ID:", task.id)

.. note::

    You may also use an HTTP client from any language and invoke the endpoints listed above directly.


Once a task is submitted, the request is enqueued in the request queue and a unique task identifier is returned immediately, which can be used to get the status and retrieve the result.

.. code-block:: python

    # Use the following code at a later time
    status = task.get_status()
    if status.value == 'success':
        print("The task runs successfully. The result is", task.get())
    elif status.value == 'failure':
        print("The task run failed.")
    else:
        print("The task is still running.")

Use ``retry()`` if a task fails or you need to rerun the task with the same parameters:

.. code-block:: python

    status = task.get_status()
    if status.value == 'failure':
        print("Task failed, retrying...")
        new_task = task.retry()
        new_status = new_task.get_status()
        print("New task status:", new_status.value)

For more information, see :doc:`/build-with-bentoml/clients`.
