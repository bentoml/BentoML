=================
Async task queues
=================

Many model inferences are best handled as long-running operations. Tasks in BentoML allow you to execute these long-running workloads in the background and retrieve the results at a later time.

This document explains how to define and call a task endpoint.

Overview
--------

Tasks are ideal for scenarios where you don't need the inference results immediately, such as:

- **Batch processing**: Handling large volumes of data or computations in a single batch.
- **Video or image generation**: Creating or manipulating media files which may take considerable time.

Waiting synchronously for such tasks could lead to inefficiencies, with the caller remaining idle for the majority of the time. With BentoML tasks, you can send prompts first and then asynchronously get the results.

Here is the general workflow of using BentoML tasks:

.. image:: ../../_static/img/get-started/tasks/task-workflow.png
    :width: 400px
    :align: center
    :alt: BentoML task workflow

Define a task endpoint
----------------------

You define a task endpoint using the ``@bentoml.task`` decorator in the Service constructor. Here's an example:

.. code-block:: python

    import bentoml
    from PIL.Image import Image

    @bentoml.service
    class ImageGenerationService:

        @bentoml.task
        def long_running_image_generation(self, prompt: str) -> Image:
            # Process the prompt in a long-running process
            return image

BentoML automatically exposes several endpoints for clients to manage the task, such as task submission and status retrieval.

Call a task endpoint
--------------------

BentoML tasks are managed via a task queue style API endpoint. You can create clients to interact with the endpoint by submitting requests and dedicated worker processes will monitor the queues for new tasks. Both ``SyncHTTPClient`` and ``AsyncHTTPClient`` clients can be used to call a task endpoint.

Here's how you can submit a task using a synchronous client:

.. code-block:: python

    import bentoml

    prompt = "a scenic mountain view that ..."
    client = bentoml.SyncHTTPClient('http://localhost:3000')
    # The arguments are the same as the Service method, just call with `.submit()`
    task = client.long_running_image_generation.submit(prompt=prompt)
    print("Task submitted, ID:", task.id)

Once a task is submitted, the client receives a task ID, which can be used to track the task status and retrieve results at a later time. Here is an example:

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
