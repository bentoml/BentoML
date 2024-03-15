===============
Lifecycle hooks
===============

Lifecycle hooks in BentoML offers mechanism to run custom logic at various stages of a Service's lifecycle. By leveraging these hooks, you can perform setup actions at startup, clean up resources before shutdown, and more. 

This document provides an overview of lifecycle hooks and how to use them in BentoML :doc:`/guides/services`.

Understand server lifecycle
---------------------------

BentoML's server lifecycle consists of several stages, each providing a unique opportunity to perform specific tasks:

1. **Deployment hooks**. These hooks run before any Service workers are spawned, making them suitable for one-time global setup tasks. They're crucial for operations that should occur once, regardless of the number of workers.
2. **Spawn workers**. BentoML then spawns worker processes according to the ``workers`` configuration specified in the ``@bentoml.service`` decorator.
3. **Service initialization and ASGI application startup**. During the startup of each worker, any :doc:`integrated ASGI application </guides/asgi>` begins its lifecycle. This is when the ``__init__`` method of your Service class is executed, allowing for instance-specific initialization.
4. **ASGI application teardown**. Finally, as the server shuts down, including the ASGI application, shutdown hooks are executed. This stage is ideal for performing cleanup tasks, ensuring a graceful shutdown.

Configure hooks in a BentoML Service
------------------------------------

This section provides code examples for configuring different BentoML hooks.

Deployment hooks
^^^^^^^^^^^^^^^^

Deployment hooks are similar to static methods as they do not receive the ``self`` argument. You can define multiple deployment hooks in a Service. Use the ``@bentoml.on_deployment`` decorator to specify a method as a deployment hook. For example:

.. code-block:: python

    import bentoml

    @bentoml.service(workers=4)
    class HookService:
        # Deployment hook does not receive `self` argument. It acts similarly to a static method.
        @bentoml.on_deployment
        def prepare():
            print("Do some preparation work, running only once.")

        # Multiple deployment hooks can be defined
        @bentoml.on_deployment
        def additional_setup():
            print("Do more preparation work if needed, also running only once.")

        def __init__(self) -> None:
            # Startup logic and initialization code 
            print("This runs on Service startup, once for each worker, so it runs 4 times.")

        @bentoml.api
        def predict(self, text) -> str:
            # Endpoint implementation logic

After the Service starts, you can see the following output on the server side in order:

.. code-block:: bash

    $ bentoml serve service:HookService

    Do some preparation work, running only once. # First on_deployment hook
    Do more preparation work if needed, also running only once. # Second on_deployment hook
    2024-03-13T03:12:33+0000 [INFO] [cli] Starting production HTTP BentoServer from "service:HookService" listening on http://localhost:3000 (Press CTRL+C to quit)
    This runs on Service startup, once for each worker, so it runs 4 times.
    This runs on Service startup, once for each worker, so it runs 4 times.
    This runs on Service startup, once for each worker, so it runs 4 times.
    This runs on Service startup, once for each worker, so it runs 4 times.

Shutdown hooks
^^^^^^^^^^^^^^

Shutdown hooks are executed as a BentoML Service is in the process of shutting down. It allows for the execution of cleanup logic such as closing connections, releasing resources, or any other necessary teardown tasks. You can define multiple shutdown hooks in a Service.

Use the ``@bentoml.on_shutdown`` decorator to specify a method as a shutdown hook. For example:

.. code-block:: python

    import bentoml

    @bentoml.service(workers=4)
    class HookService:
        @bentoml.on_deployment
        def prepare():
            print("Do some preparation work, running only once.")

        def __init__(self) -> None:
            # Startup logic and initialization code 
            print("This runs on Service startup, once for each worker, so it runs 4 times.")

        @bentoml.api
        def predict(self, text) -> str:
            # Endpoint implementation logic
            
        @bentoml.on_shutdown
        def shutdown(self):
            # Logic on shutdown
            print("Cleanup actions on Service shutdown.")
            
        @bentoml.on_shutdown
        async def async_shutdown(self):
            print("Async cleanup actions on Service shutdown.")