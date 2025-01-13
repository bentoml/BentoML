=======================
Mount ASGI applications
=======================

`ASGI (Asynchronous Server Gateway Interface <https://asgi.readthedocs.io/en/latest/>`_ is a spiritual successor to WSGI (Web Server Gateway Interface), designed to provide a standard interface between async-capable Python web servers, frameworks, and applications. ASGI supports asynchronous request handling, allowing multiple requests to be processed at the same time, making it suitable for real-time web applications, such as WebSockets, long polling, and more.

BentoML's server runs the Service API in an ASGI web serving layer, exposing REST endpoints for inference APIs (for example, ``POST /summarize``) and common infrastructure APIs (for example, ``GET /metrics``) for monitoring. This ASGI-native web serving layer allows for direct mounting of existing ASGI applications, enabling them to serve side-by-side with BentoML Services.

This document explains how to mount an ASGI application onto a BentoML Service.

Why should you mount an ASGI application
----------------------------------------

Mounting an ASGI application, such as one built with FastAPI, onto a BentoML Service can be advantageous for several reasons:

- **Extended functionality**: It enables you to extend your machine learning services with additional web functionalities, such as custom APIs for data processing, user management, or serving static files and web user interfaces, which are not directly related to model serving.
- **Custom authentication and authorization**: By integrating an ASGI application, you can implement advanced authentication and authorization mechanisms, tailoring security measures to the specific needs of your application.
- **API documentation**: With tools like FastAPI, you automatically get interactive API documentation, making it easier for end-users to understand and interact with the APIs.

Integrate BentoML with ASGI frameworks
--------------------------------------

BentoML offers seamless integration with different ASGI frameworks, allowing you to serve ML models alongside custom web application logic, such as asynchronous operations, real-time data processing, and complex web application functionalities.

When integrating the ASGI frameworks, you use ``bentoml.get_current_service()`` to retrieve the current BentoML Service instance. It is useful when you need to access the BentoML Service instance from within ASGI application routes or when injecting the Service instance into ASGI application routes using dependency injection patterns.

See the following examples for details.

FastAPI
^^^^^^^

`FastAPI <https://fastapi.tiangolo.com/>`_ is a web framework for building APIs that is built on top of ASGI, allowing it to handle asynchronous requests. To integrate a FastAPI application with a BentoML Service, you can define a FastAPI route either inside or outside the Service as below.

.. note::

   Make sure you have installed FastAPI by running ``pip install fastapi``. See `the FastAPI documentation <https://fastapi.tiangolo.com/tutorial/>`_ to learn more.

.. code-block:: python

    from fastapi import FastAPI, Depends
    import bentoml

    app = FastAPI()

    @bentoml.service
    @bentoml.asgi_app(app, path="/v1")
    class MyService:
        name = "MyService"

        @app.get('/hello')
        def hello(self):  # Inside service class, use `self` to access the service
            return f"Hello {self.name}"

    @app.get("/hello1")
    async def hello(service: MyService = Depends(bentoml.get_current_service)):
        # Outside service class, use `Depends` to get the service
        return f"Hello {service.name}"

Specifically, do the following to mount FastAPI:

1. Create a FastAPI application with ``FastAPI()``.
2. Use the ``@bentoml.asgi_app`` decorator to mount the FastAPI application to the BentoML Service, enabling them to be served together. Set the ``path`` parameter to :ref:`customize the prefix path <customize-prefix-path>`.
3. Define a FastAPI route inside or outside the Service class using ``@app.get("/<route-name>")``.

   - Inside the class: Use ``self`` to access the Service instance's attributes and methods.
   - Outside the class: Use FastAPI's dependency injection system (``Depends``) to inject the BentoML Service instance into the route function. In the code above, the ``hello1`` route uses ``Depends(bentoml.get_current_service)`` to inject the ``MyService`` instance, allowing the route to access the Service's attributes and methods.

4. Within the FastAPI route, add your desired implementation logic. This example returns a greeting message using the Service's name.

.. note::

    In addition to ``get``, you can use the other operations like ``post``, ``put``, and ``delete``. See `the FastAPI documentation <https://fastapi.tiangolo.com/tutorial/first-steps/>`_ to learn more.

.. dropdown:: Design choice: Inside vs. Outside

    Accessing the BentoML Service instance both inside and outside the Service class offers flexibility in how you structure and interact with your Service logic and dependencies. The differences in accessing the BentoML Service instance in these contexts primarily relate to scope and the intended use cases.

    Inside the Service class

    - Direct access: Within the class defining a BentoML Service, you have direct access to ``self``, which represents the instance of the Service. This allows you to directly access its attributes and methods without injecting any dependency. It's the most straightforward way to use the Service's functionality from within its own definition.
    - Contextual use: Accessing the Service instance inside the class is typical for defining the Service's internal logic, such as setting up endpoints, performing operations with the model, and handling requests directly related to the Service's primary functionality.

    Outside the Service class

    - Dependency injection: Accessing the BentoML Service instance outside the class typically requires dependency injection mechanisms, such as the ``Depends`` function in FastAPI. This approach is necessary when you want to use the Service instance in other parts of your project.
    - Modular and decoupled design: This approach allows different components of your BentoML project to interact with the Service without being tightly integrated into its class definition. For example, your ML logic can be encapsulated within the BentoML Service, while other aspects, such as custom authentication, supplementary data processing, or additional REST endpoints, can be managed externally yet still interact with the Service as needed.

The following is a more practical example of mounting FastAPI onto the Summarization Service in :doc:`/get-started/hello-world`. It defines two additional endpoints with FastAPI by accessing the Service from inside and outside the class respectively.

.. code-block:: python

    from __future__ import annotations
    import bentoml
    from transformers import pipeline
    from fastapi import FastAPI, Depends

    EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century."

    # Create a FastAPI app instance
    app = FastAPI()

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    @bentoml.asgi_app(app, path="/v1")
    class Summarization:
        def __init__(self) -> None:
            self.pipeline = pipeline('summarization')

        # Define a name attribute
        name = "MyService"

        # The original Service API endpoint for text summarization
        @bentoml.api
        def summarize(self, text: str = EXAMPLE_INPUT) -> str:
            result = self.pipeline(text)
            return result[0]['summary_text']

        # Access the Service instance inside the class
        @app.get("/hello-inside")
        def hello(self):
            # Add other logic here if needed
            return f"Hello {self.name}. You can access the Service instance inside the class."

    # Access the Service instance outside the class
    @app.get("/hello-outside")
    async def hello(service: MyService = Depends(bentoml.get_current_service)):
        # Add other logic here if needed
        return f"Hello {service.name}. You can access the Service instance outside the class."

After you start the BentoML Service, which is accessible at `http://localhost:3000 <http://localhost:3000/>`_, you can find two additional endpoints ``hello-inside`` and ``hello-outside`` exposed.

.. image:: ../../_static/img/build-with-bentoml/asgi/two-asgi-fastapi-routes.png
   :alt: Two API endpoints defined in BentoML

By sending a ``GET`` request, you can receive the corresponding output from both endpoints.

FastAPI route inside the Service class:

.. image:: ../../_static/img/build-with-bentoml/asgi/inside-the-class.png
   :alt: FastAPI route inside the BentoML Service class

FastAPI route outside the Service class:

.. image:: ../../_static/img/build-with-bentoml/asgi/outside-the-class.png
   :alt: FastAPI route outside the BentoML Service class

Quart
^^^^^

`Quart <https://quart.palletsprojects.com/en/latest/index.html>`_ is an asynchronous web framework for Python that enables you to use async/await features in your web applications to handle large volumes of concurrent connections.

The following is an example of integrating Quart with BentoML.

.. note::

    Make sure you have installed Quart by running ``pip install quart``. See `the Quart documentation <https://quart.palletsprojects.com/en/latest/tutorials/installation.html>`_ to learn more.

.. code-block:: python

    from quart import Quart

    app = Quart(__name__)

    @app.get("/hello")
    async def hello_world():
        service = bentoml.get_current_service()
        return f"Hello, {service.name}"

    @bentoml.service
    @bentoml.asgi_app(app, path="/v1")
    class MyService:
        name = "MyService"

Specifically, do the following to mount Quart:

1. Create a Quart application with ``Quart()``.
2. Use the ``@bentoml.asgi_app`` decorator to mount the Quart application to the BentoML Service, enabling them to be served together. Set the ``path`` parameter to :ref:`customize the prefix path <customize-prefix-path>`.
3. Define a Quart route outside the Service class using ``@app.get(/"<route-name>")``. Use ``bentoml.get_current_service()`` to inject the ``MyService`` instance, allowing the route to access the Service's attributes and methods.
4. Within the Quart route, add your desired implementation logic. This example returns a greeting message using the Service's name.

.. note::

    In addition to ``get``, you can use the other operations like ``post``, ``put``, and ``delete``. See `the Quart documentation <https://quart.palletsprojects.com/en/latest/tutorials/index.html>`_ to learn more.

The following is a more practical example of mounting Quart onto the Summarization Service in :doc:`/get-started/hello-world`. It defines an additional endpoint ``hello``.

.. code-block:: python

    from __future__ import annotations
    import bentoml
    from transformers import pipeline
    from quart import Quart

    EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century."

    # Create a Quart app instance
    app = Quart(__name__)

    @app.get("/hello")
    async def hello_world():
        service = bentoml.get_current_service()
        # Add other logic here if needed
        return f"Hello, {service.name}"

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    @bentoml.asgi_app(app, path="/v1")
    class Summarization:
        def __init__(self) -> None:
            self.pipeline = pipeline('summarization')

        # Define a name attribute
        name = "MyService"

        # The original Service API endpoint for text summarization
        @bentoml.api
        def summarize(self, text: str = EXAMPLE_INPUT) -> str:
            result = self.pipeline(text)
            return result[0]['summary_text']

After you start the BentoML Service, which is accessible at `http://localhost:3000 <http://localhost:3000/>`_, you can interact with the exposed endpoint ``hello``. For example:

.. code-block:: bash

    $ curl http://localhost:3000/v1/hello

    Hello, MyService

.. note::

    Unlike FastAPI, Quart does not natively support the OpenAPI specification, so the endpoint is not displayed on the Swagger UI. You can use other ways to communicate with it, such as ``curl``.

.. _customize-prefix-path:

Customize the prefix path
-------------------------

When mounting an ASGI tool onto a BentoML Service, it is possible to customize the route path by setting a prefix. This is useful for organizing your API endpoints and simplifying routing and namespace management.

To set a prefix path, simply set the ``path`` parameter in the decorator ``@bentoml.asgi_app``. Here is a FastAPI example:

.. code-block:: python

    from fastapi import FastAPI, Depends
    import bentoml

    app = FastAPI()

    @bentoml.service
    @bentoml.asgi_app(app, path="/fastapi") # Add the prefix here
    class MyService:
        name = "MyService"

        @app.get('/hello')  # This endpoint should be requested via "/fastapi/hello"
        def hello(self):
            return f"Hello {self.name}"

By specifying ``path="/fastapi"``, the entire FastAPI application is served under this prefix. This means all the routes defined within the FastAPI application will be accessible under ``/fastapi``. In this example, after you start this BentoML Service, you should interact with the ``/fastapi/hello`` endpoint.

Add custom ASGI middleware
--------------------------

``add_asgi_middleware`` is an API provided by BentoML to apply `custom ASGI middleware <https://asgi.readthedocs.io/en/latest/specs/main.html>`_. Middleware functions as a layer that processes requests and responses, allowing you to manipulate them or execute additional actions based on specific conditions. It is commonly used for implementing security measures and custom headers, managing CORS, compressing responses, and more.

Example usage:

.. code-block:: python

    from __future__ import annotations
    import bentoml
    from transformers import pipeline

    from starlette.middleware.trustedhost import TrustedHostMiddleware

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    class Summarization:
        def __init__(self) -> None:
            self.pipeline = pipeline('summarization')

        @bentoml.api
        def summarize(self, text: str) -> str:
            result = self.pipeline(text)
            return result[0]['summary_text']

    # Add TrustedHostMiddleware to ensure the Service only accepts requests from certain hosts
    Summarization.add_asgi_middleware(TrustedHostMiddleware, allowed_hosts=['example.com', '*.example.com'])

This example ensures that the ``Summarization`` Service only accepts requests from specified hosts and prevents host header attacks. You can then interact with the Service by manually specifying the ``Host`` header in requests:

.. code-block:: bash

    curl -H "Host: example.com" http://localhost:3000

.. note::

    Alternatively, you can edit your ``hosts`` file to map ``example.com`` to ``127.0.0.1`` (localhost) and then access ``http://example.com:3000/``.

While ``add_asgi_middleware`` is used to add middleware to the ASGI application that BentoML uses to serve the APIs, ``@bentoml.asgi_app`` is used to integrate the entire ASGI application into the BentoML Service. This is suitable for adding complete web applications like FastAPI or Quart applications that come with their routing logic, directly alongside your BentoML Service.

The middleware added via ``add_asgi_middleware`` applies to the entire ASGI application, including both the BentoML Service and any mounted ASGI applications. This ensures consistent processing of all requests across the application, whether they target BentoML Services or other components.
