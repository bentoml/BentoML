=====================
Customize BentoServer
=====================


.. TODO::
    Link to basic server configs.

BentoML supports customizing the API service layer by mounting any WSGI or ASGI Python Web applications to a ``bentoml.Service``.

This means any code you have written in a Python web framework can be deployed together with your BentoML service and have access to Runners 😊.


ASGI Middleware
---------------

The :code:`bentoml.Service`'s :code:`add_asgi_middleware` API supports mounting any
`ASGI middleware <https://asgi.readthedocs.io/en/latest/specs/main.html>`_ to the
BentoServer endpoints.

Users can implement their own ASGI middleware class, but before you do so, make sure
to checkout the existing middleware built by the Python community, such as:

- FastAPI middlewares: https://fastapi.tiangolo.com/advanced/middleware/
- Starlette middlewares: https://www.starlette.io/middleware/

For example, you can add do:

.. code::

    from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    svc = bentoml.Service('my_service', runners=[...])

    svc.add_asgi_middleware(TrustedHostMiddleware, allowed_hosts=['example.com', '*.example.com'])
    svc.add_asgi_middleware(HTTPSRedirectMiddleware)


Customize API Server
--------------------

BentoML provides first-class support for bundling existing WSGI or ASGI app into a BentoServer.

Bundle WSGI app (e.g. Flask)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here’s an example of mounting Flask endpoints alongside BentoML:

.. code-block:: python

    import bentoml
    from bentoml.io import JSON
    from flask import Flask

    flask_app = Flask("sample_wsgi_app")

    @flask_app.route("/hello")
    def hello_world():
        return "<p>Hello, World!</p>"

    svc = bentoml.Service("bento-server-test")
    svc.mount_wsgi_app(flask_app)

    @svc.api(input=JSON(), output=JSON())
    def predict(input_json):
        return {'input_received': input_json, 'foo': 'bar'}


As you can see, you can use flask annotations as if you were building a standalone flask app. To ensure correct coupling, ``svc.mount_wsgi_app(flask_app)`` must be invoked.

Bundle ASGI app (e.g. FastAPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here’s an example of mounting a FastAPI app alongside BentoML

.. code-block:: python

    import bentoml
    from bentoml.io import JSON

    from fastapi import FastAPI

    fastapi_app = FastAPI()

    @fastapi_app.get("/hello")
    def hello():
        return {"Hello": "World"}

    svc = bentoml.Service("hello")
    svc.mount_asgi_app(fastapi_app)

    @svc.api(input=JSON(), output=JSON())
    def predict(input_json):
        return {'input_received': input_json, 'foo': 'bar'}


Make sure to invoke ``svc.mount_asgi_app(fastapi_app)`` so that the FastAPI endpoints are initialized correctly.

