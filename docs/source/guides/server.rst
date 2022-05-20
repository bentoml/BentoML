=====================
Customize BentoServer
=====================

TODO

 middleware, custom endpoint


.. _custom-endpoints-page:

Custom HTTP endpoints
=====================

BentoML supports mounting a variety of different types of endpoints alongside it’s standard prediction endpoints. Both WSGI and ASGI python web applications are supported. So whether you already have code written in these frameworks or if it’s just a framework that you know, we support the additions.

Mounting WSGI based web frameworks
----------------------------------

Here’s an example of mounting Flask endpoints alongside BentoML

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

    app = svc.asgi_app


As you can see, you can use flask annotations just as you would if you were building a standalone flask app. In order to ensure the correct coupling, the ``svc.mount_wsgi_app(flask_app)`` must be invoked.

Mounting ASGI based web frameworks
----------------------------------

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

    app = svc.asgi_app


The primary method to invoke is ``svc.mount_asgi_app(fastapi_app)`` in order to ensure that the fastapi endpoints are initialized