============
Bento Server
============

BentoML Server runs the Service API in an `ASGI <https://asgi.readthedocs.io/en/latest/>`_
web serving layer and puts Runners in a separate worker process pool managed by BentoML. The ASGI web
serving layer will expose REST endpoints for inference APIs, such as ``POST /predict`` and common
infrastructure APIs, such as ``GET /metrics`` for monitoring.

BentoML offers a number of ways to customize the behaviors of the web serving layer to meet the needs of the consumers.


Custom Endpoint URL
-------------------

By default, the inference APIs are generated from the ``@api`` defined within a
``bentoml.Service``. The URL route for the inference API is determined by the function
name. Take the sample service from our tutorial for example, the function name ``classify``
will be used as the REST API URL ``/classify``:

.. code-block:: python

    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_arr):
        ...

However, user can customize this URL endpoint via the ``route`` option in the
``bentoml.Service#api`` decorator. For example, the following code will assign the
endpoint with URL ``/v1/models/iris_classifier/predict``, regardless of the API function name:


.. code-block:: python

    import numpy as np
    import bentoml
    from bentoml.io import NumpyNdarray

    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    @svc.api(
        input=NumpyNdarray(),
        output=NumpyNdarray(),
        route="v1/models/iris_classifier/predict"
    )
    def any_func_name(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result



ASGI Middleware
---------------

Since the web serving layer is built with the Python ASGI protocol, users can use the
:code:`bentoml.Service#add_asgi_middleware` API to mount arbitrary
`ASGI middleware <https://asgi.readthedocs.io/en/latest/specs/main.html>`_ to change
anything they may need to customize in the HTTP request to response lifecycle, such as
manipulating the request headers, modifying the response status code, or authorizing access to an endpoint.

Users can not only implement their own ASGI middleware class,
but also use existing middleware built by the Python web development community, such as:

- FastAPI middlewares: https://fastapi.tiangolo.com/advanced/middleware/
- Starlette middlewares: https://www.starlette.io/middleware/

For example, you can add do:

.. code::

    from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    svc = bentoml.Service('my_service', runners=[...])

    svc.add_asgi_middleware(TrustedHostMiddleware, allowed_hosts=['example.com', '*.example.com'])
    svc.add_asgi_middleware(HTTPSRedirectMiddleware)


Fully Customized Endpoints
--------------------------

BentoML provides first-class support for mounting existing WSGI or ASGI apps onto the
web serving layer, to enable common use cases such as serving existing Python web applications alongside
the models, performing custom authentication and authorization, handling GET requests and web UIs, or
providing streaming capabilities.



Bundle ASGI app (e.g. FastAPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BentoML's web serving layer is ASGI native, existing ASGI apps can be mounted directly
to and serving side-by-side with your BentoML Service.

Here’s an example (excerpted from :examples:`our example project <custom_web_serving/flask_example>`)
of mounting BentoML Service with an ASGI app built with FastAPI:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import bentoml
    from bentoml.io import NumpyNdarray, JSON
    from pydantic import BaseModel
    from fastapi import FastAPI

    class IrisFeatures(BaseModel):
        sepal_len: float
        sepal_width: float
        petal_len: float
        petal_width: float

    bento_model = bentoml.sklearn.get("iris_clf_with_feature_names:latest")
    iris_clf_runner = bento_model.to_runner()

    svc = bentoml.Service("iris_fastapi_demo", runners=[iris_clf_runner])

    @svc.api(input=JSON(pydantic_model=IrisFeatures), output=NumpyNdarray())
    def predict_bentoml(input_data: IrisFeatures) -> np.ndarray:
        input_df = pd.DataFrame([input_data.dict()])
        return iris_clf_runner.predict.run(input_df)

    fastapi_app = FastAPI()
    svc.mount_asgi_app(fastapi_app)

    @fastapi_app.get("/metadata")
    def metadata():
        return {"name": bento_model.tag.name, "version": bento_model.tag.version}

    # For demo purpose, here's an identical inference endpoint implemented via FastAPI
    @fastapi_app.post("/predict_fastapi")
    def predict(features: IrisFeatures):
        input_df = pd.DataFrame([features.dict()])
        results = iris_clf_runner.predict.run(input_df)
        return { "prediction": results.tolist()[0] }

    # BentoML Runner's async API is recommended for async endpoints
    @fastapi_app.post("/predict_fastapi_async")
    async def predict_async(features: IrisFeatures):
        input_df = pd.DataFrame([features.dict()])
        results = await iris_clf_runner.predict.async_run(input_df)
        return { "prediction": results.tolist()[0] }


In addition to FastAPI, application mounting is supported for any ASGI web applications built with any frameworks adhering to the ASGI standards.

Bundle WSGI app (e.g. Flask)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For WSGI web apps, such as a Flask app, BentoML provides a different API ``mount_wsgi_app``
which will internally convert the provided WSGI app into an ASGI app and serve side-by-side
with your BentoML Service.

Here’s an example (excerpted from :examples:`our example project <custom_web_serving/fastapi_example>`)
of mounting BentoML Service with an WSGI app built with Flask:

.. code-block:: python

    import numpy as np
    import bentoml
    from bentoml.io import NumpyNdarray
    from flask import Flask, request, jsonify

    bento_model = bentoml.sklearn.get("iris_clf:latest")
    iris_clf_runner = bento_model.to_runner()

    svc = bentoml.Service("iris_flask_demo", runners=[iris_clf_runner])


    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def predict_bentoml(input_series: np.ndarray) -> np.ndarray:
        return iris_clf_runner.predict.run(input_series)

    flask_app = Flask(__name__)
    svc.mount_wsgi_app(flask_app)

    @flask_app.route("/metadata")
    def metadata():
        return {"name": bento_model.tag.name, "version": bento_model.tag.version}

    # For demo purpose, here's an identical inference endpoint implemented via FastAPI
    @flask_app.route("/predict_flask", methods=["POST"])
    def predict():
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            input_arr = np.array(request.json, dtype=float)
            return jsonify(iris_clf_runner.predict.run(input_arr).tolist())
        else:
            return 'Content-Type not supported!'
