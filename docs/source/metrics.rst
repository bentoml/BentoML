=======
Metrics
=======
BentoML now supports all metrics found in Prometheus Client. 
User can create custom metrics and attach them to API endpoints and access them through the ``/metrics`` endpoint.


Creating Metrics
----------------
First thing to note about creating a metric is how to access them. 
You can access metrics directly from the ``bentoml`` package. 

Counter
~~~~~~~~~~~~~~~
.. code-block:: python

    from bentoml import metrics
    # from bentoml.metrics import Counter -- you can also impprt Histogram directly
    counter = metrics.Counter(...)


Histogram
~~~~~~~~~~~~~~~

.. code-block:: python

    from bentoml import metrics
    histogram = metrics.Histogram(...)


Summary
~~~~~~~~~~~~~~~

.. code-block:: python

    from bentoml import metrics
    summary = metrics.Summary(...)


Gauge
~~~~~~~~~~~~~~~

.. code-block:: python

    from bentoml import metrics
    gauge = metrics.Gauge(...)


Info
~~~~~~~~~~~~~~~

.. code-block:: python

    from bentoml import metrics
    info = metrics.Info(...)


Enum
~~~~~~~~~~~~~~~

.. code-block:: python

    from bentoml import metrics
    enum = metrics.Enum(...)


Metric
~~~~~~~~~~~~~~~

.. code-block:: python

    from bentoml import metrics
    metric = metrics.Metric(...)




Using Metrics
----------------
After a metrics has been created, one can start using it right away.

.. code-block:: python

    # using metric in API endpoint
    from bentoml import metrics
    from bentoml.io import NumpyNdarray

    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    my_histogram = metrics.Histogram(name="example histogram", documentation="docs for histogram")
    
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        my_histogram.observe(3)  # using metric
        result = iris_clf_runner.predict.run(input_series)
        return result

Prometheus Methods
------------------
Since bentoml metrics support all prometheus client objects and methods, you can use the following methods. 

start_http_server
~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bentoml.metrics import start_http_server
    
    start_http_server(...)


make_wsgi_app
~~~~~~~~~~~~~

.. code-block:: python

    from bentoml.metrics import make_wsgi_app

    make_wsgi_app(...)


generate_latest
~~~~~~~~~~~~~~~

.. code-block:: python

    from bentoml.metrics import generate_latest

    generate_latest(...)


text_string_to_metric_families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from bentoml.metrics import text_string_to_metric_families

    text_string_to_metric_families(...)

