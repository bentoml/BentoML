=========================
Metrics in BentoML
=========================
BentoML provides native support for all Prometheus Client Metrics. 
Users can access the metrics API by replacing  ``prometheus_client`` with ``bentoml.metrics``.

For example to write a custom metric using Prometheus Client user can write the following

.. code-block:: python
    
    import prometheus_client

    # creating metric
    my_histo = prometheus_client.Histogram(...)

The same can be accomplished by writting the following

.. code-block:: python
    
    import bentoml
    # from bentoml import metrics  <-- this would also work
    # from bentoml.metrics import Histogram  <-- this would also work
    
    # creating metric
    my_histo = bentoml.metrics.Histogram(...)



Since BentoML provides support for all metrics found in Prometheus Client, users can use the ``bentoml.metrics`` package to create any custom Metric as they would using Prometheus Client.
Here are all of the documentation for all of the metrics Prometheus Client provides.

Prometheus Client Metrics
-------------------------
Here are the signature for the Metrics provided by Prometheus Client. To access these metrics in BentoML, you have to replace ``prometheus_client`` with ``bentoml.metrics``.

.. autoclass:: prometheus_client.Counter
    :members: inc, count_exceptions
    :undoc-members:

.. autoclass:: prometheus_client.Histogram
    :members: observe, time
    :undoc-members:

.. autoclass:: prometheus_client.Summary
    :members: observe, time
    :undoc-members:

.. autoclass:: prometheus_client.Gauge
    :members: inc, dec, set, set_to_current_time, track_inprogress, time, set_function
    :undoc-members:

.. autoclass:: prometheus_client.Info
    :members: info
    :undoc-members:

.. autoclass:: prometheus_client.Enum
    :members: state
    :undoc-members:

.. autoclass:: prometheus_client.Metric
    :members: add_sample
    :undoc-members:



Using Metrics
----------------
After a metrics has been created, one can start using it right away.

.. code-block:: python

    # using metric in API endpoint
    from bentoml import metrics
    from bentoml.io import NumpyNdarray

    iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    # Creating metric
    my_histogram = metrics.Histogram(name="example histogram", documentation="docs for histogram")
    
    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        my_histogram.observe(3)  # using metric
        result = iris_clf_runner.predict.run(input_series)
        return result

As previosly mentioned, you can access metricies by replacing ``prometheus_client`` with ``bentoml.metrics``.


Accessing metrics
-----------------

Metrics can be accessed by going to ``/metrics`` for HTTP. For example, ``https://localhost:3000/metrics``. 
You can find metrics on port 3001 for gRPC. For example, ``0.0.0.0:3001``.



Additional Metric Methods
-------------------------
The following methods are availble under ``bentoml.metrics``.

start_http_server
~~~~~~~~~~~~~~~~~

.. autofunction:: bentoml.metrics.start_http_server


make_wsgi_app
~~~~~~~~~~~~~

.. autofunction:: bentoml.metrics.make_wsgi_app


generate_latest
~~~~~~~~~~~~~~~

.. autofunction:: bentoml.metrics.generate_latest



text_string_to_metric_families
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: bentoml.metrics.text_string_to_metric_families
