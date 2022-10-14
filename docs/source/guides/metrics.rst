=========================
Custom Metrics in BentoML
=========================
Metrics are used for monitoring different aspect of user-defined APIs. 
In concept, being able to closely monitor how endpoints are behaving, and being used, will give you more control over how to customize and optimization your endpoints. 
In practice, ``bentoml.metrics`` supports all exposed functions from ``prometheus_client``, which in turn means that you can create any custome metric following the ``prometheus_client`` standards.
Furthermore, by replacing ``prometheus_client`` with ``bentoml.metrics``, user can use all of the features ``prometheus_client`` offers. 


Creating Metrics
----------------
Here are the signature for the Metrics API and a short example on how to create them. 
First thing to note about creating a metric is how to access them. 

.. autoclass:: prometheus_client.metrics.Counter
    :members: inc, count_exceptions
    :undoc-members:

.. autoclass:: prometheus_client.metrics.Histogram
    :members: observe, time
    :undoc-members:

.. autoclass:: prometheus_client.metrics.Summary
    :members: observe, time
    :undoc-members:

.. autoclass:: prometheus_client.metrics.Gauge
    :members: inc, dec, set, set_to_current_time, track_inprogress, time, set_function
    :undoc-members:

.. autoclass:: prometheus_client.metrics.Info
    :members: info
    :undoc-members:

.. autoclass:: prometheus_client.metrics.Enum
    :members: state
    :undoc-members:

.. autoclass:: prometheus_client.metrics.Metric
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
