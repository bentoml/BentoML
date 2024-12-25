=======
Metrics
=======

Metrics are important measurements that provide insights into the usage and performance of :doc:`Services </build-with-bentoml/services>`. BentoML provides a set of default metrics for performance analysis while you can also define custom metrics with `Prometheus <https://prometheus.io/>`_.

In this document, you will:

- Learn and configure the default metrics in BentoML
- Create custom metrics for BentoML Services
- Use `Prometheus <https://prometheus.io/>`_ to scrape metrics
- Create a `Grafana <https://grafana.com/>`_ dashboard to visualize metrics

Understand metrics
------------------

You can access metrics via the ``metrics`` endpoint of a BentoML Service. This endpoint is enabled by default and outputs metrics that Prometheus can scrape to monitor your Services continuously.

Default metrics
^^^^^^^^^^^^^^^

BentoML automatically collects a set of default metrics for each Service. These metrics are tracked across different dimensions to provide detailed visibility into Service operations:

.. list-table::
   :widths: 30 10 60
   :header-rows: 1

   * - Name
     - Type
     - Dimension
   * - ``bentoml_service_request_in_progress``
     - Gauge
     - ``endpoint``, ``runner_name``, ``service_name``, ``service_version``
   * - ``bentoml_service_request_total``
     - Counter
     - ``endpoint``, ``service_name``, ``runner_name``, ``service_version``, ``http_response_code``
   * - ``bentoml_service_request_duration_seconds_sum``, ``bentoml_service_request_duration_seconds_count``, ``bentoml_service_request_duration_seconds_bucket``
     - Histogram
     - ``endpoint``, ``service_name``, ``runner_name``, ``service_version``, ``http_response_code``
   * - ``bentoml_service_adaptive_batch_size_sum``, ``bentoml_service_adaptive_batch_size_count``, ``bentoml_service_adaptive_batch_size_bucket``
     - Histogram
     - ``method_name``, ``service_name``, ``runner_name``, ``service_version``, ``worker_index``

- ``request_in_progress``: The number of requests that are currently being processed by a Service.
- ``request_total``: The total number of requests that a Service has processed.
- ``request_duration_seconds``: The time taken to process requests, including the total sum of request processing time, count of requests processed, and distribution across specified duration buckets.
- ``adaptive_batch_size``: The adaptive batch sizes used during Service execution, which is relevant for optimizing performance in batch processing scenarios. You need to enable :doc:`adaptive batching </get-started/adaptive-batching>` to collect this metric.

Metric types
^^^^^^^^^^^^

BentoML supports all metric types provided by Prometheus.

- ``Gauge``: A metric that represents a single numerical value that can arbitrarily go up and down.
- ``Counter``: A cumulative metric that only increases, useful for counting total requests.
- ``Histogram``: Tracks the number of observations and the sum of the observed values in configurable buckets, allowing you to calculate averages, percentiles, and so on.
- ``Summary``: Similar to Histogram but provides a total count of observations and a sum of observed values.

For more information, see `the Prometheus documentation <https://prometheus.io/docs/concepts/metric_types/>`_.

Dimensions
^^^^^^^^^^

Dimensions tracked for the default BentoML metrics include:

- ``endpoint``: The specific API endpoint being accessed.
- ``runner_name``: The name of the running Service handling the request.
- ``service_name``: The name of the Bento Service handling the request.
- ``service_version``: The version of the Service.
- ``http_response_code``: The HTTP response code of the request.
- ``worker_index``: The worker instance that is running the inference.

Configure default metrics
-------------------------

To customize how metrics are collected and reported in BentoML, use the ``metrics`` parameter within the ``@bentoml.service`` decorator:

.. code-block:: python

    @bentoml.service(metrics={
        "enabled": True,
        "namespace": "custom_namespace",
    })
    class MyService:
        # Service implementation

- ``enabled``: This option is enabled by default. When enabled, you can access the metrics through the ``metrics`` endpoint of a BentoML Service.
- ``namespace``: Follows the `labeling convention <https://prometheus.io/docs/practices/naming/#metric-and-label-naming>`_ of Prometheus. The default namespace is ``bentoml_service``, which covers most use cases.

Customize the duration bucket size
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can customize the `duration bucket size <https://prometheus.io/docs/practices/histograms/#count-and-sum-of-observations>`_ of ``request_duration_seconds`` in the following two ways:

- **Manual bucket definition**. Specify explicit steps using ``buckets``:

  .. code-block:: python

        @bentoml.service(metrics={
            "enabled": True,
            "namespace": "bentoml_service",
            "duration": {
                "buckets": [0.1, 0.2, 0.5, 1, 2, 5, 10]
            }
        })
        class MyService:
            # Service implementation

- **Exponential bucket generation**. Automatically generate exponential buckets with any given ``min``, ``max`` and ``factor`` values.

  - ``min``: The lower bound of the smallest bucket in the histogram.
  - ``max``: The upper bound of the largest bucket in the histogram.
  - ``factor``: Determines the exponential growth rate of the bucket sizes. Each subsequent bucket boundary is calculated by multiplying the previous boundary by the factor.

  .. code-block:: python

        @bentoml.service(metrics={
            "enabled": True,
            "namespace": "bentoml_service",
            "duration": {
                "min": 0.1,
                "max": 10,
                "factor": 1.2
            }
        })
        class MyService:
            # Service implementation

.. note::

    - ``duration.min``, ``duration.max`` and ``duration.factor`` are mutually exclusive with ``duration.buckets``.
    - ``duration.factor`` must be greater than 1 to ensure each subsequent bucket is larger than the previous one.
    - The buckets for the ``adaptive_batch_size`` Histogram are calculated based on the ``max_batch_size`` defined. The bucket sizes start at 1 and increase exponentially up to the ``max_batch_size`` with a factor of 2.

By default, BentoML uses the `duration buckets <https://github.com/prometheus/client_python/blob/f17a8361ad3ed5bc47f193ac03b00911120a8d81/prometheus_client/metrics.py#L544>`_ provided by Prometheus.

Create custom metrics
---------------------

You can define and use custom metrics of ``Counter``, ``Histogram``, ``Summary``, and ``Gauge`` within your BentoML Service using the ``prometheus_client`` API.

Prerequisites
^^^^^^^^^^^^^

Install the `Prometheus Python client <https://github.com/prometheus/client_python>`_ package.

.. code-block:: bash

    pip install prometheus-client

Define custom metrics
^^^^^^^^^^^^^^^^^^^^^

To define custom metrics, use the metric classes from the ``prometheus_client`` module and set the following parameters as needed:

- ``name``: A unique string identifier for the metric.
- ``documentation``: A description of what the metric measures.
- ``labelnames``: A list of strings defining the labels to apply to the metric. Labels add dimensions to the metric, which are useful for querying and aggregation purposes. When you record a metric, you specify the labels in the format ``<metric_object>.labels(<label_name>='<label_value>').<metric_function>``. Once you define a label for a metric, all instances of that metric must include that label with some value.

  The value of a label can also be dynamic, meaning it can change based on the context of the tracked metric. For example, you can use a label to log the version of model serving predictions, and this version label can change as you update the model.

- ``buckets``: A Histogram-specific parameter which defines the boundaries for Histogram buckets, useful for categorizing measurement ranges. The list should end with ``float('inf')`` to capture all values that exceed the highest defined boundary. See the Prometheus documentation on `Histogram <https://prometheus.io/docs/practices/histograms/>`_ for more details.

.. tab-set::

   .. tab-item:: Histogram

      .. code-block:: python

          import bentoml
          from prometheus_client import Histogram

          # Define Histogram metric
          inference_duration_histogram = Histogram(
              name="inference_duration_seconds",
              documentation="Time taken for inference",
              labelnames=["endpoint"],
              buckets=(
                0.005, 0.01, 0.025, 0.05, 0.075,
                0.1, 0.25, 0.5, 0.75, 1.0,
                2.5, 5.0, 7.5, 10.0, float("inf"),
              ),
          )

          @bentoml.service
          class HistogramService:
              def __init__(self) -> None:
                  # Initialization code

              @bentoml.api
              def infer(self, text: str) -> str:
                  # Track the metric
                  inference_duration_histogram.labels(endpoint='summarize').observe(512)
                  # Implementation logic

   .. tab-item:: Counter

      .. code-block:: python

          import bentoml
          from prometheus_client import Counter

          # Define Counter metric
          inference_requests_counter = Counter(
              name="inference_requests_total",
              documentation="Total number of inference requests",
              labelnames=["endpoint"],
          )

          @bentoml.service
          class CounterService:
              def __init__(self) -> None:
                  # Initialization code

              @bentoml.api
              def infer(self, text: str) -> str:
                  # Track the metric
                  inference_requests_counter.labels(endpoint='summarize').inc()  # Increment the counter by 1
                  # Implementation logic

   .. tab-item:: Summary

      .. code-block:: python

          import bentoml
          from prometheus_client import Summary

          # Define Summary metric
          response_size_summary = Summary(
              name="response_size_bytes",
              documentation="Response size in bytes",
              labelnames=["endpoint"],
          )

          @bentoml.service
          class SummaryService:
              def __init__(self) -> None:
                  # Initialization code

              @bentoml.api
              def infer(self, text: str) -> str:
                  # Track the metric
                  response_size_summary.labels(endpoint='summarize').observe(512)
                  # Implementation logic

   .. tab-item:: Gauge

      .. code-block:: python

          import bentoml
          from prometheus_client import Gauge

          # Define Gauge metric
          in_progress_gauge = Gauge(
              name="in_progress_requests",
              documentation="In-progress inference requests",
              labelnames=["endpoint"],
          )

          @bentoml.service
          class GaugeService:
              def __init__(self) -> None:
                  # Initialization code

              @bentoml.api
              def infer(self, text: str) -> str:
                  # Track the metric
                  in_progress_gauge.labels(endpoint='summarize').inc()  # Increment by 1
                  in_progress_gauge.labels(endpoint='summarize').dec()  # Decrement by 1
                  # Implementation logic

For more information on ``prometheus_client``, see the `Prometheus Python client library documentation <https://prometheus.github.io/client_python/>`_.

An example with custom metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following ``service.py`` file contains a custom Histogram and a Counter metric to measure the inference time and track the total number of requests.

.. code-block:: python

    from __future__ import annotations
    import bentoml
    from prometheus_client import Histogram, Counter
    from transformers import pipeline
    import time

    # Define the metrics
    request_counter = Counter(
        name='summary_requests_total',
        documentation='Total number of summarization requests',
        labelnames=['status']
    )

    inference_time_histogram = Histogram(
        name='inference_time_seconds',
        documentation='Time taken for summarization inference',
        labelnames=['status'],
        buckets=(0.1, 0.2, 0.5, 1, 2, 5, 10, float('inf'))  # Example buckets
    )

    EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century."

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    class Summarization:
        def __init__(self) -> None:
            self.pipeline = pipeline('summarization')

        @bentoml.api
        def summarize(self, text: str = EXAMPLE_INPUT) -> str:
            start_time = time.time()
            try:
                result = self.pipeline(text)
                summary_text = result[0]['summary_text']
                # Capture successful requests
                status = 'success'
            except Exception as e:
                # Capture failures
                summary_text = str(e)
                status = 'failure'
            finally:
                # Measure how long the inference took and update the histogram
                inference_time_histogram.labels(status=status).observe(time.time() - start_time)
                # Increment the request counter
                request_counter.labels(status=status).inc()

            return summary_text

Run this Service locally:

.. code-block:: bash

    bentoml serve service:Summarization

Make sure you have sent some requests to the ``summarize`` endpoint, then view the custom metrics by running the following command. You need to replace ``inference_time_seconds`` and ``summary_requests_total`` with your own metric names.

.. code-block:: bash

    curl -X 'GET' 'http://localhost:3000/metrics' -H 'accept: */*' | grep -E 'inference_time_seconds|summary_requests_total'

Expected output:

.. code-block:: bash

    # HELP summary_requests_total Total number of summarization requests
    # TYPE summary_requests_total counter
    summary_requests_total{status="success"} 12.0
    # HELP inference_time_seconds Time taken for summarization inference
    # TYPE inference_time_seconds histogram
    inference_time_seconds_sum{status="success"} 51.74311947822571
    inference_time_seconds_bucket{le="0.1",status="success"} 0.0
    inference_time_seconds_bucket{le="0.2",status="success"} 0.0
    inference_time_seconds_bucket{le="0.5",status="success"} 0.0
    inference_time_seconds_bucket{le="1.0",status="success"} 0.0
    inference_time_seconds_bucket{le="2.0",status="success"} 0.0
    inference_time_seconds_bucket{le="5.0",status="success"} 12.0
    inference_time_seconds_bucket{le="10.0",status="success"} 12.0
    inference_time_seconds_bucket{le="+Inf",status="success"} 12.0
    inference_time_seconds_count{status="success"} 12.0

Use Prometheus to scrape metrics
--------------------------------

You can integrate Prometheus to scrape and visualize both default and custom metrics from your BentoML Service.

1. `Install Prometheus <https://prometheus.io/docs/prometheus/latest/installation/>`_.
2. Create `a Prometheus configuration file <https://prometheus.io/docs/prometheus/latest/configuration/configuration/>`_ to define scrape jobs. Here is an example that scrapes metrics every 5 seconds from a BentoML Service.

   .. code-block:: yaml
      :caption: `prometheus.yml`

      global:
        scrape_interval: 5s
        evaluation_interval: 15s

      scrape_configs:
        - job_name: prometheus
          metrics_path: "/metrics" # The metrics endpoint of the BentoML Service
          static_configs:
            - targets: ["0.0.0.0:3000"] # The address where the BentoML Service is running

3. Make sure you have a BentoML Service running, then start Prometheus in a different terminal session using the configuration file you created:

   .. code-block:: bash

        ./prometheus --config.file=/path/to/the/file/prometheus.yml

4. Once Prometheus is running, access its web UI by visiting ``http://localhost:9090`` in your web browser. This interface allows you to query and visualize metrics collected from your BentoML Service.
5. Use `PromQL expressions <https://prometheus.io/docs/prometheus/latest/querying/basics/>`_ to query and visualize metrics. For example, to get the 99th percentile of request durations to the ``/encode`` endpoint over the last minute, use:

   .. code-block:: bash

        histogram_quantile(0.99, rate(bentoml_service_request_duration_seconds_bucket{endpoint="/encode"}[1m]))

   .. image:: ../../_static/img/build-with-bentoml/observability/metrics/prome-ui-bentoml.png
      :alt: Prometheus UI for BentoML metrics

Create a Grafana dashboard
--------------------------

Grafana is an analytics platform that allows you to create dynamic and informative `dashboards <https://grafana.com/grafana/dashboards/>`_ to visualize BentoML metrics. Do the following to create a Grafana dashboard.

1. `Install Grafana <https://grafana.com/docs/grafana/latest/setup-grafana/installation/>`_.
2. By default, Grafana runs on port ``3000``, which conflicts with BentoML's default port. To avoid this, change Grafana's default port. For example:

   .. code-block:: bash

        sudo nano /etc/grafana/grafana.ini

   Find the ``[http]`` section and change ``http_port`` to a free port like ``4000``:

   .. code-block:: bash

        ;http_port = 3000  # Change it to a port of your choice and uncomment the line by removing the semicolon
        http_port = 4000

3. Save the file and restart Grafana to apply the change:

   .. code-block:: bash

        sudo systemctl restart grafana-server

4. Access the Grafana web UI at ``http://localhost:4000/`` (use your own port). Log in with the default credentials (``admin``/``admin``).
5. In the Grafana search box at the top, enter ``Data sources`` and add Prometheus as an available option. In **Connection**, set the URL to the address of your running Prometheus instance, such as ``http://localhost:9090``. Save the configuration and test the connection to ensure Grafana can retrieve data from Prometheus.

   .. image:: ../../_static/img/build-with-bentoml/observability/metrics/grafana-bentoml-1.png
      :alt: Add Prometheus in Grafana

6. With Prometheus configured as a data source, you can create a new dashboard. Start by adding a panel and selecting a metric to visualize, such as ``bentoml_service_request_duration_seconds_bucket``. Grafana offers a wide array of visualization options, from simple line graphs to more complex representations like heatmaps or gauges.

   .. image:: ../../_static/img/build-with-bentoml/observability/metrics/grafana-bentoml-2.png
      :alt: Grafana UI for BentoML metrics

   For detailed instructions on dashboard creation and customization, read the `Grafana documentation <https://grafana.com/docs/grafana/latest/dashboards/>`_.
