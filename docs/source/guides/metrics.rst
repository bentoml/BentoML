=======
Metrics
=======

Metrics are measurements of statistics about your service, which can provide information about the usage and performance of your bentos in production.

BentoML allows users to define custom metrics with `Prometheus <https://prometheus.io/>`_ to easily enable monitoring for their Bentos.
 
This article will dive into how to add custom metrics to monitor your BentoService and how you can incorporate custom metrics into a :ref:`concepts/runner:Custom Runner`.

Make sure to have `Prometheus <https://prometheus.io/download/#prometheus>`_ installed and running before continuing.

Tracking model latency with Prometheus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will build a custom histogram to track the latency of our :ref:`custom MNIST runner <concepts/runner:Custom Runner>` in this
tutorial.

.. note::

   The source code for this custom runner is :github:`available on GitHub <bentoml/BentoML/tree/main/examples/custom_model_runner>`.

Initialize our Histogram to track inference duration:

.. code-block:: python

   import bentoml

   inference_duration = bentoml.metrics.Histogram(
       name="inference_duration",
       documentation="Duration of inference",
       labelnames=["torch_version", "device_id"],
       buckets=exponential_buckets(0.001, 1.5, 10.0),
   )

This creates a :meth:`bentoml.metrics.Histogram`, which is a metric type that tracks the distribution of events with given bucket. The
:attr:`bentoml.metrics.Histogram.buckets` is a exponential bucket with a factor of 1.5 and starts at 0.001.

Follow with creating our custom MNIST runnable:

.. code-block:: python

   mnist_model = bentoml.pytorch.get("mnist_cnn:latest").
   _BuiltinRunnable = mnist_model.to_runnable()

   class MNISTRunnable(_BuiltinRunnable):
        def __init__(self):
            super().__init__()
            import torch

            print("Running on device:", self.device_id)
            self.torch_version = torch.__version__
            print("Running on torch version:", self.torch_version)

        @bentoml.Runnable.method(batchable=True, batch_dim=0)
        def __call__(self, input_arr: np.ndarray) -> np.ndarray:
            start = time.perf_counter()
            output = super().__call__(input_arr)
            inference_duration.labels(
                  torch_version=self.torch_version, device_id=self.device_id
            ).observe(time.perf_counter() - start)
            return output.argmax(dim=1)

This runnable wraps around BentoML's built-in PyTorch runnable implementation and adds aforementioned metrics.

Initialize our custom runner, and add it to the service:

.. code-block:: python

   mnist_runner = bentoml.Runner(
      MNISTRunnable,
      method_configs={"__call__": {"max_batch_size": 50, "max_latency_ms": 600}},
   )

   svc = bentoml.Service(
      "pytorch_mnist", runners=[mnist_runner], models=[mnist_model]
   )


   @svc.api(input=bentoml.io.Image(), output=bentoml.io.NumpyNdarray())
   async def predict(image: PIL.Image.Image) -> np.ndarray:
       arr = np.array(image).reshape([-1, 1, 28, 28])
       res = await mnist_runner.async_run(arr)
       return res.numpy()

.. tab-set::

    .. tab-item:: HTTP
       :sync: http

       Serve our service:

       .. code-block:: bash

          » bentoml serve-http --production

       Use the following ``prometheus.yml`` config:

       .. literalinclude:: ../../../examples/custom_model_runner/prometheus/prometheus.http.yml
          :language: python
          :caption: `prometheus.yml`

       Startup your Prometheus server in a different terminal session:

       .. code-block:: bash

          » prometheus --config.file=prometheus.yml

       In a different terminal, send a request to our service:

       .. code-block:: bash

          » curl -X POST -F "image=@test_image.png" \
                   http://0.0.0.0:3000/predict


    .. tab-item:: gRPC
       :sync: grpc

       Serve our service:

       .. code-block:: bash

          » bentoml serve-grpc --production --enable-reflection

       Use the following ``prometheus.yml`` config:

       .. literalinclude:: ../../../examples/custom_model_runner/prometheus/prometheus.grpc.yml
          :language: python
          :caption: `prometheus.yml`

       Startup your Prometheus server in a different terminal session:

       .. code-block:: bash

          » prometheus --config.file=prometheus.yml

       In a different terminal, send a request to our service:

       .. code-block:: bash

          » grpcurl -d @ -plaintext 0.0.0.0:3000 bentoml.grpc.v1alpha1.BentoService/Call <<EOT
            {
              "apiName": "predict",
              "serializedBytes": "..."
            }
            EOT

Visit `http://localhost:9090/graph <http://localhost:9090/graph>`_ and use the following query for 95th percentile inference latency:

.. code-block:: text

   histogram_quantile(0.95, rate(inference_duration_bucket[1m]))

.. image:: ../_static/img/prometheus-metrics.png

.. TODO::

    * Grafana dashboard

.. admonition:: Help us improve the project!

    Found an issue or a TODO item? You're always welcome to make contributions to the
    project and its documentation. Check out the
    `BentoML development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_
    and `documentation guide <https://github.com/bentoml/BentoML/blob/main/docs/README.md>`_
    to get started.

