==============
Configurations
==============

BentoML provides a configuration interface that allows you to customize the runtime behavior for individual Services within a Bento. This document explains the available configuration fields and offers best practice recommendations for configuring your BentoML Services.

How do configurations work
--------------------------

BentoML's default configurations are suitable for a wide range of use cases. However, for more granular control over BentoML's features, you can customize these runtime behaviors (like resource allocation and timeout) using the ``@bentoml.service`` decorator for each Service in your ``service.py`` file.

.. note::

   If you are using BentoML versions prior to 1.2, you need to `set these runtime configurations <https://docs.bentoml.com/en/v1.1.11/guides/configuration.html>`_ via a separate ``configuration.yaml`` file.

You only need to specify the configurations you want to customize. BentoML automatically fills in any unspecified configurations with their default values. The following is an example:

.. code-block:: python

    @bentoml.service(
        resources={"gpu": 1},
        traffic={"timeout": 60},
    )
    class Summarization:
        # Service implementation

- ``"gpu": 1`` assigns one GPU to the Service's workers. See :ref:`resources` for details.
- ``"timeout": 60`` sets the maximum duration (in seconds) that the Service will wait for a response before timing out. In this case, it's set to 60 seconds.

Configuration fields
--------------------

BentoML offers a comprehensive set of configuration fields, allowing detailed customization of Services. You can use them to meet the specific requirements of different deployment environments and use cases.

.. _resources:

``resources``
^^^^^^^^^^^^^

The ``resources`` field in BentoML allows you to specify the resource allocation for a Service, including CPU, memory, and GPU. It is useful for managing and optimizing resource usage, particularly when dealing with resource-intensive tasks. Available fields are:

- ``cpu``: The number (or the amount) of CPUs that the Service should use. This is a string, like ``“200m”`` or ``“1”``.
- ``memory``: The amount of memory that the Service should use.
- ``gpu``: The number of GPUs allocated to the Service. When the Service starts, BentoML assigns this number of GPUs to the Service's workers by setting ``CUDA_VISIBLE_DEVICES``. GPU assignment is skipped if ``CUDA_VISIBLE_DEVICES`` or ``BENTOML_DISABLE_GPU_ALLOCATION`` is already set in the environment, or if ``workers`` is set to ``cpu_count``.
- ``gpu_type``: A specific type of GPU, following the naming conventions of cloud providers like AWS EC2 or GCP GPU instances.

.. note::

   ``cpu``, ``memory`` and ``gpu_type`` are stored in the Bento but have no effect when BentoML runs the Service. To limit CPU and memory or schedule the Service on a specific GPU type, configure your container runtime or orchestrator, such as Docker resource constraints or Kubernetes resource requests and node selectors.

Here is an example:

.. code-block:: python

    @bentoml.service(
        resources={
            "cpu": "1",
            "memory": "500Mi",
            "gpu": 4,
            "gpu_type": "nvidia-tesla-a100"
        }
    )
    class MyService:
        # Service implementation

``workers``
^^^^^^^^^^^

:doc:`/build-with-bentoml/parallelize-requests` are the processes that run the code logic within a Service instance. You use ``workers`` in the ``@bentoml.service`` decorator to define the process-level parallelism within a Service. This configuration is useful for optimizing performance, particularly for high-throughput or compute-intensive Services. ``workers`` defaults to ``1``.

To specify the number of workers (for example, ``3``) within a Service:

.. code-block:: python

    @bentoml.service(workers=3)
    class MyService:
        # Service implementation

``workers`` also allows for dynamic CPU allocation using the ``cpu_count`` option. This feature can be particularly useful in environments where you want to automatically scale the number of worker processes based on the available CPU cores.

.. code-block:: python

    @bentoml.service(workers="cpu_count")
    class MyService:
        # Service implementation

``image``
^^^^^^^^^

``image`` allows you to configure the runtime specifications for building a Bento.

.. code-block:: python

    my_image = bentoml.images.Image(python_version='3.11') \
        .python_packages("torch", "transformers")

    @bentoml.service(image=my_image)
    class MyService:
        # Service implementation

For more information, see :doc:`/build-with-bentoml/runtime-environment`.

``labels``
^^^^^^^^^^

``labels`` allows you to set the labels for your Bento. They are key-value pairs that you can use to organize and filter your Bentos.

.. code-block:: python

    @bentoml.service(
        labels={
            "owner": "ai-team",
            "type": "test",
            "project": "llm",
        }
    )
    class MyService:
        # Service implementation

.. _envs:

``envs``
^^^^^^^^

``envs`` allows you to set the environment variables required for your Service. Each environment variable is defined with ``name`` and ``value`` keys. To avoid exposing sensitive information, you can omit the value and set it when you run the Service, for example, with ``docker run -e HF_TOKEN=<value>``.

.. code-block:: python

    @bentoml.service(
        envs=[
            {"name": "HF_TOKEN"},  # Omit the value
            {"name": "DB_HOST", "value": "localhost"}
        ]
    )
    class MyService:
        # Service implementation

.. _config-traffic:

``traffic``
^^^^^^^^^^^

``traffic`` in BentoML allows you to manage how requests are handled by your Service. It includes settings for managing request concurrency and ensuring timely responses, which are helpful for optimizing the Service's responsiveness and load management. The following fields are available:

- ``timeout``: The maximum time in seconds the Service waits for a response to start before returning HTTP 504. For streaming responses, only the start of the response must arrive within this time. It defaults to 60 seconds.
- ``max_concurrency``: The hard limit on the number of requests a Service processes simultaneously, split evenly across its workers. Requests beyond the limit are rejected immediately with HTTP 429. Health check and metrics endpoints are not counted.
- ``concurrency``: The ideal number of simultaneous requests a Service is designed to handle. This field is reserved and has no effect in open-source BentoML.
- ``external_queue``: Whether to queue excess requests in an external request queue. This field is reserved and has no effect in open-source BentoML.

Here is an example of configuring these settings in your Service definition:

.. code-block:: python

    @bentoml.service(
        traffic={
            "timeout": 120,
            "max_concurrency": 50,
        }
    )
    class MyService:
        # Service implementation

``runner_probe``
^^^^^^^^^^^^^^^^

Configure health check settings for the Service using the endpoints ``readyz``, ``livez``, and ``healthz``. Available fields are:

- ``enabled``: Determines whether the health checks are enabled.
- ``timeout``: The maximum time in seconds to wait for a health check probe to complete before considering it failed.
- ``period``: The frequency, in seconds, at which the health check probes are performed.

Here is an example:

.. code-block:: python

    @bentoml.service(runner_probe={"enabled": True, "timeout": 1, "period": 10})
    class MyService:
        # Service implementation

``logging``
^^^^^^^^^^^

Customize server-side logging, including the content type and length of requests and responses, and trace ID formats.

Here is an example:

.. code-block:: python

    @bentoml.service(logging={
        "access": {
            "enabled": True,
            "request_content_length": True,
            "request_content_type": True,
            "response_content_length": True,
            "response_content_type": True,
            "format": {
                "trace_id": "032x",
                "span_id": "016x"
            }
        }
    })
    class MyService:
        # Service implementation

For more information, see :doc:`/build-with-bentoml/observability/logging`.

``ssl``
^^^^^^^

``ssl`` enables SSL/TLS for secure communication over HTTP requests. It is helpful for protecting sensitive data in transit and ensuring secure connections between clients and your Service.

BentoML parses all the available fields directly to `Uvicorn <https://www.uvicorn.org/settings/#https>`_. Here is an example:

.. code-block:: python

    @bentoml.service(ssl={
        "enabled": True,
        "certfile": "/path/to/certfile",
        "keyfile": "/path/to/keyfile",
        "ca_certs": "/path/to/ca_certs",
        "keyfile_password": "",
        "version": 17,
        "cert_reqs": 0,
        "ciphers": "TLSv1"
    })
    class MyService:
        # Service implementation

``http``
^^^^^^^^

``http`` allows you to customize the settings for the HTTP server that serves your BentoML Service.

By default, BentoML starts an HTTP server on port ``3000``. To change the port:

.. code-block:: python

    @bentoml.service(http={"port": 5000})
    class MyService:
        # Service implementation

You can configure `CORS <https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS>`_ settings if your Service needs to accept cross-origin requests. By default, CORS is disabled. If it is enabled, all fields under ``http.cors`` will be parsed to `CORSMiddleware <https://www.starlette.io/middleware/#corsmiddleware>`_. Here is an example:

.. code-block:: python

    @bentoml.service(http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["http://myorg.com:8080", "https://myorg.com:8080"],
            "access_control_allow_methods": ["GET", "OPTIONS", "POST", "HEAD", "PUT"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
            "access_control_allow_origin_regex": "https://.*\.my_org\.com",
            "access_control_max_age": 1200,
            "access_control_expose_headers": ["Content-Length"]
        }
    })
    class MyService:
        # Service implementation

Configuring CORS is important when your Service is accessed from web applications hosted on different domains. Proper CORS settings ensure that your Service can securely handle requests from allowed origins, enhancing both security and usability.

By customizing the ``http`` configuration, you can fine-tune how your BentoML Service interacts over HTTP, including adapting to specific network environments, securing cross-origin interactions, and ensuring compatibility with various client applications.

``monitoring``
^^^^^^^^^^^^^^

``monitoring`` allows you to collect logs and keep track of the performance and health of a Service for maintaining its reliability and efficiency. By default, BentoML provides a built-in monitoring mechanism, while you can customize it by setting a configuration file in YAML.

Here is an example:

.. code-block:: python

    @bentoml.service(monitoring={
        "enabled": True,
        "type": "default",
        "options": {
            "log_config_file": "path/to/log_config.yaml", # A configuration file for customizing monitoring behavior, using Python's logging module
            "log_path": "monitoring" # The directory where logs will be exported
        }
    })
    class MyService:
        # Service implementation

For more information, see :doc:`/build-with-bentoml/observability/monitoring-and-data-collection`.

``metrics``
^^^^^^^^^^^

``metrics`` allows you to collect and customize metrics of ``Counter``, ``Histogram``, ``Summary``, and ``Gauge`` types. By default, this feature is enabled.

Here is an example:

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

For more information, see :doc:`/build-with-bentoml/observability/metrics`.

``tracing``
^^^^^^^^^^^

You can configure tracing with different exporters like Zipkin, Jaeger, and OTLP. The specific configurations may vary depending on the exporter type defined.

Here is an example:

.. code-block:: python

    import bentoml

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
        tracing={
            # Common configurations
            "exporter_type": "jaeger",
            "sample_rate": 1.0,
            "timeout": 5,
            "max_tag_value_length": 256,
            "excluded_urls": "readyz",
            "jaeger": {
                # Specific configurations of the exporter
        }
    )
    class MyService:
       # Service implementation code

For more information, see :doc:`/build-with-bentoml/observability/tracing`.

For full schema of the configurations, see `this file <https://github.com/bentoml/BentoML/blob/1.2/src/bentoml/_internal/configuration/v2/default_configuration.yaml>`_.
