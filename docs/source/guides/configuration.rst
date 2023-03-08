=============
Configuration
=============

*time expected: 11 minutes*

BentoML provides a configuration interface that allows you to customize the runtime
behaviour of your BentoService.  This article highlight and consolidates the configuration
fields definition, as well as some of recommendation for best-practice when configuring
your BentoML.

   Configuration is best used for scenarios where the customizations can be specified once
   and applied anywhere among your organization using BentoML.

BentoML comes with out-of-the-box configuration that should work for most use cases.

However, for more advanced users who wants to fine-tune the feature suites BentoML has to offer,
users can configure such runtime variables and settings via a configuration file, often referred to as
``bentoml_configuration.yaml``.

.. note::

   This is not to be **mistaken** with the ``bentofile.yaml`` which is used to define and
   package your :ref:`Bento 🍱 <concepts/bento:What is a Bento?>`

   This configuration file are for BentoML runtime configuration.

Providing configuration during serve runtime
--------------------------------------------

BentoML configuration is a :wiki:`YAML` file which can then be specified via the environment variable ``BENTOML_CONFIG``.

For example, given the following ``bentoml_configuration.yaml`` that specify that the
server should only use 4 workers:

.. code-block:: yaml
   :caption: `~/bentoml_configuration.yaml`

   version: 1
   api_server:
     workers: 4

Said configuration then can be parsed to :ref:`bentoml serve <reference/cli:serve>` like
below:

.. code-block:: bash

   » BENTOML_CONFIG=~/bentoml_configuration.yaml bentoml serve iris_classifier:latest --production

.. note::

   Users will only have to specify a partial configuration with properties they wish to customize. BentoML
   will then fill in the rest of the configuration with the default values [#default_configuration]_.

   In the example above, the number of API workers count is overridden to 4.
   Remaining properties will take their defaults values.

.. seealso::

   :ref:`guides/configuration:Configuration fields`


Overriding configuration with environment variables
--------------------------------------------------

Users can also override configuration fields with environment variables. by defining
an oneline value of a "flat" JSON via ``BENTOML_CONFIG_OPTIONS``:

.. code-block:: yaml

   $ BENTOML_CONFIG_OPTIONS='runners.pytorch_mnist.resources."nvidia.com/gpu"[0]=0 runners.pytorch_mnist.resources."nvidia.com/gpu"[1]=2' \
            bentoml serve pytorch_mnist_demo:latest --production

Which the override configuration will be intepreted as:

.. code-block:: yaml

   runners:
    pytorch_mnist:
      resources:
        nvidia.com/gpu: [0, 2]

.. note::

   For fields that represents a iterable type, the override array must have a space
   separating each element:

   .. image:: /_static/img/configuration-override-env.png
      :alt: Configuration override environment variable


Mounting configuration to containerized Bento
---------------------------------------------

To mount a configuration file to a containerized BentoService, user can use the
|volume_mount|_ option to mount the configuration file to the container and
|env_flag|_ option to set the ``BENTOML_CONFIG`` environment variable:

.. code-block:: bash

   $ docker run --rm -v /path/to/configuration.yml:/home/bentoml/configuration.yml \
                -e BENTOML_CONFIG=/home/bentoml/configuration.yml \
                iris_classifier:6otbsmxzq6lwbgxi serve --production

Voila! You have successfully mounted a configuration file to your containerized BentoService.

.. _env_flag: https://docs.docker.com/engine/reference/commandline/run/#set-environment-variables--e---env---env-file

.. |env_flag| replace:: ``-e``

.. _volume_mount: https://docs.docker.com/storage/volumes/#choose-the--v-or---mount-flag

.. |volume_mount| replace:: ``-v``


Configuration fields
--------------------

On the top level, BentoML configuration [#default_configuration]_ has three fields:

* ``version``: The version of the configuration file. This is used to determine the
  compatibility of the configuration file with the current BentoML version.

* ``api_server``: Configuration for BentoML API server.

* ``runners`` [#runners_configuration]_: Configuration for BentoService runners.

``version``
^^^^^^^^^^^

BentoML configuration provides a ``version`` field, which enables users to easily specify
and upgrade their configuration file as BentoML evolves.

This field will follow BentoML major version number. For every patch releases that
introduces new configuration fields, a compatibility layer will be provided to ensure
there is no breaking changes.

.. epigraph::

   Note that ``version`` is not a required field, and BentoML will default to version 1 if
   it is not specified.

   However, we encourage users to always version their BentoML configuration.

``api_server``
^^^^^^^^^^^^^^

The following options are available for the ``api_server`` section:

+-------------+-------------------------------------------------------------+-------------------------------------------------+
| Option      | Description                                                 | Default                                         |
+=============+=============================================================+=================================================+
| ``workers`` | Number of API workers for to spawn                          | null [#default_workers]_                        |
+-------------+-------------------------------------------------------------+-------------------------------------------------+
| ``timeout`` | Timeout for API server in seconds                           | 60                                              |
+-------------+-------------------------------------------------------------+-------------------------------------------------+
| ``backlog`` | Maximum number of connections to hold in backlog            | 2048                                            |
+-------------+-------------------------------------------------------------+-------------------------------------------------+
| ``metrics`` | Key and values to enable metrics feature                    | See :ref:`guides/configuration:\`\`metrics\`\`` |
+-------------+-------------------------------------------------------------+-------------------------------------------------+
| ``logging`` | Key and values to enable logging feature                    | See :ref:`guides/logging:Logging Configuration` |
+-------------+-------------------------------------------------------------+-------------------------------------------------+
| ``http``    | Key and values to configure HTTP API server                 | See :ref:`guides/configuration:\`\`http\`\``    |
+-------------+-------------------------------------------------------------+-------------------------------------------------+
| ``grpc``    | Key and values to configure gRPC API server                 | See :ref:`guides/configuration:\`\`grpc\`\``    |
+-------------+-------------------------------------------------------------+-------------------------------------------------+
| ``ssl``     | Key and values to configure SSL                             | See :ref:`guides/configuration:\`\`ssl\`\``     |
+-------------+-------------------------------------------------------------+-------------------------------------------------+
| ``tracing`` | Key and values to configure tracing exporter for API server | See :doc:`/guides/tracing`                      |
+-------------+-------------------------------------------------------------+-------------------------------------------------+

``metrics``
"""""""""""

BentoML utilises `Prometheus <https://prometheus.io/>`_ to collect metrics from the API server. By default, this feature is enabled.

To disable this feature, set ``api_server.metrics.enabled`` to ``false``:

.. code-block:: yaml

   api_server:
     metrics:
       enabled: false

Following `labeling convention <https://prometheus.io/docs/practices/naming/#metric-and-label-naming>`_ set by Prometheus, metrics generated
by BentoML API server components will have ``namespace`` `bentoml_api_server`, which can
also be overridden by setting ``api_server.metrics.namespace``:

.. code-block:: yaml

   api_server:
     metrics:
       namespace: custom_namespace

.. epigraph::

   :bdg-info:`Note:` for most use cases, users should not need to change the default ``namespace`` value.

There are three types of metrics every BentoML API server will generate:

- ``request_duration_seconds``: This is a `Histogram <https://prometheus.io/docs/concepts/metric_types/#histogram>`_ that measures the HTTP request duration in seconds.

  There are two ways for users to customize `duration bucket size <https://prometheus.io/docs/practices/histograms/#count-and-sum-of-observations>`_ for this metrics:

  - Provides a manual bucket steps via ``api_server.metrics.duration.buckets``:

    .. code-block:: yaml

       api_server:
         metrics:
           duration:
             buckets: [0.1, 0.2, 0.5, 1, 2, 5, 10]

  - Automatically generate an exponential buckets with any given ``min``, ``max`` and ``factor``:

    .. code-block:: yaml

       api_server:
         metrics:
           duration:
             min: 0.1
             max: 10
             factor: 1.2

  .. note::

     - ``duration.min``, ``duration.max`` and ``duration.factor`` are mutually exclusive with ``duration.buckets``.

     - ``duration.factor`` must be greater than 1.

  By default, BentoML will respect the default `duration buckets <https://github.com/prometheus/client_python/blob/f17a8361ad3ed5bc47f193ac03b00911120a8d81/prometheus_client/metrics.py#L544>`_ provided by Prometheus.

- ``request_total``: This is a `Counter <https://prometheus.io/docs/concepts/metric_types/#counter>`_ that measures the total number of HTTP requests.

- ``request_in_progress``: This is a `Gauge <https://prometheus.io/docs/concepts/metric_types/#gauge>`_ that measures the number of HTTP requests in progress.

The following options are available for the ``metrics`` section:

+----------------------+-------------------------------------+-------------------------------------------------------+
| Option               | Description                         | Default                                               |
+======================+=====================================+=======================================================+
| ``enabled``          | Enable metrics feature              | ``true``                                              |
+----------------------+-------------------------------------+-------------------------------------------------------+
| ``namespace``        | Namespace for metrics               | ``bentoml_api_server``                                |
+----------------------+-------------------------------------+-------------------------------------------------------+
| ``duration.buckets`` | Duration buckets for Histogram      | Prometheus bucket value [#prometheus_default_bucket]_ |
+----------------------+-------------------------------------+-------------------------------------------------------+
| ``duration.factor``  | factor for exponential buckets      | null                                                  |
+----------------------+-------------------------------------+-------------------------------------------------------+
| ``duration.max``     | upper bound for exponential buckets | null                                                  |
+----------------------+-------------------------------------+-------------------------------------------------------+
| ``duration.min``     | lower bound for exponential buckets | null                                                  |
+----------------------+-------------------------------------+-------------------------------------------------------+

``http``
""""""""

Configuration under ``api_server.http`` will be used to configure the HTTP API server.

By default, BentoML will start an HTTP API server on port 3000. To change the port, set ``api_server.http.port``:

.. code-block:: yaml

   api_server:
     http:
       port: 5000

Users can also configure `CORS <https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS>`_ via ``api_server.http.cors``. By default CORS is disabled.

If specified, all fields under ``api_server.http.cors`` will then be parsed to `CORSMiddleware <https://www.starlette.io/middleware/#corsmiddleware>`_:

.. code-block:: yaml

   api_server:
     http:
       cors:
         enabled: true
         access_control_allow_origins: ["http://myorg.com:8080", "https://myorg.com:8080"]
         access_control_allow_methods: ["GET", "OPTIONS", "POST", "HEAD", "PUT"]
         access_control_allow_credentials: true
         access_control_allow_headers: ["*"]
         access_control_allow_origin_regex: 'https://.*\.my_org\.com'
         access_control_max_age: 1200
         access_control_expose_headers: ["Content-Length"]

.. deprecated:: 1.0.16
   :code:`access_control_allow_origin` is deprecated. Use :code:`access_control_allow_origins` instead.

``grpc``
""""""""

This section will go through configuration that is not yet coverred in :ref:`our guides on performance tuning <guides/grpc:Performance tuning>`.

Similar to HTTP API server, BentoML will start a gRPC API server on port 3000 by default. To change the port, set ``api_server.grpc.port``:

.. code-block:: yaml

   api_server:
     grpc:
       port: 5000

Note that when using :ref:`bentoml serve-grpc <reference/cli:serve-grpc>` and metrics is
enabled, a Prometheus metrics server will be started as a sidecar on port 3001. To change the port, set ``api_server.grpc.metrics.port``:

.. code-block:: yaml

   api_server:
     grpc:
       metrics:
         port: 50051

By default, the gRPC API server will disable reflection. To always enable :github:`server reflection <grpc/grpc/blob/master/doc/server-reflection.md>`,
set ``api_server.grpc.reflection.enabled`` to ``true``:

.. code-block:: yaml

   api_server:
     grpc:
       reflection:
         enabled: true

.. note::

   User can already enable reflection by passing ``--enable-reflection`` to :ref:`bentoml serve-grpc <reference/cli:serve-grpc>` CLI command.

   However, we also provide this option in the config file to make it easier for users who wish to always enable reflection.

``ssl``
"""""""

BentoML supports SSL/TLS for both HTTP and gRPC API server. To enable SSL/TLS, set ``api_server.ssl.enabled`` to ``true``:

.. code-block:: yaml

   api_server:
     ssl:
       enabled: true

When using HTTP API server, BentoML will parse all of the available fields directly to `Uvicorn <https://www.uvicorn.org/settings/#https>`_.

.. TODO::

   - Add instruction how one can setup SSL for gRPC API server.

----

.. rubric:: Notes

.. [#default_workers] The default number of workers is the number of CPUs count.

.. [#default_configuration] The default configuration can also be found under :github:`configuration folder <bentoml/BentoML/tree/main/src/bentoml/_internal/configuration>`.

   .. dropdown:: `Expands for default configuration`
      :icon: code

      .. literalinclude:: ../../../src/bentoml/_internal/configuration/v1/default_configuration.yaml
         :language: yaml

.. [#prometheus_default_bucket] The default buckets is specified `here <https://github.com/prometheus/client_python/blob/f17a8361ad3ed5bc47f193ac03b00911120a8d81/prometheus_client/metrics.py#L544>`_ for Python client.

.. [#runners_configuration] See :ref:`Runners' configuration <concepts/runner:Runner Configuration>`
