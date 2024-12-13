==========
Monitoring
==========

Data-centric AI is a paradigm that positions data as the cornerstone of AI systems. This approach emphasizes the importance of data quality and relevance, suggesting that the effectiveness of AI models can be primarily determined by the data they are trained on and interact with.

BentoML fully embraces this paradigm by offering APIs to implement a data-centric workflow, making it straightforward to collect inference data, monitor models, and ship the data to various destinations (for example, local storage, cloud services, and any OTLP supported tools) for your AI project.

This document explains how to implement monitoring and collect inference data in BentoML.

Benefits of model monitoring and data collection
------------------------------------------------

Implementing an effective data collection and model monitoring strategy with BentoML offers the following benefits:

- **Key business metric monitoring:** Track crucial statistical metrics to gauge the impact of your models on business objectives.
- **Early detection of data drift:** Identify shifts in input data distributions early, helping decide when model retraining is necessary.
- **Quality assurance:** Implement quality assurance for previously untracked metrics, such as model performance, accuracy, and degradation over time.
- **Enhanced interoperability:** Promote collaboration between data science and operations teams, streamlining the process of training and iterating on models.
- **Informed decision-making:** Provide insights into model behavior in production, informing future model improvements and iterations.

Implement monitoring
--------------------

In BentoML, you use the ``bentoml.monitor`` context manager to log data related to model inference. It allows you to specify a monitoring session where you can log various data types. This ensures that logging is structured and organized, making it easier to analyze the data later.

The following is an example of implementing monitoring in the Summarization Service in :doc:`/get-started/hello-world`.

.. code-block:: python

    from __future__ import annotations
    import bentoml
    from transformers import pipeline

    EXAMPLE_INPUT = "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century.'"

    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
    )
    class Summarization:
        def __init__(self) -> None:
            self.pipeline = pipeline('summarization')

        @bentoml.api
        def summarize(self, text: str = EXAMPLE_INPUT) -> str:
            # Use bentoml.monitor as a context manager
            with bentoml.monitor("text_summarization") as mon:
                # Log the input data
                mon.log(text, name="input_text", role="original_text", data_type="text")

                result = self.pipeline(text)
                summary_text = result[0]['summary_text']

                # Log the output data
                mon.log(summary_text, name="summarized_text", role="prediction", data_type="text")

                return summary_text

When you enter the ``bentoml.monitor`` context, you instantiate a monitoring session uniquely identified by a name (``"text_summarization"`` in this example), which helps you categorize and retrieve the logged data for specific monitoring tasks.

Within the ``bentoml.monitor`` context, you use the ``log`` method to record individual data points, which requires several parameters to describe the data being logged:

- ``data``: The actual data value you want to log. This could be an input/output parameter value (the input ``text`` and output ``summary_text`` in this example).
- ``name``: A string identifier for the data point, which will be displayed in logs. This helps label the data in your logs, making it clear what each logged value represents.
- ``role``: The role of the data in model inference. Default roles include:

  - ``"feature"``: Indicates that the logged data is an input feature of the model.
  - ``"prediction"``: Indicates that the logged data is a prediction made by the model.
  - ``"target"``: Indicates that the logged data is the target or label.

  You can set a custom ``role`` like ``original_text`` in the example, which will also be logged.

- ``data_type``: The type of the data. Default data types include:

  - ``"numerical"``: For quantitative data.
  - ``"categorical"``: For discrete values representing categories.
  - ``"numerical_sequence"``: For sequences or lists of numerical values.

  You can set a custom ``data_type`` like ``text`` in the example, which will also be logged.

View request and schema logs
----------------------------

When the Service starts, BentoML exports the request and schema logs to the default directory ``monitoring/<your_monitor_name>``, which contains the ``data`` and ``schema`` subdirectories.

The input and output data is stored in the ``data`` directory, including the corresponding timestamp and a unique request ID. To view the real-time data logs, run:

.. code-block:: bash

    $ tail -f monitoring/text_summarization/data/*.log

    {"input_text": "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century.'", "summarized_text": " Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly . The event is now being investigated by scientists for potential breaches in the laws of physics . Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century'", "timestamp": "2024-03-05T03:33:59.490137", "request_id": "14642743634293743168"}
    {"input_text": "Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century.'", "summarized_text": " Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly . The event is now being investigated by scientists for potential breaches in the laws of physics . Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century'", "timestamp": "2024-03-05T03:41:49.870589", "request_id": "7485759375304577245"}

The schema information is stored in the ``schema`` directory.

.. code-block:: bash

    $ cat monitoring/text_summarization/schema/*.log

    {"meta_data": {"bento_name": "", "bento_version": "not available"}, "columns": [{"name": "input_text", "role": "original_text", "type": "text"}, {"name": "summarized_text", "role": "prediction", "type": "text"}]}

BentoML logs request and schema data to rotating files. This means that new log files are created periodically or when the current log file reaches a certain size, older files are archived based on the default retention policy. You can customize the behavior by :ref:`using a configuration file <through-log-files>`.

The actual data points are logged as JSON objects, providing a structured format for storing multiple records. This format is widely supported and can be easily ingested into various data analysis tools or databases for further processing.

Ship the collected data
-----------------------

BentoML provides a general monitoring data collection API. It allows you to transmit collected data to various destinations such as data warehouses, analytics pipelines, or specialized monitoring and drift detection solutions, all without requiring any modifications to your existing codebase.

The following table outlines the available targets for shipping monitoring data, the monitoring types (read the following sections for details), and additional notes.

.. list-table::
    :widths: 33 33 34
    :header-rows: 1

    - - Destination
      - Monitoring type
      - Note
    - - ``./monitoring/<name>/data/xxx.log``
      - ``default``
      - Logs are stored locally by default.
    - - Cloud and monitoring services (Amazon S3, Azure Blob, Datadog, Elasticsearch, InfluxDB, Google BigQuery, Kafka, etc.)
      - ``otlp`` + deployed Fluent Bit
      - For more output options and configurations, see `Fluent Bit Outputs <https://docs.fluentbit.io/manual/pipeline/outputs>`_.
    - - Any OTLP supported tools
      - ``otlp``
      - Useful for environments where direct file access is restricted, like AWS Lambda.
    - - Arize
      - ``bentoml_plugins.arize.ArizeMonitor``
      - Ensure API keys and space keys are configured correctly.


.. _through-log-files:

Through log files
^^^^^^^^^^^^^^^^^

Writing monitoring data to log files is the most common way of data collection in BentoML, which is compatible with popular logging tools such as `Fluent Bit <https://fluentbit.io/>`_, `Filebeat <https://www.elastic.co/beats/filebeat>`_, and `Logstash <https://www.elastic.co/logstash/>`_. You can customize the monitoring configuration using the ``@bentoml.service`` decorator.

.. code-block:: python

    ...
    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
        monitoring={
            "enabled": True,
            "type": "default",
            "options": {
                "log_config_file": "path/to/log_config.yaml",
                "log_path": "monitoring"
            }
        }
    )
    class Summarization:
        # Service implementation code

Available fields for ``monitoring``:

- ``enabled``:  Whether monitoring is enabled for the Service. Setting it to ``True`` allows BentoML to collect and log data based on the specified configurations.
- ``type``: Specifies the type of monitoring system to use. The value ``default`` means the use of BentoML's built-in monitoring system, which collects data and logs it to files as shown in the previous section.
- ``options``: A dictionary that allows you to customize the monitoring setup.

  - ``log_config_file``: Specifies the path to a custom logging configuration file in YAML, which specifies logging behavior, such as log rotation policies, handlers, log formats, and log levels. The logging parameters should be set according to `the Python logging module's configuration schema <https://docs.python.org/3/library/logging.html>`_. If not provided, BentoML uses the default logging configuration, which are suitable for most use cases.

    Here is an example configuration file, which outputs log messages to a stream:

    .. code-block:: yaml

        version: 1
        disable_existing_loggers: false
        loggers:
          bentoml_monitor_data:
            level: INFO
            handlers: [bentoml_monitor_data]
            propagate: false
          bentoml_monitor_schema:
            level: INFO
            handlers: [bentoml_monitor_schema]
            propagate: false
        handlers:
          bentoml_monitor_data:
            class: logging.StreamHandler
            stream: "ext://sys.stdout"
            level: INFO
            formatter: bentoml_json
          bentoml_monitor_schema:
            class: logging.StreamHandler
            stream: "ext://sys.stdout"
            level: INFO
            formatter: bentoml_json
        formatters:
          bentoml_json:
            class: pythonjsonlogger.jsonlogger.JsonFormatter
            format: "()"
            validate: false


  - ``log_path``: Defines the directory where monitoring logs will be stored, which is relative to the Service's running location. It defaults to ``monitoring``.

For deployments using :doc:`the OCI-compliant image </get-started/packaging-for-deployment>`, you can persist log files by mounting the specified log directory (``monitoring`` in the example) to a volume. This ensures that your monitoring data is retained across container restarts and redeployments.

In Kubernetes, you can persist and ship logs by mounting the log directory and using a `Fluent Bit <https://fluentbit.io/>`_ DaemonSet or a sidecar container. This allows the collected log files to be automatically forwarded to your designated monitoring system or data warehouse, ensuring that your monitoring data is centralized and accessible for analysis and alerting.

Through an OTLP endpoint
^^^^^^^^^^^^^^^^^^^^^^^^

In scenarios where you can't directly access log files, such as when using AWS Lambda since it doesn't support log files, BentoML supports exporting monitoring data to an external telemetry system using the OpenTelemetry Protocol (OTLP).

.. note::

    Some log collectors like Fluent Bit also support OTLP input.

Below is an example of setting up OTLP for a BentoML Service:

.. code-block:: python

    ...
    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
        monitoring={
          "enabled": True,
          "type": "otlp",
          "options": {
            "endpoint": "http://localhost:5000",
            "insecure": True,
            "credentials": null,
            "headers": null,
            "timeout": 10,
            "compression": null,
            "meta_sample_rate": 1.0
          }
        }
    )
    class Summarization:
        # Service implementation code

Available parameters:

- ``endpoint``: Specifies the URL of the telemetry system's OTLP receiver. Data collected by BentoML will be sent to this endpoint.
- ``insecure``: A Boolean flag that specifies whether to disable transport security for the connection with the OTLP endpoint. Setting this to ``True`` means an insecure connection, which is typical for local or development environments.
- ``credentials``: If your OTLP endpoint requires authentication, you can use this parameter to provide credentials such as tokens or certificates. If set to ``null``, it means that no authentication is required.
- ``headers``: Additional headers that may be required by the OTLP endpoint, useful for passing tokens or other necessary information.
- ``timeout``: Defines the maximum duration (in seconds) that BentoML will wait for a response from the OTLP endpoint before timing out.
- ``compression``: Specifies the type of compression to use when sending data. This can help reduce bandwidth usage. Supported values include ``gzip`` or ``none``.
- ``meta_sample_rate``: Determines the sampling rate for sending metadata to the endpoint. A value of ``1.0`` means that all metadata is sent, while lower values reduce the frequency, sending only a percentage of the collected metadata.

For more information, see `the OTLP documentation <https://opentelemetry.io/docs/specs/otel/protocol/exporter/>`_.

Plugins and third-party monitoring data collectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BentoML also supports plugins and third-party monitoring data collectors. You can create a custom monitoring data collector and publish it as a Python package. Unlike the built-in collector, which is more protocol specific for general use cases, plugins could be more platform-specific.

To use a plugin, you need to install it and include it in ``bentofile.yaml``. For details, see :doc:`/reference/bentoml/bento-build-options`.

Arize AI
""""""""

For end-to-end solutions for data and model monitoring, BentoML works with `Arize AI <https://arize.com/docs/>`_ to provide a plugin for Arize. If you don't want to deploy a pipeline by yourself but still need data and model monitoring, Arize AI is a good choice. It provides a unified platform for data scientists, data engineers, and ML engineers to monitor, analyze, and debug ML models in production.

To use this plugin, make sure you have installed it first:

.. code-block:: bash

    pip install bentoml-plugins-arize

In the ``@bentoml.service`` decorator, add the ``space_key`` and ``api_key`` to connect to your Arize account.

.. code-block:: python

    ...
    @bentoml.service(
        resources={"cpu": "2"},
        traffic={"timeout": 10},
        monitoring={
          "enabled": True,
          "type": "bentoml_plugins.arize.ArizeMonitor",
          "options": {
            "space_key": <your_space_key>,
            "api_key": <your_api_key>
          }
        }
    )
    class Summarization:
      # Service implementation code

For more information about available Arize parameters, see `the Arize documentation <https://docs.arize.com/arize/api-reference/python-sdk/arize.log>`_.

The plugin should also be added in ``bentofile.yaml``:

.. code-block:: yaml

    service: "service:Summarization"
    python:
      packages:
        - bentoml-plugins-arize  # Add this plugin
