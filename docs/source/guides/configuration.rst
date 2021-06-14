.. _configuration-page:

Configuration
=============

BentoML starts with an out-of-the-box configuration that works for common use cases. For advanced users, many
features can be customized through configuration. Both BentoML CLI and Python APIs can be customized 
by the configuration. Configuration is best used for scenarios where the customizations can be specified once 
and applied to the entire team.

.. note::
    BentoML configuration underwent a major redesign in release 0.13.0. Previous configuration set through the 
    `bentoml config` CLI command is no longer compatible with the configuration releases in 0.13.0 and above. 
    Please see legacy configuration property mapping table below to upgrade configuration to the new format.

BentoML configuration is defined by a YAML file placed in a directory specified by the `BENTOML_CONFIG` 
environment variable. By default, configurations defined in `$BENTOML_HOME/bentoml.yml` is applied if present. 
The example below starts the bento server with configuration defined in `~/bentoml_configuration.yml`

.. code-block:: shell

    $ BENTOML_CONFIG=~/bentoml_configuration.yml bentoml serve-gunicorn IrisClassifier:latest

Users only need to specify a partial configuration with only the properties they wish to customize instead 
of a full configuration schema. In the example below, the microbatching workers count is overridden to 4. 
Remaining properties will take their defaults values.

.. code-block:: yaml
    :caption: BentoML Configuration

    bento_server:
        microbatch:
            workers: 4

Throughout the BentoML documentation, features that are customizable through configuration are demonstrated 
like the example above. For a full configuration schema including all customizable properties, refer to 
the BentoML configuration template defined in 
`default_configuration.yml <https://github.com/bentoml/BentoML/blob/master/bentoml/configuration/default_configuration.yml>`_. 

Docker Deployment
-----------------

Configuration file can be mounted to the Docker container using the `-v` option and specified to the BentoML 
runtime using the `-e` environment variable option.

.. code-block:: shell

    $ docker run -v /local/path/configuration.yml:/home/bentoml/configuration.yml -e BENTOML_CONFIG=/home/bentoml/configuration.yml

Configuration Priority
----------------------

Some customizable properties in the configuration can also be specified in the BentoML CLI or Python API 
parameters. Values specified through BentoML CLI and Python API parameters will always take precedence over 
the values defined in the configuration.

Legacy Property Mapping
-----------------------

Starting BentoML release `0.13.0`, the legacy `bentoml.cfg` based configuration is deprecated and no longer 
compatible with the YAML based configuration system. Please refer to the mapping below to migrate to the 
YAML based configuration.

+------------------------------------------------------------+-----------------------------------------+
| CFG Properties                                             | YAML Properties                         |
+---------------+--------------------------------------------+-----------------------------------------+
| Section       | Key                                        | Key                                     |
+---------------+--------------------------------------------+-----------------------------------------+
| core          | bentoml_deploy_version                     | bento_bundle.deployment_version         |
+---------------+--------------------------------------------+-----------------------------------------+
| core          | default_docker_base_image                  | bento_bundle.default_docker_base_image  |
+---------------+--------------------------------------------+-----------------------------------------+
| instrument    | default_namespace                          | bento_server.metrics.namespace          |
+---------------+--------------------------------------------+-----------------------------------------+
| instrument    | prometheus_multiproc_dir                   | DEPRECATED                              |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | logging_config                             | See Logging guide                       |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | console_logging_enabled                    | logging.console.enabled                 |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | file_logging_enabled                       | logging.file.enabled                    |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | level                                      | logging.level                           |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | log_format                                 | DEPRECATED                              |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | dev_log_format                             | DEPRECATED                              |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | base_log_dir                               | logging.file.directory                  |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | prediction_log_filename                    | DEPRECATED                              |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | feedback_log_filename                      | DEPRECATED                              |
+---------------+--------------------------------------------+-----------------------------------------+
| logging       | yatai_web_server_log_filename              | yatai.logging.path                      |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | url                                        | yatai.remote.url                        |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | s3_signature_version                       | yatai.repository.s3.signature_version   |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | repository_base_url                        | See Repository Base URL section         |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | db_url                                     | yatai.database.url                      |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | s3_endpoint_url                            | yatai.repository.s3.endpoint_url        |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | default_namespace                          | yatai.namespace                         |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | tls_root_ca_cert                           | yatai.remote.tls.root_ca_cert           |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | tls_client_key                             | yatai.remote.tls.client_key             |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | tls_client_cert                            | yatai.remote.tls.client_cert            |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | access_token                               | yatai.remote.access_token               |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai_service | access_token_header                        | yatai.remote.access_token_header        |
+---------------+--------------------------------------------+-----------------------------------------+
| apiserver     | default_port                               | bento_server.port                       |
+---------------+--------------------------------------------+-----------------------------------------+
| apiserver     | enable_metrics                             | bento_server.metrics.enabled            |
+---------------+--------------------------------------------+-----------------------------------------+
| apiserver     | enable_feedback                            | bento_server.feedback.enabled           |
+---------------+--------------------------------------------+-----------------------------------------+
| apiserver     | default_timeout                            | bento_server.timeout                    |
+---------------+--------------------------------------------+-----------------------------------------+
| apiserver     | default_max_request_size                   | bento_server.max_request_size           |
+---------------+--------------------------------------------+-----------------------------------------+
| apiserver     | default_image_input_accept_file_extensions | adapters.image_input.default_extensions |
+---------------+--------------------------------------------+-----------------------------------------+
| apiserver     | default_gunicorn_workers_count             | bento_server.workers                    |
+---------------+--------------------------------------------+-----------------------------------------+
| yatai         | bento_uri_default_expiration               | yatai.repository.s3.expiration OR       |
|               |                                            | yatai.repository.gcs.expiration         |
+---------------+--------------------------------------------+-----------------------------------------+

Repository Base URL
^^^^^^^^^^^^^^^^^^^

The repository base URL property has been broken down into properties for the individual repository 
implementations, instead of being derived automatically.

For file system, what was previously specified as `/user/home/bentoml/repository` should defined as 
the following in YAML.

.. code-block:: yaml
    :caption: BentoML Configuration

    yatai:
        repository:
            type: file_system
            file_system:
                directory: /user/home/bentoml/repository

For S3 or GCS, what was previously specified as `s3://s3_address` should defined as the following in 
YAML.

.. code-block:: yaml
    :caption: BentoML Configuration

    yatai:
        repository:
            type: s3
            s3:
                url: s3://s3_address


.. spelling::

    customizations
    microbatching
    customizable
    multiproc
    dir
    tls
    apiserver
    gunicorn
    uri
    gcs
