================
Preparing Models
================


save_model
load_model

how it works

get, to_runner
testing runner
runner input/output

Model signature
* batchable
* batch_dim

Dynamic batching params
* max_batch_size
* max_latency_ms

Model metadata
Model labels


Managing Models
---------------

Saved models can be managed via the :code:`bentoml models` CLI command. Try
:code:`bentoml models --help`. to learn more.

.. tabbed:: Get

   .. code:: bash

      > bentoml models get iris_clf:latest

      name: iris_clf
      version: 2uo5fkgxj27exuqj
      module: bentoml.sklearn
      labels: {}
      options: {}
      metadata: {}
      context:
        framework_name: sklearn
        framework_versions:
          scikit-learn: 1.1.0
        bentoml_version: 1.0.0
        python_version: 3.8.12
      signatures:
        predict:
          batchable: false
      api_version: v1
      creation_time: '2022-05-19T08:36:52.456990+00:00'

.. tabbed:: List

   .. code:: bash

      > bentoml models list

      Tag                        Module           Size        Creation Time        Path
      iris_clf:2uo5fkgxj27exuqj  bentoml.sklearn  5.81 KiB    2022-05-19 08:36:52  ~/bentoml/models/iris_clf/2uo5fkgxj27exuqj
      iris_clf:nb5vrfgwfgtjruqj  bentoml.sklearn  5.80 KiB    2022-05-17 21:36:27  ~/bentoml/models/iris_clf/nb5vrfgwfgtjruqj


.. tabbed:: Import / Export

   .. code:: bash

      > bentoml models export iris_clf:latest .

      INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") exported to ./iris_clf-2uo5fkgxj27exuqj.bentomodel

   .. code:: bash

      > bentoml models import ./iris_clf-2uo5fkgxj27exuqj.bentomodel

      INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") imported

   .. note::

      Model can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
      example:

      .. code:: bash

         bentoml models export iris_clf:latest s3://my_bucket/my_prefix/

.. tabbed:: Push / Pull

   If your team has `Yatai <https://github.com/bentoml/Yatai>`_ setup, you can also
   push local Models to Yatai, it provides APIs and Web UI for managing all Models
   created by your team and stores model files on cloud blob storage such as AWS S3,
   MinIO or GCS.

   .. code:: bash

      > bentoml models push iris_clf:latest

      Successfully pushed model "iris_clf:2uo5fkgxj27exuqj"                                                                                                                                                                                           │

   .. code:: bash

      > bentoml models pull iris_clf:latest

      Successfully pulled model "iris_clf:2uo5fkgxj27exuqj"

   .. image:: /_static/img/yatai-model-detail.png
     :alt: Yatai Model Details UI

.. tabbed:: Delete

   .. code:: bash

      > bentoml models delete iris_clf:latest -y

      INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") deleted


Besides the CLI commands, BentoML also provides equivalent
:doc:`Python APIs <reference/stores>` for managing models:


.. tabbed:: Get

   :code:`bentoml.models.get` returns a :code:`bentoml.Model` instance, which is a
   reference to a saved model entry in the mdoel store. The :code:`bentoml.Model`
   instances provides access to the model info and the :code:`to_runner` API for
   creating Runner instance:

   .. code:: python

      import bentoml
      bento_model: bentoml.Model = bentoml.models.get("iris_clf:latest")

      print(bento_model.path)
      print(bento_model.info.metadata)
      print(bento_model.info.labels)

      my_runner = bento_model.to_runner()

.. tabbed:: List

   :code:`bentoml.models.list` returns a list of

   .. code:: python

      import bentoml
      models = bentoml.models.list()

.. tabbed:: Import / Export

   .. code-block:: bash

      > bentoml models export iris_clf:latest .

      INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") exported to ./iris_clf-2uo5fkgxj27exuqj.bentomodel

   .. code-block:: bash

      > bentoml models import ./iris_clf-2uo5fkgxj27exuqj.bentomodel

      INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") imported

   .. note::

      Model can be exported to or import from AWS S3, GCS, FTP, Dropbox, etc. For
      example:

      .. code-block:: bash

         bentoml models export iris_clf:latest s3://my_bucket/my_prefix/

.. tabbed:: Push / Pull

   If your team has `Yatai <https://github.com/bentoml/Yatai>`_ setup, you can also
   push local Models to Yatai, it provides APIs and Web UI for managing all Models
   created by your team and stores model files on cloud blob storage such as AWS S3,
   MinIO or GCS.

   .. code-block:: bash

      > bentoml models push iris_clf:latest

      Successfully pushed model "iris_clf:2uo5fkgxj27exuqj"                                                                                                                                                                                           │

   .. code-block:: bash

      > bentoml models pull iris_clf:latest

      Successfully pulled model "iris_clf:2uo5fkgxj27exuqj"

   .. image:: _static/img/yatai-model-detail.png
     :alt: Yatai Model Details UI

.. tabbed:: Delete

   .. code-block:: bash

      > bentoml models delete iris_clf:latest -y

      INFO [cli] Model(tag="iris_clf:2uo5fkgxj27exuqj") deleted