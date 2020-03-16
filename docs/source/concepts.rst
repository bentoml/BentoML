Core Concepts
=============

The main idea behind BentoML is that Data Science team should be shipping prediction
services instead of shipping pickled models.
:ref:`bentoml.BentoService <bentoml-bentoservice-label>` is the base component for
building prediction services using BentoML. Here's the example BentoService defined in
the :doc:`Getting Started Guide <quickstart>`:

.. code-block:: python

  import bentoml
  from bentoml.handlers import DataframeHandler
  from bentoml.artifact import SklearnModelArtifact

  @bentoml.env(auto_pip_dependencies=True)
  @bentoml.artifacts([SklearnModelArtifact('model')])
  class IrisClassifier(bentoml.BentoService):

      @bentoml.api(DataframeHandler)
      def predict(self, df):
          return self.artifacts.model.predict(df)


Each BentoService can contain multiple models via the BentoML :code:`Artifact` class,
and can define multiple APIs for accessing this service. Each API should specify a type
of :code:`Handler`, which defines the expected input data format for this API, most
commonly we see the use of :code:`DataframeHandler`, :code:`TensorHandler`, and
:code:`JsonHandler`.


Once you've trained an ML model, you can use the :code:`BentoService#pack` method to
bundle it with a BentoService instance, and save the BentoService to a file directory.
In the process of :code:`BentoService#save`, BentoML serializes the model based on the
ML training framework you're using, automatically extracts all the pip dependencies
required by your BentoService class, and saves all the code, serialized model files,
and requirements.txt etc into a file directory, which we call it a SavedBundle.


.. code-block:: python

  from sklearn import svm
  from sklearn import datasets

  clf = svm.SVC(gamma='scale')
  iris = datasets.load_iris()
  X, y = iris.data, iris.target
  clf.fit(X, y)

  # Create a iris classifier service with the newly trained model
  iris_classifier_service = IrisClassifier()
  iris_classifier_service.pack("model", clf)

  # Save the entire prediction service to file bundle
  saved_path = iris_classifier_service.save()

A BentoService SavedBundle is a versioned file directory that contains all the
information needed to run this prediction service. Think of it as a docker container
image or a software binary for your ML model, which is generated in every training job,
reflecting your newest code changes and training data, and at the end of the day, makes
your model more easily testable and ready to be shipped to production.

BentoML also makes it easy to do model management. It keeping track of all the
BentoService SavedBundle you've created and provide web UI and CLI access. By default
BentoML saves all the model files and metadata in your local file system. But it is
recommended to run a shared BentoML server for your team which stores models files and
metadata in the cloud(e.g. RDS, S3). This allows your ML team to easily share, find and
use each others' models and model serving endpoints. :doc:`Read more about this
<guides/yatai_service>`

.. code-block:: bash

    > bentoml list
    BENTO_SERVICE                         CREATED_AT        APIS                       ARTIFACTS
    IrisClassifier:20200121114004_360ECB  2020-01-21 19:40  predict<DataframeHandler>  model<SklearnModelArtifact>
    IrisClassifier:20200120082658_4169CF  2020-01-20 16:27  predict<DataframeHandler>  clf<PickleArtifact>
    ...


Creating BentoService
---------------------

Users build their prediction service by subclassing
:ref:`bentoml.BentoService <bentoml-bentoservice-label>`. It is recommended to always
put the source code of your BentoService class into individual Python file and check it
into source control(e.g. git) along with your model training code.

BentoML is designed to be easily inserted to the end of your model training workflow,
where you can import your BentoService class and create a BentoService saved bundle.
This makes it easy to manage, test and deploy all the models you and your team have
created overtime.

.. note::

    The BentoService class can not be defined in the :code:`__main__` module, meaning
    the class itself should not be defined in a Jupyter notebook cell or a python
    interactive shell. You can however use the :code:`%writefile` magic command in
    jupyter notebook to write the BentoService class definition to a separate file, see
    example in `BentoML quickstart notebook <https://github.com/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb>`_.


BentoML only allow users to create prediction service in Python but you can use models
trained with other languages/frameworks with BentoML and benefit from BentoML's model
mangement and performance optimiziation such as micro batching in online serving. To do
so, you will need to :doc:`create custom artifact <guides/custom_artifact>`.


Defining Service Environment
----------------------------

The :ref:`bentoml.env <bentoml-env-label>` decorator is the API for defining the
environment settings and dependencies of your prediction service. And here are the types
of dependencies supported by BentoML

PyPI Packages
^^^^^^^^^^^^^

Python PyPI package is the most common type of dependencies. BentoML provides a
mechanism that automatically figures out the PyPI packages required by your BentoService
python class, simply use the :code:`auto_pip_dependencies=True` option.

.. code-block:: python

  @bentoml.env(auto_pip_dependencies=True)
  class ExamplePredictionService(bentoml.BentoService):

      @bentoml.api(DataframeHandler)
      def predict(self, df):
          return self.artifacts.model.predict(df)

If you have specific versions of PyPI packages required for model serving that are
different from your training environment, or the :code:`auto_pip_dependencies=True`
option does not work for your case(bug report highly appreciated), you can also specify
the list of PyPI packages manually, e.g.:

.. code-block:: python

  @bentoml.env(
    pip_dependencies=['scikit-learn']
  )
  class ExamplePredictionService(bentoml.BentoService):

      @bentoml.api(DataframeHandler)
      def predict(self, df):
          return self.artifacts.model.predict(df)


Similarly, if you already have a list of PyPI packages required for model serving in a
:code:`requirements.txt` file, then simply pass in the file path via
:code:`@bentoml.env(requirements_txt_file='./requirements.txt')`.


Conda Packages
^^^^^^^^^^^^^^

Conda packages can be specified in a similar way, here's an example prediction service
relying on an H2O model that requires the h2o conda packages:

.. code-block:: python

    @bentoml.artifacts([H2oModelArtifact('model')])
    @bentoml.env(
      pip_dependencies=['pandas', 'h2o==3.24.0.2'],
      conda_channels=['h2oai'],
      conda_dependencies=['h2o==3.24.0.2']
    )
    class ExamplePredictionService(bentoml.BentoService):

      @bentoml.api(DataframeHandler)
      def predict(self, df):
          return self.artifacts.model.predict(df)


.. note::
    One caveat with Conda Packages is that it does not work with AWS Lambda deployment
    due to the limitation of AWS Lambda platform.


Initial Setup Bash Script
^^^^^^^^^^^^^^^^^^^^^^^^^

Setup script is a way for customizing the API serving docker container. It allow user
(and trusted the user) to insert arbitary bash script into the docker build process
to install extra system dependencies or do other setups required by the prediction
service.

.. code-block:: python

  @bentoml.env(
    auto_pip_dependencies=True,
    setup_sh="./my_init_script.sh"
  )
  class ExamplePredictionService(bentoml.BentoService):
    ...

  @bentoml.env(
    auto_pip_dependencies=True,
    setup_sh="""\n
  #!/bin/bash
  set -e

  apt-get install --no-install-recommends nvidia-driver-430
  ...
    """
  )
  class ExamplePredictionService(bentoml.BentoService):
    ...

If you have a specific docker base image that you would like to use for your API server,
`contact us <mailto:contact@bentoml.ai>`_ and let us know your use case and
requirements there, as we are planning to build custom docker base image support.


Packaging Model Artifacts
-------------------------

BentoML's model artifact API allow users to specify the trained models required by a
BentoService. BentoML automatically handles model serialization and deserialization when
saving and loading a BentoService.

Thus BentoML asks the user to choose the right Artifact class for the machine learning
framework they are using. BentoML has built-in artifact class for most popular ML
frameworks and you can find the list of supported frameworks
:doc:`here <api/artifacts>`. If the ML framework you're using is not in the list,
`let us know <mailto:contact@bentoml.ai>`_  and we will consider adding its support.

To specify the model artifacts required by your BentoService, use the
:code:`bentoml.artifacts` decorator and gives it a list of artifact types. And give
each model artifact a unique name within the prediction service. Here's an example
prediction service that packs two trained models:

.. code-block:: python

    import bentoml
    from bentoml.handlers import DataframeHandler
    from bentoml.artifact import SklearnModelArtifact, XgboostModelArtifact

    @bentoml.env(auto_pip_dependencies=True)
    @artifacts([
        SklearnModelArtifact("model_a"),
        XgboostModelArtifact("model_b")
    ])
    class MyPredictionService(bentoml.BentoService):

      @bentoml.api(DataframeHandler)
      def predict(self, df):
        # assume the output of model_a will be the input of model_b in this example:
        df = self.artifacts.model_a.predict(df)

        return self.artifacts.model_b.predict(df)


.. code-block:: python

    svc = MyPredictionService()
    svc.pack('model_a', my_sklearn_model_object)
    svc.pack('model_b', my_xgboost_model_object)
    svc.save()

For most model serving scenarios, we recommend one model per prediction service, and
decouple non-related models into separate services. The only exception is when multiple
models are depending on each other, such as the example above.


API Function and Handlers
-------------------------






Saving BentoService
-------------------

Using BentoService SavedBundle
------------------------------


Model Management
----------------


Deploying BentoService
----------------------



