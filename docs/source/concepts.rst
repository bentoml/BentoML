Core Concepts
=============


Introducing BentoService
------------------------

The main idea behind BentoML is that Data Science team should be shipping prediction
services instead of shipping pickled models. BentoService is the base component for
building prediction services using BentoML. Here's the example BentoService in the
:doc:`Getting Started Guide <quickstart>`:

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


.. note::

    The BentoService class can not be defined in the :code:`__main__` module, meaning
    the class itself should not be defined in a Jupyter notebook cell or a python
    interactive shell. You can however use the :code:`%writefile` magic command in
    jupyter notebook to write the BentoService class definition to a separate file, see
    example in `BentoML quickstart notebook <https://github.com/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb>`_.

    We recommend to always put the code of every BentoService defined into a separate
    Python file and check it into source control(e.g. git) along with your model
    training code. And at the end of your model training code, import the BentoService
    class and create a BentoService saved bundle for each training job run. e.g.:



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


BentoML also provide a mechanism for easy model management. It keeping track of all the
BentoService SavedBundle you've created and provide web UI and CLI access. By default
BentoML keeps track of all the model files ad metadata in your local file system. But
it is also possible to run a BentoML server that stores those data in the cloud(e.g.
RDS, S3), and allow your ML team to easily share, find and use each others' models.

.. code-block:: bash

    > bentoml list
    BENTO_SERVICE                         CREATED_AT        APIS                       ARTIFACTS
    IrisClassifier:20200121114004_360ECB  2020-01-21 19:40  predict<DataframeHandler>  model<SklearnModelArtifact>
    IrisClassifier:20200120082658_4169CF  2020-01-20 16:27  predict<DataframeHandler>  clf<PickleArtifact>
    ...


Packaging Model Artifacts
-------------------------


Using API Handlers
------------------


Using SavedBundle
-----------------


Model Management
----------------


Deploying BentoService
----------------------



