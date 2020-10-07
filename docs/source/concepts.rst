.. _core-concepts-page:

Core Concepts
=============

.. image:: https://static.scarf.sh/a.png?x-pxid=0beb35eb-7742-4dfb-b183-2228e8caf04c

The main idea of BentoML is that the Data Science team should be able to ship their 
models in a way that is easy to test, easy to deploy, and easy to integrate with.
And to do so, Data Scientists need tools that help them build and ship prediction
services, instead of uploading pickled model files or Protobuf files to a server and
hoping things work out.

:ref:`bentoml.BentoService <bentoml-bentoservice-label>` is the base class for building
such prediction services using BentoML. And here's the minimal BentoService example from
the :doc:`Getting Started Guide <quickstart>`:

.. code-block:: python

  import bentoml
  from bentoml.adapters import DataframeInput
  from bentoml.frameworks.sklearn import SklearnModelArtifact

  @bentoml.env(infer_pip_packages=True)
  @bentoml.artifacts([SklearnModelArtifact('model')])
  class IrisClassifier(bentoml.BentoService):

      @bentoml.api(input=DataframeInput(), batch=True)
      def predict(self, df):
          return self.artifacts.model.predict(df)


Each BentoService class can contain multiple models declared through the
:code:`@bentoml.artifact` API, and multiple APIs for accessing this service. Each API
definition requires a :code:`InputAdapter` type, which defines the expected input data
format of this API. BentoML provides API input adapters that covers most model serving
use cases including :code:`DataframeInput`, :code:`TfTensorInput`, :code:`ImageInput`
and :code:`JsonInput`.


Once an ML model is trained, a BentoService instance can bundle with the trained model
with the :ref:`BentoService#pack <bentoml-bentoservice-pack-label>` method. This trained
model is then accessible within the API function code via 
:code:`self.artifacts.ARTIFACT_NAME`. In the example above, the artifact is initialized
with the name ``"model"``, so the user code can get access to the model via 
:code:`self.artifacts.model`.

The BentoService instance is now ready to be used for
inference. But more importantly, BentoML solves the problem of saving the entire
BentoService to disk, distribute the saved file, and reproduce the exact same prediction
service in testing and production environment.

To save the BentoService instance, simply call the
:ref:`BentoService#save <bentoml-bentoservice-save-label>` method. In this process, 
BentoML will:

#. Saves the model based on the ML training framework and artifact type used
#. Automatically extracts all the pip dependencies required by your BentoService class and put into a `requirements.txt` file
#. Saves all the local python code dependencies
#. Put all the generated files into one file directory, which, by default, is a location managed by BentoML


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

  # Test invoking BentoService instance
  iris_classifier_service.predict([[5.1, 3.5, 1.4, 0.2]])

  # Save the entire prediction service to file bundle
  saved_path = iris_classifier_service.save()


The saved BentoService file directory is called a Bento. It is a versioned file
directory that contains all the information required to run this prediction service.

Think of Bento as a docker container image or a software binary, but for an ML model. A
Bento can be  generated at the end of every training job, reflecting your newest code
changes and training data changes. You can then easily store and distribute the Bento
file, test the prediction service, and then update it to production model serving
endpoint.

BentoML keeps track of all the services saved and provides web UI and CLI commands for
model management. By default, BentoML saves all the model files and metadata in the
local file system. For team settings, it is recommended to run a shared BentoML server 
for the entire team, which stores all of their Bento files and metadata in the cloud
(e.g. RDS, S3). This allows your ML team to easily share, find and use each others' 
models and model serving endpoints. 
:doc:`Read more about it here <guides/yatai_service>`.

BentoML CLI Listing recent Bento:

.. code-block:: bash

    > bentoml list
    BENTO_SERVICE                         CREATED_AT        APIS                       ARTIFACTS
    IrisClassifier:20200121114004_360ECB  2020-01-21 19:40  predict<DataframeInput>  model<SklearnModelArtifact>
    IrisClassifier:20200120082658_4169CF  2020-01-20 16:27  predict<DataframeInput>  clf<PickleArtifact>
    ...


BentoML model management web UI:

.. image:: _static/img/yatai-service-web-ui-repository.png
    :alt: BentoML YataiService Bento Repository Page

.. image:: _static/img/yatai-service-web-ui-repository-detail.png
    :alt: BentoML YataiService Bento Details Page

Creating BentoService
---------------------

Users build their prediction service by subclassing
:ref:`bentoml.BentoService <bentoml-bentoservice-label>`. It is recommended to always
put the source code of your BentoService class into an individual Python file and check it
into source control(e.g. git) along with your model training code.

BentoML is designed to be easily inserted to the end of your model training workflow,
where you can import your BentoService class and create a Bento(a saved BentoService
bundled with trained model artifacts). This makes it easy to manage, test and deploy all
the models you and your team have created overtime.

.. note::

    The BentoService class can not be defined in the :code:`__main__` module, meaning
    the class itself should not be defined in a Jupyter notebook cell or a python
    interactive shell. You can however use the :code:`%writefile` magic command in
    Jupyter notebook to write the BentoService class definition to a separate file, see
    example in `BentoML quickstart notebook <https://github.com/bentoml/BentoML/blob/master/guides/quick-start/bentoml-quick-start-guide.ipynb>`_.


BentoService can only be created using Python as the programming language. But it is
possible to use models trained with other languages/frameworks with BentoML and benefit
from BentoML's model management, API server, dockerization and performance 
optimizations. To do so, you will need to :doc:`create custom artifact <guides/custom_artifact>`.
Support for R and Spark MLlib models are on our roadmap.


Defining Service Environment
----------------------------

The :ref:`bentoml.env <bentoml-env-label>` decorator is the API for defining the
environment settings and dependencies of your prediction service. And here are the types
of dependencies supported by BentoML:

PyPI Packages
^^^^^^^^^^^^^

Python PyPI package is the most common type of dependencies. BentoML provides a 
mechanism that automatically figures out the PyPI packages required by your BentoService
python class, simply use the :code:`infer_pip_packages=True` option.

.. code-block:: python

  @bentoml.env(infer_pip_packages=True)
  class ExamplePredictionService(bentoml.BentoService):

      @bentoml.api(input=DataframeInput(), batch=True)
      def predict(self, df):
          return self.artifacts.model.predict(df)

If you had specific versions of PyPI packages required for model serving that are
different from your training environment, or if the :code:`infer_pip_packages=True`
option did not work for your case(bug report highly appreciated), you can also specify
the list of PyPI packages manually, e.g.:

.. code-block:: python

  @bentoml.env(
    pip_packages=['scikit-learn']
  )
  class ExamplePredictionService(bentoml.BentoService):

      @bentoml.api(input=DataframeInput(), batch=True)
      def predict(self, df):
          return self.artifacts.model.predict(df)


Similarly, if you already have a list of PyPI packages required for model serving in a
:code:`requirements.txt` file, then simply pass in the file path via
:code:`@bentoml.env(requirements_txt_file='./requirements.txt')`.


Conda Packages
^^^^^^^^^^^^^^

Conda packages can be specified similarly, here's an example prediction service relying 
on an H2O model that requires the h2o conda packages:

.. code-block:: python

    @bentoml.artifacts([H2oModelArtifact('model')])
    @bentoml.env(
      pip_packages=['pandas', 'h2o==3.24.0.2'],
      conda_channels=['h2oai'],
      conda_dependencies=['h2o==3.24.0.2']
    )
    class ExamplePredictionService(bentoml.BentoService):

      @bentoml.api(input=DataframeInput(), batch=True)
      def predict(self, df):
          return self.artifacts.model.predict(df)


.. note::
    One caveat with Conda Packages here, is that it does not work with AWS Lambda 
    deployment due to the limitation of the AWS Lambda platform.

Using other Docker base images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, BentoML uses a default Docker base image and installs your model and its dependencies on top of it. This base image contains all of BentoML's dependencies and an installation of `conda` which helps BentoML to package and use the right Python version for your bundle.

However, there may be times when you need to use other Docker images (e.g. have some pre-build dependencies layers, company base image, using an Alpine-based image, etc.). BentoML makes it really easy to switch between base images by doing allowing you to specify a `docker_base_image`.

.. code-block:: python

  # e.g. using BentoML slim image
  @env(docker_base_image="bentoml/model-server:0.8.12-slim-py37")
  @artifacts([SklearnModelArtifact('model')])
  class ExamplePredictionService(BentoService):
    ...

In fact, one such base image that many may find useful are the BentoML slim base images. The original base image weighs in at roughly `~853MB` whereas the slim version weighs in at `~360MB`. 

.. code-block:: bash

  > docker image ls

  REPOSITORY                             TAG                   IMAGE ID            CREATED              SIZE
  bentoml/model-server                   0.8.12-slim-py37      109b451ed537        6 minutes ago        360MB
  bentoml/model-server                   0.8.12                f034fa23264c        33 minutes ago       853MB

This means that each image built on top of these slim images will be significantly smaller. 

.. code-block:: bash

  > docker image ls

  REPOSITORY                               TAG                 IMAGE ID            CREATED              SIZE
  jzhao2k19/iris                           latest              bfc9b81c7535        About a minute ago   1.54GB
  jzhao2k19/iris-slim                      latest              4e8d87a0c18a        4 minutes ago        577MB

However, as with using any alternative Docker base image, there are a few things to keep in mind. The regular base image uses `conda`, whereas the slim image does not. This has a few consequences. BentoML uses `conda` to ensure the Python version used matches the one you used to save your bundle. This means that you should manually select the right slim image for your bundle. For example, if you used Python 3.8 to train your model, you would use `bentoml/model-server:0.8.12-slim-py38`. Currently, BentoML support Python 3.6, 3.7, and 3.8. Additionally, this means that BentoML will ignore the `environment.yml`, meaning that user-defined `conda` packages and dependencies will be ignored. In the following example, only `pandas` will be installed, as the `conda_channels` and `conda_dependencies` will be ignored.

.. code-block:: python

  @bentoml.env(
    pip_packages=['pandas'],
    conda_channels=['h2oai'],
    conda_dependencies=['h2o==3.24.0.2'],
    docker_base_image="bentoml/model-server:0.8.12-slim-py37"
  )
  class ExamplePredictionService(bentoml.BentoService):
    ...

Init Bash Script
^^^^^^^^^^^^^^^^

Init setup script is used for customizing the API serving docker container. Users can
insert arbitrary bash script into the docker build process via init setup script, to 
install extra system dependencies or do other setups required by the prediction service.

.. code-block:: python

  @bentoml.env(
      infer_pip_packages=True,
      setup_sh="./my_init_script.sh"
  )
  class ExamplePredictionService(bentoml.BentoService):
      ...

  @bentoml.env(
      infer_pip_packages=True,
      setup_sh="""\
  #!/bin/bash
  set -e

  apt-get install --no-install-recommends nvidia-driver-430
  ...
    """
  )
  class ExamplePredictionService(bentoml.BentoService):
      ...

If you have a specific docker base image that you would like to use for your API server,
we are working on custom docker base image support. 
`contact us <mailto:contact@bentoml.ai>`_ if you are interested in helping with beta
testing this feature.


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
    from bentoml.adapters import DataframeInput
    from bentoml.frameworks.sklearn import SklearnModelArtifact
    from bentoml.frameworks.xgboost import XgboostModelArtifact

    @bentoml.env(infer_pip_packages=True)
    @artifacts([
        SklearnModelArtifact("model_a"),
        XgboostModelArtifact("model_b")
    ])
    class MyPredictionService(bentoml.BentoService):

        @bentoml.api(input=DataframeInput(), batch=True)
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

.. _concepts-api-func-and-adapters:

API Function and Adapters
-------------------------

BentoService API is the entry point for clients to access a prediction service. It is
defined by writing the API handling function(a class method within the BentoService
class) which gets called when client sent an inference request. User will need to
annotate this method with :code:`@bentoml.api` decorator and pass in an InputAdapter
instance, which defines the desired input format for the API function. For example,
if your model is expecting tabular data as input, you can use :code:`DataframeInput`
for your API,
e.g.:


.. code-block:: python


  class ExamplePredictionService(bentoml.BentoService):

      @bentoml.api(input=DataframeInput(), batch=True)
      def predict(self, df):
          assert type(df) == pandas.core.frame.DataFrame
          return postprocessing(model_output)


When using DataframeInput, BentoML will convert the inference requests sent from the
client, either in the form of a JSON HTTP request or a CSV file, into a
:code:`pandas.DataFrame` object and pass it down to the user-defined API function.

User can write arbitrary python code within the API function that process the data.
Besides passing the prediction input data to the model for inference, user can also
write Python code for data fetching, data pre-processing and post-processing within the
API function. For example:

.. code-block:: python

  from my_lib import preprocessing, postprocessing, fetch_user_profile_from_database

  class ExamplePredictionService(bentoml.BentoService):

      @bentoml.api(input=DataframeInput(), batch=True)
      def predict(self, df):
          user_profile_column = fetch_user_profile_from_database(df['user_id'])
          df['user_profile'] = user_profile_column
          model_input = preprocessing(df)
          model_output = self.artifacts.model.predict(model_input)
          return postprocessing(model_output)

.. note::

    Check out the :doc:`list of API InputAdapters <api/adapters>` that BentoML provides.


It is important to notice that in BentoML, the input variable passed into the
user-defined API function **is always a list of inference inputs**. BentoML users
must make sure their API function code is processing a batch of input data.

This design made it possible for BentoML to do Micro-Batching in online API serving, 
which is one of the most effective optimization technique for model serving systems.


API Function Return Value
^^^^^^^^^^^^^^^^^^^^^^^^^

The output of an API function can be any of the follow types:

.. code-block:: python

    pandas.DataFrame
    pandas.Series
    numpy.ndarray
    tensorflow.Tensor

    # JSON = t.Union[str, int, float, bool, None, t.Mapping[str, 'JSON'], t.List['JSON']]
    JSON
    # For batch enabled API, List of JSON Serializable
    List[JSON]

    # For fine-grained control
    bentoml.types.InferenceResult
    # For batch enabled API
    List[InferenceResult]
    bentoml.types.InferenceError
    # For batch enabled API
    List[InferenceError]

.. note::

    For API with batch enabled, it is user API function's responsibility to make sure
    the list of prediction results matches the order of input sequence and have the
    exact same length.


Defining a Batch API
^^^^^^^^^^^^^^^^^^^^

For APIs with ``batch=True``, the user-defined API function will be required to process
a list of input item at a time, and return a list of results of the same length. On the
contrary, @api by default uses batch=False, which processes one input item at a time.
Implementing a batch API allow your workload to benefit from BentoML's adaptive
micro-batching mechanism when serving online traffic, and also will speed up offline
batch inference job. We recommend using batch=True if performance & throughput is a
concern. Non-batch APIs are usually easier to implement, good for quick POC, simple
use cases, and deploying on Serverless platforms such as AWS Lambda, Azure function,
and Google KNative.

``DataframeInput`` and ``TfTensorInput`` are special input types that only support
accepting a batch of input at one time.

**Input data validation while handling batch input**

When the API function received a list of input, it is now possible to reject a subset
of the input data and return an error code to the client, if the input data is invalid
or malformatted. Users can do this via the InferenceTask#discard API, here's an example:

.. code-block:: python

    from typings import List
    from bentoml import env, artifacts, api, BentoService
    from bentoml.adapters import JsonInput
    from bentoml.types import JsonSerializable, InferenceTask  # type annotations are optional

    @env(infer_pip_packages=True)
    @artifacts([SklearnModelArtifact('classifier')])
    class MyPredictionService(BentoService):

            @api(input=JsonInput(), batch=True)
            def predict_batch(self, parsed_json_list: List[JsonSerializable], tasks: List[InferenceTask]):
                 model_input = []
                 for json, task in zip(parsed_json_list, tasks):
                      if "text" in json:
                          model_input.append(json['text'])
                      else:
                          task.discard(http_status=400, err_msg="input json must contain `text` field")

                results = self.artifacts.classifier(model_input)

                return results

The number of tasks got discarded plus the length of the results array returned, should
be equal to the length of the input list, this will allow BentoML to match the results
back to tasks that have not yet been discarded.

*Allow fine-grained control of the HTTP response, CLI inference job output, etc. E.g.:*

.. code-block:: python

    import bentoml
    from bentoml.types import JsonSerializable, InferenceTask, InferenceError  # type annotations are optional

    class MyService(bentoml.BentoService):

        @bentoml.api(input=JsonInput(), batch=False)
        def predict(self, parsed_json: JsonSerializable, task: InferenceTask) -> InferenceResult:
            if task.http_headers['Accept'] == "application/json":
                predictions = self.artifact.model.predict([parsed_json])
                return InferenceResult(
                    data=predictions[0],
                    http_status=200,
                    http_headers={"Content-Type": "application/json"},
                )
            else:
                return InferenceError(err_msg="application/json output only", http_status=400)

Or when ``batch=True``:

.. code-block:: python

    import bentoml
    from bentoml.types import JsonSerializable, InferenceTask, InferenceError  # type annotations are optional

    class MyService(bentoml.BentoService):

        @bentoml.api(input=JsonInput(), batch=True)
        def predict(self, parsed_json_list: List[JsonSerializable], tasks: List[InferenceTask]) -> List[InferenceResult]:
            rv = []
            predictions = self.artifact.model.predict(parsed_json_list)
            for task, prediction in zip(tasks, predictions):
                if task.http_headers['Accept'] == "application/json":
                    rv.append(
                        InferenceResult(
                            data=prediction,
                            http_status=200,
                            http_headers={"Content-Type": "application/json"},
                    ))
                else:
                    rv.append(InferenceError(err_msg="application/json output only", http_status=400))
                    # or task.discard(err_msg="application/json output only", http_status=400)
            return rv


Service with Multiple APIs
^^^^^^^^^^^^^^^^^^^^^^^^^^

A BentoService can contain multiple APIs, which makes it easy to build prediction
service that supports different access patterns for different clients, e.g.:

.. code-block:: python

  from my_lib import process_custom_json_format

  class ExamplePredictionService(bentoml.BentoService):

      @bentoml.api(input=DataframeInput(), batch=True)
      def predict(self, df: pandas.Dataframe):
          return self.artifacts.model.predict(df)

      @bentoml.api(input=JsonInput(), batch=True)
      def predict_json(self, json_arr):
          df = process_custom_json_format(json-arr)
          return self.artifacts.model.predict(df)


Make sure to give each API a different name. BentoML uses the method name as the API's
name, which will become part the serving endpoint it generates.

Operational API
^^^^^^^^^^^^^^^

User can also create APIs that, instead of handling an inference request, handles
request for updating prediction service configs or retraining models with new arrived
data. Operational API is still a beta feature, `contact us <mailto:contact@bentoml.ai>`_
if you're interested in learning more.


Customize Web UI
----------------

With ``@web_static_content`` decorator, you can add your web frontend project directory
to your BentoService class and BentoML will automatically bundle all the web UI files
and host them when starting the API server.

.. code-block:: python

    @env(auto_pip_dependencies=True)
    @artifacts([SklearnModelArtifact('model')])
    @web_static_content('./static')
    class IrisClassifier(BentoService):

        @api(input=DataframeInput(), batch=True)
        def predict(self, df):
            return self.artifacts.model.predict(df)

Here is an example project `bentoml/gallery@master/scikit-learn/iris-classifier <https://github.com/bentoml/gallery/blob/master/scikit-learn/iris-classifier/iris-classifier.ipynb>`_

.. image:: https://raw.githubusercontent.com/bentoml/gallery/master/scikit-learn/iris-classifier/webui.png


Saving BentoService
-------------------

After writing your model training/evaluation code and BentoService definition, here are
the steps required to create a BentoService instance and save it for serving:

#. Model Training
#. Create BentoService instance
#. Pack trained model artifacts with :ref:`BentoService#pack <bentoml-bentoservice-pack-label>`
#. Save to a Bento with :ref:`BentoService#save <bentoml-bentoservice-save-label>`

As illustrated in the previous example:

.. code-block:: python

  from sklearn import svm
  from sklearn import datasets

  # 1. Model training
  clf = svm.SVC(gamma='scale')
  iris = datasets.load_iris()
  X, y = iris.data, iris.target
  clf.fit(X, y)

  # 2. Create BentoService instance
  iris_classifier_service = IrisClassifier()

  # 3. Pack trained model artifacts
  iris_classifier_service.pack("model", clf)

  # 4. Save
  saved_path = iris_classifier_service.save()


How Save Works
^^^^^^^^^^^^^^

:ref:`BentoService#save_to_dir(path) <bentoml-bentoservice-save-label>` is the primitive
operation for saving the BentoService to a target directory. :code:`save_to_dir`
serializes the model artifacts and saves all the related code, dependencies and configs
into a the given path.

Users can then use :ref:`bentoml.load(path) <bentoml-load-label>` to load the exact same
BentoService instance back from the saved file path. This made it possible to easily
distribute your prediction service to test and production environment in a consistent
manner.

:ref:`BentoService#save <bentoml-bentoservice-save-label>` essentially calls
:ref:`BentoService#save_to_dir(path) <bentoml-bentoservice-save-label>` under the hood,
while keeping track of all the prediction services you've created and maintaining the
file structures and metadata information of those saved bundle.



.. _concepts-model-serving:

Model Serving
-------------

Once a BentoService is saved as a Bento, it is ready to be deployed for many different
types of serving workloads.

There are 3 main types of model serving - 

* **Online Serving** - clients access predictions via API endpoints in near real-time
* **Offline Batch Serving** - pre-compute predictions and save results in a storage system
* **Edge Serving** - distribute model and run it on mobile or IoT devices

BentoML has great support for online serving and offline batch serving. It has a 
high-performance API server that can load a saved Bento and expose a REST API for client
access. It also provide tools to load the Bento and feed it with a batch of inputs
for offline inference. Edge serving is only supported when the client has the Python
runtime, e.g. model serving in a router or a Raspberry Pi.

Online API Serving
^^^^^^^^^^^^^^^^^^

Once a BentoService is saved, you can easily start the REST API server to test out
sending request and interacting with the server. For example, after saving the 
BentoService in the :doc:`Getting Started Guide <quickstart>`, you can start a API
server right away with:

.. code-block:: bash

    bentoml serve IrisClassifier:latest


If you are using :ref:`save_to_dir <bentoml-bentoservice-save-label>` , or you have 
directly copied the saved Bento file directory from other machine, the BentoService
``IrisClassifier`` is not registered with your local BentoML repository. In that case,
you can still start the server by providing the path to the saved BentoService:

.. code-block:: bash

    bentoml serve $saved_path

The REST API request format is determined by each API's input type and input config.
More details can be found in the :ref:`BentoML API InputAdapters References <bentoml-api-adapters-label>`.

For running production API server, make sure to run ``bentoml serve-gunicorn`` 
command instead, or use Docker container for deployment.

.. code-block:: bash

    bentoml serve-gunicorn $saved_path --enable-microbatch --workers=2 --port=3000


API Server Dockerization
^^^^^^^^^^^^^^^^^^^^^^^^

When you are ready to deploy the service to production, a docker image containing your
model API server can be easily created with BentoML. When saving a Bento, a
``Dockerfile`` is also generated by BentoML in the same directory. ``Dockerfile`` is a 
text document that contains all the commands required for creating a docker image, and
``docker build`` command builds an image from a ``Dockerfile``.


.. code-block:: bash

    # Find the local path of the latest version IrisClassifier saved bundle
    saved_path=$(bentoml get IrisClassifier:latest --print-location --quiet)


    # Build docker image using saved_path directory as the build context, replace the
    # {username} below to your docker hub account name
    docker build -t {username}/iris_classifier_bento_service $saved_path

    # Run a container with the docker image built and expose port 5000
    docker run -p 5000:5000 {username}/iris_classifier_bento_service

    # Push the docker image to docker hub for deployment
    docker push {username}/iris_classifier_bento_service 


Here's an example deployment you can create in a Kubernetes cluster using the docker
image built above:

.. code-block:: yaml

  apiVersion: apps/v1 # for k8s versions before 1.9.0 use apps/v1beta2
  kind: Deployment
  metadata:
    name: iris_classifier
  spec:
    selector:
      matchLabels:
        app: iris_classifier
    replicas: 3
    template:
      metadata:
        labels:
          app: iris_classifier
      spec:
        containers:
        - name: iris_classifier_bento_service
          image: {username}/iris_classifier_bento_service:latest
          ports:
          - containerPort: 5000


Adaptive Micro-Batching
^^^^^^^^^^^^^^^^^^^^^^^

Micro batching is a technique where incoming prediction requests are grouped into small
batches to achieve the performance advantage of batch processing in model inference
tasks. BentoML implemented such a micro batching layer that is inspired by the paper
`Clipper: A Low-Latency Online Prediction Serving System 
<https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf>`_.


Given the mass performance improvement a model serving system get from micro-batching, 
BentoML APIs were designed to work with micro-batching without any code changes on the 
user side. It is why all the API InputAdapters are designed to accept a list of input data, 
as described in the :ref:`concepts-api-func-and-adapters` section.

Currently, micro-batching is still a beta feature, users can enable micro-batching by
passing a flag when running BentoML API server:

.. code-block:: bash

    # Launch micro batching API server from CLI
    bentoml serve-gunicorn $saved_path --enable-microbatch

    # Launch model server docker image with micro batching enabled
    docker run -p 5000:5000 -e BENTOML_ENABLE_MICROBATCH=True {username}/iris-classifier:latest


Programmatic Access
^^^^^^^^^^^^^^^^^^^

A saved BentoService can also be loaded from saved Bento and access directly from 
Python. There are two main ways this can be done:


1. Load from a saved Bento directory with :ref:`bentoml.load(path) <bentoml-load-label>` API

  .. code-block:: python

      import bentoml

      bento_service = bentoml.load(saved_path)
      result = bento_service.predict(input_data)

  The benefit of this approach is its flexibility. Users can easily invoke saved
  BentoService in their backend applications, and programmatically choose which model to
  load and how they are used for inference. 

2. Install BentoService as a PyPI package

  A Bento directory is also pip-installable as demonstrated in the 
  :doc:`Getting Started Guide <quickstart>`:

  .. code-block:: bash

      pip install $saved_path

  .. code-block:: python

    # Your bentoML model class name will become the package name
    import IrisClassifier

    installed_svc = IrisClassifier.load()
    installed_svc.predict([[5.1, 3.5, 1.4, 0.2]])

  This approach made sure that all the required pip dependencies are installed for the 
  BentoService when being installed. It is convenient when your Data Science team is
  shipping the prediction service as a standalone python package that can be shared
  by a variety of different developers to integrate with.

3. Command-Line Access

  Similarly, a Bento can be loaded for inference using the BentoML CLI tool. The CLI
  command `bentoml` is available once you've installed BentoML via ``pip``. And to load
  a saved Bento file, simply use the :code:`bentoml run` command and give it either the
  name and version pair, or the Bento's path:

  .. code-block:: bash
      
      # With BentoService name and version pair
      bentoml run IrisClassifier:latest predict --input='[[5.1, 3.5, 1.4, 0.2]]'
      bentoml run IrisClassifier:latest predict --input='./iris_test_data.csv'

      # With BentoService's saved path
      bentoml run $saved_path predict --input='[[5.1, 3.5, 1.4, 0.2]]'
      bentoml run $saved_path predict --input='./iris_test_data.csv'

  Or if you have already pip-installed the BentoService, it provides a CLI command
  specifically for this BentoService. The CLI command is the same as the BentoService
  class name:

  .. code-block:: bash
      
      IrisClassifier run predict --input='[[5.1, 3.5, 1.4, 0.2]]'
      IrisClassifier run predict --input='./iris_test_data.csv'



Offline Batch Serving
^^^^^^^^^^^^^^^^^^^^^

All three methods in the Programmatic Access section above, can be used for doing 
single-machine batch offline model serving. Depends on the format of input data. An
inference computation job can be started either with BentoService's Python API or Bash
CLI command. This made it very easy to integrate with Job scheduling tools such as 
`Apache Airflow <https://airflow.apache.org/>`_ and
`Celery <http://www.celeryproject.org/>`_.


For batch serving on large dataset running on a cluster, BentoML team is building a
Apache Spark UDF loader for BentoService. This feature is still in Beta testing phase. 
`Contact us <mailto:contact@bentoml.ai>`_ if you are interested in helping to test or
improve it.


Model Management
----------------

By default, :ref:`BentoService#save <bentoml-bentoservice-save-label>` will save all the
BentoService saved bundle files under :code:`~/bentoml/repository/` directory, following
by the service name and service version as sub-directory name. And all the metadata of
saved BentoService are stored in a local SQLite database file at
:code:`~/bentoml/storage.db`.

Users can easily query and use all the BentoService they have created, for example, to
list all the BentoService created:

.. code-block:: bash

    > bentoml list
    BENTO_SERVICE                                   AGE                  APIS                        ARTIFACTS
    IrisClassifier:20200323212422_A1D30D            1 day and 22 hours   predict<DataframeInput>   model<SklearnModelArtifact>
    IrisClassifier:20200304143410_CD5F13            3 weeks and 4 hours  predict<DataframeInput>   model<SklearnModelArtifact>
    SentimentAnalysisService:20191219090607_189CFE  13 weeks and 6 days  predict<DataframeInput>   model<SklearnModelArtifact>
    TfModelService:20191216125343_06BCA3            14 weeks and 2 days  predict<JsonInput>        model<TensorflowSavedModelArtifact>

    > bentoml get IrisClassifier
    BENTO_SERVICE                         CREATED_AT        APIS                       ARTIFACTS
    IrisClassifier:20200121114004_360ECB  2020-01-21 19:45  predict<DataframeInput>  model<SklearnModelArtifact>
    IrisClassifier:20200121114004_360ECB  2020-01-21 19:40  predict<DataframeInput>  model<SklearnModelArtifact>

    > bentoml get IrisClassifier:20200323212422_A1D30D
    {
      "name": "IrisClassifier",
      "version": "20200323212422_A1D30D",
      "uri": {
        "type": "LOCAL",
        "uri": "/Users/chaoyu/bentoml/repository/IrisClassifier/20200323212422_A1D30D"
      },
      "bentoServiceMetadata": {
        "name": "IrisClassifier",
        "version": "20200323212422_A1D30D",
        "createdAt": "2020-03-24T04:24:39.517239Z",
        "env": {
          "condaEnv": "name: bentoml-IrisClassifier\nchannels:\n- defaults\ndependencies:\n- python=3.7.5\n- pip\n",
          "pipDependencies": "bentoml==0.6.3\nscikit-learn",
          "pythonVersion": "3.7.5"
        },
        "artifacts": [
          {
            "name": "model",
            "artifactType": "SklearnModelArtifact"
          }
        ],
        "apis": [
          {
            "name": "predict",
            "InputType": "DataframeInput",
            "docs": "BentoService API",
            "inputConfig": {
              "orient": "records",
              "typ": "frame",
              "dtypes": null
            }
          }
        ]
      }
    }

Similarly, the Bento name and version pair can be used to load and run those
BentoService directly. For example:

.. code-block:: bash

    > bentoml serve IrisClassifier:latest
    * Serving Flask app "IrisClassifier" (lazy loading)
    * Environment: production
      WARNING: This is a development server. Do not use it in a production deployment.
      Use a production WSGI server instead.
    * Debug mode: off
    * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

    > bentoml run IrisClassifier:latest predict --input='[[5.1, 3.5, 1.4, 0.2]]'
    [0]


Customizing Model Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BentoML has a standalone component :code:`YataiService` that handles model storage and
deployment. BentoML uses a local :code:`YataiService` instance by default, which saves
BentoService files to :code:`~/bentoml/repository/` directory and other metadata to
:code:`~/bentoml/storage.db`.

Users can also customize this to make it work for team settings, making it possible
for a team of data scientists to easily share, use and deploy models and prediction
services created by each other. To do so, the user will need to setup a host server
that runs :code:`YataiService`, from BentoML cli command `yatai-service-start`:

.. code-block:: bash

    > bentoml yatai-service-start --help
    Usage: bentoml yatai-service-start [OPTIONS]

      Start BentoML YataiService for model management and deployment

    Options:
      --db-url TEXT         Database URL following RFC-1738, and usually can
                            include username, password, hostname, database name as
                            well as optional keyword arguments for additional
                            configuration
      --repo-base-url TEXT  Base URL for storing saved BentoService bundle files,
                            this can be a filesystem path(POSIX/Windows), or an S3
                            URL, usually starts with "s3://"
      --grpc-port INTEGER   Port for Yatai server
      --ui-port INTEGER     Port for Yatai web UI
      --ui / --no-ui        Start BentoML YataiService without Web UI
      -q, --quiet           Hide all warnings and info logs
      --verbose, --debug    Show debug logs when running the command
      --help                Show this message and exit.


BentoML provides a pre-built docker image for running YataiService. For each BentoML 
release, a new image will be pushed to [docker hub](https://hub.docker.com/r/bentoml/yatai-service/tags) under :code:`bentoml/yatai-service` 
with the same image tag as the PyPI package version. For example, use the following 
command to start a YataiService of BentoML version 0.8.6, loading data from your local
BentoML repository under the local ``~/bentoml`` directory:

.. code-block:: bash

    > docker run -v ~/bentoml:/bentoml \
        -p 3000:3000 \
        -p 50051:50051 \
        bentoml/yatai-service:0.8.6 \
        --db-url=sqlite:///bentoml/storage.db \
        --repo-base-url=/bentoml/repository


The recommended way to deploy :code:`YataiService` for teams, is to back it by a
remote PostgreSQL database and an S3 bucket. For example, deploy the following docker
container to run a YataiService configured with remote database and S3 storage, as well
as AWS credentials for managing deployments created on AWS: 

.. code-block:: bash

    > docker run -p 3000:3000 -p 50051:50051 \
        -e AWS_SECRET_ACCESS_KEY=... -e AWS_ACCESS_KEY_ID=...  \
        bentoml/yatai-service \
        --db-url postgresql://scott:tiger@localhost:5432/bentomldb \
        --repo-base-url s3://my-bentoml-repo/

    * Starting BentoML YataiService gRPC Server
    * Debug mode: off
    * Web UI: running on http://127.0.0.1:3000
    * Running on 127.0.0.1:50051 (Press CTRL+C to quit)
    * Usage: `bentoml config set yatai_service.url=127.0.0.1:50051`
    * Help and instructions: https://docs.bentoml.org/en/latest/guides/yatai_service.html
    * Web server log can be found here: /Users/chaoyu/bentoml/logs/yatai_web_server.log


After deploying the YataiService server, get the server IP address and run the following 
command to configure BentoML client to use this remote YataiService for model management
and deployments. You will need to replace ``127.0.0.1`` with an IP address or URL
that is accessible for your team:

.. code-block:: bash

    bentoml config set yatai_service.url=127.0.0.1:50051

Once you've run the command above, all the BentoML model management operations will be
sent to the remote server, including saving BentoService, query saved BentoServices or
creating model serving deployments.


.. note::

    BentoML's :code:`YataiService` does not provide any kind of authentication. To
    secure your deployment, we recommend only make the server accessible within your
    VPC for you data science team to have access.

    BentoML team also provides hosted YataiService for enterprise teams, that has all
    the security best practices built-in, to bootstrap the end-to-end model management 
    and model serving deployment workflow. `Contact us <mailto:contact@bentoml.ai>`_ to
    learn more about our offerings.


Labels
------

Labels are key/value pairs for BentoService and deployment to be used to identify
attributes that are relevant to the users. Labels do not have any direct implications
to YataiService.  Each key must be unique for the given resource.

Valid label name and value must be 63 characters or less, beginning and ending with an
alphanumeric character([a-zA-Z0-9]) with dashes (`-`), underscores (`_`), dots(`.`),
and alphanumeric between.

Example labels:

* `"cicd-status": "success"`
* `"data-cohort": "2020.9.10-2020.9.11"`
* `"created_by": "Tim_Apple"`


**Set labels for Bentos**

Currently, the only way to set labels for Bento is during save Bento as Bento bundle.

.. code-block:: python

    svc = MyBentosService()
    svc.pack('model', model)
    svc.save(labels={"framework": "xgboost"})


**Set labels for deployments**

Currently, CLI is the only way to set labels for deployments. In the upcoming release, BentoML
provides alternative ways to set and update labels.

.. code-block:: bash

    $ # In any of the deploy command, you can add labels via --label option
    $ bentoml azure-functions deploy my_deployment --bento service:name \
        --labels key1:value1,key2:value2


Label selector
^^^^^^^^^^^^^^

BentoML provides label selector for the user to identify BentoServices or deployments.
The label selector query supports two type of selector: `equality-based` and `set-based`.
A label selector query can be made of multiple requirements which are comma-separated.
In the case of multiple requirements, the comma separator acts as a logical AND operator.

**Equality-based requirements**

Equality-based requirements allow filtering by label keys and values, matching resources
must satisfy the specified label constraint. The available operators are `=` and `!=`.
`=` represents equality, and `!=` represents inequality.

Examples:

* ``framework=pytorch``
* ``cicd_result!=failed``

**Set-based requirements**

Set-based requirements allow you to filter keys according to a set of values. BentoML
supports four type of operators, `In`, `NotIn`, `Exists`, `DoesNotExist`.

Example:

* ``framework In (xgboost, lightgbm)``

    This example selects all resources with key equals to `framework` and value equal to `xgboost` or `lightgbm`

* ``platform NotIn (lambda, azure-function)``

    This label selector selects all resources with key equals to `platform` and value not equal to `lambda` or `azure-function`.

* ``fb_cohort Exists``

    This example selects all resources that has a label with key equal to `fb_cohort`

* ``cicd DoesNotExist``

    This label selector selects all resources that does not have a label with key equal to `cicd`.


**Use label selector in CLI**

There are several CLI commands supported label selector. More ways to interact with label
selector will be available in the future versions.

Supported CLI commands:

* ``bentoml list``
* ``bentoml get``

    ``--labels`` option will be ignored if the version is provided.
    ``$ bentoml get bento_name --labels "key1=value1, key2 In (value2, value3)"``
* ``bentoml deployment list``
* ``bentoml lambda list``
* ``bentoml sagemaker list``
* ``bentoml azure-functions list``


Retrieving BentoServices
------------------------

After saving your Model services to BentoML, you can retrieve the artifact bundle using the CLI from any environment configured to use the YataiService. The :code:`--target_dir` flag specifies where the artifact bundle will be populated. If the directory exists, it will not be overwritten to avoid inconsistent bundles.

.. code-block:: bash

    > bentoml retrieve --help
    Usage: bentoml retrieve [OPTIONS] BENTO

      Retrieves BentoService artifacts into a target directory

    Options:
      --target_dir TEXT   Directory to put artifacts into. Defaults to pwd.
      -q, --quiet         Hide all warnings and info logs
      --verbose, --debug  Show debug logs when running the command
      --help              Show this message and exit.

This command extends BentoML to be useful in a CI workflow or to provide a rapid way to share Services with others.

.. code-block:: bash

    bentoml retrieve ModelServe --target_dir=~/bentoml_bundle/


.. spelling::

    pre
    init
    deserialization
    malformatted
    frontend
    IoT
    programmatically
    Jupyter
    jupyter
    installable
    zA
