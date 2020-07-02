Deploying to Clipper Cluster
============================

Clipper(http://clipper.ai/) is a low-latency prediction serving system for machine learning.

It provides a powerful way to orchestrate ML model containers and supports features such as `micro batching`_ which is
critical for building low latency online model serving systems.

BentoML makes it easier to build custom containers that can be deployed to Clipper, users can easily add Clipper
specify API inputs to their prediction service created with BentoML, and deploy them into clipper cluster.
In this guide, we will demonstrate how to deploy a scikit-learn model to clipper, using BentoML.

.. _micro batching: https://www.usenix.org/system/files/conference/nsdi17/nsdi17-crankshaw.pdf

Prerequisites
-------------

* Clipper cluster http://clipper.ai

* Docker is installed and running on the machine.

  * Install instruction: https://docs.docker.com/install


Build BentoService for Clipper
------------------------------

===========================
Train Iris classifier model
===========================

.. code-block:: python

    >>> from sklearn import svm
    >>> from sklearn import datasets

    >>> clf = svm.SVC()
    >>> iris = datasets.load_iris()
    >>> X, y = iris.data, iris.target
    >>> clf.fit(X, y)


BentoML provides input types that are specific for use with Clipper, including `ClipperBytesInput`,
`ClipperIntsInput`, `ClipperFloatsInput`, `ClipperDoublesInput`, `ClipperStringsInput` each
corresponding to one input type that clipper support.

Other than using Clipper specific input, the rest are the same as defining a regular BentoService class:

.. code-block:: python

    >>> # save this to a separate iris_classifier.py file
    >>> from bentoml import BentoService, api, env, artifacts
    >>> from bentoml.artifact import PickleArtifact
    >>> from bentoml.adapters import DataframeInput, ClipperFloatsInput

    >>> @artifacts([PickleArtifact('model')])
    >>> @env(pip_dependencies=["scikit-learn"])
    >>> class IrisClassifier(BentoService):

    >>>     @api(input=DataframeInput())
    >>>     def predict(self, df):
    >>>         return self.artifacts.model.predict(df)
    >>>
    >>>     @api(input=ClipperFloatsInput())
    >>>     def predict_clipper(self, inputs):
    >>>         return self.artifacts.model.predict(inputs)


Save the BentoService

.. code-block:: python

    >>> # 1) import the custom BentoService defined above
    >>> from iris_classifier import IrisClassifier

    >>> # 2) `pack` it with required artifacts
    >>> svc = IrisClassifier()
    >>> svc.pack('model', clf)

    >>> # 3) save packed BentoService as archive
    >>> saved_path = svc.save()

    running sdist
    running egg_info
    writing requirements to BentoML.egg-info/requires.txt
    writing BentoML.egg-info/PKG-INFO
    writing top-level names to BentoML.egg-info/top_level.txt
    writing dependency_links to BentoML.egg-info/dependency_links.txt
    writing entry points to BentoML.egg-info/entry_points.txt
    reading manifest file 'BentoML.egg-info/SOURCES.txt'
    ...
    ...
    ...
    [2019-11-13 15:41:24,395] INFO - BentoService bundle 'IrisClassifier:20191113154121_E7D3CE' created at: /Users/chaoyuyang/bentoml/repository/IrisClassifier/20191113154121_E7D3CE


Test the clipper input directly with a list of floats as input

.. code-block:: python

    >>> svc.predict_clipper([X[0]])

    array([0])


Deploying BentoService to local Clipper cluster
-----------------------------------------------

The sample code below assumes you have docker setup and starts a local Clipper cluster using Docker.


Start the Clipper cluster

.. code-block:: python

    >>> from clipper_admin import ClipperConnection, DockerContainerManager
    >>> cl = ClipperConnection(DockerContainerManager())
    >>> cl.start_clipper()

    19-11-13:15:43:33 INFO     [docker_container_manager.py:184] [default-cluster] Starting managed Redis instance in Docker
    19-11-13:15:43:37 INFO     [docker_container_manager.py:276] [default-cluster] Metric Configuration Saved at /private/var/folders/ns/vc9qhmqx5dx_9fws7d869lqh0000gn/T/tmp_V3qv1.yml
    19-11-13:15:43:38 INFO     [clipper_admin.py:162] [default-cluster] Clipper is running


Register an application on the clipper cluster

.. code-block:: python

    >>> cl.register_application('bentoml-test', 'floats', 'default_pred', 100000)

    19-11-13:15:43:58 INFO     [clipper_admin.py:236] [default-cluster] Application bentoml-test was successfully registered


Now you can deploy the saved BentoService using this Clipper connection and BentoML's `bentoml.clipper.deploy_bentoml` API,
which will first build a clipper model docker image that containing your BentoService and then deploy it to the cluster.

.. code-block:: python

    >>> from bentoml.clipper import deploy_bentoml

    >>> saved_path = "/Users/chaoyuyang/bentoml/repository/IrisClassifier/20191113154121_E7D3CE"

    >>> clipper_model_name, clipper_model_version = deploy_bentoml(cl, saved_path, 'predict_clipper')

    [2019-11-13 15:45:49,422] WARNING - BentoML local changes detected - Local BentoML repository including all code changes will be bundled together with the BentoService archive. When used with docker, the base docker image will be default to same version as last PyPI release at version: 0.4.9. You can also force bentoml to use a specific version for deploying your BentoService archive, by setting the config 'core/bentoml_deploy_version' to a pinned version or your custom BentoML on github, e.g.:'bentoml_deploy_version = git+https://github.com/{username}/bentoml.git@{branch}'
    [2019-11-13 15:45:49,444] WARNING - BentoArchive version mismatch: loading archive bundled in version 0.4.9,  but loading from version 0.4.9+7.g429b9ec.dirty
    [2019-11-13 15:45:49,772] INFO - Step 1/10 : FROM clipper/python-closure-container:0.4.1
    [2019-11-13 15:45:49,775] INFO -

    [2019-11-13 15:45:49,777] INFO -  ---> e9b89c285ef8

    [2019-11-13 15:45:49,780] INFO - Step 2/10 : COPY . /container

    ...
    ...
    ...

    [2019-11-13 15:46:45,596] INFO -  ---> 8d5863be7a60

    [2019-11-13 15:46:45,598] INFO - Successfully built 8d5863be7a60

    [2019-11-13 15:46:45,604] INFO - Successfully tagged clipper-model-irisclassifier:20191113154121_E7D3CE

    [2019-11-13 15:46:45,606] INFO - Successfully built docker image clipper-model-irisclassifier:20191113154121_E7D3CE for Clipper deployment
    19-11-13:15:46:45 INFO     [docker_container_manager.py:409] [default-cluster] Found 0 replicas for irisclassifier-predict-clipper:20191113154121-e7d3ce. Adding 1
    19-11-13:15:46:46 INFO     [clipper_admin.py:724] [default-cluster] Successfully registered model irisclassifier-predict-clipper:20191113154121-e7d3ce
    19-11-13:15:46:46 INFO     [clipper_admin.py:642] [default-cluster] Done deploying model irisclassifier-predict-clipper:20191113154121-e7d3ce.


Use `get_all_models` api to check is the model properly linked and deployed.

.. code-block:: python

    >>> cl.get_all_models()

    [u'irisclassifier-predict-clipper:20191113154121-e7d3ce']

Link the deployed model with the `bentoml-test` application created above

.. code-block:: python

    >>> cl.link_model_to_app('bentoml-test', clipper_model_name)


    19-11-13:15:47:05 INFO     [clipper_admin.py:303] [default-cluster] Model irisclassifier-predict-clipper is now linked to application bentoml-test


Let's test the application by sending prediction request with sample data.

.. code-block:: python

    >>> import requests, json
    >>> # Get Address
    >>> addr = cl.get_query_addr()
    >>> # Post Query
    >>> response = requests.post(
    >>>     "http://%s/%s/predict" % (addr, 'bentoml-test'),
    >>>     headers={"Content-type": "application/json"},
    >>>     data=json.dumps({
    >>>         'input': [6.5, 3.0 , 5.8, 2.2]
    >>>     }))

    >>> result = response.json()
    >>> if response.status_code == requests.codes.ok and result["default"]:
    >>>     print('A default prediction was returned.')
    >>>     print(result)

    >>> elif response.status_code != requests.codes.ok:
    >>>     print(result)
    >>> #     raise BenchmarkException(response.text)
    >>> else:
    >>>     print('Prediction Returned:', result)

    ('Prediction Returned:', {u'default': False, u'output': 2, u'query_id': 0})
