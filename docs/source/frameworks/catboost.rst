========
CatBoost
========


CatBoost is a machine learning algorithm that uses gradient boosting on decision trees. It is available as an open source library.
To learn more about CatBoost, visit their `documentation <https://catboost.ai/en/docs/>`_.

BentoML provides native support for `CatBoost <https://github.com/catboost/catboost>`_, and this guide provides an overview of how to use BentoML with CatBoost.

Saving a trained CatBoost model
--------------------------------

In this example, we will train a new model using UCI's `breast cancer dataset <https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)>`_.

.. code-block:: python

   import bentoml

   import catboost as cbt

   from sklearn.datasets import load_breast_cancer

   cancer = load_breast_cancer()

   X = cancer.data
   y = cancer.target

   model = cbt.CatBoostClassifier(
       iterations=2,
       depth=2,
       learning_rate=1,
       loss_function="Logloss",
       verbose=False,
   )

   # train the model
   model.fit(X, y)


Use :obj:`~bentoml.catboost.save_model` to save the model instance to BentoML model store:

.. code-block:: python

   bento_model = bentoml.catboost.save_model("catboost_cancer_clf", model)


To verify that the saved learner can be loaded properly:

.. code-block:: python

   model = bentoml.catboost.load_model("catboost_cancer_clf:latest")

   model.predict(cbt.Pool([[1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
       4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
       1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
       1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
       1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]]))


Building a Service using CatBoost
---------------------------------

.. seealso::

   :ref:`Building a Service <concepts/service:Service and APIs>`: more information on creating a prediction service with BentoML.

.. code-block:: python

   import bentoml

   import numpy as np

   from bentoml.io import NumpyNdarray

   runner = bentoml.catboost.get("catboost_cancer_clf:latest").to_runner()

   svc = bentoml.Service("cancer_clf", runners=[runner])


   @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
   async def classify_cancer(input: np.ndarray) -> np.ndarray:
      # returns sentiment score of a given text
      res = await runner.predict.async_run(input)
      return res


When constructing a :ref:`bentofile.yaml <concepts/bento:Bento Build Options>`,
there are two ways to include CatBoost as a dependency, via ``python`` or
``conda``:

.. tab-set::

   .. tab-item:: python

      .. code-block:: yaml

         python:
	   packages:
	     - catboost

   .. tab-item:: conda

      .. code-block:: yaml

         conda:
           channels:
           - conda-forge
           dependencies:
           - catboost


Using Runners
-------------

.. seealso::

   See :ref:`concepts/runner:Using Runners` doc for a general introduction to the Runner concept and its usage.

A CatBoost :obj:`~bentoml.Runner` can be created as follows:

.. code-block:: python

   runner = bentoml.catboost.get("model_name:model_version").to_runner()

``runner.predict.run`` is generally a drop-in replacement for ``model.predict``.

While a `Pool <https://catboost.ai/en/docs/concepts/python-reference_pool>`_ can be passed to a CatBoost Runner, BentoML does not support adaptive batching for ``Pool`` objects.

To use adaptive batching feature from BentoML, we recommend our users to use either NumPy ``ndarray`` or Pandas ``DataFrame`` instead.

.. note::

   Currently ``staged_predict`` callback is not yet supported with :code:`bentoml.catboost`.

Using GPU
---------

CatBoost Runners will automatically use ``task_type=GPU`` if a GPU is detected.

This behavior can be disabled using the :ref:`BentoML configuration file<guides/configuration:Configuration>`:

access:

.. code-block:: yaml

   runners:
      # resources can be configured at the top level
      resources:
         nvidia.com/gpu: 0
      # or per runner
      my_runner_name:
         resources:
             nvidia.com/gpu: 0

Adaptive batching 
~~~~~~~~~~~~~~~~~

.. seealso::

   :ref:`guides/batching:Adaptive Batching`: a general introduction to adaptive batching in BentoML.

CatBoost's ``model.predict`` supports taking batch input for inference. This is disabled by
default, but can be enabled using the appropriate signature when saving your model.

.. note::

   BentoML does not currently support adaptive batching for ``Pool`` input. In order to enable
   batching, use either a NumPy ``ndarray`` or a Pandas ``DataFrame`` instead.

.. code-block:: python

   bento_model = bentoml.catboost.save_model(
    "catboost_cancer_clf", model, signatures={"predict": {"batchable": True}}
    )
