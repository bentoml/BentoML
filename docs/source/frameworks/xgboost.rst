=======
XGBoost
=======

XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable. This guide provides an overview of using `XGBoost <https://xgboost.readthedocs.io/en/stable/>`_ with BentoML.

Compatibility
~~~~~~~~~~~~~

BentoML has been validated to work with XGBoost version 0.7post3 and higher.

Saving a Trained Booster
------------------------

First, train or load a booster. In this example, we will be training a new booster using UCI's
`breast cancer dataset <https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)>`_.
If you've already saved a model using XGBoost, simply load it back into Python using
``Booster.load_model``.

.. code-block:: python

   import xgboost as xgb
   from sklearn.datasets import load_breast_cancer

   cancer = load_breast_cancer()

   X = cancer.data
   y = cancer.target

   dt = xgb.DMatrix(X, label=y)

   param = {"max_depth": 3, "eta": 0.3, "objective": "multi:softprob", "num_class": 2}
   bst = xgb.train(param, dt)


After training, use :obj:`~bentoml.xgboost.save_model()` to save the Booster instance to BentoML model store. XGBoost has no
framework-specific save options.

.. code-block:: python

   import bentoml
   bento_model = bentoml.xgboost.save_model("booster_tree", bst)

To verify that the saved learner can be loaded properly:

.. code-block:: python

   import bentoml
   booster = bentoml.xgboost.load_model("booster_tree:latest")
   booster.predict(xgb.DMatrix([[1.308e+01, 1.571e+01, 8.563e+01, 5.200e+02, 1.075e-01, 1.270e-01,
       4.568e-02, 3.110e-02, 1.967e-01, 6.811e-02, 1.852e-01, 7.477e-01,
       1.383e+00, 1.467e+01, 4.097e-03, 1.898e-02, 1.698e-02, 6.490e-03,
       1.678e-02, 2.425e-03, 1.450e+01, 2.049e+01, 9.609e+01, 6.305e+02,
       1.312e-01, 2.776e-01, 1.890e-01, 7.283e-02, 3.184e-01, 8.183e-02]]))

.. note::
   ``load_model`` should only be used when the booster object itself is required. When using a saved
   booster in a BentoML service, use :obj:`~bentoml.xgboost.get` and create a runner as described
   below.

Building a Service
------------------

.. seealso::

   :ref:`Building a Service <concepts/service:Service and APIs>`: more information on creating a
   prediction service with BentoML.

Create a ``service.py`` file separate from your training code that will be used to define the
BentoML service:

.. code-block:: python

   import bentoml
   from bentoml.io import NumpyNdarray
   import numpy as np

   # create a runner from the saved Booster
   runner = bentoml.xgboost.get("booster_tree:latest").to_runner()

   # create a BentoML service
   svc = bentoml.Service("cancer_classifier", runners=[runner])

   # define a new endpoint on the BentoML service
   @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
   async def classify_tumor(input: np.ndarray) -> np.ndarray:
       # use 'runner.predict.run(input)' instead of 'booster.predict'
       res = await runner.predict.async_run(input)
       return res

Take note of the name of the service (``svc`` in this example) and the name of the file.

You should also have a ``bentofile.yaml`` alongside the service file that specifies that
information, as well as the fact that it depends on XGBoost. This can be done using either
``python`` (if using pip), or ``conda``:

.. tab-set::
   .. tab-item:: pip

      .. code-block:: yaml

         service: "service:svc"
         description: "My XGBoost service"
         python:
	   packages:
	     - xgboost

   .. tab-item:: conda

      .. code-block:: yaml

         service: "service:svc"
         description: "My XGBoost service"
         conda:
           channels:
           - conda-forge
           dependencies:
           - xgboost

Using Runners
~~~~~~~~~~~~~
.. seealso::

   :ref:`concepts/runner:Using Runners`: a general introduction to the Runner concept and its usage.

A runner for a Booster is created like so:

.. code-block:: python

   bentoml.xgboost.get("model_name:model_version").to_runner()

``runner.predict.run`` is generally a drop-in replacement for ``booster.predict``. However, while it
is possible to pass a ``DMatrix`` as input, BentoML does not support adaptive batching in that case.
It is therefore recommended to use a NumPy ``ndarray`` or Pandas ``DataFrame`` as input instead.

There are no special options for loading XGBoost.

Runners must to be initialized in order for their ``run`` methods to work. This is done by BentoML
internally when you serve a bento with ``bentoml serve``. See the :ref:`runner debugging guide
<concepts/service:Debugging Runners>` for more information about initializing runners locally.


GPU Inference
~~~~~~~~~~~~~

If there is a GPU available, the XGBoost Runner will automatically use ``gpu_predictor`` by default.
This can be disabled by using the
:ref:`BentoML configuration file <guides/configuration:Configuration>` to disable Runner GPU
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


Adaptive Batching
~~~~~~~~~~~~~~~~~

.. seealso::

   :ref:`guides/batching:Adaptive Batching`: a general introduction to adaptive batching in BentoML.

XGBoost's ``booster.predict`` supports taking batch input for inference. This is disabled by
default, but can be enabled using the appropriate signature when saving your booster.

.. note

   BentoML does not currently support adaptive batching for ``DMatrix`` input. In order to enable
   batching, use either a NumPy ``ndarray`` or a Pandas ``DataFrame`` instead.

.. code-block:: python

   bento_model = bentoml.xgboost.save_model("booster_tree", booster, signatures={"predict": {"batchable": True}})

.. note::

   You can find more examples for **XGBoost** in our :examples:`bentoml/examples/xgboost
   <xgboost>` directory.

.. currentmodule:: bentoml.xgboost
