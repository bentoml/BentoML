=======
XGBoost
=======

Users can now use XGBoost with BentoML with the following API: :code:`save_model`, :code:`load_model`, and :code:`get` as follows:

.. code-block:: python

   import bentoml
   import xgboost as xgb

   def xgboost_model() -> "xgb.Booster":
      from sklearn.datasets import load_breast_cancer

      # read in data
      cancer = load_breast_cancer()

      X = cancer.data
      y = cancer.target

      dt = xgb.DMatrix(X, label=y)

      # specify parameters via map
      param = {"max_depth": 3, "eta": 0.3, "objective": "multi:softprob", "num_class": 2}
      bst = xgb.train(param, dt)

      return bst

   model = xgboost_model()

   # `save` a given model and retrieve coresponding tag:
   tag = bentoml.xgboost.save_model("booster_tree", model, booster_params={"disable_default_eval_metric": 1, "nthread": 2, "tree_method": "hist"})

   # retrieve metadata with `bentoml.xgboost.get`:
   metadata = bentoml.xgboost.get(tag)

   # `load` the model back in memory:
   loaded = bentoml.xgboost.load_model("booster_tree")

.. note::

   You can find more examples for **XGBoost** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.xgboost

.. autofunction:: bentoml.xgboost.save_model

.. autofunction:: bentoml.xgboost.load_model

.. autofunction:: bentoml.xgboost.get
