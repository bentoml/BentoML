XGBoost
-------

| XGBoost is an optimized distributed gradient boosting library designed
| to be highly efficient, flexible and portable. It implements machine learning algorithms
| under the `Gradient Boosting <https://en.wikipedia.org/wiki/Gradient_boosting>`_ framework.
| XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science
| problems in a fast and accurate way. The same code runs on major distributed environment (Hadoop, SGE, MPI)
| and can solve problems beyond billions of examples. - `Source <https://xgboost.readthedocs.io/en/stable/>`_

Users can now use XGBoost with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

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
   tag = bentoml.xgboost.save("booster_tree", model, booster_params={"disable_default_eval_metric": 1, "nthread": 2, "tree_method": "hist"})

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   loaded = bentoml.xgboost.load("booster_tree")

.. note::
   You can find more examples for **XGBoost** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.xgboost

.. autofunction:: bentoml.xgboost.save

.. autofunction:: bentoml.xgboost.load

.. autofunction:: bentoml.xgboost.load_runner