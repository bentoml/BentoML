LightGBM
--------

| LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages:
| - Faster training speed and higher efficiency.
| - Lower memory usage.
| - Better accuracy.
| - Support of parallel, distributed, and GPU learning.
| - Capable of handling large-scale data. - `Source <https://lightgbm.readthedocs.io/en/latest/>`_

Users can now use LightGBM with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml
   import lightgbm as lgb
   import pandas as pd

   # load a dataset
   df_train = pd.read_csv("regression.train", header=None, sep="\t")
   df_test = pd.read_csv("regression.test", header=None, sep="\t")

   y_train = df_train[0]
   y_test = df_test[0]
   X_train = df_train.drop(0, axis=1)
   X_test = df_test.drop(0, axis=1)

   # create dataset for lightgbm
   lgb_train = lgb.Dataset(X_train, y_train)
   lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

   # specify your configurations as a dict
   params = {
      "boosting_type": "gbdt",
      "objective": "regression",
      "metric": {"l2", "l1"},
      "num_leaves": 31,
      "learning_rate": 0.05,
   }

   # train
   gbm = lgb.train(
      params, lgb_train, num_boost_round=20, valid_sets=lgb_eval
   )

   # `save` a given classifier and retrieve coresponding tag:
   tag = bentoml.lightgbm.save("my_lightgbm_model", gbm, booster_params=params)

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   model = bentoml.lightgbm.load("my_lightgbm_model")

   # Run a given model under `Runner` abstraction with `load_runner`
   input_data = pd.from_csv("/path/to/csv")
   runner = bentoml.lightgbm.load_runner("my_lightgbm_model:latest")
   runner.run(input_data)

.. note::
   You can find more examples for **LightGBM** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.lightgbm

.. autofunction:: bentoml.lightgbm.save

.. autofunction:: bentoml.lightgbm.load

.. autofunction:: bentoml.lightgbm.load_runner