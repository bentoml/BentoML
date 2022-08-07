========
LightGBM
========

Users can now use LightGBM with BentoML with the following API: :code:`load_model`,
:code:`save_model`, and :code:`get` as follow:

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
   bentoml.lightgbm.save_model("my_lightgbm_model", gbm, booster_params=params)

   # retrieve metadata with `bentoml.models.get`:
   bento_model = bentoml.models.get("my_lightgbm_model:latest")

   # `load` the model back in memory:
   loaded_model = bentoml.lightgbm.load_model("my_lightgbm_model")

   # Run a given model under `Runner` abstraction with `to_runner`
   input_data = pd.from_csv("/path/to/csv")
   runner = bentoml.lightgbm.get("my_lightgbm_model:latest").to_runner()
   runner.init_local()
   runner.run(input_data)

.. note::

   You can find more examples for **LightGBM** in our `bentoml/examples <https://github.com/bentoml/BentoML/tree/main/examples>`_ repo.


