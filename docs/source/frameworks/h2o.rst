H2O
---

Users can now use H2O with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml
   import h2o
   import h2o.model
   import h2o.automl

   H2O_PORT = 54323

   def train_h2o_aml() -> h2o.automl.H2OAutoML:

      h2o.init(port=H2O_PORT)
      h2o.no_progress()

      df = h2o.import_file(
         "https://github.com/yubozhao/bentoml-h2o-data-for-testing/raw/master/"
         "powerplant_output.csv"
      )
      splits = df.split_frame(ratios=[0.8], seed=1)
      train = splits[0]
      test = splits[1]

      aml = h2o.automl.H2OAutoML(
         max_runtime_secs=60, seed=1, project_name="powerplant_lb_frame"
      )
      aml.train(y="HourlyEnergyOutputMW", training_frame=train, leaderboard_frame=test)

      return aml
   
   model = train_h2o_aml()

   # `save` a model to BentoML modelstore:
   tag = bentoml.h2o.save("h2o_model", model.leader)

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   h2o_loaded: h2o.model.model_base.ModelBase = bentoml.h2o.load(
       tag,
       init_params=dict(port=H2O_PORT),
   )

.. note::

   You can find more examples for **H2O** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.h2o

.. autofunction:: bentoml.h2o.save

.. autofunction:: bentoml.h2o.load

.. autofunction:: bentoml.h2o.load_runner
