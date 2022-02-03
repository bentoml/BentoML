statsmodels
-----------

Users can now use statsmodels with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml
   import pandas as pd

   import statsmodels
   from statsmodels.tsa.holtwinters import HoltWintersResults
   from statsmodels.tsa.holtwinters import ExponentialSmoothing

   def holt_model() -> "HoltWintersResults":
       df: pd.DataFrame = pd.read_csv(
           "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv"
       )

       # Taking a test-train split of 80 %
       train = df[0 : int(len(df) * 0.8)]
       test = df[int(len(df) * 0.8) :]

       # Pre-processing the  Month  field
       train.Timestamp = pd.to_datetime(train.Month, format="%m-%d")
       train.index = train.Timestamp
       test.Timestamp = pd.to_datetime(test.Month, format="%m-%d")
       test.index = test.Timestamp

       # fitting the model based on  optimal parameters
       return ExponentialSmoothing(
           np.asarray(train["Sales"]),
           seasonal_periods=7,
           trend="add",
           seasonal="add",
       ).fit()

   # `save` a given classifier and retrieve coresponding tag:
   tag = bentoml.statsmodels.save("holtwinters", HoltWintersResults())

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   model = bentoml.statsmodels.load("holtwinters:latest")

   # Run a given model under `Runner` abstraction with `load_runner`
   input_data = pd.from_csv("/path/to/csv")
   runner = bentoml.statsmodels.load_runner(tag)
   runner.run(pd.DataFrame("/path/to/csv"))

.. note::

   You can find more examples for **statsmodels** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.statsmodels

.. autofunction:: bentoml.statsmodels.save

.. autofunction:: bentoml.statsmodels.load

.. autofunction:: bentoml.statsmodels.load_runner
