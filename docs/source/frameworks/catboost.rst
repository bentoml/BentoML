CatBoost
--------

Users can now use CatBoost with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml
   import catboost as cbt

   from sklearn.datasets import load_breast_cancer

   cancer = load_breast_cancer()

   X = cancer.data
   y = cancer.target

   clf = cbt.CatBoostClassifier(
       iterations=2,
       depth=2,
       learning_rate=1,
       loss_function="Logloss",
       verbose=False,
   )

   # train the model
   clf.fit(X, y)

   # `save` a given classifier and retrieve coresponding tag:
   tag = bentoml.catboost.save("cancer_clf", clf)

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   model = bentoml.catboost.load("cancer_clf:latest")

   # Run a given model under `Runner` abstraction with `load_runner`
   input_data = pd.from_csv("/path/to/csv")
   runner = bentoml.catboost.load_runner("cancer_clf:latest")
   runner.run(cbt.Pool(input_data))

.. note::

   You can find more examples for **CatBoost** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.catboost

.. autofunction:: bentoml.catboost.save

.. autofunction:: bentoml.catboost.load

.. autofunction:: bentoml.catboost.load_runner
