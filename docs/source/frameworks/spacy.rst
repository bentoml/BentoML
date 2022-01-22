SpaCy
-----

| Industrial-Strength Natural Language Processing in Python - `Source <https://spacy.io/>`_

Users can now use SpaCy with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml
   import spacy as cbt

   from sklearn.datasets import load_breast_cancer

   cancer = load_breast_cancer()

   X = cancer.data
   y = cancer.target

   clf = cbt.SpaCyClassifier(
       iterations=2,
       depth=2,
       learning_rate=1,
       loss_function="Logloss",
       verbose=False,
   )

   # train the model
   clf.fit(X, y)

   # `save` a given classifier and retrieve coresponding tag:
   tag = bentoml.spacy.save("cancer_clf", clf)

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   model = bentoml.spacy.load("cancer_clf:latest")

   # Run a given model under `Runner` abstraction with `load_runner`
   input_data = pd.from_csv("/path/to/csv")
   runner = bentoml.spacy.load_runner("cancer_clf:latest")
   runner.run(cbt.Pool(input_data))

.. note::
   You can find more examples for **SpaCy** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.spacy

.. autofunction:: bentoml.spacy.save

.. autofunction:: bentoml.spacy.load

.. autofunction:: bentoml.spacy.projects

.. autofunction:: bentoml.spacy.load_runner