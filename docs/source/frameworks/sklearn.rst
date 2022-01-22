Scikit-Learn
------------

| Simple and efficient tools for predictive data analysis.
| Accessible to everybody, and reusable in various contexts.
| Built on NumPy, SciPy, and matplotlib.
| Open source, commercially usable - BSD license. - `Source <https://scikit-learn.org/stable/>`_

Users can now use scikit-learn with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml

   from sklearn.datasets import load_iris
   from sklearn.neighbors import KNeighborsClassifier

   model = KNeighborsClassifier()
   iris = load_iris()
   X = iris.data[:, :4]
   Y = iris.target
   model.fit(X, Y)

   # `save` a given classifier and retrieve coresponding tag:
   tag = bentoml.sklearn.save('kneighbors', model)

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # load the model back:
   loaded = bentoml.sklearn.load("kneighbors:latest")

   # Run a given model under `Runner` abstraction with `load_runner`
   runner = bentoml.sklearn.load_runner(tag)
   runner.run([[1,2,3,4,5]])

.. note::
   You can find more examples for **scikit-learn** in our `gallery <https://github.com/bentoml/gallery>`_ repo.


.. currentmodule:: bentoml.sklearn

.. autofunction:: bentoml.sklearn.save

.. autofunction:: bentoml.sklearn.load

.. autofunction:: bentoml.sklearn.load_runner