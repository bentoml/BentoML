============
Scikit-Learn
============

Below is a simple example of using scikit-learn with BentoML:

.. code:: python

    import bentoml

    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier()
    iris = load_iris()
    X = iris.data[:, :4]
    Y = iris.target
    model.fit(X, Y)

    # `save` a given classifier and retrieve coresponding tag:
    bento_model = bentoml.sklearn.save_model('kneighbors', model)

    # retrieve metadata with `bentoml.models.get`:
    metadata = bentoml.models.get(bento_model.tag)

    # load the model back:
    loaded = bentoml.sklearn.load_model("kneighbors:latest")

    # Run a given model under `Runner` abstraction with `to_runner`
    runner = bentoml.sklearn.get(bento_model.tag).to_runner()
    runner.init_local()
    runner.run([[1,2,3,4]])

.. note::

   You can find more examples for **scikit-learn** in our :examples:`bentoml/examples <>` directory.
