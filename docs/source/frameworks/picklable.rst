===============
Picklable Model
===============

Here's an example of saving any Python object or function as model, and create a runner
instance:

.. code:: python

    import bentoml

    class MyPicklableModel:
        def predict(self, some_integer: int):
            return some_integer ** 2

    # `save` a given model or function
    model = MyPicklableModel()
    tag = bentoml.picklable_model.save_model(
        'mypicklablemodel',
        model,
        signatures={"predict": {"batchable": False}}
    )

    # retrieve metadata with `bentoml.models.get`:
    metadata = bentoml.picklable_model.get(tag)

    # load the model back:
    loaded = bentoml.picklable_model.load_model("mypicklablemodel:latest")

    # Run a given model under `Runner` abstraction with `load_runner`
    runner = bentoml.picklable_model.get(tag).to_runner()
    runner.init_local()
    runner.predict.run(7)


.. currentmodule:: bentoml.picklable_model

.. autofunction:: bentoml.picklable_model.save_model

.. autofunction:: bentoml.picklable_model.load_model

.. autofunction:: bentoml.picklable_model.get
