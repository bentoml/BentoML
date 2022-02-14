Picklable Model
---------------

Users can now save any given python method or object as a loadable model in BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner`:

.. code-block:: python

   import bentoml

    class MyPicklableModel:
        def predict(self, some_integer: int):
            return some_integer ** 2

   # `save` a given model or function
   model = MyPicklableModel()
   tag = bentoml.picklable_model.save('mypicklablemodel', model, batch=False, method="predict")

   # Or load save a model which takes in a batch of values to take advantage of BentoML's adaptive batching
   # class MyPicklableModelBatch:
   #         def predict(self, some_integers: t.List[int]):
   #             return list(map(lambda x: x ** 2, some_integers))
   #
   # model = MyPicklableModel()
   # tag = bentoml.picklable_model.save('mypicklablemodel', model, batch=True, method="predict")


   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # load the model back:
   loaded = bentoml.picklable_model.load("mypicklablemodel:latest")

   # Run a given model under `Runner` abstraction with `load_runner`
   runner = bentoml.picklable_model.load_runner(tag)
   runner.run(7)

.. note::

   You can find more examples for **picklable-model** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.picklable_model

.. autofunction:: bentoml.picklable_model.save

.. autofunction:: bentoml.picklable_model.load

.. autofunction:: bentoml.picklable_model.load_runner
