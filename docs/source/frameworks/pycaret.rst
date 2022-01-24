PyCaret
-------

| PyCaret is an open source, low-code machine learning library in
| Python that allows you to go from preparing your data to deploying your 
| model within minutes in your choice of notebook environment. - `Source <https://pycaret.org/>`_

Users can now use PyCaret with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml
   import pandas as pd

   from pycaret.datasets import get_data
   from pycaret.classification import setup as pycaret_setup
   from pycaret.classification import save_model
   from pycaret.classification import tune_model
   from pycaret.classification import create_model
   from pycaret.classification import predict_model
   from pycaret.classification import finalize_model

   dataset = get_data("credit")
   data = dataset.sample(frac=0.95, random_state=786)
   data_unseen = dataset.drop(data.index)
   data.reset_index(inplace=True, drop=True)
   data_unseen.reset_index(inplace=True, drop=True)

   pycaret_setup(data=data, target="default", session_id=123, silent=True)
   dt = create_model("dt")
   tuned_dt = tune_model(dt)
   final_dt = finalize_model(tuned_dt)

   # `save` a given classifier and retrieve coresponding tag:
   tag = bentoml.pycaret.save("cls", final_dt)

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   model = bentoml.pycaret.load("cls:latest")

   # Run a given model under `Runner` abstraction with `load_runner`
   input_data = pd.from_csv("/path/to/csv")
   runner = bentoml.pycaret.load_runner(tag)
   runner.run(pd.DataFrame("/path/to/csv"))

.. admonition:: btw
   :class: customNotesFmt

   You can find more examples for **PyCaret** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.pycaret

.. autofunction:: bentoml.pycaret.save

.. autofunction:: bentoml.pycaret.load

.. autofunction:: bentoml.pycaret.load_runner
