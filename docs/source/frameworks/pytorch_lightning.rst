PyTorch Lightning
-----------------

| You do the research. Lightning will do everything else.
| The ultimate PyTorch research framework. Scale your models, without the boilerplate. - `Source <https://www.pytorchlightning.ai/>`_

Users can now use PyTorch Lightning with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import bentoml
   import torch
   import pytorch_lightning as pl

   class AdditionModel(pl.LightningModule):
      def forward(self, inputs):
         return inputs.add(1)

   # `save` a given classifier and retrieve coresponding tag:
   tag = bentoml.pytorch_lightning.save("addition_model", AdditionModel())

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   model = bentoml.pytorch_lightning.load("addition_model:latest")

   # Run a given model under `Runner` abstraction with `load_runner`
   runner = bentoml.pytorch_lightning.load_runner(tag)
   runner.run_batch(torch.from_numpy(np.array([[1,2,3,4]])))

.. admonition:: btw
   :class: customNotesFmt

   You can find more examples for **PyTorch Lightning** in our `gallery <https://github.com/bentoml/gallery>`_ repo.


.. currentmodule:: bentoml.pytorch_lightning

.. autofunction:: bentoml.pytorch_lightning.save

.. autofunction:: bentoml.pytorch_lightning.load

.. autofunction:: bentoml.pytorch_lightning.load_runner