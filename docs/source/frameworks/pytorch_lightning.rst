=================
PyTorch Lightning
=================

Here's a simple example of using PyTorch Lightning with BentoML:

.. code:: python

    import bentoml
    import torch
    import pytorch_lightning as pl

    class AdditionModel(pl.LightningModule):
        def forward(self, inputs):
            return inputs.add(1)

    # `save` a given classifier and retrieve coresponding tag:
    tag = bentoml.pytorch_lightning.save_model("addition_model", AdditionModel())

    # retrieve metadata with `bentoml.models.get`:
    metadata = bentoml.models.get(tag)

    # `load` the model back in memory:
    model = bentoml.pytorch_lightning.load_model("addition_model:latest")

    # Run a given model under `Runner` abstraction with `to_runner`
    runner = bentoml.pytorch_lightning.get(tag).to_runner()
    runner.init_local()
    runner.run(torch.from_numpy(np.array([[1,2,3,4]])))

.. note::

   You can find more examples for **PyTorch Lightning** in our :github:`bentoml/examples <tree/main/examples>` directory.


.. currentmodule:: bentoml.pytorch_lightning

.. autofunction:: bentoml.pytorch_lightning.save_model

.. autofunction:: bentoml.pytorch_lightning.load_model

.. autofunction:: bentoml.pytorch_lightning.get

