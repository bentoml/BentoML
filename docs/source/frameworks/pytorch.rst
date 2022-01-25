PyTorch
-------

Users can now use PyTorch with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import pandas as pd
   import torch
   import bentoml

   class NGramLanguageModeler(nn.Module):

      def __init__(self, vocab_size, embedding_dim, context_size):
            super(NGramLanguageModeler, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear1 = nn.Linear(context_size * embedding_dim, 128)
            self.linear2 = nn.Linear(128, vocab_size)

      def forward(self, inputs):
            embeds = self.embeddings(inputs).view((1, -1))
            out = F.relu(self.linear1(embeds))
            out = self.linear2(out)
            log_probs = F.log_softmax(out, dim=1)
            return log_probs

   # `save` a given classifier and retrieve coresponding tag:
   tag = bentoml.pytorch.save("ngrams", NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE))

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   model = bentoml.pytorch.load("ngrams:latest")

   # Run a given model under `Runner` abstraction with `load_runner`
   input_data = pd.from_csv("/path/to/csv")
   runner = bentoml.pytorch.load_runner(tag)
   runner.run(pd.DataFrame("/path/to/csv"))

.. note::

   You can find more examples for **PyTorch** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.pytorch

.. autofunction:: bentoml.pytorch.save

.. autofunction:: bentoml.pytorch.load

.. autofunction:: bentoml.pytorch.load_runner
