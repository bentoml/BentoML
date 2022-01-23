MXNet Gluon
-----------

| MXNet - A flexible and efficient library for deep learning - `Source <https://mxnet.apache.org/versions/1.6/>`_

| The Gluon library in Apache MXNet provides a clear, concise, and simple API for deep learning. It makes it easy to prototype,
| build, and train deep learning models without sacrificing training speed. - `Documentation <https://mxnet.apache.org/versions/1.6/api/python/docs/api/gluon/index.html>`_

Users can now use MXNet Gluon with BentoML with the following API: :code:`load`, :code:`save`, and :code:`load_runner` as follow:

.. code-block:: python

   import mxnet
   import mxnet.gluon as gluon
   import bentoml


   def train_gluon_classifier() -> gluon.nn.HybridSequential:
      net = mxnet.gluon.nn.HybridSequential()
      net.hybridize()
      net.forward(mxnet.nd.array(0))
      return net

   model = train_gluon_classifier()

   # `save` a model to BentoML modelstore:
   tag = bentoml.gluon.save("gluon_block", model)

   # retrieve metadata with `bentoml.models.get`:
   metadata = bentoml.models.get(tag)

   # `load` the model back in memory:
   loaded: gluon.Block = bentoml.gluon.load(tag)

.. admonition:: btw
   :class: customNotesFmt

   You can find more examples for **MXNet Gluon** in our `gallery <https://github.com/bentoml/gallery>`_ repo.

.. currentmodule:: bentoml.gluon

.. autofunction:: bentoml.gluon.save

.. autofunction:: bentoml.gluon.load

.. autofunction:: bentoml.gluon.load_runner