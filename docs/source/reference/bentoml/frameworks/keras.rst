=====
Keras
=====

.. admonition:: About this page

   This is an API reference for Keras in BentoML. Please refer to
   :ref:`Keras guide <frameworks/keras:Keras>` for more information about how to use
   Keras in BentoML.

.. note::

   ``bentoml.keras`` supports both Keras 2 (TensorFlow backend) and Keras 3
   multi-backend models (TensorFlow, PyTorch, JAX). The active backend is
   detected from ``keras.config.backend()`` at save and runtime. Only the
   TensorFlow backend supports ``tf_signatures`` and ``tf_save_options``.

.. currentmodule:: bentoml.keras

.. autofunction:: bentoml.keras.save_model

.. autofunction:: bentoml.keras.load_model

.. autofunction:: bentoml.keras.get
