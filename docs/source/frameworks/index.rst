=========================
Framework-specific Guides
=========================

Here are the all of the supported ML frameworks for BentoML. You can find the official
BentoML example projects in the `bentoml/gallery <https://github.com/bentoml/gallery>`__
repository.


.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/frameworks/keras`
        :link: /frameworks/keras
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/picklable`
        :link: /frameworks/picklable
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/pytorch`
        :link: /frameworks/pytorch
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/pytorch_lightning`
        :link: /frameworks/pytorch_lightning
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/sklearn`
        :link: /frameworks/sklearn
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/tensorflow`
        :link: /frameworks/tensorflow
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/transformers`
        :link: /frameworks/transformers
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/xgboost`
        :link: /frameworks/xgboost
        :link-type: doc


The following frameworks are supported in BentoML prior to 1.0.0 release and is still
being migrated to the new API design. Before they are officially supported in BentoML
1.0, users may use :ref:`Custom Runner <concepts/runner:Custom Runner>` to serve models
from these ML libraries with BentoML.

- Catboost
- Detectron
- EasyOCR
- EvalML
- FastText
- FastAI
- Gluon
- H2O
- LightGBM
- MLFlow
- ONNX
- ONNX-MLIR
- PaddlePaddle
- PyCaret
- Spacy
- Statsmodels
- Flax
- Jax
- Neuropod
- PyTorch ignite
- Spark MLlib
- SnapML



.. admonition:: Help us improve the project!

    Found an issue or a TODO item? You're always welcome to make contributions to the
    project and its documentation. Check out the
    `BentoML development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_
    and `documentation guide <https://github.com/bentoml/BentoML/blob/main/docs/README.md>`_
    to get started.


.. toctree::
  :hidden:

  picklable_model
  pytorch
  pytorch_lightning
  keras
  sklearn
  tensorflow
  transformers
  xgboost

