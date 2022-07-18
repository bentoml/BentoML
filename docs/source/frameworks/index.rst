================
Framework Guides
================

Here is the list of ML libraries and formats supported in BentoML. You can find example
projects in the `bentoml/gallery <https://github.com/bentoml/gallery>`__ repository.


.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/frameworks/keras`
        :link: /frameworks/keras
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/onnx`
        :link: /frameworks/onnx
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

    .. grid-item-card:: :doc:`/frameworks/fastai`
        :link: /frameworks/fastai
        :link-type: doc

    .. grid-item-card:: :doc:`/integrations/mlflow`
        :link: /integrations/mlflow
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/catboost`
        :link: /frameworks/catboost
        :link-type: doc

    .. grid-item-card:: :doc:`/frameworks/lightgbm`
        :link: /frameworks/lightgbm
        :link-type: doc


Custom Models
-------------

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/frameworks/picklable`
        :link: /frameworks/picklable
        :link-type: doc

    .. grid-item-card:: :ref:`concepts/runner:Custom Runner`
        :link: /concepts/runner
        :link-type: doc


.. TODO:
    .. grid-item-card:: :doc:`/guides/custom_ml_framework`
        :link: /guides/custom_ml_framework
        :link-type: doc


Roadmap
-------

The following frameworks are supported in pre-1.0 BentoML versions and they are
still being migrated to the new API in 1.0. Before they are officially supported,
users may use :ref:`Custom Models <frameworks/index:Custom Models>` as an workaround.

- Detectron
- EasyOCR
- EvalML
- FastText
- Flax
- Gluon
- H2O
- Jax
- Neuropod
- ONNX-MLIR
- PaddlePaddle
- PyCaret
- PyTorch ignite
- SnapML
- Spacy
- Spark MLlib
- Statsmodels


.. admonition:: Help us improve the project!

    Found an issue or a TODO item? You're always welcome to make contributions to the
    project and its documentation. Check out the
    `BentoML development guide <https://github.com/bentoml/BentoML/blob/main/DEVELOPMENT.md>`_
    and `documentation guide <https://github.com/bentoml/BentoML/blob/main/docs/README.md>`_
    to get started.


.. toctree::
    :hidden:

    catboost
    fastai
    keras
    onnx
    picklable
    pytorch
    pytorch_lightning
    sklearn
    tensorflow
    transformers
    xgboost
