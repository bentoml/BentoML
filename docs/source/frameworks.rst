.. _frameworks-page:

Frameworks
==========

Here are the all of the supported ML frameworks for BentoML. You can find the official
BentoML example projects in the `bentoml/gallery <https://github.com/bentoml/gallery>`__
repository, group by the ML training frameworks used in the project.

You can download the examples below and run them on your computer. Links to run them on
Google Colab are also available, although some of the features demoed in the notebooks
does not work in the Colab environment due to its limitations, including running the
BentoML API model server, building docker image or creating cloud deployment.


============
Scikit-Learn
============

Example Projects:

* Sentiment Analysis - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/scikit-learn/sentiment-analysis/sklearn-sentiment-analysis.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/scikit-learn/sentiment-analysis/sklearn-sentiment-analysis.ipynb>`__

.. autoclass:: bentoml.frameworks.sklearn.SklearnModelArtifact

=======
PyTorch
=======

Example Projects:

* Fashion MNIST - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/pytorch/fashion-mnist/pytorch-fashion-mnist.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/pytorch/fashion-mnist/pytorch-fashion-mnist.ipynb>`__
* CIFAR-10 Image Classification - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/pytorch/cifar10-image-classification/pytorch-cifar10-image-classification.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/pytorch/cifar10-image-classification/pytorch-cifar10-image-classification.ipynb>`__

.. autoclass:: bentoml.frameworks.pytorch.PytorchModelArtifact


===========================
Tensorflow 2.0 (Native API)
===========================

Example Projects:

* tf.Function model - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb>`__
* Fashion MNIST - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb>`__
* Movie Review Sentiment with BERT - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/bert/bert_movie_reviews.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/tensorflow/bert/bert_movie_reviews.ipynb>`__

.. autoclass:: bentoml.frameworks.tensorflow.TensorflowSavedModelArtifact


=====================================
Keras (Tensorflow 2.0 as the backend)
=====================================

Example Projects:

* Fashion MNIST - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/legacy-keras/fashion-mnist/keras-fashion-mnist.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/legacy-keras/fashion-mnist/keras-fashion-mnist.ipynb>`__
* Text Classification - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/legacy-keras/text-classification/keras-text-classification.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/legacy-keras/text-classification/keras-text-classification.ipynb>`__
* Toxic Comment Classifier - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/legacy-keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/legacy-keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb>`__

.. autoclass:: bentoml.frameworks.keras.KerasModelArtifact


==============
Tensorflow 1.0
==============

Example Projects:

* Fashion MNIST - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_1_fashion_mnist.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_1_fashion_mnist.ipynb>`__

.. autoclass:: bentoml.frameworks.tensorflow.TensorflowSavedModelArtifact
    :noindex:


=========
FastAI v1
=========

Example Projects:

* Pet Image Classification - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/pet-image-classification/fast-ai-pet-image-classification.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/fast-ai/pet-image-classification/fast-ai-pet-image-classification.ipynb>`__
* Salary Range Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/salary-range-prediction/fast-ai-salary-range-prediction.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/fast-ai/salary-range-prediction/fast-ai-salary-range-prediction.ipynb>`__

.. autoclass:: bentoml.frameworks.fastai.Fastai1ModelArtifact

=========
FastAI v2
=========

Example Projects:

* Medical image classification - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/fastai2_medical/medical_imaging.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/fast-ai/fastai2_medical/medical_imaging.ipynb>`__

.. autoclass:: bentoml.frameworks.fastai.FastaiModelArtifact

=======
XGBoost
=======

Example Projects:

* Titanic Survival Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb>`__
* League of Legend win Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb>`__

.. autoclass:: bentoml.frameworks.xgboost.XgboostModelArtifact

========
LightGBM
========

Example Projects:

* Titanic Survival Prediction -  `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb>`__

.. autoclass:: bentoml.frameworks.lightgbm.LightGBMModelArtifact


========
FastText
========

Example Projects:

* Text Classification - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/fasttext/text-classification/text-classification.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/fast-text/text-classification/text-classification.ipynb>`__

.. autoclass:: bentoml.frameworks.fasttext.FasttextModelArtifact

===
H2O
===

Example Projects:

* Loan Default Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb>`__
* Prostate Cancer Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb>`__

.. autoclass:: bentoml.frameworks.h2o.H2oModelArtifact


======
CoreML
======

.. autoclass:: bentoml.frameworks.coreml.CoreMLModelArtifact


====
ONNX
====

Example Projects:

* Image Classification with ResNet50 - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/onnx/resnet50/resnet50.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/onnx/resnet50/resnet50.ipynb>`__

.. autoclass:: bentoml.frameworks.onnx.OnnxModelArtifact


=========
ONNX-MLIR
=========

.. autoclass:: bentoml.frameworks.onnxmlir.OnnxMlirModelArtifact


=====
Spacy
=====

.. autoclass:: bentoml.frameworks.spacy.SpacyModelArtifact


============
Transformers
============

.. autoclass:: bentoml.frameworks.transformers.TransformersModelArtifact


===========
Statsmodels
===========

For statsmodels, we recommend using PickleArtifact:

.. autoclass:: bentoml.service.artifacts.common.PickleArtifact
    :noindex:

Example Projects:

* Shampoo Sales Prediction -  `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/statsmodels_holt/bentoml_statsmodels.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/statsmodels_holt/bentoml_statsmodels.ipynb>`__


=====
Gluon
=====

.. autoclass:: bentoml.frameworks.gluon.GluonModelArtifact


=================
Pytorch Lightning
=================

.. autoclass:: bentoml.frameworks.pytorch.PytorchLightningModelArtifact


=========
Detectron
=========

.. autoclass:: bentoml.frameworks.detectron.DetectronModelArtifact


======
Paddle
======

Example Projects:

* Boston Housing Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/paddlepaddle/LinearRegression/LinearRegression.ipynb>`__ / `Notebook Source <https://github.com/bentoml/gallery/blob/master/paddlepaddle/LinearRegression/LinearRegression.ipynb>`__

.. autoclass:: bentoml.frameworks.paddle.PaddlePaddleModelArtifact

========
EasyOCR
========

.. autoclass:: bentoml.frameworks.easyocr.EasyOCRArtifact

======
EvalML
======

.. autoclass:: bentoml.frameworks.evalml.EvalMLModelArtifact


.. spelling::

    MLModel
    tokenizer
    AutoModelWithLMHead
    nn
    fasttext
    onnx
    Statsmodels
    mxnet
    env
    GluonModelArtifact
    JsonInput
    nd
    mx
    asnumpy
    svc
    Pytorch
    pytorch
    detectron
    Detectron
    DetectionCheckpointer
    evalml
    Evalml
    EvalML
    EasyOCR
    easyocr
    mlir
    operationalized
    pyruntime
    llvm
    pybind