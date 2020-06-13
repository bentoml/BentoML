Frameworks
==========

Here are the all of the supported ML frameworks for BentoML. You can find the official
BentoML example projects in the `bentoml/gallery <https://github.com/bentoml/gallery>`_
repository, group by the ML training frameworks used in the project.

You can download the examples below and run them on your computer. Links to run them on
Google Colab are also available, although some of the features demo'd in the notebooks
does not work in the Colab environment due to its limitations, including running the 
BentoML API model server, building docker image or creating cloud deployment.


============
Scikit-Learn
============

Example Projects:

* Sentiment Analysis - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/scikit-learn/sentiment-analysis/sklearn-sentiment-analysis.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/scikit-learn/sentiment-analysis/sklearn-sentiment-analysis.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/scikit-learn/sentiment-analysis/sklearn-sentiment-analysis.ipynb>`_

.. autoclass:: bentoml.artifact.SklearnModelArtifact

=======
PyTorch
=======

Example Projects:

* Fashion MNIST - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/pytorch/fashion-mnist/pytorch-fashion-mnist.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/pytorch/fashion-mnist/pytorch-fashion-mnist.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/pytorch/fashion-mnist/pytorch-fashion-mnist.ipynb>`_
* CIFAR-10 Image Classification - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/pytorch/cifar10-image-classification/pytorch-cifar10-image-classification.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/pytorch/cifar10-image-classification/pytorch-cifar10-image-classification.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/pytorch/cifar10-image-classification/pytorch-cifar10-image-classification.ipynb>`_

.. autoclass:: bentoml.artifact.PytorchModelArtifact


==============
Tensorflow 2.0
==============

Example Projects:

* tf.Function model - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/tensorflow/echo/tensorflow-echo.ipynb>`_
* Fashion MNIST - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/tensorflow/fashion-mnist/tensorflow_2_fashion_mnist.ipynb>`_
* Movie Review Sentiment with BERT - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/tensorflow/bert/bert_movie_reviews.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/tensorflow/bert/bert_movie_reviews.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/tensorflow/bert/bert_movie_reviews.ipynb>`_

.. autoclass:: bentoml.artifact.TensorflowSavedModelArtifact

================
Tensorflow Keras
================

Example Projects:

* Fashion MNIST - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/keras/fashion-mnist/keras-fashion-mnist.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/keras/fashion-mnist/keras-fashion-mnist.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/keras/fashion-mnist/keras-fashion-mnist.ipynb>`_
* Text Classification - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/keras/text-classification/keras-text-classification.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/keras/text-classification/keras-text-classification.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/keras/text-classification/keras-text-classification.ipynb>`_
* Toxic Comment Classifier - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/keras/toxic-comment-classification/keras-toxic-comment-classification.ipynb>`_

.. autoclass:: bentoml.artifact.KerasModelArtifact

======
FastAI
======

Example Projects:

* Pet Image Classification - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/pet-image-classification/fast-ai-pet-image-classification.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/fast-ai/pet-image-classification/fast-ai-pet-image-classification.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/fast-ai/pet-image-classification/fast-ai-pet-image-classification.ipynb>`_
* Salary Range Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/fast-ai/salary-range-prediction/fast-ai-salary-range-prediction.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/fast-ai/salary-range-prediction/fast-ai-salary-range-prediction.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/fast-ai/salary-range-prediction/fast-ai-salary-range-prediction.ipynb>`_

.. autoclass:: bentoml.artifact.FastaiModelArtifact

.. autoclass:: bentoml.adapters.FastaiImageInput

=======
XGBoost
=======

Example Projects:

* Titanic Survival Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/xgboost/titanic-survival-prediction/xgboost-titanic-survival-prediction.ipynb>`_
* League of Legend win Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/xgboost/league-of-legend-win-prediction/xgboost-league-of-legend-win-prediction.ipynb>`_

.. autoclass:: bentoml.artifact.XgboostModelArtifact

========
LightGBM
========

Example Projects:

* Titanic Survival Prediction -  `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/lightbgm/titanic-survival-prediction/lightbgm-titanic-survival-prediction.ipynb>`_

.. autoclass:: bentoml.artifact.LightGBMModelArtifact


========
FastText
========

Example Projects:

* Text Classification - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/fasttext/text-classification/text-classification.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/fasttext/text-classification/text-classification.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/fast-text/text-classification/text-classification.ipynb>`_

.. autoclass:: bentoml.artifact.FasttextModelArtifact


===
H2O
===

Example Projects:

* Loan Default Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/h2o/loan-prediction/h2o-loan-prediction.ipynb>`_
* Prostate Cancer Prediction - `Google Colab <https://colab.research.google.com/github/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb>`_ | `nbviewer <https://nbviewer.jupyter.org/github/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb>`_ | `source <https://github.com/bentoml/gallery/blob/master/h2o/prostate-cancer-classification/h2o-prostate-cancer-classification.ipynb>`_

.. autoclass:: bentoml.artifact.H2oModelArtifact
