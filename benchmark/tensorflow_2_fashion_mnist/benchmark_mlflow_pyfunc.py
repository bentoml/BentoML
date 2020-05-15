from __future__ import print_function

import os
import pickle

import base64
import pandas as pd
import numpy as np
import six

import tensorflow as tf

import mlflow
import mlflow.pyfunc
import mlflow.pyfunc.model
from mlflow.models import Model


try:
    tf.config.set_visible_devices([], 'GPU')  # disable GPU, required when served in docker
except:
    pass


def _load_pyfunc(path):
    tf_model = tf.saved_model.load(path)
    class Model:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        def predict(self, inputs):
            _bytes = [base64.b64decode(i) for i in inputs['str'].to_numpy().tolist()]
            inputs = tf.constant(_bytes, dtype=tf.string)
            outputs = tf_model.predict_image(inputs)
            output_classes = tf.math.argmax(outputs, axis=1)
            return [self.class_names[i] for i in output_classes]
    return Model()
        


if __name__ == '__main__':
    tmpdir = 'mlflow_tmp'
    tf_model_path = os.path.join(str(tmpdir), "tf.pkl")
    model_path = os.path.join(str(tmpdir), "model")

    model_config = Model(run_id="test")
    mlflow.pyfunc.save_model(path=model_path,
                             data_path=tf_model_path,
                             loader_module=os.path.basename(__file__)[:-3],
                             code_path=[__file__],
                             mlflow_model=model_config)

    reloaded_model = mlflow.pyfunc.load_pyfunc(model_path)
    print(reloaded_model)
