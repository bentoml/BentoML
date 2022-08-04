"""Trains and evaluate a simple MLP
on the Reuters newswire topic classification task.
"""
# pylint: disable=no-name-in-module
import numpy as np

# The following import and function call are the only additions to code required
# to automatically log metrics and parameters to MLflow.
import mlflow.keras
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.text import Tokenizer

import bentoml

mlflow.keras.autolog()

max_words = 1000
batch_size = 32
epochs = 5

print("Loading data...")
(x_train, y_train), (x_test, y_test) = reuters.load_data(
    num_words=max_words, test_split=0.2
)

print(len(x_train), "train sequences")
print(len(x_test), "test sequences")

num_classes = np.max(y_train) + 1
print(num_classes, "classes")

print("Vectorizing sequence data...")
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode="binary")
x_test = tokenizer.sequences_to_matrix(x_test, mode="binary")
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

print(
    "Convert class vector to binary class matrix (for use with categorical_crossentropy)"
)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("Building model...")
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_split=0.1,
)
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
print("Test score:", score[0])
print("Test accuracy:", score[1])

# Find URI to the logged MLFlow model
run_id = mlflow.last_active_run().info.run_id
artifact_path = "model"
model_uri = f"runs:/{run_id}/{artifact_path}"

# Option 1: directly save trained model to BentoML
bento_model_1 = bentoml.keras.save_model("keras_native", model)
print("\nBentoML: model imported as native keras model: %s" % bento_model_1)

# Option 2: Import logged MLFlow model to BentoML
bento_model_2 = bentoml.mlflow.import_model("mlflow_keras", model_uri)
print("\nBentoML: model imported as MLFlow pyfunc model: %s" % bento_model_2)

# Option 3: loaded keras model from MLFlow artifact and save with bentoml.keras natively
loaded_keras_model = mlflow.keras.load_model(model_uri)
bento_model_3 = bentoml.keras.save_model("keras_native", loaded_keras_model)
print(
    "\nBentoML: import native keras model loaded from MLflow artifact: %s"
    % bento_model_3
)

# Test loading model from BentoML model store:
for bento_model in [
    bentoml.keras.get(bento_model_1.tag),
    bentoml.mlflow.get(bento_model_2.tag),
    bentoml.keras.get(bento_model_3.tag),
]:
    test_runner = bento_model.to_runner()
    test_runner.init_local()
    assert np.allclose(test_runner.predict.run(x_test), model.predict(x_test))
