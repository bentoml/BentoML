import numpy as np

import bentoml

# Load the model by specifying the model tag
iris_model = bentoml.mlflow.load_model("iris:74px7hboeo25fjjt")

input_data = np.array([[5.9, 3, 5.1, 1.8]])
res = iris_model.predict(input_data)
print(res)
