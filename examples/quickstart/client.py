import bentoml
import numpy as np

input_series=[
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.9, 4.3, 1.3],
    [5.9, 3.0, 5.1, 1.8],
    [4.6, 3.1, 1.5, 0.2],
    [6.7, 3.1, 4.4, 1.4],
    [5.5, 2.6, 4.4, 1.2],
    [7.7, 3.0, 6.1, 2.3],
    [4.9, 3.0, 1.4, 0.2]
]

with bentoml.SyncHTTPClient('http://localhost:3000') as client:
    pred: np.ndarray = client.classify(input_series)
    print(pred)
