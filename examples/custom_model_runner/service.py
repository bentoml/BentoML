import numpy as np
import bentoml

mnist_model = bentoml.pytorch.get("mnist_cnn:latest")
mnist_runnable = mnist_model.to_runnable()

class CustomMnistRunnable(mnist_runnable):
    def __init__(self):
        super().__init__()
        import torch
        print("Running on device:", self.device_id)
        print("Running on torch version:", torch.__version__)

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def __call__(self, input_array):
        output = super().__call__(input_array)
        return output.argmax(dim=1)

mnist_runner = bentoml.Runner(CustomMnistRunnable, method_configs={
    '__call__': {
        'max_batch_size': 50, 'max_latency_ms': 600
    }
})

svc = bentoml.Service("pytorch_mnist_demo", runners=[mnist_runner], models=[mnist_model])

@svc.api(input=bentoml.io.Image(), output=bentoml.io.NumpyNdarray())
def predict(input_img):
    input_arr = np.array(input_img).reshape([-1, 1, 28, 28])
    return mnist_runner.run(input_arr)[0]
