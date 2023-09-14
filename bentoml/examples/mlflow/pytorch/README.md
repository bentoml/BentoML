# MLFlow + PyTorch + BentoML Example

This example is built on https://github.com/mlflow/mlflow/tree/master/examples/pytorch

Train a PyTorch mnist example model, log the model to mlflow and import MLFlow model to BentoML for serving.

```bash
python mnist.py
```

Start the prediction service defined with BentoML in `service.py`, using the imported MLflow model:

```bash
bentoml serve service.py:svc
```

Test out the serving endpoint:

```bash
curl -X POST -H "Content-Type:application/json" \
  -d @test_input.json \
  http://localhost:3000/predict
```

Build Bento and containerize BentoServier for deployment:

```bash
bentoml build

bentoml containerize mlflow_pytorch_mnist_demo:latest
```
