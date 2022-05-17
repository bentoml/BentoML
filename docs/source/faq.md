FAQ
===
Frequently asked questions from the [BentoML slack community](https://l.linklyhq.com/l/ktOX).

## Does BentoML support horizontal auto-scaling?

Yes! The recommended way of running large scale BentoML workloads, is to do it with 
Yatai on Kubernetes. Yatai can create deployment of BentoML Service with powerful 
autoscaling, and it will scale resources required for each Runner/Model separately based
on workloads and resource consumption.

Many other deployment options that BentoML provide also comes with certain level of 
auto-scaling capability. For example, you can use `bentoctl` to deploy Bentos to AWS EC2
with an auto-scaling group automatically setup for your deployment.

## How does BentoML compare to Tensorflow-serving or Triton inference server?

Model-server projects like Tensorflow-serving, TorchServe, and Triton typically turns a 
saved model file into a Tensor based endpoint for running model inference. BentoML 
serves not just the model, it allows users to define an extended serving pipeline, 
including additional feature extraction code, business logic code, and compose multiple 
models to run in parallel or sequential.

BentoML's Runner component is essentially an equivalent to a typical model-server. We 
are building a Runner adapter which allow users to utilize Triton and Onnx-runtime as
a drop-in replacement for the BentoML Runner, for running the model inference part of
your serving pipeline.

## How does BentoML compare to AWS SageMaker?

BentoML makes deploying models to SageMaker easier! Try deploying your model with 
BentoML and bentoctl to SageMaker: https://docs.bentoml.org/en/latest/deployment/aws_sagemaker.html
Model deployment on SageMaker natively requires users to build their API server with 
Flask/FastAPI and containerize the flask app by themselves, when not using the built-in 
algorithms. BentoML provides a high-performance API server for users without the need 
for lower-level web server development.

## How does BentoML model store compare to MLFlow model registry?

MLFlow model registry is designed for capturing models created from experimentation 
stage, comparing models trained with different parameters or architectures. It is meant 
for tracking models created from the development stage and for data scientists to find 
the best model.

Whereas BentoML model store is commonly considered for "finalized models", containing 
models from your periodic training pipelines running in production. It is designed for 
managing artifacts for build, test, and deployment.

BentoML natively integrates with MLFlow. Users can easily port over models from MLFlow
model registry to BentoML models store for model serving, using the 
`bentoml.mlflow.import_from_uri` API.


## Does BentoML provide model monitoring and drift detection?

BentoML does not provide a built-in model monitoring solution, but it is quite easy to
setup BentoML with a model monitoring solution provider, in the service definition code.

Since most monitoring solutions requires sending model input and output data over an 
HTTP request, it is recommneded to define an async endpoint in this type of use cases.
Here's an example:

```python
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def predict(input_series: np.ndarray) -> np.ndarray:
    result = await my_model_runer.predict.async_run(input_series)
    await async_post_data_for_monitoring(input_series, result)
    return result
```

## Does BentoML support Kubeflow?

Yes, BentoML is designed to be cloud-native and integrate nicely with other tools in the
ecosystem. Refer to the [Kubeflow doc](https://www.kubeflow.org/docs/external-add-ons/serving/bentoml/)
on model serving with BentoML for more details. Note that this doc was written for 
BentoML version 0.13, an update is coming soon for BentoML 1.0 version.


## Who is behind this project?

We are a startup company(bentoml.com) based in San Francisco, California.
Similar to many other open source companies in the enterprise infrastructure space(
Docker, MongoDB, Databricks, etc), we build the open source framework for the community 
for free, and provide a fully managed service for our customers who don't want to
get into the trouble of hosting, managing and securing our open source products 
themselves.

Our commercial product is still under private beta, contact us at contact@bentoml.com if 
you are interested in learning more.


Something Missing?
------------------

If something is missing in the documentation or if you found some part confusing, please 
file an issue [here](https://github.com/bentoml/BentoML/issues/new/choose) with your 
suggestions for improvement, or tweet at the [@bentomlai](http://twitter.com/bentomlai)
account. We love hearing from you!

