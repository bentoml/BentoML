# Fraud Detection Service Example

1. Install dependencies:
```bash
pip install -r ./dev-requirements.txt
```

2. Download dataset from Kaggle

Before downloading, set up Kaggle API Credentials https://github.com/Kaggle/kaggle-api#api-credentials 
and accept dataset rules: https://www.kaggle.com/competitions/ieee-fraud-detection/data

```bash
./download_data.sh
```

3. Train the fraud detection xgboost model. For details, see the ./IEEE-CIS-Fraud-Detection.ipynb
   notebook:
```bash
./train.sh
```

This creates 3 variations of the model:

```bash
$ bentoml models list

Tag                                         Module           Size        Creation Time
ieee-fraud-detection-tiny:qli6n3f6jcta3uqj  bentoml.xgboost  141.40 KiB  2023-03-08 23:03:36
ieee-fraud-detection-lg:o7wqb5f6jcta3uqj    bentoml.xgboost  18.07 MiB   2023-03-08 23:03:17
ieee-fraud-detection-sm:5yblgmf6i2ta3uqj    bentoml.xgboost  723.00 KiB  2023-03-08 22:52:16
```


4. Run the ML Service locally:
```bash
bentoml serve
```

5. Send test requests:

Visit http://localhost:3000/ in a browser and send test requests via the UI.

Alternatively, send test payloads via CLI:

```bash
head --lines=200 ./data/test_transaction.csv | curl -X POST -H 'Content-Type: text/csv' --data-binary @- http://0.0.0.0:3000/is_fraud
```

6. Build a docker image for deployment

Build a Bento to lock the model version and dependency tree:
```bash
bentoml build
```

Ensure docker is installed and running, build a docker image with `bentoml containerize`
```bash
bentoml containerize fraud_detection:latest
```

Test out the docker image built:

```bash
docker run -it --rm -p 3000:3000 fraud_detection:{YOUR BENTO VERSION}
```

7. Optional: use GPU for model inference

Use `bentofile-gpu.yaml` to build a new Bento, which adds the following two lines to the YAML.
This ensures the docker image comes with GPU libraries installed and BentoML will automatically
load models on GPU when running the docker image with GPU devices available.

```yaml
docker:
  cuda_version: "11.6.2"
```

Build Bento with GPU support:
```bash
bentoml build -f ./bentofile-gpu.yaml
```

Build docker image with GPU support:
```bash
bentoml containerize fraud_detection:latest

docker run --gpus all --device /dev/nvidia0 \
           --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools \
           --device /dev/nvidia-modeset --device /dev/nvidiactl \
           fraud_detection:{YOUR BENTO VERSION}
```

8. Optional: multi-model inference graph

BentoML makes it efficient to create ML service with multiple ML models. Users can choose to run
models sequentially or in parallel using the Python AsyncIO APIs along with Runners APIs. This makes
it possible create inference graphes or multi-stage inference pipeline all from Python APIs.

Here's an example that runs all three models simutaneously and aggregate their results:

```bash
bentoml serve inference_graph_service:svc

bentoml build -f ./inference_graph_bentofile.yaml
```

Sample code available in the `./inference_graph_service.py` file.
