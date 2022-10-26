# MNIST with custom model runner

This example showcases how one can extend BentoML's provided runner and build a custom Runner. See [our documentation][#custom-runner] on Runners.

This example will also demonstrate how one can create custom metrics to monitor model's performance.
We will provide two Prometheus configs as well as a Grafana dashboard for demonstration.

### Requirements

Install requirements with:

```bash
pip install -r requirements.txt
```

### Instruction

1. Train and save model:

```bash
python -m train
```

2. Start a development server:

```bash
bentoml serve
```

3. Download test data

```bash
bazel run //:download_mnist_data
```

4. Send test requests

```bash
curl -F 'image=@mnist_png/testing/8/1007.png' http://127.0.0.1:3000/predict
```

5. Load testing

Start production server:

```bash
bentoml serve --production
```

From another terminal:

```bash
pip install locust
locust -H http://0.0.0.0:3000
```

[#custom-runner]: https://docs.bentoml.org/en/latest/concepts/runner.html#custom-runner
