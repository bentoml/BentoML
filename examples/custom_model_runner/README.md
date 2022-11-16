# Custom PyTorch MNIST runner

This example showcases how one can extend BentoML's provided runner and build a custom Runner. See [our documentation][#custom-runner] on Runners.

This example will also demonstrate how one can create custom metrics to monitor the model's performance.
We will provide two Prometheus configs to use for either HTTP or gRPC BentoServer for demonstration.

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

2. Download test data

```bash
wget -qO- https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz | tar xz
```

3. Start a development server:

<table>
<tr>
<td> Protocol </td> <td> Command </td>
</tr>
<tr>
<td> <code>HTTP</code> </td>
<td>

```bash
bentoml serve-http service.py:svc
```

</td>
</tr>
<tr>
<td> <code>gRPC</code> </td>
<td>

```bash
bentoml serve-grpc service.py:svc
```

</td>
</tr>
</table>

4. Send test requests

<table>
<tr>
<td> Protocol </td> <td> Command </td>
</tr>
<tr>
<td> <code>HTTP</code> </td>
<td>

```bash
curl -F 'image=@mnist_png/testing/8/1007.png' http://127.0.0.1:3000/predict
```

</td>
</tr>
<tr>
<td> <code>gRPC</code> </td>
<td>

```bash
grpcurl -d @ -plaintext 0.0.0.0:3000 bentoml.grpc.v1.BentoService/Call <<EOM
{
  "apiName": "classify",
  "file": {
    "content": "..."  # bytes from a one of the testdata.
  }
}
EOM
```
</table>

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
