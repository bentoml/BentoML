# Custom Runner with pre-trained NLTK model

For ML libraries that provide built-in trained models, such as NLTK, users may create a
custom Runner directly without saving the model to the model store.

This example showcases how one can create a custom Runner directly without saving models
to model store for frameworks that provide buil-tin trained models. See [our documentation][#custom-runner] on Runners.

This example will also demonstrate how one can create custom metrics to monitor the model's performance.
We will provide two Prometheus configs to use for either HTTP or gRPC BentoServer for demonstration.

### Requirements

Install requirements with:

```bash
pip install -r requirements.txt
```

### Instruction

1. Download required NLTK pre-trained models:

```bash
python -m download_nltk_models
```

2. Run the service:

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

3. Send in test request:

<table>
<tr>
<td> Protocol </td> <td> Command </td>
</tr>
<tr>
<td> <code>HTTP</code> </td>
<td>

```bash
curl -X POST -H "content-type: application/text" --data "BentoML is great" http://127.0.0.1:3000/analysis
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
  "text": "BentoML is great"
}
EOM
```

4. Build Bento

```
bentoml build
```

5. Build docker image

```
bentoml containerize sentiment_analyzer:latest
```

[#custom-runner]: https://docs.bentoml.org/en/latest/concepts/runner.html#custom-runner
