# Developing BentoServer 


## Run BentoServer with sample Service
Create a sample Servie in `hello.py`:

```python
# hello.py
import bentoml
from bentoml.io import JSON

svc = bentoml.Service("bento-server-test")

@svc.api(input=JSON(), output=JSON())
def predict(input_json):
    return {'input_received': input_json, 'foo': 'bar'}

@svc.api(input=JSON(), output=JSON())
async def classify(input_json):
    return {'input_received': input_json, 'foo': 'bar'}

# make sure to expose the asgi_app from the service instance
app = svc.asgi_app
```

Run the BentoServer:
```bash
bentoml serve hello:svc --reload
```

Alternatively, run the BentoServer directly with `uvicorn`:
```bash
uvicorn hello:app --reload
```

Send test request to the server:
```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"abc": 123}'
```