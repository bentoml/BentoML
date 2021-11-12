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


## Mounting a Flask App to Service

Use `Service#mount_wsgi_app` to mount flask app to a Service. This works for any WSGI based python web application. 

```python
# hello.py
import bentoml
from bentoml.io import JSON
from flask import Flask

flask_app = Flask("sample_wsgi_app")

@flask_app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"


svc = bentoml.Service("bento-server-test")
svc.mount_wsgi_app(flask_app)

@svc.api(input=JSON(), output=JSON())
def predict(input_json):
    return {'input_received': input_json, 'foo': 'bar'}

app = svc.asgi_app
```


## Mounting a FastAPI app to Service

Use `Service#mount_asgi_app` to mount FastAPI app to a Service. This works for any ASGI based python web application. 

```python
import bentoml
from bentoml.io import JSON

from fastapi import FastAPI

fastapi_app = FastAPI()

@fastapi_app.get("/hello")
def hello():
    return {"Hello": "World"}

svc = bentoml.Service("hello")
svc.mount_asgi_app(fastapi_app)

@svc.api(input=JSON(), output=JSON())
def predict(input_json):
    return {'input_received': input_json, 'foo': 'bar'}

app = svc.asgi_app
```