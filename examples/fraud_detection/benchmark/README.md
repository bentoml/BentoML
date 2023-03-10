# Benchmark

1. Install dependencies:
```bash
pip install -r ./dev-requirements.txt
```

2. Benchmark with Locust
```bash
bentoml serve service:svc --production
```
```bash
locust -H http://0.0.0.0:3000 -u 200 -r 10
```

Visit http://0.0.0.0:8089/ and start the test.

3. Testing other serving methods

* BentoML with distributed Runner architecture (default, recommended for most use cases)
  ```bash
  bentoml serve service:svc --production
  ```

* BentoML with embedded local Runner (recommended for light-weight models)
  ```bash
  bentoml serve service_local_runner:svc --production
  ```

* A typical FastAPI implementation with XGBoost syncrounous API for inference:

  ```bash
  uvicorn fastapi_main_load_model:app --workers=10 --port=3000
  ```

* FastAPI endpoint with BentoML runner async API for inference:

  ```bash
  uvicorn fastapi_main_local_runner:app --workers=10 --port=3000
  ```

* BentoML deployed on Ray

  serve run ray_deploy:deploy -p 3000 
