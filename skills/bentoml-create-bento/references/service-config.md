# `@bentoml.service` configuration

Everything is optional. The decorator takes named parameters plus any key of
`bentoml.ServiceConfig`. Full list:
https://docs.bentoml.com/en/latest/reference/bentoml/configurations.html

## Decorator parameters

| Parameter | Effect |
|---|---|
| `name="..."` | Overrides the Service name (default: the class name). Deploy configs, clients and `BENTOML_SERVE_DEPENDS` key off this — renaming is a breaking change |
| `image=Image(...)` | Runtime environment. Only the **entry** Service's image is used |
| `description="..."` | Shown in the Swagger UI |
| `path_prefix="/v1"` | Prefixes every API route, mounted ASGI app **and** `/livez` + `/readyz` |
| `envs=[{"name": "HF_TOKEN"}, {"name": "MODE", "value": "fast"}]` | Declares env vars. Value omitted = supplied at deploy time. Never put a secret value here |
| `labels={"owner": "ml-team"}` | Metadata on the bento |
| `cmd=["uvicorn", "app:api", "--port", "$PORT"]` | Run an external server instead of BentoML's; BentoML proxies to it. Pair with `http={"proxy_port": N}` if it doesn't listen on 8000 |

## Config keys that matter locally

| Key | Default | Notes |
|---|---|---|
| `workers` | `1` | Processes, each with its own model copy. `"cpu_count"` uses every core — but **skips GPU assignment** |
| `threads` | `1` | Concurrent in-flight requests per worker for **sync** APIs (an `anyio` capacity limiter). Measured: four concurrent 1-second requests take 4.0 s at the default and 1.0 s with `threads=4`. The usual cause of serial throughput |
| `traffic={"timeout": s}` | `60.0` | Per-request deadline. Raise it for long generation |
| `traffic={"max_concurrency": n}` | unset | Requests beyond this are rejected rather than queued |
| `resources={"gpu": n}` | unset | Sets `CUDA_VISIBLE_DEVICES` per worker locally. `BENTOML_DISABLE_GPU_ALLOCATION=1` turns that off |
| `http={"port": 3000, "host": "0.0.0.0", "cors": {...}}` | as shown | CORS is **off** by default — a browser front end needs it enabled explicitly |
| `endpoints={"livez": "/healthz"}` | `/livez`, `/readyz` | Rename health checks if a load balancer demands specific paths |
| `metrics={"enabled": True, "namespace": "..."}` | enabled | Prometheus at `/metrics` |
| `logging={"access": {"skip_paths": [...]}}` | health paths skipped | Access-log noise control |
| `extra_ports=[8080]` | unset | Additional ports the container should expose |

## Config keys that are hints only

Set them for the deployment platform; `bentoml serve` ignores them.

| Key | Consumed by |
|---|---|
| `resources={"cpu": "2", "memory": "4Gi"}` | BentoCloud instance sizing; on Kubernetes the real requests/limits come from the deploy config, not from here |
| `resources={"gpu_type": "nvidia-l4", "tpu_type": ...}` | BentoCloud instance selection |
| `traffic={"concurrency": n, "external_queue": True}` | BentoCloud autoscaling and its request queue |

## Overriding at run time without editing code

```bash
BENTOML_CONFIG_OPTIONS='services.Summarization.workers=4' bentoml serve
BENTOML_CONFIG=./bentoml_config.yaml bentoml serve
```

Useful for retuning a built bento or a running container. In a Kubernetes deployment this is
what `config_overrides` in the deploy config compiles down to. Note the key is the Service's
**own** name — an override on the caller does not reach its dependency.

## Lifecycle hooks

| Hook | Runs | Signature | Use for |
|---|---|---|---|
| `@bentoml.on_deployment` | Once, before any worker starts | no `self` (static) | One-time global setup: a migration, a download, a cache warm |
| `@bentoml.on_startup` | Once per worker, before the APIs accept traffic | takes `self`; may be `async` | Per-worker resources: DB connections, clients |
| `__init__` | Once per worker | `self` | Loading the model. The usual place |
| `@bentoml.on_shutdown` | Per worker, on shutdown | `self` | Closing connections, flushing buffers |

https://docs.bentoml.com/en/latest/build-with-bentoml/lifecycle-hooks.html
