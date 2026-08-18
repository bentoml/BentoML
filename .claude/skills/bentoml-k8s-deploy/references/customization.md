# What can I customize, and where?

**Everything the user edits lives in one file: `config.yml`.** The Kubernetes manifests
under `k8s/` are rendered from it and are not edited (the exception, `manifests_dir`, is at
the bottom of this page). Inside that one file there are still two different *kinds* of
knob, and they do not overlap:

1. **Kubernetes-shaped keys** (Table 1) — everything about the *pod*: how many, how big,
   where it runs, how it is reached, how it is probed. Each one names the Kubernetes field
   it renders into.
2. **`services.<Name>.config_overrides`** (Table 2) — everything about the *BentoML server
   inside* the pod: timeouts, workers, concurrency limits, access logging, CORS, TLS,
   tracing, metrics. The renderer turns it into `BENTOML_CONFIG_OVERRIDES` (compact nested
   JSON) on that service's container, nested under the service's own name. It **beats the
   `@bentoml.service(...)` decorator**, so an operator retunes a prebuilt image without a
   rebuild or a source change.

Nothing in a BentoML decorator constrains Kubernetes, and no Kubernetes field changes
BentoML's server behaviour. If a knob is not in Table 1, it does not belong in the
Kubernetes-shaped keys; if it is not in Table 2, do not put it in `config_overrides`.

> This whole surface is effectively **undocumented upstream** — the values below were read
> out of the BentoML source (`bentoml/_internal/configuration/v2/default_configuration.yaml`
> and its schema, `_bentoml_impl/server/app.py`, `_bentoml_impl/server/allocator.py`,
> `bentoml/_internal/server/http/{traffic,instruments}.py`,
> `_bentoml_impl/client/proxy2.py` — note `client/proxy.py` also defines a `RemoteProxy`
> but nothing imports it; dependency wiring goes through `proxy2.py`). Verify against the
> installed version before relying on
> an exotic key; treat the "inert" rows as version-specific.

---

## Table 1 — what you set in `config.yml`

`<Name>` is the BentoML service name, verbatim, as used for the `services:` block keys.

| Knob | Config path | Renders into | Notes / interactions |
|---|---|---|---|
| Replica count | `services.<Name>.replicas` | `spec.replicas` (deployment) | Per BentoML service. **Omitted from the manifest when `autoscaling.enabled` is true**, so an HPA and a replica count can never fight on apply. |
| CPU / memory requests | `services.<Name>.resources.requests.{cpu,memory}` | `resources.requests` (deployment) | What the scheduler reserves. `@bentoml.service(resources={"cpu","memory"})` is **inert in OSS** — nothing reads it at serve time — so these values are the only place they take effect. A CPU-utilization HPA is meaningless without a CPU request. |
| CPU / memory limits | `services.<Name>.resources.limits.{cpu,memory}` | `resources.limits` (deployment) | Over-limit CPU throttles (latency spikes); over-limit memory is OOMKilled. Note `workers: "cpu_count"` counts **cgroup** CPUs, i.e. it follows the CPU *limit*. |
| GPUs | `services.<Name>.resources.limits["nvidia.com/gpu"]` | `resources.limits` (deployment) | Integer, limits-only, never shared. Must be present for a service whose decorator declares `resources={"gpu": N}` — that decorator key is the one resource key that *is* live in OSS (it sets `CUDA_VISIBLE_DEVICES`), but it cannot make the kubelet hand the pod a device. See `gpu-scheduling.md`. |
| Dependency wiring | `services.<Name>.depends` | `BENTOML_SERVE_DEPENDS` env (deployment) | Direct callees only, by BentoML service name. Also derives the rollout order. Every non-leaf service needs its own list; leaves get no env var. Never hand-write the URLs, and never `BENTOML_RUNNER_MAP`. |
| Exposure | `services.<Entry>.expose.{type,node_port,annotations}` | `spec.type`, `spec.ports[].nodePort`, `metadata.annotations` (service) | **Entry service only** — rejected elsewhere, because non-entry services must stay `ClusterIP`. Entry options: ClusterIP + port-forward, NodePort, LoadBalancer. See `exposure-options.md`. |
| Ingress | `services.<Entry>.ingress.{enabled,class_name,host,tls_secret,annotations}` | `ingress.yaml` | Entry service only, one Ingress per bento. `annotations` is where controller tuning goes — keep proxy timeouts **≥ the service's `traffic.timeout`**, or the proxy 504s before BentoML does, and raise the body-size limit to fit the payload. `host` is required when `enabled`. |
| Extra labels | `kubernetes.extra_labels` | `metadata.labels` on every object | Merged onto every rendered object (team, cost center, env). The four `app.kubernetes.io/*` identity labels are managed by the renderer and **win over these** — an override of `app.kubernetes.io/name` would desync a Deployment's selector from its own pod template, which the API server rejects and a client-side dry-run does not catch. `spec.selector` is **immutable** after apply, so changing a slug means delete + re-apply. |
| Object names / hostnames | `services.<Name>.slug` | object names, dependency URLs | DNS-1035 (starts with a letter), unique. Changing it rewrites every caller's dependency URL automatically. |
| Probe timings | `services.<Name>.probes.{startup_failure_threshold,readiness_timeout_seconds}` | `startupProbe`, `readinessProbe` (deployment) | Paths are `/livez` and `/readyz` and are **not** configurable — but `@bentoml.service(path_prefix="/x")` moves them (and `/metrics`) under the prefix, which the renderer cannot express (a `manifests_dir` case). `startup_failure_threshold × 10 s` is the model-loading budget. **`readiness_timeout_seconds` must be ≥ 6**: the dependency fan-out gives each dependency a hard-coded 5 s, so `/readyz` can legitimately exceed 5 s and a 3 s kubelet timeout makes the pod permanently unready. Liveness is fixed at 3 s because `/livez` never cascades. |
| Graceful shutdown | (not configurable; fixed at 60 s) | `terminationGracePeriodSeconds` | Above the longest expected request so in-flight inference drains. Longer than 60 s is a `manifests_dir` case. |
| Image pull secret | `kubernetes.image_pull_secret` | `imagePullSecrets` (every deployment) | One per namespace; every service of the bento runs the same image, so every Deployment gets it. See `private-registries.md`. |
| Image | `image.registry` + `image.repository` (+ the tag chosen at deploy time) | `containers[].image` | The renderer writes the full reference directly — there is no sentinel to substitute. A new build means re-render + apply, not a config change. Pass `--image REF` when rendering: the default tag is the project's short git SHA, which is not the image you just pushed. |
| Registry auth | `image.registry_type`, `image.ecr_region`, `image.local_image_preloaded` | (build/push behaviour) | `ecr` logs in with `aws ecr get-login-password`; `ecr_region` is derived from a standard `*.dkr.ecr.<region>.amazonaws.com` host and only needs setting for a non-standard/VPC endpoint. `generic` assumes `docker login` was already done. `none` pushes nothing and requires `local_image_preloaded: true`. |
| Node placement | `services.<Name>.node_selector`, `services.<Name>.tolerations` | `nodeSelector`, `tolerations` (deployment) | GPU pools, arch (`kubernetes.io/arch: amd64`). GPU nodes are usually tainted → needs a matching toleration. `affinity` / `topologySpreadConstraints` are not in the schema (a `manifests_dir` case). |
| Autoscaling | `services.<Name>.autoscaling.{enabled,min_replicas,max_replicas,metric,target}` | `<slug>-hpa.yaml` | Per service; `metric: cpu` or `concurrency`. See the HPA section below. |
| Rollout / verification budgets | `kubernetes.rollout_timeout_seconds`, `verify.readyz_timeout_seconds`, `verify.inference.timeout_seconds` | (client-side waits) | Keep the rollout timeout above the startup-probe budget, and the inference timeout ≥ the entry service's `traffic.timeout`. |
| Quantity quoting | every `resources.*` value | `cpu`, `memory`, `nvidia.com/gpu` | Always quoted strings: `cpu: "1"`, not `cpu: 1`. An unquoted whole number is a YAML int while the API server stores the quantity as a string, so the strategic-merge patch diffs on every apply — `kubectl apply` says `configured` forever and `kubectl diff` / GitOps drift detection report phantom drift. The loader rejects unquoted quantities so the error names your line. |
| Plain env vars / secret env | `services.<Name>.env`, `services.<Name>.env_from_secrets` | `env`, `envFrom.secretRef` (deployment) | Application config (`HF_TOKEN`, endpoints of external systems). `env_from_secrets` holds Secret **names** only — secret values never go in the config file. `BENTOML_RUNNER_MAP`/`BENTOML_SERVE_RUNNER_MAP` in `env` are rejected at load time; `BENTOML_SERVE_DEPENDS` / `BENTOML_CONFIG_OVERRIDES` set there replace the derived value entirely (nothing is emitted twice) and are warned about. **A Secret's contents are invisible to every static check** — a Secret carrying a runner map still reaches the pod and silently re-enables the in-process fallback, so keep runtime knobs out of Secrets. |
| Volumes, PDBs, sidecars, affinity | not in the schema | — | Volumes for model caches (PVC), a writable `emptyDir` for `/tmp` on a read-only root, a `ConfigMap`-mounted BentoML config file (then point `BENTOML_CONFIG` at it instead of using overrides), a `PodDisruptionBudget` (`minAvailable: 1`, worth it only at ≥2 replicas). These need `kubernetes.manifests_dir` — see the bottom of this page. |

---

## Table 2 — belongs in `BENTOML_CONFIG_OVERRIDES`, i.e. `config_overrides`

In `config.yml` you write the bare keys under the service that runs them:

```yaml
services:
  Summarization:
    config_overrides:
      workers: 2
      traffic: {timeout: 300, max_concurrency: 16}
```

and the renderer emits, on that container only:

```yaml
- name: BENTOML_CONFIG_OVERRIDES
  value: '{"services":{"Summarization":{"workers":2,"traffic":{"timeout":300,"max_concurrency":16}}}}'
```

The nesting under the **named service** is what makes the override win; a global
`{"services":{"traffic":...}}` loses to the decorator (see the precedence gotcha below),
which is why the renderer does the keying and you do not. `services.<ServiceName>` uses the
**BentoML service name verbatim** (`Summarization`, not the slug) — the same name that keys
the `services:` blocks. One consequence: a block keyed to a *different* service (the
`traffic.timeout` of a service this pod *calls* — see the gotchas) cannot be expressed
through `config_overrides`. Do that with the callee's decorator, or, without a rebuild,
with a raw `env: {BENTOML_CONFIG_OVERRIDES: '<json>'}` entry on the caller **while its own
`config_overrides` is empty** (two sources for one env var collide).

| Path under `services.<ServiceName>` | Default | What it does |
|---|---|---|
| `traffic.timeout` | `60` (seconds) | Server-side request deadline; exceeding it returns **HTTP 504**. Also the basis of the client timeout used when *calling* this service (× 1.01). |
| `traffic.max_concurrency` | unset (unlimited) | Cap on concurrent requests. Implemented as a semaphore **per worker** sized `ceil(max_concurrency / workers)`; over the cap the request is **rejected immediately with HTTP 429** `{"error":"Too many requests"}` — never queued. Health and metrics paths bypass it. |
| `workers` | `1` | Uvicorn worker processes per pod. `"cpu_count"` uses the cgroup CPU count (so it follows the container's CPU limit) — **but `"cpu_count"` also disables BentoML's GPU assignment**, so never use it on a GPU service. Divides `max_concurrency`; wants ~1 CPU each. |
| `backlog` | `2048` | Socket accept backlog. |
| `metrics.enabled` | `true` | Turning it off removes `/metrics` and breaks concurrency-based HPAs. |
| `metrics.namespace` | `bentoml_service` | Prefix for the request metrics (`request_total`, `request_in_progress`, `request_duration_seconds`, `last_request_timestamp_seconds`). Changing it renames those — update your HPA/PromQL. It does **not** rename `bentoml_service_adaptive_batch_size`, whose namespace is hard-coded. |
| `metrics.duration.buckets` | client default | Explicit histogram buckets for `request_duration_seconds`. **`min`/`max`/`factor` win:** if all three of `metrics.duration.min`/`max`/`factor` are set, exponential buckets are used and `buckets` is ignored entirely; `buckets` applies only when they are not all set. Setting only some of the three falls through to `buckets`/the default. |
| `logging.access.enabled` | `true` | Per-request access log line. |
| `logging.access.request_content_length` / `request_content_type` / `response_content_length` / `response_content_type` | `true` | Which fields the access log carries. |
| `logging.access.skip_paths` | `["/metrics","/healthz","/livez","/readyz"]` | Prefix-matched (`startswith`) for the access log, and additionally suppresses `last_request_timestamp_seconds` for those paths. That is **all** it does. It does **not** gate the concurrency limiter (the serving app overwrites that middleware's skip list with just the livez/readyz endpoints, matched **exactly**, not by prefix), and it does **not** exclude the paths from `request_total` / `request_in_progress` / `request_duration_seconds` — probe traffic is counted there. See the HPA section. |
| `logging.access.format.trace_id` / `format.span_id` | `032x` / `016x` | Format specs for trace/span ids in the access log. |
| `http.cors.enabled` | `false` | Needed only for browser callers. |
| `http.cors.access_control_allow_origins` | unset | **Required** when CORS is enabled (startup asserts otherwise). Also `..._allow_credentials`, `..._allow_methods`, `..._allow_headers`, `..._allow_origin_regex`, `..._max_age`, `..._expose_headers`. |
| `http.response.trace_id` | `false` | Echo the trace id in the response. |
| `http.host` / `http.port` | `0.0.0.0` / `3000` | Changing the port means changing `containerPort`, the Service `targetPort`, the probes **and** every dependency URL. `BENTOML_PORT` (and the container's `$PORT`) do the same thing more simply. |
| `ssl.enabled` + `ssl.certfile` / `keyfile` / `keyfile_password` / `ca_certs` / `version` / `cert_reqs` / `ciphers` | off | In-pod TLS. Usually the wrong layer in Kubernetes — terminate at the Ingress or a mesh instead; if you do use it, the cert files have to be mounted from a Secret. |
| `tracing.exporter_type` | unset | `otlp` \| `jaeger` \| `zipkin` \| `in_memory`. With no exporter, tracing is off. |
| `tracing.sample_rate` | `0` (no traces) | `0.0`–`1.0`. Leaving it unset means nothing is exported even with an exporter configured. |
| `tracing.excluded_urls`, `tracing.timeout`, `tracing.max_tag_value_length` | unset | Filtering and exporter limits. |
| `tracing.otlp.protocol` / `.endpoint` / `.compression` (+ `.http.certificate_file`, `.http.headers`, `.grpc.insecure`, `.grpc.headers`) | unset | OTLP collector wiring — the usual choice on Kubernetes; point `endpoint` at the collector Service. |
| `tracing.jaeger.protocol` / `.collector_endpoint` / `.thrift.*` / `.grpc.insecure` | `thrift` | Jaeger wiring. |
| `tracing.zipkin.endpoint` / `.local_node_*` | unset | Zipkin wiring; the exporter is only used if at least one `zipkin.*` value is set. |
| `monitoring.enabled` / `monitoring.type` / `monitoring.options` | `true` / `default` / `{log_path: monitoring}` | Inference-data monitoring sink. The default writes files inside the container — mount a volume or switch it off unless you have a collector. |
| `runner_probe.enabled` | `true` | Whether this service's `/readyz` also probes its `bentoml.depends()` dependencies. See the cascading-readiness gotcha. |
| `runner_connection.max_requests` / `max_age` | `100` / `300.0` | Recycling of the HTTP connections this pod uses to call its dependencies. **Same both-pods keying trap as `traffic.timeout` above:** the value is read from the **caller's** config under the **callee's** service name, so to change how the gateway recycles connections to `Sentiment` you set `services.Sentiment.runner_connection.*` on the **gateway's** Deployment. A global setting does not cascade. |

Other env vars worth knowing: `BENTOML_PORT` / `BENTOML_HOST`; `BENTOML_CONFIG` (path to a
full YAML config file — mount it from a ConfigMap when the override JSON gets unwieldy);
`BENTOML_SERVE_SERVICE_NAME` (equivalent to `--service-name`); `BENTOML_SERVE_DEPENDS`
(equivalent to `--depends`). String values in the config are expanded with
`os.path.expandvars` + `expanduser` (repeatedly, so nesting works), which supports `$VAR`,
`${VAR}` and `~`. Shell-style defaults do **not** exist: `${VAR:-fallback}` is left in the
value verbatim, and an undefined `$VAR` is also left literal rather than becoming empty.

### Table 2b — looks tunable, is not

Do not offer these as env knobs; they silently do nothing.

| Key / var | Reality |
|---|---|
| `services.<Svc>.threads` | Read from the **decorator config** only, not the merged container config. Changing it via env has no effect — it must be `@bentoml.service(threads=N)`. |
| `services.<Svc>.endpoints.livez` / `.readyz` | Not in the config schema at all; only a `@bentoml.service(endpoints=...)` kwarg, and even then it does not move the served routes (they stay `/livez`, `/healthz`, `/readyz`) — it only affects log/metric skip paths and the URL a dependency client polls. **Keep the manifest probes on `/livez` and `/readyz`.** The one thing that *does* move them is `path_prefix` — next row. |
| `@bentoml.service(path_prefix="/x")` | **Not** an env knob (decorator/code only), but it is the one setting that relocates the system routes: `/livez`, `/healthz`, `/readyz` **and** `/metrics` all move under the prefix. Consequences: the manifest's probe paths must become `/x/readyz` and `/x/livez`, the Prometheus scrape path becomes `/x/metrics`, and the concurrency limiter's hard-coded bypass list (`/metrics`, `/healthz`, `/livez`, `/readyz`, matched exactly) no longer matches — so under `path_prefix` probe requests start consuming `max_concurrency` slots and can be answered with 429. Check for it during topology discovery. |
| `services.<Svc>.batching.*` | Inert. Batching is configured per API method: `@bentoml.api(batchable=True, max_batch_size=..., max_latency_ms=...)`. An overloaded batch dispatcher returns **HTTP 503** `"process is overloaded"`, not 429. |
| `BENTOML_API_WORKERS` | Silently ignored on BentoML 1.2+ (only wired into the legacy `bentoml.legacy.Service` path). Use `workers`. |
| `BENTOML_TIMEOUT` / `--timeout` | Accepted by the 1.2+ worker and never applied. Use `traffic.timeout`. |
| `BENTOML_CONFIG_OPTIONS` | Deprecated flat `a.b.c=value` form; emits a DeprecationWarning. Use `BENTOML_CONFIG_OVERRIDES`. |
| `services.<Svc>.max_runner_connections`, `runner_probe.timeout`, `runner_probe.period` | Legacy 1.1 runner-server keys; unused on the 1.2+ serving path. The dependency readiness probe uses a hard-coded 5 s timeout. |
| `services.<Svc>.grpc.*` | The 1.2+ SDK serves HTTP only; gRPC config applies to the legacy path. |
| `services.<Svc>.extra_ports`, `replicate_process`, `http.proxy_port` | Either BentoCloud-only or decorator-only (custom-command services). |
| `api_server.*` at the top level | The per-service config is deep-merged **over** it at startup for `traffic`, `metrics`, `logging`, `ssl`, `http`, `grpc`, `backlog`, `runner_probe`, `max_runner_connections` — so a top-level `api_server` block for those keys gets clobbered. Always use `services.<ServiceName>.*`. |

---

## Table 3 — BentoCloud-only: ignore on vanilla Kubernetes

These are accepted by the decorator and appear in `bento.yaml`, but nothing in
open-source BentoML reads them at serve time. They are instructions to BentoCloud's
autoscaler and instance-type picker.

| Key | Kubernetes equivalent, if any |
|---|---|
| `resources.gpu_type` (e.g. `nvidia-tesla-t4`) | `nodeSelector` on the GPU pool's node labels (e.g. `nvidia.com/gpu.product`) plus a toleration. |
| `resources.tpu_type` | Node selector for the TPU pool; there is no BentoML-side support. |
| `traffic.concurrency` | The *autoscaling target*, not a limit. Use `HorizontalPodAutoscaler` (see below); use `traffic.max_concurrency` if you actually want a hard cap. |
| `traffic.external_queue` | No equivalent. If you need queueing rather than 429s, put a queue/broker in front yourself. |
| `resources.cpu` / `resources.memory` | Live only as manifest `requests`/`limits` — this is why the rendered Deployment always sets both. |

---

## Gotchas (multi-service deployments)

- **Never set `BENTOML_RUNNER_MAP` / `BENTOML_SERVE_RUNNER_MAP`.** The serving process
  overwrites it, and any dependency it does not resolve is then instantiated **in-process**
  instead of being called over HTTP: every pod loads every model, memory blows up, and the
  pod still reports healthy. There is no error message. Wire dependencies with
  `BENTOML_SERVE_DEPENDS` (`--depends`) only — which is what `services.<Name>.depends`
  renders into; there is no config key that can produce a runner map.
- **`BENTOML_SERVE_DEPENDS` is whitespace-separated**, not comma-separated:
  `"A=http://a.ns.svc.cluster.local:3000 B=http://b.ns.svc.cluster.local:3000"`.
- **Dependency URLs must contain no `=`.** Each pair is split on the first `=`, so a URL
  with a query string raises a ValueError at startup.
- **The dependency key is the BentoML service name, exactly** — so every `depends` entry
  (and every `services:` block key) is that name: `Sentiment`, not `sentiment` and not the
  slug. Note the service name is `name or inner.__name__`, so an
  explicit `@bentoml.service(name="...")` overrides the class name. A mismatch is not an
  error — it falls back to the in-process behaviour above.
- **Neither `/readyz` nor a correct answer detects a mis-wired dependency.** The readiness
  fan-out iterates only the dependencies that are *remote proxies*; an in-process
  dependency is simply absent from that list, so a mis-wired pod returns **200** and looks
  healthier than a correctly wired one. And the inference result is **correct**, because
  the same code ran — just in the wrong pod, with the wrong memory footprint. The only
  reliable check is on the dependency side: an access-log line, or a non-zero
  `bentoml_service_request_total`, in the dependency's own pod.
- **Probes that hit `/readyz` need `timeoutSeconds` ≥ 6**
  (`services.<Name>.probes.readiness_timeout_seconds`; the loader enforces the floor). The fan-out gives each
  dependency a hard-coded 5 s, so `/readyz` can legitimately take just over 5 s. A 3 s
  kubelet timeout cuts it off first and the pod is **permanently unready** — with no
  explanatory log line in the pod itself. `runner_probe.timeout` will not change the 5 s.
- **Cascading readiness.** `runner_probe.enabled` defaults to `true`, so a service's
  `/readyz` issues a `GET /readyz` (5 s timeout, hard-coded) to every one of its
  `bentoml.depends()` dependencies and returns **503** if any is not ready. Consequences:
  roll out dependencies before their callers (which is exactly the order derived from
  `depends`), and expect the entry pod to sit un-ready
  while a dependency restarts. `/livez` never cascades, which is why liveness must stay on
  `/livez` — putting liveness on `/readyz` turns one sick dependency into a cluster-wide
  restart loop. Opt out per service to make readiness pod-local, via that service's
  `config_overrides: {runner_probe: {enabled: false}}`.
- **A dependency's timeout must be raised in BOTH pods.** The client timeout for a call is
  read from the **calling** pod's own config for the callee's name
  (`services.<Callee>.traffic.timeout`, × 1.01), not from the callee. So a slow dependency
  needs `{"services":{"<Callee>":{"traffic":{"timeout":300}}}}` in the **caller's** pod (to
  stop the client giving up) *and* the same in the **callee's** pod (to stop its own 504
  middleware firing). Setting it in only one place produces a confusing half-fix. **This is
  the one thing `config_overrides` cannot express**, since the renderer keys it to the
  service that owns the block: raise the timeout in the callee's decorator
  (`@bentoml.service(traffic={"timeout": 300})`, whose value every pod of the bento sees, at
  the cost of a rebuild), or put the raw JSON in the caller's `env`
  (`env: {BENTOML_CONFIG_OVERRIDES: '...'}`) while leaving that service's
  `config_overrides` empty — two sources for one env var collide.
- **Per-service beats global, and the decorator lives in the per-service block.** A global
  `{"services":{"traffic":{"timeout":300}}}` is merged *under* each named service's config,
  which already contains the decorator's values — so a global override loses to any
  decorator-declared value. **Always use `services.<ServiceName>.<key>`.**
- **The override JSON is parsed at import time by the v1 schema first**, which is not
  lenient about unknown *top-level* keys — a typo like `{"service":{...}}` (singular) can
  raise `BentoMLConfigException` on startup rather than being ignored. Keep the top level
  to `services` (and `version`).
- **`BENTOML_API_WORKERS` is silently ignored on 1.2+** — a pod "configured" with it runs
  one worker. Use `workers` in the override JSON.
- **`workers: "cpu_count"` disables GPU assignment.** On a GPU service, set an integer.
- **Kubernetes' NVIDIA device plugin already sets `CUDA_VISIBLE_DEVICES`** in the
  container, and BentoML skips its own GPU allocation when that variable is present. That
  is the desired behaviour — but it means `resources={"gpu": N}` in the decorator does not
  control which devices a pod sees; `resources.limits["nvidia.com/gpu"]` in the config
  does.
- **Only the entry service may be exposed.** Dependency traffic is
  `application/vnd.bentoml+pickle`; a NodePort/LoadBalancer/Ingress in front of a
  dependency is an unauthenticated deserialization endpoint, i.e. remote code execution.
  `expose:`/`ingress:` under a non-entry service is a load-time error, and hand-written
  manifests under `manifests_dir` are the only way to break the invariant.
- **Unquoted whole-number CPU makes every apply non-idempotent.** `cpu: 1` renders a YAML
  integer; the stored quantity is the string `"1"`, so the patch never converges and
  `kubectl apply` reports `configured` on every run (no pod churn, but drift detection and
  the generated deploy script both lie). `cpu: "1"` reports `configured` once, then
  `unchanged`. Same for `memory` and `nvidia.com/gpu` — and the config loader rejects
  unquoted quantities so the error names your `config.yml` line, not the rendered file.
- **`services[].config` in `bento.yaml` is a snapshot of the decorator kwargs** (no
  defaults merged) and is *not* re-applied at runtime. It is excellent for pre-filling
  `config.yml` defaults, and worthless as a statement of what the running pod is doing.
- **Rendered manifests are output.** Editing `k8s/*.yaml` works exactly until the next
  render, which overwrites it; the change also never reaches CI, which renders from
  `config.yml` too. Change the config and re-render. There is exactly **one** renderer and
  one validator — the deploy bundle's — so interactive and CI runs cannot drift apart, and
  the container env is emitted in a stable order (derived wiring first, then `env` sorted)
  so re-rendering does not churn pods for cosmetic reasons.

---

## HPA: metrics, labels, and the concurrency trap

BentoML 1.2+ pods expose these on **`/metrics`, port 3000** (prefix = `metrics.namespace`,
default `bentoml_service`):

| Metric | Type | Labels |
|---|---|---|
| `bentoml_service_request_in_progress` | Gauge (`livesum` across workers) | `endpoint`, `service_name`, `service_version`, `runner_name` |
| `bentoml_service_request_total` | Counter | `endpoint`, `service_name`, `service_version`, `http_response_code`, `runner_name` |
| `bentoml_service_request_duration_seconds` | Histogram | same as `request_total` |
| `bentoml_service_last_request_timestamp_seconds` | Gauge | `service_name`, `service_version`, `runner_name` |
| `bentoml_service_adaptive_batch_size` | Histogram | `runner_name`, `worker_index`, `method_name`, `service_version`, `service_name` |

**The label trap.** On these metrics:
- `runner_name` = the **BentoML service name** of the pod (`Sentiment`, `TextPipeline`)
- `service_name` = the **bento name** (`text_pipeline`)

So to select one service you filter on `runner_name`. Filtering
`service_name="Sentiment"` matches nothing, the adapter returns no series, and the HPA
sits at `<unknown>` indefinitely with no error anywhere. (The legacy
`bentoml_api_server_*` metrics have no `runner_name` at all — if you see those names, the
pod is on the pre-1.2 serving path.)

**Variant A — CPU utilization.** Works with a stock metrics-server. It works *only*
because the rendered Deployment always sets `resources.requests.cpu`: utilization is
used/requested, and a pod with no request has undefined utilization. Fine for CPU-bound
services, misleading for GPU inference (a saturated GPU pod can idle at 5% CPU).

**Variant B — in-flight requests per pod.** Requires prometheus-adapter or KEDA. Check
with `kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1" | head`. prometheus-adapter
rule:

```yaml
rules:
  custom:
    - seriesQuery: 'bentoml_service_request_in_progress{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: namespace}
          pod: {resource: pod}
      name: {matches: "^bentoml_service_request_in_progress$", as: "bentoml_inflight"}
      metricsQuery: 'sum(bentoml_service_request_in_progress{<<.LabelMatchers>>,runner_name="<ServiceName>",endpoint!~"/(livez|readyz|healthz|metrics)"}) by (<<.GroupBy>>)'
```

then target it as a `Pods` metric with `averageValue: "<in-flight per pod>"`. Because the
gauge is `livesum`, its value already covers all workers in the pod, so the target is
per-pod, not per-worker.

**Exclude the health endpoints from the query.** `logging.access.skip_paths` keeps the
probes out of the access log, not out of these metrics, so every kubelet `/readyz` and
`/livez` hit is counted in `request_in_progress` for as long as it is in flight. On a
service whose `/readyz` fans out to dependencies that can be seconds per probe period —
enough to keep an autoscaler warm on a completely idle service. Hence the
`endpoint!~"/(livez|readyz|healthz|metrics)"` matcher in the `metricsQuery` above; add it
even if you otherwise sum over all endpoints.

**`max_concurrency` vs the HPA target.** `traffic.max_concurrency` sheds load with a 429
the instant it is exceeded; an HPA needs ~30–60 s to notice and place a new pod, plus
model-load time. If `max_concurrency` is at or below `workers × averageValue`, the service
starts returning 429s before it ever scales up — the autoscaler is effectively disabled.
Set `max_concurrency` comfortably above the HPA target (or leave it unset and let latency
degrade), and treat it as a last-resort overload guard rather than a scaling signal.

**Also, when adding an HPA:** `autoscaling.enabled: true` makes the renderer drop
`spec.replicas` from that service's Deployment (a declared replica count and an HPA fight on
every apply); `behavior.scaleDown.stabilizationWindowSeconds` is rendered at 300 s, which
should be at least the worst-case model load; and scale each BentoML service on its own
signal — the DAG's bottleneck is rarely the entry service.

---

## The escape hatch: `kubernetes.manifests_dir`

Setting it makes the deployment apply those files **as-is**; `services:` no longer shapes
the workload. That is how you get anything the schema has no key for — sidecars, volumes,
PodDisruptionBudgets, affinity/topology spread, a longer
`terminationGracePeriodSeconds`, `path_prefix`-adjusted probe paths.

What you give up is everything Table 1 derived for you: rollout order,
`BENTOML_SERVE_DEPENDS` wiring, per-service selectors, ClusterIP-only dependencies,
HPA-vs-`spec.replicas` exclusivity, and the image reference (hand-owned files pin whatever
tag they contain). Migration path: render once into a directory, commit it, point
`manifests_dir` at it, and keep `templates/*.yaml` in this skill as the checklist of the
invariants you now maintain yourself.
