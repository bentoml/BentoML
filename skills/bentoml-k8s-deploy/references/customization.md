# What can I customize, and where?

**Everything the user edits is `config.yml`.** The manifests are rendered from it
(`manifests_dir`, at the bottom, is the one exception). The file is small because the
bento's topology, the image tag, the build platform and the registry type are **derived,
not declared** — a config that customizes nothing is five lines. What remains is two kinds
of knob that never overlap:

1. **Kubernetes-shaped keys** (Table 1) — the *pod*: how many, how big, where it runs, how
   it is reached, how it is probed. Each names the field it renders into.
2. **`services.<Name>.config_overrides`** (Table 2) — the *BentoML server inside* the pod:
   timeouts, workers, concurrency, access logging, CORS, TLS, tracing, metrics. Rendered
   into `BENTOML_CONFIG_OVERRIDES` on that container, nested under the service's own name.
   It **beats the `@bentoml.service(...)` decorator**, so an operator retunes a prebuilt
   image without a rebuild.

No decorator constrains Kubernetes, and no Kubernetes field changes BentoML's server
behaviour.

> This surface is effectively **undocumented upstream**. The values below were read out of
> the source (`bentoml/_internal/configuration/v2/default_configuration.yaml` and its
> schema, `_bentoml_impl/server/app.py`, `_bentoml_impl/server/allocator.py`,
> `bentoml/_internal/server/http/{traffic,instruments}.py`, `_bentoml_impl/client/proxy2.py`
> — `client/proxy.py` also defines a `RemoteProxy`, but nothing imports it). Verify against
> the installed version before relying on an exotic key, and treat "inert" rows as
> version-specific.

---

## Derived, not configurable

Worked out on every run, so a declaration can never disagree with reality:

| What | Where it comes from |
|---|---|
| Service list, entry service, dependency DAG | `bento.yaml` — `entry_service`, `services[].name`, `services[].dependencies[].service`; read from the local store after a build, out of the image under `--skip-build`, or from the `deploy/.bento-topology.json` cache |
| Rollout order and `BENTOML_SERVE_DEPENDS` | topological sort of that DAG (deepest tier first, alphabetical within a tier) + each dependency's slug + `kubernetes.namespace` |
| Image **tag** | the bento version — the short git SHA by default; `--version TAG` overrides, `--image <ref>` bypasses build and push |
| Registry type and ECR region | the `image` URL host: `*.dkr.ecr.<region>.amazonaws.com` ⇒ ECR (automatic login, repository created if missing); anything else ⇒ you keep `docker login` valid |
| Build platform | builder arch (`docker info`) vs. node arch (`kubectl get nodes -L kubernetes.io/arch`); a mismatch adds `--opt platform=linux/<node arch>` and logs it |
| Slugs, labels, selectors | the service name, snake_cased with `_` → `-` (`services.<Name>.slug` remains as an override for a DNS-1035 collision or a legacy name) |
| The `app.kubernetes.io/part-of` label | `bento.yaml`'s `name` |

Deriving the DAG removes a whole class of bug (cycles, unknown `depends` targets, two entry
services, a middle tier with no dependencies) — and the worst one: a stale hand-written copy
wires nothing, so the dependency is instantiated in-process, `/readyz` still returns 200 and
the answers stay correct while every model loads into the caller.

Two consequences worth telling a user:

- **Changing the topology never touches `config.yml`.** Add a service or a
  `bentoml.depends()` edge, rebuild, re-render, apply. The only config change a rename can
  force is a `services:` block keyed to the old name — a load-time error, not a silent
  no-op, and the config's whole remaining typo surface.
- **Verification follows the topology.** A single-service bento has no dependency-metrics
  step; growing a dependency adds one, with no config edit.

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
  overwrites it, and any dependency it does not resolve is instantiated **in-process** with
  no error message: every pod loads every model and still reports healthy. Wire dependencies
  with `BENTOML_SERVE_DEPENDS` only — derived from the DAG; no config key can produce a
  runner map.
- **`BENTOML_SERVE_DEPENDS` is whitespace-separated**, not comma-separated:
  `"A=http://a.ns.svc.cluster.local:3000 B=http://b.ns.svc.cluster.local:3000"`.
- **Dependency URLs must contain no `=`.** Each pair is split on the first `=`, so a URL
  with a query string raises a ValueError at startup.
- **The dependency key is the BentoML service name, exactly** — which is also what every
  `services:` block key must be: `Sentiment`, not `sentiment` and not the slug. The
  service name is `name or inner.__name__`, so an
  explicit `@bentoml.service(name="...")` overrides the class name. A mismatch is not an
  error — it falls back to the in-process behaviour above.
- **Neither `/readyz` nor a correct answer detects a mis-wired dependency.** The readiness
  fan-out iterates only *remote proxies*, so an in-process dependency is absent from the list
  and the pod returns **200**; the answer is correct too, because the same code ran in the
  wrong pod. The only reliable check is on the dependency side: an access-log line, or a
  non-zero `bentoml_service_request_total`, in its own pod.
- **Probes that hit `/readyz` need `timeoutSeconds` ≥ 6**
  (`services.<Name>.probes.readiness_timeout_seconds`; the loader enforces the floor).
  The fan-out gives each
  dependency a hard-coded 5 s, so `/readyz` can legitimately take just over 5 s. A 3 s
  kubelet timeout cuts it off first and the pod is **permanently unready** — with no
  explanatory log line in the pod itself. `runner_probe.timeout` will not change the 5 s.
- **Cascading readiness.** `runner_probe.enabled` defaults to `true`: a service's `/readyz`
  issues `GET /readyz` (hard-coded 5 s) to each `bentoml.depends()` dependency and returns
  **503** if any is not ready. So roll out dependencies first (the derived order does), and
  expect the entry pod to go un-ready while a dependency restarts. `/livez` never cascades —
  putting liveness on `/readyz` turns one sick dependency into a cluster-wide restart loop.
  Opt out per service with `config_overrides: {runner_probe: {enabled: false}}`.
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
- **Rendered manifests are output.** Editing `k8s/*.yaml` lasts until the next render, and
  never reaches CI, which renders from `config.yml` too. One renderer and one validator (the
  bundle's) means interactive and CI runs cannot drift, and the container env is emitted in a
  stable order (derived wiring first, then `env` sorted) so re-rendering does not churn pods
  cosmetically.

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

**The label trap.** `runner_name` is the **BentoML service name** (`Sentiment`);
`service_name` is the **bento name** (`text_pipeline`). So select a service with
`runner_name`. `service_name="Sentiment"` matches nothing, the adapter returns no series, and
the HPA sits at `<unknown>` forever with no error anywhere. (Legacy `bentoml_api_server_*`
metrics have no `runner_name` — those names mean a pre-1.2 serving path.)

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

What you give up is everything the config derived for you: rollout order,
`BENTOML_SERVE_DEPENDS` wiring, per-service selectors, ClusterIP-only dependencies,
HPA-vs-`spec.replicas` exclusivity, and the image reference (hand-owned files pin whatever
tag they contain). Migration path: render once into a directory, commit it, point
`manifests_dir` at it, and keep `templates/*.yaml` in this skill as the checklist of the
invariants you now maintain yourself.
