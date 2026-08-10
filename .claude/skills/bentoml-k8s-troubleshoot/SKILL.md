---
name: bentoml-k8s-troubleshoot
description: Diagnose and fix BentoML services deployed to Kubernetes with the bentoml-k8s-deploy skill (plain Deployment + Service, port 3000, /livez + /readyz probes). Use when the user says things like "my BentoML deployment is failing", "pods are crashing / CrashLoopBackOff / ImagePullBackOff", "pod stuck Pending", "readiness probe failing", "rollout stuck", "can't reach my service on Kubernetes", or "inference requests return 4xx/5xx errors".
---

# Troubleshoot a BentoML service on Kubernetes

You are diagnosing a BentoML service deployed as a plain `Deployment` + `Service`
(created by the sibling skill `bentoml-k8s-deploy`): container port **3000**, liveness
probe **`GET /livez`**, readiness probe **`GET /readyz`**, and a generous `startupProbe`
on `/readyz` because model loading can take minutes.

Work the decision tree below: run the triage block, match the symptom, run the
diagnostics for that symptom, state the likely cause, then apply the fix.

## Placeholders used throughout

- `<ns>` — namespace of the deployment (ask the user; the deploy skill asked them too)
- `<name>` — the app name, i.e. the value of the `app.kubernetes.io/name` label
  (the service class name, snake_cased and then `_`→`-`,
  e.g. `Summarization` → `summarization`, `MyService` → `my-service`)
- `<pod>` — a concrete pod name from `kubectl get pods` output
- `<deploy>` — the Deployment name (usually the same as `<name>`)
- Other `<...>` tokens are literal values from the user's setup (registry, credentials,
  image, endpoint name, etc.)

## Safety rules (binding)

- **Never delete** resources these skills did not create. Never `kubectl delete`
  Deployments, Services, Secrets, or namespaces as a "fix". Prefer `kubectl apply` of a
  corrected manifest (the deploy skill writes manifests to the project's `k8s/` dir) or
  `kubectl rollout restart deployment/<deploy> -n <ns>`.
- **Always show the user the target context and namespace before any mutating command**
  and get their confirmation:
  ```sh
  kubectl config current-context
  kubectl config view --minify -o jsonpath='{..namespace}{"\n"}'   # empty = default
  ```
- Read-only diagnostics (`get`, `describe`, `logs`, `events`, `port-forward`, `curl`)
  may run freely once the context is confirmed.

## Step 0 — Triage (always run first)

```sh
kubectl get pods -n <ns> -l app.kubernetes.io/name=<name> -o wide
kubectl get deploy,svc -n <ns> -l app.kubernetes.io/name=<name>
kubectl describe pod -n <ns> <pod>          # read Events at the bottom first
kubectl get events -n <ns> --sort-by=.lastTimestamp | tail -30
kubectl logs -n <ns> <pod> --tail=100
kubectl logs -n <ns> <pod> --previous --tail=100   # if the container restarted
```

Route on the pod STATUS / describe output:

| Observation | Go to |
|---|---|
| `ImagePullBackOff` / `ErrImagePull` | 1 |
| `CrashLoopBackOff` / `Error` | 2 |
| `OOMKilled` (in `Last State` of describe) | 3 |
| `Pending` | 4 |
| `Running` but `READY 0/1`; probe failures in events | 5 |
| `kubectl rollout status` never completes | 6 |
| Pods Ready but requests don't reach the service | 7 |
| Service reachable but inference returns 4xx/5xx | 8 |
| `kubectl apply` rejected a manifest (validation error, e.g. `Invalid value` for a name) | 9 |
| port-forward "works" but responses look wrong / content is from another app | 10 |

## 1. ImagePullBackOff / ErrImagePull

Diagnose — the exact pull error is in events:

```sh
kubectl describe pod -n <ns> <pod> | grep -A5 -i "failed to pull\|pull image"
kubectl get pod -n <ns> <pod> -o jsonpath='{.spec.containers[0].image}{"\n"}'
kubectl get pod -n <ns> <pod> -o jsonpath='{.spec.imagePullSecrets}{"\n"}'
```

Likely causes and fixes:

- **`not found` / `manifest unknown`** — wrong image ref or tag never pushed. Compare
  the pod's image against what was actually pushed (`docker images | grep <name>`).
  Remember BentoML tags images as `name:version` (e.g. `summarization:abc123`), which
  must be retagged to `<registry>/<repo>:<tag>` and pushed. Fix the `image:` field in
  `k8s/deployment.yaml` and `kubectl apply -n <ns> -f k8s/deployment.yaml`.
- **`unauthorized` / `authentication required`** — private registry without
  `imagePullSecrets`. Create the secret and reference it (never write credentials into
  a manifest file):
  ```sh
  kubectl create secret docker-registry regcred -n <ns> \
    --docker-server=<registry> --docker-username=<user> \
    --docker-password=<password-or-token>
  ```
  Add to the Deployment pod spec: `spec.template.spec.imagePullSecrets: [{name: regcred}]`,
  then `kubectl apply`.
- **`no match for platform in manifest`** (containerd) or **`no matching manifest for
  linux/amd64 in the manifest list entries`** (Docker) — arch mismatch: image built on
  Apple Silicon (arm64) for an amd64 cluster. Rebuild with
  `bentoml containerize --opt platform=linux/amd64 <name>:<version>`, retag, push, and
  `kubectl rollout restart deployment/<deploy> -n <ns>`.
- **kind/minikube cluster can't pull at all** — local clusters can't see the host's
  Docker images. Use `kind load docker-image <image> --name <cluster>` or
  `minikube image load <image>`, and set `imagePullPolicy: IfNotPresent`.

## 2. CrashLoopBackOff

The answer is almost always in the logs of the *previous* attempt:

```sh
kubectl logs -n <ns> <pod> --previous --tail=200
kubectl describe pod -n <ns> <pod> | grep -A3 "Last State"
```

Read the log tail bottom-up (see "Reading BentoML logs" below). Likely causes:

- **Python traceback ending in `KeyError`/`ValidationError` about an env var, or code
  reading `os.environ[...]`** — missing env var or Secret. Check what the pod actually
  has: `kubectl exec` won't work on a crashing pod, so inspect the spec:
  ```sh
  kubectl get pod -n <ns> <pod> -o jsonpath='{.spec.containers[0].env}{"\n"}'
  kubectl get pod -n <ns> <pod> -o jsonpath='{.spec.containers[0].envFrom}{"\n"}'
  kubectl get secret -n <ns>
  ```
  Fix: create the missing secret (`kubectl create secret generic <name>-env
  --from-literal=HF_TOKEN=...` — ask the user for the value, never invent one), wire it
  via `envFrom.secretRef` in the Deployment, `kubectl apply`.
- **`NotFound: model ... does not exist` or an HF download error** — model missing.
  Models are baked into the image at `bentoml build` time by default; if the traceback
  shows a download attempt, the service downloads at runtime and likely needs
  `HF_TOKEN` (see above) or network egress. If the model should be baked in, the Bento
  was built without it — rebuild (`bentoml build`), re-containerize, push, restart.
- **`ModuleNotFoundError` / `ImportError`** — a Python dependency missing from the
  Bento's declared packages. Add it to `bentofile.yaml` (or the
  `bentoml.images.Image(...)` spec), rebuild, re-containerize, push, rollout restart.
- **`exec format error`** (often the only log line, or in describe events) — wrong CPU
  architecture; same fix as the arch mismatch in section 1
  (`bentoml containerize --opt platform=linux/amd64 ...`).
- **Exit code 137 with no traceback** — see section 3 (OOMKilled).
- Do **not** "fix" crashes by overriding `command:` in the pod spec — the image
  entrypoint already runs `serve`; overriding it is a common way to break the container.

## 3. OOMKilled

Confirm, then check current limits:

```sh
kubectl describe pod -n <ns> <pod> | grep -B2 -A5 "OOMKilled"
kubectl get pod -n <ns> <pod> -o jsonpath='{.spec.containers[0].resources}{"\n"}'
```

Likely cause: memory limit smaller than model weights + inference working set. Rule of
thumb: limit >= ~1.5-2x the model files' size on disk (weights expand in RAM). Fix by
raising `resources.limits.memory` (and `requests.memory`) in `k8s/deployment.yaml` and
`kubectl apply -n <ns> -f k8s/deployment.yaml`. If the node itself is too small, that
becomes a Pending problem (section 4). OOM during startup (kill within the first
minutes, logs stop mid model-load) is the same fix — it is not a probe problem.

## 4. Pending

The scheduler's reason is in events:

```sh
kubectl describe pod -n <ns> <pod> | grep -A10 Events
kubectl describe nodes | grep -A6 "Allocated resources"
```

- **`Insufficient cpu` / `Insufficient memory`** — requests exceed free node capacity.
  Lower `resources.requests` in `k8s/deployment.yaml` (keep limits >= requests) and
  apply, or the user must add/resize nodes.
- **`Insufficient nvidia.com/gpu`** — either no GPU nodes, or the NVIDIA device plugin
  is not installed so GPUs aren't advertised. Check:
  ```sh
  kubectl get nodes -o custom-columns='NAME:.metadata.name,GPU:.status.allocatable.nvidia\.com/gpu'
  ```
  If GPU nodes exist but show `<none>`, the device plugin is missing — installing it is
  the cluster admin's responsibility; tell the user rather than modifying their cluster.
  If the service doesn't actually need a GPU, remove the `nvidia.com/gpu` limit and apply.
- **`didn't match Pod's node affinity/selector`** — a `nodeSelector`/affinity in the
  Deployment matches no node. Compare against `kubectl get nodes --show-labels`; fix or
  remove the selector in the manifest and apply.
- **`FailedScheduling ... untolerated taint`** — target nodes are tainted (common on GPU
  pools). Add the matching `tolerations` to the pod spec and apply.

## 5. Running but not Ready (readiness/startup probe failing)

First decide: *still loading* vs *actually broken*.

```sh
kubectl describe pod -n <ns> <pod> | grep -iA4 "unhealthy\|probe failed"
kubectl logs -n <ns> <pod> --tail=50
```

- If logs show active model loading (download progress bars, checkpoint shards) and the
  pod is only a few minutes old: **wait**. The deploy skill's startupProbe budget is
  about 10 minutes (`periodSeconds: 5`, `failureThreshold: 120`). If loading genuinely
  needs longer, raise `failureThreshold` in `k8s/deployment.yaml` and apply — do not
  shrink or remove the probe.
- If the startup log line already appeared (see log hints below) but readiness still
  fails, ask `/readyz` yourself and **read the response body** — BentoML puts the reason
  there:
  ```sh
  kubectl port-forward -n <ns> <pod> 3100:3000 &   # local 3100: port 3000 is often squatted (section 10)
  for i in $(seq 1 10); do curl -so /dev/null http://localhost:3100/livez && break; sleep 1; done
  curl -si http://localhost:3100/readyz
  curl -si http://localhost:3100/livez
  kill %1
  ```
  Run every `port-forward ... & ... kill %1` sequence in this skill as ONE shell
  invocation — job IDs (`%1`) don't survive across separate tool calls — and always
  wait for the tunnel to bind (the retry loop above) before trusting a curl result.
  - `200` with empty/whitespace body (it is a single newline) → ready; if K8s still
    reports failure, the probe path/port in the manifest is wrong (must be `/readyz`
    on port 3000) — fix and apply.
  - `503` with `"Service is not ready because .__is_ready__() returns False."` → the
    user's own `__is_ready__` hook is returning False; read their `service.py`.
  - `503` with `"Runners are not ready."` → a dependent service in a multi-service Bento
    isn't up; check its logs.
  - Connection refused / connection reset / empty reply even after waiting for the
    tunnel (retry loop above) → server never bound port 3000; treat as CrashLoop
    (section 2). (Exact error text varies: kubectl may accept the local connection
    and then fail the forward.)

## 6. Rollout stuck

```sh
kubectl rollout status deployment/<deploy> -n <ns>
kubectl get rs -n <ns> -l app.kubernetes.io/name=<name>
kubectl get pods -n <ns> -l app.kubernetes.io/name=<name>
```

A stuck rollout means new-ReplicaSet pods never became Ready — diagnose those pods with
sections 1-5. If the newly pushed image is broken and the user wants the previous
version back while they fix it, prefer fixing the image ref in `k8s/deployment.yaml`
and re-applying; `kubectl rollout undo deployment/<deploy> -n <ns>` is acceptable
(it only rolls the Deployment back, deletes nothing) — confirm with the user first.
If the same tag was re-pushed with new content, nodes may run a stale cached image:
set a fresh unique tag instead of reusing tags.

## 7. Pods Ready but service unreachable

Check that the Service actually selects the pods:

```sh
kubectl get endpoints -n <ns> <name>   # deprecated on K8s >=1.33; modern equivalent:
kubectl get endpointslices -n <ns> -l kubernetes.io/service-name=<name>
kubectl get svc -n <ns> <name> -o yaml | grep -A5 "selector:\|ports:"
kubectl get pods -n <ns> -l app.kubernetes.io/name=<name> --show-labels
```

- **`ENDPOINTS <none>`** — Service selector doesn't match pod labels (deploy skill uses
  `app.kubernetes.io/name: <name>`), or matching pods aren't Ready (sections 2-5). Fix
  the selector/labels in `k8s/` and apply.
- **Endpoints exist but curl fails** — wrong port wiring. Service `targetPort` must be
  `3000` (or the port name `http`). Bypass the Service to isolate:
  ```sh
  kubectl port-forward -n <ns> <pod> 3100:3000 &        # pod direct (local 3100, see section 10)
  curl -sf --retry 5 --retry-connrefused http://localhost:3100/livez && echo POD-OK
  kill %1
  kubectl port-forward -n <ns> svc/<name> 3100:3000 &   # via the service (service port 3000, named http)
  curl -sf --retry 5 --retry-connrefused http://localhost:3100/livez && echo SVC-OK
  kill %1
  ```
  Pod OK + Service broken → fix Service `port`/`targetPort` and apply.
- **Ingress path broken (pod and svc both OK)**:
  ```sh
  kubectl get ingress -n <ns> -o wide
  kubectl describe ingress -n <ns> <name>
  kubectl get ingressclass
  ```
  Common: `ingressClassName` names a class that doesn't exist / no controller installed
  (ADDRESS stays empty), wrong `service.name`/`port` in the backend, or DNS for the host
  rule not pointing at the controller. Fix the ingress manifest and apply; installing an
  ingress controller is the user's call.

## 8. Reachable but inference returns 4xx/5xx

Reproduce with the real error body — BentoML returns JSON error details:

```sh
kubectl port-forward -n <ns> svc/<name> 3100:3000 &   # local 3100, see section 10
for i in $(seq 1 10); do curl -so /dev/null http://localhost:3100/livez && break; sleep 1; done
curl -si -X POST http://localhost:3100/<endpoint> \
  -H 'Content-Type: application/json' -d '{"<arg>": "<value>"}'
```

Then check the actual API schema the server exposes — never guess the payload shape.
(This block reuses the tunnel from the previous block — run both in the same shell
invocation, with `kill %1` only at the end:)

```sh
curl -s http://localhost:3100/docs.json | python3 -m json.tool | head -100   # OpenAPI spec
# Swagger UI is served at the service root path in a browser
kill %1
```

- **404** — wrong path. BentoML API endpoints are `POST /<method_name>` (e.g.
  `POST /summarize`); list them from the OpenAPI spec's `paths`.
- **400 / 422** — payload doesn't match the method signature: wrong field names, wrong
  types, or wrong content type. Match the request body schema in `docs.json` exactly;
  keys are the Python parameter names.
- **405** — used GET on an inference route; they are POST-only.
- **500** — server-side exception. The full traceback is in the pod logs at the moment
  of the request:
  ```sh
  kubectl logs -n <ns> <pod> --tail=50
  ```
- **503 / 429** — overloaded or concurrency limit hit; check for OOM/restarts
  (`kubectl get pods -n <ns>` RESTARTS column) or retry after load drops.

## 9. `kubectl apply` rejected a manifest (validation error)

`kubectl apply -f k8s/` applies file-by-file, so a rejection leaves a half-applied
state — run the Step 0 triage to see which objects exist. Common cause: invalid name —
K8s **Service** names are DNS-1035 (must start with a *letter*) while most other names
are DNS-1123, so e.g. `1st-model` fails as a Service name. Fix the name in the `k8s/`
manifests and re-apply. Renames create *new* objects: it is OK to `kubectl delete` the
misnamed objects **this deploy-skill run just created** and re-apply — that is the one
sanctioned delete; the never-delete rule still protects everything pre-existing.

## 10. port-forward "works" but responses look wrong

If curl returns 200 but the body isn't BentoML (another app's page, wrong JSON), the
local port is probably occupied by an unrelated process and the port-forward silently
failed to bind (its error is easy to miss when output is redirected). Check the tunnel
is alive (`kill -0 <port-forward-pid>`) and prefer an uncommon local port, e.g.
`kubectl port-forward -n <ns> svc/<name> 3100:3000` then curl `localhost:3100`.
Always verify response CONTENT (e.g. `/docs.json` mentions your endpoints), not just
the status code — then follow the tunnel-wait guidance in section 5.

## Reading BentoML logs (hints)

- **Healthy startup** ends with a line like:
  `Starting production HTTP BentoServer from "<bento_identifier>" listening on http://0.0.0.0:3000 (Press CTRL+C to quit)`
  If this line is present, the server bound the port — later failures are probe config,
  networking, or per-request errors, not startup.
- Before that line you'll see worker/model-loading output (framework logs, HF download
  progress). No startup line + logs stop mid-load → still loading, crashed during load
  (traceback), or OOMKilled (logs cut off with no traceback, exit 137).
- **Tracebacks go to the pod's stderr/stdout** — `kubectl logs` shows them; the last
  frame of the last traceback is the actual error. For request-time errors the
  traceback is logged when the request fails, so correlate by timestamp.
- Access logs print one line per request with method, path, and status — useful to
  confirm requests reach the pod at all (note: `/livez` and `/readyz` probe hits are
  excluded from access logs by default, so their absence is normal).

## Escalation

If none of the branches match, collect a bundle and reason from it (do not delete or
recreate resources speculatively):

```sh
kubectl get all -n <ns> -l app.kubernetes.io/name=<name> -o wide
kubectl describe deploy -n <ns> <deploy>
kubectl get events -n <ns> --sort-by=.lastTimestamp
kubectl logs -n <ns> <pod> --all-containers --timestamps --tail=300
```
