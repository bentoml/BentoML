---
name: bentoml-k8s-deploy
description: Deploy a containerized BentoML service to a vanilla Kubernetes cluster using plain kubectl manifests (no Helm, no operators, no BentoCloud/Yatai). Takes a pushed container image (built by the bentoml-containerize skill or `bentoml containerize`), generates Deployment/Service/Ingress manifests under k8s/, applies them, and verifies the rollout with a real inference request. Use when the user says things like "deploy my BentoML service to Kubernetes", "deploy this bento image to my cluster", "run my bento on k8s", "create k8s manifests for my bento", or "expose my BentoML service in Kubernetes".
---

# Deploy a BentoML service to vanilla Kubernetes

You will take a **pushed container image reference** (e.g. `ghcr.io/acme/summarization:v1`,
produced by the `bentoml-containerize` skill), generate plain Kubernetes manifests from the
templates in this skill's `templates/` directory, apply them with `kubectl`, and verify the
service end to end.

> For production / CI-CD deployments, generate a standalone script bundle (no agent needed at deploy time) with the `bentoml-deploy-scriptgen` skill.

Facts that are always true for images built by `bentoml containerize`:
- The HTTP server listens on **port 3000**.
- Health endpoints: **`/livez`** (liveness), **`/readyz`** (readiness). `/metrics` serves Prometheus metrics.
- The image entrypoint already starts the server — **never set `command:`/`args:`** in the pod spec.
- Models are usually baked into the image; models downloaded at runtime (e.g. gated HF models) need env vars such as `HF_TOKEN` via a Kubernetes Secret.

Safety rules (apply throughout):
- **Never apply anything before the user has confirmed the kubectl context AND namespace.**
- Never `kubectl delete` anything this skill did not create in this session.
- Never write secret values into files on disk — secrets are created only with `kubectl create secret ... --from-literal`.

## Step 0 — Preflight checks

Run, and stop with a clear message if any check fails:

```bash
kubectl version --client        # kubectl installed?
kubectl config get-contexts     # list contexts; note the current one (*)
```

Then **ask the user which context to use — never assume the current context is the
intended cluster**, even if there is only one. Show them the list and get an explicit
answer. Use `--context <ctx>` on every subsequent kubectl command (do not silently
switch their current context with `kubectl config use-context`).

With the confirmed context, verify connectivity and permissions:

```bash
kubectl --context <ctx> get nodes -L kubernetes.io/arch   # cluster reachable? note node arch
kubectl --context <ctx> auth can-i create deployment -n <ns>   # run once the namespace is known (Step 1)
```

If the image was built on Apple Silicon and the `ARCH` column above shows `amd64`
nodes, warn the user now: the image must have been containerized with
`--opt platform=linux/amd64`, otherwise pods will crash with "exec format error".

## Step 1 — Gather deployment parameters

Ask the user for anything you cannot detect. Present defaults and let them accept in one go:

| Parameter | Default | Notes |
|---|---|---|
| Image reference | (required) | Full pushed ref incl. registry + tag. If the user only ran `bentoml containerize`, the image may exist only locally — it must be pushed (or `kind load` / `minikube image load` for local clusters; see `references/private-registries.md`). |
| Service name | bento name, DNS-1035-sanitized | Prefer the suggested name from the `bentoml-containerize` handoff (bento name, `_`→`-`); the image repo basename works only when it is meaningful. Must be **DNS-1035** (see naming rule below), e.g. `summarization:v1` → `summarization`. |
| Namespace | `default` | Offer to create a dedicated one. |
| Replicas | `1` | |
| CPU request/limit | `500m` / `2` | |
| Memory request/limit | `1Gi` / `4Gi` | Size to the model: OOMKilled pods usually mean the limit is below model memory. |
| GPU | none | If needed: how many `nvidia.com/gpu` per pod. First read `references/gpu-scheduling.md` and check the cluster actually advertises the resource. |
| Env vars (plain) | none | Non-sensitive config only. |
| Secrets | none | Tokens/keys (e.g. `HF_TOKEN`). Handled in Step 2, never written to manifests. |
| Private registry? | detect/ask | If pulling needs auth → imagePullSecret, Step 2. See `references/private-registries.md`. |
| Exposure | `port-forward` | One of: port-forward (testing), NodePort, LoadBalancer, Ingress. Read `references/exposure-options.md` before recommending; check what the cluster supports (`kubectl --context <ctx> get ingressclass`, cloud LB availability). |

**Service naming rule (DNS-1035 — stricter than most K8s names).** Kubernetes
`Service` objects require an RFC 1035 label: lowercase alphanumerics and `-`, **must
start with a letter**, must end with an alphanumeric, max 63 chars. Since
`{{SERVICE_NAME}}` names the Service too, always enforce DNS-1035:

- Validate the derived name against `^[a-z]([-a-z0-9]*[a-z0-9])?$` before rendering.
- If the image repo basename starts with a digit or is a meaningless random name (e.g. a
  ttl.sh UUID like `1f0d3c2a-...`), do NOT use it — derive the name from the bento name
  instead (the service class name lowercased/snake_cased, e.g. bento `text_summarizer:v1`),
  converting underscores to `-` (→ `text-summarizer`). If nothing usable can be derived,
  ask the user.
- **`kubectl apply --dry-run=client` does NOT reliably catch DNS-1035 violations for
  Services** — a bad name passes dry-run, then the real apply rejects the Service while
  the Deployment succeeds, leaving a half-applied state (see Step 4). Validate the name
  yourself; never rely on dry-run for this.

## Step 2 — Namespace and Secrets (imperative, before manifests)

Create the namespace if it does not exist (or render `templates/namespace.yaml` into
`k8s/namespace.yaml` if the user wants it tracked in git):

```bash
kubectl --context <ctx> get namespace <ns> || kubectl --context <ctx> create namespace <ns>
```

**Application secrets** — one Secret holding all sensitive env vars. Ask the user to
provide values (or confirm reading them from their shell env); never echo values back,
never put them in a file:

```bash
kubectl --context <ctx> create secret generic <service-name>-env -n <ns> \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  --from-literal=OTHER_KEY="..."
```

**Registry pull secret** (only for private registries):

```bash
kubectl --context <ctx> create secret docker-registry <service-name>-regcred -n <ns> \
  --docker-server=<registry> --docker-username=<user> \
  --docker-password="$REGISTRY_TOKEN"
```

Registry-specific details (GHCR, ECR, Docker Hub, local kind/minikube):
`references/private-registries.md`.

## Step 3 — Render manifests into the project's `k8s/` directory

For each needed manifest, in this exact order:

1. **Render**: run sed on the file in this skill's `templates/` directory, substituting
   every placeholder you have a value for (including placeholders inside OPTIONAL blocks
   that apply), and write the output to `<project>/k8s/<file>.yaml`. Never modify the
   templates themselves.
2. **Prune**: edit the `k8s/` copy in place and **delete every `OPTIONAL` block that does
   not apply**, including its comment lines (imagePullSecrets, env, envFrom, GPU line,
   Ingress TLS). Each block's comment states exactly how many lines to delete.
3. **Check**: a finished manifest must contain **no `{{...}}` left** (verified below).

Which files to write:
- `k8s/deployment.yaml` — always.
- `k8s/service.yaml` — always (`SERVICE_TYPE`: `ClusterIP` for port-forward and Ingress; `NodePort` or `LoadBalancer` if that exposure was chosen).
- `k8s/namespace.yaml` — only if the user wants the namespace in git (otherwise Step 2 already created it).
- `k8s/ingress.yaml` — only for Ingress exposure.

### Placeholder reference (all templates)

| Placeholder | Used in | Meaning / example |
|---|---|---|
| `{{SERVICE_NAME}}` | all | DNS-1035 name (see naming rule, Step 1), e.g. `summarization`. Also used as label value. |
| `{{NAMESPACE}}` | all | Target namespace, e.g. `ml-services`. |
| `{{IMAGE}}` | deployment | Full pushed image ref, e.g. `ghcr.io/acme/summarization:v1`. |
| `{{REPLICAS}}` | deployment | Integer, e.g. `1`. |
| `{{CPU_REQUEST}}` / `{{CPU_LIMIT}}` | deployment | e.g. `500m` / `2`. |
| `{{MEMORY_REQUEST}}` / `{{MEMORY_LIMIT}}` | deployment | e.g. `1Gi` / `4Gi`. |
| `{{IMAGE_PULL_SECRET}}` | deployment (optional) | docker-registry Secret name from Step 2, e.g. `summarization-regcred`. |
| `{{ENV_NAME}}` / `{{ENV_VALUE}}` | deployment (optional) | One plain env var; duplicate the pair per extra var. |
| `{{ENV_SECRET_NAME}}` | deployment (optional) | generic Secret name from Step 2, e.g. `summarization-env`. |
| `{{GPU_COUNT}}` | deployment (optional) | Integer GPUs per pod, e.g. `1`. |
| `{{SERVICE_TYPE}}` | service | `ClusterIP`, `NodePort`, or `LoadBalancer`. |
| `{{INGRESS_CLASS_NAME}}` | ingress | From `kubectl --context <ctx> get ingressclass`, e.g. `nginx`. |
| `{{INGRESS_HOST}}` | ingress | e.g. `summarization.example.com`. |
| `{{TLS_SECRET_NAME}}` | ingress (optional) | Existing TLS Secret, e.g. `summarization-tls`. |

Example — step 1 (render). Add extra `-e` expressions for any optional placeholders that
apply (`{{IMAGE_PULL_SECRET}}`, `{{ENV_SECRET_NAME}}`, `{{GPU_COUNT}}`, ...):

```bash
mkdir -p k8s
sed -e 's|{{SERVICE_NAME}}|summarization|g' \
    -e 's|{{NAMESPACE}}|ml-services|g' \
    -e 's|{{IMAGE}}|ghcr.io/acme/summarization:v1|g' \
    -e 's|{{REPLICAS}}|1|g' \
    -e 's|{{CPU_REQUEST}}|500m|g'  -e 's|{{CPU_LIMIT}}|2|g' \
    -e 's|{{MEMORY_REQUEST}}|1Gi|g' -e 's|{{MEMORY_LIMIT}}|4Gi|g' \
    <path-to-skill>/templates/deployment.yaml > k8s/deployment.yaml
```

Then step 2 (prune): edit `k8s/deployment.yaml` in place (your file-editing tool, or
`sed -i`) to delete the non-applicable OPTIONAL blocks — e.g. with no private registry,
no env vars, no secrets and no GPU, delete all four blocks listed in the template
comments. Only after pruning does the placeholder check below pass.

Sanity-check every rendered file before showing it to the user:

```bash
grep -n '{{' k8s/*.yaml && echo "ERROR: unreplaced placeholders" || echo OK
kubectl --context <ctx> apply --dry-run=client -f k8s/
```

Suggest adding `k8s/` to `.gitignore` only if it contains environment-specific values the
user does not want committed; otherwise committing it is fine (it contains no secrets).

## Step 4 — Confirm, then apply

Show the user the rendered manifests plus a one-line summary —
**`context=<ctx>  namespace=<ns>  image=<image>`** — and wait for explicit confirmation.
Then:

```bash
kubectl --context <ctx> apply -n <ns> -f k8s/
```

**If apply partially fails** (some manifests created, one rejected — e.g. an invalid
Service name that dry-run did not catch): fix the offending manifest, then re-apply the
whole `k8s/` directory. Two rules for the cleanup:

- Resources **this skill just created in this run** are yours to fix: deleting and
  re-applying them is allowed and is NOT covered by the "never delete" safety rule —
  that rule protects pre-existing resources you did not create.
- If the fix involves **renaming** (e.g. correcting `{{SERVICE_NAME}}`), `kubectl apply`
  would create a second object under the new name instead of updating the old one —
  first delete the just-created misnamed objects
  (`kubectl --context <ctx> delete -n <ns> deployment/<old-name> service/<old-name> ...`,
  only the ones created moments ago in this run), update the name in **all** manifests
  in `k8s/`, then re-apply the whole directory.

## Step 5 — Verify

1. Wait for the rollout (budget matches the startupProbe: up to ~10 min for model loading):

```bash
kubectl --context <ctx> rollout status deployment/<service-name> -n <ns> --timeout=600s
```

If it stalls, inspect
(`kubectl --context <ctx> get pods -n <ns> -l app.kubernetes.io/name=<service-name>`,
`kubectl --context <ctx> describe pod <pod> -n <ns>`,
`kubectl --context <ctx> logs <pod> -n <ns>`) and hand off to the
`bentoml-k8s-troubleshoot` skill for diagnosis.

2. Readiness check plus **one real inference request** through a port-forward. Derive
   the inference endpoint and payload from the user's `service.py` (each `@bentoml.api`
   method `def foo(self, text: str)` is `POST /foo` with JSON body `{"text": "..."}`);
   if the source is not available, fetch the live schema from
   `http://127.0.0.1:3100/docs.json` (while the port-forward is up) or ask the user.
   If you need the schema first, run the block below once with only the readyz check
   plus a `curl -s http://127.0.0.1:3100/docs.json`, then re-run it with the inference
   call filled in.

   Forward to an **uncommon local port (3100)**, not local 3000: port 3000 is commonly
   occupied by dev servers, and if the port-forward fails to bind, your curls silently
   hit whatever local process squats there — producing convincing but fake results.
   For the same reason the block checks that the port-forward process is still alive
   before trusting any curl.

   The whole sequence below — port-forward, checks, kill — **must run in ONE shell
   invocation**: the background process and `$PF_PID` do not survive across separate
   shell calls.

```bash
PF_ERR=$(mktemp)
kubectl --context <ctx> port-forward svc/<service-name> -n <ns> 3100:3000 >"$PF_ERR" 2>&1 &
PF_PID=$!
OK=""
for i in $(seq 1 15); do
  kill -0 $PF_PID 2>/dev/null || { echo "port-forward exited:"; cat "$PF_ERR"; exit 1; }
  curl -sfo /dev/null http://127.0.0.1:3100/readyz && { OK=1; break; }
  sleep 2
done
[ -n "$OK" ] || { echo "not ready after 30s; port-forward log:"; cat "$PF_ERR"; kill $PF_PID; exit 1; }
echo READY

# Real inference request — substitute the endpoint/payload derived above:
curl -s -X POST http://127.0.0.1:3100/summarize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Kubernetes is an open-source container orchestration system."}'

kill $PF_PID
rm -f "$PF_ERR"
```

**Judge the inference response by its content, not the status code.** Only a correct
inference result proves the deployment works: HTTP 200 alone proves nothing, and a
4xx/5xx or error body means the payload or service needs fixing. If the output looks
unrelated to the service's API, suspect a local port squatter — confirm the port-forward
owns the port (`kill -0 $PF_PID`, check `$PF_ERR`) before believing anything.

## Step 6 — Tell the user how to reach the service

Print the access instructions matching the chosen exposure (full details and commands in
`references/exposure-options.md`):

- **port-forward**: `kubectl --context <ctx> port-forward svc/<service-name> -n <ns> 3000:3000` → `http://127.0.0.1:3000`
- **NodePort**: `http://<node-ip>:<node-port>` (get with `kubectl --context <ctx> get svc <service-name> -n <ns>` and `kubectl --context <ctx> get nodes -o wide`)
- **LoadBalancer**: external IP/hostname from `kubectl --context <ctx> get svc <service-name> -n <ns> -w`
- **Ingress**: `http(s)://<INGRESS_HOST>/` once DNS points at the ingress controller

Also mention: interactive API docs at `/` (Swagger UI), Prometheus metrics at `/metrics`.

## Out of scope

Autoscaling, scale-to-zero, canary/blue-green rollouts, and metric-based scaling were
BentoCloud features and are **not** part of this skill. If the user asks about scaling,
point them at a standard CPU-based HorizontalPodAutoscaler
(`kubectl --context <ctx> autoscale deployment <service-name> -n <ns> --min=1 --max=5 --cpu-percent=80`)
and stop there.

## References (read on demand)

- `references/exposure-options.md` — choosing and configuring port-forward / NodePort / LoadBalancer / Ingress, TLS, inference-friendly timeouts.
- `references/private-registries.md` — imagePullSecrets for GHCR/ECR/Docker Hub/self-hosted; kind/minikube local image loading; ttl.sh for throwaway tests.
- `references/gpu-scheduling.md` — verifying the NVIDIA device plugin, requesting `nvidia.com/gpu`, node selectors and taints.
