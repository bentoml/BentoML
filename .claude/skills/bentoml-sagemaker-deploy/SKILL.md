---
name: bentoml-sagemaker-deploy
description: >
  Deploy a BentoML service to an AWS SageMaker real-time inference endpoint
  (bring-your-own-container). Adapts the user's service to the SageMaker
  contract (port 8080, GET /ping, POST /invocations) with a small validated
  patch, containerizes and pushes to ECR via the bentoml-containerize skill,
  then creates the SageMaker model, endpoint config, and endpoint with the AWS
  CLI and verifies with a real invoke-endpoint call. Use when the user says
  things like "deploy my BentoML service to SageMaker", "create a SageMaker
  endpoint from my bento", "host my bento on SageMaker", or "serve my BentoML
  model as a SageMaker real-time endpoint". For plain VMs use
  bentoml-ec2-deploy; for Kubernetes use bentoml-k8s-deploy.
---

# Deploy a BentoML service to a SageMaker real-time endpoint

> For production / CI-CD deployments, generate a standalone script bundle (no agent needed at deploy time) with the `bentoml-deploy-scriptgen` skill.

You will adapt the user's BentoML service to SageMaker's bring-your-own-container
(BYOC) contract, build and push the image to **ECR in the endpoint's region**, and
create the endpoint with the AWS CLI: `create-model` → `create-endpoint-config` →
`create-endpoint` → `wait endpoint-in-service` → `invoke-endpoint`.

The SageMaker BYOC contract (verified against
[docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html)):
- SageMaker launches the container as **`docker run IMAGE serve`**.
- The container must serve HTTP on **port 8080**.
- **`GET /ping`** must return **200** (empty body is fine). The container gets
  **8 minutes after startup** to start passing pings, and each ping has a 2 s timeout.
- **`POST /invocations`** handles inference; it must respond **within 60 seconds**
  (hard platform limit) and request/response bodies are capped at **6 MB**.
- Env vars come from `ContainerDefinition.Environment` in `create-model` — the run
  command cannot be changed, so all configuration must be env-var-only.

How BentoML meets it (verified against BentoML source and by a local end-to-end test):
- The image entrypoint translates `docker run IMAGE serve` into `bentoml serve <bento>`
  (`src/bentoml/_internal/container/frontend/dockerfile/entrypoint.sh`).
- `bentoml serve` reads the port from the **`BENTOML_PORT`** env var (the `--port`
  option's envvar) — so `Environment={BENTOML_PORT=8080}` is the whole port fix.
- `@bentoml.api(route="/invocations")` puts an API method on POST /invocations.
- `@bentoml.asgi_app(...)` mounts a tiny Starlette app that serves GET /ping.

Safety and cost rules (apply throughout — no exceptions):
- **Before EVERY mutating AWS CLI command** (`ecr create-repository`, `iam create-role`,
  `iam put-role-policy`, `sagemaker create-*`, `sagemaker delete-*`, `ecr delete-*`):
  show the user the **exact command** plus a **cost note**, and wait for explicit
  confirmation. Read-only commands (`describe-*`, `list-*`, `get-*`,
  `sts get-caller-identity`, `wait`, `logs tail`) may run freely.
- **`create-endpoint` bills per instance-hour from InService until you delete the
  endpoint** — a forgotten ml.m5.large endpoint costs roughly $80-90/month. Never run
  it without an explicit "yes" after showing the instance type, count, and hourly rate.
- Track every resource you create (ECR repo, IAM role name, model name, endpoint config
  name, endpoint name) — the Teardown section operates on exactly that list and nothing else.
- Never delete or modify SageMaker/IAM/ECR resources this skill did not create in this session.

## Step 0 — Preflight checks

Run, and stop with a clear message if any check fails:

```bash
aws --version                       # AWS CLI v2 installed?
aws sts get-caller-identity         # credentials valid? note the Account ID
docker info --format '{{.ServerVersion}} {{.Architecture}}'   # daemon reachable (for build+push)
command -v bentoml && bentoml --version
```

Then **confirm the region WITH the user — never assume**:

```bash
aws configure get region            # show as the default suggestion, then ask
```

Set `AWS_REGION` and `ACCOUNT_ID` and use them explicitly everywhere:

```bash
AWS_REGION=us-east-1                                   # confirmed with the user
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
```

The image MUST live in **ECR in `$AWS_REGION`** — SageMaker requires the inference
image to be in the same region as the model/endpoint. Also note: SageMaker `ml.m5`,
`ml.c5`, `ml.g4dn`, etc. instances are **linux/amd64** — if building on Apple Silicon
the image must be containerized with `--opt platform=linux/amd64`.

## Step 1 — Adapt the service to the SageMaker contract

Read the user's `service.py`. Identify the `@bentoml.service` class and the primary
`@bentoml.api` inference method (its name, parameters, and return type). Then apply the
**additive patch** below (validated: it does not change any existing route — `/predict`,
`/readyz` etc. keep working). Show the user the diff before writing it.

**Legacy project check first:** if `service.py` has no `@bentoml.service`-decorated
class but instead builds a v1.1-style `svc = bentoml.Service(...)` with
`@svc.api(input=..., output=...)` handlers, this patch does NOT apply — tell the user
their project uses the legacy runner API and must be migrated to the v2 class-based SDK
before SageMaker deployment, and stop. Do not attempt to adapt a legacy project.

Insert near the top of `service.py` (Starlette is already a BentoML dependency — no
new packages):

```python
# --- SageMaker adaptation (added by bentoml-sagemaker-deploy) ---
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

async def _sagemaker_ping(request):
    return JSONResponse({})

_sagemaker_ping_app = Starlette(routes=[Route("/ping", _sagemaker_ping, methods=["GET"])])
# --- end SageMaker adaptation ---
```

Add one decorator line **above** the existing `@bentoml.service(...)` line:

```python
@bentoml.asgi_app(_sagemaker_ping_app, path="/")  # SageMaker adaptation: GET /ping -> 200
@bentoml.service(...)          # existing decorator, unchanged
class MyService:
```

Add one delegating method **inside** the class, mirroring the primary method's exact
signature (here the primary method is `def predict(self, text: str) -> dict`):

```python
    # --- SageMaker adaptation: POST /invocations alias ---
    @bentoml.api(route="/invocations")
    def invocations(self, text: str) -> dict:
        return self.predict.local(text)
    # --- end SageMaker adaptation ---
```

Three rules, all load-bearing:
- Mount the Starlette app at `path="/"`, NOT `path="/ping"`. Mounting a bare handler at
  `/ping` makes Starlette answer `GET /ping` with a **307 redirect** to `/ping/`, which
  fails SageMaker's health check (it requires a literal 200). At `path="/"` BentoML's
  `PassiveMount` claims only the routes the app defines, so nothing else is shadowed.
- Call the primary method through **`.local`** (`self.predict.local(...)`) — that invokes
  it in-process. If the primary method is `async`, make `invocations` async and `await` it.
- The `invocations` signature must mirror the primary method's parameters — that is what
  defines the JSON body SageMaker clients will send.

A full annotated copy of the patch is in
[templates/sagemaker_adaptation.py](templates/sagemaker_adaptation.py). If the user does
not want `service.py` touched at all, use the (also validated) wrapper-file variant in
[references/service-adaptation.md](references/service-adaptation.md) — same reference
covers async methods, multiple API methods, and non-JSON inputs.

**Validate locally NOW** (30 seconds, catches almost every later endpoint failure).
Run in one shell invocation from the project directory:

```bash
# LPORT is only for this LOCAL test — if 8080 is busy, use 18080 or any free port.
# The env var is what matters; the SageMaker container itself always uses 8080 internally.
LPORT=8080
BENTOML_PORT=$LPORT bentoml serve . > serve.log 2>&1 &  SERVE_PID=$!
for i in $(seq 1 30); do   # retry loop: model loading can take a while, don't trust a fixed sleep
  curl -sfo /dev/null http://127.0.0.1:$LPORT/ping && break; sleep 2
done
curl -s -o /dev/null -w 'ping:%{http_code}\n' http://127.0.0.1:$LPORT/ping      # expect ping:200
curl -s -X POST http://127.0.0.1:$LPORT/invocations \
  -H 'Content-Type: application/json' -d '{"text": "hello sagemaker"}'          # expect real JSON result
kill $SERVE_PID
# Server output is in ./serve.log — read it if either check fails.
```

Judge the `/invocations` response by its **content**. Do not continue until both checks pass.

## Step 2 — Build, containerize, push to ECR

Use the **`bentoml-containerize`** skill for the mechanics (build → containerize →
smoke test → push; its `references/registries.md` has the ECR section). The
SageMaker-specific requirements you must impose on it:

```bash
BENTO_TAG=$(bentoml build -o tag | grep '^__tag__:' | sed 's/^__tag__://')
REPO_NAME=$(echo "$BENTO_TAG" | cut -d: -f1 | tr '_.' '--')   # SageMaker names derived from this forbid '_'/'.' (ECR itself allows them)
IMAGE="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:$(echo "$BENTO_TAG" | cut -d: -f2)"

# amd64 is REQUIRED for standard ml.* instances (skip --opt platform if already on amd64)
bentoml containerize "$BENTO_TAG" -t "$IMAGE" --opt platform=linux/amd64

# create-repository is a MUTATING command — show it and confirm like any other mutation.
# Keep stderr visible: a permission error here must surface now, not as a confusing push failure.
aws ecr describe-repositories --repository-names "$REPO_NAME" --region "$AWS_REGION" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "$REPO_NAME" --region "$AWS_REGION"
aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
docker push "$IMAGE"
```

If the local machine is amd64, also re-run the Step 1 curls against the **container**
exactly as SageMaker will run it (this is the definitive contract test):

```bash
docker run -d --name sm-sim -e BENTOML_PORT=8080 -p 18080:8080 "$IMAGE" serve
for i in $(seq 1 30); do   # retry loop, up to ~60s — model loading takes time
  curl -sfo /dev/null http://127.0.0.1:18080/ping && break; sleep 2
done
curl -s -o /dev/null -w 'ping:%{http_code}\n' http://127.0.0.1:18080/ping
curl -s -X POST http://127.0.0.1:18080/invocations -H 'Content-Type: application/json' -d '<payload>'
docker rm -f sm-sim
```

## Step 3 — IAM execution role

SageMaker needs an execution role (trusting `sagemaker.amazonaws.com`) to pull the ECR
image and write CloudWatch logs. **Permissions heads-up:** creating the role needs
`iam:CreateRole` + `iam:PutRolePolicy` on the caller, and `create-model` (Step 4) needs
`iam:PassRole` on the chosen role. Common setups like the AWS-managed
**PowerUserAccess** policy have **no IAM write permissions at all** — so always
**prefer an existing role**.

Find candidates by **trust policy**, not by name (any role trusting
`sagemaker.amazonaws.com` works, whatever it is called; this needs only IAM read
access, which PowerUser-style setups do have):

```bash
aws iam list-roles \
  --query "Roles[?AssumeRolePolicyDocument.Statement[?contains(to_string(Principal.Service), 'sagemaker.amazonaws.com')]].[RoleName,Arn]" \
  --output table
```

(The `contains(to_string(...))` form also matches trust statements whose
`Principal.Service` is a list of services.) A listed role with
`AmazonSageMakerFullAccess` — or the ECR pull + CloudWatch actions — attached works.
Show the list, let the user pick, set `ROLE_ARN`.

If none exists, create a minimal one (trust policy + the AWS-documented CreateModel
permission set: CloudWatch metrics/logs + ECR pull) — exact JSON and commands in
[references/aws-setup.md](references/aws-setup.md). **If `iam create-role` returns
`AccessDenied`:**
1. Ask the user whether they have an execution role ARN the scan could not see
   (cross-account listing gaps, restrictive `iam:ListRoles`) — any ARN whose trust
   policy includes `sagemaker.amazonaws.com` is fine.
2. If they have none, print the ready-to-send **admin snippet** from
   `references/aws-setup.md` (the two policy JSONs plus the two `aws iam` commands)
   for them to forward to an account admin, then **STOP — do not attempt workarounds**.

Set `ROLE_ARN` once a role exists. Note: a freshly created role can take ~10 s to
propagate; retry `create-model` once if it fails with an assume-role error.

## Step 4 — Create the SageMaker model

`BENTOML_PORT=8080` in the container environment is the entire BentoML-side port
configuration — the run command stays SageMaker's fixed `serve`.

```bash
MODEL_NAME="${REPO_NAME}-model"          # names: max 63 chars, pattern ^[a-zA-Z0-9](-*[a-zA-Z0-9])*
                                         # (alphanumerics and '-'; no leading/trailing hyphen, no '_' or '.')
aws sagemaker create-model --region "$AWS_REGION" \
  --model-name "$MODEL_NAME" \
  --primary-container "Image=${IMAGE},Environment={BENTOML_PORT=8080}" \
  --execution-role-arn "$ROLE_ARN"
```

Extra runtime env vars (e.g. `HF_TOKEN` for models downloaded at startup) go in the same
`Environment={...}` map — but AWS says don't put sensitive data there (it is visible to
anyone who can `describe-model`); prefer baking models into the image (the BentoML default).

## Step 5 — Create the endpoint config

Default: **1 × ml.m5.large** (2 vCPU, 8 GiB — CPU inference). Ask before choosing
anything else; sizing/cost table in [references/aws-setup.md](references/aws-setup.md).
If model loading takes minutes, raise `ContainerStartupHealthCheckTimeoutInSeconds`
(default budget is 8 minutes, configurable 60-3600):

```bash
EPC_NAME="${REPO_NAME}-config"
aws sagemaker create-endpoint-config --region "$AWS_REGION" \
  --endpoint-config-name "$EPC_NAME" \
  --production-variants "VariantName=AllTraffic,ModelName=${MODEL_NAME},InitialInstanceCount=1,InstanceType=ml.m5.large,ContainerStartupHealthCheckTimeoutInSeconds=600"
```

## Step 6 — Create the endpoint (BILLING STARTS HERE — confirm first)

Show the user: region, instance type, count, and the approximate hourly price
(ml.m5.large ≈ $0.115/h ≈ $83/month in us-east-1; check
https://aws.amazon.com/sagemaker/pricing/ for their region), and state plainly:
**"This bills hourly until you run the teardown commands."** Only after an explicit yes:

```bash
EP_NAME="${REPO_NAME}-endpoint"
aws sagemaker create-endpoint --region "$AWS_REGION" \
  --endpoint-name "$EP_NAME" \
  --endpoint-config-name "$EPC_NAME"

aws sagemaker wait endpoint-in-service --endpoint-name "$EP_NAME" --region "$AWS_REGION"
aws sagemaker describe-endpoint --endpoint-name "$EP_NAME" --region "$AWS_REGION" \
  --query '{Status:EndpointStatus,Reason:FailureReason}'
```

Creation typically takes 5-10 minutes; `wait` polls until InService (or fails — then go
to Troubleshooting). If the endpoint ends up `Failed`, **it still exists — delete it**
(see Teardown) before retrying with a fixed image/config.

## Step 7 — Invoke and verify

Derive the JSON payload from the `invocations` method's parameters. AWS CLI v2 needs
`--cli-binary-format raw-in-base64-out` to pass a raw JSON string, and the output file
is a required positional argument:

```bash
aws sagemaker-runtime invoke-endpoint \
  --region "$AWS_REGION" \
  --endpoint-name "$EP_NAME" \
  --content-type application/json \
  --cli-binary-format raw-in-base64-out \
  --body '{"text": "SageMaker runs BentoML containers happily"}' \
  /tmp/sm-response.json
cat /tmp/sm-response.json; echo
```

**Judge the response by its content, not the exit code** — only a correct inference
result proves the deployment works. A `ModelError` (HTTP 424) means the container
returned 4xx/5xx: read its `OriginalStatusCode`/`OriginalMessage` and check the payload
against the method signature. Remember the platform limits: 60 s per request, 6 MB body.

Report to the user: the endpoint name/region, this invoke command as the usage example,
and the Teardown block below (verbatim) so they can stop billing at any time.

## Teardown (run when the endpoint is no longer needed — REQUIRED knowledge)

Deleting the **endpoint** is what stops billing; config and model are free but keep the
account tidy. Delete in this order:

```bash
aws sagemaker delete-endpoint        --endpoint-name        "$EP_NAME"   --region "$AWS_REGION"   # stops billing
aws sagemaker wait endpoint-deleted  --endpoint-name        "$EP_NAME"   --region "$AWS_REGION"
aws sagemaker delete-endpoint-config --endpoint-config-name "$EPC_NAME"  --region "$AWS_REGION"
aws sagemaker delete-model           --model-name           "$MODEL_NAME" --region "$AWS_REGION"

# Optional — only if this skill created the ECR repo in this session:
aws ecr delete-repository --repository-name "$REPO_NAME" --region "$AWS_REGION" --force

# Optional — ONLY if this skill created the IAM role in this session
# ($ROLE_NAME/$POLICY_NAME as set in references/aws-setup.md; never delete a pre-existing role):
aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name "$POLICY_NAME"
aws iam delete-role --role-name "$ROLE_NAME"
```

## Troubleshooting (compact)

First stop for anything after `create-endpoint`: **CloudWatch container logs** —

```bash
aws logs tail "/aws/sagemaker/Endpoints/${EP_NAME}" --region "$AWS_REGION" --since 1h
```

| Symptom | Likely cause → fix |
|---|---|
| Endpoint stuck `Creating` then `Failed`, FailureReason mentions ping | Container never passed `GET /ping` in 8 min. Either the port fix is missing (logs say `listening on http://localhost:3000` → `BENTOML_PORT=8080` absent from the model's `Environment`) or the `/ping` route is missing/redirecting (adaptation not applied, or mounted at `/ping` instead of `/` → 307). Reproduce locally with `docker run -e BENTOML_PORT=8080 -p 18080:8080 IMAGE serve`. |
| Logs show `exec format error` or container exits instantly | Image is arm64. Rebuild with `bentoml containerize ... --opt platform=linux/amd64`, push, create a **new** model (+config+endpoint — SageMaker models are immutable). |
| `Failed` with ECR/access error in FailureReason | Execution role can't pull: wrong region ECR repo (must equal endpoint region), or role missing the ECR pull actions — see `references/aws-setup.md`. |
| Slow model load kills startup | Raise `ContainerStartupHealthCheckTimeoutInSeconds` (max 3600) in a new endpoint config. |
| `invoke-endpoint` → `ModelError` (424) | Container returned 4xx/5xx. `OriginalMessage` has the body: usually payload keys not matching the `invocations` parameters, or missing `--content-type application/json`. 5xx → read CloudWatch logs. |
| `invoke-endpoint` hangs/times out | 60 s hard limit per invocation; payloads capped at 6 MB. Long-running inference does not fit real-time endpoints. |
| Permission-denied errors inside the container in CloudWatch logs | BentoML images run as non-root user `bentoml`; AWS recommends root. This worked in our containerized tests, but if writes outside `/home/bentoml` fail, that is why. |

Model/config/endpoint are immutable-ish: to ship a fixed image, create a new model and
endpoint config, then either `update-endpoint` or delete + recreate the endpoint.

## Out of scope

Serverless inference, async inference, multi-model endpoints, and autoscaling are out of
scope — one-liners only: serverless (`ServerlessConfig` in the endpoint config), async
(`CreateEndpointConfig` `AsyncInferenceConfig`), autoscaling
(`aws application-autoscaling` on `sagemaker:variant:DesiredInstanceCount`). Stop there.

## References (read on demand)

- [references/service-adaptation.md](references/service-adaptation.md) — wrapper-file variant (no edits to `service.py`), async methods, multiple APIs, non-JSON input, and exactly why each piece of the patch is required (with the source-level mechanics).
- [references/aws-setup.md](references/aws-setup.md) — minimal IAM role JSON + creation commands, instance type/cost table, CloudWatch log debugging, ECR specifics.
- [templates/sagemaker_adaptation.py](templates/sagemaker_adaptation.py) — the full adaptation snippet with placeholders.
