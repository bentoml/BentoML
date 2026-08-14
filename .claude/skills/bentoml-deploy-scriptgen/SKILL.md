---
name: bentoml-deploy-scriptgen
description: >
  Generate a standalone, committable production deploy-script bundle
  (deploy/deploy.py + deploy.config.json + rendered k8s manifests) that
  builds, containerizes, pushes, deploys, and verifies a BentoML service
  without any agent involved — runnable from a terminal or CI/CD. Use when
  the user says things like "generate a deployment script", "deploy from
  CI/CD", "set up a production deployment pipeline", "deploy without the
  agent", "give me a script I can commit to deploy this", or "automate my
  BentoML deploys". Complements the interactive skills: bentoml-containerize,
  bentoml-k8s-deploy, bentoml-ec2-deploy, and bentoml-sagemaker-deploy do a
  one-off deploy with you in the loop; this skill emits scripts that repeat
  it forever. Kubernetes, EC2, and SageMaker targets.
---

# Generate a production deploy-script bundle

You are a **script generator, not a script author**. The Python files under
this skill's `templates/deploy/` are static, reviewed, e2e-tested artifacts.

**HARD RULE — copy the `.py` files VERBATIM. Never edit, patch, "improve",
or regenerate their code, and never write new script logic for the user.**
Everything user-specific flows through exactly two rendered files
(`deploy.config.json`, `README.md`) plus the k8s manifests. If a user need
cannot be expressed in `deploy.config.json`, say so and report it as a
limitation — do not fork the templates.

What the generated bundle does at runtime (all driven by config):

- `python3 deploy/deploy.py --target k8s` → preflight → `bentoml build
  --version <tag>` (tag defaults to the project's short git SHA) →
  `bentoml containerize -t <registry>/<repo>:<tag>` → push (ECR login +
  describe-or-create handled automatically) → `kubectl apply` of the
  manifests in `deploy/k8s/` with the image sentinel `__DEPLOY_IMAGE__`
  rewritten in memory → rollout status → `/readyz` + inference smoke test
  through a port-forward.
- `python3 deploy/deploy.py --target ec2` → same preflight/build/push, then
  per host over SSH: optional ECR token-over-stdin login → `docker pull` →
  `rm -f` + `run -d --restart unless-stopped -p <host_port>:3000` (secret
  env values travel via stdin, never in a command line) → container-Up gate
  → `/readyz` + inference smoke test through a hardened SSH tunnel (or
  direct HTTP). **Existing instances only — the script never provisions
  EC2.**
- `python3 deploy/deploy.py --target sagemaker` → same preflight/build/push
  (the image **must** be ECR in the endpoint's region), then per-tag
  `create-model` (BENTOML_PORT=8080 always injected) + endpoint config
  named `<endpoint_name>-<tag>` → **create the endpoint on the first run,
  `update-endpoint` on every later one** (Failed endpoints are
  delete+recreated; concurrent operations abort with exit 6) →
  `wait endpoint-in-service` with a Python-side timeout →
  `invoke-endpoint` smoke test. **The execution role must already exist —
  the script never creates IAM**, and the service must already carry the
  SageMaker adaptation (`/ping` + `/invocations`) from the interactive
  skill's Step 1.
- Flags: `--check-only` (with `--local-only` for credential-free CI PR
  gates), `--skip-build` (which preflights that the image actually exists),
  `--image REF`, `--version TAG`, `--no-verify`, `--config`,
  `--output-json`. Exit codes 0..7 and a JSON summary as the last stdout
  line (contract documented in the bundle's README).

## Step 0 — Locate the project and confirm scope

Find the BentoML project (directory containing `service.py` /
`bentofile.yaml`, same detection as the `bentoml-containerize` skill). The
bundle is generated into `<project>/deploy/`. If `deploy/` already exists,
show what is there and get explicit confirmation before overwriting.

Scope: the **k8s**, **ec2**, and **sagemaker** targets all generate working
scripts. Standing prerequisites the script cannot create for the user:

- ec2 deploys to **existing instances only**; if an instance must be
  created, provision it first with the interactive `bentoml-ec2-deploy`
  skill, then generate this bundle against the resulting host(s).
- sagemaker needs an **existing execution role** (the script never creates
  IAM — find or request one via the interactive `bentoml-sagemaker-deploy`
  skill's Step 3), and the service must **already carry the SageMaker
  adaptation** (`GET /ping` + `POST /invocations`, applied and locally
  validated by that skill's Step 1). If `service.py` lacks the adaptation,
  run that step first — the generated script deploys the image as-is and an
  unadapted service only fails minutes later on the endpoint's health
  checks.

## Step 1 — Gather parameters (one round of questions)

Same conventions as the interactive deploy skills. Detect what you can
(read `service.py` for the service class and `@bentoml.api` methods; run
`kubectl config get-contexts` for the context list — **never assume the
current context**), then ask for the rest in one go:

| Parameter | Placeholder | Default / notes |
|---|---|---|
| Build target | `project.service` (ships `null`) | `module:Class`, e.g. `service:TextPipeline`. Leave `null` if `bentoml build` finds it alone; required when `service.py` defines several services. |
| Service name | `{{SERVICE_NAME}}` | Bento name, `_`→`-`, must match DNS-1035 `^[a-z]([-a-z0-9]*[a-z0-9])?$` (same naming rule as bentoml-k8s-deploy Step 1). Names the Deployment and Service. |
| Registry host | `{{IMAGE_REGISTRY}}` | e.g. `123456789012.dkr.ecr.us-west-1.amazonaws.com`, `ghcr.io`, `docker.io`. `""` for local-only images. |
| Repository | `{{IMAGE_REPOSITORY}}` | e.g. `acme/text-pipeline`. Lowercase. |
| Registry type | `{{REGISTRY_TYPE}}` | `ecr` (auth + repo-create automated), `generic` (user must `docker login`), or `none` (kind/minikube local load — nothing pushed; the user must load the image and set `image.local_image_preloaded` to `true`, which preflight enforces). |
| Platform | `image.platform` (ships `null`) | Set `"linux/amd64"` when cluster nodes are amd64 and builds may run on arm64; leave `null` when archs match. |
| kubectl context | `{{K8S_CONTEXT}}` | From `kubectl config get-contexts`; user must confirm explicitly. |
| Namespace | `{{NAMESPACE}}` | Must exist (preflight checks it; the script never creates namespaces). |
| Replicas / CPU / memory / GPU | (manifest) | Same defaults as bentoml-k8s-deploy: 1, `500m`/`2`, `1Gi`/`4Gi`, none. |
| Exposure | (manifest) | ClusterIP (port-forward) / NodePort / LoadBalancer / Ingress — same rules as bentoml-k8s-deploy. |
| Fixed NodePort | `targets.k8s.node_port` (ships `null`) | Only for NodePort exposure when the user wants a stable URL: pick 30000–32767, set the config key AND render the same value as a `nodePort:` line under the Service port (preflight cross-checks them). Leave `null` to let the cluster allocate (kubectl apply preserves an existing allocation). |
| Pull secret | `targets.k8s.image_pull_secret` (ships `null`) | Secret name if the registry is private and the cluster cannot pull natively (EKS→ECR usually can). The script checks existence and warns when an ECR-token secret is older than 11 h (heuristic — see the bundle README); creating/refreshing it stays a manual/CI step. |
| Inference smoke test | `{{INFERENCE_PATH}}` `{{INFERENCE_BODY}}` `{{EXPECT_SUBSTRING}}` | Derive from a `@bentoml.api` method: path `/<method>`, JSON body of its params, and a substring the response must contain (e.g. a result key). Set `"inference": null` only if the user declines. |

The k8s rows (context, namespace, replicas/resources, exposure, NodePort,
pull secret) apply only when generating the k8s target; the manifest rows
render into `deploy/k8s/`.

### EC2 target parameters

Same question round as the interactive `bentoml-ec2-deploy` skill's Step 0,
minus provisioning (the script only ever deploys to existing instances):

| Parameter | Placeholder / config key | Default / notes |
|---|---|---|
| Hosts | `{{EC2_HOST}}` → `targets.ec2.hosts` | JSON array of public IPs/DNS names of **existing** instances (1..N; bare hosts, no `user@`). Ask for all of them now — the script loops per host, fail-fast. |
| SSH user | `{{EC2_SSH_USER}}` → `targets.ec2.ssh_user` | `ec2-user` (Amazon Linux) or `ubuntu` (Ubuntu). |
| SSH key path | `{{EC2_SSH_KEY_PATH}}` → `targets.ec2.ssh_key_path` | Path to the private key (`~` expands; relative paths resolve against `deploy/`). Point at the key's usual home (e.g. `~/.ssh/...`) — never copy a private key into the repo. Preflight enforces mode 600. |
| Container name | `targets.ec2.container_name` (ships `{{SERVICE_NAME}}`) | Bento name with `_`→`-`. Names the container that `docker rm -f` + `run` replaces on every deploy. |
| Host port | `targets.ec2.host_port` (ships `3000`) | Published as `-p <host_port>:3000`. Preflight fails if anything other than our container holds it. |
| Runtime env var names | `targets.ec2.env_names` (ships `[]`) | NAMES only, e.g. `["HF_TOKEN"]` — values come from the deploying shell/CI environment at run time and are never written anywhere. |
| Registry auth | `{{EC2_REGISTRY_AUTH}}` → `targets.ec2.registry_auth` | `"ecr-token-over-ssh"` for ECR images when the deploying machine/CI holds AWS credentials (a fresh token is piped to each host per run — the only method that never touches instance IAM). `"preauthed"` when the instances pull on their own: instance profile, a docker login the user maintains, or a public image. |
| Verify path | `targets.ec2.verify_via` (ships `"tunnel"`) | `"tunnel"` (default; works with no inbound rule for the host port) or `"direct"` (only when the security group allows `host_port` from the deploying machine — ask, never assume). |
| Local tunnel port | `targets.ec2.local_tunnel_port` (ships `3230`) | Only for tunnel verify; change it if 3230 is taken locally. |

Arch matching needs no extra question: preflight compares each host's
`uname -m` against `image.platform` (or the local build arch) and fails
before anything mutates.

### SageMaker target parameters

Prerequisites first (same as the interactive `bentoml-sagemaker-deploy`
skill, which handles both interactively): the service must already carry
the SageMaker adaptation (its Step 1 — apply and locally validate it now if
missing), and an execution role must already exist (its Step 3 trust-policy
scan / admin snippet — this bundle **never creates IAM**). The registry
must be **ECR in the endpoint's region** (`registry_type: "ecr"`), and
standard `ml.*` instances are amd64 — set `image.platform` to
`"linux/amd64"` when building on arm64. Then:

| Parameter | Placeholder / config key | Default / notes |
|---|---|---|
| Region | `{{SAGEMAKER_REGION}}` → `targets.sagemaker.region` | Confirm with the user — never assume. Must equal the ECR registry region (preflight cross-checks). |
| Endpoint name | `{{SAGEMAKER_ENDPOINT_NAME}}` → `targets.sagemaker.endpoint_name` | e.g. `<service-name>-endpoint`. Max 63 chars, `^[a-zA-Z0-9](-*[a-zA-Z0-9])*$`. Keep it short: `-<version-tag>` is appended to derive the per-tag model/endpoint-config names, which share the 63-char limit. The endpoint is long-lived — created once, updated in place by every later run. |
| Execution role ARN | `{{SAGEMAKER_EXECUTION_ROLE_ARN}}` → `targets.sagemaker.execution_role_arn` | REQUIRED, pre-existing; trust policy must include `sagemaker.amazonaws.com`. Preflight verifies existence + trust via `iam get-role` (a denied `iam:GetRole` warns instead of failing). |
| Instance type | `targets.sagemaker.instance_type` (ships `"ml.m5.large"`) | ~$0.115/h ≈ $83/month **per instance, billed from InService until delete-endpoint** — state this cost when asking. Sizing table in the interactive skill's `references/aws-setup.md`. |
| Instance count | `targets.sagemaker.instance_count` (ships `1`) | Fixed count; autoscaling is out of scope. |
| Startup health timeout | `targets.sagemaker.startup_health_timeout_seconds` (ships `600`) | `ContainerStartupHealthCheckTimeoutInSeconds`, 60–3600. Raise it when model loading takes many minutes. |
| Extra container env vars | `targets.sagemaker.environment` (ships `{}`) | String map set on the model definition; visible via `describe-model`, so **no secrets** — bake models into the image instead. `BENTOML_PORT` is rejected here because the script always injects `BENTOML_PORT=8080`. |

The smoke-test rows from the main table double as the `invoke-endpoint`
verification; `verify.inference.path` should be `/invocations` (SageMaker
always posts there — any other value is ignored with a warning).

## Step 2 — Copy the bundle VERBATIM and render the config

```bash
mkdir -p <project>/deploy
cp -R <this-skill>/templates/deploy/. <project>/deploy/
```

Then render placeholders in **exactly two files**: `deploy/deploy.config.json`
and `deploy/README.md`. Substitute only there, file by file — **never run a
blanket sed across the bundle**: the `.py` files must stay byte-identical to
the templates, and some contain literal `{{SERVICE_NAME}}`-style text in
comments that a global substitution would corrupt. JSON gotchas:

- The template config ships **all three** target blocks (`targets.k8s`,
  `targets.ec2`, and `targets.sagemaker`). **Delete the block(s) for
  targets the user did not select** — a leftover `{{...}}` placeholder
  anywhere makes the config fail to load (exit 2), by design. Likewise
  prune the README sections (and CI/CD jobs) for targets that were not
  generated.
- `{{INFERENCE_BODY}}` is substituted with a JSON object (no quotes), e.g.
  `{"text": "A great day"}`.
- ec2 specifics: `hosts` is a JSON array of strings (render one
  `{{EC2_HOST}}` entry per instance); `env_names` stays `[]` unless the
  service needs runtime env vars (names only — remind the user the values
  come from the deploying environment); `registry_auth` and `verify_via`
  per the Step 1 table. `ssh_key_path` must point at the key where it
  already lives — never move or copy a private key into the project.
- sagemaker specifics: `environment` stays `{}` unless the service needs
  extra non-secret env vars (never add `BENTOML_PORT` — the script injects
  it); `instance_type`/`instance_count`/`startup_health_timeout_seconds`
  ship with usable defaults, change them only per the Step 1 answers. The
  role ARN and region must be the confirmed real values — the config
  loader rejects malformed ARNs and regions at exit 2.
- Optional keys ship as JSON `null` (`project.service`, `image.platform`,
  `targets.k8s.image_pull_secret`, `targets.k8s.node_port`) or `false`
  (`image.local_image_preloaded`) — replace the null/false with the real
  value only when the parameter applies (no quotes around `null`, numbers,
  or booleans). With `registry_type: "none"` set `registry` to `""` and
  `local_image_preloaded` to `true` once the user has loaded the image.
- Validate: `python3 -c "import json;json.load(open('deploy/deploy.config.json'))"`
  and `grep -nE '\{\{[A-Z][A-Z0-9_]*\}\}' deploy/deploy.config.json deploy/README.md`
  must find nothing (README placeholders must be substituted too; the
  pattern matches only generator placeholders, not the `${{ secrets.* }}`
  expressions that belong in the README's GitHub Actions examples).

## Step 3 — Render the k8s manifests into `deploy/k8s/` (k8s target only)

The ec2 target needs no manifests — skip this step entirely for an
ec2-only bundle (and do not create `deploy/k8s/`).

Use the **`bentoml-k8s-deploy` skill's templates and rules** (its
`templates/deployment.yaml`, `service.yaml`, optional `namespace.yaml` /
`ingress.yaml`, rendered per its SKILL.md Step 3: sed-substitute
placeholders, prune non-applicable OPTIONAL blocks, no `{{...}}` left), with
exactly one difference:

- **`{{IMAGE}}` is rendered as the literal sentinel `__DEPLOY_IMAGE__`**
  (not a real image ref). `deploy.py` rewrites the sentinel in memory on
  every run and pipes the manifests to `kubectl apply -f -` — the files on
  disk stay tag-free and committable.

Additional notes:
- Output dir is `deploy/k8s/` (matching `targets.k8s.manifests_dir`).
- Fixed NodePort chosen (see Step 1): add `nodePort: <value>` under the
  Service's port entry (alongside `port`/`targetPort`) with the same value
  as `targets.k8s.node_port`.
- For images only loaded into kind/minikube (registry_type `none`, or an
  image loaded by hand), add `imagePullPolicy: IfNotPresent` under the
  container `image:` line, per the k8s-deploy skill's private-registries
  reference.
- Sanity check: `grep -rn '{{' deploy/k8s/` empty, and
  `grep -rn '__DEPLOY_IMAGE__' deploy/k8s/` hits exactly the Deployment
  image line.

## Step 4 — Prove the bundle works, then hand it over

Run the preflight gate exactly as CI would, once per generated target:

```bash
python3 deploy/deploy.py --target k8s --check-only
python3 deploy/deploy.py --target ec2 --check-only        # connects to every host over SSH
python3 deploy/deploy.py --target sagemaker --check-only  # read-only AWS calls
```

For ec2, the full `--check-only` needs the SSH key, per-host reachability,
any `env_names` values exported, and (with `ecr-token-over-ssh`) AWS
credentials — it probes each host's docker daemon, architecture, and host
port in one SSH round trip per host, without changing anything.

For sagemaker, the full `--check-only` needs AWS credentials: it runs
`sts get-caller-identity`, checks the AWS CLI is v2, that the image ref is
ECR in `targets.sagemaker.region`, that the derived
`<endpoint_name>-<tag>` names fit SageMaker's rules, and verifies the
execution role's existence and trust policy via `iam get-role` (a denied
`iam:GetRole` degrades to a warning). All read-only. Without credentials
on this machine, fall back to `--check-only --local-only` and say so.

Show the user the check list and the summary JSON. It must exit 0 before
you hand off. If this machine lacks some credentials, degrade explicitly
and tell the user what was skipped (the summary's `skipped_checks` lists
it):

- no bentoml CLI / registry credentials, but cluster access works:
  `python3 deploy/deploy.py --target k8s --check-only --skip-build --image <known-ref>`
- no cluster/host/AWS or registry access at all (e.g. CI PR gate):
  `python3 deploy/deploy.py --target <k8s|ec2|sagemaker> --check-only --local-only`

If a check fails, fix the *environment or config* it names — never the
scripts.

Then tell the user:

1. **What was generated**: `deploy/deploy.py`, `deploy/deploy.config.json`
   (the only file they edit), `deploy/_internal/` (never edit),
   `deploy/k8s/*.yaml` (k8s target only), `deploy/README.md`.
2. **How to deploy**: `python3 deploy/deploy.py --target <k8s|ec2|sagemaker>`
   (full build+push+deploy+verify), and the `--skip-build --image REF` form
   for redeploys/rollbacks. For ec2, secrets named in `env_names` must be
   exported in the deploying shell/CI environment first. For sagemaker,
   state the standing cost plainly: the endpoint bills per instance-hour
   from InService until `delete-endpoint` (default ml.m5.large ≈ $0.115/h ≈
   $83/month), and the README's teardown section stops it.
3. **Commit it**: the bundle contains no secrets —
   `git add deploy/ && git commit -m "Add production deploy bundle"`.
4. **Wire CI later**: point them at the CI/CD chapter in the generated
   `deploy/README.md` (complete GitHub Actions workflow with AWS OIDC and
   per-target deploy jobs, a GitLab CI equivalent, and
   `--check-only --local-only` as the fork-safe PR gate).
