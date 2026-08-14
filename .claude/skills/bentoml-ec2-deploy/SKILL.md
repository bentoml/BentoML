---
name: bentoml-ec2-deploy
description: >
  Deploy a containerized BentoML service directly onto one or more plain AWS EC2
  instances with Docker — no Kubernetes. Takes a pushed container image (built by
  the bentoml-containerize skill), either uses the user's existing instances over
  SSH or provisions a new instance via the AWS CLI (SSM AMI lookup, security
  group, key pair), runs the container with restart-on-reboot, and verifies with
  a real inference request. Use when the user says things like "deploy my BentoML
  service to EC2", "run my bento on an AWS VM", "deploy this bento image to an
  EC2 instance", "run my BentoML container on AWS without Kubernetes", or "put my
  bento on a cloud VM". For Kubernetes targets use bentoml-k8s-deploy instead.
---

# Deploy a BentoML service to plain EC2 instances

> For production / CI-CD deployments, generate a standalone script bundle (no agent needed at deploy time) with the `bentoml-deploy-scriptgen` skill.

You will take a **pushed container image reference** (e.g.
`123456789012.dkr.ecr.us-east-1.amazonaws.com/summarization:v1`, produced by the
`bentoml-containerize` skill), get it running under Docker on one or more EC2
instances, and verify it end to end. Docker is the only runtime dependency on the
instance — no Kubernetes, no agents.

Facts that are always true for images built by `bentoml containerize`:
- The HTTP server listens on **port 3000**.
- Health endpoints: **`/livez`** (liveness), **`/readyz`** (readiness). `/metrics` serves Prometheus metrics; `/` is Swagger UI.
- The image entrypoint already starts the server — never pass a command to `docker run`.
- Models are usually baked into the image (expect multi-GB images → size the root EBS volume accordingly); models downloaded at runtime need env vars such as `HF_TOKEN` passed with `-e`.

Safety rules (apply throughout — no exceptions):
- **Before EVERY mutating AWS CLI command** (`run-instances`, `create-security-group`,
  `authorize-security-group-ingress`, `create-key-pair`, `iam create-*`/`put-*`/`add-*`,
  `associate-iam-instance-profile`, `terminate-instances`, `delete-*`): show the user the
  **exact command** plus a **cost note** (hourly/monthly estimate, or "free"), and wait for
  explicit confirmation. Read-only commands (`describe-*`, `get-*`, `sts get-caller-identity`,
  `ssm get-parameters`, `wait`) may run freely.
- **Every mutating SSH command echoes the target host first**: `echo ">>> on $HOST:" ...`.
- Never terminate, stop, reboot, or modify instances, security groups, key pairs, or IAM
  resources this skill did not create in this session. This includes attaching instance
  profiles to existing instances — that modifies the instance (see the registry-auth reference).
- Track every resource ID you create (instance IDs, SG ID, key pair name, IAM names) — the
  teardown section operates on exactly that list and nothing else.
- Secrets are passed at `docker run` time with `-e NAME="$NAME"` expanded from the local
  shell env — never written into files, user-data, or the manifest of any command you save.
- Use `sudo docker ...` in all SSH commands: `usermod -aG docker` does not take effect for
  already-open sessions and races with fresh instances; `sudo` always works.

## Step 0 — Preflight checks

Always required (stop with a clear message if it fails):

```bash
ssh -V                              # ssh client present
```

Gather from the `bentoml-containerize` handoff (or ask):
1. **Image reference** — full pushed ref including registry and tag.
2. **Registry access** — ECR / other private registry / public. Determines Step 2.
3. **Runtime env var names** the service needs (values come from the user's shell env later).
4. **Image architecture** (`amd64` or `arm64`) — must match the instance
   (`t3.*`/`m5.*`/`c5.*` = amd64, `t4g.*`/`m7g.*` = arm64), otherwise the container
   dies with `exec format error`.
5. **Container/service name** — the bento name with underscores as hyphens
   (e.g. `my_service` → `my-service`).

Ask in the same question round:

6. **Mode**:
   - **A — existing instance(s)**: the user provides SSH host(s), user, and key.
   - **B — provision new**: this skill creates the instance(s) via the AWS CLI.
7. **ECR auth path** (only if the image is in ECR — decide it NOW, not in Step 2):
   - **Instance profile** (preferred for Mode B — but it needs IAM write permissions,
     which PowerUserAccess-style SSO roles typically lack despite full EC2/ECR access):
     the IAM role/profile must be created **before launch** so it can be attached at
     `run-instances` time — see [references/registry-auth.md](references/registry-auth.md).
     If the user's role may lack IAM writes, probe first:
     `aws iam get-role --role-name nonexistent-probe 2>&1` — `NoSuchEntity` means IAM
     read works (writes are plausible); `AccessDenied` means not even IAM read, so pick
     token-over-SSH now. Either way, if the first IAM create in Way 1 returns
     `AccessDenied`, fall back to token-over-SSH — no cleanup needed, nothing was created.
   - **Token over SSH**: nothing to create or install; the only option that doesn't
     modify a Mode A instance.

**AWS CLI checks** — required for Mode B and for any ECR image; **skip both for Mode A
with a non-ECR image** (that path never runs an aws command — don't block it):

```bash
aws --version                       # AWS CLI v2 installed?
aws sts get-caller-identity         # credentials valid? note the Account
```

Then **confirm the region WITH the user — never assume**:

```bash
aws configure get region            # show as the default suggestion, then ask
```

Use `--region "$AWS_REGION"` explicitly on every subsequent aws command. For ECR
images the region embedded in the image ref
(`<account>.dkr.ecr.<region>.amazonaws.com/...`) must match the login commands — check it now.

## Step 1A — Existing instance(s) (SSH mode)

Collect: `HOST` (public IP/DNS), `SSH_USER` (`ec2-user` for Amazon Linux, `ubuntu`
for Ubuntu), `KEY` (path to the private key). Set
`SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$SSH_USER@$HOST")`
and use `"${SSH[@]}"` everywhere below.

```bash
"${SSH[@]}" 'echo connected; uname -m;
  if ! command -v docker >/dev/null 2>&1; then echo NO_DOCKER;
  elif ! sudo docker info >/dev/null 2>&1;  then echo DAEMON_DOWN;
  else sudo docker info --format {{.ServerVersion}}; fi'
```

- `uname -m`: `x86_64` = amd64, `aarch64` = arm64 — must match the image arch (Step 0.4).
- If `NO_DOCKER`: offer to install it (this mutates the user's instance — echo the host and
  get confirmation first). Amazon Linux 2023:
  `sudo dnf install -y docker && sudo systemctl enable --now docker`; Ubuntu:
  `sudo apt-get update && sudo apt-get install -y docker.io && sudo systemctl enable --now docker`.
- If `DAEMON_DOWN`: docker is installed but the daemon isn't running — offer
  `sudo systemctl enable --now docker` (also a mutation: echo host + confirm). `enable`
  matters, not just `start`: `--restart unless-stopped` only survives reboots if the
  daemon itself comes back.
- Also confirm the instance's **security group** allows the access path you'll verify with
  (port 3000 from the user's IP, or plan on an SSH tunnel — Step 4). Never edit an existing
  instance's security group; tell the user what rule is missing and let them add it, or use
  the tunnel.

Multiple hosts: collect them all now; Steps 2–4 loop per host — read
[references/multi-node.md](references/multi-node.md).

## Step 1B — Provision new instance(s) (AWS CLI mode)

Read [references/provisioning.md](references/provisioning.md) and follow it fully. The
sequence (every mutating call shown + cost note + confirmation first):

1. **AMI via SSM public parameters — never hardcode or guess AMI IDs.** Default
   Amazon Linux 2023 (`ec2-user`), alternative Ubuntu 24.04 (`ubuntu`):

```bash
# Amazon Linux 2023 (pick the suffix matching the image arch: x86_64 | arm64)
AMI=$(aws ssm get-parameters --region "$AWS_REGION" \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameters[0].Value' --output text)

# Ubuntu 24.04 LTS (amd64 | arm64)
AMI=$(aws ssm get-parameters --region "$AWS_REGION" \
  --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query 'Parameters[0].Value' --output text)
echo "$AMI"     # must look like ami-...; show it to the user
```

2. **Key pair**: reuse the user's existing one, or create a new one (free).
3. **Security group** (free): new SG allowing **22 from the user's IP** (detect with
   `curl -s https://checkip.amazonaws.com`, confirm with the user — VPNs skew it) and
   **port 3000 with a scope the user chooses**: their IP (default), an SSH tunnel only
   (no 3000 rule), or `0.0.0.0/0` — the last **only after an explicit warning** that it
   exposes an unauthenticated inference API to the whole internet, and explicit confirmation.
4. **ECR instance profile** (only if Step 0 chose it): create the role + profile per
   [references/registry-auth.md](references/registry-auth.md) **now** — it must exist to be
   attached at launch; there is no later attach step.
5. **Instance**: default `t3.medium` (~$0.04/hr on-demand in us-east-1; region-dependent —
   always state the estimate; the public IPv4 address adds $0.005/hr). `t4g.medium` for
   arm64 images. Size memory to the model; size the **root volume to the image** (default
   8 GiB is too small for most bentos — 30–50 GiB gp3, ~$0.08/GB-month). User-data installs
   and enables Docker. Tag everything `managed-by=bentoml-ec2-deploy`.
6. **Wait and connect**: `aws ec2 wait instance-running`, fetch the public IP (`HOST`),
   then set the same SSH array used everywhere below —
   `SSH_USER=ec2-user` (AL2023; `ubuntu` for Ubuntu AMIs) and
   `SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$SSH_USER@$HOST")`
   — and retry SSH until cloud-init has finished installing Docker.

Record `INSTANCE_ID`, `SG_ID`, key pair name, and (if created) IAM names — they are the
complete teardown list.

## Step 2 — Registry auth on the instance

Read [references/registry-auth.md](references/registry-auth.md) and execute the path
already chosen in Step 0 (for ECR + instance profile, the IAM setup happened before
launch in Step 1B — only the on-instance login remains here):

- **ECR + instance profile** (preferred for instances this skill provisions): minimal
  pull-only IAM policy, attached at launch; login on the instance uses its own role.
- **ECR + token over SSH** (works for any instance, nothing to install): pipe
  `aws ecr get-login-password` from the local machine into `docker login` on the host.
  Tokens expire after **12 hours** — fine for deploying, remember it for redeploys.
- **Non-ECR private registry** (Docker Hub, GHCR, private): pipe the token from the local
  env into `docker login` over SSH.
- **Public image**: skip this step.

## Step 3 — Pull and run

For each host (echo it first — this mutates the instance):

```bash
echo ">>> on $HOST: pulling $IMAGE"
"${SSH[@]}" "sudo docker pull '$IMAGE'"
```

If a container named `$SERVICE_NAME` already exists on the host, do **not** remove it
unless this skill created it earlier in this session or the user explicitly confirms —
show them `sudo docker ps -a --filter name=$SERVICE_NAME` output and ask.

```bash
echo ">>> on $HOST: starting container $SERVICE_NAME"
"${SSH[@]}" "sudo docker run -d --name '$SERVICE_NAME' \
  --restart unless-stopped \
  -p 3000:3000 \
  '$IMAGE'"
```

- Runtime env vars: insert `-e HF_TOKEN='$HF_TOKEN'` style flags **before** the image ref.
  Use double quotes around the remote command so values expand from the **local** shell env;
  never write them into a file or user-data.
- `--restart unless-stopped` survives container crashes and instance reboots (provided the
  Docker service is enabled — the provisioning user-data does this).
- GPU instances: add `--gpus all` (requires the NVIDIA driver + nvidia-container-toolkit on
  the host — a GPU AMI concern, out of this skill's scope to install).

Confirm it stayed up before verifying (a crash-looping container shows `Restarting`).
The `sleep 3` is load-bearing: checked immediately after `docker run -d`, even a container
that crashes 2 seconds later still shows `Up less than a second` — give it time to die:

```bash
sleep 3
"${SSH[@]}" "sudo docker ps --filter name='$SERVICE_NAME' --format '{{.Status}}'"   # must show "Up ..."
```

## Step 4 — Verify: /readyz + one real inference request

Choose the path that matches the port-3000 scope from Step 1:

**Direct** (port 3000 open to your IP): probe `http://$HOST:3000` from the local machine.

**SSH tunnel** (port 3000 not opened, or direct probing fails): forward an **uncommon
local port (3200)** — local 3000 is commonly occupied by dev servers, and if the tunnel
fails to bind, curls silently hit whatever squats on the port, producing convincing but
fake results. The whole block must run in **one shell invocation** (the background PID
does not survive across separate shells):

```bash
TUN_ERR=$(mktemp)
# ExitOnForwardFailure=yes is load-bearing: without it, ssh prints "Could not
# request local forwarding." and KEEPS RUNNING when local 3200 is taken — the
# kill -0 guard below would pass while your curls hit the squatter.
ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
  -o ExitOnForwardFailure=yes \
  -N -L 3200:127.0.0.1:3000 "$SSH_USER@$HOST" >"$TUN_ERR" 2>&1 &
TUNNEL_PID=$!
BASE=http://127.0.0.1:3200
OK=""
for i in $(seq 1 60); do            # model loading can take minutes
  kill -0 $TUNNEL_PID 2>/dev/null || { echo "tunnel exited:"; cat "$TUN_ERR"; rm -f "$TUN_ERR"; exit 1; }
  curl -sfo /dev/null "$BASE/readyz" && { OK=1; break; }
  sleep 5
done
[ -n "$OK" ] || { echo "not ready after 5 min; tunnel log:"; cat "$TUN_ERR"; rm -f "$TUN_ERR"; kill $TUNNEL_PID; exit 1; }
echo READY

# Real inference request — derive route/payload from service.py (@bentoml.api methods
# are POST /<method_name> with a JSON body of the parameters); if the source is not
# available, fetch $BASE/docs.json for the schema, then re-run with the call filled in:
curl -s -X POST "$BASE/summarize" \
  -H 'Content-Type: application/json' \
  -d '{"text": "EC2 is a web service that provides resizable compute capacity."}'

kill $TUNNEL_PID
rm -f "$TUN_ERR"
```

For the direct path, run the same readiness loop and inference call with
`BASE=http://$HOST:3000` (no tunnel, no PID checks) — but first re-check
`sudo docker ps` over SSH so you know the responder is your container.

**Judge the inference response by its content, not the status code.** Only a plausible,
correct result for the user's service proves the deployment works. HTTP 200 alone proves
nothing; an error body means the payload or the service needs fixing. If the output looks
unrelated to the service's API, suspect a local port squatter (tunnel path) or a stale
process on the host.

Multiple hosts: verify **every** host individually — a load-balanced fleet with one dead
node still answers most probes.

## Step 5 — Tell the user how to reach the service

- Base URL: `http://<public-ip-or-dns>:3000` (only from IPs the security group allows) —
  Swagger UI at `/`, Prometheus metrics at `/metrics`, health at `/livez` and `/readyz`.
- Tunnel-only setups: `ssh -i <key> -N -L 3000:127.0.0.1:3000 <user>@<host>` →
  `http://127.0.0.1:3000`.
- The public IP **changes when the instance is stopped and started** — for a stable address
  point them at an Elastic IP (`aws ec2 allocate-address`) or DNS. Note: since Feb 2024 AWS
  bills **every** public IPv4 address at $0.005/hr (~$3.65/mo), attached or not — an EIP
  costs the same as the auto-assigned IP the instance already has, so stability is free;
  an *unattached* EIP keeps billing with nothing to show for it.
- Plain HTTP on a raw port is fine for testing; for production suggest TLS in front
  (nginx/caddy on the host, or an ALB — see below). Remind them the service has **no
  authentication** unless they built it in.

## Multi-node and load balancing

Deploying to N hosts = loop Steps 2–4 over the host list; details and a ready-made loop in
[references/multi-node.md](references/multi-node.md). Load balancing across nodes is **out
of scope** for this skill — the standard AWS answer is an Application Load Balancer
(`aws elbv2 create-load-balancer` + a target group on port 3000 with health check path
`/readyz`); point the user at the reference's short ALB note and stop there.

## Teardown

Offer this at the end of every session; run it only on **resources this skill created in
this session**, each command shown + confirmed first.

Container only (any mode — echo the host):

```bash
"${SSH[@]}" "sudo docker rm -f '$SERVICE_NAME'"
```

Provisioned infrastructure (Mode B) — order matters; full commands and IAM cleanup in
[references/provisioning.md](references/provisioning.md):

```bash
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"   # billing stops at termination
aws ec2 wait instance-terminated --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
aws ec2 delete-security-group --region "$AWS_REGION" --group-id "$SG_ID"           # fails while the instance is alive — hence the wait
aws ec2 delete-key-pair --region "$AWS_REGION" --key-name "$KEY_NAME"              # only if this session created it; also rm the local .pem
```

Registry side: if the deployment is being **fully retired** AND the ECR repository was
created just for it (e.g. by the bentoml-containerize skill in this workflow — never a
pre-existing repo), offer to delete it too; ECR bills for image storage (~$0.10/GB-month)
until the repo is gone:

```bash
aws ecr delete-repository --region "$ECR_REGION" --repository-name "$REPO_NAME" --force
# --force deletes the images inside — confirm the user wants them gone, not just the instance
```

Remind users who keep the instance: it **bills every hour until terminated** (a stopped
instance still bills its EBS volume).

## Troubleshooting (self-contained — do not hand off to bentoml-k8s-troubleshoot)

Work symptom → command → fix. All docker commands run over SSH with `sudo`.

| Symptom | Diagnose | Likely cause / fix |
|---|---|---|
| SSH timeout / connection refused | `aws ec2 describe-instances --instance-ids ... --query '...State.Name'`; SG rules | Instance still booting (wait), SG missing 22-from-your-IP, wrong public IP, or your IP changed (VPN). |
| SSH `Permission denied (publickey)` | — | Wrong `SSH_USER` (`ec2-user` vs `ubuntu`), wrong key, or key perms (`chmod 600`). |
| `docker: command not found` right after boot | `"${SSH[@]}" 'cloud-init status'` | user-data still running — wait until `status: done`, then recheck. |
| Pull fails: `no basic auth credentials` / `denied` / `authorization token has expired` | re-run the login from Step 2 | ECR token older than 12 h, login done for the wrong region/account, or the instance profile lacks the pull policy. |
| Pull fails: `no space left on device` | `"${SSH[@]}" 'df -h /'` | Root volume too small for the image — reprovision with a bigger `--block-device-mappings`, or `sudo docker system prune -af`. |
| Container status `Restarting (…)` or `Exited` | `"${SSH[@]}" "sudo docker logs --tail 100 '$SERVICE_NAME'"` | Read the traceback: missing env var (add `-e`), missing model/dependency (rebuild via bentoml-containerize), port clash inside logs. |
| `exec format error` in logs | `uname -m` on host vs image arch | Arch mismatch — rebuild with `--opt platform=linux/amd64` (or use a `t4g` instance for arm64 images). |
| `port is already allocated` | `"${SSH[@]}" 'sudo docker ps; sudo ss -ltnp \| grep :3000'` | Another container/process owns 3000 — pick a different host port (`-p 8080:3000`) or stop the squatter (only if the user says so). |
| Container `Up` but `/readyz` never 200 | `sudo docker logs -f` | Model still loading (large models take minutes — keep waiting), or startup error after the server bound the port. |
| `/readyz` ok locally on host (`curl 127.0.0.1:3000/readyz` over SSH) but not from your machine | compare with SG rules | Network path: SG has no 3000 rule for your IP, or corporate egress blocks it — use the SSH tunnel. |
| Killed / OOM | `"${SSH[@]}" 'sudo dmesg \| grep -i oom \| tail'` and `sudo docker inspect --format '{{.State.OOMKilled}}' $SERVICE_NAME` | Instance memory too small for the model — bigger instance type. |
| Inference 4xx/5xx with readyz green | response body; `$BASE/docs.json` | Wrong route or payload shape — endpoints are `POST /<method_name>` with JSON params. |

## Out of scope

Autoscaling groups, ALB creation, HTTPS termination, spot instances, and multi-region were
never part of basic deployment. If the user asks, give the one-line ALB pointer above and
stop there. For fleet-scale EC2 operations (launch templates, Auto Scaling groups, Spot,
SSM Session Manager), AWS's official plugin covers exactly that ground — the user can
install it with `/plugin install aws-core@claude-plugins-official` (its `aws-compute`
skill picks up where this one ends).

## References (read on demand)

- `references/provisioning.md` — full Mode B flow: IP detection, key pair, security group, SSM AMI lookup, run-instances with user-data, root volume sizing, waits, cost table, exact teardown incl. IAM.
- `references/registry-auth.md` — ECR via instance profile (minimal policy) and via token-over-SSH; 12-hour expiry; Docker Hub/GHCR/private/public registries.
- `references/multi-node.md` — deploying the same container to N hosts; short ALB pointer.
