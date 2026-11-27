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
license: Apache-2.0
compatibility: >-
  Requires an ssh client and AWS CLI v2 with valid credentials. Docker runs on the EC2 instance, not on your machine.
---

# Deploy a BentoML service to plain EC2 instances

> For production / CI-CD, generate a standalone script bundle (no agent at deploy time)
> with the `bentoml-deploy-scriptgen` skill.

In: a pushed image ref from `bentoml-containerize` (e.g.
`123456789012.dkr.ecr.us-east-1.amazonaws.com/summarization:v1`). Out: it running under
Docker on one or more EC2 instances, verified. Docker is the instance's only dependency
— no Kubernetes, no agents.

Image facts (`bentoml containerize`):
- Serves HTTP on **port 3000**. **`/livez`** liveness, **`/readyz`** readiness,
  `/metrics` Prometheus, `/` Swagger UI.
- The entrypoint starts the server — never pass a command to `docker run`.
- Models are usually baked in, so images run multi-GB; size volumes to match. Runtime
  model downloads need env vars such as `HF_TOKEN` via `-e`.

Safety rules, no exceptions:
- **Before EVERY mutating AWS CLI command** (`run-instances`, `create-security-group`,
  `authorize-security-group-ingress`, `create-key-pair`, `iam create-*`/`put-*`/`add-*`,
  `associate-iam-instance-profile`, `terminate-instances`, `delete-*`): show the exact
  command + a cost note (hourly/monthly, or free), then wait for explicit confirmation.
  Read-only (`describe-*`, `get-*`, `sts get-caller-identity`, `ssm get-parameters`,
  `wait`) runs freely.
- Mutating SSH commands echo their host first: `echo ">>> on $HOST:" ...`.
- Never terminate, stop, reboot or modify instances, security groups, key pairs or IAM
  resources this skill did not create this session. Attaching an instance profile to an
  existing instance counts as modifying it.
- Track every ID you create (instances, SG, key pair, IAM). Teardown touches only that
  list.
- Secrets reach the container as `-e NAME="$NAME"` from the local shell env — never a
  file, user-data or a saved command.
- Always `sudo docker ...` over SSH: `usermod -aG docker` misses open sessions and races
  fresh instances.

## Step 0 — Preflight

`ssh -V` must succeed. From the `bentoml-containerize` handoff, or ask:
1. **Image ref** — full, with registry and tag.
2. **Registry access** — ECR / other private / public. Picks Step 2.
3. **Runtime env var names**; values come from the user's shell env.
4. **Image arch** (`amd64`/`arm64`) — must match the instance (`t3.*`/`m5.*`/`c5.*`
   amd64, `t4g.*`/`m7g.*` arm64), else `exec format error`.
5. **Container/service name** — bento name, underscores → hyphens (`my_service` →
   `my-service`).
6. **Mode** — **A**: user's existing instance(s) over SSH. **B**: provision new via AWS
   CLI.
7. **ECR auth path** (ECR only, decide now, see
   [references/registry-auth.md](references/registry-auth.md)):
   - **Instance profile** (preferred for Mode B) must exist **before launch**, to attach
     at `run-instances`. Needs IAM writes, which PowerUserAccess-style SSO roles usually
     lack despite full EC2/ECR. Probe: `aws iam get-role --role-name nonexistent-probe
     2>&1` — `NoSuchEntity` = IAM read works; `AccessDenied` = pick token-over-SSH now.
     On `AccessDenied` from the first IAM create (Way 1), fall back; nothing was
     created.
   - **Token over SSH**: nothing to create or install, and the only path that leaves a
     Mode A instance unmodified.

**AWS CLI checks** — Mode B and any ECR image. **Skip for Mode A + non-ECR image**: that
path runs no aws command, so don't block on them.

```bash
aws --version                       # AWS CLI v2 installed?
aws sts get-caller-identity         # credentials valid? note the Account
aws configure get region            # suggest as default, then ASK — never assume the region
```

`--region "$AWS_REGION"` goes on every later aws command. For ECR, the region in the ref
(`<account>.dkr.ecr.<region>.amazonaws.com/...`) must match the login commands.

## Step 1A — Existing instance(s) (SSH mode)

Collect `HOST` (public IP/DNS), `SSH_USER` (`ec2-user` Amazon Linux, `ubuntu` Ubuntu),
`KEY` (private key path), then set the array used everywhere below:

```bash
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$SSH_USER@$HOST")
"${SSH[@]}" 'echo connected; uname -m;
  if ! command -v docker >/dev/null 2>&1; then echo NO_DOCKER;
  elif ! sudo docker info >/dev/null 2>&1;  then echo DAEMON_DOWN;
  else sudo docker info --format {{.ServerVersion}}; fi'
```

- `uname -m`: `x86_64` = amd64, `aarch64` = arm64 — must match the image arch (Step
  0.4).
- `NO_DOCKER`: offer to install (a mutation — echo host, confirm). AL2023: `sudo dnf
  install -y docker && sudo systemctl enable --now docker`. Ubuntu: `sudo apt-get update
  && sudo apt-get install -y docker.io && sudo systemctl enable --now docker`.
- `DAEMON_DOWN`: offer `sudo systemctl enable --now docker` (also a mutation). `enable`,
  not just `start`: `--restart unless-stopped` needs the daemon back after a reboot.
- Check the **security group** allows your verification path: 3000 from the user's IP,
  or an SSH tunnel (Step 4). Never edit an existing instance's SG — name the missing
  rule, let the user add it, or tunnel.

Multiple hosts: collect all now; Steps 2–4 loop per host, with a ready-made loop in
[references/multi-node.md](references/multi-node.md).

## Step 1B — Provision new instance(s) (AWS CLI mode)

Follow [references/provisioning.md](references/provisioning.md) fully; every mutating
call gets command + cost note + confirmation.

1. **AMI from SSM public parameters (reference §4) — never hardcode or guess AMI IDs.**
   AL2023 → `ec2-user`, Ubuntu 24.04 → `ubuntu`. Show the resolved `ami-...`.
2. **Key pair**: reuse the user's, or create one (free).
3. **Security group** (free): **22 from the user's IP** (detect and confirm it — VPNs
   skew it) and **port 3000 at a scope the user picks**: their IP (default), tunnel-only
   (no 3000 rule), or `0.0.0.0/0` — that last **only after warning** that it exposes an
   unauthenticated inference API to the whole internet, plus explicit confirmation.
4. **ECR instance profile** (if Step 0 chose it): create role + profile per
   [references/registry-auth.md](references/registry-auth.md) **now**; it must exist to
   be attached at launch, and there is no later attach step.
5. **Instance**: default `t3.medium` (~$0.04/hr on-demand us-east-1, region-dependent —
   always state the estimate; the public IPv4 address adds $0.005/hr), `t4g.medium` for
   arm64. Size RAM to the model, and the **root volume to the image**: the 8 GiB AMI
   default is too small for most bentos — 30–50 GiB gp3, ~$0.08/GB-month. User-data
   installs and enables Docker. Tag everything `managed-by=bentoml-ec2-deploy`.
6. **Wait and connect**: `aws ec2 wait instance-running`, public IP as `HOST`,
   `SSH_USER=ec2-user` (AL2023; `ubuntu` for Ubuntu AMIs), the Step 1A `SSH=(...)`
   array, then retry SSH until cloud-init finishes installing Docker.

Record `INSTANCE_ID`, `SG_ID`, key pair name, IAM names — the complete teardown list.

## Step 2 — Registry auth on the instance

Execute the Step 0 path per [references/registry-auth.md](references/registry-auth.md):

- **ECR + instance profile**: the instance logs in with its own role (IAM done in Step
  1B).
- **ECR + token over SSH**: pipe `aws ecr get-login-password` into `docker login` on the
  host. Tokens expire after **12 hours** — matters for redeploys.
- **Other private registry** (Docker Hub, GHCR): pipe the token from the local env into
  `docker login` over SSH.
- **Public image**: skip.

## Step 3 — Pull and run

Per host, echoed first because this mutates the instance:

```bash
echo ">>> on $HOST: pulling $IMAGE"
"${SSH[@]}" "sudo docker pull '$IMAGE'"

echo ">>> on $HOST: starting container $SERVICE_NAME"
"${SSH[@]}" "sudo docker run -d --name '$SERVICE_NAME' --restart unless-stopped -p 3000:3000 '$IMAGE'"
```

- A container already named `$SERVICE_NAME`: do **not** remove it unless this skill
  created it this session or the user confirms — show `sudo docker ps -a --filter
  name=$SERVICE_NAME`, ask.
- Env vars: `-e HF_TOKEN='$HF_TOKEN'` style flags **before** the image ref. Double
  quotes around the remote command expand values from the **local** env; never a file or
  user-data.
- `--restart unless-stopped` survives crashes and reboots if the Docker service is
  enabled (user-data does that).
- `-p <host_port>:3000` if 3000 is taken on the host; the container side is always 3000.
- GPU: add `--gpus all` (needs NVIDIA driver + nvidia-container-toolkit on the host, not
  installed here).

Confirm it stayed up; crash-loops show `Restarting`. `sleep 3` is load-bearing: a
container that dies 2 seconds in still reports `Up less than a second` right after
`docker run -d`.

```bash
sleep 3
"${SSH[@]}" "sudo docker ps --filter name='$SERVICE_NAME' --format '{{.Status}}'"   # must show "Up ..."
```

## Step 4 — Verify: /readyz + one real inference request

Match the port-3000 scope from Step 1. **Direct** (3000 open to your IP): probe
`http://$HOST:3000`. **SSH tunnel** (no 3000 rule, or direct probing fails): forward an
**uncommon local port (3200)** — local 3000 often hosts a dev server, and a tunnel that
fails to bind sends curls to the squatter, faking success. Keep it to one shell
invocation: the background PID dies with the shell.

```bash
TUN_ERR=$(mktemp)
# ExitOnForwardFailure=yes is load-bearing: without it ssh prints "Could not request
# local forwarding." and KEEPS RUNNING when local 3200 is taken — the kill -0 guard
# would pass while curls hit the squatter.
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

# Real inference request — route/payload from service.py (@bentoml.api methods are
# POST /<method_name> with a JSON body of the parameters). No source? Fetch
# $BASE/docs.json for the schema, then re-run with the call filled in:
curl -s -X POST "$BASE/summarize" \
  -H 'Content-Type: application/json' \
  -d '{"text": "EC2 is a web service that provides resizable compute capacity."}'

kill $TUNNEL_PID
rm -f "$TUN_ERR"
```

Direct path: same loop and call with `BASE=http://$HOST:3000`, no tunnel, no PID checks
— but re-check `sudo docker ps` over SSH first, so you know your container is the
responder.

**Judge the response body, not the status code.** Only a plausible, correct result
proves the deployment. HTTP 200 proves nothing; an error body means the payload or the
service needs fixing; output unrelated to the API means a port squatter (tunnel) or a
stale host process. Verify **every** host individually — a fleet with one dead node
still answers most probes.

## Step 5 — Tell the user how to reach the service

- `http://<public-ip-or-dns>:3000`, only from IPs the SG allows. `/` Swagger UI,
  `/metrics`, `/livez`, `/readyz`.
- Tunnel-only: `ssh -i <key> -N -L 3000:127.0.0.1:3000 <user>@<host>` →
  `http://127.0.0.1:3000`.
- The public IP **changes on stop/start**; for a stable address use an Elastic IP (`aws
  ec2 allocate-address`) or DNS. Since Feb 2024 AWS bills **every** public IPv4 address
  $0.005/hr (~$3.65/mo), attached or not — an EIP costs the same as the auto-assigned IP
  (stability is free), but an *unattached* EIP bills for nothing.
- Raw HTTP is fine for testing; for production suggest TLS in front (nginx/caddy, or an
  ALB). Say plainly: the service has **no authentication** unless they built it in.

## Teardown

Offer at the end of every session, for session-created resources only, each command
shown + confirmed. Container only, any mode (echo the host): `"${SSH[@]}" "sudo docker
rm -f '$SERVICE_NAME'"`. Mode B infrastructure — order matters, IAM cleanup in
[references/provisioning.md](references/provisioning.md):

```bash
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"   # billing stops at termination
aws ec2 wait instance-terminated --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
aws ec2 delete-security-group --region "$AWS_REGION" --group-id "$SG_ID"           # fails while the instance lives — hence the wait
aws ec2 delete-key-pair --region "$AWS_REGION" --key-name "$KEY_NAME"              # only if this session created it; also rm the local .pem
```

Registry: if the deployment is **fully retired** AND the ECR repo was created just for
it (by bentoml-containerize here, never a pre-existing repo), offer to delete it. ECR
bills storage (~$0.10/GB-month) until the repo is gone:

```bash
aws ecr delete-repository --region "$ECR_REGION" --repository-name "$REPO_NAME" --force
# --force deletes the images inside — confirm the user wants them gone, not just the instance
```

If they keep the instance: it **bills every hour until terminated**, and a stopped
instance still bills its EBS volume and its public IPv4 address.

## Troubleshooting

Handle EC2 symptoms here; the Kubernetes runbook
(`bentoml-k8s-deploy/references/troubleshooting.md`) does not apply. Docker commands run
over SSH with `sudo`.

| Symptom | Diagnose | Likely cause / fix |
|---|---|---|
| SSH timeout / refused | `aws ec2 describe-instances --instance-ids ... --query '...State.Name'`; SG rules | Still booting (wait), SG missing 22-from-your-IP, wrong public IP, or your IP changed (VPN). |
| SSH `Permission denied (publickey)` | — | Wrong `SSH_USER` (`ec2-user` vs `ubuntu`), wrong key, or key perms (`chmod 600`). |
| `docker: command not found` after boot | `"${SSH[@]}" 'cloud-init status'` | user-data still running — wait for `status: done`, recheck. |
| Pull: `no basic auth credentials` / `denied` / `authorization token has expired` | re-run the Step 2 login | ECR token older than 12 h, login for the wrong region/account, or the instance profile lacks the pull policy. |
| Pull: `no space left on device` | `"${SSH[@]}" 'df -h /'` | Root volume too small — reprovision with a bigger `--block-device-mappings`, or `sudo docker system prune -af`. |
| Container `Restarting (…)` or `Exited` | `"${SSH[@]}" "sudo docker logs --tail 100 '$SERVICE_NAME'"` | Traceback says which: missing env var (add `-e`), missing model/dependency (rebuild via bentoml-containerize), port clash. |
| `exec format error` in logs | `uname -m` on host vs image arch | Arch mismatch — rebuild with `--opt platform=linux/amd64`, or use `t4g` for arm64. |
| `port is already allocated` | `"${SSH[@]}" 'sudo docker ps; sudo ss -ltnp \| grep :3000'` | Something else owns 3000 — use another host port (`-p 8080:3000`) or stop the squatter (only if the user says so). |
| Container `Up`, `/readyz` never 200 | `sudo docker logs -f` | Model still loading (large ones take minutes — keep waiting), or a startup error after the port was bound. |
| `/readyz` ok on the host (`curl 127.0.0.1:3000/readyz` over SSH), not from your machine | SG rules | No 3000 rule for your IP, or corporate egress blocks it — use the SSH tunnel. |
| Killed / OOM | `"${SSH[@]}" 'sudo dmesg \| grep -i oom \| tail'` and `sudo docker inspect --format '{{.State.OOMKilled}}' $SERVICE_NAME` | Instance RAM too small for the model — bigger instance type. |
| Inference 4xx/5xx, readyz green | response body; `$BASE/docs.json` | Wrong route or payload shape — endpoints are `POST /<method_name>` with JSON params. |

## Out of scope

Autoscaling groups, HTTPS termination, spot instances, multi-region. **Load balancing
too**: the AWS answer is an ALB (`aws elbv2 create-load-balancer` + a target group on
port 3000, health check `/readyz`) — hand over that pointer plus the ALB note in
`references/multi-node.md`, then stop. For fleet-scale EC2 work (launch templates, Auto
Scaling groups, Spot, SSM Session Manager): `/plugin install
aws-core@claude-plugins-official`; its `aws-compute` skill picks up where this ends.

References, all linked above where they are needed: `references/provisioning.md` (Mode B
provisioning, cost table, teardown including IAM), `references/registry-auth.md` (ECR by
instance profile or token-over-SSH, other registries), `references/multi-node.md` (N
hosts, ALB pointer).
