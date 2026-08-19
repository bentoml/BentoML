# Registry authentication on EC2 instances

Docker on the instance must be able to pull `$IMAGE`. Pick the section matching the
registry. In every case the login happens with `sudo docker login` (root's Docker config),
because all `docker` commands in this skill run under `sudo`.

## Amazon ECR

An ECR image ref is `<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>`. The login
must target **that** account+region — extract both from the ref, do not assume they equal
the deploy region:

```bash
ECR_REGISTRY=${IMAGE%%/*}                      # 123456789012.dkr.ecr.us-east-1.amazonaws.com
ECR_ACCOUNT=${ECR_REGISTRY%%.*}                # the account that OWNS the repo — from the ref,
                                               # NOT from sts get-caller-identity (cross-account!)
ECR_REGION=$(echo "$ECR_REGISTRY" | cut -d. -f4)
```

Facts (docs-verified): the token from `aws ecr get-login-password` is valid for
**12 hours**; the docker login username is always `AWS`.

### Way 1 — Instance profile (preferred for instances this skill provisions)

The instance gets an IAM role with pull-only access; the instance's own AWS CLI fetches
tokens with no credentials copied anywhere. **Requires IAM write permissions on the
deploying credentials**: `iam:CreateRole`, `iam:PutRolePolicy`, `iam:CreateInstanceProfile`,
`iam:AddRoleToInstanceProfile`, plus `iam:PassRole` at `run-instances` time.
PowerUserAccess-style SSO roles have full EC2/ECR but **none of these** — probe per
SKILL.md Step 0.7, and on the first `AccessDenied` below fall back to Way 2 (nothing to
clean up: the failed create made nothing). All four IAM commands are mutating (cost:
free) — show + confirm each, and record `$ROLE_NAME` / `$PROFILE_NAME` for teardown.

```bash
ROLE_NAME="bentoml-${SERVICE_NAME}-ec2-role"
PROFILE_NAME="bentoml-${SERVICE_NAME}-ec2-profile"

aws iam create-role --role-name "$ROLE_NAME" \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{"Effect": "Allow",
                   "Principal": {"Service": "ec2.amazonaws.com"},
                   "Action": "sts:AssumeRole"}]}'

# Minimal pull-only policy. GetAuthorizationToken does not support resource
# scoping (must be *); the three pull actions are scoped to the one repository.
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name ecr-pull \
  --policy-document "{
    \"Version\": \"2012-10-17\",
    \"Statement\": [
      {\"Effect\": \"Allow\",
       \"Action\": \"ecr:GetAuthorizationToken\",
       \"Resource\": \"*\"},
      {\"Effect\": \"Allow\",
       \"Action\": [\"ecr:BatchCheckLayerAvailability\",
                    \"ecr:GetDownloadUrlForLayer\",
                    \"ecr:BatchGetImage\"],
       \"Resource\": \"arn:aws:ecr:${ECR_REGION}:${ECR_ACCOUNT}:repository/${REPO_NAME}\"}]}"

aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME"
aws iam add-role-to-instance-profile \
  --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
```

(`REPO_NAME` is the path between the registry host and the `:tag`. `ECR_ACCOUNT` comes
from the image ref above — if it differs from the deploying account, the repository's
**resource policy** in that other account must additionally grant these pull actions to
this role, otherwise pulls are denied regardless of this identity policy.)

Create the profile **before** `run-instances` and pass
`--iam-instance-profile Name="$PROFILE_NAME"` at launch (provisioning.md step 6). Then log
in on the instance using its own role:

```bash
echo ">>> on $HOST: ECR login via instance profile"
"${SSH[@]}" "aws ecr get-login-password --region '$ECR_REGION' \
  | sudo docker login --username AWS --password-stdin '$ECR_REGISTRY'"
```

- Amazon Linux 2023 ships AWS CLI v2 (`awscli-2` package) preinstalled; guard anyway:
  `command -v aws || sudo dnf install -y awscli-2`. Ubuntu AMIs do **not** ship it — either
  install it (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
  or use Way 2, which needs nothing on the instance.
- **Existing instances (Mode A):** attaching a profile requires
  `aws ec2 associate-iam-instance-profile`, which **modifies an instance this skill did not
  create** — do it only if the user explicitly asks, after spelling that out. Default to
  Way 2 for Mode A.
- Long-lived nodes that re-pull often can install the Amazon ECR credential helper
  (https://github.com/awslabs/amazon-ecr-credential-helper) to stop thinking about the
  12-hour expiry — one-line mention, don't set it up unasked.

### Way 2 — Token over SSH (works everywhere, nothing installed on the instance)

The **local** machine (which passed `aws sts get-caller-identity` in preflight) mints the
token and pipes it straight into `docker login` on the host — it never touches a file:

```bash
echo ">>> on $HOST: ECR login (token piped from local AWS credentials)"
aws ecr get-login-password --region "$ECR_REGION" \
  | ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$SSH_USER@$HOST" \
      "sudo docker login --username AWS --password-stdin '$ECR_REGISTRY'"
```

The login survives on the instance until the token expires (**12 hours**): pulls after
that fail with `authorization token has expired` or `no basic auth credentials` — just
re-run the login. A running container is unaffected; only new pulls need fresh auth.

## Docker Hub / GHCR / generic private registry

Same pattern — pipe the secret from the local env over SSH, never into a file:

```bash
echo ">>> on $HOST: docker login $REGISTRY_HOST"
printf '%s' "$REGISTRY_TOKEN" \
  | ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$SSH_USER@$HOST" \
      "sudo docker login '$REGISTRY_HOST' --username '$REGISTRY_USER' --password-stdin"
```

- **Docker Hub**: `REGISTRY_HOST=docker.io` (may be omitted); use an access token, not the
  account password.
- **GHCR**: `REGISTRY_HOST=ghcr.io`, token is a PAT with `read:packages`. New GHCR packages
  are **private by default** — a `denied` on pull usually means visibility, not credentials.
- **Self-hosted HTTP/self-signed registries** need `insecure-registries` in
  `/etc/docker/daemon.json` on the instance + a Docker restart — that modifies the user's
  host config; confirm first. Prefer a TLS registry.

## Public images / no registry

- Public Docker Hub, public GHCR, public ECR (`public.ecr.aws/...`): no login needed —
  skip straight to `docker pull`.
- **ttl.sh** images (from the containerize skill's throwaway path) pull anonymously but
  **expire** (default 24 h max): after expiry the instance can never re-pull, so a
  `--restart unless-stopped` node loses the image on the first `docker rm` or host disk
  cleanup. Fine for a demo; warn the user and push a real registry for anything persistent.
