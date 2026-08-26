# Registry authentication on EC2 instances

The instance must be able to pull `$IMAGE`. Logins use `sudo docker login` (root's
Docker config), since every `docker` command here runs under `sudo`.

## Amazon ECR

An ECR ref is `<account>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>`. Log in to
**that** account+region, extracted from the ref — it need not match the deploy region:

```bash
ECR_REGISTRY=${IMAGE%%/*}                      # 123456789012.dkr.ecr.us-east-1.amazonaws.com
ECR_ACCOUNT=${ECR_REGISTRY%%.*}                # the account that OWNS the repo — from the ref,
                                               # NOT from sts get-caller-identity (cross-account!)
ECR_REGION=$(echo "$ECR_REGISTRY" | cut -d. -f4)
```

The token from `aws ecr get-login-password` is valid for **12 hours**; the login
username is always `AWS`.

### Way 1 — Instance profile (preferred for provisioned instances)

The instance gets a pull-only role and fetches its own tokens; no credentials are
copied. **Requires IAM writes on the deploying credentials**: `iam:CreateRole`,
`iam:PutRolePolicy`, `iam:CreateInstanceProfile`, `iam:AddRoleToInstanceProfile`, plus
`iam:PassRole` at `run-instances` — probe and fall back to Way 2 per SKILL.md Step 0.7;
a failed create leaves nothing to clean up. All four commands are mutating (free): show
+ confirm each, record `$ROLE_NAME` / `$PROFILE_NAME` for teardown.

```bash
ROLE_NAME="bentoml-${SERVICE_NAME}-ec2-role"
PROFILE_NAME="bentoml-${SERVICE_NAME}-ec2-profile"

aws iam create-role --role-name "$ROLE_NAME" --assume-role-policy-document \
  '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

# Minimal pull-only policy: GetAuthorizationToken cannot be resource-scoped (must be *);
# the three pull actions are scoped to the one repository.
aws iam put-role-policy --role-name "$ROLE_NAME" --policy-name ecr-pull --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [
    {\"Effect\": \"Allow\", \"Action\": \"ecr:GetAuthorizationToken\", \"Resource\": \"*\"},
    {\"Effect\": \"Allow\",
     \"Action\": [\"ecr:BatchCheckLayerAvailability\", \"ecr:GetDownloadUrlForLayer\", \"ecr:BatchGetImage\"],
     \"Resource\": \"arn:aws:ecr:${ECR_REGION}:${ECR_ACCOUNT}:repository/${REPO_NAME}\"}]}"

aws iam create-instance-profile --instance-profile-name "$PROFILE_NAME"
aws iam add-role-to-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
```

`REPO_NAME` is the path between the registry host and the `:tag`. If `ECR_ACCOUNT` is
not the deploying account, that account's repository **resource policy** must grant
these actions to the role too, or pulls are denied regardless.

Create the profile **before** `run-instances`, pass `--iam-instance-profile
Name="$PROFILE_NAME"` at launch (provisioning.md step 6), then log in with the
instance's role:

```bash
echo ">>> on $HOST: ECR login via instance profile"
"${SSH[@]}" "aws ecr get-login-password --region '$ECR_REGION' \
  | sudo docker login --username AWS --password-stdin '$ECR_REGISTRY'"
```

- AL2023 ships AWS CLI v2 (`awscli-2`); guard anyway: `command -v aws || sudo dnf
  install -y awscli-2`. Ubuntu AMIs don't: install it
  (https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) or use
  Way 2, which installs nothing.
- **Mode A:** attaching a profile (`aws ec2 associate-iam-instance-profile`) **modifies
  an instance this skill did not create** — only on explicit request, after saying so.
  Default to Way 2.
- Nodes that re-pull often can install the Amazon ECR credential helper
  (https://github.com/awslabs/amazon-ecr-credential-helper) and forget the expiry —
  mention it, don't set it up unasked.

### Way 2 — Token over SSH (works everywhere, installs nothing)

The **local** machine (it passed `aws sts get-caller-identity` in preflight) mints the
token and pipes it into `docker login` on the host — never a file, a command line or
user-data:

```bash
echo ">>> on $HOST: ECR login (token piped from local AWS credentials)"
aws ecr get-login-password --region "$ECR_REGION" \
  | ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$SSH_USER@$HOST" \
      "sudo docker login --username AWS --password-stdin '$ECR_REGISTRY'"
```

The login lasts **12 hours**; later pulls fail with `authorization token has expired` or
`no basic auth credentials`, so re-run it. Running containers are fine — only pulls need
auth.

## Docker Hub / GHCR / generic private registry

Same pattern: pipe the secret from the local env over SSH, never into a file.

```bash
echo ">>> on $HOST: docker login $REGISTRY_HOST"
printf '%s' "$REGISTRY_TOKEN" \
  | ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$SSH_USER@$HOST" \
      "sudo docker login '$REGISTRY_HOST' --username '$REGISTRY_USER' --password-stdin"
```

- **Docker Hub**: `REGISTRY_HOST=docker.io` (omittable); an access token, not the
  account password.
- **GHCR**: `REGISTRY_HOST=ghcr.io`, token = PAT with `read:packages`. New packages are
  **private by default**, so `denied` usually means visibility, not credentials.
- **Self-hosted HTTP/self-signed registries** need `insecure-registries` in
  `/etc/docker/daemon.json` plus a Docker restart — that edits host config, so confirm
  first. Prefer TLS.

## Public images / no registry

- Public Docker Hub, public GHCR, public ECR (`public.ecr.aws/...`): no login, straight
  to `docker pull`.
- **ttl.sh** images (the containerize skill's throwaway path) pull anonymously but
  **expire** (24 h max by default). After that the instance can never re-pull, so a
  `--restart unless-stopped` node loses the image at the first `docker rm` or disk
  cleanup. Fine for a demo — warn the user, push to a real registry for anything
  persistent.
