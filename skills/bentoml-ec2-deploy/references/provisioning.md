# Provisioning a new EC2 instance for a BentoML container (Mode B)

Every mutating command below follows the skill's rule: show the exact command plus a cost
note, get explicit confirmation, then run. Read-only lookups run freely. Use
`--region "$AWS_REGION"` on every command. Record every created ID — teardown at the
bottom operates on exactly that list.

Working variables used throughout:

```bash
AWS_REGION=us-east-1                 # confirmed with the user in preflight
SERVICE_NAME=summarization           # bento name, hyphens for underscores
NAME_PREFIX="bentoml-${SERVICE_NAME}"
```

The names are only defaults — if the user's org mandates naming conventions, substitute
theirs; nothing in the commands depends on the exact strings (teardown targets recorded
IDs, not names).

## 1. Detect the user's public IP (for SSH + service access rules)

```bash
MY_IP=$(curl -s https://checkip.amazonaws.com)
echo "$MY_IP"
```

Show it and **confirm with the user** — VPNs and corporate NAT can make this IP wrong or
unstable. If they can't confirm, ask them for the CIDR to allow.

## 2. Key pair (free)

Prefer an existing key pair (`aws ec2 describe-key-pairs --region "$AWS_REGION"` — ask
which one and where the local `.pem` lives). To create a new one (cost: free):

```bash
KEY_NAME="${NAME_PREFIX}-key"
aws ec2 create-key-pair --region "$AWS_REGION" \
  --key-name "$KEY_NAME" --key-type ed25519 --key-format pem \
  --query 'KeyMaterial' --output text > ~/.ssh/"${KEY_NAME}".pem
chmod 600 ~/.ssh/"${KEY_NAME}".pem
```

The private key is shown exactly once at creation — if the file is lost the key pair is
useless. Record `KEY_NAME` and the local path (only session-created key pairs are torn down).

## 3. Security group (free)

Created in the default VPC unless the user names one (then add `--vpc-id`). Accounts
**without a default VPC** fail right here with `VPCIdNotSpecified` (before launch ever
comes up) — ask the user for a VPC ID, add `--vpc-id`, and remember they must also supply
a **public subnet** in that VPC at launch (step 6).

```bash
SG_ID=$(aws ec2 create-security-group --region "$AWS_REGION" \
  --group-name "${NAME_PREFIX}-sg" \
  --description "BentoML ${SERVICE_NAME} - created by bentoml-ec2-deploy" \
  --tag-specifications "ResourceType=security-group,Tags=[{Key=managed-by,Value=bentoml-ec2-deploy}]" \
  --query 'GroupId' --output text)
echo "$SG_ID"

# SSH from the user's IP only:
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" \
  --group-id "$SG_ID" --protocol tcp --port 22 --cidr "${MY_IP}/32"
```

**Port 3000 scope — ask the user, do not decide for them.** Options:

1. **User's IP (default, recommended):**
   ```bash
   aws ec2 authorize-security-group-ingress --region "$AWS_REGION" \
     --group-id "$SG_ID" --protocol tcp --port 3000 --cidr "${MY_IP}/32"
   ```
2. **A specific CIDR** they name (office/VPC range) — same command, their CIDR.
3. **No 3000 rule at all** — access only via SSH tunnel (SKILL.md Step 4). Most secure.
4. **`0.0.0.0/0` (whole internet)** — only after this explicit warning and their
   confirmation: *"This exposes an unauthenticated inference API to the entire internet.
   Anyone can run inference on your instance (your compute, your cost) and probe the
   service for vulnerabilities. Recommended only for short-lived demos."*

## 4. AMI via SSM public parameters — never hardcode AMI IDs

AMI IDs differ per region and rot as images are deprecated; the SSM public parameters
always resolve to the current image in the chosen region.

Amazon Linux 2023 (default — SSH user `ec2-user`, AWS CLI v2 preinstalled, root device
`/dev/xvda`):

```bash
# x86_64 for amd64 images | arm64 for arm64 images
AMI=$(aws ssm get-parameters --region "$AWS_REGION" \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameters[0].Value' --output text)
```

Ubuntu 24.04 LTS (SSH user `ubuntu`, no AWS CLI preinstalled, root device `/dev/sda1`):

```bash
# amd64 | arm64
AMI=$(aws ssm get-parameters --region "$AWS_REGION" \
  --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
  --query 'Parameters[0].Value' --output text)
```

(Ubuntu 22.04 uses `.../22.04/stable/current/<arch>/hvm/ebs-gp2/ami-id` — note **gp2**.)

Show the resolved `$AMI` to the user, and don't trust the root-device names above blindly —
query the AMI (read-only):

```bash
ROOT_DEV=$(aws ec2 describe-images --region "$AWS_REGION" --image-ids "$AMI" \
  --query 'Images[0].RootDeviceName' --output text)
```

## 5. Instance type, disk, user-data

Instance type — match the **image architecture** and size memory to the model:

| Type | Arch | vCPU / RAM | On-demand (us-east-1, approx.) |
|---|---|---|---|
| `t3.medium` (default) | amd64 | 2 / 4 GiB | ~$0.04/hr (~$30/mo) |
| `t3.large` | amd64 | 2 / 8 GiB | ~$0.08/hr |
| `t4g.medium` | arm64 | 2 / 4 GiB | ~$0.03/hr |
| `m5.xlarge` | amd64 | 4 / 16 GiB | ~$0.19/hr |

Prices vary by region — state the estimate for the chosen region/type in the cost note
(check https://aws.amazon.com/ec2/pricing/on-demand/ if unsure). Add to every estimate:
the **public IPv4 address** costs $0.005/hr (~$3.65/mo) — since Feb 2024 this applies to
all public IPv4 addresses, including the auto-assigned one from
`--associate-public-ip-address`. GPU types (`g4dn.*`,
`g5.*`) cost 10–25x more and need a GPU AMI or driver install — flag this clearly.

**Root volume**: AMI defaults (8 GiB) are too small for most bento images (models are baked
in). Size it to at least 2x the image size, minimum 30 GiB. gp3 costs ~$0.08/GB-month
(50 GiB ≈ $4/mo).

**User-data** installs and enables Docker so `--restart unless-stopped` containers survive
reboots. Write it to a scratch file (no secrets in user-data — it is readable from the
instance metadata service):

```bash
# Amazon Linux 2023:
cat > /tmp/bentoml-user-data.sh <<'EOF'
#!/bin/bash
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user
EOF

# Ubuntu instead:
#   apt-get update && apt-get install -y docker.io
#   systemctl enable --now docker
#   usermod -aG docker ubuntu
```

## 6. Launch

Cost note to show: instance $/hr + EBS $/mo, "bills until terminated". If using an ECR
instance profile, create it first (see `registry-auth.md`) and add
`--iam-instance-profile Name="$PROFILE_NAME"` to this command.

```bash
INSTANCE_ID=$(aws ec2 run-instances --region "$AWS_REGION" \
  --image-id "$AMI" \
  --instance-type t3.medium \
  --count 1 \
  --key-name "$KEY_NAME" \
  --security-group-ids "$SG_ID" \
  --associate-public-ip-address \
  --user-data file:///tmp/bentoml-user-data.sh \
  --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEV\",\"Ebs\":{\"VolumeSize\":50,\"VolumeType\":\"gp3\"}}]" \
  --tag-specifications \
    "ResourceType=instance,Tags=[{Key=Name,Value=${NAME_PREFIX}},{Key=managed-by,Value=bentoml-ec2-deploy}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "$INSTANCE_ID"
```

- No `--subnet-id` → AWS picks a default-VPC subnet. Accounts without a default VPC get an
  error; then ask the user for a **public** subnet ID and pass `--subnet-id`.
- Multi-node: `--count N`, then capture all IDs with `--query 'Instances[].InstanceId'`.
- If launch fails with `InvalidParameterValue` for a just-created instance profile, IAM
  propagation is lagging — wait ~10 s and retry once.

## 7. Wait, get the address, wait for Docker

```bash
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
HOST=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "$HOST"
```

SSH becomes available before user-data finishes; retry until cloud-init is done and Docker
answers (one shell invocation):

```bash
SSH_USER=ec2-user      # ubuntu for Ubuntu AMIs
for i in $(seq 1 36); do   # up to ~3 min
  if ssh -i ~/.ssh/"${KEY_NAME}".pem -o StrictHostKeyChecking=accept-new \
       -o ConnectTimeout=5 -o BatchMode=yes "$SSH_USER@$HOST" \
       'sudo docker info >/dev/null 2>&1 && echo DOCKER_READY' \
       2>/dev/null | grep -q DOCKER_READY; then echo READY; break; fi
  sleep 5
done
```

Still not ready after the loop? `cloud-init status` over SSH shows whether user-data is
still running (`status: running`) or failed (`status: error` → check
`/var/log/cloud-init-output.log`).

Once ready, define the SSH array every later step uses (SKILL.md Steps 2–4, teardown,
registry-auth.md):

```bash
KEY=~/.ssh/"${KEY_NAME}".pem
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$SSH_USER@$HOST")
```

If it never becomes ready: `aws ec2 get-console-output --region "$AWS_REGION" --instance-id "$INSTANCE_ID" --latest --output text | tail -50`
(read-only) shows boot problems. Then continue with SKILL.md Step 2 (registry auth).

## Teardown (session-created resources only, confirm each command)

Order matters: the SG cannot be deleted while the instance references it, and the IAM role
cannot be deleted while attached to the profile.

```bash
# 1. Instance — billing stops here
aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-terminated --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"

# 2. Security group (free to keep, but clean up)
aws ec2 delete-security-group --region "$AWS_REGION" --group-id "$SG_ID"

# 3. Key pair — only if created this session; also remove the local file
aws ec2 delete-key-pair --region "$AWS_REGION" --key-name "$KEY_NAME"
rm -f ~/.ssh/"${KEY_NAME}".pem

# 4. IAM (only if registry-auth.md created these this session)
aws iam remove-role-from-instance-profile \
  --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
aws iam delete-instance-profile --instance-profile-name "$PROFILE_NAME"
aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name ecr-pull
aws iam delete-role --role-name "$ROLE_NAME"
```

Never run any of these against IDs the user supplied (Mode A hosts, pre-existing SGs, their
own key pairs, pre-existing roles) — those are theirs, not this session's.
