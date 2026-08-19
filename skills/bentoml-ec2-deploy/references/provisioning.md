# Provisioning a new EC2 instance for a BentoML container (Mode B)

Every mutating command: shown with a cost note, confirmed, then run. Read-only lookups
run freely. `--region "$AWS_REGION"` throughout. Record every created ID — teardown uses
that list only. Names are defaults; substitute the user's, since teardown targets IDs.

```bash
AWS_REGION=us-east-1                 # confirmed with the user in preflight
SERVICE_NAME=summarization           # bento name, hyphens for underscores
NAME_PREFIX="bentoml-${SERVICE_NAME}"
```

## 1. The user's public IP (for SSH + service rules)

```bash
MY_IP=$(curl -s https://checkip.amazonaws.com); echo "$MY_IP"
```

Confirm with the user: VPNs and corporate NAT skew it; if unsure, ask for the CIDR.

## 2. Key pair (free)

Prefer an existing pair (`aws ec2 describe-key-pairs --region "$AWS_REGION"`): ask
which, and where its `.pem` lives. To create one (free):

```bash
KEY_NAME="${NAME_PREFIX}-key"
aws ec2 create-key-pair --region "$AWS_REGION" --key-name "$KEY_NAME" --key-type ed25519 --key-format pem \
  --query 'KeyMaterial' --output text > ~/.ssh/"${KEY_NAME}".pem
chmod 600 ~/.ssh/"${KEY_NAME}".pem
```

The private key is shown once, at creation: lose the file, lose the pair. Record
`KEY_NAME` + path; only session-created pairs are torn down.

## 3. Security group (free)

Default VPC unless the user names one (add `--vpc-id`). Accounts **without a default
VPC** fail here with `VPCIdNotSpecified`, before launch: ask for a VPC ID, add
`--vpc-id`, and they must also supply a **public subnet** at launch (step 6).

```bash
SG_ID=$(aws ec2 create-security-group --region "$AWS_REGION" --group-name "${NAME_PREFIX}-sg" \
  --description "BentoML ${SERVICE_NAME} - created by bentoml-ec2-deploy" \
  --tag-specifications "ResourceType=security-group,Tags=[{Key=managed-by,Value=bentoml-ec2-deploy}]" \
  --query 'GroupId' --output text)
echo "$SG_ID"

# SSH from the user's IP only:
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" --protocol tcp --port 22 --cidr "${MY_IP}/32"
```

**Port 3000 scope — ask the user, don't decide for them:**

1. **User's IP (default, recommended):**
   ```bash
   aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$SG_ID" --protocol tcp --port 3000 --cidr "${MY_IP}/32"
   ```
2. **A CIDR they name** (office/VPC range) — same command, their CIDR.
3. **No 3000 rule** — tunnel only (SKILL.md Step 4). Most secure.
4. **`0.0.0.0/0` (whole internet)** — only after this warning and their confirmation:
   *"This exposes an unauthenticated inference API to the entire internet. Anyone can
   run inference on your instance (your compute, your cost) and probe the service for
   vulnerabilities. Recommended only for short-lived demos."*

## 4. AMI via SSM public parameters — never hardcode AMI IDs

AMI IDs differ per region and rot as images are deprecated; these parameters always
resolve to the region's current image.

```bash
# Amazon Linux 2023 — SSH user ec2-user, AWS CLI v2 preinstalled, root device /dev/xvda. Suffix: x86_64 | arm64.
AMI=$(aws ssm get-parameters --region "$AWS_REGION" --output text --query 'Parameters[0].Value' --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64)

# Ubuntu 24.04 LTS — SSH user ubuntu, no AWS CLI preinstalled, root device /dev/sda1. Arch: amd64 | arm64.
# (Ubuntu 22.04 uses .../22.04/stable/current/<arch>/hvm/ebs-gp2/ami-id — note gp2.)
AMI=$(aws ssm get-parameters --region "$AWS_REGION" --output text --query 'Parameters[0].Value' --names /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id)
```

Show the resolved `$AMI`. Verify the root device rather than the comments (read-only):
`ROOT_DEV=$(aws ec2 describe-images --region "$AWS_REGION" --image-ids "$AMI" --query
'Images[0].RootDeviceName' --output text)`

## 5. Instance type, disk, user-data

Match the **image architecture**, size RAM to the model:

| Type | Arch | vCPU / RAM | On-demand (us-east-1, approx.) |
|---|---|---|---|
| `t3.medium` (default) | amd64 | 2 / 4 GiB | ~$0.04/hr (~$30/mo) |
| `t3.large` | amd64 | 2 / 8 GiB | ~$0.08/hr |
| `t4g.medium` | arm64 | 2 / 4 GiB | ~$0.03/hr |
| `m5.xlarge` | amd64 | 4 / 16 GiB | ~$0.19/hr |

Prices vary by region — state the estimate for the chosen region/type
(https://aws.amazon.com/ec2/pricing/on-demand/ if unsure). Every estimate also carries
the **public IPv4 address**: $0.005/hr (~$3.65/mo), charged since Feb 2024 on all public
IPv4 addresses, the auto-assigned one from `--associate-public-ip-address` included. GPU
types (`g4dn.*`, `g5.*`) cost 10–25x more and need a GPU AMI or driver install — flag
that.

**Root volume**: the 8 GiB AMI default is too small, since models are baked in. At least
2x the image size, minimum 30 GiB; gp3 ~$0.08/GB-month (50 GiB ≈ $4/mo).

**User-data** installs and enables Docker so `--restart unless-stopped` survives
reboots. Write it to a scratch file. No secrets in it: the metadata service serves it
back.

```bash
# Amazon Linux 2023:
cat > /tmp/bentoml-user-data.sh <<'EOF'
#!/bin/bash
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user
EOF

# Ubuntu instead: apt-get update && apt-get install -y docker.io; systemctl enable --now docker; usermod -aG docker ubuntu
```

## 6. Launch

Cost note: instance $/hr + EBS $/mo, "bills until terminated". With an ECR instance
profile, create it first (`registry-auth.md`) and add `--iam-instance-profile
Name="$PROFILE_NAME"`.

```bash
INSTANCE_ID=$(aws ec2 run-instances --region "$AWS_REGION" --image-id "$AMI" --instance-type t3.medium --count 1 \
  --key-name "$KEY_NAME" --security-group-ids "$SG_ID" --associate-public-ip-address \
  --user-data file:///tmp/bentoml-user-data.sh \
  --block-device-mappings "[{\"DeviceName\":\"$ROOT_DEV\",\"Ebs\":{\"VolumeSize\":50,\"VolumeType\":\"gp3\"}}]" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME_PREFIX}},{Key=managed-by,Value=bentoml-ec2-deploy}]" \
  --query 'Instances[0].InstanceId' --output text)
echo "$INSTANCE_ID"
```

- No `--subnet-id` → a default-VPC subnet. Accounts without one error out; ask for a
  **public** subnet ID, pass `--subnet-id`.
- Multi-node: `--count N`, capture IDs with `--query 'Instances[].InstanceId'`.
- `InvalidParameterValue` on a just-created instance profile = IAM propagation lag; wait
  ~10 s, retry once.

## 7. Wait, get the address, wait for Docker

```bash
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"
HOST=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text); echo "$HOST"
```

SSH answers before user-data finishes, so retry until Docker answers (one shell
invocation):

```bash
SSH_USER=ec2-user      # ubuntu for Ubuntu AMIs
for i in $(seq 1 36); do   # up to ~3 min
  if ssh -i ~/.ssh/"${KEY_NAME}".pem -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -o BatchMode=yes \
       "$SSH_USER@$HOST" 'sudo docker info >/dev/null 2>&1 && echo DOCKER_READY' 2>/dev/null | grep -q DOCKER_READY
  then echo READY; break; fi
  sleep 5
done
```

Still not ready? `cloud-init status` over SSH says running (`status: running`) or failed
(`status: error` → `/var/log/cloud-init-output.log`). Boot problems: `aws ec2
get-console-output --region "$AWS_REGION" --instance-id "$INSTANCE_ID" --latest --output
text | tail -50` (read-only).

Then define the SSH array used by SKILL.md Steps 2–4, teardown and registry-auth.md, and
continue with SKILL.md Step 2:

```bash
KEY=~/.ssh/"${KEY_NAME}".pem
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$SSH_USER@$HOST")
```

## Teardown (session-created resources only, confirm each command)

Order matters: the SG can't go while the instance references it, nor the role while
attached to the profile.

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
aws iam remove-role-from-instance-profile --instance-profile-name "$PROFILE_NAME" --role-name "$ROLE_NAME"
aws iam delete-instance-profile --instance-profile-name "$PROFILE_NAME"
aws iam delete-role-policy --role-name "$ROLE_NAME" --policy-name ecr-pull
aws iam delete-role --role-name "$ROLE_NAME"
```

Never aim these at IDs the user supplied: Mode A hosts, pre-existing SGs, their own key
pairs and pre-existing roles are theirs.
