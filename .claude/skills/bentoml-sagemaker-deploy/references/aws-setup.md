# AWS setup details — IAM role, instance sizing, ECR, CloudWatch debugging

## Minimal execution role (create only if no suitable role exists)

Trust policy and permission set taken from the AWS docs
([execution roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html),
"CreateModel API: Execution Role Permissions"). Confirm with the user before each
`iam` mutating command (free, but it creates account-level resources).

**Caller permissions required:** `iam:CreateRole` and `iam:PutRolePolicy` for this
section, plus `iam:PassRole` on the resulting role at `create-model` time. The
AWS-managed **PowerUserAccess** policy (very common) includes NONE of these IAM writes —
if `iam create-role` returns `AccessDenied`, do not retry or work around it: follow the
"AccessDenied branch" below.

```bash
ROLE_NAME=bentoml-sagemaker-exec
POLICY_NAME="${ROLE_NAME}-policy"

cat > /tmp/sm-trust.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "sagemaker.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

cat > /tmp/sm-perms.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "cloudwatch:PutMetricData",
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams",
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
EOF

aws iam create-role --role-name "$ROLE_NAME" \
  --assume-role-policy-document file:///tmp/sm-trust.json
aws iam put-role-policy --role-name "$ROLE_NAME" \
  --policy-name "$POLICY_NAME" \
  --policy-document file:///tmp/sm-perms.json

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query Role.Arn --output text)
```

## AccessDenied branch (caller cannot create IAM roles)

If `iam create-role` (or `put-role-policy`) fails with `AccessDenied`:

1. Ask the user for an **existing execution role ARN** — any role whose trust policy
   includes `sagemaker.amazonaws.com` works; the name is irrelevant. (The trust-policy
   scan in SKILL.md Step 3 finds them when the caller has IAM read access.)
2. If no such role exists in the account, hand the user this **ready-to-send admin
   snippet** (the two heredocs above written to files, plus these two commands) and
   **STOP — do not attempt workarounds** (no inline policies on the caller, no
   wildcard-role guessing):

   ```text
   Please create a SageMaker execution role for me and send back its ARN:

   1. Save the trust policy JSON as sm-trust.json and the permission policy JSON
      as sm-perms.json (contents included below).
   2. Run:
      aws iam create-role --role-name bentoml-sagemaker-exec \
        --assume-role-policy-document file://sm-trust.json
      aws iam put-role-policy --role-name bentoml-sagemaker-exec \
        --policy-name bentoml-sagemaker-exec-policy \
        --policy-document file://sm-perms.json
   3. Also make sure my IAM principal is allowed iam:PassRole on that role
      (needed when I call sagemaker create-model).
   ```

   Attach both JSON documents verbatim when sending the snippet. Resume the skill at
   Step 3 once the user supplies the ARN.

Optionally scope the `ecr:BatchCheckLayerAvailability`/`GetDownloadUrlForLayer`/
`BatchGetImage` actions to
`arn:aws:ecr:<region>:<account>:repository/<repo>` instead of `*` (AWS shows both
forms; `ecr:GetAuthorizationToken`, logs, and metrics need `*`). No S3 permissions
are needed because BentoML bakes the model into the image — there is no `ModelDataUrl`.

Notes:
- IAM propagation: a freshly created role can take ~10 s to become assumable; if
  `create-model` fails with a role/assume error immediately after creation, wait and retry once.
- The **caller's** identity (not the execution role) also needs `iam:PassRole` on this
  role to run `create-model` — usually already true for admin-ish users; surface the
  error clearly if not.
- Teardown of a skill-created role (only if created this session): `aws iam
  delete-role-policy --role-name "$ROLE_NAME" --policy-name "$POLICY_NAME"` then
  `aws iam delete-role --role-name "$ROLE_NAME"`.

## Instance types and cost (real-time endpoints)

Hourly prices are region-dependent — always point the user at
https://aws.amazon.com/sagemaker/pricing/ and label numbers as approximate (us-east-1):

| Instance | vCPU / RAM | GPU | ~$/hour | Use for |
|---|---|---|---|---|
| ml.t2.medium | 2 / 4 GiB | — | ~$0.06 | tiny demos only (burstable) |
| **ml.m5.large** (default) | 2 / 8 GiB | — | ~$0.115 | CPU inference default |
| ml.m5.xlarge | 4 / 16 GiB | — | ~$0.23 | bigger CPU models |
| ml.c5.xlarge | 4 / 8 GiB | — | ~$0.20 | compute-bound CPU |
| ml.g4dn.xlarge | 4 / 16 GiB | 1× T4 | ~$0.74 | small GPU models |
| ml.g5.xlarge | 4 / 16 GiB | 1× A10G | ~$1.41 | mid GPU / small LLMs |

Rules:
- An endpoint bills **per instance-hour from InService until `delete-endpoint`**, idle
  or not. Always leave the user with the teardown block.
- All of the above are **linux/amd64**. GPU instances need no image changes beyond the
  model's own CUDA deps (SageMaker provides the NVIDIA runtime; don't bundle drivers).
- Memory sizing: the image is loaded and the model unpacked in RAM — if the container
  OOMs (endpoint fails, logs stop abruptly), move up a size.

## ECR specifics

- The image **must** be in ECR **in the endpoint's region** (verified:
  `ContainerDefinition.Image` docs — "must be in the same region as the model or
  endpoint you are creating"). Cross-region or Docker Hub images will not work here
  (private non-ECR registries require VPC config — out of scope).
- Repository names: lowercase; convert bento-name underscores to hyphens.
- Push flow (login, create-repository, naming) is in the `bentoml-containerize` skill's
  `references/registries.md` (AWS ECR section) — reuse it.
- ECR storage costs ~$0.10/GB-month — worth mentioning for multi-GB model images; the
  optional teardown deletes the repo with `--force` (removes all images in it).

## CloudWatch debugging

Container stdout/stderr lands in log group `/aws/sagemaker/Endpoints/<endpoint-name>`,
one stream per instance (`AllTraffic/<instance-id>`):

```bash
aws logs describe-log-streams \
  --log-group-name "/aws/sagemaker/Endpoints/${EP_NAME}" \
  --region "$AWS_REGION" --query 'logStreams[].logStreamName'

aws logs tail "/aws/sagemaker/Endpoints/${EP_NAME}" --region "$AWS_REGION" --since 1h --follow
```

What to look for:
- `Starting production HTTP BentoServer ... listening on http://localhost:3000` —
  the port env var is missing: the model's `Environment` must contain `BENTOML_PORT=8080`
  (models are immutable — create a new model with the fix).
- Python import errors / missing packages — the bento's runtime spec is incomplete; go
  back to the `bentoml-containerize` skill's Step 1.
- No log group at all — the container never started: usually image pull failure (role
  permissions, wrong region) or `exec format error` (arm64 image); `describe-endpoint`'s
  `FailureReason` says which.
- Endpoint metrics (invocations, errors, latency) are in CloudWatch under the
  `AWS/SageMaker` namespace if deeper analysis is needed.
