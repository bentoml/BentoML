# Multi-node deployment (N identical hosts)

Multi-node = the single-node flow (SKILL.md Steps 2–4) per host. No coordination; each
node runs its own copy.

Provision N at once: `run-instances --count N` (provisioning.md step 6), capturing
IDs/IPs:

```bash
INSTANCE_IDS=$(aws ec2 run-instances ... --count 3 --query 'Instances[].InstanceId' --output text)
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids $INSTANCE_IDS
HOSTS=$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids $INSTANCE_IDS \
  --query 'Reservations[].Instances[].PublicIpAddress' --output text)
```

Deploy loop — echo the host before each mutating step, and **fail loudly per host**:

```bash
for HOST in $HOSTS; do
  SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -o BatchMode=yes "$SSH_USER@$HOST")

  # Freshly provisioned host? user-data may still be installing Docker — wait for the daemon (same gate as provisioning.md §7):
  DOCKER_OK=""
  for i in $(seq 1 36); do   # up to ~3 min per host
    "${SSH[@]}" 'sudo docker info >/dev/null 2>&1' 2>/dev/null && { DOCKER_OK=1; break; }
    sleep 5
  done
  [ -n "$DOCKER_OK" ] || { echo "DOCKER NOT READY on $HOST"; continue; }
  echo ">>> on $HOST: registry login"   # per registry-auth.md (the instance-profile way needs no per-host secret)
  echo ">>> on $HOST: pull + run"
  "${SSH[@]}" "sudo docker pull '$IMAGE'" || { echo "PULL FAILED on $HOST"; continue; }
  "${SSH[@]}" "sudo docker run -d --name '$SERVICE_NAME' --restart unless-stopped -p 3000:3000 '$IMAGE'" || { echo "RUN FAILED on $HOST"; continue; }
done
```

**Verify every host** (SKILL.md Step 4: readyz loop + one content-checked inference
request each) and report a table of host, container status, readyz, inference OK/FAIL —
one dead node hides behind healthy siblings.

Teardown covers **all** session-created instance IDs in one `terminate-instances` call.

## Load balancing (out of scope)

This skill creates no load balancers; the standard AWS answer is an **Application Load
Balancer**:

- `aws elbv2 create-load-balancer` + `create-target-group` (HTTP, port 3000, health
  check `/readyz`) + `register-targets` with the instance IDs + `create-listener`.
- The ALB needs its own SG; the instances' SG then allows 3000 **from the ALB's SG**,
  not a user IP.
- Cost: ~$16/mo base + LCU charges, billing until deleted.
- Docs:
  https://docs.aws.amazon.com/elasticloadbalancing/latest/application/application-load-balancer-getting-started.html

Hand over the pointer and stop; managing an ALB is beyond basic deployment.
