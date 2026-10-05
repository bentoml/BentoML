============
Agent skills
============

.. meta::
    :description lang=en:
        Use BentoML's agent skills to let a coding agent build, containerize, and deploy your BentoML Service to Kubernetes or EC2.

An **agent skill** is a set of instructions that a coding agent loads on demand when your request matches what the skill covers. The format is defined by the open `Agent Skills specification <https://agentskills.io/specification>`_ and works across hosts such as Claude Code, OpenAI Codex, and Cursor.

BentoML provides several skills in the `skills/ <https://github.com/bentoml/BentoML/tree/main/skills>`_ directory of the repository. They teach your agent to create a BentoML project, build a Bento, and then take it all the way to a verified, running deployment on infrastructure you own, such as a Kubernetes cluster or plain AWS EC2 instances. They use only open-source tools, including the ``bentoml`` CLI, Docker, ``kubectl`` with plain manifests, ``ssh``, and the AWS CLI.

The skills follow the same pipeline you would follow by hand. See :doc:`hello-world` and :doc:`packaging-for-deployment` for that workflow in detail.

.. note::

   The skills cover **basic deployment on self-managed infrastructure**, not on BentoCloud.

Install the skills
------------------

Each skill is a directory containing a ``SKILL.md`` file plus optional ``references/`` and ``templates/``.

.. tab-set::

    .. tab-item:: Any agent (recommended)

        The `skills <https://github.com/vercel-labs/skills>`_ CLI copies them straight from GitHub.

        .. code-block:: bash

            npx skills add bentoml/BentoML -a claude-code -g   # user scope
            npx skills add bentoml/BentoML -a codex -g
            npx skills add bentoml/BentoML                     # project scope; prompts for the agent

        Use ``-s <skill>`` for a subset and ``npx skills update`` to refresh.

    .. tab-item:: Claude Code plugin

        The BentoML repository is a Claude Code plugin marketplace, so all the skills are installed and updated together.

        .. code-block:: bash

            claude plugin marketplace add bentoml/BentoML
            claude plugin install bentoml-deploy@bentoml

        Alternatively, run ``/plugin marketplace add bentoml/BentoML`` inside a session. Update with ``/plugin update bentoml-deploy@bentoml``. If you previously copied the skills into ``~/.claude/skills/`` by hand, delete those copies. Otherwise both sets stay active.

    .. tab-item:: Manual copy

        This works for every host, and lets you commit the skills into your own project so the whole team shares one set.

        .. code-block:: bash

            git clone --depth 1 https://github.com/bentoml/BentoML.git /tmp/bentoml
            mkdir -p ~/.claude/skills
            cp -r /tmp/bentoml/skills/bentoml-* ~/.claude/skills/

            # or per project, committed to your repository:
            mkdir -p my-ml-project/.claude/skills
            cp -r /tmp/bentoml/skills/bentoml-* my-ml-project/.claude/skills/

To confirm the installation, start a new session and type ``/``. The skills appear as commands. Plain requests work too: "deploy my BentoML Service to my Kubernetes cluster" loads the right skill. If nothing appears, check that each skill is a **directory** containing a ``SKILL.md`` file (copying the ``SKILL.md`` files alone is the usual mistake) and restart the agent.

Understand the skills
---------------------

.. list-table::
   :widths: 28 72
   :header-rows: 1

   * - Skill
     - What it does
   * - ``bentoml-create-bento``
     - Creates the project itself: the ``service.py``, its runtime environment, and a built Bento. It works from scratch or converts existing code, such as a script, a notebook, a FastAPI or Flask app, an MLflow model, or a BentoML 1.1 Runner project. Start here if you don't have a Bento yet.
   * - ``bentoml-containerize``
     - Builds your project into a Bento, containerizes it, smoke-tests the container locally, and pushes it to your registry (Docker Hub, GHCR, ECR, a private registry, a ``kind``/``minikube`` local load, or ``ttl.sh``). The entry point for every target.
   * - ``bentoml-k8s-deploy``
     - Deploys a pushed image to Kubernetes. It writes one ``deploy/config.yml``, renders plain manifests from it - one Deployment + Service per BentoML Service, plus optional HPA and Ingress - applies them in dependency order, and verifies with a real inference request. It ships a troubleshooting runbook indexed by symptom.
   * - ``bentoml-ec2-deploy``
     - Runs a pushed image under Docker on one or more plain EC2 instances: your existing instances over SSH, or a fresh instance provisioned through the AWS CLI. It includes ECR authentication, verification, and teardown.
   * - ``bentoml-deploy-scriptgen``
     - Generates a standalone, committable deploy bundle (``deploy/deploy.py`` plus one ``config.yml``) that builds, pushes, deploys, and verifies with no agent involved. You can use it for your own CI/CD workflow.

How to use the skills
---------------------

Suppose you have the text summarization model from :doc:`hello-world`, written the way you would first write it. It's a plain script with no BentoML in it:

.. code-block:: python
    :caption: `summarize.py`

    from transformers import pipeline

    pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    def summarize(text: str) -> str:
        return pipe(text)[0]["summary_text"]

    if __name__ == "__main__":
        print(summarize("Breaking News: In an astonishing turn of events, the small town of "
                        "Willow Creek has been taken by storm as local resident Jerry "
                        "Thompson's cat, Whiskers, performed what witnesses are calling a "
                        "'miraculous and gravity-defying leap.' ..."))

The skills take it from here to a deployment you can call. Work in an agent session started from the project directory. For example:

.. code-block:: bash

    cd ~/projects/summarization && claude

Step 1: Convert the script into a BentoML project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Describe what you want in plain language. ``bentoml-create-bento`` loads on its own and will return an ordinary BentoML project you can read and edit, similar to the :doc:`hello-world` example. Skip to the next step if you already have a BentoML project.

.. code-block:: bash

    > convert summarize.py into a BentoML service

    I'll start by looking at the file.
    This is exactly what the bentoml-create-bento skill covers — let me load it.
    Skill(bentoml-deploy:bentoml-create-bento)
      Successfully loaded skill
    Path B (conversion). Let me check the environment and capture the anchor case by running the original script.
    Let me capture the anchor output from the original script.
    Background command "Run original script to capture anchor output" completed (exit code 0)
    Anchor captured. Now writing the project files.
    Wrote service.py, requirements.txt, .bentoignore, test_service.py
    bentoml serve -> /readyz 200; /docs.json shows summarize(text: str) -> str
    Anchor request -> "Whiskers, a tabby cat, jumped 20 feet to catch a fly..."
    Same input through summarize.py -> identical output
    pytest -> 1 passed
    Next: /bentoml-containerize

Step 2: Containerize and push
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The agent may ask you some questions for clarification, with defaults you can accept.

.. code-block:: bash

    > containerize it and load it into my kind cluster

    Loading skill: bentoml-containerize
      Preflight: bentoml 1.4.x OK - docker daemon OK (x86_64) - found ./service.py
      1. Which registry?  2. Target CPU architecture?
    > kind load, amd64
      bentoml build -> summarization:lyt7deofjkmparqe
      bentoml containerize -> summarization:lyt7deofjkmparqe (linux/amd64)
      Smoke test: docker run -> /readyz 200, POST /summarize -> correct summary
      kind load docker-image summarization:lyt7deofjkmparqe --name kind

Step 3: Deploy
^^^^^^^^^^^^^^

Nothing is applied to your cluster or AWS account until you confirm it. The deployment will be verified based on the response body.

.. tab-set::

    .. tab-item:: Kubernetes

        .. code-block:: bash

            > deploy it to my kind cluster

            Loading skill: bentoml-k8s-deploy
              Contexts: kind-kind (current), prod-eks - which one should I deploy to?
            > kind-kind
              bento.yaml: 1 Service (Summarization), entry Summarization, no dependencies
              Parameters: namespace default - replicas 1 - cpu 500m/2 - memory 1Gi/4Gi
              Wrote deploy/config.yml. Review without touching the cluster:
                python3 deploy/deploy.py --target k8s --render-only /tmp/look
              About to apply: context=kind-kind namespace=default
            > yes
              rollout status: deployment "summarization" successfully rolled out
              POST /summarize through a port-forward -> the anchor output
              Reach it: kubectl port-forward svc/summarization 3000:3000

    .. tab-item:: EC2

        .. code-block:: bash

            > run it on a new EC2 instance in us-west-2 instead

            Loading skill: bentoml-ec2-deploy
              Preflight: AWS CLI v2 OK - credentials OK - image is in ECR
              Instance type t3.medium (~$0.04/hr + EBS + $0.005/hr public IPv4),
              root volume 50 GiB, port 3000 open to your IP only, ECR auth via
              instance profile. Proceed?
            > yes
              run-instances -> i-0abc123 - security group - key pair
              docker run -d -p 3000:3000 --restart unless-stopped <ecr-image>
              /readyz 200 over an SSH tunnel
              POST /summarize -> the anchor output
              Teardown when you are done: /bentoml-ec2-deploy, "tear down i-0abc123"

Step 4: Make it repeatable
^^^^^^^^^^^^^^^^^^^^^^^^^^

The first deployment is worth doing interactively. You can automate future deployments by using the scripts created by the ``bentoml-deploy-scriptgen`` skill.

.. code-block:: bash

    > generate a deploy script I can run from CI

    Loading skill: bentoml-deploy-scriptgen
      Wrote deploy/deploy.py, deploy/config.yml, deploy/README.md
      Verified: --check-only --local-only passes without touching the cluster
      python3 deploy/deploy.py --target k8s   # build -> push -> deploy -> verify

The same pipeline can run from your terminal or from CI, and the agent is only needed when something changes.

Onboarding best practices
-------------------------

The skills already handle the general mechanics. You may need to provide additional information to make it work for your specific use case:

- **Have an anchor case ready**, like one concrete input and the output you expect for it. The skills can verify against a response body.
- **Know where your environment variables come from**, such as ``HF_TOKEN`` for a gated model. The agent puts names into the deployment and you need to supply the values.
- **Write down the project facts the code does not show** in a ``CLAUDE.md`` or ``AGENTS.md`` file, such as which cluster is which, which registry to use, and who owns the namespace.
- **Say the target in your first message**, such as ``deploy this to my EKS cluster in us-west-2``. The skills ask fewer questions when the target is unambiguous.
- **When something breaks, say what you see.** The skills carry troubleshooting references indexed by symptom: ``ImagePullBackOff``, ``CrashLoopBackOff``, ``OOMKilled``, ``Pending``, probe failures, unreachable Services, and inference errors.

Deployment best practices
-------------------------

The skills carry the deployment rules with them. The decisions that stay yours:

- **Cost on EC2 is yours to stop.** For example, instances bill by the hour until you terminate them, whether or not they serve traffic. Ask the skill to tear down what you no longer need.
- **Decide how the endpoint is exposed.** If it must be reachable from outside your network, put an authenticating load balancer, API gateway, or ingress in front of it rather than opening port 3000 to the world.
- **Scale after it works.** Start at one replica and add autoscaling once the Service is verified against a real request.
- **Build your CI/CD workflow with the script bundle generated by bentoml-deploy-scriptgen.** Commit and wire your scripts into CI so every deployment is reproducible and agent-free. The generated bundle ships its own README with a GitHub Actions workflow and a GitLab CI equivalent.
