============
Agent skills
============

.. meta::
    :description lang=en:
        Use BentoML's agent skills to let a coding agent build, containerize, and deploy your BentoML Service to Kubernetes or EC2.

An **agent skill** is a set of instructions that a coding agent loads on demand when your request matches what the skill covers. The format is defined by the open `Agent Skills specification <https://agentskills.io/specification>`_ and works across hosts such as Claude Code, OpenAI Codex, and Cursor.

BentoML provides several skills in the `skills/ <https://github.com/bentoml/BentoML/tree/main/skills>`_ directory of the repository. They teach your agent to create a BentoML project, build a Bento, and then take it all the way to a verified, running deployment on infrastructure you own, such as a Kubernetes cluster or plain AWS EC2 instances. They use only open-source tools, including the ``bentoml`` CLI, Docker, ``kubectl`` with plain manifests, ``ssh``, and the AWS CLI.

.. note::

   The skills cover **basic deployment on self-managed infrastructure**, not on BentoCloud.

How BentoML works
-----------------

Learn about the general workflow before diving into the skills. This is the same pipeline you follow by hand, but with the skills, your agent does it for you.

.. code-block:: bash

    model or existing code                ->  a script, a notebook, a FastAPI app, etc.
              |
              v
    service.py + runtime environment      ->  your Python project
              |  bentoml build
              v
    Bento (~/bentoml/bentos/<name>/<ver>) ->  standardized, versioned package
              |  bentoml containerize
              v
    OCI image (port 3000)                 ->  runs anywhere Docker/OCI runs
              |  push
              v
    Registry  ->  Kubernetes | EC2 | any Docker host

1. A Python project
^^^^^^^^^^^^^^^^^^^

Everything starts with a ``service.py`` file. A class decorated with ``@bentoml.service`` is a **Service**. Each method decorated with ``@bentoml.api`` becomes an HTTP endpoint whose request and response schemas are inferred from your type annotations.

.. code-block:: python
    :caption: `service.py`

    import bentoml

    @bentoml.service
    class Summarization:
        def __init__(self) -> None:
            from transformers import pipeline
            self.pipeline = pipeline("summarization")

        @bentoml.api
        def summarize(self, text: str) -> str:
            return self.pipeline(text)[0]["summary_text"]

A project can contain several Services wired together with ``bentoml.depends()``. The one you build is the **entry Service**; it receives external traffic and calls the others.

For more information, see :doc:`/build-with-bentoml/services`, :doc:`/build-with-bentoml/iotypes`, and :doc:`/build-with-bentoml/distributed-services`.

2. The runtime environment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Declare the dependencies for your project.

.. tab-set::

    .. tab-item:: Python SDK (recommended)

        .. code-block:: python
            :caption: `service.py`

            import bentoml

            my_image = bentoml.images.Image(python_version="3.11") \
                .python_packages("torch", "transformers")

            @bentoml.service(
                image=my_image,
                envs=[{"name": "HF_TOKEN"}],
            )
            class Summarization:
                ...

    .. tab-item:: bentofile.yaml

        .. code-block:: yaml
            :caption: `bentofile.yaml`

            service: "service:Summarization"
            include:
              - "*.py"
            python:
              packages:
                - torch
                - transformers
              # or: requirements_txt: "./requirements.txt"

    .. tab-item:: pyproject.toml

        .. code-block:: toml
            :caption: `pyproject.toml`

            [tool.bentoml.build]
            service = "service:Summarization"

            [tool.bentoml.build.python]
            requirements_txt = "./requirements.txt"

For more information, see :doc:`/build-with-bentoml/runtime-environment` and :doc:`/reference/bentoml/bento-build-options`.

3. Build a Bento
^^^^^^^^^^^^^^^^

.. code-block:: bash

    bentoml build

A **Bento** is the standardized, versioned package of everything needed to serve your Service: source code, the runtime environment spec, model references, and metadata. Where things live locally:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Path
     - Contents
   * - ``~/bentoml/bentos/<name>/<version>/``
     - Built Bentos. ``bentoml list`` shows them; ``bentoml get <tag> -o json`` prints one.
   * - ``~/bentoml/models/<name>/<version>/``
     - Models saved to the local Model Store. ``bentoml models list`` shows them.
   * - ``~/bentoml/`` (``BENTOML_HOME``)
     - Root of both stores. Set ``BENTOML_HOME`` to relocate them, which is useful in CI, where a runner's home directory is ephemeral.

Inside each Bento is a ``bento.yaml`` recording the **topology**: every Service, which one is the entry Service, and the dependency graph between them.

Use a ``.bentoignore`` file to keep datasets, notebooks, and virtual environments out of the build context. For more information, see :doc:`packaging-for-deployment` and :doc:`/build-with-bentoml/model-loading-and-management`.

4. Containerize
^^^^^^^^^^^^^^^

.. code-block:: bash

    bentoml containerize summarization:latest

Convert your Bento into a plain OCI image. The skills use the following by default:

- Port **3000** for HTTP serving.
- ``/livez`` for liveness, ``/readyz`` for readiness, ``/metrics`` for Prometheus, ``/`` for the Swagger UI.

Models are usually baked into the image. Size your disks and volumes accordingly.

5. Deploy anywhere
^^^^^^^^^^^^^^^^^^

Once the image is in a registry your target can pull from, the deployment is ordinary container operations:

- **Any Docker host**: ``docker run -p 3000:3000 <image>``.
- **Kubernetes**: one Deployment + Service per BentoML Service, plus optional HPA and Ingress.
- **EC2**: Docker on a VM, with ``--restart unless-stopped``.

The skills automate these steps and verify the result with a real inference request.

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

The skills
----------

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

An example session
------------------

Here is an example session that converts an existing script into a Bento:

.. code-block:: bash

    $ cd ~/projects/summarization && claude
    > turn predict.py into a BentoML service

    Loading skill: bentoml-create-bento
      Read predict.py: model loaded at module level, summarize(text) -> str,
      weights from sshleifer/distilbart-cnn-12-6
      Before I write anything: one input and the output you expect for it?
    > "the quick brown fox..." -> a one-sentence summary containing "fox"
      Wrote service.py, requirements.txt, .bentoignore, test_service.py
      bentoml serve -> /readyz OK, /docs.json shows summarize(text: str) -> str
      Anchor request -> "A fox jumped over the dog." Matches, and identical to
      predict.py on the same input
      pytest -> 1 passed
      bentoml build -> summarization:lyt7deofjkmparqe
      Next step: /bentoml-containerize

And here is one that deploys the result. Nothing is applied to a cluster or an AWS account before you confirm it.

.. code-block:: bash

    $ cd ~/projects/summarization && claude
    > containerize my BentoML service and deploy it to my kind cluster

    Loading skill: bentoml-containerize
      Preflight: bentoml 1.4.x OK - docker daemon OK (x86_64) - found ./service.py
      1. Which registry?  2. Target CPU architecture?
    > kind load, amd64
      bentoml build -> summarization:lyt7deofjkmparqe -> bentoml containerize
      Smoke test: GET /readyz -> 200, POST /summarize -> plausible output
      kind load docker-image summarization:lyt7deofjkmparqe --name kind

    Loading skill: bentoml-k8s-deploy
      Contexts: kind-kind (current), prod-eks - which one should I deploy to?
    > kind-kind
      bento.yaml: 1 service (Summarization), entry Summarization, no dependencies
      Parameters: namespace default - replicas 1 - cpu 500m/2 - memory 1Gi/4Gi
      Wrote deploy/config.yml. Review without touching the cluster:
        python3 deploy/deploy.py --target k8s --render-only /tmp/look
      About to apply: context=kind-kind namespace=default
    > yes
      rollout status: deployment "summarization" successfully rolled out
      POST /summarize through a port-forward -> correct output

Onboarding best practices
-------------------------

To prepare the project for your agent:

- Keep ``service.py`` at the project root, or somewhere obvious, with one clear entry Service. If you don't have one yet, ``bentoml-create-bento`` writes it for you.
- Declare the runtime environment in code, and keep it next to the Service it describes.
- Add a ``.bentoignore`` file so builds do not sweep in datasets, checkpoints, and ``.venv``.
- Write down what the Service expects. One concrete input and the output you expect for it - an anchor case - is the single most useful thing you can hand the agent, because every skill verifies against a response body rather than a status code.
- Note any required environment variables (such as ``HF_TOKEN`` for gated models) and where their values come from. Names go into the deployment; values never do.
- Add a ``CLAUDE.md`` or ``AGENTS.md`` file recording project-specific facts the code does not show: which cluster is which, which registry to use, who owns the namespace.

When creating or converting a Service:

- **Settle the anchor case before any code is written.** ``bentoml-create-bento`` asks for it up front and refuses to declare success without it. For a conversion, it also runs your original code on the same input and diffs the two outputs.
- **Convert one real Service first.** Pick something small that already works, run ``bentoml-create-bento`` on it, and compare the Bento's output against the original code before going further.
- **Keep model references at class scope**, not inside ``__init__``. Class scope is what declares a model a dependency of the Bento. Declared inside ``__init__``, it isn't packaged, and the deployment fails with a model ``NotFound`` error even though it worked locally.
- **Annotate every parameter and return value.** The annotations are the schema, the OpenAPI spec, and the client. An unannotated parameter becomes ``Any``, with no validation and no docs.

To adopt the skills across a team:

1. **Start local and free.** A first deployment to ``kind`` or ``minikube`` exercises the whole pipeline with no cloud spend.
2. **Commit the skills into the repository** (``.claude/skills/`` or ``.codex/skills/``) so everyone runs the same version, and update them deliberately rather than per person.
3. **Commit deploy/config.yml** and treat it as the single reviewable artifact. Edit it per your own needs and use it for your custom workflow.
4. **Review before applying.** ``--render-only`` prints exactly what would be applied. Make that the habit, especially for the first deployment to a shared cluster.
5. **Give each Service its own namespace or cluster per environment**, and let the agent ask which context to use every time. It never assumes your current context.

When working alongside the agent:

- Say where you want to deploy in your first message ("deploy this to my EKS cluster in ``us-west-2``"). The skills ask fewer questions when the target is unambiguous.
- Answer the batched questions in one go. They are asked up front and have defaults you can accept.
- When something breaks, say what you see. The skills carry troubleshooting references indexed by symptom: ``ImagePullBackOff``, ``CrashLoopBackOff``, ``OOMKilled``, ``Pending``, probe failures, unreachable Services, and inference errors for Kubernetes.

Deployment best practices
-------------------------

General deployment guidelines:

- **Declare every dependency** in the runtime environment, not just in your local virtual environment.
- **Never put secrets in an image or a manifest.** Pass them as environment variables at run time - a Kubernetes Secret referenced by name, or an ``-e`` flag expanded from your shell on EC2.
- **Verify with a real inference request**, not a health check. A Pod can be ``Ready`` and still return errors because the wrong model was baked in. The skills judge the response body.
- **Pin the tag.** The image tag is the Bento version; ``latest`` in a manifest makes rollbacks guesswork.
- **Test the container locally first**: ``docker run -p 3000:3000 <image>``, then open the Swagger UI at ``http://localhost:3000``.

To deploy to Kubernetes:

- **Set resources in the deployment configuration, not in code.** The ``resources`` argument of ``@bentoml.service`` is honored by BentoCloud but is inert in open-source BentoML. The manifest's requests and limits are the only thing constraining a Pod.
- **Size memory to the model.** ``OOMKilled`` often means the limit is below what the model needs.
- **Give model loading a long startup probe.** Liveness on ``/livez``, readiness and startup on ``/readyz``, with a startup budget generous enough to load weights. Otherwise Kubernetes may keep restarting the Pod mid-load.
- **Confirm the cluster can pull the image** before deploying. Every ECR registry is private: you need an image pull secret in the namespace, on the namespace's default service account, or node-level credentials.
- **Keep the configuration as the source of truth.** Manifests are rendered from ``deploy/config.yml`` on every run. Use ``--render-only`` to review or to commit into a GitOps repository.
- **Scale after it works.** Start at one replica. Add ``autoscaling`` (a stock CPU-based HPA) once the Service is verified.

To deploy to EC2 instances:

- **Instances bill by the hour until you terminate them**, whether or not they serve traffic. A stopped instance still bills its EBS volume, and every public IPv4 address bills separately. Tear down what you no longer need.
- **Size the root volume to the image.** Bentos bake models in. The AMI default can be too small to hold them.
- **Do not open port 3000 to the world.** If the endpoint must be public, put an authenticating load balancer or API gateway in front.
- **Prefer an instance profile over copied credentials** for ECR authentication. Registry tokens expire after 12 hours; an instance profile does not.
- **Restart on reboot**. Run the container with ``--restart unless-stopped``.
- **Match the instance family to the image architecture** - ``t3``/``m5`` for ``amd64``, ``t4g``/``m7g`` for ``arm64``.

To build your CI/CD workflow:

Do the **first** deployment interactively. It needs judgment, and the agent asks the questions you would otherwise forget. Then generate the script bundle with ``bentoml-deploy-scriptgen``, commit it, and wire it into CI so every deployment after that is reproducible and agent-free:

- The bundle repeats the exact build → containerize → push → deploy → verify pipeline with no questions asked.
- Preflight checks fail fast, exit codes are a stable contract, and the last line of standard output is a JSON summary you can assert on.
- Use ``--check-only --local-only`` as a pull request gate: it validates the configuration and builds without touching your cluster or cloud account.
- Authenticate CI with short-lived credentials (AWS OIDC for ECR and EKS) rather than long-lived keys.

The generated bundle ships its own README with a GitHub Actions workflow and a GitLab CI equivalent.
