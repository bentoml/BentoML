===============
Deploying Bento
===============


Deployment Overview
-------------------

Here are the four most common deployment options with BentoML:

.. list-table::
   :header-rows: 1

   * - Deploy Method 
     - Pros ❤️ 
     - Cons 💔 
   * - `🍱 BentoCloud <https://www.bentoml.com/>`_
     - Serverless cloud for AI, fully managed, made for BentoML
     - Invite only today, `sign up here <https://www.bentoml.com/bento-cloud/>`_ for early access
   * - Docker Containers 
     - Flexible for deploying almost anywhere, easy to get started 
     - Rigid deployment as a single container. Inflexible for scaling and maximizing hardware utilization.
   * - `Yatai on Kubernetes <https://github.com/bentoml/Yatai>`_
     - Cloud-native AI deployment on Kubernetes, comes with advanced auto-scaling and CI/CD workflows
     - Requires professional DevOps team to maintain and operate
   * - Cloud Deployment with `BentoCTL <https://github.com/bentoml/bentoctl>`_ 
     - Great for proof-of-concept deployments directly running on public cloud services (EC2, ECS, SageMaker, Lambda, GCP, etc) 
     - Requires working knowledge of Cloud Services and their limitations for AI-specific workloads



Feature comparison across deployment options:

.. list-table::
   :header-rows: 1

   * - Feature
     - `🍱 BentoCloud <https://www.bentoml.com/>`_
     - Docker Containers 
     - `Yatai on Kubernetes <https://github.com/bentoml/Yatai>`_
     - Cloud Deployment with `BentoCTL <https://github.com/bentoml/bentoctl>`_ 
   * - Auto-scaling
     - ✅ Fast auto-scaling optimized for AI
     - 💦 Possible via container orchastration tools
     - ✅ Kubernetes-native with custom metrics
     - ❌ Depending on the Cloud Service, supported on ECS
   * - Scaling-to-zero
     - ✅ Scaling at individual Model/Runner level
     - 💦 Possible via Knative, no optimization for AI or GPU workload
     - ❓ Not currently supported
     - 💦 Supported on AWS Lambda, GCP Functions with limitations
   * - GPU Support
     - ✅
     - ✅
     - ✅
     - 💦 Supported on EC2, AWS SageMaker, requires manual configurations
   * - Monitoring
     - ✅ Auto-generated dashboards for key metrics
     - 💦 Requires setting up Prometheus scraping job
     - 💦 Requires setting up Prometheus scraping job
     - ❌ Requires manual configurations with cloud provider
   * - Logging
     - ✅ Turnkey solution for log collection and delivery
     - 💦 Manual setup
     - 💦 Manual setup
     - 💦 Manual setup
   * - Tracing
     - ✅ Turnkey solution for OpenTelemetry and 3rd party integration
     - 💦 Manual setup
     - 💦 Manual setup
     - 💦 Manual setup


Docker Containers
-----------------

Containerizing bentos as Docker images allows users to easily distribute and deploy
bentos. Once services are built as bentos and saved to the bento store, we can
containerize saved bentos with the CLI command :ref:`bentoml containerize <reference/cli:containerize>`.

Start the Docker engine. Verify using ``docker info``.

.. code-block:: bash

    $ docker info

Run ``bentoml list`` to view available bentos in the store.

.. code-block:: bash

    $ bentoml list

    Tag                               Size        Creation Time        Path
    iris_classifier:ejwnswg5kw6qnuqj  803.01 KiB  2022-05-27 00:37:08  ~/bentoml/bentos/iris_classifier/ejwnswg5kw6qnuqj
    iris_classifier:h4g6jmw5kc4ixuqj  644.45 KiB  2022-05-27 00:02:08  ~/bentoml/bentos/iris_classifier/h4g6jmw5kc4ixuqj


Run ``bentoml containerize`` to start the containerization process.

.. code-block:: bash

    $ bentoml containerize iris_classifier:latest

    INFO [cli] Building docker image for Bento(tag="iris_classifier:ejwnswg5kw6qnuqj")...
    [+] Building 21.2s (20/20) FINISHED
    ...
    INFO [cli] Successfully built docker image "iris_classifier:ejwnswg5kw6qnuqj"


.. dropdown:: For Mac with Apple Silicon
   :icon: cpu

   Specify the :code:`--platform` to avoid potential compatibility issues with some
   Python libraries.

   .. code-block:: bash

      $ bentoml containerize --opt platform=linux/amd64 iris_classifier:latest


View the built Docker image:

.. code-block:: bash

    $ docker images

    REPOSITORY          TAG                 IMAGE ID       CREATED         SIZE
    iris_classifier     ejwnswg5kw6qnuqj    669e3ce35013   1 minutes ago   1.12GB

Run the generated docker image:

.. code-block:: bash

    $ docker run -p 3000:3000 iris_classifier:ejwnswg5kw6qnuqj serve --production

.. seealso::

   :ref:`guides/containerization:Containerization with different container engines.`
   goes into more details on our containerization process and how to use different container runtime.

.. todo::

    - Add sample code for working with GPU and --gpu flag


Deploy with Yatai on Kubernetes
-------------------------------

Yatai helps ML teams to deploy large scale model serving workloads on Kubernetes. It
standardizes BentoML deployment on Kubernetes, provides UI and APis for managing all
your ML models and deployments in one place, and enables advanced GitOps and CI/CD
workflows.

Yatai is Kubernetes native, integrates well with other cloud native tools in the K8s
eco-system.

To get started, get an API token from Yatai Web UI and login from your :code:`bentoml`
CLI command:

.. code-block:: bash

    bentoml yatai login --api-token {YOUR_TOKEN_GOES_HERE} --endpoint http://yatai.127.0.0.1.sslip.io

Push your local Bentos to yatai:

.. code-block:: python

    bentoml push iris_classifier:latest


Yatai is designed to be a cloud-native tool, providing
For DevOps managing production model serving workloads along with other kubernetes
resources, the best option is to use :code:`kubectl` and directly create
:code:`BentoDeployment` objects in the cluster, which will be handled by the Yatai
deployment CRD controller.

.. code-block:: yaml

    # my_deployment.yaml
    apiVersion: serving.yatai.ai/v1alpha2
    kind: BentoDeployment
    metadata:
      name: demo
    spec:
      bento_tag: iris_classifier:3oevmqfvnkvwvuqj
      resources:
        limits:
          cpu: 1000m
        requests:
          cpu: 500m

.. code-block:: bash

    kubectl apply -f my_deployment.yaml



Deploy with BentoControl
------------------------

:code:`bentoctl` is a CLI tool for deploying Bentos to run on any cloud platform. It
supports all major cloud providers, including AWS, Azure, Google Cloud, and many more.

Underneath, :code:`bentoctl` is powered by Terraform. :code:`bentoctl` adds required
modifications to Bento or service configurations, and then generate terraform templates
for the target deploy platform for easy deployment.

The :code:`bentoctl` deployment workflow is optimized for CI/CD and GitOps. It is highly
customizable, users can fine-tune all configurations provided by the cloud platform. It
is also extensible, for users to define additional terraform templates to be attached
to a deployment.

Here's an example of using :code:`bentoctl` for deploying to AWS Lambda. First, install
the `aws-lambda` operator plugin:

.. code-block:: bash

    bentoctl operator install aws-lambda

Initialize a bentoctl project. This enters an interactive mode asking users for related
deployment configurations:

.. code-block:: bash

    $ bentoctl init

    Bentoctl Interactive Deployment Config Builder
    ...

    deployment config generated to: deployment_config.yaml
    ✨ generated template files.
      - bentoctl.tfvars
      - main.tf


Deployment config will be saved to :code:`./deployment_config.yaml`:

.. code-block:: yaml

    api_version: v1
    name: quickstart
    operator:
        name: aws-lambda
    template: terraform
    spec:
        region: us-west-1
        timeout: 10
        memory_size: 512

Now, we are ready to build the deployable artifacts required for this deployment. In
most cases, this step will product a new docker image specific to the target deployment
configuration:


.. code-block:: bash

    bentoctl build -b iris_classifier:btzv5wfv665trhcu -f ./deployment_config.yaml

Next step, use :code:`terraform` CLI command to apply the generated deployment configs
to AWS. This will require user setting up AWS credentials on the environment.


.. code-block:: bash

    $ terraform init
    $ terraform apply -var-file=bentoctl.tfvars --auto-approve

    ...
    base_url = "https://ka8h2p2yfh.execute-api.us-west-1.amazonaws.com/"
    function_name = "quickstart-function"
    image_tag = "192023623294.dkr.ecr.us-west-1.amazonaws.com/quickstart:btzv5wfv665trhcu"


Testing the endpoint deployed:

.. code-block:: bash

    URL=$(terraform output -json | jq -r .base_url.value)classify
    curl -i \
        --header "Content-Type: application/json" \
        --request POST \
        --data '[5.1, 3.5, 1.4, 0.2]' \
        $URL


Supported Cloud Platforms
^^^^^^^^^^^^^^^^^^^^^^^^^

- AWS Lambda: https://github.com/bentoml/aws-lambda-deploy
- AWS SageMaker: https://github.com/bentoml/aws-sagemaker-deploy
- AWS EC2: https://github.com/bentoml/aws-ec2-deploy
- Google Cloud Run: https://github.com/bentoml/google-cloud-run-deploy
- Google Compute Engine: https://github.com/bentoml/google-compute-engine-deploy
- Azure Functions: https://github.com/bentoml/azure-functions-deploy
- Azure Container Instances: https://github.com/bentoml/azure-container-instances-deploy
- Heroku: https://github.com/bentoml/heroku-deploy
