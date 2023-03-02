===============
Deploying Bento
===============


Deployment Overview
-------------------

The three most common deployment options with BentoML are:

- 🐳 Generate container images from Bento for custom docker deployment
- `🦄️ Yatai <https://github.com/bentoml/Yatai>`_: Model Deployment at scale on Kubernetes
- `🚀 bentoctl <https://github.com/bentoml/bentoctl>`_: Fast model deployment on any cloud platform


Containerize Bentos
-------------------

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


Deploy with Yatai
-----------------

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

.. tip::
    Yatai will automatically start building container images for a new Bento pushed.


Deploy via Web UI
^^^^^^^^^^^^^^^^^

Although not always recommended for production workloads, Yatai offers an easy-to-use
web UI for quickly creating deployments. This is convenient for data scientists to test
out Bento deployments end-to-end from a development or testing environment:

.. image:: /_static/img/yatai-deployment-creation.png
    :alt: Yatai Deployment creation UI

The web UI is also very helpful for viewing system status, monitoring services, and
debugging issues.

.. image:: /_static/img/yatai-deployment-details.png
    :alt: Yatai Deployment Details UI

Commonly we recommend using APIs or Kubernetes CRD objects to automate the deployment
pipeline for production workloads.

Deploy via API
^^^^^^^^^^^^^^

Yatai's REST API specification can be found under the :code:`/swagger` endpoint. If you
have Yatai deployed locally with minikube, visit:
http://yatai.127.0.0.1.sslip.io/swagger/. The Swagger API spec covers all core Yatai
functionalities ranging from model/bento management, cluster management to deployment
automation.

.. note::

    Python APIs for creating deployment on Yatai is on our roadmap. See :issue:`2405`.
    Current proposal looks like this:

    .. code-block:: python

        yatai_client = bentoml.YataiClient.from_env()

        bento = yatai_client.get_bento('my_svc:v1')
        assert bento and bento.status.is_ready()

        yatai_client.create_deployment('my_deployment', bento.tag, ...)

        # For updating a deployment:
        yatai_client.update_deployment('my_deployment', bento.tag)

        # check deployment_info.status
        deployment_info = yatai_client.get_deployment('my_deployment')


Deploy via kubectl and CRD
^^^^^^^^^^^^^^^^^^^^^^^^^^

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



Deploy with bentoctl
--------------------

:code:`bentoctl` is a CLI tool for deploying Bentos to run on any cloud platform. It
supports all major cloud providers, including AWS, Azure, Google Cloud, and many more.

Underneath, :code:`bentoctl` is powered by Terraform. :code:`bentoctl` adds required
modifications to Bento or service configurations, and then generate terraform templates
for the target deploy platform for easy deployment.

The :code:`bentoctl` deployment workflow is optimized for CI/CD and GitOps. It is highly
customizable, users can fine-tune all configurations provided by the cloud platform. It
is also extensible, for users to define additional terraform templates to be attached
to a deployment.

Quick Tour
^^^^^^^^^^

Install aws-lambda plugin for :code:`bentoctl` as an example:

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

.. TODO::
    Explain limitations of each platform, e.g. GPU support
    Explain how to customize the terraform workflow


About Horizontal Auto-scaling
-----------------------------

Auto-scaling is one of the most sought-after features when it comes to deploying models. Autoscaling helps optimize resource usage and cost by automatically provisioning up and scaling down depending on incoming traffic.

Among deployment options introduced in this guide, Yatai on Kubernetes is the
recommended approach if auto-scaling and resource efficiency are required for your team’s workflow.
Yatai enables users to fine-tune resource requirements and
auto-scaling policy at the Runner level, which inherently improves interoperability between auto-scaling and data aggregated at Runner's adaptive batching layer in real-time.

Many of bentoctl’s deployment targets also come with a certain level of auto-scaling
capabilities, including AWS EC2 and AWS Lambda.
