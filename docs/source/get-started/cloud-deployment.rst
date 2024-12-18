================
Cloud deployment
================

BentoCloud is an Inference Management Platform and Compute Orchestration Engine built on top of BentoML's open-source serving framework. It provides a complete stack for building fast and scalable AI systems with any model, on any cloud.

Why developers love BentoCloud:

- **Flexible Pythonic APIs** for building inference APIs, batch jobs, and compound AI systems
- **Blazing fast cold start** with a container infrastructure stack rebuilt for ML/AI workloads
- Support for **any ML frameworks and inference runtimes** (vLLM, TensorRT, Triton, etc.)
- **Streamlined workflows** across development, testing, deployment, monitoring, and CI/CD
- Easy access to various GPUs like L4 and A100, in **our cloud or yours**

Log in to BentoCloud
--------------------

1. Visit the `BentoML website <https://www.bentoml.com/>`_ to sign up.
2. Install BentoML.

   .. code-block:: bash

      pip install bentoml

3. Log in to BentoCloud with the ``bentoml cloud login`` command. Follow the on-screen instructions to :ref:`create a new API token <creating-an-api-token>`.

   .. code-block:: bash

      $ bentoml cloud login

      ? How would you like to authenticate BentoML CLI? [Use arrows to move]
      > Create a new API token with a web browser
        Paste an existing API token

Deploy your first model
-----------------------

1. Clone the :doc:`hello-world` example.

   .. code-block:: bash

      git clone https://github.com/bentoml/quickstart.git
      cd quickstart

2. Deploy it to BentoCloud from the project directory. Optionally, use the ``-n`` flag to set a name.

   .. code-block:: bash

      bentoml deploy . -n my-first-bento

   Sample output:

   .. code-block:: bash

      🍱 Built bento summarization:ngfnciv5g6nxonry
      Successfully pushed Bento "summarization:ngfnciv5g6nxonry"
      ✅ Created deployment "my-first-bento" in cluster "google-cloud-us-central-1"
      💻 View Dashboard: https://demo.cloud.bentoml.com/deployments/my-first-bento

   The first Deployment might take a minute or two. Wait until it's fully ready:

   .. code-block:: bash

      ✅ Deployment "my-first-bento" is ready: https://demo.cloud.bentoml.com/deployments/my-first-bento

3. On the BentoCloud console, navigate to the **Deployments** page, and click your Deployment. Once it's up and running, you can interact with it using the **Form** section on the **Playground** tab.

   .. image:: ../_static/img/get-started/cloud-deployment/first-bento-on-bentocloud.png
      :alt: A summarization model running on BentoCloud

Call the Deployment endpoint
----------------------------

1. Retrieve the Deployment URL via CLI. Replace ``my-first-bento`` if you use another name.

   .. code-block:: bash

      bentoml deployment get my-first-bento -o json | jq ."endpoint_urls"

   .. note::

      Ensure ``jq`` is installed for processing JSON output.

2. Create :doc:`a BentoML client </build-with-bentoml/clients>` to call the exposed endpoint. Replace the example URL with your Deployment's URL:

   .. code-block:: python

      import bentoml

      client = bentoml.SyncHTTPClient("https://my-first-bento-e3c1c7db.mt-guc1.bentoml.ai")
      result: str = client.summarize(
            text="Breaking News: In an astonishing turn of events, the small town of Willow Creek has been taken by storm as local resident Jerry Thompson's cat, Whiskers, performed what witnesses are calling a 'miraculous and gravity-defying leap.' Eyewitnesses report that Whiskers, an otherwise unremarkable tabby cat, jumped a record-breaking 20 feet into the air to catch a fly. The event, which took place in Thompson's backyard, is now being investigated by scientists for potential breaches in the laws of physics. Local authorities are considering a town festival to celebrate what is being hailed as 'The Leap of the Century.",
         )
      print(result)

Configure scaling
-----------------

The replica count defaults to ``1``. You can update the minimum and maximum replicas allowed for scaling:

.. code-block:: bash

   bentoml deployment update my-first-bento --scaling-min 0 --scaling-max 3

Cleanup
-------

To terminate this Deployment, click **Stop** in the top right corner of its details page or simply run:

.. code-block:: bash

   bentoml deployment terminate my-first-bento

More resources
--------------

If you are a first-time user of BentoCloud, we recommend you read the following documents to get started:

- Deploy :doc:`example projects </examples/overview>` to BentoCloud
- :doc:`/scale-with-bentocloud/deployment/manage-deployments`
- :doc:`/scale-with-bentocloud/deployment/create-deployments`
- :doc:`/scale-with-bentocloud/manage-api-tokens`
