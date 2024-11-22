====================
Batch inference jobs
====================

Some AI-powered tasks are best suited for batch inference, such as embedding generation for RAG systems, periodic updates to recommendation systems, or bulk image processing for feature extraction.

Using BentoML and BentoCloud, you can efficiently manage these batch inference jobs with several key advantages:

- **On-demand Deployment**: Deploy your model only when needed and terminate the Deployment after the job completes, ensuring you pay only for the resources you use. You can run batch inference jobs once or on a recurring basis.
- **Automatic scaling**: Scale your resources automatically based on the traffic demands for your job.
- **Dedicated hardware for inference**: Run model inference on dedicated GPUs, ensuring that the inference tasks do not interfere with batch processing.

This document explains how to run batch inference jobs with BentoML and BentoCloud.

Create jobs
-----------

The following example demonstrates the full lifecycle of job execution.

Step 1: Prepare a BentoML project
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Make sure you have an existing BentoML project or a Bento. The example below is a typical BentoML Service setup for a RAG system, where endpoints ``ingest_pdf_batch`` and ``ingest_text_batch`` are used for batch ingestion of files. They can compute embeddings for documents and write them to a vector database for indexing. Unlike regular Services that might require constant availability, these endpoints can be activated on-demand, making them ideal for batch inference jobs, as resources are only consumed during active job execution.

.. code-block:: python

    ...
    @bentoml.service(
        resources={
            "gpu": 1,
        },
        traffic={
            "timeout": 30,
            "concurrency": 5,
            "external_queue": True,
        }
    )
    class RAGService:
        # Initialization setup
        ...

        @bentoml.api
        def ingest_pdf_batch(self, pdf: Annotated[Path, bentoml.validators.ContentType("application/pdf")]) -> str:

            import pypdf
            reader = pypdf.PdfReader(pdf)
            texts = []
            for page in reader.pages:
                text = page.extract_text()
                texts.append(text)
            all_text = "".join(texts)
            doc = Document(text=all_text)
            # Insert document into vector index and persist to storage
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    [doc], storage_context=self.storage_context
                )
            else:
                self.index.insert(doc)

            self.index.storage_context.persist()
            return "Successfully Loaded Document"


        @bentoml.api
        def ingest_text_batch(self, txt: Annotated[Path, bentoml.validators.ContentType("text/plain")]) -> str:

            with open(txt) as f:
                text = f.read()

            doc = Document(text=text)

            # Insert document into vector index and persist to storage
            if self.index is None:
                self.index = VectorStoreIndex.from_documents(
                    [doc], storage_context=self.storage_context
                )
            else:
                self.index.insert(doc)

            self.index.storage_context.persist()
            return "Successfully Loaded Document"

        @bentoml.api
        def query(self, query: str) -> str:
            # Implementation code for query handling
            ...

You can find the full example code in the `rag-tutorials <https://github.com/bentoml/rag-tutorials>`_ repository.

Step 2: Create a Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To deploy this BentoML project as a batch job, create a script to start the Deployment with specific :doc:`configurations </scale-with-bentocloud/deployment/configure-deployments>`.

.. code-block:: python

    import bentoml

    # Define the path to your BentoML project or the Bento package
    BENTO_PATH = "./path_to_your_project"
    DEPLOYMENT_NAME = "my_batch_job"

    # Create a Deployment
    deployment = bentoml.deployment.create(
        bento=BENTO_PATH,
        name=DEPLOYMENT_NAME,
        scaling_min=1,
        scaling_max=3
    )

    # Optionally, wait for the Deployment to become ready
    deployment.wait_until_ready(timeout=3600)

Step 3: Run inference against the Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once your Deployment is active, you can interact with it by creating a client that calls its endpoints. Below is a script that uses the client to perform a file ingestion task.

.. code-block:: python

    import bentoml
    from pathlib import Path

    deployment = bentoml.deployment.get(name=DEPLOYMENT_NAME)

    # Get synchronous HTTP client for the Deployment
    client = deployment.get_client()
    # Call the available endpoints to ingest files
    result = client.ingest_text_batch(txt=Path("file_to_ingest.txt"))

Step 4: Clean up
^^^^^^^^^^^^^^^^

After completing the job, it's important to terminate the Deployment to conserve resources.

.. code-block:: python

    import bentoml

    # Clean-up: terminate the Deployment after job completion
    bentoml.deployment.terminate(name=DEPLOYMENT_NAME)

    # Optionally check and print the final status
    final_status = bentoml.deployment.get(name=DEPLOYMENT_NAME).get_status()
    print("Final status:", final_status.to_dict())

Schedule jobs
-------------

To automate and schedule your batch inference tasks, you can utilize a variety of job scheduling tools that best fit your operational environment and requirements. Here are some commonly used schedulers:

- `Cron <https://man7.org/linux/man-pages/man5/crontab.5.html>`_
- `Apache Airflow <https://airflow.apache.org/>`_
- `Kubernetes CronJobs <https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/>`_
