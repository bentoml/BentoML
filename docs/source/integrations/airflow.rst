=======
Airflow
=======

Apache Airflow is a platform to programmatically author, schedule and monitor workflows.
It is a commonly used framework for building model training pipelines in ML projects.
BentoML provides a flexible set of APIs for integrating natively with Apache Airflow.
Users can use Airflow to schedule their model training pipelines and use BentoML to keep
tracked of trained model artifacts and optionally deploy them to production in an
automated fashion.

This is especially userful for teams that can benefit from retraining models often with
newly arrived data, and want to update their production models regularly with
confidence.

For more in-depth Airflow tutorials, please visit `the Airflow documentation <https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html>`_.


Overview
--------

A typical Airflow pipeline with a BentoML serving & deployment workflow look like this:

1. Fetch new data batches from a data source
2. Split the data in train and test sets
3. Perform feature extraction on training data set
4. Train a new model using the training data set
5. Perform model evaluation and validation
6. :doc:`Save model with BentoML </concepts/model>`
7. :ref:`Push saved model to Yatai registry (or export model to s3) <concepts/model:Managing Models>`
8. :doc:`Build a new Bento using the newly trained model </concepts/bento>`
9. Run integration test on the Bento to verify the entire serving pipeline
10. :ref:`Push the Bento to a Yatai (or export bento to s3) <concepts/bento:Managing Bentos>`
11. (Optional) Trigger a redeployment via Yatai, bentoctl, or custom deploy script


Pro Tips
--------

Pipeline Dependencies
~~~~~~~~~~~~~~~~~~~~~

The default PythonOperator requires all the dependencies to be installed on the Airflow
environment. This can be challenging to manage when the pipeline is running on a remote
Airflow deployment and running a mix of different tasks.

To avoid this, we recommend managing dependencies of your ML pipeline with the
`PythonVirtualenvOperator <https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/python.html#pythonvirtualenvoperator>`_,
which runs your code in a virtual environment. This allows you to define your Bento's
dependencies in a ``requirements.txt`` file and use it across training pipeline and the
bento build process. For example:

.. code-block:: python

    from datetime import datetime, timedelta
    from airflow import DAG
    from airflow.decorators import task

    with DAG(
        dag_id='example_bento_build_operator',
        description='A simple tutorial DAG with BentoML',
        schedule_interval=timedelta(days=1),
        start_date=datetime(2021, 1, 1),
        catchup=False,
        tags=['example'],
    ) as dag:

        @task.virtualenv(
            task_id="bento_build",
            requirements='./requirements.txt',
            system_site_packages=False,
            provide_context=True,
        )
        def build_bento(**context):
            """
            Perform Bento build in a virtual environment.
            """
            import bentoml
            bento = bentoml.bentos.build(
                "service.py:svc",
                labels={
                    "job_id": context.run_id
                },
                python={
                    requirements_txt: "./requirements.txt"
                },
                include=["*"],
            )

        build_bento_task = build_bento()



Artifact Management
~~~~~~~~~~~~~~~~~~~

Since Airflow is a distributed system, it is important to save the
:doc:`Models </concepts/model>` and :doc:`Bentos </concepts/bento>` produced in your
Airflow pipeline to a central location that is accessible by all the nodes in the
Airflow cluster, and also by the workers in your production deployment environment.

For a simple setup, we recommend using the Import/Export API for
:ref:`Model <concepts/model:Managing Models>` and
:ref:`Bento <concepts/bento:Managing Bentos>`. This allows you to export the model files
directly to cloud storage, and import them from the same location when needed. E.g:

.. code-block:: python

    bentoml.models.export_model('s3://my_bucket/folder/')
    bentoml.models.import_model('s3://my_bucket/folder/iris_clf-3vl5n7qkcwqe5uqj.bentomodel')

    bentoml.export_bento('s3://my_bucket/bentos/')
    bentoml.import_bento('s3://my_bucket/bentos/iris_classifier-7soszfq53sv6huqj.bento')

For a more advanced setup, we recommend using the Model and Bento Registry feature
provided in `Yatai <https://github.com/bentoml/Yatai>`_, which provides additional
management features such as filtering, labels, and a web UI for browsing and managing
models. E.g:

.. code-block:: python

    bentoml.models.push("iris_clf:latest")
    bentoml.models.pull("iris_clf:3vl5n7qkcwqe5uqj")

    bentoml.push("iris_classifier:latest")
    bentoml.pull("iris_classifier:mcjbijq6j2yhiusu")


Python API or CLI
~~~~~~~~~~~~~~~~~

BentoML provides both Python APIs and CLI commands for most workflow management tasks,
such as building Bento, managing Models/Bentos, and deploying to production.

When using the Python APIs, you can organize your code in a Airflow PythonOperator task.
And for CLI commands, you can use the `BashOperator <https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/bash.html>`_
instead.


Validating new Bento
~~~~~~~~~~~~~~~~~~~~

It is important to validate the new Bento before deploying it to production. The
`bentoml.testing` module provides a set of utility functions for building behavior tests
for your BentoML Service, by launching the API server in a docker container and sending
test requests to it.

The BentoML community is also building a standardized way of defining and running
test cases for your Bento, that can be easily integrated with your CI/CD pipeline in
an Airflow job. See `#2967 <https://github.com/bentoml/BentoML/issues/2967>`_ for the
latest progress.

Saving model metadata
~~~~~~~~~~~~~~~~~~~~~

When saving a model with BentoML, you can pass in a dictionary of metadata to be saved
together with the model. This can be useful for tracking model evaluation metrics and
training context, such as the training dataset timestamp, training code version, or
training parameters.


Sample Project
--------------

The following is a sample project created by the BentoML community member Sarah Floris，
that demonstrates how to use BentoML with Airflow:

* 📖 `Deploying BentoML using Airflow <https://medium.com/codex/deploying-bentoml-using-airflow-28972343ac68>`_
* 💻 `Source Code <https://github.com/sdf94/bentoml-airflow>`_

