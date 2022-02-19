.. _containerize-bentos-page:

Containerize Bentos as Docker Images
************************************

Containerizing bentos as Docker images allows users to easily distribute and deploy bentos. 
Once services are built as bentos and saved to the bento store, we can containerize saved bentos 
with the CLI command `bentoml containerize`.

Start the Docker engine. Verify using `docker info`.

.. code-block:: bash

    > docker info

Run `bentoml list` to view available bentos in the store.

.. code-block:: bash

    > bentoml list
    BENTO                   VERSION                    LABELS      CREATED
    iris_classifier_service v5mgcacfgzi6zdz7vtpeqaare  iris,prod   2021/09/19 10:15:50

Run `bentoml containerize` to start the containerization process.

.. code-block:: bash

    > bentoml containerize iris_classifier_service:latest
    Containerizing iris_classifier_service:v5mgcacfgzi6zdz7vtpeqaare with docker daemon from local environment
    ✓ Build container image: iris_classifier_service:v5mgcacfgzi6zdz7vtpeqaare

Built Docker images are stored to the local Docker repository.

.. code-block:: bash

    > docker images
    REPOSITORY               TAG               IMAGE ID       CREATED         SIZE
    IrisClassifierService    20210919_UN30CA   669e3ce35013   1 minutes ago   1.21GB

We can run the images with `docker run`.

.. code-block:: bash

    > docker run IrisClassifierService:20210919_UN30CA
    [INFO] Starting BentoML API server in development mode with auto-reload enabled
    [INFO] Serving BentoML Service "IrisClassifierService" defined in "bento.py"
    [INFO] API Server running on http://0.0.0.0:5000

.. todo::

    Add a further reading section

