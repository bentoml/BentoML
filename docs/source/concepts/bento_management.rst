.. _bento-management-page:

Model and Bento Management
**************************

BentoML provides easy to use local and centralized stores for managing models and bentos. This article 
focuses on the use of local file system based model and bento stores. To learn more about the centralized 
store solution, see BentoML Yatai. To connect the CLI to a remote `Yatai <yatai-service-page>`, 
use the `bentoml login` command.

.. todo::

    Link to BentoML Yatai documentation.

Managing Models
---------------

Creating Models
^^^^^^^^^^^^^^^

Recall the :ref:`Getting Started <getting-started-page>` guide, models are saved using the framework 
specific `save()` function. In the example, we used the `save()` function from the `sklearn` module for 
the Scikit Learn framework.

.. code-block:: python

    import bentoml.sklearn
    bentoml.sklearn.save("iris_classifier_model", clf)

Models can also be imported from support framework specific registries. In the example below, a model 
is imported from the MLFlow Model Registry.

.. code-block:: python

    import bentoml.mlflow
    bentoml.mlflow.import_from_uri("mlflow_model", uri=mlflow_registry_uri)

Saved and imported models are added to the local file system based model store located in the 
`$HOME/bentoml/models` directory by default.

Listing Models
^^^^^^^^^^^^^^

To list all the models created, use either the `list()` Python function in the `bentoml.models` 
modules or the `models list` CLI command:

.. tabs::

    .. code-tab:: python

        import bentoml.models

        bentoml.models.list() # get a list of all models
        # [
        #   {
        #     tag: Tag("iris_classifier_model", "vkorlosfifi6zhqqvtpeqaare"),
        #     framework: "SKLearn",
        #     created: 2021/11/14 03:55:11
        #   },
        #    {
        #     tag: Tag("iris_classifier_model", "vlqdohsfifi6zhqqvtpeqaare"),
        #     framework: "SKLearn",
        #     created: 2021/11/14 03:55:15
        #   },
        #   {
        #     tag: Tag("iris_classifier_model", "vmiqwpcfifi6zhqqvtpeqaare"),
        #     framework: "SKLearn",
        #     created: 2021/11/14 03:55:25
        #   },
        #   {
        #     tag: Tag("fraud_detection_model", "5v4pdccfifi6zhqqvtpeqaare"),
        #     framework: "PyTorch",
        #     created: 2021/11/14 03:57:01
        #   },
        #   {
        #     tag: Tag("fraud_detection_model", "5xorursfifi6zhqqvtpeqaare"),
        #     framework: "PyTorch",
        #     created: 2021/11/14 03:57:45
        #   },
        # ]
        bentoml.models.list("iris_classifier_model") # get a list of all versions of a specific model
        bentoml.models.list(Tag("iris_classifier_model", None))
        # [
        #   {
        #     tag: Tag("iris_classifier_model", "vkorlosfifi6zhqqvtpeqaare"),
        #     framework: "SKLearn",
        #     created: 2021/11/14 03:55:11
        #   },
        #    {
        #     tag: Tag("iris_classifier_model", "vlqdohsfifi6zhqqvtpeqaare"),
        #     framework: "SKLearn",
        #     created: 2021/11/14 03:55:15
        #   },
        #   {
        #     tag: Tag("iris_classifier_model", "vmiqwpcfifi6zhqqvtpeqaare"),
        #     framework: "SKLearn",
        #     created: 2021/11/14 03:55:25
        #   },
        # ]

    .. code-tab:: bash

        > bentoml models list # list all models
        MODEL                 FRAMEWORK   VERSION                    CREATED
        iris_classifier_model SKLearn     vkorlosfifi6zhqqvtpeqaare  2021/11/14 03:55:11
        iris_classifier_model SKLearn     vlqdohsfifi6zhqqvtpeqaare  2021/11/14 03:55:15
        iris_classifier_model SKLearn     vmiqwpcfifi6zhqqvtpeqaare  2021/11/14 03:55:25
        fraud_detection_model PyTorch     5v4pdccfifi6zhqqvtpeqaare  2021/11/14 03:57:01
        fraud_detection_model PyTorch     5xorursfifi6zhqqvtpeqaare  2021/11/14 03:57:45
        > bentoml models list iris_classifier # list all version of my-model
        MODEL           FRAMEWORK   VERSION          CREATED
        iris_classifier_model PyTorch     vkorlosfifi6zhqqvtpeqaare  2021/11/14 03:55:11
        iris_classifier_model PyTorch     vlqdohsfifi6zhqqvtpeqaare  2021/11/14 03:55:15
        iris_classifier_model SKLearn     vmiqwpcfifi6zhqqvtpeqaare  2021/11/14 03:55:25

To get model information, use either the `get()` function under the `bentoml.models` module or 
the models get CLI command.

.. tabs::

    .. code-tab:: python

        import bentoml.models

        bentoml.models.get("iris_classifier_model:vmiqwpcfifi6zhqqvtpeqaare")
        bentoml.models.get(Tag("iris_classifier_model", "vmiqwpcfifi6zhqqvtpeqaare"))
        # Model(
        #   tag: Tag("iris_classifier_model", "vmiqwpcfifi6zhqqvtpeqaare"),
        #   framework: "SKLearn",
        #   created: 2021/11/14 03:55:25
        #   description: "The iris classifier model"
        #   path: "/user/home/bentoml/models/iris_classifier_model/vmiqwpcfifi6zhqqvtpeqaare"
        # )
    
    .. code-tab:: bash

        > bentoml models get iris_classifier_model:vmiqwpcfifi6zhqqvtpeqaare
        TAG         iris_classifier_model:vmiqwpcfifi6zhqqvtpeqaare
        FRAMEWORK   SKLearn
        CREATED     2021/9/21 10:07:45
        DESCRIPTION The iris classifier model
        PATH        /user/home/bentoml/models/iris_classifier_model/vmiqwpcfifi6zhqqvtpeqaare

Deleting Models
^^^^^^^^^^^^^^^

To delete models in the model store, use either the `delete()` function under the `bentoml.models` 
module or the `models delete` CLI command.

.. tabs::

    .. code-tab:: python

        import bentoml.models

        bentoml.models.delete("iris_classifier_model:vmiqwpcfifi6zhqqvtpeqaare", skip_confirm=True)
    
    .. code-tab:: bash

        > bentoml models delete iris_classifier_model:vmiqwpcfifi6zhqqvtpeqaare

Pushing Models
^^^^^^^^^^^^^^

Once you are happy with a model and ready to share with other collaborators, you can upload it to a 
remote `Yatai <yatai-service-page>` model store with the `push()` function under the `bentoml.models` 
module or the `models push` CLI command.

.. tabs::

    .. code-tab:: python 

        import bentoml.models

        bentoml.models.delete("iris_classifier_model:vmiqwpcfifi6zhqqvtpeqaare", skip_confirm=True)
    
    .. code-tab:: bash

        > bentoml models delete iris_classifier_model:vmiqwpcfifi6zhqqvtpeqaare

Pulling Models
^^^^^^^^^^^^^^

Previously pushed models can be downloaded from `Yatai <yatai-service-page>` and saved local model 
store with the `pull()` function under the `bentoml.models` module or the `models pull` CLI command.

.. tabs::

    .. code-tab:: python 

        import bentoml.models

        bentoml.modles.pull("iris_classifier_model:vmiqwpcfifi6zhqqvtpeqaare", url=yatai_url)
    
    .. code-tab:: bash

        > bentoml models pull iris_classifier_model:vmiqwpcfifi6zhqqvtpeqaare

Managing Bentos
---------------

Creating Bentos
^^^^^^^^^^^^^^^

Bentos are created through the bento build process. Recall the :ref:`Getting Started <getting-started-page>` 
guide, bentos are built with the `build` CLI command. See :ref:`Building Bentos <building-bentos-page>` 
for more details. Built bentos are added to the local file system based bento store located under 
the `$HOME/bentoml/bentos` by default.

.. code-block:: bash

    > bentoml build ./bento.py:svc

Listing Bentos
^^^^^^^^^^^^^^

To view bentos in the bento store, use the `list` CLI command.

.. code-block:: bash

    > bentoml list
    BENTO                   VERSION                    LABELS      CREATED
    iris_classifier_service v5mgcacfgzi6zdz7vtpeqaare  iris,prod   2021/09/19 10:15:50

Deleting Bentos
^^^^^^^^^^^^^^^

To delete bentos in the bento store, use  the `delete` CLI command.

.. code-block:: bash
    
    > bentoml delete iris_classifier_service:v5mgcacfgzi6zdz7vtpeqaare

Pushing Bentos
^^^^^^^^^^^^^^

To upload bento in the local file system store to a remote `Yatai <yatai-service-page>` bento store 
for collaboration and deployment, use the `push` CLI command.

.. code-block:: bash
    
    > bentoml push iris_classifier_service:v5mgcacfgzi6zdz7vtpeqaare

Pulling Bentos
^^^^^^^^^^^^^^

To download a bento from a remote `Yatai <yatai-service-page>` bento store to the local file system 
bento store for troubleshooting, use the `pull` CLI command.

.. code-block:: bash

    > bentoml pull iris_classifier_service:v5mgcacfgzi6zdz7vtpeqaare

Further Reading
---------------
- Install Yatai
- Yatai System Admin Guide

.. todo::

    Complete and link the further reading documentation
