===========================
Manage Models and Bentos
===========================

In this guide, we'll walk through the process of managing your Models and Bentos
effectively in BentoML. This includes pulling and pushing Models and Bentos,
dealing with new versions, and viewing their status.

Prerequisite
============

Ensure you have BentoML installed and are properly logged into your BentoCloud
account. If you're not yet logged in, use the following command:

.. code-block:: bash

   bentoml cloud login --api-token <your-api-token> --endpoint <your-bentocloud-endpoint>

.. note::

   Refer to :doc:`Manage Access Tokens <manage-access-token>`
   on how to obtain your access token.

Local and Remote Repositories
=============================

Models and Bentos are initially created and stored locally.
BentoCloud provides a remote repository feature, similar to Git,
which enables you to push these local assets to a remote repository
for efficient management and team collaboration.

Pull and Push Models
====================

New Model Versions
------------------

As your ML models evolve, you'll often need to create new versions.
Here's how you can manage those versions:

1. **Pull a Model:** To pull a model from the repository,
   use the ``bentoml models pull`` CLI command followed by the model name and tag.

.. code-block:: bash

   bentoml models pull <model_name>:<tag>

1. **Push a Model:** Push a new model version to the repository
   using the ``bentoml models push`` command followed by the model name and tag.

.. code-block:: bash

   bentoml models push <model_name>:<tag>

You can view your models in BentoCloud’s UI in `cloud.bentoml.com/models <http://cloud.bentoml.com/models>`_.

.. image:: ../../_static/img/bentocloud/manage-models.gif
   :alt: manage-models.gif

Pull and Push Bento
===================

New Bento Versions
------------------

Similarly, you can manage different versions of Bentos:

1. **Pull a Bento:** To pull a specific Bento version from the repository,
   use the ``bentoml pull`` CLI command followed by the Bento name and tag.

.. code-block:: bash

   bentoml pull <bento_name>:<tag>

1. **Push a Bento:** Push a new Bento version to the repository
   using the ``bentoml push`` command followed by the Bento name and tag.

.. code-block:: bash

   bentoml push <bento_name>:<tag>

You can view your models in BentoCloud’s UI in `cloud.bentoml.com/bento_repositories <http://cloud.bentoml.com/bento_repositories>`_.

.. image:: ../../_static/img/bentocloud/manage-bentos.gif
   :alt: manage-bentos.gif

Python SDK
==========

In addition to CLI operations, BentoCloud also provides a Pythonic API for managing models and bentos.

.. TODO::
    Link Python API reference.

That's it! You've learned how to effectively manage your Models and Bentos in BentoML.
By understanding these fundamental operations, you can improve your model development
workflow and make your team's work more efficient.
