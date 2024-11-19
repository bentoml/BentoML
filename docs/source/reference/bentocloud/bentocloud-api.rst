==============
BentoCloud API
==============

This page provides API reference for creating and managing BentoCloud Deployments.

.. seealso::

    See :doc:`/scale-with-bentocloud/deployment/manage-deployments` for usage details.

Create
------

Create a Deployment on BentoCloud.

.. autofunction:: bentoml.deployment.create

For more information, see :doc:`/scale-with-bentocloud/deployment/create-deployments`.

Get
---

Retrieve details about a specific Deployment.

.. autofunction:: bentoml.deployment.get

For more information, see :ref:`scale-with-bentocloud/deployment/manage-deployments:view`.

List
----

List all Deployments on BentoCloud.

.. autofunction:: bentoml.deployment.list

Update
------

Update the configuration of a specific Deployment.

.. autofunction:: bentoml.deployment.update

For more information, see :ref:`scale-with-bentocloud/deployment/manage-deployments:update`.

Apply
-----

Create or update a Deployment based on the specifications provided.

.. autofunction:: bentoml.deployment.apply

For more information, see :ref:`scale-with-bentocloud/deployment/manage-deployments:apply`.

Terminate
---------

Stop a Deployment, which can be restarted later.

.. autofunction:: bentoml.deployment.terminate

For more information, see :ref:`scale-with-bentocloud/deployment/manage-deployments:terminate`.

Delete
------

Remove a Deployment from BentoCloud.

.. autofunction:: bentoml.deployment.delete

For more information, see :ref:`scale-with-bentocloud/deployment/manage-deployments:delete`.
