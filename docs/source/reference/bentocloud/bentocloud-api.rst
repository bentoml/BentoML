==============
BentoCloud API
==============

This page provides API reference for managing BentoCloud resources including Deployments and API tokens.

.. seealso::

    - :doc:`/scale-with-bentocloud/deployment/manage-deployments` for Deployment usage details
    - :doc:`/scale-with-bentocloud/manage-api-tokens` for API token usage details

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

API Token Management
====================

The ``bentoml.api_token`` module provides functions for managing API tokens on BentoCloud programmatically.

List API tokens
---------------

List all API tokens in your organization.

.. autofunction:: bentoml.api_token.list

Create an API token
-------------------

Create a new API token with specified scopes and expiration.

.. autofunction:: bentoml.api_token.create

Get API token
-------------

Retrieve details about a specific API token by its UID.

.. autofunction:: bentoml.api_token.get

Delete API token
----------------

Delete an API token by its UID.

.. autofunction:: bentoml.api_token.delete

For more information and examples, see :doc:`/scale-with-bentocloud/manage-api-tokens`.
