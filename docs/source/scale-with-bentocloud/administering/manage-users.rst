============
Manage users
============

You can add and delete BentoCloud users to control and customize access to your resources.

Invite a new user
-----------------

1. Click your profile image in the top-right corner of any BentoCloud page, then select **Members**.
2. On the **Members** page, click **Invite new member**.
3. In the dialog that appears, provide the following information:

   - **Name**: The user's name.
   - **Email**: A valid email address to receive an invitation link to activate the account.
   - **Role**: The role assigned to the user. Available roles in BentoCloud include:

     - **Admin**: The Admin role has the highest level of access within BentoCloud. Users assigned with the Admin role are able to manage all resources in BentoCloud.
     - **Developer**: The Developer role is designed for users who need comprehensive access to the development and operational features but do not require control over billing or user management.
     - **Endpoint User**: The Endpoint User role is designed for users who primarily interact with already deployed models for inference purposes.

     For details, see :ref:`role-permissions`.

   .. image:: ../../_static/img/bentocloud/how-to/manage-users/invite-new-member.png
      :align: center

4. Click **Submit**.

Both the inviter and the new user receive the invitation email. Clicking the link in the email directs the new user to the login page, where they can choose to log in using either their Google or GitHub account.

Delete a user
-------------

1. Click your profile image in the top-right corner of any BentoCloud page, then select **Members**.
2. On the **Members** page, click **Delete Member** for the user you want to delete.
3. Click **Delete** in the dialog that appears.

.. _role-permissions:

Role permissions
----------------

The tasks you are able to perform depend on your user role. The following table lists the permissions allowed for each role.

.. list-table::
   :widths: 45 10 10 15
   :header-rows: 1

   * - Permission
     - Admin
     - Developer
     - Endpoint User
   * - View and edit billing information
     - ✓
     -
     -
   * - Add and remove users
     - ✓
     -
     -
   * - Change user roles
     - ✓
     -
     -
   * - Create, terminate, restore, and delete Deployments
     - ✓
     - ✓
     -
   * - View and update Deployment configurations
     - ✓
     - ✓
     -
   * - Create API tokens with Developer Operations Access
     - ✓
     - ✓
     -
   * - Create API tokens with Protected Endpoint Access
     - ✓
     - ✓
     - ✓
   * - Run inference with Deployments
     - ✓
     - ✓
     - ✓
   * - View Activities, Logging, and Monitoring information on a Deployment details page
     - ✓
     - ✓
     -
   * - View and pull Bentos
     - ✓
     - ✓
     - ✓
   * - View and pull models
     - ✓
     - ✓
     - ✓
   * - Push Bentos
     - ✓
     - ✓
     -
   * - Delete Bentos
     - ✓
     - ✓
     -
   * - Delete models
     - ✓
     - ✓
     -
   * - Configure standby instances
     - ✓
     -
     -
