===================================
BentoCloud BYOC Setup Guide for AWS
===================================

As part of our commitment to ensuring a smooth and efficient onboarding process, we have prepared this setup guide to help your DevOps team integrate BentoCloud into your AWS account.

Setup
-----

This setup process will establish an IAM Role to provide BentoCloud with the necessary access to a specific set of AWS services. This access is crucial for BentoCloud to deploy and manage cloud components within your AWS environment. The process utilizes a predefined CloudFormation template, as outlined in the steps below.

1. Log in to your organization's account on AWS.
2. Go to the `CloudFormation <http://console.aws.amazon.com/cloudformation/home>`_ web console. Ensure you are in the desired AWS region. Switch region if necessary.
3. On the **Stacks** page, choose **Create stack** > **With new resources (standard)**.

   .. image:: ../../_static/img/bentocloud/byoc/aws/stack-with-new-resources.png

4. In the **Create stack** section, select **Choose an existing template** and **Amazon S3 URL** to paste the following URL, and then click **Next**:

   .. code-block:: bash

        https://customer-helpdesk.s3.us-west-1.amazonaws.com/create-bentocloud-copilot-role.json

   .. image:: ../../_static/img/bentocloud/byoc/aws/create-stack.png

5. In the **Specify stack details** section, provide the following information and click **Next**.

   - **Stack name**: ``bentocloud``
   - **BentoCloudCopilot**: ``arn:aws:iam::303081928216:user/bentocloud-copilot-[organization_name]``

   .. image:: ../../_static/img/bentocloud/byoc/aws/specify-stack-details.png

6. In the **Configuring stack options** section, keep the default selections and click **Next**.
7. In the **Review and create** section, scroll down to **Capabilities** to acknowledge IAM Role creation, and click **Submit**.
8. Share the ``InstallerRole`` value with the BentoML team.

   a. Go to **CloudFormation** > **Stacks** > **bentocloud**, and wait until the stack enters the following state:

      .. image:: ../../_static/img/bentocloud/byoc/aws/aws-state-one.png

      .. image:: ../../_static/img/bentocloud/byoc/aws/aws-state-two.png

   b. Go to the **Outputs** tab, and copy the value of ``InstallerRole``.

      .. image:: ../../_static/img/bentocloud/byoc/aws/value.png

Post setup
----------

Please inform your main BentoML contact once the steps above are completed, and share the **InstallerRole's value**, and **your AWS region** with the BentoML team.

After that, the BentoML automation will continue the cluster setup, which typically takes 1 business day. The BentoML team typically will run a small test deployment on your AWS account to ensure the system is working correctly end-to-end.

(Optional) Granting and revoking authorization
----------------------------------------------

You may revoke the authorization to BentoCloud copilot when there is no ongoing support ticket.

1. Go to the `Identity and Access Management (IAM) console <https://console.aws.amazon.com/iam/>`_.
2. In the navigation pane, select **Roles** > **The copilot role.**

   .. image:: ../../_static/img/bentocloud/byoc/aws/bentocloud-copilot-iam.png

3. Removing/adding the ``AWS`` line from the ``Principal`` field will revoke/grant authorization to BentoCloud copilot.

Getting help and troubleshooting
--------------------------------

Please reach out to us if you encounter any issues or have questions during the setup process. Our support team is available to assist you with:

- Detailed walkthroughs of each step
- Troubleshooting common issues
- Best practices for managing BentoCloud in your AWS environment

You can contact our support team at support@bentoml.com or through our support Slack channel.
