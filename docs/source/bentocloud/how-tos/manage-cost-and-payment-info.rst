===================================
Manage cost and payment information
===================================

This page provides instructions on how to access and manage your BentoCloud cost and payment details on the BentoCloud console.

Add or edit payment methods
---------------------------

1. Click your username in the top-right corner of any BentoCloud page, then select **Payment**.
2. In the **Payment Method** section, you may see different options depending on whether you have an existing payment method.

   * Without prior payment methods: Click **Add on Stripe** and you will be redirected to complete the setup process on Stripe's payment platform.
   * With an existing payment method: Click **Edit on Stripe** to view or update your information as needed on Stripe's payment platform.

.. note::

    For Free Tier users, you can also click **Update Now** on the top banner, and click **Add on Stripe** in the dialog that appears to add a payment method. After that, click **Upgrade to Starter** to upgrade your plan.

View cost information
---------------------

1. Click your username in the top-right corner of any BentoCloud page, then select **Payment**.
2. On the **Payments** page, view detailed information on credits and current usage by instance type and by billing cycle.

   .. image:: ../../_static/img/bentocloud/how-to/manage-cost-and-payment/cost-and-payment-page.jpg

   * **Credit:** Displays the available amount of free credits in your BentoCloud account. Free credits are granted to your account by default once you have gained access to BentoCloud.
   * **Current Usage**: Displays a breakdown of costs associated with your usage by instance type. Depending on your plan, you may have access to different instance types. **Usage Quantity** represents the total time each instance type has been utilized. **Total Cost** = **Usage Quantity (Second)** × **Unit Cost**.
   * **Usage Dashboard**: Displays cost information by billing cycle, typically on a monthly basis. Billing statuses are defined as follows:

     * **Draft**: A preliminary bill has been automatically generated, pending review or adjustments before being issued.
     * **Active**: The bill has been finalized and sent to the customer, awaiting payment.
     * **Paid**: Payment for the bill has been received and recorded.
     * **Void**: The bill has been voided.
