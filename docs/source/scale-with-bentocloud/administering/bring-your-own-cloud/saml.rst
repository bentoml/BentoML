BentoCloud SSO with SAML
========================

SAML is an XML-based open standard for transferring user authentication and authorization data between an identity provider (IdP) and a service provider (SP). In this context, BentoCloud acts as the service provider, allowing your users to authenticate using your organization's identity provider. This means that your users can utilize their existing corporate or organizational credentials to access BentoCloud resources securely.

Obtain SAML configuration parameters
------------------------------------

You will need to provide the following SAML configuration parameters to the BentoML team through a secure channel.

- **SAML Issuer**: This is the issuer (entity ID) of your identity provider.
- **SAML Identity Provider Certificate**: The certificate your IdP uses to sign SAML assertions.

.. note::

    The BentoML team will invite you to join their secure vault platform to share these parameters. If you are not already set up, please provide the BentoML team with an email address to send the secure vault invitation.

Configure your identity provider
--------------------------------

In your identity provider (e.g., Okta), create a new SAML 2.0 application (or the equivalent). You will need to add the following settings:

- **Application Callback URL**:

  .. code-block:: text

     https://bentoml-prod.firebaseapp.com/__/auth/handler

- **Application Login URI**:

  .. code-block:: text

     https://<your-org-name>.cloud.bentoml.com/signup

Make sure to replace ``<your-org-name>`` with the subdomain or organization name provided by BentoCloud. The Login URI is the endpoint within BentoCloud that users will be directed to in order to initiate the SAML login process.

Verify and test SAML setup
--------------------------

After BentoML configures SAML on BentoCloud, verify that you can access the BentoCloud sign-up/login page at ``https://<your-org-name>.cloud.bentoml.com/signup``. You should see only one **Continue with SSO** option, which should redirect you to your identity provider’s login page. Log in with your identity provider credentials, after which you should be directed back to BentoCloud with an active session.

Manage users
------------

When a new user signs in through SSO, BentoCloud automatically creates a new user account under your organization. By default, newly created user accounts are assigned the Developer role. If you wish to change the role of a new user, an organization admin can modify the user’s role through the BentoCloud admin interface.

It is important to note that removing or disabling a user in your SSO provider does not automatically remove them from BentoCloud. The user's session will remain valid until it expires. To completely remove a user’s access to BentoCloud, an organization admin must manually delete the user’s membership within the BentoCloud platform.
