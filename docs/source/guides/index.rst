======
Guides
======

This chapter introduces the key features of BentoML. We recommend you read :doc:`/get-started/quickstart` before diving into this chapter.

.. grid:: 1 2 2 2
    :gutter: 3
    :margin: 0
    :padding: 3 4 0 0

    .. grid-item-card:: :doc:`/guides/services`
        :link: /guides/services
        :link-type: doc

        Understand the BentoML Service and its key components.

    .. grid-item-card:: :doc:`/guides/iotypes`
        :link: /guides/iotypes
        :link-type: doc

        Customize the input and output type of BentoML Services.

    .. grid-item-card:: :doc:`/guides/deployment`
        :link: /guides/deployment
        :link-type: doc

        Gain a general understanding of BentoCloud deployment.

    .. grid-item-card:: :doc:`/guides/containerization`
        :link: /guides/containerization
        :link-type: doc

        Create an OCI-compliant image for your BentoML project and deploy it anywhere.

    .. grid-item-card:: :doc:`/guides/build-options`
        :link: /guides/build-options
        :link-type: doc

        Customize the build configurations of a Bento.

    .. grid-item-card:: :doc:`/guides/model-store`
        :link: /guides/model-store
        :link-type: doc

        Use the BentoML local Model Store to manage your models in a unified way.

    .. grid-item-card:: :doc:`/guides/distributed-services`
        :link: /guides/distributed-services
        :link-type: doc

        Create distributed Services for advanced use cases.

    .. grid-item-card:: :doc:`/guides/concurrency`
        :link: /guides/concurrency
        :link-type: doc

        Set concurrency to enable your Service to handle multiple requests simultaneously.

    .. grid-item-card:: :doc:`/guides/testing`
        :link: /guides/testing
        :link-type: doc

        Create tests to verify the functionality of your model and the operational aspect of your Service.

    .. grid-item-card:: :doc:`/guides/clients`
        :link: /guides/clients
        :link-type: doc

        Use BentoML clients to interact with your Service.

    .. grid-item-card:: :doc:`/guides/adaptive-batching`
        :link: /guides/adaptive-batching
        :link-type: doc

        Enable adaptive batching to batch requests for reduced latency and optimized resource use.

    .. grid-item-card:: :doc:`/guides/asgi`
        :link: /guides/asgi
        :link-type: doc

        Integrate ASGI frameworks in a BentoML Service to provide additional features to exposed endpoints.

    .. grid-item-card:: :doc:`/guides/configurations`
        :link: /guides/configurations
        :link-type: doc

        Customize the runtime behaviors of your Service.

    .. grid-item-card:: :doc:`/guides/lifecycle-hooks`
        :link: /guides/lifecycle-hooks
        :link-type: doc

        Confgiure hooks to run custom logic at different stages of a Service's lifecycle.

.. toctree::
    :hidden:

    services
    iotypes
    deployment
    containerization
    build-options
    model-store
    distributed-services
    concurrency
    testing
    clients
    adaptive-batching
    asgi
    configurations
    lifecycle-hooks
