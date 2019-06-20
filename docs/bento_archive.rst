Using Bento Archive
===================

There are few ways to utilize Bento archive to work with different serving
scenarios.  We will list out few ways to use Bento archive and the serving
scenarios that they would be great fit with.


Using inside python application
*******************************

The easiest way to use Bento archive is with python application. Import
BentoML package into your application and load archive with
``bentoml.load`` function

Load function works with local file system as well as AWS S3 path.
You can find more information at API reference page :ref:`api-load-ref`
section.

.. code-block:: python

    import bentoml

    bento_service = bentoml.load('/path/to/bento/archive')

    api_list = bento_service.get_service_apis()

    # the get_service_apis function will return a list of user defined APIs.
    print(api_list)

    print(bento_service.predict(INPUT_DATA))


Install with ``pip install``
****************************

.. code-block:: bash

    $ cd BENTO/ARCHIVE/PATH
    $ pip install .


.. code-block:: python

    import YourBentoService

    service = YourBentoService.load()

    print(service.get_service_apis())

    print(service.predict(INPUT_DATA))


Using generated CLI tool
++++++++++++++++++++++++

After ``pip install``, you can use the generated CLI tool.

.. code-block:: bash

    $ YourBentoService info


.. code-block:: bash

    $ YourBentoService API_name --input=JSON/FILE/PATH
    $ YourBentoService API_name --input='{"key": "value", "json": "string"}'


.. code-block:: bash

    $ YourBentoService serve --port 5001
