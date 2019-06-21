Using Bento Archive
===================

There are few ways to utilize Bento archive to work with different serving
scenarios.  We will list out few ways to use Bento archive and the serving
scenarios that they would be great fit with.


Using inside python application
*******************************

It is easy to  use Bento archive with python application. Import
BentoML package into your application and load archive with
``bentoml.load`` function

Load function works with local file system as well as AWS S3 path.
You can find more information at API reference page :ref:`api-load-ref`
section.

.. code-block:: python
    :linenos:

    import bentoml

    bento_service = bentoml.load('my/bento_archive/path/IrisClassifier')

    api_list = bento_service.get_service_apis()

    # the get_service_apis function will return a list of user defined APIs.
    print(api_list)

    print(bento_service.predict(INPUT_DATA))

In the example above, we imported bentoml and use load function to load our
IrisClassifier archive. After we load the archive into a bento service, we
can make prediction with ``bento_service.predict``.

If you defined more than one API functions, you can get the list of defined
API functions with ``get_service_apis`` method that's availabel on bento
service.


Install with ``pip install``
****************************

Bento archive includes a generated ``setup.py`` file that we can use to
distribute and use the archive as PyPi package.

Just navigate to the archive's directory and run ``pip install .``

.. code-block:: bash

    $ cd my/bento_archive/path/IrisClassifier
    $ pip install .

We can also publish this archive as python package to pypi.org or private PyPi index,
after you configured your ``.pypirc`` file.

.. code-block:: bash

    $ cd my/bento_archive/path/IrisClassifier
    $ python setup.py sdist upload


We can import and use bento archive just like a normal python package after
``pip install``. After you import package, you will need to call ``load``
function to initialize it into a bento service.

.. code-block:: python

    import IrisClassifier

    service = IrisClassifier.load()

    print(service.predict(INPUT_DATA))


Using generated CLI tool
++++++++++++++++++++++++

Another benefit of using ``pip install`` approach is the generated CLI tool.
Bento archive created a customized CLI tool for you.

You can check API info by calling `info` command.

.. code-block:: bash

    $ IrisClassifier info


Make prediction with data from local file or string.

.. code-block:: bash

    $ IrisClassifier predict --input JSON/FILE/PATH
    $ IrisClassifier predict --input '{"key": "value", "json": "string"}'

Start a local REST API server with `serve` command.  Default port is 5000, use
``--port`` options to change.

.. code-block:: bash

    $ IrisClassifier serve --port 5001


Using BentoML CLI tool
**********************

You can use BentoML's CLI tool without install the archive with pip. Just
provide the archive's location as local path or s3 location.

.. code-block:: bash

    $ bentoml info my/bento_archive/IrisClassifier
    $ bentoml info s3://my_bucket/bento_archive_path/IrisClassifier

.. code-block:: bash

    $ bentoml predict my/bento_archive/IrisClassifier --input JSON/FILE/PATH

.. code-block:: bash

    $ bentoml serve my/bento_archive/IrisClassifier --port 5001

We can start a gunicorn server to maximize on utilizing our computing
resources.

.. code-block:: bash

    $ bentoml serve-gunicorn my/bento_archive/IrisClassifier --port 5001 --workers 2


Generate Docker Image
*********************

BentoML also support to build docker image with bento archive.  Docker is
one the most common packaging format for deploying applications.
With bento archive, navigate to the archive directory and run ``docker build``

.. code-block:: bash

    $ cd my/bento_archive/IrisClassifier
    $ docker build . -t iris-classifier

After finish building docker image, we can run the image with ``docker run``

.. code-block:: bash

    $ docker run -p 5000:5000 iris-classifier

