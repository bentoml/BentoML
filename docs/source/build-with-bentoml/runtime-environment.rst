==============================
Define the runtime environment
==============================

Since v1.3.20, BentoML introduces a new Python SDK for configuring the runtime environment of a Bento. You can set it alongside your BentoML Service code in ``service.py``.

Essentially, the runtime environment contains a set of Bento build options, such as:

- Python version
- Operating system (distro)
- System packages
- Python packages
- Package installation options
- Custom commands

.. important::

   We recommend you use the new Python SDK to configure the runtime environment. If you prefer the previous methods using ``bentofile.yaml`` or ``pyproject.toml``, refer to the :doc:`/reference/bentoml/bento-build-options` document.

Basic usage
-----------

Create a ``PythonImage`` instance with the necessary configurations and attach it to your Service:

.. code-block:: python

    import bentoml

    my_image = bentoml.images.PythonImage(python_version='3.11') \
        .python_packages("torch", "transformers")

    @bentoml.service(image=my_image)
    class MyService:
        # Service implementation

This example specifies:

- Python 3.11 as the runtime version
- Installation of PyTorch and Transformers libraries

Constructor parameters
----------------------

You can initialize a ``PythonImage`` instance with the following parameters:

- ``python_version``: The Python version to use. It defaults to the Python version in your build environment.
- ``distro``: The Linux distribution for the base image. It defaults to ``debian``.
- ``base_image``: A custom Docker base image, which overrides all other attributes of the image.
- ``lock_python_packages``: Whether to lock all package versions and dependencies. It defaults to ``True``. You can set it to ``False`` if you have already specified versions for all packages.

Example usage:

.. code-block:: python

    import bentoml

    # Specify Python version and distro
    image_two = bentoml.images.PythonImage(python_version='3.11', distro='alpine')

    # Specify a custom base image and disable version locking
    image_one = bentoml.images.PythonImage(base_image="python:3.11-slim-buster", lock_python_packages=False)

Configuration methods
---------------------

The ``PythonImage`` class provides various methods to customize the build process.

``.python_packages()``
^^^^^^^^^^^^^^^^^^^^^^

Install specific Python dependencies by listing them directly. It supports version constraints.

.. code-block:: python

    import bentoml

    image = bentoml.images.PythonImage(python_version='3.11') \
        .python_packages(
            "numpy>=1.20.0",
            "pandas",
            "scikit-learn==1.2.0"
        )

To configure PyPI indexes and other pip options:

.. code-block:: python

    import bentoml

    # Using custom PyPI index
    image = bentoml.images.PythonImage(python_version='3.11') \
        .python_packages(
            "--index-url https://download.pytorch.org/whl/cpu",
            "torch",
            "torchvision",
            "torchaudio"
        )

    # Multiple pip options
    image = bentoml.images.PythonImage(python_version='3.11') \
        .python_packages(
            "--index-url https://pypi.org/simple",
            "--extra-index-url https://my.private.pypi/simple",
            "--trusted-host my.private.pypi",
            "my-private-package"
        )

``.requirements_file()``
^^^^^^^^^^^^^^^^^^^^^^^^

You can also install Python dependencies from a ``requirements.txt`` file instead of using ``.python_packages()``.

.. code-block:: python

    import bentoml

    image = bentoml.images.PythonImage(python_version='3.11') \
        .requirements_file("./path/to/requirements.txt")

``.system_packages()``
^^^^^^^^^^^^^^^^^^^^^^

Install system-level dependencies in the runtime environment.

.. code-block:: python

    import bentoml

    image = bentoml.images.PythonImage(python_version='3.11') \
        .system_packages("curl", "git")

``.run()``
^^^^^^^^^^

Run custom commands during the build process. It supports chaining with other methods.

.. code-block:: python

    import bentoml

    image = bentoml.images.PythonImage(python_version='3.11') \
        .run('echo "Starting build process..."') \
        .system_packages("curl", "git") \
        .run('echo "System packages installed"') \
        .python_packages("pillow", "fastapi") \
        .run('echo "Python packages installed"')
