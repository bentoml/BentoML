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

   We recommend you use the new Python SDK to configure the runtime environment. It allows you to integrate all build configurations in a single file and leverage Python's dynamic features like subclassing for more customization. If you prefer the previous methods using ``bentofile.yaml`` or ``pyproject.toml``, refer to the :doc:`/reference/bentoml/bento-build-options` document.

Basic usage
-----------

Create a ``Image`` instance with the desired configurations and attach it to your Service:

.. code-block:: python

    import bentoml

    my_image = bentoml.images.Image(python_version='3.11') \
        .python_packages("torch", "transformers")

    @bentoml.service(image=my_image)
    class MyService:
        # Service implementation

This example specifies:

- Python 3.11 as the runtime version
- Installation of PyTorch and Transformers libraries

.. note::

   Currently, it's not possible to define unique runtime environments for each Service in a multi-Service Bento deployment, but it's on our roadmap. For now, you only need to configure the runtime image of the entry Service (the one that calls other Services as dependencies).

Constructor parameters
----------------------

You can initialize a ``Image`` instance with the following parameters:

- ``python_version``: The Python version to use. If not specified, it defaults to the Python version in your current build environment.
- ``distro``: The Linux distribution for the base image. It defaults to ``debian``.
- ``base_image``: A custom Docker base image, which overrides all other attributes of the image.
- ``lock_python_packages``: Whether to lock all package versions and dependencies. It defaults to ``True``. You can set it to ``False`` if you have already specified versions for all packages.

Example usage:

.. code-block:: python

    import bentoml

    # Specify Python version and distro
    image_two = bentoml.images.Image(python_version='3.11', distro='alpine')

    # Specify a custom base image and disable version locking
    image_one = bentoml.images.Image(base_image="python:3.11-slim-buster", lock_python_packages=False)

Configuration methods
---------------------

The ``Image`` class provides various methods to customize the build process.

``python_packages()``
^^^^^^^^^^^^^^^^^^^^^^

Install specific Python dependencies by listing them directly. It supports version constraints.

.. code-block:: python

    import bentoml

    image = bentoml.images.Image(python_version='3.11') \
        .python_packages(
            "numpy>=1.20.0",
            "pandas",
            "scikit-learn==1.2.0",
            "git+https://github.com/username/mylib.git@main"
        )

.. note::

    You don't need to specify BentoML as a dependency in this field since the current version of BentoML will be added to the list by default. However, you can override this by specifying a different BentoML version.

To include a package from a GitHub repository, use `the pip requirements file format <https://pip.pypa.io/en/stable/reference/requirements-file-format/>`_. You can specify the repository URL, the branch, tag, or commit to install from, and the subdirectory if the Python package is not in the root of the repository.

.. code-block:: python

    import bentoml

    image = bentoml.images.Image(python_version='3.11') \
        .python_packages(
            "git+https://github.com/username/repository.git@branch_name",
            "git+https://github.com/username/repository.git@v1.0.0",
            "git+https://github.com/username/repository.git@abcdef1234567890abcdef1234567890abcdef12",
            "git+https://github.com/username/repository.git@branch_name#subdirectory=package_dir"
        )

If your project depends on a private GitHub repository, you can include the Python package from the repository via SSH. Make sure that the environment where BentoML is running has the appropriate SSH keys configured and that `these keys are added to GitHub <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account>`_.

.. code-block:: python

    import bentoml

    image = bentoml.images.Image(python_version='3.11') \
        .python_packages("git+ssh://git@github.com/username/repository.git@branch_name")

To configure PyPI indexes and other pip options (e.g. custom package sources and private repositories):

.. code-block:: python

    import bentoml

    # Using custom PyPI index
    image = bentoml.images.Image(python_version='3.11') \
        .python_packages(
            "--index-url https://download.pytorch.org/whl/cpu",
            "torch",
            "torchvision",
            "torchaudio"
        )

    # Configuring multiple PyPI options, including a private repository
    image = bentoml.images.Image(python_version='3.11') \
        .python_packages(
            "--index-url https://pypi.org/simple", # Default PyPI index
            "--extra-index-url https://my.private.pypi/simple", # Additional private repository
            "--trusted-host my.private.pypi", # Mark the private host as trusted
            "my-private-package"
        )

If your private package requires authentication:

.. code-block:: python

    import bentoml

    # Securely configure authentication for a private PyPI repository
    image = bentoml.images.Image(python_version='3.11') \
        .python_packages(
            # Use environment variables for security
            "--extra-index-url https://${USERNAME}:${PASSWORD}@my.private.pypi/simple",
            "my-private-package"
        )

    @bentoml.service(
        image=image,
        envs=[
            {"name": "USERNAME"},  # You can omit value and set it when deploying the Service
            {"name": "PASSWORD"},
        ]
    )
    class MyService:
       ...

``requirements_file()``
^^^^^^^^^^^^^^^^^^^^^^^^

You can also install Python dependencies from a ``requirements.txt`` file instead of using ``.python_packages()``.

.. code-block:: python

    import bentoml

    image = bentoml.images.Image(python_version='3.11') \
        .requirements_file("./path/to/requirements.txt")

``system_packages()``
^^^^^^^^^^^^^^^^^^^^^^

Install system-level dependencies in the runtime environment.

.. code-block:: python

    import bentoml

    image = bentoml.images.Image(python_version='3.11') \
        .system_packages("curl", "git")

``run()``
^^^^^^^^^^

Run custom commands during the build process. It supports chaining with other methods. This means you can freely combine all the above methods to create custom runtime environments.

Here is an example:

.. code-block:: python

    import bentoml

    image = bentoml.images.Image(python_version='3.11') \
        .run('echo "Starting build process..."') \
        .system_packages("curl", "git") \
        .run('echo "System packages installed"') \
        .python_packages("pillow", "fastapi") \
        .run('echo "Python packages installed"')

``run()`` is context-sensitive. For example, commands placed before ``.python_packages()`` are executed before installing Python dependencies, while those placed after are executed after installation. This allows you to perform certain tasks in the correct order.

Exclude files
-------------

You can define a ``.bentoignore`` file to exclude specific files when building your Bento. It uses standard pathspec patterns and the specified paths should be relative to the ``build_ctx`` directory (typically, this is the same directory where your ``service.py`` resides). This helps reduce the size of your Bento and keeps your runtime clean and efficient.

Here is an example:

.. code-block:: bash
   :caption: .bentoignore

   __pycache__/
   *.py[cod]
   *$py.class
   .ipynb_checkpoints/
   training_data/
   venv/

Next step
---------

After you've configured the environment specifications, you can :doc:`build a Bento </get-started/packaging-for-deployment>` or :doc:`deploy your Service to BentoCloud </get-started/cloud-deployment>`.
