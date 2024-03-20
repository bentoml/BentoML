=======
Testing
=======

Testing is important for ensuring your code behaves as expected under various conditions. After creating a BentoML project, you can design different tests to verify both the functionality of the machine learning (ML) model and the operational aspects of the Service.

Testing provides multiple benefits, including:

- **Reliability**: Ensure your BentoML Service behaves as expected, increasing confidence in the its stability.
- **Regularity**: Facilitate regular and automated checking of the codebase for errors, helping catch issues early in the development cycle.
- **Refactorability**: Make the codebase more maintainable and adaptable to change, as tests provide a safety net for modifications.

This document explains how to design and run tests for BentoML Services. It uses the :doc:`Summarization Service in Quickstart </get-started/quickstart>` as an example for testing.

Prerequisites
-------------

Tests can be run using a test runner like ``pytest``. Install ``pytest`` via ``pip`` if you haven't already:

.. code-block:: bash

    pip install pytest

For more information, see `the pytest documentation <https://docs.pytest.org/en/latest/index.html>`_.

Unit tests
----------

Unit tests verify the smallest testable parts of a project, such as functions or methods, in isolation from the rest of the code. The purpose is to ensure that each component performs correctly as designed.

When dealing with ML models or Services like Summarization, where the output might not be exactly fixed, you can mock dependencies and output, and focus on testing the behavior and logic of the Service code rather than the model's output. You might not test the model's output directly but ensure that the BentoML Service interacts correctly with the model pipeline and processes inputs and outputs as expected.

An example:

.. code-block:: python
    :caption: `test_unit.py`

    from unittest.mock import patch, MagicMock
    from service import Summarization, EXAMPLE_INPUT # Imported from the Summarization service.py file


    @patch('service.pipeline')
    def test_summarization(mock_pipeline):
        # Setup a mock return value that resembles the model's output structure
        mock_pipeline.return_value = MagicMock(return_value=[{"summary_text": "Mock summary"}])

        service = Summarization()
        summary = service.summarize(EXAMPLE_INPUT)

        # Check that the mocked pipeline method was called exactly once
        mock_pipeline.assert_called_once()
        # Check the type of the response
        assert isinstance(summary, str), "The output should be a string."
        # Verify the length of the summarized text is less than the original input
        assert len(summary) < len(EXAMPLE_INPUT), "The summarized text should be shorter than the input."

This unit test does the following:

1. Use ``unittest.mock.patch`` to mock the ``pipeline`` function from the Transformers library.
2. Create a mock object that simulates the behavior of the callable object returned by the real ``pipeline`` function. Whenever this mock callable object is called, it returns a list containing a single dictionary with the key ``"summary_text"`` and value ``"Mock summary"``. For more information, see `mock object library <https://docs.python.org/3/library/unittest.mock.html>`_.
3. Make assertions to ensure the Service is functioning correctly.

.. note::

    When the output is fixed and known (for example, a function that returns a constant value or a predictable result based on the input), you can write tests that directly assert the expected output. In such cases, mocking might still be used to isolate the function from any dependencies it has, but the focus of the test can be on asserting that the function returns the exact expected value.

Run the unit test:

.. code-block:: bash

    pytest test_unit.py -v

Expected output:

.. code-block:: bash

    ====================================================================== test session starts ======================================================================
    platform linux -- Python 3.11.7, pytest-8.0.2, pluggy-1.4.0 -- /home/demo/Documents/summarization/summarization/bin/python
    cachedir: .pytest_cache
    rootdir: /home/demo/Documents/summarization
    plugins: anyio-4.3.0
    collected 1 item

    test_unit.py::test_summarization PASSED                                                                                                                   [100%]

    ======================================================================= 1 passed in 2.08s =======================================================================

Integration tests
-----------------

Integration tests assess the combined operation of two or more components. The goal is to ensure that different parts of your project work together as intended, including interactions with databases, external APIs, and other services.

Integration tests for a BentoML Service can involve starting the Service and sending HTTP requests to verify its response.

An example:

.. code-block:: python
    :caption: `test_integration.py`

    import bentoml
    import subprocess

    from service import EXAMPLE_INPUT # Imported from the Summarization service.py file

    def test_summarization_service_integration():
        with subprocess.Popen(["bentoml", "serve", "service:Summarization", "-p", "50001"]) as server_proc:
            try:
                client = bentoml.SyncHTTPClient("http://localhost:50001", server_ready_timeout=10)
                summarized_text = client.summarize(text=EXAMPLE_INPUT)

                # Ensure the summarized text is not empty
                assert summarized_text, "The summarized text should not be empty."
                # Check the type of the response
                assert isinstance(summarized_text, str), "The response should be a string."
                # Verify the length of the summarized text is less than the original input
                assert len(summarized_text) < len(EXAMPLE_INPUT), "The summarized text should be shorter than the input."
            finally:
                server_proc.terminate()

This integration test does the following:

1. Use the ``subprocess`` module to start the ``Summarization`` Service in a separate process on port ``50001``.
2. Create a :doc:`client </guides/clients>` and send a request. ``server_ready_timeout=10`` means the client will wait 10 seconds for the server to become ready before proceeding with the call.
3. Make assertions to ensure the Service is functioning correctly.

Run the integration test:

.. code-block:: bash

    pytest test_integration.py -v

Expected output:

.. code-block:: bash

    ====================================================================== test session starts ======================================================================
    platform linux -- Python 3.11.7, pytest-8.0.2, pluggy-1.4.0 -- /home/demo/Documents/summarization/summarization/bin/python
    cachedir: .pytest_cache
    rootdir: /home/demo/Documents/summarization
    plugins: anyio-4.3.0
    collected 1 item

    test_integration.py::test_summarization_service_integration PASSED                                                                                        [100%]

    ====================================================================== 1 passed in 19.29s =======================================================================

HTTP behavior tests
-------------------

To test the HTTP behavior of a BentoML Service, you can simulate HTTP requests and assert the responses match expected outcomes.

You can use the ``httpx`` library to create a test client. This allows you to send HTTP requests directly to your BentoML Service, which can be converted to an :doc:`ASGI application </guides/asgi>` via the ``to_asgi()`` method. The ``init`` parameter of ``to_asgi()``, when set to ``true``, initializes the middleware, routing, and other necessary configurations, preparing the application for handling requests.

An example:

.. code-block:: python
    :caption: `test_http.py`

    import httpx
    from service import Summarization, EXAMPLE_INPUT # Imported from the Summarization service.py file
    import pytest

    @pytest.mark.asyncio
    async def test_request():
        # Initialize the ASGI transport with the Summarization Service
        transport=httpx.ASGITransport(app=Summarization.to_asgi(init=True))

        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
            response = await test_client.post("/summarize", json={"text": EXAMPLE_INPUT})
            # Retrieve the text from the response for validation
            summarized_text = response.text
            # Assert that the HTTP response status code is 200, indicating success
            assert response.status_code == 200
            # Assert that the summarized text is not empty
            assert summarized_text, "The summary should not be empty"

This test does the following:

- Define a test function with ``@pytest.mark.asyncio``, which allows the test to perform asynchronous operations.
- Create an `asynchronous HTTP client <https://www.python-httpx.org/async/>`_, which interacts with the ASGI application converted from the ``Summarization`` Service through ``to_asgi(init=True)``. ``base_url="http://testserver"`` configures the client to send requests to a test server.
- Send a ``POST`` request to the ``/summarize`` endpoint. It simulates a client sending input data to the ``Summarization`` Service for processing.
- Make assertions to ensure the Service is functioning correctly.

Run the HTTP behavior test:

.. code-block:: bash

    pytest test_http.py -v

.. note::

    You need a plugin like ``pytest-asyncio`` to run async tests. You can install it by running ``pip install pytest-asyncio``.

Expected output:

.. code-block:: bash

    ================================================================================== test session starts ===================================================================================
    platform linux -- Python 3.11.7, pytest-8.0.2, pluggy-1.4.0 -- /home/demo/Documents/summarization/summarization/bin/python
    cachedir: .pytest_cache
    rootdir: /home/demo/Documents/summarization
    plugins: anyio-4.3.0, asyncio-0.23.5.post1
    asyncio: mode=Mode.STRICT
    collected 1 item

    test_http.py::test_request PASSED                                                                                                                                                  [100%]

    =================================================================================== 1 passed in 6.13s ====================================================================================

Best practices
--------------

Consider the following when designing your tests:

* Keep unit tests isolated; mock external dependencies to ensure tests are not affected by external factors.
* Automate tests using CI/CD pipelines to ensure they are run regularly.
* Keep tests simple and focused. A test should ideally verify one behavior.
* Ensure your testing environment closely mirrors your production environment to avoid "it works on my machine" issues.
* To `customize or configure <https://docs.pytest.org/en/stable/reference/customize.html>`_ ``pytest`` and make your testing process more efficient and tailored to your needs, you can create a ``pytest.ini`` configuration file. By specifying settings in ``pytest.ini``, you ensure that ``pytest`` consistently recognizes your project structure and preferences across different environments and setups. Here is an example:

  .. code-block:: ini

     [pytest]
     # Add current directory to PYTHONPATH for easy module imports
     pythonpath = .

     # Specify where pytest should look for tests, in this case, a directory named `test`
     testpaths = test

     # Optionally, configure pytest to use specific markers
     markers =
        integration: mark tests as integration tests.
        unit: mark tests as unit tests.

  Navigate to the root directory of your project (where ``pytest.ini`` is located), then run the following command to start testing:

  .. code-block:: bash

        pytest -v

  Expected output:

  .. code-block:: bash

        ================================================================================== test session starts ===================================================================================
        platform linux -- Python 3.11.7, pytest-8.0.2, pluggy-1.4.0 -- /home/demo/Documents/summarization/summarization/bin/python
        cachedir: .pytest_cache
        rootdir: /home/demo/Documents/summarization
        configfile: pytest.ini
        testpaths: test
        plugins: anyio-4.3.0, asyncio-0.23.5.post1
        asyncio: mode=Mode.STRICT
        collected 3 items

        test/test_http.py::test_request PASSED                                                                                                                                             [ 33%]
        test/test_integration.py::test_summarization_service_integration PASSED                                                                                                            [ 66%]
        test/test_unit.py::test_summarization PASSED                                                                                                                                       [100%]

        =================================================================================== 3 passed in 17.57s ===================================================================================
