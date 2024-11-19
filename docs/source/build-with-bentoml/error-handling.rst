=========================
Customize error responses
=========================

Effective error handling is important for building user-friendly AI applications. BentoML facilitates this by allowing you to customize error handling logic for a :doc:`Service </build-with-bentoml/services>`.

This document describes how to define a custom exception class and error handling logic in a BentoML Service.

Custom exception class
----------------------

To define a custom exception, inherit from ``BentoMLException`` or one of its subclasses, such as ``InvalidArgument`` in the example below.

.. note::

   BentoML reserves error codes 401, 403, and any above 500. Avoid using these in custom exceptions.

.. code-block:: python

    import bentoml
    from bentoml.exceptions import InvalidArgument, BentoMLException
    from http import HTTPStatus

    # Define a custom exception for method not allowed errors
    class MyCustomException(BentoMLException):
        error_code = HTTPStatus.METHOD_NOT_ALLOWED

    # Define a simple custom exception for invalid argument errors
    class MyCustomInvalidArgsException(InvalidArgument):
        pass


    @bentoml.service
    class MyService:

        @bentoml.api
        def test1(self, text: str) -> str:
            # Raise the custom method not allowed exception
            raise MyCustomException("This is a custom error message.")

        @bentoml.api
        def test2(self, text: str) -> str:
            # Raise the custom invalid argument exception
            raise MyCustomInvalidArgsException("This is a custom error message.")

For more information, see :doc:`exception APIs </reference/bentoml/exceptions>`.
