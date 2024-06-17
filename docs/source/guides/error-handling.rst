=============
Error Handling
=============
BentoML allows user to define custom error handling logic for their Service.

This document describes how to define a custom exception class and error handling logic in a BentoML Service.

Custom Exception Class
-----------------------
User can define custom exception class by inheriting from `BentoMLException` or any of its subclasses, such as `InvalidArgument` in the example below.

.. note::

   Error Code 401, 403, and >500 is reserved by BentoML, user should not use these error codes in their custom exception class.

.. code-block:: python

    import bentoml
    from bentoml.exceptions import InvalidArgument, BentoMLException
    from http import HTTPStatus

    class MyCustomException(BentoMLException):
        error_code = HTTPStatus.METHOD_NOT_ALLOWED

    class MyCustomInvalidArgsException(InvalidArgument):
        pass


    @bentoml.service
    class MyService:

        @bentoml.api
        def test1(self, text: str) -> str:
            raise MyCustomException("this is a custom error message")

        @bentoml.api
        def test2(self, text: str) -> str:
            raise MyCustomInvalidArgsException("this is a custom error message")
