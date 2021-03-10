Customizing InputAdapter
========================

BentoML allows user to create customize InputAdapter to better suit for their system.

This guide demonstrate how to extending how to create a custom InputAdapter that check the
incoming data and then uses the customized InputAdapter to create and test with BentoService

--------------------------
Create custom InputAdapter
--------------------------

The following code create a subclass from the `StringInput` and throw an AttributeError
if the `cohort_id` is missing in the data. If the incoming data is invalid, the input adapter
will discard the task with appropriate status code and message.

.. code-block:: python

    # my_custom_input.py

    import json
    from bentoml.adapters.string_input import StringInput


    class MyCustomJsonInput(StringInput):
        # See more information about tasks at https://docs.bentoml.org/en/latest/api/types.html#bentoml.types.InferenceTask
        def extract_user_func_args(self, tasks):
            json_inputs = []
            for task in tasks:
                try:
                    parsed_json = json.loads(task.data)
                    if parsed_json.get('cohort_id', None) is None:
                        raise AttributeError('Missing cohort data')
                    json_inputs.append(parsed_json)
                except json.JSONDecodeError:
                    task.discard(http_status=400, err_msg="Not a valid JSON format")
                except AttributeError:
                    task.discard(http_status=400, err_msg="Invalid request data")
                except Exception:  # pylint: disable=broad-except
                    err = traceback.format_exc()
                    task.discard(http_status=500, err_msg=f"Internal Server Error: {err}")
            return (json_inputs,)


-------------------------------------------------------------
Define and save BentoService with the customized InputAdapter
-------------------------------------------------------------

.. code-block:: python

    # my_bento_service.py

    import bentoml
    from bentoml.service.artifacts.common import PickleArtifact
    from my_custom_input import MyCustomJsonInput

    @bentoml.env(infer_pip=True)
    class MyService(bentoml.BentoService):
        @bentoml.api(input=MyCustomJsonInput(), batch=True)
        def predict(self, input_data):
            result = []
            for data in input_data:
                result.append(data.get('name') + data.get('cohort_id'))
            return result


.. code-block:: python

    from my_bento_service import MyService

    svc = MyService()
    svc.save()


----------------------
Test with example data
----------------------

.. code-block:: shell

    $ bentoml serve MyService:latest


.. code-block:: shell

    $ curl -i --header "Content-Type: application/json" \
      --request POST --data '{"name": "foo"}' \
      http://localhost:5000/predict

    # Output
    HTTP/1.0 400 BAD REQUEST
    X-Request-Id: cb63a61e-dc2a-4e12-a91c-8b15316a99df
    Content-Type: text/html; charset=utf-8
    Content-Length: 20
    Server: Werkzeug/0.15.4 Python/3.7.3
    Date: Wed, 10 Mar 2021 01:47:38 GMT

    Invalid request data%

    $ curl -i --header "Content-Type: application/json" \
      --request POST --data '{"name": "foo", "cohort_id": "1"}' \
      http://localhost:5000/predict

    # Output
    HTTP/1.0 200 OK
    Content-Type: application/json
    X-Request-Id: 34ad9963-4be5-47a1-afcf-774b9d866e76
    Content-Length: 6
    Server: Werkzeug/0.15.4 Python/3.7.3
    Date: Wed, 10 Mar 2021 01:48:37 GMT

    "foo1"%

