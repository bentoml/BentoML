Customizing InputAdapter
========================

BentoML allows user to create customize InputAdapter to better suit for their system.

This guide demonstrate how to extending how to create a custom InputAdapter that check the
incoming data and then uses the customized InputAdapter to create and test with BentoService

-----------------------------
1. Create custom InputAdapter
-----------------------------

The following code create a subclass from the `StringInput` and makes the incoming data into a pandas dataframe.
Batch processing is possible because multiple inputs are considered. If `df_data` does not exist in incoming data,
the input adapter will discard the task with appropriate status code and message of the custom exception class.

.. code-block:: python

    # my_custom_input.py

    import json
    import traceback
    import pandas as pd
    from enum import Enum
    from typing import Iterable, Sequence, Tuple
    from bentoml.adapters.string_input import StringInput
    from bentoml.types import InferenceTask, JsonSerializable

    ApiFuncArgs = Tuple[
        Sequence[JsonSerializable],
    ]

    class ErrorCode(Enum):
        INPUT_FORMAT_INVALID = ("1000", "Missing df_data")
        def __init__(self, code, msg):
            self.code = code
            self.msg = msg

    class MyCustomException(Exception):
        def __init__(self ,code, msg):
            self.code = code
            self.msg = msg

    class MyCustomDataframeInput(StringInput):
        def extract_user_func_args(self, tasks: Iterable[InferenceTask[str]]) -> ApiFuncArgs:
            json_inputs = []
            for task in tasks:
                try:
                    parsed_json = json.loads(task.data)
                    if parsed_json.get("df_data") is None:
                        raise MyCustomException(
                            msg=ErrorCode.INPUT_FORMAT_INVALID.msg, code=ErrorCode.INPUT_FORMAT_INVALID.code
                        )
                    else:
                        df_data = parsed_json.get("df_data")
                        task.batch = len(df_data)
                        json_inputs.extend(df_data)
                except json.JSONDecodeError:
                    task.discard(http_status=400, err_msg="Not a valid JSON format")
                except MyCustomException as e:
                    task.discard(http_status=200, err_msg="Msg : {msg}, Error Code : {code}".format(msg=e.msg, code=e.code))
                except Exception:
                    err = traceback.format_exc()
                    task.discard(http_status=500, err_msg=f"Internal Server Error: {err}")

            df_inputs=pd.DataFrame.from_dict(json_inputs, orient='columns')

            return (df_inputs,)


----------------------------------------------------------------
2. Define and save BentoService with the customized InputAdapter
----------------------------------------------------------------

.. code-block:: python

    # my_bento_service.py

    import bentoml
    import logging
    import pandas as pd
    from bentoml.frameworks.keras import KerasModelArtifact
    from my_custom_input import MyCustomDataframeInput

    bentoml_logger = logging.getLogger("bentoml")

    @bentoml.env(infer_pip_packages=True)
    @bentoml.artifacts([KerasModelArtifact('my_model')])
    class MyService(bentoml.BentoService):
        @bentoml.api(input=MyCustomDataframeInput(), batch=True)
        def predict(self, df: pd.DataFrame):
            return self.artifacts.my_model.predict(df)


.. code-block:: python

    from my_bento_service import MyService

    svc = MyService()
    svc.pack("my_model", my_model)
    svc.save()


-------------------------
3. Test with example data
-------------------------

.. code-block:: shell

    $ bentoml serve MyService:latest


In another terminal to make a `curl` request

.. code-block:: shell

    $ curl -i --header "Content-Type: application/json" \
      --request POST --data '{"df_data":[{"feature1":0.0013,"feature2":0.0234,"feature3":0.0234}]}' \
      http://localhost:5000/predict

    # Output
    HTTP/2 200
    content-type: application/json
    content-length: 22
    date: Fri, 19 Nov 2021 15:53:23 GMT
    server: Python/3.6 aiohttp/3.7.4.post0

    [[0.9023]]

    $ curl -i --header "Content-Type: application/json" \
      --request POST --data '{"df_data":[{"feature1":0.0013,"feature2":0.0234,"feature3":0.0234},{"feature1":0.0029,"feature2":0.0287,"feature3":0.0980}]}' \
      http://localhost:5000/predict

    # Output
    HTTP/2 200
    content-type: application/json
    content-length: 22
    date: Fri, 19 Nov 2021 15:53:23 GMT
    server: Python/3.6 aiohttp/3.7.4.post0

    [[0.9023], [0.7283]]

    $ curl -i --header "Content-Type: application/json" \
      --request POST --data '{"not_valid":[{"feature1":0.0013,"feature2":0.0234,"feature3":0.0234}]}' \
      http://localhost:5000/predict

    # Output
    HTTP/2 200
    content-length: 40
    content-type: text/plain; charset=utf-8
    date: Fri, 19 Nov 2021 15:51:59 GMT
    server: Python/3.6 aiohttp/3.7.4.post0

    Msg : Missing df_data, Error Code : 1000

    $ curl -i --header "Content-Type: application/json" \
      --request POST --data '"not_valid":[{"feature1":0.0013,"feature2":0.0234,"feature3":0.0234}]}' \
      http://localhost:5000/predict

    # Output
    HTTP/2 400
    content-length: 23
    content-type: text/plain; charset=utf-8
    date: Fri, 19 Nov 2021 16:04:36 GMT
    server: Python/3.6 aiohttp/3.7.4.post0

    Not a valid JSON format