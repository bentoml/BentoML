import numpy as np
import json

import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "..", ".."))
import bentoml


def test_predict_function(query, func, input_type):
    """Tests that the user's function has the correct signature and can be 
    properly saved and loaded.

    The function should take a dict request object like the query frontend
    expects JSON, the predict function, and the input type for the model.

    For example, the function can be called like:
        clipper_conn.test_predict_function({"input": [1.0, 2.0, 3.0]}, predict_func, "doubles")

    Parameters
    ----------
    query: JSON or list of dicts
        Inputs to test the prediction function on.
    func: function
        Predict function to test.
    input_type: str
        The input_type to be associated with the registered app and deployed model.
        One of "integers", "floats", "doubles", "bytes", or "strings".
    """
    query_data = list(x for x in list(query.values()))
    query_key = list(query.keys())

    if query_key[0] == "input_batch":
        query_data = query_data[0]

    print(query_data)
    try:
        flattened_data = [item for sublist in query_data for item in sublist]
    except TypeError:
        return "Invalid input type or JSON key"

    numpy_data = None

    if input_type == "bytes":
        numpy_data = list(np.int8(x) for x in query_data)
        for x in flattened_data:
            if type(x) != bytes:
                return "Invalid input type"

    if input_type == "integers":
        numpy_data = list(np.int32(x) for x in query_data)
        for x in flattened_data:
            if type(x) != int:
                return "Invalid input type"

    if input_type == "floats" or input_type == "doubles":
        if input_type == "floats":
            numpy_data = list(np.float32(x) for x in query_data)
        else:
            numpy_data = list(np.float64(x) for x in query_data)
        for x in flattened_data:
            if type(x) != float:
                return "Invalid input type"

    if input_type == "strings":
        numpy_data = list(np.str_(x) for x in query_data)
        for x in flattened_data:
            if type(x) != str:
                return "Invalid input type"

    print('func', func)
    print('data', numpy_data)
    return func(numpy_data)


saved_path = "./model"
api_name = "predict"
bento_service = bentoml.load(saved_path)
api = bento_service.get_service_apis()[0]

fake_data = {"input": '[[5.0, 4.0, 1.2, 3.8]]'}

result = test_predict_function(fake_data, api.handle_clipper_strings, "strings")
print(result)
