from typing import List

import numpy as np

import bentoml


def my_python_model(input_list: List[int]) -> List[int]:
    return np.square(np.array(input_list))


if __name__ == "__main__":
    # `save_model` saves a given python object or function
    saved_model = bentoml.picklable_model.save_model(
        "my_python_model", my_python_model, signatures={"__call__": {"batchable": True}}
    )
    print(f"Model saved: {saved_model}")
