from __future__ import annotations

import typing as t

import numpy as np
import easyocr
import requests
from PIL import Image

import bentoml

from . import FrameworkTestModel
from . import FrameworkTestModelInput as Input
from . import FrameworkTestModelConfiguration as Config

framework = bentoml.easyocr
backward_compatible = False

url = "https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/english.png"


def check_output(output: list[tuple[list[t.Any], str, float]]):
    assert (
        " ".join([x[1] for x in output])
        == "Reduce your risk of coronavirus infection: Clean hands with soap and water or alcohol-based hand rub Cover nose and mouth when coughing and sneezing with tissue or flexed elbow Avoid close contact with anyone with cold or flu-like symptoms Thoroughly cook meat and eggs No unprotected contact with live wild or farm animals World Health Organization"
    )


en_reader = FrameworkTestModel(
    name="en_reader",
    model=easyocr.Reader(["en"]),
    configurations=[
        Config(
            test_inputs={
                "readtext": [
                    Input(
                        input_args=(
                            [
                                np.asarray(
                                    Image.open(
                                        requests.get(url, stream=True).raw
                                    ).convert("RGB")
                                )
                            ]
                        ),
                        expected=check_output,
                    )
                ]
            }
        )
    ],
)

models = [en_reader]
